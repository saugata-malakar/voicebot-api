"""
nlp/train.py – Fine-tune BERT for customer-support intent classification.

Usage
─────
  # 1. Generate dataset (if not already done)
  python nlp/dataset.py

  # 2. Train
  python nlp/train.py

Outputs
───────
  models/intent_classifier/
    ├── config.json
    ├── pytorch_model.bin  (or model.safetensors)
    ├── tokenizer_config.json
    ├── vocab.txt
    └── label_map.json

  data/
    └── evaluation_report.png   (confusion matrix + metrics bar chart)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# ── Path resolution so we can run from any working directory ──────────────────
_BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND))

from config import nlp_config, train_config, DATA_DIR
from utils.logger import logger


# ── PyTorch Dataset ────────────────────────────────────────────────────────────

class IntentDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_len: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


# ── Visualisation helpers ──────────────────────────────────────────────────────

def _plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    out_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Intent Classifier – Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved → {}", out_path)


def _plot_metrics(report_dict: dict, out_path: Path) -> None:
    intents = [k for k in report_dict if k not in ("accuracy", "macro avg", "weighted avg")]
    precisions = [report_dict[i]["precision"] for i in intents]
    recalls = [report_dict[i]["recall"] for i in intents]
    f1s = [report_dict[i]["f1-score"] for i in intents]

    x = np.arange(len(intents))
    width = 0.25

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - width, precisions, width, label="Precision", color="#4C72B0")
    ax.bar(x, recalls, width, label="Recall", color="#55A868")
    ax.bar(x + width, f1s, width, label="F1-Score", color="#C44E52")

    ax.set_xticks(x)
    ax.set_xticklabels(intents, rotation=45, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Per-Intent Evaluation Metrics", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Metrics bar chart saved → {}", out_path)


def _plot_training_loss(train_losses: list[float], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, marker="o", color="#4C72B0", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Training Loss")
    ax.set_title("Training Loss Curve", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Loss curve saved → {}", out_path)


# ── Training entry-point ───────────────────────────────────────────────────────

def train() -> None:
    logger.info("═" * 60)
    logger.info("  INTENT CLASSIFIER TRAINING")
    logger.info("  Base model : {}", train_config.OUTPUT_DIR / ".." if False else nlp_config.BASE_MODEL)
    logger.info("  Epochs     : {}", train_config.EPOCHS)
    logger.info("  Batch size : {}", train_config.BATCH_SIZE)
    logger.info("  LR         : {}", train_config.LR)
    logger.info("═" * 60)

    # ── Load dataset ──────────────────────────────────────────────────────────
    if not train_config.DATA_PATH.exists():
        logger.info("Dataset not found – generating…")
        from nlp.dataset import save_dataset
        save_dataset()

    df = pd.read_csv(train_config.DATA_PATH)
    logger.info("Dataset loaded: {} samples, {} intents", len(df), df["intent"].nunique())

    texts = df["text"].tolist()
    labels = df["intent_id"].tolist()

    # ── Split ─────────────────────────────────────────────────────────────────
    x_temp, x_test, y_temp, y_test = train_test_split(
        texts, labels,
        test_size=train_config.TEST_SPLIT,
        random_state=train_config.SEED,
        stratify=labels,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp,
        test_size=train_config.EVAL_SPLIT / (1 - train_config.TEST_SPLIT),
        random_state=train_config.SEED,
        stratify=y_temp,
    )
    logger.info("Split → train={}, val={}, test={}", len(x_train), len(x_val), len(x_test))

    # ── Tokenizer & model ─────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: {}", device)

    tokenizer = AutoTokenizer.from_pretrained(nlp_config.BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        nlp_config.BASE_MODEL,
        num_labels=nlp_config.NUM_LABELS,
        id2label=nlp_config.ID2LABEL,
        label2id=nlp_config.LABEL2ID,
    )
    model.to(device)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_ds = IntentDataset(x_train, y_train, tokenizer, nlp_config.MAX_LENGTH)
    val_ds   = IntentDataset(x_val,   y_val,   tokenizer, nlp_config.MAX_LENGTH)
    test_ds  = IntentDataset(x_test,  y_test,  tokenizer, nlp_config.MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=train_config.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=train_config.BATCH_SIZE)
    test_loader  = DataLoader(test_ds,  batch_size=train_config.BATCH_SIZE)

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.LR, weight_decay=0.01)
    total_steps = len(train_loader) * train_config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    train_losses: list[float] = []

    for epoch in range(1, train_config.EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.perf_counter()

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_t = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_t)
            loss = outputs.loss
            epoch_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Validation
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(batch["labels"].numpy())

        val_acc = accuracy_score(val_true, val_preds)
        elapsed = time.perf_counter() - t0
        logger.info(
            "Epoch {}/{} | loss={:.4f} | val_acc={:.4f} | {:.1f}s",
            epoch, train_config.EPOCHS, avg_loss, val_acc, elapsed,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(str(train_config.OUTPUT_DIR))
            tokenizer.save_pretrained(str(train_config.OUTPUT_DIR))
            logger.info("  ✓ Best model saved (val_acc={:.4f})", best_val_acc)

    # ── Test-set evaluation ───────────────────────────────────────────────────
    logger.info("Running test-set evaluation…")
    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            test_preds.extend(preds)
            test_true.extend(batch["labels"].numpy())

    class_names = [nlp_config.ID2LABEL[i] for i in range(nlp_config.NUM_LABELS)]

    acc  = accuracy_score(test_true, test_preds)
    prec = precision_score(test_true, test_preds, average="weighted", zero_division=0)
    rec  = recall_score(test_true, test_preds, average="weighted", zero_division=0)
    f1   = f1_score(test_true, test_preds, average="weighted", zero_division=0)

    logger.info("══ TEST RESULTS ══════════════════════════")
    logger.info("  Accuracy  : {:.4f}", acc)
    logger.info("  Precision : {:.4f}", prec)
    logger.info("  Recall    : {:.4f}", rec)
    logger.info("  F1-Score  : {:.4f}", f1)
    logger.info("══════════════════════════════════════════")

    report_dict = classification_report(
        test_true, test_preds, target_names=class_names, output_dict=True, zero_division=0
    )

    # Save metrics JSON
    metrics_path = DATA_DIR / "evaluation_metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(
            {
                "accuracy": acc, "precision": prec,
                "recall": rec, "f1": f1,
                "classification_report": report_dict,
            },
            fh, indent=2,
        )
    logger.info("Metrics JSON saved → {}", metrics_path)

    # Save visualisations
    _plot_confusion_matrix(test_true, test_preds, class_names, DATA_DIR / "confusion_matrix.png")
    _plot_metrics(report_dict, DATA_DIR / "metrics_per_intent.png")
    _plot_training_loss(train_losses, DATA_DIR / "training_loss.png")

    # Save label map
    label_map_path = train_config.OUTPUT_DIR / "label_map.json"
    with open(label_map_path, "w") as fh:
        json.dump({"id2label": nlp_config.ID2LABEL, "label2id": nlp_config.LABEL2ID}, fh, indent=2)
    logger.info("Label map saved → {}", label_map_path)

    logger.info("Training complete ✓ — Model in {}", train_config.OUTPUT_DIR)


if __name__ == "__main__":
    train()
