"""
evaluate/metrics.py – Evaluation utilities.

Functions
─────────
• evaluate_asr        – compute WER on a test set
• evaluate_classifier – load saved metrics JSON and return summary
• generate_report     – combine both into a full HTML report
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from config import DATA_DIR
from utils.logger import logger


def evaluate_asr(audio_paths: list[str], references: list[str]) -> dict:
    """Thin wrapper – calls asr.whisper_asr.evaluate_wer."""
    from asr.whisper_asr import evaluate_wer
    return evaluate_wer(audio_paths, references)


def load_classifier_metrics() -> Optional[dict]:
    """Load classifier evaluation metrics saved during training."""
    path = DATA_DIR / "evaluation_metrics.json"
    if not path.exists():
        logger.warning("No classifier metrics file found at {}", path)
        return None
    with open(path) as fh:
        return json.load(fh)


def generate_report() -> dict:
    """
    Return a combined evaluation report dict.
    Suitable for the /evaluate API endpoint.
    """
    clf = load_classifier_metrics()
    report: dict = {}

    if clf:
        report["intent_classifier"] = {
            "accuracy":  clf.get("accuracy"),
            "precision": clf.get("precision"),
            "recall":    clf.get("recall"),
            "f1":        clf.get("f1"),
            "confusion_matrix_image": str(DATA_DIR / "confusion_matrix.png"),
            "metrics_chart_image":    str(DATA_DIR / "metrics_per_intent.png"),
            "loss_curve_image":       str(DATA_DIR / "training_loss.png"),
        }
    else:
        report["intent_classifier"] = {"status": "not_trained"}

    return report
