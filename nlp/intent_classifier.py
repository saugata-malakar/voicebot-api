"""
nlp/intent_classifier.py – Inference wrapper around the fine-tuned BERT model.

Supports
────────
• Single-text prediction with confidence score
• Batch prediction
• Top-k intent ranking
• Graceful fallback when the model is not yet trained
  (returns "general_inquiry" so the API stays functional)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import nlp_config
from utils.logger import logger


# ── Singleton loader ───────────────────────────────────────────────────────────
_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForSequenceClassification] = None
_device: Optional[torch.device] = None


def _load_model() -> tuple:
    global _tokenizer, _model, _device
    if _model is not None:
        return _tokenizer, _model, _device

    model_dir = nlp_config.MODEL_DIR

    if not model_dir.exists() or not (model_dir / "config.json").exists():
        logger.warning(
            "Intent model not found at '{}'. Using rule-based fallback.", model_dir
        )
        return None, None, None

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading intent classifier from '{}' on {}…", model_dir, _device)
    t0 = time.perf_counter()
    _tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    _model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    _model.to(_device)
    _model.eval()
    logger.info("Intent model loaded in {:.2f}s", time.perf_counter() - t0)
    return _tokenizer, _model, _device


# ── Rule-based fallback (keyword matching) ────────────────────────────────────
_KEYWORD_MAP: dict[str, list[str]] = {
    "order_status":            ["order status", "where is my order", "track", "delivery status", "when will"],
    "cancel_order":            ["cancel order", "cancel my order", "stop order", "revoke", "cancel purchase"],
    "refund_request":          ["refund", "money back", "reimbursement", "chargeback", "return money"],
    "subscription_management": ["subscription", "upgrade plan", "downgrade", "cancel subscription", "plan"],
    "password_reset":          ["password", "reset", "locked out", "forgot", "login credentials"],
    "account_issues":          ["account", "suspended", "delete account", "account details", "hacked"],
    "payment_problems":        ["payment", "charge", "credit card", "declined", "billing failed"],
    "shipping_inquiry":        ["shipping", "delivery", "carrier", "express", "delivery address"],
    "product_complaint":       ["defective", "broken", "damaged", "wrong item", "poor quality"],
    "return_request":          ["return", "exchange", "send back", "return label", "return policy"],
    "technical_support":       ["error", "crash", "bug", "not working", "install", "technical", "app issue"],
    "billing_inquiry":         ["invoice", "bill", "receipt", "billing date", "statement"],
}


def _rule_based_predict(text: str) -> dict:
    text_lower = text.lower()
    scores: dict[str, float] = {intent: 0.0 for intent in nlp_config.INTENTS}

    for intent, keywords in _KEYWORD_MAP.items():
        for kw in keywords:
            if kw in text_lower:
                scores[intent] += 1.0

    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0:
        best = "general_inquiry"

    total = sum(scores.values()) or 1.0
    confidence = scores[best] / total

    top_k = sorted(scores.items(), key=lambda x: -x[1])[:5]
    return {
        "intent": best,
        "confidence": round(min(confidence + 0.3, 0.95), 4),   # bias upward for rules
        "top_k": [{"intent": k, "score": round(v / total, 4)} for k, v in top_k],
        "method": "rule_based",
    }


# ── Neural prediction ──────────────────────────────────────────────────────────

def predict(text: str, top_k: int = 5) -> dict:
    """
    Predict the customer-support intent for a single text.

    Returns
    -------
    {
        "intent": str,
        "confidence": float,
        "top_k": [{"intent": str, "score": float}, ...],
        "method": "neural" | "rule_based",
        "inference_time": float,
    }
    """
    t0 = time.perf_counter()
    text = text.strip()
    if not text:
        raise ValueError("Input text is empty.")

    tokenizer, model, device = _load_model()

    if model is None:
        logger.debug("Using rule-based fallback for intent prediction.")
        result = _rule_based_predict(text)
        result["inference_time"] = round(time.perf_counter() - t0, 4)
        return result

    # Neural path
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=nlp_config.MAX_LENGTH,
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0]

    scores = probs.cpu().numpy()
    best_idx = int(scores.argmax())
    best_label = nlp_config.ID2LABEL[best_idx]
    best_score = float(scores[best_idx])

    sorted_idx = scores.argsort()[::-1][:top_k]
    top_k_list = [
        {"intent": nlp_config.ID2LABEL[i], "score": round(float(scores[i]), 4)}
        for i in sorted_idx
    ]

    # Apply confidence threshold
    if best_score < nlp_config.CONFIDENCE_THRESHOLD:
        logger.debug(
            "Low confidence ({:.4f}) for '{}'; falling back to rule-based.", best_score, text
        )
        rb = _rule_based_predict(text)
        rb["inference_time"] = round(time.perf_counter() - t0, 4)
        rb["neural_confidence"] = round(best_score, 4)
        return rb

    result = {
        "intent": best_label,
        "confidence": round(best_score, 4),
        "top_k": top_k_list,
        "method": "neural",
        "inference_time": round(time.perf_counter() - t0, 4),
    }

    logger.debug(
        "Intent: '{}' | conf={:.4f} | text='{}'",
        best_label, best_score, text[:60],
    )
    return result


def predict_batch(texts: list[str]) -> list[dict]:
    """Predict intents for a list of texts."""
    return [predict(t) for t in texts]
