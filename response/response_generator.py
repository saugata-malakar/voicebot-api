"""
response/response_generator.py – Contextual response generation.

Strategy
────────
1. Primary: Intent-to-response template bank (fast, deterministic, zero hallucination)
2. Context hints: Extracts entities (order IDs, dates) from the user text
   and interpolates them into the response for a natural feel.
3. Low-confidence / unknown: Polite escalation to a human agent.

All responses are scoped strictly to customer-support topics.
"""

from __future__ import annotations

import re
import random
from typing import Optional

from utils.logger import logger


# ── Response templates ─────────────────────────────────────────────────────────
# Each intent has several paraphrases; one is chosen at random to avoid
# repetitive-sounding replies.

_RESPONSES: dict[str, list[str]] = {

    "order_status": [
        "I can check that for you right away! Could you please share your order number? "
        "Once I have it, I'll pull up the latest tracking information.",
        "Of course! To look up your order status, please provide your order ID or the "
        "email address used at checkout.",
        "I'm happy to help track your order. Please share your order number and I'll "
        "give you a real-time update.",
    ],

    "cancel_order": [
        "I can help you cancel your order. Please note that orders can be cancelled "
        "within 1 hour of placement. Could you share your order number?",
        "No problem! To process a cancellation, I'll need your order number. "
        "If the order has already shipped, we can arrange a return instead.",
        "I'll do my best to cancel that for you. Please provide the order ID and "
        "I'll check if cancellation is still possible.",
    ],

    "refund_request": [
        "I understand you'd like a refund. Refunds are typically processed within "
        "5–7 business days back to your original payment method. Could you share "
        "your order number?",
        "I'll be happy to initiate your refund. Please provide your order ID and "
        "a brief reason for the refund so I can process it promptly.",
        "Refund requests are processed within 3–5 business days. To get started, "
        "could you confirm your order number and the email on your account?",
    ],

    "subscription_management": [
        "I can assist with your subscription. Whether you'd like to upgrade, "
        "downgrade, pause, or cancel, just let me know the change and I'll "
        "update your plan right away.",
        "Managing your subscription is easy! Please tell me what change you'd like — "
        "upgrade, downgrade, pause, or cancel — and I'll take care of it.",
        "Of course! Could you let me know which subscription action you need: "
        "upgrade, downgrade, pause, or cancellation? I'll handle it immediately.",
    ],

    "password_reset": [
        "I can send you a password reset link right away. Please confirm the email "
        "address associated with your account.",
        "No worries! I'll send a password reset email to the address on file. "
        "Could you confirm your email address?",
        "To reset your password, I'll send a secure link to your registered email. "
        "Please share the email address you used to sign up.",
    ],

    "account_issues": [
        "I'm sorry to hear you're having trouble with your account. Could you describe "
        "the issue in a bit more detail? I'll get it sorted for you.",
        "I can look into your account right away. Please share your registered email "
        "address and describe the problem you're experiencing.",
        "Let me help resolve your account issue. To verify your identity, could you "
        "provide your account email address?",
    ],

    "payment_problems": [
        "I'm sorry for the payment inconvenience. Could you let me know the error "
        "you're seeing? I'll investigate and resolve it as quickly as possible.",
        "Payment issues can be frustrating — I apologise. Please provide your account "
        "email and a description of the problem so I can look into it.",
        "I'll help you sort out the payment issue right away. Could you share the "
        "transaction ID or the date of the charge?",
    ],

    "shipping_inquiry": [
        "Standard shipping takes 5–7 business days. Express shipping (2–3 days) is "
        "also available at checkout. Is there a specific order you'd like to inquire about?",
        "We ship to most countries worldwide. Domestic orders typically arrive within "
        "3–5 business days. Would you like express shipping details?",
        "I can help with shipping information. Could you let me know whether this is "
        "for a new order or an existing one? I'll provide the most accurate details.",
    ],

    "product_complaint": [
        "I sincerely apologise for the experience with your product. Could you describe "
        "the issue and provide your order number? We'll arrange a replacement or refund.",
        "I'm very sorry to hear that. Please share your order ID and a description of "
        "the defect — I'll escalate this to our quality team immediately.",
        "That's certainly not the experience we want for you. Could you share your "
        "order number? We'll either replace the item or issue a full refund.",
    ],

    "return_request": [
        "Of course! We accept returns within 30 days of delivery. Please provide "
        "your order number and I'll generate a prepaid return label for you.",
        "I can help you start a return. Could you share your order ID and the "
        "reason for the return? I'll send the return instructions to your email.",
        "Returns are easy with us. Provide your order number and I'll walk you "
        "through the steps — and arrange a free return label if applicable.",
    ],

    "technical_support": [
        "I'm sorry you're experiencing a technical issue. Could you describe the "
        "problem in detail and let me know the device/browser you're using?",
        "I'd be happy to help troubleshoot. Please share the error message you're "
        "seeing and the steps that led to the problem.",
        "Technical issues are a priority for us. Could you provide more details "
        "about the error? Our tech team will resolve it as quickly as possible.",
    ],

    "billing_inquiry": [
        "I can pull up your billing information. Please confirm your account email "
        "and I'll share the latest invoice and billing details.",
        "Of course! Could you let me know what billing information you need — "
        "invoice copy, charge explanation, or billing date? I'll get that for you.",
        "I'll be happy to help with billing. Please provide your account email and "
        "I'll send the requested billing documentation to your inbox.",
    ],

    "general_inquiry": [
        "Thank you for reaching out! I'm here to help. Could you please describe "
        "your question or issue in more detail?",
        "Hi there! I'm your customer support assistant. Please share what you need "
        "help with and I'll do my best to assist you.",
        "Of course, I'd be happy to help! Could you give me a bit more detail "
        "about what you're looking for?",
    ],
}

# Fallback for truly unknown intents
_FALLBACK_RESPONSES = [
    "I'm not quite sure I understood that. Could you rephrase your question? "
    "If needed, I can connect you with a human support agent.",
    "I'm sorry, I couldn't fully understand your request. Would you like me to "
    "transfer you to a live agent for further assistance?",
    "That's a bit outside what I can help with directly. Let me connect you with "
    "a specialist who can assist you better.",
]


# ── Entity extraction ──────────────────────────────────────────────────────────

def _extract_order_id(text: str) -> Optional[str]:
    match = re.search(r"\b(ORD\d{5,}|\d{5,8})\b", text, re.IGNORECASE)
    return match.group(0) if match else None


def _extract_email(text: str) -> Optional[str]:
    match = re.search(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else None


# ── Main generate function ─────────────────────────────────────────────────────

def generate(
    intent: str,
    confidence: float,
    original_text: str = "",
    confidence_threshold: float = 0.4,
) -> dict:
    """
    Generate a customer-support response for the given intent.

    Returns
    -------
    {
        "response": str,
        "intent_used": str,
        "escalated": bool,      # True if sent to human agent
        "order_id": str | None,
        "email": str | None,
    }
    """
    # Low-confidence → escalate
    if confidence < confidence_threshold:
        logger.debug("Low confidence ({:.4f}) — escalating.", confidence)
        return {
            "response": random.choice(_FALLBACK_RESPONSES),
            "intent_used": "unknown",
            "escalated": True,
            "order_id": None,
            "email": None,
        }

    templates = _RESPONSES.get(intent, _FALLBACK_RESPONSES)
    response_text = random.choice(templates)

    # Entity interpolation
    order_id = _extract_order_id(original_text)
    email = _extract_email(original_text)

    if order_id and "{order_id}" in response_text:
        response_text = response_text.replace("{order_id}", order_id)

    logger.debug("Generated response for intent='{}' | conf={:.4f}", intent, confidence)

    return {
        "response": response_text,
        "intent_used": intent,
        "escalated": False,
        "order_id": order_id,
        "email": email,
    }
