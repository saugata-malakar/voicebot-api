"""
nlp/dataset.py – Synthetic customer-support intent dataset generator.

Produces a CSV with 1 000+ labelled utterances across 13 intents.
Run directly:  python nlp/dataset.py
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from config import DATA_DIR, nlp_config

# ── Seed for reproducibility ───────────────────────────────────────────────────
random.seed(42)

# ── Templates per intent ───────────────────────────────────────────────────────
_TEMPLATES: dict[str, list[str]] = {

    "order_status": [
        "Where is my order?",
        "What's the status of my order #{id}?",
        "Can you tell me when my package will arrive?",
        "I haven't received my order yet.",
        "When will order #{id} be delivered?",
        "I want to track my shipment.",
        "My order {num} hasn't shipped yet.",
        "Has my recent purchase been dispatched?",
        "I need an update on my order from {date}.",
        "Can you check the delivery status for me?",
        "Why is my order still processing?",
        "I placed an order {num} days ago — where is it?",
        "The tracking page says my order is delayed.",
        "Order #{id} shows no tracking information.",
        "I need to know when I'll get my delivery.",
        "Please look up my order status.",
        "My parcel seems to be stuck.",
        "No update since {date} on my order.",
        "Is my order on its way?",
        "I can't find any update on my recent purchase.",
    ],

    "cancel_order": [
        "I want to cancel my order.",
        "Please cancel order #{id}.",
        "How do I cancel a recent purchase?",
        "I need to stop my order from being shipped.",
        "Cancel my order placed on {date}.",
        "I changed my mind — please cancel.",
        "I'd like to cancel order number {id}.",
        "Can I still cancel before it ships?",
        "Stop my order immediately.",
        "I accidentally placed an order; please cancel it.",
        "I no longer want the item in my cart.",
        "Please reverse my recent order.",
        "Is it too late to cancel?",
        "I want to revoke my purchase.",
        "Cancel order #{id} right away.",
        "Please halt the shipment of my order.",
        "I need this order cancelled asap.",
        "Pull back my order, I don't need it.",
        "Can you cancel my last order?",
        "I want to undo my purchase.",
    ],

    "refund_request": [
        "I want a refund for my order.",
        "How do I get my money back?",
        "Please process a refund for order #{id}.",
        "I was charged incorrectly; I need a refund.",
        "My item arrived damaged — I need a refund.",
        "Request refund for #{id}.",
        "I returned my item and haven't received the refund.",
        "When will I get my refund?",
        "I need a reimbursement for order #{id}.",
        "I've been waiting for my refund since {date}.",
        "Please return the money to my account.",
        "My order never arrived — I need a full refund.",
        "Refund request for defective product.",
        "I was double-charged. Please refund one payment.",
        "I want my purchase price back.",
        "How long does a refund take?",
        "Give me back my money for the broken item.",
        "Issue a refund for the cancelled service.",
        "I'd like to request a chargeback.",
        "Process a refund to my original payment method.",
    ],

    "subscription_management": [
        "I want to upgrade my subscription.",
        "How do I downgrade my plan?",
        "Cancel my subscription.",
        "I'd like to change my subscription plan.",
        "Pause my subscription for a month.",
        "Renew my annual plan.",
        "I want to switch from monthly to yearly billing.",
        "How do I add more users to my plan?",
        "What are the available subscription tiers?",
        "I want to manage my subscription settings.",
        "Remove a feature from my current plan.",
        "Add the premium tier to my account.",
        "Extend my trial period.",
        "My free trial ended — how do I subscribe?",
        "Can I share my subscription with family?",
        "Tell me about the enterprise plan.",
        "I want to freeze my subscription.",
        "Move me to the basic plan.",
        "I need to modify my subscription.",
        "End my subscription at the next billing cycle.",
    ],

    "password_reset": [
        "I forgot my password.",
        "How do I reset my password?",
        "I can't log in — reset my password.",
        "Send me a password reset link.",
        "I need to change my password.",
        "My password is not working.",
        "I'm locked out of my account.",
        "Reset my account password, please.",
        "How can I update my password?",
        "I never received the password reset email.",
        "The reset link expired.",
        "I need help recovering my password.",
        "Password reset not working.",
        "Can you send a new reset email?",
        "I want to set a new password.",
        "My account password is wrong.",
        "Force reset my password.",
        "I've forgotten my login credentials.",
        "Help me get back into my account.",
        "The reset code is invalid.",
    ],

    "account_issues": [
        "I can't log into my account.",
        "My account has been suspended.",
        "How do I delete my account?",
        "I want to update my account details.",
        "My email address changed — update the account.",
        "I think someone hacked my account.",
        "My account shows the wrong information.",
        "I need to reactivate my account.",
        "My account has been locked.",
        "Create a new account for me.",
        "I can't verify my email.",
        "Account login is failing.",
        "The verification code isn't working.",
        "My profile picture won't update.",
        "I can't change my username.",
        "Merge two accounts.",
        "My account was incorrectly banned.",
        "Change the email on my account.",
        "Why is my account restricted?",
        "I need to access my old account.",
    ],

    "payment_problems": [
        "My payment failed.",
        "Why was I charged twice?",
        "My credit card isn't being accepted.",
        "I can't complete my purchase.",
        "There's an unauthorized charge on my account.",
        "My payment is pending for too long.",
        "Update my payment method.",
        "I need to add a new card.",
        "Remove my old credit card.",
        "Why was my payment declined?",
        "The checkout keeps failing.",
        "I can't enter my card details.",
        "My PayPal payment isn't processing.",
        "Reverse the duplicate charge.",
        "I was overcharged on my last order.",
        "My promo code didn't apply to the payment.",
        "The payment gateway shows an error.",
        "I need a receipt for my payment.",
        "Can I pay by bank transfer?",
        "My billing address won't verify.",
    ],

    "shipping_inquiry": [
        "How long does shipping take?",
        "What are your delivery options?",
        "Do you offer express shipping?",
        "Can I change my delivery address?",
        "I need expedited shipping.",
        "What is the shipping cost?",
        "Do you ship internationally?",
        "My package shows delivered but I didn't get it.",
        "The courier left my parcel in the wrong place.",
        "Can I pick up my order in store?",
        "How do I change my shipping address?",
        "What carriers do you use?",
        "I need Saturday delivery.",
        "My order was returned to sender.",
        "When is the last day to order for holiday delivery?",
        "Is free shipping available?",
        "My tracking number isn't working.",
        "What countries do you deliver to?",
        "The estimated delivery passed — where is my order?",
        "Can you ship to a PO box?",
    ],

    "product_complaint": [
        "The product I received is defective.",
        "My item broke after one use.",
        "The product doesn't match the description.",
        "I received the wrong item.",
        "The product quality is very poor.",
        "My order was incomplete — missing items.",
        "The product arrived damaged.",
        "The size I received is wrong.",
        "The colour is different from what I ordered.",
        "I'm not satisfied with the product.",
        "The item looks nothing like the picture.",
        "The product stopped working.",
        "I received an expired item.",
        "One part of the product is missing.",
        "The product leaks.",
        "The packaging was completely destroyed.",
        "The product has a manufacturing defect.",
        "Wrong model was sent to me.",
        "Product doesn't work as advertised.",
        "I want to report a safety issue with a product.",
    ],

    "return_request": [
        "I want to return an item.",
        "How do I send a product back?",
        "Start a return for order #{id}.",
        "I'd like to exchange this product.",
        "What is your return policy?",
        "I need a return label.",
        "My return request was rejected — why?",
        "Return a damaged item.",
        "How long do I have to return an order?",
        "I sent the item back, when will I get my refund?",
        "Can I return a sale item?",
        "I'd like to initiate a return.",
        "How do I return a digital purchase?",
        "Print a return shipping label for me.",
        "The return process is not clear.",
        "I'd like to swap sizes.",
        "Replace the defective product.",
        "Where do I send my return?",
        "I mailed the return — track it for me.",
        "Exchange this for a different colour.",
    ],

    "technical_support": [
        "The app keeps crashing.",
        "I can't install the software.",
        "The website won't load.",
        "I'm getting an error message.",
        "How do I enable notifications?",
        "The app is very slow.",
        "I can't connect to the server.",
        "My device is not compatible.",
        "The update broke something.",
        "I keep getting a 404 error.",
        "The download link isn't working.",
        "I need help setting up the device.",
        "The integration isn't syncing.",
        "API returning an unexpected error.",
        "The login page is down.",
        "Dark mode isn't working.",
        "I can't upload files.",
        "The video player won't play.",
        "The feature I need is missing.",
        "The mobile app shows a blank screen.",
    ],

    "billing_inquiry": [
        "Can I see my invoice?",
        "When is my next billing date?",
        "I need a copy of my latest receipt.",
        "What is my current plan cost?",
        "Explain the charge on my statement.",
        "How is my bill calculated?",
        "Send the invoice to my email.",
        "I didn't get my monthly invoice.",
        "Change my billing email.",
        "I need a VAT invoice.",
        "Why did my bill increase?",
        "I want to see my full billing history.",
        "What is the annual cost of my plan?",
        "I need a formal invoice for tax purposes.",
        "My company needs a billing statement.",
        "I see an unknown charge — explain it.",
        "Bill me quarterly instead of monthly.",
        "How much will I be charged next month?",
        "Generate a billing report.",
        "What does line item X mean on my invoice?",
    ],

    "general_inquiry": [
        "Hi, I need some help.",
        "Can someone assist me?",
        "What are your business hours?",
        "How do I contact customer support?",
        "I have a question.",
        "Who can help me with my issue?",
        "I'd like to speak to a human agent.",
        "What services do you offer?",
        "Hello, is anyone there?",
        "I need general information.",
        "Tell me about your company.",
        "Where is your headquarters?",
        "What is your return policy?",
        "Can I give feedback about my experience?",
        "I'd like to file a complaint.",
        "How do I use your website?",
        "I need help navigating the app.",
        "Connect me to support.",
        "What's your phone number?",
        "I want to leave a review.",
    ],
}

# ── Placeholder substitutions ──────────────────────────────────────────────────
_ORDER_IDS = [f"ORD{random.randint(10000,99999)}" for _ in range(50)]
_DATES = [
    "Jan 5", "Feb 14", "March 3", "April 20", "last Tuesday",
    "3 days ago", "yesterday", "last week", "two weeks ago",
]
_NUMS = [str(random.randint(2, 15)) for _ in range(10)]


def _fill(template: str) -> str:
    return (
        template
        .replace("{id}", random.choice(_ORDER_IDS))
        .replace("{date}", random.choice(_DATES))
        .replace("{num}", random.choice(_NUMS))
    )


# ── Dataset builder ────────────────────────────────────────────────────────────

def build_dataset(samples_per_intent: int = 80) -> pd.DataFrame:
    """
    Generate a synthetic dataset.

    Parameters
    ----------
    samples_per_intent : int
        Approximate utterances per intent (templates are repeated with variation).

    Returns
    -------
    pd.DataFrame with columns [text, intent, intent_id]
    """
    rows: list[dict] = []
    for intent, templates in _TEMPLATES.items():
        label_id = nlp_config.LABEL2ID[intent]
        count = 0
        while count < samples_per_intent:
            tmpl = random.choice(templates)
            text = _fill(tmpl)
            # Add light augmentation: lowercase occasionally
            if random.random() < 0.3:
                text = text.lower()
            rows.append({"text": text, "intent": intent, "intent_id": label_id})
            count += 1

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def save_dataset(samples_per_intent: int = 80) -> Path:
    df = build_dataset(samples_per_intent)
    out = DATA_DIR / "intent_dataset.csv"
    df.to_csv(out, index=False)
    print(f"Dataset saved → {out}  ({len(df)} rows, {df['intent'].nunique()} intents)")
    return out


if __name__ == "__main__":
    save_dataset()
