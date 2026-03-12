"""
config.py – Central configuration, loaded once at startup.
All settings are driven by environment variables (or .env file).
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Resolve project root & load .env ──────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


# ── Helper ─────────────────────────────────────────────────────────────────────
def _bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).lower() in ("1", "true", "yes")


def _int(key: str, default: int = 0) -> int:
    return int(os.getenv(key, str(default)))


def _float(key: str, default: float = 0.0) -> float:
    return float(os.getenv(key, str(default)))


# ── App ────────────────────────────────────────────────────────────────────────
class AppConfig:
    NAME: str = os.getenv("APP_NAME", "VoiceBot Customer Support")
    VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", os.getenv("APP_PORT", "8000")))
    DEBUG: bool = _bool("DEBUG", False)
    CORS_ORIGINS: list[str] = [
        o.strip()
        for o in os.getenv(
            "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:5500"
        ).split(",")
    ]


# ── ASR ────────────────────────────────────────────────────────────────────────
class ASRConfig:
    MODEL: str = os.getenv("ASR_MODEL", "base")
    LANGUAGE: str = os.getenv("ASR_LANGUAGE", "en")
    DEVICE: str = os.getenv("ASR_DEVICE", "cpu")
    SUPPORTED_FORMATS: tuple[str, ...] = (".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm")


# ── Intent Classifier ──────────────────────────────────────────────────────────
class NLPConfig:
    MODEL_DIR: Path = BASE_DIR / os.getenv("INTENT_MODEL_DIR", "models/intent_classifier")
    BASE_MODEL: str = os.getenv("INTENT_BASE_MODEL", "bert-base-uncased")
    CONFIDENCE_THRESHOLD: float = _float("INTENT_CONFIDENCE_THRESHOLD", 0.5)
    MAX_LENGTH: int = _int("INTENT_MAX_LENGTH", 128)

    # All supported intents
    INTENTS: list[str] = [
        "order_status",
        "cancel_order",
        "refund_request",
        "subscription_management",
        "password_reset",
        "account_issues",
        "payment_problems",
        "shipping_inquiry",
        "product_complaint",
        "return_request",
        "technical_support",
        "billing_inquiry",
        "general_inquiry",
    ]

    NUM_LABELS: int = len(INTENTS)
    ID2LABEL: dict[int, str] = {i: label for i, label in enumerate(INTENTS)}
    LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(INTENTS)}


# ── TTS ────────────────────────────────────────────────────────────────────────
class TTSConfig:
    ENGINE: str = os.getenv("TTS_ENGINE", "gtts")
    LANGUAGE: str = os.getenv("TTS_LANGUAGE", "en")
    SLOW: bool = _bool("TTS_SLOW", False)
    RATE: int = _int("TTS_RATE", 150)
    VOLUME: float = _float("TTS_VOLUME", 1.0)


# ── Logging ────────────────────────────────────────────────────────────────────
class LogConfig:
    LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    DIR: Path = BASE_DIR / os.getenv("LOG_DIR", "logs")
    DIR.mkdir(parents=True, exist_ok=True)
    FILE: Path = DIR / "voicebot.log"


# ── Training ───────────────────────────────────────────────────────────────────
class TrainConfig:
    EPOCHS: int = _int("TRAIN_EPOCHS", 5)
    BATCH_SIZE: int = _int("TRAIN_BATCH_SIZE", 16)
    LR: float = _float("TRAIN_LR", 2e-5)
    DATA_PATH: Path = BASE_DIR / os.getenv("TRAIN_DATA_PATH", "data/intent_dataset.csv")
    OUTPUT_DIR: Path = NLPConfig.MODEL_DIR
    EVAL_SPLIT: float = 0.15
    TEST_SPLIT: float = 0.10
    SEED: int = 42


# ── Directories ────────────────────────────────────────────────────────────────
MODELS_DIR = BASE_DIR / "models"
AUDIO_DIR = BASE_DIR / "audio_samples"
DATA_DIR = BASE_DIR / "data"
TEMP_DIR = BASE_DIR / "temp"

for _d in (MODELS_DIR, AUDIO_DIR, DATA_DIR, TEMP_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ── Expose grouped config ──────────────────────────────────────────────────────
app_config = AppConfig()
asr_config = ASRConfig()
nlp_config = NLPConfig()
tts_config = TTSConfig()
log_config = LogConfig()
train_config = TrainConfig()
