"""
utils/logger.py – Structured, colored logging via Loguru.
All modules import `logger` from here.
"""

import sys
from pathlib import Path

from loguru import logger as _logger

from config import log_config

# ── Remove default Loguru sink ─────────────────────────────────────────────────
_logger.remove()

# ── Console sink (coloured) ────────────────────────────────────────────────────
_logger.add(
    sys.stderr,
    level=log_config.LEVEL,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> – "
        "<level>{message}</level>"
    ),
    colorize=True,
)

# ── Rotating file sink ─────────────────────────────────────────────────────────
_logger.add(
    str(log_config.FILE),
    level=log_config.LEVEL,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} – {message}",
    rotation="10 MB",
    retention="7 days",
    compression="zip",
    enqueue=True,
)

# ── Public export ──────────────────────────────────────────────────────────────
logger = _logger
