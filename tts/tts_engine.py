"""
tts/tts_engine.py – Text-to-Speech engine.

Engines
───────
• gTTS  (primary)  – Google TTS, requires internet; returns MP3
• pyttsx3 (offline fallback) – system TTS; returns WAV

Both return raw audio bytes so the API can stream them back.
"""

from __future__ import annotations

import io
import os
import tempfile
import time
from pathlib import Path
from typing import Literal

from config import tts_config, TEMP_DIR
from utils.logger import logger

# Lazy imports so the server starts even if a library is missing
try:
    from gtts import gTTS as _gTTS
    _GTTS_AVAILABLE = True
except ImportError:
    _GTTS_AVAILABLE = False
    logger.warning("gTTS not installed — will use pyttsx3 fallback.")

try:
    import pyttsx3 as _pyttsx3
    _PYTTSX3_AVAILABLE = True
except ImportError:
    _PYTTSX3_AVAILABLE = False
    logger.warning("pyttsx3 not installed.")


# ── gTTS ───────────────────────────────────────────────────────────────────────

def _synthesize_gtts(text: str, language: str, slow: bool) -> bytes:
    if not _GTTS_AVAILABLE:
        raise RuntimeError("gTTS is not installed. Run: pip install gTTS")

    t0 = time.perf_counter()
    tts = _gTTS(text=text, lang=language, slow=slow)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    audio_bytes = buf.getvalue()
    logger.debug("gTTS synthesis: {:.2f}s | {} chars", time.perf_counter() - t0, len(text))
    return audio_bytes     # MP3


# ── pyttsx3 ────────────────────────────────────────────────────────────────────

def _synthesize_pyttsx3(text: str, rate: int, volume: float) -> bytes:
    if not _PYTTSX3_AVAILABLE:
        raise RuntimeError("pyttsx3 is not installed. Run: pip install pyttsx3")

    t0 = time.perf_counter()
    engine = _pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.setProperty("volume", volume)

    with tempfile.NamedTemporaryFile(suffix=".wav", dir=TEMP_DIR, delete=False) as tmp:
        tmp_path = tmp.name

    try:
        engine.save_to_file(text, tmp_path)
        engine.runAndWait()
        with open(tmp_path, "rb") as fh:
            audio_bytes = fh.read()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    logger.debug("pyttsx3 synthesis: {:.2f}s | {} chars", time.perf_counter() - t0, len(text))
    return audio_bytes     # WAV


# ── Public API ──────────────────────────────────────────────────────────────────

def synthesize(
    text: str,
    engine: Literal["gtts", "pyttsx3", "auto"] = "auto",
    language: str | None = None,
    slow: bool | None = None,
    rate: int | None = None,
    volume: float | None = None,
) -> dict:
    """
    Convert text to speech.

    Parameters
    ----------
    text     : str  – Text to synthesise (max ~5 000 chars recommended for gTTS)
    engine   : str  – 'gtts' | 'pyttsx3' | 'auto' (auto prefers gTTS, falls back to pyttsx3)
    language : str  – BCP-47 language code (e.g. 'en', 'fr')
    slow     : bool – Slow speech mode (gTTS only)
    rate     : int  – Words per minute (pyttsx3 only)
    volume   : float – 0.0–1.0 (pyttsx3 only)

    Returns
    -------
    {
        "audio_bytes": bytes,
        "format": "mp3" | "wav",
        "engine_used": str,
        "duration_chars": int,
        "synthesis_time": float,
    }
    """
    text = text.strip()
    if not text:
        raise ValueError("Cannot synthesise empty text.")

    language = language or tts_config.LANGUAGE
    slow = slow if slow is not None else tts_config.SLOW
    rate = rate if rate is not None else tts_config.RATE
    volume = volume if volume is not None else tts_config.VOLUME

    effective_engine = tts_config.ENGINE if engine == "auto" else engine

    t0 = time.perf_counter()

    if effective_engine == "gtts":
        try:
            audio = _synthesize_gtts(text, language, slow)
            fmt = "mp3"
            used = "gtts"
        except Exception as exc:
            if _PYTTSX3_AVAILABLE:
                logger.warning("gTTS failed ({}); retrying with pyttsx3.", exc)
                audio = _synthesize_pyttsx3(text, rate, volume)
                fmt = "wav"
                used = "pyttsx3"
            else:
                raise RuntimeError(f"gTTS failed: {exc}. pyttsx3 not available on this platform.")
    elif effective_engine == "pyttsx3":
        if not _PYTTSX3_AVAILABLE:
            logger.warning("pyttsx3 not available, falling back to gTTS.")
            audio = _synthesize_gtts(text, language, slow)
            fmt = "mp3"
            used = "gtts"
        else:
            audio = _synthesize_pyttsx3(text, rate, volume)
            fmt = "wav"
            used = "pyttsx3"
    else:
        audio = _synthesize_gtts(text, language, slow)
        fmt = "mp3"
        used = "gtts"

    synthesis_time = round(time.perf_counter() - t0, 3)
    logger.info(
        "TTS complete | engine={} | format={} | chars={} | time={:.2f}s",
        used, fmt, len(text), synthesis_time,
    )

    return {
        "audio_bytes": audio,
        "format": fmt,
        "engine_used": used,
        "duration_chars": len(text),
        "synthesis_time": synthesis_time,
    }
