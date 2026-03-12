"""
asr/whisper_asr.py – Automatic Speech Recognition using OpenAI Whisper.

Features
────────
• Singleton model loader (loaded once, reused across requests)
• WAV + other formats via pydub conversion
• WER evaluation against reference transcripts
• Graceful handling of noise / short audio
• Returns transcript + word-level confidence where available
"""

from __future__ import annotations

import io
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import whisper
from jiwer import wer as compute_wer
from pydub import AudioSegment

from config import asr_config, TEMP_DIR
from utils.logger import logger


# ── Singleton loader ───────────────────────────────────────────────────────────
_model: Optional[whisper.Whisper] = None


def _get_model() -> whisper.Whisper:
    global _model
    if _model is None:
        logger.info(
            "Loading Whisper '{}' model on device '{}'…",
            asr_config.MODEL,
            asr_config.DEVICE,
        )
        t0 = time.perf_counter()
        _model = whisper.load_model(asr_config.MODEL, device=asr_config.DEVICE)
        logger.info("Whisper model loaded in {:.2f}s", time.perf_counter() - t0)
    return _model


# ── Audio preprocessing ────────────────────────────────────────────────────────

def _to_wav_bytes(audio_bytes: bytes, original_ext: str = ".wav") -> bytes:
    """
    Convert any supported audio format to 16-kHz mono WAV bytes.
    For WAV files, uses soundfile directly. For other formats, tries pydub.
    Returns original bytes unchanged if already compatible WAV.
    """
    ext = original_ext.lower()
    if ext not in asr_config.SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format '{ext}'. "
            f"Supported: {asr_config.SUPPORTED_FORMATS}"
        )

    # For WAV files, try to use soundfile directly (no ffmpeg needed)
    if ext in (".wav", "wav"):
        try:
            import scipy.io.wavfile as scipy_wav
            buf_in = io.BytesIO(audio_bytes)
            data, sr = sf.read(buf_in, dtype="float32")
            
            # Resample to 16kHz if needed
            if sr != 16000:
                import librosa
                data = librosa.resample(data, orig_sr=sr, target_sr=16000)
                sr = 16000
            
            # Convert to mono if stereo
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            
            # Convert to 16-bit PCM WAV
            buf_out = io.BytesIO()
            sf.write(buf_out, data, sr, format='WAV', subtype='PCM_16')
            return buf_out.getvalue()
        except Exception as e:
            logger.warning("soundfile processing failed: {}, trying pydub", e)
    
    # Fallback to pydub for non-WAV formats or if soundfile fails
    try:
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format=ext.lstrip("."))
        seg = seg.set_channels(1).set_frame_rate(16_000)
        buf = io.BytesIO()
        seg.export(buf, format="wav")
        return buf.getvalue()
    except Exception as e:
        raise ValueError(f"Audio conversion failed: {e}. Make sure ffmpeg is installed for non-WAV formats.")


def _validate_audio(audio_bytes: bytes) -> None:
    """Raise ValueError if audio is too short or silent."""
    buf = io.BytesIO(audio_bytes)
    data, sr = sf.read(buf, dtype="float32")
    duration = len(data) / sr
    if duration < 0.3:
        raise ValueError("Audio is too short (< 0.3 s). Please speak longer.")
    rms = float(np.sqrt(np.mean(data ** 2)))
    if rms < 1e-4:
        raise ValueError("Audio appears to be silent. Check your microphone.")


# ── Main transcription function ────────────────────────────────────────────────

def transcribe(
    audio_bytes: bytes,
    original_ext: str = ".wav",
    language: Optional[str] = None,
) -> dict:
    """
    Transcribe audio bytes → text.

    Returns
    -------
    {
        "text": str,
        "language": str,
        "duration": float,          # seconds
        "inference_time": float,    # seconds
        "segments": list[dict],     # per-segment details
    }
    """
    t_start = time.perf_counter()
    language = language or asr_config.LANGUAGE

    logger.debug("Preprocessing audio (ext={})…", original_ext)
    wav_bytes = _to_wav_bytes(audio_bytes, original_ext)
    _validate_audio(wav_bytes)

    # Write to a temp file because Whisper reads from disk
    with tempfile.NamedTemporaryFile(
        suffix=".wav", dir=TEMP_DIR, delete=False
    ) as tmp:
        tmp.write(wav_bytes)
        tmp_path = tmp.name

    try:
        model = _get_model()
        logger.debug("Running Whisper inference…")
        result = model.transcribe(
            tmp_path,
            language=language,
            fp16=(asr_config.DEVICE == "cuda"),
            verbose=False,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    inference_time = time.perf_counter() - t_start
    transcript = result["text"].strip()
    detected_lang = result.get("language", language)
    duration = result.get("segments", [{}])[-1].get("end", 0.0) if result.get("segments") else 0.0

    logger.info(
        "Transcription complete | lang={} | duration={:.1f}s | infer={:.2f}s | text='{}'",
        detected_lang,
        duration,
        inference_time,
        transcript[:80],
    )

    return {
        "text": transcript,
        "language": detected_lang,
        "duration": duration,
        "inference_time": round(inference_time, 3),
        "segments": result.get("segments", []),
    }


# ── WER Evaluation ─────────────────────────────────────────────────────────────

def evaluate_wer(
    audio_paths: list[str],
    references: list[str],
) -> dict:
    """
    Compute Word Error Rate on a list of (audio_file, reference_text) pairs.

    Returns
    -------
    {
        "wer": float,
        "num_samples": int,
        "results": list[{"file", "reference", "hypothesis", "sample_wer"}]
    }
    """
    if len(audio_paths) != len(references):
        raise ValueError("audio_paths and references must have equal length.")

    model = _get_model()
    hypotheses: list[str] = []
    results: list[dict] = []

    for path, ref in zip(audio_paths, references):
        p = Path(path)
        if not p.exists():
            logger.warning("Audio file not found: {}", path)
            hyp = ""
        else:
            res = model.transcribe(str(p), language=asr_config.LANGUAGE, fp16=False)
            hyp = res["text"].strip()

        sample_wer = compute_wer(ref.lower(), hyp.lower())
        hypotheses.append(hyp)
        results.append(
            {"file": path, "reference": ref, "hypothesis": hyp, "sample_wer": round(sample_wer, 4)}
        )

    overall_wer = compute_wer(
        [r.lower() for r in references],
        [h.lower() for h in hypotheses],
    )
    logger.info("WER evaluation complete | WER={:.4f} | n={}", overall_wer, len(audio_paths))

    return {
        "wer": round(overall_wer, 4),
        "num_samples": len(audio_paths),
        "results": results,
    }
