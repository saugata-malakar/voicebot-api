"""
tests/test_api.py – Automated tests for all API endpoints.

Run with:  pytest tests/ -v
"""

import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure backend is in path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from main import app

client = TestClient(app)


# ── Health ─────────────────────────────────────────────────────────────────────
def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ── Intents list ───────────────────────────────────────────────────────────────
def test_intents():
    r = client.get("/intents")
    assert r.status_code == 200
    data = r.json()
    assert "intents" in data
    assert len(data["intents"]) == 13


# ── Intent prediction ──────────────────────────────────────────────────────────
def test_predict_intent_order_status():
    r = client.post("/predict-intent", json={"text": "Where is my order?", "top_k": 5})
    assert r.status_code == 200
    data = r.json()
    assert data["intent"] == "order_status"
    assert data["confidence"] > 0.3
    assert len(data["top_k"]) == 5


def test_predict_intent_cancel():
    r = client.post("/predict-intent", json={"text": "Please cancel my order"})
    assert r.status_code == 200
    assert r.json()["intent"] == "cancel_order"


def test_predict_intent_empty_text():
    r = client.post("/predict-intent", json={"text": ""})
    assert r.status_code == 422     # validation error


# ── Response generation ────────────────────────────────────────────────────────
def test_generate_response_known_intent():
    r = client.post("/generate-response", json={
        "intent": "order_status",
        "confidence": 0.95,
        "original_text": "Where is my order?",
    })
    assert r.status_code == 200
    data = r.json()
    assert len(data["response"]) > 10
    assert data["escalated"] is False


def test_generate_response_low_confidence():
    r = client.post("/generate-response", json={
        "intent": "general_inquiry",
        "confidence": 0.1,     # below threshold → escalate
        "original_text": "umm",
    })
    assert r.status_code == 200
    assert r.json()["escalated"] is True


def test_generate_response_unknown_intent():
    r = client.post("/generate-response", json={
        "intent": "flying_unicorn",
        "confidence": 0.99,
    })
    assert r.status_code == 422


# ── Evaluate endpoint ──────────────────────────────────────────────────────────
def test_evaluate():
    r = client.get("/evaluate")
    assert r.status_code == 200
    data = r.json()
    assert "intent_classifier" in data


# ── Transcribe (with a synthesised silent WAV) ─────────────────────────────────
def _make_silent_wav(duration_secs: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Create a minimal valid WAV file with silence."""
    import struct, math
    num_samples = int(sample_rate * duration_secs)
    data_bytes = bytes(num_samples * 2)   # 16-bit silence

    # WAV header
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + len(data_bytes),
        b'WAVE',
        b'fmt ',
        16,      # chunk size
        1,       # PCM
        1,       # mono
        sample_rate,
        sample_rate * 2,
        2,       # block align
        16,      # bits per sample
        b'data',
        len(data_bytes),
    )
    return header + data_bytes


def test_transcribe_silent_wav():
    wav_bytes = _make_silent_wav(0.5)
    r = client.post(
        "/transcribe",
        data={"language": "en"},
        files={"file": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")},
    )
    # Silent audio may raise 422 (too silent) or 200 (empty transcript) — both are valid
    assert r.status_code in (200, 422)
