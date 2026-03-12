"""
generate_sample_audio.py – Generate synthetic WAV test files using pyttsx3.

These act as sample audio files for testing the /voicebot endpoint without a microphone.

Usage:  python generate_sample_audio.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path

SAMPLES = [
    ("where_is_my_order.wav",        "Where is my order? I placed it five days ago and haven't received it."),
    ("cancel_order.wav",              "I would like to cancel my recent order please."),
    ("refund_request.wav",            "I need a refund for order number ORD12345. The item arrived damaged."),
    ("password_reset.wav",            "I forgot my password. Can you send me a reset link?"),
    ("subscription_management.wav",   "I want to upgrade my subscription to the premium plan."),
    ("technical_support.wav",         "The app keeps crashing whenever I try to open it on my phone."),
    ("billing_inquiry.wav",           "Can I get a copy of my invoice from last month?"),
]

def generate():
    out_dir = Path(__file__).parent / "audio_samples"
    out_dir.mkdir(exist_ok=True)

    # Use pyttsx3 for WAV generation (works offline)
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        for filename, text in SAMPLES:
            out = out_dir / filename
            if out.exists():
                print(f"  (skip) {filename}")
                continue
            engine.save_to_file(text, str(out))
            engine.runAndWait()
            print(f"  ✓ {filename}")
        print(f"\n✅ Audio samples saved to {out_dir}")
    except Exception as e:
        print(f"❌ Could not generate audio samples: {e}")
        print("Install gTTS:  pip install gTTS pydub")


if __name__ == "__main__":
    generate()
