"""
main.py – FastAPI application entry-point for the Voice Bot.

Endpoints
─────────
  POST /transcribe           – audio → text  (ASR)
  POST /predict-intent       – text  → intent + confidence
  POST /generate-response    – intent + text → response text
  POST /synthesize           – text  → audio
  POST /voicebot             – audio → audio (full pipeline)

  GET  /health               – liveness probe
  GET  /evaluate             – classifier evaluation metrics
  GET  /intents              – list of supported intents
  GET  /docs                 – Swagger UI (built-in)
"""

from __future__ import annotations

import base64
import io
import time
import uuid
from pathlib import Path
from typing import Annotated, Optional

from fastapi import (
    FastAPI, File, Form, HTTPException, Query, Request, UploadFile
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import app_config, nlp_config, TEMP_DIR
from utils.logger import logger

# ── App factory ────────────────────────────────────────────────────────────────
app = FastAPI(
    title=app_config.NAME,
    version=app_config.VERSION,
    description=(
        "Production-ready AI Voice Bot for Customer Support Automation.\n\n"
        "Pipeline: **Audio → ASR → Intent → Response → TTS → Audio**"
    ),
    contact={"name": "VoiceBot Team"},
    license_info={"name": "MIT"},
)

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Request-ID", "X-Process-Time",
        "X-Transcript", "X-Intent", "X-Confidence",
        "X-Response-Text", "X-Escalated",
        "X-TTS-Engine", "X-Total-Latency",
        "X-Engine-Used", "X-Synthesis-Time",
    ],
)

# ── Serve static frontend ──────────────────────────────────────────────────────
_FRONTEND = Path(__file__).parent.parent / "frontend"
if _FRONTEND.exists():
    app.mount("/ui", StaticFiles(directory=str(_FRONTEND), html=True), name="frontend")


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic schemas
# ═══════════════════════════════════════════════════════════════════════════════

class TranscribeResponse(BaseModel):
    request_id: str
    text: str
    language: str
    duration: float
    inference_time: float

class IntentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500, example="Where is my order?")
    top_k: int = Field(5, ge=1, le=13)

class IntentResponse(BaseModel):
    request_id: str
    text: str
    intent: str
    confidence: float
    top_k: list[dict]
    method: str
    inference_time: float

class GenerateRequest(BaseModel):
    intent: str = Field(..., example="order_status")
    confidence: float = Field(..., ge=0.0, le=1.0, example=0.95)
    original_text: str = Field("", example="Where is my order?")

class GenerateResponse(BaseModel):
    request_id: str
    response: str
    intent_used: str
    escalated: bool
    order_id: Optional[str]
    email: Optional[str]

class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    engine: str = Field("auto", description="gtts | pyttsx3 | auto")
    language: str = Field("en")
    slow: bool = Field(False)
    rate: int = Field(150, ge=50, le=300)
    volume: float = Field(1.0, ge=0.0, le=1.0)

class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float

class VoiceBotResponse(BaseModel):
    request_id: str
    transcript: str
    language: str
    intent: str
    confidence: float
    response_text: str
    escalated: bool
    tts_engine: str
    total_latency: float


# ── Startup timer ──────────────────────────────────────────────────────────────
_start_time = time.time()


# ═══════════════════════════════════════════════════════════════════════════════
# Request middleware – log every request
# ═══════════════════════════════════════════════════════════════════════════════
@app.middleware("http")
async def _log_requests(request: Request, call_next):
    rid = str(uuid.uuid4())[:8]
    t0 = time.perf_counter()
    logger.info("→ {} {} [rid={}]", request.method, request.url.path, rid)
    response = await call_next(request)
    elapsed = time.perf_counter() - t0
    logger.info("← {} {} | {:.0f}ms [rid={}]", request.method, request.url.path, elapsed * 1000, rid)
    response.headers["X-Request-ID"] = rid
    response.headers["X-Process-Time"] = f"{elapsed:.3f}"
    return response


# ═══════════════════════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════════════════════
def _req_id() -> str:
    return str(uuid.uuid4())


# ═══════════════════════════════════════════════════════════════════════════════
# GET /health
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness probe."""
    return HealthResponse(
        status="ok",
        version=app_config.VERSION,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# GET /intents
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/intents", tags=["NLP"])
async def list_intents():
    """Return the list of all supported customer-support intents."""
    return {"intents": nlp_config.INTENTS, "count": len(nlp_config.INTENTS)}


# ═══════════════════════════════════════════════════════════════════════════════
# GET /evaluate
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/evaluate", tags=["Evaluation"])
async def get_evaluation_report():
    """Return the latest classifier evaluation metrics."""
    from evaluate.metrics import generate_report
    return generate_report()


# ═══════════════════════════════════════════════════════════════════════════════
# POST /transcribe
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/transcribe", response_model=TranscribeResponse, tags=["ASR"])
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, OGG, FLAC, M4A)"),
    language: str = Form("en", description="BCP-47 language code"),
):
    """
    Convert uploaded audio to text using OpenAI Whisper.

    - Accepts WAV, MP3, OGG, FLAC, M4A
    - Returns transcript, detected language, and duration
    """
    from asr.whisper_asr import transcribe

    ext = Path(file.filename or "audio.wav").suffix.lower() or ".wav"
    audio_bytes = await file.read()

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    try:
        result = transcribe(audio_bytes, original_ext=ext, language=language)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("ASR error: {}", exc)
        raise HTTPException(status_code=500, detail=f"ASR failed: {exc}")

    return TranscribeResponse(
        request_id=_req_id(),
        text=result["text"],
        language=result["language"],
        duration=result["duration"],
        inference_time=result["inference_time"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# POST /predict-intent
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/predict-intent", response_model=IntentResponse, tags=["NLP"])
async def predict_intent(body: IntentRequest):
    """
    Classify the customer-support intent from text.

    - Returns intent label, confidence score, and top-k ranking
    - Uses fine-tuned BERT; falls back to keyword matching if model not found
    """
    from nlp.intent_classifier import predict

    try:
        result = predict(body.text, top_k=body.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Intent prediction error: {}", exc)
        raise HTTPException(status_code=500, detail=f"Intent prediction failed: {exc}")

    return IntentResponse(
        request_id=_req_id(),
        text=body.text,
        intent=result["intent"],
        confidence=result["confidence"],
        top_k=result["top_k"],
        method=result["method"],
        inference_time=result["inference_time"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# POST /generate-response
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/generate-response", response_model=GenerateResponse, tags=["NLP"])
async def generate_response(body: GenerateRequest):
    """
    Generate a customer-support response for the given intent.

    - Low-confidence intents are escalated to a human agent
    - Entities (order IDs, emails) are extracted and reflected in the response
    """
    from response.response_generator import generate

    if body.intent not in nlp_config.INTENTS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown intent '{body.intent}'. Valid: {nlp_config.INTENTS}",
        )

    result = generate(body.intent, body.confidence, body.original_text)

    return GenerateResponse(
        request_id=_req_id(),
        **result,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# POST /synthesize
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/synthesize", tags=["TTS"])
async def synthesize_speech(body: SynthesizeRequest):
    """
    Convert text to speech audio.

    Returns raw audio bytes (MP3 for gTTS, WAV for pyttsx3).
    """
    from tts.tts_engine import synthesize

    try:
        result = synthesize(
            text=body.text,
            engine=body.engine,
            language=body.language,
            slow=body.slow,
            rate=body.rate,
            volume=body.volume,
        )
    except Exception as exc:
        logger.exception("TTS error: {}", exc)
        raise HTTPException(status_code=500, detail=f"TTS failed: {exc}")

    audio = result["audio_bytes"]
    fmt = result["format"]
    mime = "audio/mpeg" if fmt == "mp3" else "audio/wav"

    return Response(
        content=audio,
        media_type=mime,
        headers={
            "X-Engine-Used": result["engine_used"],
            "X-Synthesis-Time": str(result["synthesis_time"]),
            "Content-Disposition": f'attachment; filename="response.{fmt}"',
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# POST /voicebot  – Full audio → audio pipeline
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/voicebot", tags=["VoiceBot"])
async def voicebot(
    file: UploadFile = File(..., description="Input audio file"),
    language: str = Form("en"),
    tts_engine: str = Form("auto"),
    return_audio: bool = Form(True, description="If false, return JSON metadata only"),
    response_format: str = Form("audio", description="'audio' = raw stream, 'json' = JSON with base64 audio"),
):
    """
    **Unified Audio → Audio Pipeline**

    1. ASR   : Convert audio to text (Whisper)
    2. NLP   : Classify intent (fine-tuned BERT)
    3. Gen   : Generate contextual response
    4. TTS   : Synthesise response audio (gTTS / pyttsx3)
    5. Return: Response audio stream + JSON metadata header

    If `return_audio=false`, returns JSON with all pipeline metadata.
    """
    from asr.whisper_asr import transcribe
    from nlp.intent_classifier import predict
    from response.response_generator import generate
    from tts.tts_engine import synthesize

    t_total = time.perf_counter()
    rid = _req_id()

    # ── Step 1: ASR ────────────────────────────────────────────────────────────
    ext = Path(file.filename or "audio.wav").suffix.lower() or ".wav"
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    try:
        asr_result = transcribe(audio_bytes, original_ext=ext, language=language)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"ASR: {exc}")
    except Exception as exc:
        logger.exception("ASR pipeline error: {}", exc)
        raise HTTPException(status_code=500, detail=f"ASR failed: {exc}")

    transcript = asr_result["text"]
    if not transcript:
        raise HTTPException(status_code=422, detail="Could not transcribe audio. Please speak clearly.")

    # ── Step 2: Intent ─────────────────────────────────────────────────────────
    try:
        intent_result = predict(transcript)
    except Exception as exc:
        logger.exception("Intent pipeline error: {}", exc)
        raise HTTPException(status_code=500, detail=f"Intent prediction failed: {exc}")

    # ── Step 3: Response Generation ────────────────────────────────────────────
    gen_result = generate(
        intent=intent_result["intent"],
        confidence=intent_result["confidence"],
        original_text=transcript,
    )

    response_text = gen_result["response"]

    # ── Step 4: TTS ────────────────────────────────────────────────────────────
    try:
        tts_result = synthesize(text=response_text, engine=tts_engine, language=language)
    except Exception as exc:
        logger.exception("TTS pipeline error: {}", exc)
        raise HTTPException(status_code=500, detail=f"TTS failed: {exc}")

    total_latency = round(time.perf_counter() - t_total, 3)
    logger.info(
        "[{}] Pipeline complete | transcript='{}' | intent='{}' | latency={:.2f}s",
        rid, transcript[:50], intent_result["intent"], total_latency,
    )

    # ── Response ───────────────────────────────────────────────────────────────
    if not return_audio:
        return JSONResponse({
            "request_id": rid,
            "transcript": transcript,
            "language": asr_result["language"],
            "intent": intent_result["intent"],
            "confidence": intent_result["confidence"],
            "response_text": response_text,
            "escalated": gen_result["escalated"],
            "tts_engine": tts_result["engine_used"],
            "total_latency": total_latency,
        })

    audio = tts_result["audio_bytes"]
    fmt = tts_result["format"]
    mime = "audio/mpeg" if fmt == "mp3" else "audio/wav"

    # JSON mode: return everything as JSON with base64-encoded audio
    if response_format == "json":
        return JSONResponse({
            "request_id": rid,
            "transcript": transcript,
            "language": asr_result["language"],
            "intent": intent_result["intent"],
            "confidence": intent_result["confidence"],
            "response_text": response_text,
            "escalated": gen_result["escalated"],
            "tts_engine": tts_result["engine_used"],
            "total_latency": total_latency,
            "audio_base64": base64.b64encode(audio).decode("ascii"),
            "audio_format": fmt,
        })

    return Response(
        content=audio,
        media_type=mime,
        headers={
            "X-Request-ID": rid,
            "X-Transcript": transcript[:100],
            "X-Intent": intent_result["intent"],
            "X-Confidence": str(intent_result["confidence"]),
            "X-Response-Text": response_text[:200],
            "X-Escalated": str(gen_result["escalated"]).lower(),
            "X-TTS-Engine": tts_result["engine_used"],
            "X-Total-Latency": str(total_latency),
            "Content-Disposition": f'attachment; filename="voicebot_response.{fmt}"',
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# POST /wer-evaluate  – ASR evaluation
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/wer-evaluate", tags=["Evaluation"])
async def wer_evaluate(
    files: list[UploadFile] = File(...),
    references: str = Form(..., description="JSON array of reference transcripts"),
):
    """
    Compute Word Error Rate (WER) on uploaded audio + reference transcripts.

    `references` should be a JSON string like:
    `["hello world", "please cancel my order", ...]`
    """
    import json as _json
    from asr.whisper_asr import transcribe
    from jiwer import wer as compute_wer

    try:
        refs: list[str] = _json.loads(references)
    except Exception:
        raise HTTPException(status_code=422, detail="references must be a valid JSON array of strings.")

    if len(files) != len(refs):
        raise HTTPException(status_code=422, detail="Number of files and references must match.")

    hypotheses = []
    results = []
    for uf, ref in zip(files, refs):
        ext = Path(uf.filename or "audio.wav").suffix.lower() or ".wav"
        audio = await uf.read()
        try:
            res = transcribe(audio, original_ext=ext)
            hyp = res["text"]
        except Exception as exc:
            hyp = ""
            logger.warning("Transcription failed for {}: {}", uf.filename, exc)
        sample_wer = compute_wer(ref.lower(), hyp.lower())
        hypotheses.append(hyp)
        results.append({"file": uf.filename, "reference": ref, "hypothesis": hyp, "sample_wer": round(sample_wer, 4)})

    overall_wer = compute_wer(
        [r.lower() for r in refs],
        [h.lower() for h in hypotheses],
    )

    return {
        "overall_wer": round(overall_wer, 4),
        "num_samples": len(files),
        "results": results,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Root redirect
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/", include_in_schema=False)
async def root():
    return {"message": f"{app_config.NAME} v{app_config.VERSION}",
            "docs": "/docs", "ui": "/ui"}


# ═══════════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=app_config.HOST,
        port=app_config.PORT,
        reload=app_config.DEBUG,
        log_level=app_config.DEBUG and "debug" or "info",
    )
