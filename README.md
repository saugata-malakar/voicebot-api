# VoiceBot AI - Customer Support Backend

AI-powered Voice Bot for customer support automation.

**Pipeline:** Audio  Whisper ASR  BERT Intent  Response  gTTS  Audio

## Deploy to Render (One-Click)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/saugata-malakar/voicebot-api)

## API Endpoints

| Method | Endpoint | Description |
|--------|---------|-------------|
| GET | /health | Health check |
| GET | /intents | List supported intents |
| GET | /evaluate | Model evaluation metrics |
| GET | /docs | Swagger API documentation |
| POST | /transcribe | Audio  Text (ASR) |
| POST | /predict-intent | Text  Intent |
| POST | /generate-response | Intent  Response |
| POST | /synthesize | Text  Audio (TTS) |
| POST | /voicebot | Full pipeline (Audio  Audio) |

## Tech Stack

- **ASR:** OpenAI Whisper
- **NLP:** BERT Intent Classifier
- **TTS:** Google Text-to-Speech
- **API:** FastAPI + Uvicorn
- **Container:** Docker (Python 3.11)
