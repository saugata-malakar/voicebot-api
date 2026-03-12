FROM python:3.11-slim

# Install system dependencies (ffmpeg for whisper audio processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (Docker cache layer)
COPY requirements-cloud.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p models audio_samples data temp logs

# Expose port (Render sets PORT env var)
EXPOSE 8000

# Start server - Render provides PORT env var
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
