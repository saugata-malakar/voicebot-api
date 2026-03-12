FROM python:3.11-slim

# Install system dependencies (ffmpeg for whisper audio processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install setuptools with pkg_resources (removed in setuptools>=75)
RUN pip install --no-cache-dir --upgrade pip "setuptools<75.0" wheel

# Install openai-whisper first WITHOUT build isolation so it uses our setuptools
RUN pip install --no-cache-dir --no-build-isolation openai-whisper==20231117

# Copy and install remaining requirements
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
