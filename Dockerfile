FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download FastPitch model at build time (important for cold start)
RUN python3 - <<'EOF'
from TTS.api import TTS
TTS(model_name="tts_models/en/fastpitch", gpu=False)
print("FastPitch model downloaded.")
EOF

# Copy handler
COPY handler.py .

CMD ["python3", "-u", "handler.py"]
