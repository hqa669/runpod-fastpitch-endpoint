FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    git \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel Cython

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Preload models
RUN python3 - <<'EOF'
from nemo.collections.tts.models import FastPitchModel, HifiGanModel
FastPitchModel.from_pretrained("tts_en_fastpitch")
HifiGanModel.from_pretrained("tts_hifigan")
EOF

COPY handler.py .

CMD ["python3", "-u", "handler.py"]
