FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6"

# System deps
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------
# 🔥 PRELOAD MODELS AT BUILD TIME
# -------------------------------
RUN python3 - <<'EOF'
import torch
from nemo.collections.tts.models import FastPitchModel, HifiGanModel

print("Downloading FastPitch...")
FastPitchModel.from_pretrained("tts_en_fastpitch")

print("Downloading HiFi-GAN...")
HifiGanModel.from_pretrained("tts_hifigan")

print("Models downloaded successfully.")
EOF

# Copy handler last (best cache usage)
COPY handler.py .

CMD ["python3", "-u", "handler.py"]
