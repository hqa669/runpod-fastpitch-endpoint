import runpod
import torch
import io
import base64
import soundfile as sf
from TTS.api import TTS

# -------------------------
# Device setup
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# -------------------------
# Load FastPitch model ONCE
# -------------------------
tts = TTS("tts_models/en/ljspeech/fast_pitch").to(DEVICE)

# -------------------------
# RunPod handler
# -------------------------
def handler(event):
    """
    Expected request format:
    {
      "input": {
        "text": "Hello world"
      }
    }
    """

    # ✅ Read input text safely
    input_data = event.get("input", {})
    text = input_data.get("text")

    if not text or not isinstance(text, str):
        return {
            "error": "Missing or invalid 'text' field in input."
        }

    # ✅ Generate speech
    wav = tts.tts(text)

    # ✅ Encode WAV to Base64 in-memory
    buffer = io.BytesIO()
    sf.write(buffer, wav, samplerate=22050, format="WAV")

    audio_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "audio_base64": audio_base64,
        "sample_rate": 22050,
        "format": "wav",
        "model": "fast_pitch"
    }

# -------------------------
# IMPORTANT: block forever
# -------------------------
runpod.serverless.start({"handler": handler})
