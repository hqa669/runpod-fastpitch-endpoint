import base64
import io

import runpod
import soundfile as sf
from TTS.api import TTS


# Load model once (cold start only)
tts = TTS(
    model_name="tts_models/en/fastpitch",
    gpu=True
)


def handler(event):
    """
    Expected input:
    {
        "input": {
            "text": "Hello world"
        }
    }
    """
    try:
        text = event["input"].get("text", "").strip()
        if not text:
            raise ValueError("Input 'text' is required")

        # Generate waveform (numpy array)
        wav = tts.tts(text)

        # Write WAV to memory
        buffer = io.BytesIO()
        sf.write(buffer, wav, samplerate=22050, format="WAV")
        buffer.seek(0)

        return {
            "audio_base64": base64.b64encode(buffer.read()).decode("utf-8"),
            "format": "wav",
            "sample_rate": 22050,
            "model": "coqui_fastpitch"
        }

    except Exception as e:
        return {
            "error": str(e)
        }


runpod.serverless.start({"handler": handler})
