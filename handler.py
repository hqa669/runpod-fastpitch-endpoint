import base64
import io
import runpod
import torch
import soundfile as sf

from nemo.collections.tts.models import FastPitchModel, HifiGanModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[Init] Using device: {DEVICE}")

# Load models once (cold start only)
print("[Init] Loading FastPitch...")
fastpitch = FastPitchModel.from_pretrained(
    model_name="tts_en_fastpitch"
).to(DEVICE).eval()

print("[Init] Loading HiFi-GAN...")
hifigan = HifiGanModel.from_pretrained(
    model_name="tts_hifigan"
).to(DEVICE).eval()


@torch.no_grad()
def synthesize(text: str) -> bytes:
    """
    Convert text → waveform bytes (WAV)
    """
    parsed = fastpitch.parse(text)
    spectrogram = fastpitch.generate_spectrogram(tokens=parsed)
    audio = hifigan.convert_spectrogram_to_audio(spec=spectrogram)

    audio = audio.squeeze().cpu().numpy()

    buffer = io.BytesIO()
    sf.write(buffer, audio, samplerate=22050, format="WAV")
    buffer.seek(0)
    return buffer.read()


def handler(event):
    """
    RunPod handler
    Input:
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

        wav_bytes = synthesize(text)
        audio_base64 = base64.b64encode(wav_bytes).decode("utf-8")

        return {
            "audio_base64": audio_base64,
            "format": "wav",
            "sample_rate": 22050,
            "model": "fastpitch"
        }

    except Exception as e:
        return {
            "error": str(e)
        }


runpod.serverless.start({"handler": handler})
