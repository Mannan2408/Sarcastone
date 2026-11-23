"""
SarcasTone — Speech-to-Text Transcriber
=========================================
Uses OpenAI Whisper (runs 100% locally, no API key) to transcribe
recorded audio into text.
"""

import numpy as np
import torch

# Whisper import with fallback
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

from src.config import WHISPER_MODEL_SIZE, SAMPLE_RATE


class Transcriber:
    """Transcribes audio to text using Whisper."""

    def __init__(self, model_size: str = WHISPER_MODEL_SIZE):
        if not WHISPER_AVAILABLE:
            print("[Transcriber] ⚠️  Whisper not installed. Transcription disabled.")
            print("              Install with: pip install openai-whisper")
            self.model = None
            return

        print(f"[Transcriber] Loading Whisper ({model_size})...")
        self.model = whisper.load_model(model_size)
        print(f"[Transcriber] Ready.")

    def transcribe(self, audio_path: str = None, audio_array: np.ndarray = None,
                   sr: int = SAMPLE_RATE) -> str:
        """
        Transcribe audio to text.

        Args:
            audio_path: Path to audio file.
            audio_array: Raw audio numpy array (float32, mono).
            sr: Sample rate of audio_array.

        Returns:
            Transcribed text string.
        """
        if self.model is None:
            return "[Whisper not available — transcription disabled]"

        try:
            if audio_array is not None:
                # Whisper expects float32 numpy array at 16kHz
                if sr != 16000:
                    import librosa
                    audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)

                audio_array = audio_array.astype(np.float32)

                # Pad or trim to 30 seconds (Whisper requirement)
                audio_array = whisper.pad_or_trim(audio_array)

                # Create mel spectrogram
                mel = whisper.log_mel_spectrogram(audio_array).to(self.model.device)

                # Decode
                options = whisper.DecodingOptions(language="en", fp16=False)
                result = whisper.decode(self.model, mel, options)
                return result.text.strip()

            elif audio_path is not None:
                result = self.model.transcribe(audio_path, language="en", fp16=False)
                return result["text"].strip()

            else:
                return "[No audio provided]"

        except Exception as e:
            print(f"[Transcriber] Error: {e}")
            return f"[Transcription error: {str(e)}]"


# Singleton
_transcriber_instance = None


def get_transcriber() -> Transcriber:
    global _transcriber_instance
    if _transcriber_instance is None:
        _transcriber_instance = Transcriber()
    return _transcriber_instance
