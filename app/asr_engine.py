"""
HearLink ASL — ASR Engine
Streaming Automatic Speech Recognition using Faster-Whisper.
"""

import io
import time
import threading
from typing import Callable, Optional

import numpy as np

try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    HAS_FASTER_WHISPER = False
    print("[ASREngine] faster-whisper not installed. Using mock ASR.")


class ASREngine:
    """
    Streaming ASR engine wrapping Faster-Whisper.
    Accumulates audio chunks and transcribes on speech boundaries.
    """

    def __init__(self, model_size: str = "small.en",
                 compute_type: str = "int8",
                 language: str = "en",
                 vad_threshold: float = 0.5,
                 min_chunk_duration: float = 1.0,
                 max_chunk_duration: float = 10.0):
        """
        Initialize the ASR engine.

        Args:
            model_size: Whisper model size ('tiny.en', 'base.en', 'small.en', etc.)
            compute_type: Compute precision ('int8', 'float16', 'float32')
            language: Language code
            vad_threshold: Voice Activity Detection threshold
            min_chunk_duration: Min audio duration before transcription (seconds)
            max_chunk_duration: Max audio buffer before forced transcription
        """
        self.model_size = model_size
        self.language = language
        self.vad_threshold = vad_threshold
        self.min_chunk_duration = min_chunk_duration
        self.max_chunk_duration = max_chunk_duration
        self.sample_rate = 16000

        # Audio buffer
        self.audio_buffer = np.array([], dtype=np.float32)
        self.silence_frames = 0
        self.speech_frames = 0
        self.is_speaking = False

        # Model
        self.model = None
        if HAS_FASTER_WHISPER:
            try:
                print(f"[ASREngine] Loading Whisper model: {model_size} ({compute_type})")
                self.model = WhisperModel(
                    model_size,
                    compute_type=compute_type,
                    device="auto",
                    download_root="app/models/whisper",
                )
                print("[ASREngine] Model loaded successfully")
            except Exception as e:
                print(f"[ASREngine] Failed to load model: {e}")
                self.model = None

        # Lock for thread safety
        self._lock = threading.Lock()

    def process_audio_chunk(self, audio_data: bytes,
                             sample_rate: int = 16000) -> Optional[dict]:
        """
        Process an incoming audio chunk.

        Args:
            audio_data: Raw PCM audio bytes (16-bit mono)
            sample_rate: Sample rate of the audio

        Returns:
            Transcription result dict or None if not enough audio
        """
        with self._lock:
            # Convert bytes to float32
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Resample if necessary
            if sample_rate != self.sample_rate:
                ratio = self.sample_rate / sample_rate
                new_length = int(len(audio_array) * ratio)
                indices = np.linspace(0, len(audio_array) - 1, new_length)
                audio_array = np.interp(indices, np.arange(len(audio_array)), audio_array)

            # Add to buffer
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_array])

            # Calculate buffer duration
            buffer_duration = len(self.audio_buffer) / self.sample_rate

            # Simple energy-based VAD
            energy = np.sqrt(np.mean(audio_array ** 2))
            is_speech = energy > 0.01  # Simple threshold

            if is_speech:
                self.speech_frames += 1
                self.silence_frames = 0
                self.is_speaking = True
            else:
                self.silence_frames += 1

            # Determine if we should transcribe
            should_transcribe = False

            # Force transcription at max duration
            if buffer_duration >= self.max_chunk_duration:
                should_transcribe = True

            # Transcribe on speech boundary (silence after speech)
            elif (self.is_speaking and
                  self.silence_frames > 10 and  # ~300ms of silence
                  buffer_duration >= self.min_chunk_duration):
                should_transcribe = True
                self.is_speaking = False

            if should_transcribe:
                return self._transcribe()

        return None

    def _transcribe(self) -> dict:
        """Transcribe the current audio buffer."""
        audio = self.audio_buffer.copy()
        self.audio_buffer = np.array([], dtype=np.float32)
        self.speech_frames = 0
        self.silence_frames = 0

        if len(audio) < self.sample_rate * 0.3:  # Less than 300ms
            return {"text": "", "segments": [], "is_partial": False}

        if self.model is not None:
            return self._transcribe_whisper(audio)
        else:
            return self._transcribe_mock(audio)

    def _transcribe_whisper(self, audio: np.ndarray) -> dict:
        """Transcribe using Faster-Whisper."""
        start_time = time.perf_counter()

        segments_gen, info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(
                threshold=self.vad_threshold,
                min_speech_duration_ms=250,
                min_silence_duration_ms=300,
            ),
        )

        segments = []
        full_text = []
        for segment in segments_gen:
            segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "words": [
                    {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                    for w in (segment.words or [])
                ],
            })
            full_text.append(segment.text.strip())

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        text = " ".join(full_text).strip()

        return {
            "text": text,
            "segments": segments,
            "language": info.language if hasattr(info, 'language') else self.language,
            "latency_ms": elapsed_ms,
            "is_partial": False,
        }

    def _transcribe_mock(self, audio: np.ndarray) -> dict:
        """Mock transcription for testing without Whisper installed."""
        duration = len(audio) / self.sample_rate
        energy = np.sqrt(np.mean(audio ** 2))

        # Generate mock text based on audio energy
        if energy > 0.05:
            text = "hello how are you"
        elif energy > 0.02:
            text = "thank you"
        elif energy > 0.01:
            text = "yes"
        else:
            text = ""

        return {
            "text": text,
            "segments": [{"start": 0, "end": duration, "text": text}] if text else [],
            "language": self.language,
            "latency_ms": 10.0,
            "is_partial": False,
            "mock": True,
        }

    def flush(self) -> Optional[dict]:
        """Force transcription of remaining audio buffer."""
        with self._lock:
            if len(self.audio_buffer) > self.sample_rate * 0.3:
                return self._transcribe()
        return None

    def reset(self):
        """Clear the audio buffer and reset state."""
        with self._lock:
            self.audio_buffer = np.array([], dtype=np.float32)
            self.silence_frames = 0
            self.speech_frames = 0
            self.is_speaking = False
