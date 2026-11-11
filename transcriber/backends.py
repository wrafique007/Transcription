"""Speech-to-text backends behind one small interface.

The pipeline only depends on `Backend.transcribe(audio, language)`. That keeps
the engine swappable (local Whisper today, a hosted API or a fine-tuned model
tomorrow) and lets the whole pipeline be tested without a GPU or model download
via `MockBackend`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Segment:
    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    words: list[dict] = field(default_factory=list)


@dataclass
class BackendResult:
    segments: list[Segment]
    language: str | None
    language_probability: float | None = None


class Backend(Protocol):
    name: str

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> BackendResult:
        """`audio` is 16 kHz mono float32. Timestamps returned are chunk-relative."""
        ...


class FasterWhisperBackend:
    """Whisper via CTranslate2 (faster-whisper).

    Why faster-whisper over openai-whisper: same model weights and accuracy, but
    ~4x faster and much lower memory thanks to CTranslate2 + int8 quantisation,
    and it runs well on CPU. It also ships Silero VAD (`vad_filter`), which skips
    long silences — both faster and the main fix for Whisper "hallucinating"
    text during silence.
    """

    name = "faster-whisper"

    def __init__(self, model_size: str = "small", device: str = "auto",
                 compute_type: str = "default", beam_size: int = 5,
                 word_timestamps: bool = False):
        from faster_whisper import WhisperModel  # imported lazily: optional dependency

        log.info("Loading faster-whisper model=%s device=%s", model_size, device)
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.beam_size = beam_size
        self.word_timestamps = word_timestamps

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> BackendResult:
        seg_iter, info = self.model.transcribe(
            audio,
            language=language,                 # None => auto-detect
            beam_size=self.beam_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            word_timestamps=self.word_timestamps,
            condition_on_previous_text=False,  # limits repetition loops
        )
        segments = [
            Segment(
                start=s.start, end=s.end, text=s.text,
                avg_logprob=s.avg_logprob, no_speech_prob=s.no_speech_prob,
                words=[{"start": w.start, "end": w.end, "word": w.word,
                        "probability": w.probability} for w in (s.words or [])],
            )
            for s in seg_iter  # generator: decoding actually happens here
        ]
        return BackendResult(segments, info.language, info.language_probability)


class MockBackend:
    """Deterministic stand-in: finds voiced regions by energy, emits placeholder text.

    Produces realistic timestamps for whatever audio it gets, so chunking, offset
    arithmetic, post-processing and output formats can be tested end to end.
    """

    name = "mock"

    def __init__(self, sample_rate: int = 16_000, threshold: float = 0.01,
                 max_segment_s: float = 10.0):
        self.sr = sample_rate
        self.threshold = threshold
        self.max_segment_s = max_segment_s

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> BackendResult:
        frame = int(self.sr * 0.03)
        n = len(audio) // frame
        if n == 0:
            return BackendResult([], language or "en", 1.0)
        rms = np.sqrt((audio[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
        voiced = rms > self.threshold

        segments: list[Segment] = []
        i = 0
        while i < n:
            if not voiced[i]:
                i += 1
                continue
            j = i
            while j < n and voiced[j] and (j - i) * 0.03 < self.max_segment_s:
                j += 1
            start, end = round(i * 0.03, 2), round(j * 0.03, 2)
            if end - start >= 0.2:
                segments.append(Segment(start, end,
                                        f" [mock speech {start:.2f}-{end:.2f}s]",
                                        avg_logprob=-0.2, no_speech_prob=0.01))
            i = j
        return BackendResult(segments, language or "en", 1.0)


def get_backend(name: str, **kwargs) -> Backend:
    name = name.lower()
    if name in ("faster-whisper", "whisper", "fw"):
        return FasterWhisperBackend(**kwargs)
    if name == "mock":
        return MockBackend()
    raise ValueError(f"Unknown backend '{name}'. Use 'faster-whisper' or 'mock'.")
