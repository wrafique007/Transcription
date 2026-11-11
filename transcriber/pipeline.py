"""End-to-end pipeline: validate -> normalise -> chunk -> transcribe -> post-process.

    input file ──► probe (ffprobe) ──► normalise to 16k mono WAV (ffmpeg)
                                              │
                         energy scan ◄────────┘
                              │
                     plan silence-aware chunks
                              │
            for each chunk: read lazily ─► backend.transcribe()
                              │             (chunk-relative timestamps)
                     + chunk.start offset ─► global timestamps
                              │
               post-process (clean, de-dupe, confidence flags)
                              │
                 TranscriptionResult ─► JSON / SRT / VTT / TXT
"""
from __future__ import annotations

import logging
import math
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from . import audio as audio_mod
from .backends import Backend, Segment
from .chunking import frame_energies, iter_chunks, plan_chunks

log = logging.getLogger(__name__)

LOW_CONFIDENCE_LOGPROB = -1.0
HALLUCINATION_NO_SPEECH = 0.6

ProgressFn = Callable[[int, int], None]  # (chunks_done, chunks_total)


@dataclass
class OutputSegment:
    id: int
    start: float
    end: float
    text: str
    confidence: float | None      # exp(avg_logprob): rough 0..1 score
    low_confidence: bool
    chunk: int
    words: list[dict] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    text: str
    language: str | None
    duration: float
    segments: list[OutputSegment]
    metadata: dict

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PipelineConfig:
    language: str | None = None   # None => auto-detect (on first chunk, then pinned)
    max_chunk_s: float = 300.0
    search_window_s: float = 20.0
    keep_words: bool = False


class TranscriptionPipeline:
    def __init__(self, backend: Backend, config: PipelineConfig | None = None):
        self.backend = backend
        self.config = config or PipelineConfig()

    def run(self, path: str | Path, progress: ProgressFn | None = None) -> TranscriptionResult:
        t0 = time.perf_counter()
        info = audio_mod.probe(path)
        log.info("Input %s: %s/%s %dHz %dch %.1fs", info.path.name, info.format_name,
                 info.codec, info.sample_rate, info.channels, info.duration)

        with tempfile.TemporaryDirectory(prefix="stt_") as tmp:
            wav = audio_mod.normalize(path, Path(tmp) / "normalized.wav")
            energies, _ = frame_energies(wav)
            duration = len(energies) * 0.03 if len(energies) else info.duration
            duration = min(duration, info.duration) if info.duration else duration
            chunks = plan_chunks(energies, duration, self.config.max_chunk_s,
                                 self.config.search_window_s)
            log.info("Planned %d chunk(s)", len(chunks))

            language = self.config.language
            raw: list[tuple[int, Segment]] = []
            for chunk, samples in iter_chunks(wav, chunks):
                result = self.backend.transcribe(samples, language=language)
                # Pin the detected language after the first chunk: avoids a later
                # chunk of music/silence being mis-detected as another language.
                if language is None and result.language:
                    language = result.language
                for s in result.segments:
                    s.start = min(s.start + chunk.start, chunk.end)
                    s.end = min(s.end + chunk.start, chunk.end)
                    for w in s.words:
                        w["start"] = round(w["start"] + chunk.start, 3)
                        w["end"] = round(w["end"] + chunk.start, 3)
                    raw.append((chunk.index, s))
                if progress:
                    progress(chunk.index + 1, len(chunks))

        segments = postprocess(raw, keep_words=self.config.keep_words)
        elapsed = time.perf_counter() - t0
        return TranscriptionResult(
            text=" ".join(s.text for s in segments),
            language=language,
            duration=round(duration, 3),
            segments=segments,
            metadata={
                "source_file": info.path.name,
                "source_format": info.format_name,
                "source_codec": info.codec,
                "source_sample_rate": info.sample_rate,
                "source_channels": info.channels,
                "backend": self.backend.name,
                "chunks": len(chunks),
                "processing_seconds": round(elapsed, 3),
                "real_time_factor": round(elapsed / duration, 4) if duration else None,
                "low_confidence_segments": sum(s.low_confidence for s in segments),
            },
        )


_WS = re.compile(r"\s+")


def postprocess(raw: list[tuple[int, Segment]], keep_words: bool = False) -> list[OutputSegment]:
    """Make model output safe for downstream consumers.

    - normalise whitespace, drop empty segments
    - drop likely hallucinations (model says "no speech" AND is unsure of text)
    - collapse consecutive duplicate segments (a known Whisper failure mode)
    - enforce monotonic, non-overlapping timestamps, rounded to ms
    - attach a confidence score + low_confidence flag for human review queues
    """
    out: list[OutputSegment] = []
    last_end = 0.0
    for chunk_idx, s in sorted(raw, key=lambda t: t[1].start):
        text = _WS.sub(" ", s.text).strip()
        if not text:
            continue
        if (s.no_speech_prob or 0) > HALLUCINATION_NO_SPEECH and \
                (s.avg_logprob if s.avg_logprob is not None else 0) < LOW_CONFIDENCE_LOGPROB:
            continue
        if out and text.lower() == out[-1].text.lower():
            out[-1].end = round(max(out[-1].end, s.end), 3)
            continue

        start = round(max(s.start, last_end), 3)
        end = round(max(s.end, start), 3)
        conf = round(math.exp(s.avg_logprob), 3) if s.avg_logprob is not None else None
        out.append(OutputSegment(
            id=len(out), start=start, end=end, text=text, confidence=conf,
            low_confidence=s.avg_logprob is not None and s.avg_logprob < LOW_CONFIDENCE_LOGPROB,
            chunk=chunk_idx, words=s.words if keep_words else [],
        ))
        last_end = end
    return out
