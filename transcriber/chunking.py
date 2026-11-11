"""Long-audio handling: silence-aware chunking with bounded memory.

Problem: a 2-hour recording at 16 kHz is ~115M samples. Loading it all at once,
or sending it to a model in one call, means high memory use, no progress
reporting, no partial retries and (for hosted APIs) hard size limits — OpenAI's
API caps uploads at 25 MB, for example.

Approach:
1. Stream through the normalised WAV once, computing RMS energy for every
   30 ms frame. Only the energy array is kept (~120k floats per hour = <1 MB).
2. Walk forward in steps of `max_chunk_s`. Rather than cutting blindly at the
   limit (which would split a word in half), look back over a search window
   (default: the last 20 s before the limit) and cut at the *quietest* point,
   i.e. a pause between words/sentences.
3. Each chunk is read lazily from disk only when it is about to be transcribed,
   so peak memory is one chunk (~10 MB for 5 minutes), whatever the file length.
4. Every chunk carries its absolute start offset; the pipeline adds it to the
   model's chunk-relative timestamps, so final timestamps are global.

Because cuts land in silence, we don't need overlapping windows and the fiddly
de-duplication of words transcribed twice that overlap requires.
"""
from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

FRAME_MS = 30


@dataclass(frozen=True)
class Chunk:
    index: int
    start: float  # seconds, absolute
    end: float    # seconds, absolute

    @property
    def duration(self) -> float:
        return self.end - self.start


def frame_energies(wav_path: str | Path, frame_ms: int = FRAME_MS,
                   block_frames: int = 2000) -> tuple[np.ndarray, int]:
    """Return per-frame RMS energy (float32) and the sample rate, streaming."""
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        assert wf.getsampwidth() == 2 and wf.getnchannels() == 1, "expects normalised WAV"
        frame_len = int(sr * frame_ms / 1000)
        out: list[np.ndarray] = []
        while True:
            raw = wf.readframes(frame_len * block_frames)
            if not raw:
                break
            x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            n = len(x) // frame_len
            if n:
                frames = x[: n * frame_len].reshape(n, frame_len)
                out.append(np.sqrt((frames ** 2).mean(axis=1)))
            rem = len(x) - n * frame_len
            if rem:  # trailing partial frame (only at EOF)
                tail = x[n * frame_len:]
                out.append(np.array([np.sqrt((tail ** 2).mean())], dtype=np.float32))
    energies = np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
    return energies, sr


def plan_chunks(energies: np.ndarray, total_duration: float,
                max_chunk_s: float = 300.0, search_window_s: float = 20.0,
                min_chunk_s: float = 5.0, frame_ms: int = FRAME_MS) -> list[Chunk]:
    """Choose cut points at the quietest moment near each max-length boundary."""
    if total_duration <= max_chunk_s:
        return [Chunk(0, 0.0, total_duration)]

    fs = frame_ms / 1000.0
    # Smooth energy a little so a single quiet frame mid-word isn't chosen.
    kernel = np.ones(5, dtype=np.float32) / 5
    smooth = np.convolve(energies, kernel, mode="same") if len(energies) >= 5 else energies

    chunks: list[Chunk] = []
    start = 0.0
    while total_duration - start > max_chunk_s:
        hard_end = start + max_chunk_s
        win_lo = max(start + min_chunk_s, hard_end - search_window_s)
        lo, hi = int(win_lo / fs), int(hard_end / fs)
        if hi > lo and hi <= len(smooth):
            cut = (lo + int(np.argmin(smooth[lo:hi])) + 0.5) * fs  # centre of quietest frame
        else:
            cut = hard_end
        chunks.append(Chunk(len(chunks), start, cut))
        start = cut

    # Fold a tiny tail into the previous chunk rather than transcribing 1 s alone.
    if total_duration - start < min_chunk_s and chunks:
        last = chunks.pop()
        chunks.append(Chunk(last.index, last.start, total_duration))
    else:
        chunks.append(Chunk(len(chunks), start, total_duration))
    return chunks


def read_chunk(wav_path: str | Path, chunk: Chunk) -> np.ndarray:
    """Lazily read one chunk as float32 in [-1, 1] (the format Whisper expects)."""
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        start_f = int(round(chunk.start * sr))
        n = int(round(chunk.end * sr)) - start_f
        wf.setpos(min(start_f, wf.getnframes()))
        raw = wf.readframes(max(n, 0))
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def iter_chunks(wav_path: str | Path, chunks: list[Chunk]) -> Iterator[tuple[Chunk, np.ndarray]]:
    for c in chunks:
        yield c, read_chunk(wav_path, c)
