"""Audio ingestion: validation, probing and normalisation.

Design decision: instead of supporting N formats inside the pipeline, every input
is converted ONCE, at the edge, into a single canonical format:

    16 kHz, mono, 16-bit signed PCM WAV

That is exactly what Whisper-family models consume internally, so nothing
downstream (chunking, VAD, the model) ever needs to know whether the user sent
MP3, M4A, OGG, a video file, or a 48 kHz stereo WAV. ffmpeg does the decoding,
resampling and down-mixing, which gives us every format ffmpeg supports for free.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2  # int16

# Allow-list of extensions we accept. ffmpeg can decode far more, but an explicit
# list gives clear error messages and avoids feeding arbitrary files to a decoder.
SUPPORTED_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus",
    ".webm", ".wma", ".amr", ".aiff", ".aif", ".mp4", ".mov", ".mkv",
}

MAX_DURATION_SECONDS = 4 * 60 * 60  # 4h hard limit; configurable guardrail


class AudioError(Exception):
    """Raised for user-facing audio problems (bad format, corrupt file, ...)."""


@dataclass(frozen=True)
class AudioInfo:
    path: Path
    format_name: str
    codec: str
    sample_rate: int
    channels: int
    duration: float  # seconds


def _require(binary: str) -> str:
    found = shutil.which(binary)
    if not found:
        raise RuntimeError(f"'{binary}' not found on PATH; install ffmpeg.")
    return found


def probe(path: str | Path) -> AudioInfo:
    """Inspect a file with ffprobe. We trust the *content*, not the extension."""
    path = Path(path)
    if not path.is_file():
        raise AudioError(f"File not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise AudioError(
            f"Unsupported file extension '{path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    cmd = [
        _require("ffprobe"), "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", "-select_streams", "a", str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AudioError(f"Could not read audio (corrupt or not media): {proc.stderr.strip()}")

    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams") or []
    if not streams:
        raise AudioError("File contains no audio stream.")
    stream, fmt = streams[0], data.get("format", {})

    duration = float(stream.get("duration") or fmt.get("duration") or 0.0)
    if duration <= 0:
        raise AudioError("Audio has zero/unknown duration.")
    if duration > MAX_DURATION_SECONDS:
        raise AudioError(f"Audio is {duration/3600:.1f}h; limit is {MAX_DURATION_SECONDS/3600:.0f}h.")

    return AudioInfo(
        path=path,
        format_name=fmt.get("format_name", "unknown"),
        codec=stream.get("codec_name", "unknown"),
        sample_rate=int(stream.get("sample_rate") or 0),
        channels=int(stream.get("channels") or 0),
        duration=duration,
    )


def normalize(src: str | Path, dst: str | Path) -> Path:
    """Decode any supported input into 16 kHz mono s16le WAV.

    -vn          drop any video track (mp4/mov/webm uploads)
    -ac 1        down-mix to mono
    -ar 16000    resample to the model's native rate
    -sample_fmt  16-bit PCM (half the size of float32, lossless for speech)
    """
    dst = Path(dst)
    cmd = [
        _require("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src), "-vn",
        "-ac", str(TARGET_CHANNELS), "-ar", str(TARGET_SAMPLE_RATE),
        "-sample_fmt", "s16", "-f", "wav", str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dst.exists():
        raise AudioError(f"ffmpeg failed to decode audio: {proc.stderr.strip()}")
    return dst
