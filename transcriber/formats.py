"""Serialise a TranscriptionResult for different downstream consumers."""
from __future__ import annotations

import json

from .pipeline import TranscriptionResult


def _ts(seconds: float, sep: str) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_json(r: TranscriptionResult) -> str:
    return json.dumps(r.to_dict(), indent=2, ensure_ascii=False)


def to_srt(r: TranscriptionResult) -> str:
    blocks = [f"{i}\n{_ts(s.start, ',')} --> {_ts(s.end, ',')}\n{s.text}\n"
              for i, s in enumerate(r.segments, 1)]
    return "\n".join(blocks)


def to_vtt(r: TranscriptionResult) -> str:
    blocks = [f"{_ts(s.start, '.')} --> {_ts(s.end, '.')}\n{s.text}\n" for s in r.segments]
    return "WEBVTT\n\n" + "\n".join(blocks)


def to_txt(r: TranscriptionResult) -> str:
    return "\n".join(f"[{_ts(s.start, '.')} - {_ts(s.end, '.')}] {s.text}" for s in r.segments) + "\n"


FORMATTERS = {"json": to_json, "srt": to_srt, "vtt": to_vtt, "txt": to_txt}
MEDIA_TYPES = {"json": "application/json", "srt": "application/x-subrip",
               "vtt": "text/vtt", "txt": "text/plain"}
