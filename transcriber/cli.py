"""Command-line entry point.

    python -m transcriber.cli meeting.mp3                      # JSON to stdout
    python -m transcriber.cli talk.m4a -f srt -o talk.srt      # subtitles
    python -m transcriber.cli long.wav --model medium --language en --chunk 600
    python -m transcriber.cli sample.wav --backend mock        # no model needed
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .audio import AudioError
from .backends import get_backend
from .formats import FORMATTERS
from .pipeline import PipelineConfig, TranscriptionPipeline


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Transcribe an audio file with segment timestamps.")
    p.add_argument("audio", help="Path to audio/video file (wav, mp3, m4a, flac, ogg, ...)")
    p.add_argument("-f", "--format", choices=FORMATTERS, default="json")
    p.add_argument("-o", "--output", help="Write to file instead of stdout")
    p.add_argument("--backend", default="faster-whisper", help="faster-whisper | mock")
    p.add_argument("--model", default="small", help="tiny|base|small|medium|large-v3|turbo")
    p.add_argument("--device", default="auto", help="auto|cpu|cuda")
    p.add_argument("--compute-type", default="default", help="e.g. int8, float16")
    p.add_argument("--language", default=None, help="ISO code (e.g. en, ur); default auto-detect")
    p.add_argument("--chunk", type=float, default=300.0, help="Max chunk length in seconds")
    p.add_argument("--words", action="store_true", help="Include word-level timestamps")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)

    kwargs = {}
    if a.backend != "mock":
        kwargs = dict(model_size=a.model, device=a.device, compute_type=a.compute_type,
                      word_timestamps=a.words)
    backend = get_backend(a.backend, **kwargs)
    pipeline = TranscriptionPipeline(backend, PipelineConfig(
        language=a.language, max_chunk_s=a.chunk, keep_words=a.words))

    def progress(done: int, total: int) -> None:
        if total > 1:
            print(f"  chunk {done}/{total} done", file=sys.stderr)

    try:
        result = pipeline.run(a.audio, progress=progress)
    except AudioError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    rendered = FORMATTERS[a.format](result)
    if a.output:
        Path(a.output).write_text(rendered, encoding="utf-8")
        print(f"wrote {a.output} ({len(result.segments)} segments, "
              f"{result.duration:.1f}s audio, lang={result.language})", file=sys.stderr)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
