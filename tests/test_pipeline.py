"""Tests run with the MockBackend, so no model download or GPU is needed.

Synthetic audio: a 440 Hz tone that is ON for 4 s then OFF for 1 s, repeating.
That gives known "speech" regions, so we can assert where segments and chunk
cuts must land.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from transcriber.audio import AudioError, normalize, probe
from transcriber.backends import MockBackend
from transcriber.chunking import frame_energies, plan_chunks
from transcriber.formats import to_srt, to_vtt
from transcriber.pipeline import PipelineConfig, TranscriptionPipeline

TONE = "0.5*sin(2*PI*440*t)*lt(mod(t,5),4)"  # 4 s tone, 1 s silence


def make_audio(path: Path, seconds: float, extra: list[str] | None = None,
               sr: int = 44100, channels: int = 2) -> Path:
    expr = "|".join([TONE] * channels)
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi",
           "-i", f"aevalsrc='{expr}':s={sr}:d={seconds}", *(extra or []), str(path)]
    subprocess.run(cmd, check=True)
    return path


@pytest.fixture
def pipeline():
    return TranscriptionPipeline(MockBackend())


# --- Q1/Q4: accept files in different formats ------------------------------

@pytest.mark.parametrize("name,extra", [
    ("a.wav", []),
    ("a.mp3", ["-c:a", "libmp3lame", "-b:a", "64k"]),
    ("a.flac", []),
    ("a.ogg", ["-c:a", "libvorbis"]),
    ("a.opus", ["-c:a", "libopus"]),
    ("a.m4a", ["-c:a", "aac"]),
])
def test_formats_produce_same_segments(tmp_path, pipeline, name, extra):
    src = make_audio(tmp_path / name, 15, extra)
    r = pipeline.run(src)
    # Tone is on at 0-4, 5-9, 10-14 => 3 segments regardless of container/codec.
    assert len(r.segments) == 3
    for seg, expected_start in zip(r.segments, [0, 5, 10]):
        assert abs(seg.start - expected_start) < 0.15
        assert abs((seg.end - seg.start) - 4) < 0.2
    assert abs(r.duration - 15) < 0.1


def test_normalize_outputs_16k_mono(tmp_path):
    src = make_audio(tmp_path / "in.wav", 3, sr=48000, channels=2)
    out = normalize(src, tmp_path / "out.wav")
    info = probe(out)
    assert (info.sample_rate, info.channels, info.codec) == (16000, 1, "pcm_s16le")


def test_video_container_audio_track_is_used(tmp_path, pipeline):
    src = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "color=c=black:s=64x64:d=10",
                    "-f", "lavfi", "-i", f"aevalsrc='{TONE}':s=44100:d=10",
                    "-c:v", "libx264", "-c:a", "aac", "-shortest", str(src)], check=True)
    assert len(pipeline.run(src).segments) == 2


def test_rejects_unsupported_extension(tmp_path, pipeline):
    bad = tmp_path / "notes.txt"
    bad.write_text("hello")
    with pytest.raises(AudioError, match="Unsupported"):
        pipeline.run(bad)


def test_rejects_corrupt_file(tmp_path, pipeline):
    bad = tmp_path / "broken.mp3"
    bad.write_bytes(b"\x00\x01garbage" * 100)
    with pytest.raises(AudioError):
        pipeline.run(bad)


def test_rejects_file_without_audio(tmp_path, pipeline):
    src = tmp_path / "silent_video.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", "color=c=black:s=64x64:d=2", "-c:v", "libx264", str(src)], check=True)
    with pytest.raises(AudioError, match="no audio"):
        pipeline.run(src)


# --- Q3: timestamps per segment + output formats ---------------------------

def test_segment_timestamps_are_monotonic_and_serialisable(tmp_path, pipeline):
    r = pipeline.run(make_audio(tmp_path / "a.wav", 20))
    starts = [s.start for s in r.segments]
    assert starts == sorted(starts)
    assert all(s.end >= s.start for s in r.segments)
    assert all(a.end <= b.start for a, b in zip(r.segments, r.segments[1:]))
    json.loads(json.dumps(r.to_dict()))
    srt = to_srt(r)
    assert srt.count("-->") == len(r.segments) == 4
    assert "\n2\n00:00:0" in srt and "00:00:00,000 --> 00:00:04," in srt
    assert to_vtt(r).startswith("WEBVTT")


# --- Q5: long audio --------------------------------------------------------

def test_chunk_cuts_land_in_silence(tmp_path):
    wav = normalize(make_audio(tmp_path / "long.wav", 125, channels=1), tmp_path / "n.wav")
    energies, _ = frame_energies(wav)
    chunks = plan_chunks(energies, 125, max_chunk_s=30, search_window_s=6)
    assert len(chunks) >= 4
    for c in chunks[:-1]:
        assert c.duration <= 30 + 1e-6
        assert 4.0 <= c.end % 5 <= 5.0, f"cut at {c.end:.2f}s is inside a tone"
    # contiguous, full coverage
    assert chunks[0].start == 0 and abs(chunks[-1].end - 125) < 0.1
    assert all(abs(a.end - b.start) < 1e-9 for a, b in zip(chunks, chunks[1:]))


def test_long_audio_chunked_matches_unchunked(tmp_path):
    src = make_audio(tmp_path / "long.mp3", 125, ["-c:a", "libmp3lame"], channels=1)
    whole = TranscriptionPipeline(MockBackend(), PipelineConfig(max_chunk_s=1000)).run(src)
    chunked = TranscriptionPipeline(MockBackend(), PipelineConfig(max_chunk_s=30,
                                                                   search_window_s=6)).run(src)
    assert whole.metadata["chunks"] == 1 and chunked.metadata["chunks"] >= 4
    # Same segments, with global (not chunk-relative) timestamps.
    assert len(chunked.segments) == len(whole.segments) == 25
    for a, b in zip(whole.segments, chunked.segments):
        assert abs(a.start - b.start) < 0.1 and abs(a.end - b.end) < 0.1
    assert chunked.segments[-1].start > 115


def test_progress_callback(tmp_path):
    src = make_audio(tmp_path / "a.wav", 70, channels=1)
    seen = []
    TranscriptionPipeline(MockBackend(), PipelineConfig(max_chunk_s=30, search_window_s=6)) \
        .run(src, progress=lambda d, t: seen.append((d, t)))
    assert seen[-1][0] == seen[-1][1] and len(seen) == seen[-1][1]


# --- post-processing -------------------------------------------------------

def test_postprocess_cleans_dedupes_and_flags():
    from transcriber.backends import Segment
    from transcriber.pipeline import postprocess
    raw = [
        (0, Segment(0.0, 2.0, "  Hello   world ", avg_logprob=-0.1, no_speech_prob=0.0)),
        (0, Segment(2.0, 4.0, "hello world", avg_logprob=-0.1, no_speech_prob=0.0)),   # dup
        (0, Segment(4.0, 5.0, "   ", avg_logprob=-0.1, no_speech_prob=0.0)),           # empty
        (0, Segment(5.0, 6.0, "Thanks for watching!", avg_logprob=-1.5, no_speech_prob=0.9)),  # hallucination
        (1, Segment(6.0, 8.0, "mumbled bit", avg_logprob=-1.3, no_speech_prob=0.1)),   # low conf
    ]
    out = postprocess(raw)
    assert [s.text for s in out] == ["Hello world", "mumbled bit"]
    assert out[0].end == 4.0
    assert out[1].low_confidence and not out[0].low_confidence


# --- CLI -------------------------------------------------------------------

def test_cli_writes_srt(tmp_path):
    from transcriber.cli import main
    src = make_audio(tmp_path / "a.ogg", 10, ["-c:a", "libvorbis"])
    out = tmp_path / "a.srt"
    assert main([str(src), "--backend", "mock", "-f", "srt", "-o", str(out)]) == 0
    assert out.read_text().count("-->") == 2


def test_cli_bad_file_exit_code(tmp_path):
    from transcriber.cli import main
    bad = tmp_path / "x.wav"
    bad.write_bytes(b"nope")
    assert main([str(bad), "--backend", "mock"]) == 2
