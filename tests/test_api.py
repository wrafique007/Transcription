"""API tests (skipped automatically if fastapi/httpx aren't installed)."""
from __future__ import annotations

import os
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
os.environ["STT_BACKEND"] = "mock"

from fastapi.testclient import TestClient  # noqa: E402

from transcriber.api import app  # noqa: E402

from .test_pipeline import make_audio  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "backend": "mock"}


def test_sync_transcribe_json(client, tmp_path):
    f = make_audio(tmp_path / "a.mp3", 10, ["-c:a", "libmp3lame"])
    r = client.post("/transcribe", files={"file": ("a.mp3", f.open("rb"), "audio/mpeg")})
    assert r.status_code == 200
    body = r.json()
    assert len(body["segments"]) == 2 and body["segments"][1]["start"] > 4.5


def test_sync_transcribe_srt(client, tmp_path):
    f = make_audio(tmp_path / "a.wav", 10)
    r = client.post("/transcribe?format=srt", files={"file": ("a.wav", f.open("rb"))})
    assert r.status_code == 200 and r.text.count("-->") == 2


def test_rejects_bad_type_and_corrupt(client, tmp_path):
    assert client.post("/transcribe", files={"file": ("x.txt", b"hi")}).status_code == 415
    assert client.post("/transcribe", files={"file": ("x.wav", b"junk" * 50)}).status_code == 422


def test_async_job_flow(client, tmp_path):
    f = make_audio(tmp_path / "a.flac", 20)
    r = client.post("/jobs", files={"file": ("a.flac", f.open("rb"))})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    for _ in range(100):
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            break
        time.sleep(0.05)
    assert job["status"] == "completed", job
    assert len(job["result"]["segments"]) == 4
    assert client.get(f"/jobs/{job_id}?format=vtt").text.startswith("WEBVTT")
    assert client.get("/jobs/nope").status_code == 404
