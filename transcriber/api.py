"""HTTP service.

    uvicorn transcriber.api:app --host 0.0.0.0 --port 8000

Endpoints
    GET  /health
    POST /transcribe          multipart upload, synchronous; for short audio
    POST /jobs                multipart upload, asynchronous; returns 202 + job id
    GET  /jobs/{job_id}       status, progress (chunks done/total) and result

Why two modes: a 3-minute voice note should return in one request, but a 2-hour
meeting would hold an HTTP connection open for minutes and trip load-balancer
timeouts. Long files go through the job API: upload, poll, fetch. Here the job
store is an in-process dict + thread pool; in production that becomes a queue
(Celery/RQ/SQS) + Redis/Postgres for state + object storage for audio, so API
pods and GPU workers can scale independently.

Config via env vars: STT_BACKEND, STT_MODEL, STT_DEVICE, STT_COMPUTE_TYPE,
STT_WORKERS, STT_MAX_UPLOAD_MB, STT_SYNC_MAX_SECONDS.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse

from .audio import SUPPORTED_EXTENSIONS, AudioError, probe
from .backends import get_backend
from .formats import FORMATTERS, MEDIA_TYPES
from .pipeline import PipelineConfig, TranscriptionPipeline

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MAX_UPLOAD_BYTES = int(os.getenv("STT_MAX_UPLOAD_MB", "500")) * 1024 * 1024
SYNC_MAX_SECONDS = float(os.getenv("STT_SYNC_MAX_SECONDS", "600"))
UPLOAD_DIR = Path(tempfile.gettempdir()) / "stt_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# One model instance for the process (loading takes seconds and GBs of RAM);
# a small pool bounds concurrent inference so we don't oversubscribe CPU/GPU.
_executor = ThreadPoolExecutor(max_workers=int(os.getenv("STT_WORKERS", "1")))
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


@lru_cache(maxsize=1)
def _backend():
    name = os.getenv("STT_BACKEND", "faster-whisper")
    if name == "mock":
        return get_backend("mock")
    return get_backend(name, model_size=os.getenv("STT_MODEL", "small"),
                       device=os.getenv("STT_DEVICE", "auto"),
                       compute_type=os.getenv("STT_COMPUTE_TYPE", "default"))


def _save_upload(upload: UploadFile) -> Path:
    """Stream to disk in 1 MB pieces (never hold the whole upload in RAM)."""
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(415, f"Unsupported file type '{suffix or '?'}'. "
                                 f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    dst = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    written = 0
    with dst.open("wb") as f:
        while piece := upload.file.read(1024 * 1024):
            written += len(piece)
            if written > MAX_UPLOAD_BYTES:
                f.close()
                dst.unlink(missing_ok=True)
                raise HTTPException(413, f"File exceeds {MAX_UPLOAD_BYTES // 2**20} MB limit.")
            f.write(piece)
    if written == 0:
        dst.unlink(missing_ok=True)
        raise HTTPException(400, "Empty file.")
    return dst


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _backend()  # load model at boot, not on the first user's request
    yield
    _executor.shutdown(wait=False, cancel_futures=True)
    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)


app = FastAPI(title="Transcription Service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "backend": _backend().name}


@app.post("/transcribe")
def transcribe(file: UploadFile = File(...),
               format: str = Query("json", pattern="^(json|srt|vtt|txt)$"),
               language: str | None = Query(None, description="ISO code; omit to auto-detect")):
    path = _save_upload(file)
    try:
        try:
            info = probe(path)
        except AudioError as e:
            raise HTTPException(422, str(e))
        if info.duration > SYNC_MAX_SECONDS:
            raise HTTPException(413, f"Audio is {info.duration:.0f}s; sync limit is "
                                     f"{SYNC_MAX_SECONDS:.0f}s. Use POST /jobs instead.")
        pipeline = TranscriptionPipeline(_backend(), PipelineConfig(language=language))
        try:
            result = _executor.submit(pipeline.run, path).result()
        except AudioError as e:
            raise HTTPException(422, str(e))
        if format == "json":
            return result.to_dict()
        return PlainTextResponse(FORMATTERS[format](result), media_type=MEDIA_TYPES[format])
    finally:
        path.unlink(missing_ok=True)


def _run_job(job_id: str, path: Path, language: str | None) -> None:
    def progress(done: int, total: int) -> None:
        with _jobs_lock:
            _jobs[job_id]["progress"] = {"chunks_done": done, "chunks_total": total}

    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
    try:
        pipeline = TranscriptionPipeline(_backend(), PipelineConfig(language=language))
        result = pipeline.run(path, progress=progress)
        with _jobs_lock:
            _jobs[job_id].update(status="completed", result=result.to_dict())
    except AudioError as e:
        with _jobs_lock:
            _jobs[job_id].update(status="failed", error=str(e))
    except Exception as e:  # noqa: BLE001 - report, don't crash the worker
        log.exception("job %s failed", job_id)
        with _jobs_lock:
            _jobs[job_id].update(status="failed", error=f"internal error: {e}")
    finally:
        path.unlink(missing_ok=True)


@app.post("/jobs", status_code=202)
def create_job(file: UploadFile = File(...), language: str | None = Query(None)):
    path = _save_upload(file)
    try:
        info = probe(path)  # fail fast on corrupt input, before queueing
    except AudioError as e:
        path.unlink(missing_ok=True)
        raise HTTPException(422, str(e))
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {"id": job_id, "status": "queued", "filename": file.filename,
                         "duration": info.duration, "progress": None,
                         "result": None, "error": None}
    _executor.submit(_run_job, job_id, path, language)
    return {"job_id": job_id, "status": "queued", "status_url": f"/jobs/{job_id}"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str, format: str = Query("json", pattern="^(json|srt|vtt|txt)$")):
    with _jobs_lock:
        job = _jobs.get(job_id)
        job = dict(job) if job else None
    if not job:
        raise HTTPException(404, "Job not found")
    if format != "json" and job["status"] == "completed":
        from .pipeline import OutputSegment, TranscriptionResult
        r = job["result"]
        result = TranscriptionResult(r["text"], r["language"], r["duration"],
                                     [OutputSegment(**s) for s in r["segments"]], r["metadata"])
        return PlainTextResponse(FORMATTERS[format](result), media_type=MEDIA_TYPES[format])
    return job
