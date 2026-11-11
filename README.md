# Transcription Pipeline

A small, production-minded speech-to-text pipeline. It takes an audio (or video)
file in any common format, transcribes it with **Whisper** (via `faster-whisper`), and
returns clean text with **per-segment timestamps** as JSON, SRT, VTT or plain text.
It runs as a **CLI** or an **HTTP service** (FastAPI), and has sync and async-job modes.

```
input file ──► validate + probe (ffprobe) ──► normalise → 16 kHz mono PCM WAV (ffmpeg)
                                                      │
                                     streaming energy scan (30 ms frames)
                                                      │
                                 plan silence-aware chunks (≤ 5 min each)
                                                      │
              for each chunk: lazy read ─► backend.transcribe()  (chunk-relative times)
                                                      │
                                   + chunk offset ─► global timestamps
                                                      │
                    post-process: clean, de-dupe, drop hallucinations, confidence flags
                                                      │
                                 TranscriptionResult ─► JSON / SRT / VTT / TXT
```

> **Part 2 (system design)** is in [`SYSTEM_DESIGN.md`](SYSTEM_DESIGN.md).

## Project layout

| File | Responsibility |
|---|---|
| `transcriber/audio.py` | Validate input, probe with ffprobe, normalise with ffmpeg (**Q1, Q4**) |
| `transcriber/chunking.py` | Streaming energy scan, silence-aware chunk planning, lazy chunk reads (**Q5**) |
| `transcriber/backends.py` | `Backend` interface, `FasterWhisperBackend`, `MockBackend` (**Q2**) |
| `transcriber/pipeline.py` | Orchestration, timestamp offsetting, post-processing (**Q3**) |
| `transcriber/formats.py` | JSON / SRT / VTT / TXT serialisers |
| `transcriber/cli.py` | Command-line entry point |
| `transcriber/api.py` | FastAPI service: `/transcribe` (sync), `/jobs` (async) |
| `tests/` | 18 pipeline tests + API tests, all using real ffmpeg-generated audio |

## Quick start

```bash
# prerequisites: Python 3.10+ and ffmpeg on PATH
pip install -r requirements.txt

# CLI
python -m transcriber.cli meeting.mp3                          # JSON to stdout
python -m transcriber.cli talk.m4a -f srt -o talk.srt          # subtitles
python -m transcriber.cli interview.wav --model medium --language en --words
python -m transcriber.cli sample.wav --backend mock            # no model download

# Service
uvicorn transcriber.api:app --port 8000
curl -F file=@speech.m4a "localhost:8000/transcribe?format=json"
curl -F file=@2h_podcast.mp3 localhost:8000/jobs               # -> {"job_id": "..."}
curl localhost:8000/jobs/<job_id>                              # status + progress + result
curl "localhost:8000/jobs/<job_id>?format=srt"

# Docker
docker build -t stt . && docker run -p 8000:8000 stt

# Tests (mock backend: no GPU or model download needed)
pytest -q
```

### Example output (illustrative shape, abridged)

```json
{
  "text": "Welcome everyone. Let's get started with the quarterly review.",
  "language": "en",
  "duration": 4200.0,
  "segments": [
    {"id": 0, "start": 0.0, "end": 2.34, "text": "Welcome everyone.",
     "confidence": 0.91, "low_confidence": false, "chunk": 0, "words": []},
    {"id": 1, "start": 2.8, "end": 5.6, "text": "Let's get started with the quarterly review.",
     "confidence": 0.88, "low_confidence": false, "chunk": 0, "words": []}
  ],
  "metadata": {"source_format": "mp3", "source_sample_rate": 44100, "source_channels": 2,
               "backend": "faster-whisper", "chunks": 15, "processing_seconds": 412.7,
               "real_time_factor": 0.098, "low_confidence_segments": 3}
}
```

---

## Answers to the questions

### 1. A service/script that accepts an audio file (WAV or MP3)

There are two entry points over the same `TranscriptionPipeline`:

- **CLI**, `python -m transcriber.cli <file>`: for batch jobs and local use.
- **HTTP**, `POST /transcribe` / `POST /jobs` (multipart upload): for other services.

Input is validated before any expensive work happens:
- an extension allow-list gives a clear `415` for things like `.txt`, `.pdf`
- **ffprobe checks the actual content**, so a renamed or corrupt file gets a `422`
  with ffmpeg's reason instead of a crash deep inside the model
- files with no audio stream, zero duration, or over the duration limit are rejected
- uploads are **streamed to disk in 1 MB pieces** with a size cap, so a 500 MB upload never sits in RAM

### 2. A service/script that transcribes speech into text

**Engine choice: Whisper via `faster-whisper`.**
- Whisper is open-source, multilingual (~99 languages, including Urdu), handles accents
  and noise well, and detects the language automatically.
- `faster-whisper` uses the same weights but runs on CTranslate2: about 4× faster, less memory, and int8
  quantisation, so it works on CPU, which matters for cost.
- It includes **Silero VAD** (`vad_filter=True`), which skips silence. That speeds things up and
  is the main fix for Whisper "hallucinating" phrases like *"Thanks for watching!"* during silence.
- `condition_on_previous_text=False` limits Whisper's known repetition loops.

**The engine sits behind an interface.** The pipeline only calls
`backend.transcribe(audio_16k_float32, language) -> segments`. To switch to a hosted API
(OpenAI, Deepgram, AssemblyAI) or a fine-tuned model, you add a single class. A `MockBackend`
returns realistic timestamps without a model, so CI tests the whole pipeline in seconds.

The model is **loaded once per process** (it takes seconds and GBs) and warmed at startup. A
bounded worker pool caps concurrent inference so the CPU/GPU isn't oversubscribed.

**Model size is a config knob** (`--model tiny|base|small|medium|large-v3|turbo`), trading
accuracy against speed. `small` is the default as a CPU-friendly middle ground. On a GPU I'd use
`large-v3-turbo`.

### 3. Return the transcription with timestamps per segment

Every segment has `start`/`end` (seconds, ms precision), `text`, a `confidence` score
(`exp(avg_logprob)`), a `low_confidence` flag, and the source `chunk`. `--words` adds
word-level timestamps.

The post-processing step is there to make the output **safe for downstream use**:
- **Global timestamps**: chunk-relative model times + chunk offset, clamped to chunk bounds
- **Monotonic and non-overlapping**, so subtitle players and alignment tools don't break
- whitespace normalised, empty segments dropped
- **hallucination filter**: drops segments where the model reports high `no_speech_prob`
  *and* low confidence
- **consecutive duplicate collapse**, for Whisper's repetition failure mode
- **`low_confidence` flag**, so a product can send those spans to human review instead of
  silently trusting them

There are output formats for different consumers: **JSON** (APIs, search indexing, LLM
summarisation), **SRT/VTT** (video players, captions), **TXT** (people reading).

### 4. How do you handle different audio formats?

**Normalise once, at the edge, into one canonical format: 16 kHz, mono, 16-bit PCM WAV.**

```
ffmpeg -i <anything> -vn -ac 1 -ar 16000 -sample_fmt s16 -f wav normalized.wav
```

- **Why ffmpeg:** it decodes essentially every codec and container (MP3, AAC/M4A, FLAC,
  OGG/Vorbis, Opus, WebM, WMA, AMR, AIFF), including **audio tracks inside video**
  (MP4/MOV/MKV, where `-vn` drops the video). We get all of that without writing format-specific code.
- **Why 16 kHz mono:** that's what Whisper uses internally. Resampling once, explicitly,
  means the chunker, VAD and model all see identical input. It also makes behaviour
  **deterministic across formats**. The tests check that WAV, MP3, FLAC, OGG, Opus and M4A versions of
  the same audio give the same segments.
- **Stereo → mono** down-mix: speech recognition doesn't benefit from stereo, and it halves the data.
  (For call-centre recordings with one speaker per channel, I'd split the channels instead of
  mixing them. That's an easy way to get diarisation.)
- **int16 rather than float32** on disk: half the size, and lossless for speech.
- **Trust content, not extension:** ffprobe reads the real codec/sample rate/channels.
  These are reported in `metadata` for debugging ("why is this file bad?" usually turns out to be
  8 kHz telephone audio).
- Clear, distinct errors for unsupported type, corrupt file, no audio stream, and too long.

### 5. How do you deal with long audio files?

The risks with a 2-hour file: running out of memory, HTTP timeouts, no progress
visibility, losing all the work on one failure, and hard limits on hosted APIs (e.g. a
25 MB upload cap). The approach has four parts:

**a) Silence-aware chunking with bounded memory** (`chunking.py`)
1. Stream through the normalised WAV once, keeping only RMS energy per 30 ms frame
   (about 120k floats per hour, under 1 MB).
2. Chunk targets are ≤ 5 min (configurable). Instead of cutting blindly at 5:00, which
   could split a word, it looks back over the last 20 s and **cuts at the quietest point**,
   i.e. a natural pause.
3. Chunks are **read lazily** right before inference, so peak memory is one chunk
   (~10 MB of audio) however long the file is.
4. Each chunk's absolute start offset is added to the model's timestamps, giving **global
   timestamps**.

Because cuts land in silence, **overlapping windows aren't needed**. That avoids the
fiddly work of de-duplicating words transcribed twice in an overlap. The test suite checks
that cuts land in silence, and that a chunked and an unchunked run of the same file give identical
segments and timestamps.

*Measured:* a **70-minute MP3** ran through the pipeline in 15 chunks with **~110 MB peak
RSS** (mock backend, so this measures the pipeline's own overhead, not the model's).

**b) Language pinning:** the language is auto-detected on the first chunk and then fixed, so a
later chunk of music or silence can't be mis-detected as another language.

**c) Async job API for the HTTP service:** short files (< 10 min, configurable) use the sync
`POST /transcribe`. Longer files are refused there with a pointer to `POST /jobs`, which returns
`202 + job_id` right away. Clients poll `GET /jobs/{id}` for status and
**progress (`chunks_done / chunks_total`)**, then fetch the result in any format. Nothing holds
an HTTP connection open for minutes.

**d) What changes at production scale** (deliberately left out of this assignment):
- in-memory job dict → **queue (Celery/RQ/SQS) + Redis/Postgres** for job state, plus
  **object storage (S3/GCS)** for audio, so API pods and GPU workers scale independently
- **per-chunk retries and checkpointing**: a failure in chunk 9 of 24 retries just
  chunk 9, and finished chunks are persisted
- **parallel chunk fan-out** across GPU workers (chunks are independent, so this is easy),
  or `faster-whisper`'s `BatchedInferencePipeline` on a single GPU
- **webhook callback** on completion as an alternative to polling
- streaming partial results (SSE/WebSocket) as each chunk finishes

---

## Engineering decisions and trade-offs

| Decision | Why | Trade-off |
|---|---|---|
| faster-whisper over openai-whisper | ~4× faster, int8 on CPU, built-in VAD | Extra dependency (CTranslate2) |
| Local model over hosted API | No per-minute cost, data stays on-prem, no upload limits | You run the compute; the backend interface keeps an API easy to swap in |
| Normalise everything to 16k mono WAV | One code path; deterministic; matches the model | One extra decode pass (cheap compared to inference) |
| Silence-based cuts, no overlap | No duplicate-word merging; cuts at natural pauses | Continuous speech with no pause gets cut at the hard limit (rare; window is configurable) |
| Energy-based cut finder (not a neural VAD) | Zero dependencies, fast, good enough to find pauses | The model's own Silero VAD still does fine-grained speech detection |
| Sync + async APIs | Simple UX for short clips; robust for long ones | Two code paths to maintain |
| Confidence flags rather than silent output | Downstream can route doubtful spans to review | Thresholds are heuristics and should be tuned on real data |

## Part 2: system design decisions (summary)

Full reasoning is in [`SYSTEM_DESIGN.md`](SYSTEM_DESIGN.md). The main principle is to
**separate accepting work from doing it**: a lightweight API takes requests, a queue
holds jobs, and workers scale independently.

| Question | Decision | Why | Trade-off |
|---|---|---|---|
| Concurrent uploads | Clients upload straight to S3 via presigned (multipart, resumable) URLs; the API only creates a job and returns `202` | API servers never carry audio bytes, so they stay cheap and responsive under load | Two-step client flow (upload, then create job) |
| | Queue + separate CPU (prep) and GPU (inference) worker pools, autoscaled on backlog in *audio-hours* | Spikes are absorbed by the queue; GPU spend follows real demand; chunk-level tasks spread long files across GPUs | More moving parts than a single server |
| | Per-tenant quotas, a priority lane for short jobs, `Idempotency-Key`, content-hash dedupe | One heavy user can't starve others; client retries don't create duplicate (billed) jobs | Quota tuning per customer tier |
| Storage | **S3** for original audio (immutable), normalised audio (short TTL cache) and versioned transcript JSON | Cheap, durable, lifecycle tiering; originals allow re-transcription with better models | Reads need presigned URLs |
| | **Postgres** for job and chunk state; **search index** for segments only if search is needed | Each store does what it's good at; thousands of segments per file don't bloat the DB | Data lives in two/three places |
| | SRT/VTT rendered from canonical JSON on request; encryption, retention policies, full delete endpoint | One source of truth; privacy/GDPR compliance | Small render cost per request |
| Retry / recovery | Permanent errors (corrupt file, no audio) fail fast; transient ones (crash, OOM, timeout) retry with exponential backoff + jitter; repeat failures go to a dead-letter queue | Retrying a corrupt file wastes GPU time; transient failures usually succeed on retry | Classification must be maintained as new errors appear |
| | Retry **per chunk**, with results checkpointed; worker leases + heartbeats; idempotent writes | A crash costs one chunk, not a 2-hour job; duplicate processing is harmless | Chunk bookkeeping table |
| | Outbox pattern (job row and queue message saved in one transaction); stuck-job sweeper; `partial` status | No lost or orphaned jobs; partial results beat nothing | Extra relay process |
| API | Versioned REST (`/v1`), async by default; sync shortcut only for short clips | Transcription takes minutes; long-held HTTP connections break behind load balancers | Clients must poll or receive webhooks |
| | HMAC-signed webhooks with retries; polling as fallback | Clients learn about completion without polling loops, and can verify the sender | Clients need a public endpoint |
| | Upload separate from transcription; tenant-scoped auth; stable error codes (RFC 7807); rate-limit headers | Re-run audio with new settings without re-uploading; predictable client error handling | Slightly larger API surface |
| | Live/real-time transcription as a separate WebSocket streaming service | Streaming needs different models and latency targets than batch | A second service to run |

## Possible extensions

- **Speaker diarisation** (`pyannote.audio` or WhisperX) to label "who spoke when"
- **Evaluation harness:** WER on a labelled sample set, to choose model size and confidence
  thresholds from data
- **Downstream NLP:** punctuation/casing restoration, PII redaction, summarisation,
  keyword extraction
- **Observability:** real-time factor, queue depth, failure rate and low-confidence ratio as metrics

## Testing note

All pipeline logic (format handling, chunking, timestamp offsets, post-processing, output
formats, CLI) is tested end-to-end with real ffmpeg-generated audio in WAV, MP3, FLAC, OGG,
Opus, M4A and MP4 video, using `MockBackend`. With `pip install -r requirements.txt`, the
API tests (`tests/test_api.py`) run too, and switching `--backend faster-whisper` runs the real model.
