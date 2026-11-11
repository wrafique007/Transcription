# Part 2 — System Design

How the Part 1 pipeline would grow into a multi-tenant production service. The core
idea is to **separate taking in work from doing the work**. The API accepts requests and
returns quickly. Audio goes straight to object storage. A queue holds the jobs, and workers
scale on their own to process them. Every step is written to a database, so any failure can
be picked up again from the last saved step.

```
                     ┌──────────────────────────── control plane ───────────────────────────┐
 Client ──HTTPS──► API Gateway / LB ──► API service (stateless, N pods)                      │
   │                (auth, rate limit)        │  1. create upload  → presigned URL           │
   │                                          │  2. create job     → Postgres (+ outbox)     │
   │                                          │  3. status/results ← Postgres / S3           │
   │                                          ▼                                              │
   │  PUT audio (presigned, multipart)   ┌──────────┐   outbox relay   ┌──────────────────┐   │
   └────────────────────────────────────►│  Object  │                  │  Queue           │   │
                                         │  storage │                  │  (SQS / Rabbit)  │   │
                                         │  (S3)    │                  │  prep · chunks · │   │
                                         └────▲─────┘                  │  DLQ             │   │
                                              │                        └────────┬─────────┘   │
                     ┌────────────────────────┼─────────────────────────────────┘             │
                     ▼                        │                                               │
            Prep workers (CPU)                │            GPU workers (Whisper)              │
            probe · normalise · plan chunks ──┴──► chunk tasks ──► transcribe ──► chunk result│
                                                                                │             │
                                     Assembler: all chunks done → post-process → transcript   │
                                                    │                                         │
                                       Postgres (jobs, chunks) · search index · webhooks ─────┘
```

---

## 1. How would you handle concurrent uploads?

**a) Keep audio bytes off the API servers.** The client asks for an upload, receives a
**presigned S3 URL**, and uploads directly to object storage. For files over ~100 MB this is
an **S3 multipart upload**, so the upload can resume after a dropped connection (useful on
mobile networks). API pods only deal with small JSON, so 1,000 uploads at the same time don't
tie up API threads, memory or bandwidth. This also replaces Part 1's
"stream to local disk" step, which only works with a single node.

**b) Accepting a request ≠ processing it.** `POST /transcriptions` writes a job row and
returns `202` within milliseconds. The queue absorbs spikes. Ingestion scales with API pods,
which are cheap. Processing scales with GPU workers, which are expensive. Each side scales
separately.

**c) Two kinds of worker pools, autoscaled on queue depth** (e.g. KEDA on Kubernetes):
- **CPU prep workers:** ffprobe validation, ffmpeg normalisation, chunk planning.
  This is cheap work and stays off the GPU.
- **GPU workers:** one model instance per GPU, loaded once and kept warm. Each worker
  pulls *chunk* tasks, not whole files, so a 3-hour file is spread across many GPUs
  instead of tying up one GPU for 15 minutes. That cuts the time to finish a long file.
  On a single GPU, faster-whisper's batched pipeline can process several chunks together.
- Autoscaling uses *backlog seconds of audio ÷ throughput*, not raw message count. A
  queue of 10 two-hour files is a very different load from 10 voice notes.

**d) Fairness and backpressure:**
- **Per-tenant rate limits and quotas** at the gateway, e.g. concurrent jobs and
  audio-hours per day. Over the limit → `429` with `Retry-After`.
- **Priority lanes:** a separate queue for short or interactive jobs, so one customer
  uploading 500 hours of backlog can't starve everyone's 30-second voice notes.
  Workers take from the queues by weight.
- A hard global queue-depth limit: beyond it, new batch jobs are rejected quickly rather
  than accepted and left for hours with no feedback.

**e) Duplicates and safe retries:**
- An **`Idempotency-Key` header** on job creation: if a client retries after a timeout,
  it gets the same job back instead of two transcriptions billed twice.
- A **SHA-256 hash of the audio content** (computed during prep): if the same file is
  transcribed again with the same model and settings, the stored result is reused at no
  GPU cost.

**Rough capacity sizing (illustrative):** GPU-hours needed ≈ audio-hours × RTF, where RTF
(real-time factor) is processing time ÷ audio duration. If benchmarks show an RTF of ~0.05
for `large-v3-turbo` on the chosen GPU, 1,000 audio-hours/day ≈ 50 GPU-hours/day, i.e.
~2–3 GPUs on average, with more provisioned for peaks. The real number should come from a
benchmark on our own hardware. Part 1 already reports `real_time_factor` in its metadata
for this.

---

## 2. How would you store audio and transcripts?

Use the right store for each shape of data: **large immutable files go in object storage;
state and queryable metadata go in a relational DB; full-text search goes in a search index.**

### Object storage (S3 / GCS): the bytes

```
s3://stt-audio/{tenant}/{job_id}/original.mp3          # exactly what the user sent, immutable
s3://stt-audio/{tenant}/{job_id}/normalized.wav        # 16 kHz mono, reproducible → short TTL
s3://stt-transcripts/{tenant}/{job_id}/chunks/{i}.json # per-chunk results (checkpoints)
s3://stt-transcripts/{tenant}/{job_id}/v{n}/transcript.json   # canonical final output
```
- **The original is kept as uploaded and never changed.** It's the source of truth, and
  it lets us re-transcribe later with a better model.
- **The normalised WAV is a cache** and can be rebuilt from the original, so it gets a
  short lifecycle (e.g. deleted after 7 days).
- **The canonical transcript is the JSON** from Part 1. SRT/VTT/TXT are cheap to
  render from it on request, so they aren't stored separately (or are cached in a CDN).
- **Versioned transcripts** (`v1`, `v2`…) with the model name and version recorded, so
  re-running with a newer model never silently overwrites results a customer already used.
- **Lifecycle rules:** audio moves to infrequent-access or cold storage after 30 days and is
  deleted after the tenant's retention period. Transcripts are small and kept longer.
- **Security:** server-side encryption with KMS (per-tenant keys for enterprise customers),
  a tenant prefix in every key, a private bucket, and access only through short-lived
  presigned GET URLs.

### Postgres: state and metadata
```
jobs(id, tenant_id, status, idempotency_key, audio_key, audio_sha256, duration_s,
     source_format, language, model, model_version, options jsonb, progress,
     attempts, error_code, error_message, created_at, started_at, finished_at)
chunks(job_id, idx, start_s, end_s, status, attempts, result_key, worker_id,
       lease_expires_at, error)            -- PK (job_id, idx)
transcripts(job_id, version, transcript_key, text_preview, language,
            low_confidence_count, created_at)
outbox(id, topic, payload, created_at, sent_at)  -- see §3
```
- The **chunks table** is what makes recovery cheap (§3): each chunk's state is tracked
  on its own.
- Segments are **not** one row per segment in Postgres by default. A 2-hour meeting
  produces thousands, so they go in the S3 JSON. If the product needs "search inside
  transcripts", segments with `start/end` are indexed into **OpenSearch/Elasticsearch**
  (or Postgres `tsvector` at small scale). A search hit can then jump to the right moment
  in the audio.

### Compliance
- A **`DELETE /transcriptions/{id}`** that removes the audio, all transcript versions and
  search documents. This is needed for GDPR and other right-to-erasure requests.
- Optional **PII redaction** as a post-processing step, storing a redacted version
  alongside the original.
- Audit log of who accessed which transcript.

---

## 3. How do you retry or recover failed transcriptions?

**First step: sort failures into ones that can't be fixed by retrying and ones that can.**

| Type | Examples | Action |
|---|---|---|
| **Permanent** | corrupt file, no audio stream, unsupported codec, over the duration limit | Fail immediately with a clear `error_code`; **never retry** (it would fail the same way every time) |
| **Transient** | worker crash, spot/preemptible instance reclaimed, GPU out-of-memory (OOM), S3/network timeout, external API 5xx/429 | Retry with **exponential backoff + jitter**, up to N attempts (e.g. 3) |
| **Poison** | input that fails over and over for an unknown reason | Sent to the **dead-letter queue** after N attempts, an alert fires, and someone looks at it |

Part 1 already makes this split: `AudioError` (permanent) is separate from everything else.

**Retry at the chunk level, not the job level.** A 2-hour file is ~24 chunks. If chunk 17
fails, only chunk 17 is retried; chunks 1–16 and 18–24 are already saved in S3 and marked
`done` in the `chunks` table. Because Part 1's chunk planning gives the same chunks for the
same audio, and the plan is stored, a retry always lines up with the existing pieces.

**Plan for workers dying unexpectedly** (e.g. a reclaimed spot instance with no time to clean up):
- **At-least-once delivery with leases:** a worker claims a chunk with
  `lease_expires_at` and sends heartbeats while it works (the queue's visibility timeout
  plays the same role). If it dies, the lease expires and the chunk is picked up by
  another worker.
- **Idempotent writes:** a chunk result is written to a fixed key
  (`chunks/{i}.json`), and the row is updated with a conditional `UPDATE … WHERE
  status != 'done'`. If two workers process the same chunk, the outcome is the same and
  nothing is counted twice.
- **Stuck-job sweeper:** a periodic job finds work stuck in `running` past its lease or
  a maximum wall-clock time and puts it back in the queue (or fails it).

**Keep the DB and the queue consistent (outbox pattern).** Creating a job writes the
job row *and* an `outbox` row in one DB transaction. A relay process publishes outbox
rows to the queue. This rules out "job saved but message lost" or "message sent but DB write
rolled back", which would otherwise leave jobs that never run.

**Smarter retries for known failure types:**
- GPU OOM → retry the chunk with smaller chunks or a lower-precision `compute_type`.
- Repeated failure on the primary model → optionally fall back to a smaller model or a
  hosted API provider. The fallback is recorded in metadata so it's visible.
- Output with poor quality (many low-confidence segments) isn't treated as a failure.
  The job completes and the transcript is flagged for human review.

**Partial results and manual recovery:**
- If some chunks still fail after all retries, the job ends as **`partial`**: the
  transcript is returned with the failed time ranges clearly marked, which is often more
  useful than nothing.
- `POST /transcriptions/{id}/retry` (user or admin) re-runs only the failed chunks.
  The DLQ has a replay tool for after a bug fix is deployed.

**Visibility:** metrics on failure rate by `error_code`, retry count, DLQ depth,
queue backlog in audio-hours, end-to-end latency, real-time factor (RTF), and low-confidence
ratio. Alerts fire on DLQ growth and on a rising failure rate after a deploy. Every log line
carries a `job_id`/`chunk_id` trace ID.

---

## 4. How would you expose this as an API?

A **REST API, versioned (`/v1`), asynchronous by default**, with a sync shortcut for short
clips and webhooks so clients don't have to poll. OpenAPI documentation is generated
automatically by FastAPI.

### Resources and endpoints

```
POST   /v1/uploads                          → { upload_id, url (presigned), expires_at }
POST   /v1/transcriptions                   → 202 { id, status: "queued", links }
         headers: Idempotency-Key
         body: { upload_id | audio_url, language?, model?, word_timestamps?,
                 diarize?, callback_url?, priority? }
GET    /v1/transcriptions/{id}              → status, progress, metadata, result (when done)
GET    /v1/transcriptions/{id}/transcript?format=json|srt|vtt|txt
GET    /v1/transcriptions?status=&created_after=&cursor=   (cursor pagination)
POST   /v1/transcriptions/{id}/cancel
POST   /v1/transcriptions/{id}/retry
DELETE /v1/transcriptions/{id}              (erases audio + transcripts)

POST   /v1/transcribe                       sync shortcut: small file inline,
                                            ≤ ~5 min audio, returns the transcript directly
GET    /v1/health  ·  GET /v1/models        (available models & languages)
```

**Job lifecycle:**
`queued → preprocessing → transcribing (progress: chunks_done/chunks_total) →
completed | partial | failed | cancelled`

**Example: create and fetch**
```http
POST /v1/transcriptions
Idempotency-Key: 7f9c…
{ "upload_id": "upl_123", "language": "ur", "callback_url": "https://client.app/hooks/stt" }

202 Accepted
{ "id": "tr_8x2", "status": "queued",
  "links": { "self": "/v1/transcriptions/tr_8x2" } }
```

### Design choices
- **Async by default:** transcription takes seconds to many minutes, and holding HTTP
  connections open that long breaks behind load balancers and mobile networks. The sync
  endpoint is a convenience with a strict duration limit (Part 1's 413 response
  "use /jobs instead" is the same idea).
- **Webhooks** (`transcription.completed`, `.failed`, `.partial`): **signed with HMAC**
  so receivers can verify they came from us, retried with backoff, and carrying an event
  ID so receivers can ignore duplicates. Polling stays available as a fallback.
- **Upload separate from transcription:** the same uploaded audio can be transcribed again
  with different settings or models without uploading it again. Clients can also pass an
  `audio_url` they already host.
- **Auth:** API keys for server-to-server use, OAuth2/JWT for user-facing apps. Every query
  is scoped to the caller's tenant, so one tenant can never read another's job IDs.
- **Consistent errors** (RFC 7807 `application/problem+json`) with stable codes such as
  `unsupported_format`, `audio_too_long`, `quota_exceeded`, `no_audio_stream`, so clients
  can branch on the code instead of parsing error text. Correct status codes:
  400/401/403/404/409 (idempotency conflict)/413/415/422/429.
- **Rate-limit headers** (`X-RateLimit-Remaining`, `Retry-After`) and usage reporting
  in audio-minutes, which is also the natural unit for billing.
- **Versioning:** breaking changes go in `/v2`. The `model_version` in each response means
  results are reproducible and model upgrades are visible to clients.

### Real-time use cases
Live captioning or voice agents need a **different path**: a **WebSocket (or gRPC
bidirectional stream)** at `/v1/stream`. The client sends audio frames, and the server
returns *partial* (may still change) and *final* segments. That requires streaming-specific
inference: short windows, VAD-based endpointing and low-latency models. It's best built as a
separate service sharing the same post-processing and storage, rather than forced into the
batch pipeline.

---

## Summary of key decisions

| Concern | Decision | Why |
|---|---|---|
| Concurrent uploads | Presigned direct-to-S3 multipart + queue + autoscaled workers | API stays light; spikes are absorbed; GPU spend follows the actual backlog |
| Fairness | Per-tenant quotas, priority lanes | One heavy user can't starve the rest |
| Storage | S3 for audio/transcripts, Postgres for state, search index for segments | Each store does what it's good at; cheap lifecycle tiering |
| Recovery | Chunk-level checkpoints, leases, idempotent writes, outbox, DLQ | Failures cost one chunk, not one job; no lost or duplicated work |
| API | Async REST + webhooks, sync shortcut, idempotency keys | Matches how long the work really takes; safe for clients to retry |
