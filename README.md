# Relay ML

Bhavya-owned perception service for Relay. Grades returned products into a
structured **ConditionPassport**, provides fit intelligence, embeddings for
next-owner matching, and buyer-intent scoring.

## Current Status

**Fully functional** — all endpoints live, 55 tests passing.

| Endpoint | Description | Status |
|---|---|---|
| `GET /health` | Liveness + model/mode status | ✅ |
| `POST /grade-image` | Grade single image → ConditionPassport | ✅ |
| `POST /grade-images` | Grade 1-8 multi-angle images → ConditionPassport | ✅ |
| `POST /grade-video` | Grade video via keyframe extraction | ✅ |
| `POST /fit-flags` | Aggregate-backed fit flags | ✅ |
| `POST /fit-flags/multi` | Bayesian MultiFlags from return counts | ✅ |
| `POST /embed` | 384-d embedding vector | ✅ |
| `POST /wish-score` | Buyer-intent confidence score (0–1) | ✅ |
| `POST /return-clusters` | NLP clustering of return reasons | ✅ |

## Grading Modes

Controlled by `GRADING_MODE` env var:

| Mode | Description |
|---|---|
| `bedrock_only` **(default)** | AWS Bedrock Nova Lite — most accurate, handles undamaged products correctly |
| `cnn` | Local MobileNetV3 CNN, escalates to Bedrock on low confidence |
| `mock` | Deterministic stub (no AI) for relay-api local dev |

## Quick Start

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` with your AWS credentials:

```bash
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
GRADING_MODE=bedrock_only
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

Run:

```bash
uvicorn app.main:app --reload --port 8001
```

Verify:

```bash
# Health check
curl http://localhost:8001/health

# OpenAPI docs
open http://localhost:8001/docs
```

## Test

```bash
.venv\Scripts\python.exe -m pytest
# Expected: 55 passed
```

Live product testing:

```bash
.venv\Scripts\python.exe scripts/run_tests_now.py
```

## Integration

Set in downstream services:

```
ML_SERVICE_URL=http://localhost:8001
```

Embedding dimension: **384** (matches relay-api pgvector schema).

## Model Artifacts

### For Bedrock mode (default) — no local model needed

Just AWS credentials with Bedrock Nova Lite enabled in us-east-1.

### For CNN mode

Local-only (git-ignored):

```text
models/grade_cnn_v1.pt     (6,248,127 bytes)
```

Tracked metadata:

```text
models/grade_cnn_v1.metadata.json
models/label_map.json
```

## Embedding Backends

| Setting | Backend | Torch needed? |
|---|---|---|
| `EMBEDDING_MODEL=all-MiniLM-L6-v2` | Local sentence-transformers | Yes |
| `EMBEDDING_MODEL=bedrock-titan` | AWS Bedrock Titan V2 (512→384 truncated) | No |

Both produce normalized 384-d vectors compatible with pgvector `VECTOR(384)`.

## Dataset Setup (optional)

```bash
pip install -r requirements-data.txt
python scripts/download_datasets.py --dataset all
python scripts/build_fit_aggregates.py
```

See `data/README.md` for details.

## Docker

```bash
docker build -t relay-ml .
docker run --env-file .env -p 8001:8001 relay-ml
```

For lightweight builds without torch (Bedrock-only mode), see `HANDOFF.md`.

## Docs

- `HANDOFF.md` — full integration details, endpoint contracts, caveats
- `models/README.md` — model artifact documentation

