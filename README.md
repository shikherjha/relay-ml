# Relay ML

Bhavya-owned perception service for Relay. This service exposes image/video
grading and fit-intelligence endpoints used by `relay-api`.

## Current Status

Implemented:

- FastAPI app skeleton
- `GET /health`
- Pydantic schema mirrors for `ConditionPassport`
- CNN-backed `POST /grade-image`
- Keyframe-aggregated `POST /grade-video`
- Aggregate-backed `/fit-flags` with rules fallback
- Dockerfile and environment template
- Health, hash, image, video, and fit tests

The current local model artifact is `models/grade_cnn_v1.pt`. It is required
for real CNN inference but is intentionally ignored by git until Git LFS or
release artifact storage is configured.

## Phase 2 Dataset Setup

Dataset acquisition is scripted but raw data is intentionally ignored by git.

```bash
.venv\Scripts\python.exe -m pip install -r requirements-data.txt
.venv\Scripts\python.exe scripts\download_datasets.py --dataset all
```

See `data/README.md` for dataset-specific notes and credential handling.

## Model Artifacts

Tracked:

- `models/grade_cnn_v1.metadata.json`
- `models/label_map.json`

Local-only unless Git LFS or release storage is configured:

- `models/grade_cnn_v1.pt`

Expected local artifact:

```text
models/grade_cnn_v1.pt
size: 6,248,127 bytes
architecture: mobilenetv3_small_100
input_size: 224
defect_labels: crack, damaged, other
grade_labels: A+, A, B+, B, C, D
```

Health should report `model_loaded=true` when the checkpoint is present:

```bash
.venv\Scripts\python.exe -c "from app.main import health; print(health().model_loaded); print(health().model_bytes)"
```

## Run Locally

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
```

Open:

```text
http://localhost:8001/health
http://localhost:8001/docs
```

For handoff details, endpoint smoke examples, and artifact notes, see
`HANDOFF.md`.

## Test

```bash
.venv\Scripts\python.exe -m pytest
```

## Docker

```bash
docker build -t relay-ml .
docker run --env-file .env.example -p 8001:8001 relay-ml
```

## API Contract

### `GET /health`

Returns service and model-load status.

### `POST /fit-flags`

Returns aggregate-backed category fit flags when
`data/processed/fit_aggregates.json` is present, with a rules fallback.

### `POST /grade-image`

Returns a CNN-backed `ConditionPassport` for JPEG/PNG images.

### `POST /grade-video`

Extracts representative keyframes, grades each frame with the image pipeline,
and returns a worst-frame aggregated `ConditionPassport`.
