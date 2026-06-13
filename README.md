# Relay ML

Bhavya-owned perception service for Relay. This service exposes image/video
grading and fit-intelligence endpoints used by `relay-api`.

## Phase 1 Status

Implemented:

- FastAPI app skeleton
- `GET /health`
- Pydantic schema mirrors for `ConditionPassport`
- Placeholder grading routes
- Simple T1-style `/fit-flags` rules endpoint
- Dockerfile and environment template
- Basic health test

The current `models/grade_cnn_v1.pt` file is a placeholder. Replace it with
trained weights in the CNN phase.

## Phase 2 Dataset Setup

Dataset acquisition is scripted but raw data is intentionally ignored by git.

```bash
.venv\Scripts\python.exe -m pip install -r requirements-data.txt
.venv\Scripts\python.exe scripts\download_datasets.py --dataset all
```

See `data/README.md` for dataset-specific notes and credential handling.

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

Temporary rules endpoint for Phase 1/early T1.

### `POST /grade-image`

Placeholder in Phase 1. This will return a full `ConditionPassport` after image
grading is implemented.

### `POST /grade-video`

Placeholder in Phase 1. This will aggregate frame-level image grades.
