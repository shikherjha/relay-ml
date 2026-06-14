# Relay ML Handoff

Current status: local demo-ready ML service with CNN-backed image grading,
keyframe-backed video grading, and aggregate-backed fit flags.

## Required Local Artifacts

The model checkpoint is intentionally local-only:

```text
models/grade_cnn_v1.pt
```

Expected checkpoint:

```text
size_bytes: 6248127
architecture: mobilenetv3_small_100
input_size: 224
model_tier_used: cnn-v1
```

Small metadata files should be committed:

```text
models/grade_cnn_v1.metadata.json
models/label_map.json
models/README.md
```

Fit aggregates are also local/generated data:

```text
data/processed/fit_aggregates.json
```

## Setup

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Optional dataset tooling:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-data.txt
```

## Verify

```powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -c "from app.main import health; print(health().model_loaded); print(health().model_bytes)"
```

Expected current verification:

```text
tests: 20 passed
model_loaded: True
model_bytes: 6248127
```

## Run

```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8001
```

Open:

```text
http://localhost:8001/health
http://localhost:8001/docs
```

Set downstream services to:

```text
ML_SERVICE_URL=http://localhost:8001
```

## Endpoint Notes

### Health

```powershell
Invoke-RestMethod http://localhost:8001/health
```

### Fit Flags

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8001/fit-flags `
  -ContentType application/json `
  -Body '{"sku_id":"SKU-123","category":"coat"}'
```

Expected source when aggregates exist:

```text
fit_aggregates_v1:renttherunway
```

### Grade Image

Use `/docs` for multipart upload, or post:

```text
image: JPEG/PNG file
unit_id: UUID string
category: lower-ish product category, e.g. electronics, jeans, headphones
return_id: optional UUID string
```

Returns a contract-shaped `ConditionPassport`.

### Grade Video

Use `/docs` for multipart upload, or post:

```text
video: MP4/MOV/WEBM/AVI file
unit_id: UUID string
category: product category
return_id: optional UUID string
```

The pipeline extracts 5 representative keyframes, runs image grading on each
frame, and returns a worst-frame aggregate passport.

## Current Caveats

- `models/grade_cnn_v1.pt` is ignored by git. Do not commit it unless Git LFS
  or release artifact storage is configured.
- The v1 defect classifier only predicts `crack`, `damaged`, and `other`.
  Category-aware mapping expands those into contract defect types, but it is
  still a demo-grade approximation.
- The local Hugging Face image dataset download is partial. The v1 checkpoint
  is usable for demos, not final model quality.
- Bedrock fallback is intentionally not implemented yet.
- Video grading is CPU-heavy because it runs five image inferences per upload.
