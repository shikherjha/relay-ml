# Relay ML Logs

This file records implementation decisions and progress so future work can resume with context.

## 2026-06-14 - Phase 1 FastAPI Skeleton

- Started Phase 1 for Bhavya's `relay-ml` service.
- Existing repo already had the intended folder layout, but most files were empty.
- Added a minimal FastAPI application with `GET /health`.
- Added Pydantic schema mirrors for `ConditionPassport`, defects, health, and fit flags.
- Added placeholder routers for grading and fit endpoints so the API shape is visible early.
- Added Docker, environment example, README, and initial tests.
- Current model file `models/grade_cnn_v1.pt` exists but is empty, so health reports the model as a placeholder instead of claiming it is loaded.

### Verification

- Initial test run failed because the system `python` points to Python 3.14 and had no FastAPI/pytest installed.
- A Python 3.14 venv also failed because `pydantic-core` attempted a Rust source build.
- Recreated `.venv` using installed Python 3.12 via `py -3.12 -m venv .venv`.
- Installed dependencies from `requirements.txt` successfully.
- Ran `.venv\Scripts\python.exe -m pytest`: `1 passed`.
- Ran app import check: `Relay ML Service`.
- Added `.gitignore` so `.venv`, caches, raw datasets, and large training artifacts are not committed by accident.
