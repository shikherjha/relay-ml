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

## 2026-06-14 - PR Setup Note

- User asked Codex to open the Phase 1 PR.
- GitHub repo only had remote branch `feat/ml-health`; no remote `main` branch existed yet.
- Plan: create a clean `main` baseline, create a PR-ready branch from it, replay Phase 1 changes there, then open a PR into `main`.
- Created draft PR: https://github.com/shikherjha/relay-ml/pull/1

## 2026-06-14 - Phase 2 Dataset Setup

- Started Phase 2 on branch `feat/ml-dataset`, based on the Phase 1 PR branch so the FastAPI skeleton remains available.
- Avoiding the Phase 1 mistake of using default `python`; all commands should use the Python 3.12 venv at `.venv`.
- Phase 2 goal: make dataset acquisition reproducible without committing raw datasets to git.
- Plan: add dataset-specific requirements, a download script, a dataset inventory template, and precise README steps for Hugging Face + Kaggle.
- Added `requirements-data.txt`, `scripts/download_datasets.py`, `scripts/inspect_fit_dataset.py`, and `data/dataset_manifest.example.json`.
- Installed Phase 2 optional dependencies successfully in the Python 3.12 venv.
- Downloaded Kaggle clothing fit dataset successfully through `kagglehub`.
- Wrote local ignored manifest at `data/dataset_manifest.json`.
- Inspected fit datasets:
  - `modcloth_final_data.json`: 82,790 rows; fit 56,757; large 13,059; small 12,974.
  - `renttherunway_final_data.json`: 192,544 rows; fit 142,058; small 25,779; large 24,707.
- Checked Hugging Face dataset file list successfully: 6,034 files.
- Hugging Face full image download partially completed, then failed due to `cas-bridge.xethub.hf.co` read timeout / DNS resolution failure.
- Local partial HF folder currently has 918 files; rerun `scripts/download_datasets.py --dataset hf-defects --max-workers 1` or `--max-workers 4` to resume.
- Updated downloader to support `--max-workers` and to merge dataset manifest records instead of overwriting existing successful entries.
