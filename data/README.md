# Relay ML Data

Track every dataset used by the ML service here.

## Required First Downloads

| Dataset | Source | Use | Status |
|---|---|---|---|
| AI-generated e-commerce defect images | Hugging Face: `prajwalkothwal/ai-generated-ecommerce-images` | Grade CNN, defect labels, demo images | scripted |
| Clothing fit dataset | Kaggle: `rmisra/clothing-fit-dataset-for-size-recommendation` | Fit flags and size signals | scripted |

## Local Setup

Use Python 3.12 for this repo. The default `python` on Bhavya's machine points to
Python 3.14, which is not compatible with the current pinned FastAPI/Pydantic
stack.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -r requirements-data.txt
```

## Download Commands

Download both Phase 2 datasets:

```powershell
.venv\Scripts\python.exe scripts\download_datasets.py --dataset all
```

Download only the Hugging Face defect dataset:

```powershell
.venv\Scripts\python.exe scripts\download_datasets.py --dataset hf-defects --max-workers 4
```

Download only the Kaggle fit dataset:

```powershell
.venv\Scripts\python.exe scripts\download_datasets.py --dataset kaggle-fit
```

The script writes:

```text
data/dataset_manifest.json
```

Raw datasets are ignored by git through `.gitignore`.

If the Hugging Face image download times out, rerun the same command. The
download cache and local folder should resume already-fetched files. If the
network is unstable, lower concurrency:

```powershell
.venv\Scripts\python.exe scripts\download_datasets.py --dataset hf-defects --max-workers 1
```

## Kaggle Notes

`kagglehub` can download public datasets when the environment is configured
correctly. If it asks for authentication, create or download your Kaggle API
token from Kaggle account settings and keep it outside git.

Never commit:

```text
kaggle.json
KAGGLE_USERNAME
KAGGLE_KEY
```

## Fit Dataset Inspection

After downloading the Kaggle dataset, inspect JSON files with:

```powershell
.venv\Scripts\python.exe scripts\inspect_fit_dataset.py <dataset-folder>
```

Use the folder printed in `data/dataset_manifest.json` for the Kaggle dataset.

Current local inspection result:

| File | Rows | Fit counts |
|---|---:|---|
| `modcloth_final_data.json` | 82,790 | fit: 56,757; large: 13,059; small: 12,974 |
| `renttherunway_final_data.json` | 192,544 | fit: 142,058; small: 25,779; large: 24,707 |

These counts are enough to build the first aggregate `/fit-flags` rules in the
next phase.

## Documentation Template

For each dataset, add:

- source URL
- license or terms
- downloaded date
- local path
- row/image count
- labels used
- train/validation/test split
- any preprocessing steps

Do not commit large raw datasets unless the repo is configured for Git LFS or
the team explicitly agrees to store a small demo subset.

## Phase 2 Acceptance Criteria

- Dataset download process is reproducible from scripts.
- Raw dataset files stay out of git.
- `data/dataset_manifest.json` records downloaded locations.
- Dataset licenses/sources are documented before model training.
- Fit dataset columns and fit-label distribution are inspected before building
  `/fit-flags` from real aggregates.
