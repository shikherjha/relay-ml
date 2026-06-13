from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


HF_DEFECT_DATASET = "prajwalkothwal/ai-generated-ecommerce-images"
KAGGLE_FIT_DATASET = "rmisra/clothing-fit-dataset-for-size-recommendation"


@dataclass
class DatasetRecord:
    name: str
    source: str
    local_path: str
    downloaded_at: str
    status: str
    notes: str


def write_manifest(records: list[DatasetRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_records = load_existing_records(output_path)
    for record in records:
        merged_records[record.name] = record

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "datasets": [asdict(record) for record in merged_records.values()],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_existing_records(manifest_path: Path) -> dict[str, DatasetRecord]:
    if not manifest_path.exists():
        return {}

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {}
    for item in data.get("datasets", []):
        record = DatasetRecord(**item)
        records[record.name] = record
    return records


def download_hf_defects(output_dir: Path, max_workers: int) -> DatasetRecord:
    from huggingface_hub import snapshot_download

    target_dir = output_dir / "hf_ecommerce_defects"
    local_path = snapshot_download(
        repo_id=HF_DEFECT_DATASET,
        repo_type="dataset",
        local_dir=target_dir,
        max_workers=max_workers,
    )

    return DatasetRecord(
        name="AI-generated e-commerce defect images",
        source=f"https://huggingface.co/datasets/{HF_DEFECT_DATASET}",
        local_path=str(Path(local_path)),
        downloaded_at=datetime.now(timezone.utc).isoformat(),
        status="downloaded",
        notes="Used for grade CNN baseline, defect labels, and demo-grade images.",
    )


def download_kaggle_fit(output_dir: Path) -> DatasetRecord:
    import kagglehub

    downloaded_path = kagglehub.dataset_download(KAGGLE_FIT_DATASET)
    target_dir = output_dir / "kaggle_clothing_fit"
    target_dir.mkdir(parents=True, exist_ok=True)

    # KaggleHub caches the dataset. Keep the cached path in the manifest instead
    # of copying unknown-size files into the repo workspace.
    return DatasetRecord(
        name="Clothing fit dataset for size recommendation",
        source=f"https://www.kaggle.com/datasets/{KAGGLE_FIT_DATASET}",
        local_path=str(Path(downloaded_path)),
        downloaded_at=datetime.now(timezone.utc).isoformat(),
        status="downloaded",
        notes=(
            "Used for Phase 2 fit-flag EDA. KaggleHub stores the actual files "
            "in its cache; do not commit raw files."
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Relay ML Phase 2 datasets.")
    parser.add_argument(
        "--dataset",
        choices=["all", "hf-defects", "kaggle-fit"],
        default="all",
        help="Dataset to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory for datasets that can be stored locally.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/dataset_manifest.json"),
        help="Path for the generated dataset manifest.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum concurrent Hugging Face downloads. Lower this if downloads time out.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records: list[DatasetRecord] = []
    if args.dataset in {"all", "hf-defects"}:
        records.append(download_hf_defects(args.output_dir, args.max_workers))
    if args.dataset in {"all", "kaggle-fit"}:
        records.append(download_kaggle_fit(args.output_dir))

    write_manifest(records, args.manifest)
    print(f"Wrote dataset manifest to {args.manifest}")


if __name__ == "__main__":
    main()
