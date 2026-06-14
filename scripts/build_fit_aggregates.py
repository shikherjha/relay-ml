import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = Path("data/raw/kaggle_clothing_fit")
DEFAULT_OUTPUT_PATH = Path("data/processed/fit_aggregates.json")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _source_name(path: Path) -> str:
    name = path.name.lower()
    if "modcloth" in name:
        return "modcloth"
    if "renttherunway" in name:
        return "renttherunway"
    return path.stem


def _category(row: dict[str, Any]) -> str:
    for key in ("category", "rented for", "body type"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return "unknown"


def build_aggregates(paths: list[Path]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"fit_count": 0, "small_count": 0, "large_count": 0, "total": 0}
    )

    for path in paths:
        source = _source_name(path)
        for row in _read_jsonl(path):
            fit = str(row.get("fit", "")).strip().lower()
            if fit not in {"fit", "small", "large"}:
                continue

            category = _category(row)
            bucket = buckets[(source, category)]
            bucket["total"] += 1
            bucket[f"{fit}_count"] += 1

    records: list[dict[str, Any]] = []
    for (source, category), counts in sorted(buckets.items()):
        total = counts["total"]
        records.append(
            {
                "source": source,
                "category": category,
                "fit_count": counts["fit_count"],
                "small_count": counts["small_count"],
                "large_count": counts["large_count"],
                "total": total,
                "runs_small_rate": counts["small_count"] / total if total else 0.0,
                "runs_large_rate": counts["large_count"] / total if total else 0.0,
            }
        )

    return records


def _default_paths(input_dir: Path) -> list[Path]:
    return sorted(input_dir.rglob("*final_data.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build category fit aggregates from Kaggle JSONL files.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args()

    input_paths = args.files or _default_paths(args.input_dir)
    if not input_paths:
        raise SystemExit(f"No Kaggle fit JSONL files found under {args.input_dir}")

    records = build_aggregates(input_paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} fit aggregate rows to {args.output}")


if __name__ == "__main__":
    main()
