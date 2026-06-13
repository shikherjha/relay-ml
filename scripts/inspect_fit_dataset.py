from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the Kaggle clothing fit dataset after download."
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Directory returned by KaggleHub or the folder containing the JSON files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_files = sorted(args.dataset_dir.glob("*.json"))
    if not json_files:
        raise SystemExit(f"No JSON files found in {args.dataset_dir}")

    for json_file in json_files:
        df = pd.read_json(json_file, lines=True)
        print(f"\n{json_file.name}")
        print(f"rows: {len(df)}")
        print(f"columns: {', '.join(df.columns)}")
        if "fit" in df.columns:
            print("fit counts:")
            print(df["fit"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
