import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


REQUIRED_FILES = {
    "train_transaction.csv",
    "train_identity.csv",
    "test_transaction.csv",
    "test_identity.csv",
}

OUTPUT_NAME_MAP = {
    "train_transaction.csv": "raw_train_transaction.csv",
    "train_identity.csv": "raw_train_identity.csv",
    "test_transaction.csv": "raw_test_transaction.csv",
    "test_identity.csv": "raw_test_identity.csv",
}


def extract_required_files(zip_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with ZipFile(zip_path, "r") as zf:
        members = {Path(name).name: name for name in zf.namelist()}
        missing = [name for name in REQUIRED_FILES if name not in members]
        if missing:
            raise FileNotFoundError(f"Missing required files in zip: {missing}")

        for required in REQUIRED_FILES:
            source_name = members[required]
            target_path = output_dir / OUTPUT_NAME_MAP[required]
            if target_path.exists():
                continue

            with zf.open(source_name) as src, target_path.open("wb") as dst:
                dst.write(src.read())


def build_processed_datasets(output_dir: Path) -> dict:
    train_tx = pd.read_csv(output_dir / "raw_train_transaction.csv")
    train_id = pd.read_csv(output_dir / "raw_train_identity.csv")
    test_tx = pd.read_csv(output_dir / "raw_test_transaction.csv")
    test_id = pd.read_csv(output_dir / "raw_test_identity.csv")

    train_merged = train_tx.merge(train_id, on="TransactionID", how="left")
    test_merged = test_tx.merge(test_id, on="TransactionID", how="left")

    train_merged.to_parquet(output_dir / "train_merged.parquet", index=False)
    test_merged.to_parquet(output_dir / "test_merged.parquet", index=False)

    summary = {
        "train_shape": list(train_merged.shape),
        "test_shape": list(test_merged.shape),
        "target_column": "isFraud",
        "id_column": "TransactionID",
        "time_column": "TransactionDT",
        "fraud_rate": float(train_merged["isFraud"].mean()),
    }

    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and prepare IEEE Fraud Detection data"
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        required=True,
        help="Path to ieee-fraud-detection.zip downloaded from Kaggle",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/ieee_fraud_detection"),
        help="Directory where extracted and processed files are written",
    )
    args = parser.parse_args()

    if not args.zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {args.zip_path}")

    extract_required_files(args.zip_path, args.output_dir)
    summary = build_processed_datasets(args.output_dir)
    print("Prepared IEEE dataset:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
