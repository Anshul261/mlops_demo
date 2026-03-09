import argparse
import gc
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, RobustScaler, StandardScaler


def identify_feature_groups(
    df: pd.DataFrame, target_col: str, drop_cols: list[str]
) -> dict:
    feature_cols = [c for c in df.columns if c not in set(drop_cols + [target_col])]
    x = df[feature_cols]

    groups = {
        "continuous_scale": [],
        "continuous_robust": [],
        "binary": [],
        "categorical": [],
    }

    for col in x.columns:
        s = pd.Series(x[col])
        if pd.api.types.is_numeric_dtype(s):
            unique_count = s.nunique(dropna=True)
            if unique_count == 2:
                vals = set(
                    pd.Series(s.dropna().tolist()).astype(float).unique().tolist()
                )
                if vals.issubset({0.0, 1.0}):
                    groups["binary"].append(col)
                    continue

            if any(
                k in col.lower() for k in ["amt", "amount", "dist", "value", "price"]
            ):
                groups["continuous_robust"].append(col)
            else:
                groups["continuous_scale"].append(col)
        else:
            groups["categorical"].append(col)

    return groups


def build_preprocessor(groups: dict) -> ColumnTransformer:
    transformers = []

    if groups["continuous_scale"]:
        transformers.append(
            (
                "continuous_scale",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                groups["continuous_scale"],
            )
        )

    if groups["continuous_robust"]:
        transformers.append(
            (
                "continuous_robust",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", RobustScaler()),
                    ]
                ),
                groups["continuous_robust"],
            )
        )

    if groups["binary"]:
        transformers.append(
            (
                "binary",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]),
                groups["binary"],
            )
        )

    if groups["categorical"]:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                                dtype=np.float32,
                            ),
                        ),
                    ]
                ),
                groups["categorical"],
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_output_names(preprocessor: ColumnTransformer) -> list[str]:
    out_names = preprocessor.get_feature_names_out().tolist()
    return [name.split("__", 1)[1] if "__" in name else name for name in out_names]


def write_transformed_parquet_in_chunks(
    input_path: Path,
    output_path: Path,
    preprocessor: ColumnTransformer,
    feature_cols: list[str],
    output_names: list[str],
    target_col: str | None,
    chunk_size: int,
) -> tuple[int, int]:
    parquet_file = pq.ParquetFile(str(input_path))
    read_cols = feature_cols + ([target_col] if target_col else [])
    writer = None
    total_rows = 0
    total_cols = 0

    try:
        for batch in parquet_file.iter_batches(
            batch_size=chunk_size, columns=read_cols
        ):
            chunk_df = batch.to_pandas()
            x_chunk = chunk_df.reindex(columns=feature_cols)

            x_chunk_t = preprocessor.transform(x_chunk)
            if hasattr(x_chunk_t, "toarray"):
                x_chunk_t = x_chunk_t.toarray()
            x_chunk_t = np.asarray(x_chunk_t, dtype=np.float32)

            out_df = pd.DataFrame(x_chunk_t, columns=output_names)
            if target_col:
                out_df[target_col] = chunk_df[target_col].astype(np.int8).values

            table = pa.Table.from_pandas(out_df, preserve_index=False)
            if writer is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(str(output_path), table.schema)
                total_cols = len(out_df.columns)

            writer.write_table(table)
            total_rows += len(out_df)

            del chunk_df, x_chunk, x_chunk_t, out_df, table
            gc.collect()
    finally:
        if writer is not None:
            writer.close()

    return total_rows, total_cols


def sample_for_fitting(train_path: Path, max_rows: int) -> pd.DataFrame:
    parquet_file = pq.ParquetFile(str(train_path))
    batches = []
    rows = 0
    for batch in parquet_file.iter_batches(batch_size=10000):
        bdf = batch.to_pandas()
        batches.append(bdf)
        rows += len(bdf)
        if rows >= max_rows:
            break
    df = pd.concat(batches, ignore_index=True)
    if len(df) > max_rows:
        df = df.iloc[:max_rows].copy()
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smart preprocessing for IEEE fraud dataset"
    )
    parser.add_argument(
        "--params", type=Path, default=Path("params.yaml"), help="Path to params yaml"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=10000, help="Rows per transform chunk"
    )
    parser.add_argument(
        "--fit-rows", type=int, default=200000, help="Rows used to fit preprocessors"
    )
    args = parser.parse_args()

    with args.params.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    target_col = data_cfg["target_column"]
    drop_cols = data_cfg.get("drop_columns", [])
    train_path = Path(data_cfg["train_path"])
    test_path = Path(data_cfg["test_path"])

    fit_df = sample_for_fitting(train_path, args.fit_rows)
    groups = identify_feature_groups(fit_df, target_col, drop_cols)

    feature_cols = [c for c in fit_df.columns if c not in set(drop_cols + [target_col])]
    x_fit = fit_df[feature_cols]

    preprocessor = build_preprocessor(groups)
    preprocessor.fit(x_fit)
    output_names = get_output_names(preprocessor)

    del x_fit, fit_df
    gc.collect()

    preproc_dir = Path(data_cfg["preprocessing_artifact_dir"])
    preproc_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, preproc_dir / "smart_preprocessor.joblib")

    train_out = Path(data_cfg["processed_train_path"])
    test_out = Path(data_cfg["processed_test_path"])

    train_rows, train_cols = write_transformed_parquet_in_chunks(
        input_path=train_path,
        output_path=train_out,
        preprocessor=preprocessor,
        feature_cols=feature_cols,
        output_names=output_names,
        target_col=target_col,
        chunk_size=args.chunk_size,
    )
    test_rows, test_cols = write_transformed_parquet_in_chunks(
        input_path=test_path,
        output_path=test_out,
        preprocessor=preprocessor,
        feature_cols=feature_cols,
        output_names=output_names,
        target_col=None,
        chunk_size=args.chunk_size,
    )

    feature_meta = {
        "continuous_scale": groups["continuous_scale"],
        "continuous_robust": groups["continuous_robust"],
        "binary": groups["binary"],
        "categorical": groups["categorical"],
        "feature_count_after_processing": int(train_cols - 1),
        "fit_rows": args.fit_rows,
    }
    with (preproc_dir / "feature_groups.json").open("w", encoding="utf-8") as f:
        json.dump(feature_meta, f, indent=2)

    summary = {
        "processed_train_shape": [train_rows, train_cols],
        "processed_test_shape": [test_rows, test_cols],
        "chunk_size": args.chunk_size,
        "fit_rows": args.fit_rows,
    }
    with (preproc_dir / "preprocessing_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
