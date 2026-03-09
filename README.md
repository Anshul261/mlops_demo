MLOps pipeline for IEEE-CIS Fraud Detection using DVC + MLflow.

## Current project status
- Phase 1 documentation: `PHASE_1_README.md`
- Phase 2 documentation: `PHASE_2_README.md`
- DVC pipeline includes `prepare_data` -> `smart_preprocess` -> `train_models`
- DVC remote is local-only (`.dvc/local_remote_store`) for a self-contained demo
- Smart preprocessing is tracked as a DVC data stage
- MLflow registry integration is implemented in training code (registration + aliases)
- LightGBM is expected to be skipped in this environment due to `libgomp.so.1` missing

## Execution policy
- Long model-training commands are user-run.
- The assistant will avoid running expensive training commands unless explicitly requested.

## What this project does
- Extracts Kaggle IEEE fraud data from a local zip file.
- Builds merged train/test datasets (`transaction` + `identity`).
- Applies a smart preprocessing pipeline and versions processed datasets with DVC.
- Trains baseline fraud models with CV and holdout metrics on processed data.
- Tracks data/model artifacts with DVC and experiment/model lifecycle with MLflow.

## Dataset source
- Kaggle competition: `ieee-fraud-detection`
- Local zip expected at: `/home/ai_alpha_2026/Downloads/ieee-fraud-detection.zip`
- Only `train_*` and `test_*` files are used. `sample_submission.csv` is ignored.

## Run pipeline
```bash
uv run dvc repro
```

This runs:
- `prepare_data`: extracts and creates
  - `data/ieee_fraud_detection/raw_train_transaction.csv`
  - `data/ieee_fraud_detection/raw_train_identity.csv`
  - `data/ieee_fraud_detection/raw_test_transaction.csv`
  - `data/ieee_fraud_detection/raw_test_identity.csv`
  - `data/ieee_fraud_detection/train_merged.parquet`
  - `data/ieee_fraud_detection/test_merged.parquet`
- `train_models`: trains models and writes
  - `artifacts/models/`
  - `artifacts/plots/roc_curves.png`
  - `artifacts/reports/metrics.json`
  - `artifacts/reports/leaderboard.csv`

Plus:
- `smart_preprocess`: creates
  - `data/ieee_fraud_detection/processed_train.parquet`
  - `data/ieee_fraud_detection/processed_test.parquet`
  - `data/ieee_fraud_detection/preprocessing/smart_preprocessor.joblib`
  - `data/ieee_fraud_detection/preprocessing/feature_groups.json`

## Recommended stage-by-stage run (for stability)
```bash
uv run dvc repro prepare_data
uv run dvc repro smart_preprocess
uv run dvc repro train_models
```

If resources are tight, run one stage at a time and verify outputs before moving on.

## MLflow
- Tracking URI: `file:./mlruns` (configured in `params.yaml`)
- Start UI:
```bash
uv run mlflow ui --backend-store-uri file:./mlruns
```
- Registered model name: `ieee-fraud-detector`
- Alias policy:
  - best run version -> `testing`
  - optional promotion to `production` via `mlflow.promote_best_to_production`

Check registry and aliases:
```bash
uv run python -c "import mlflow; mlflow.set_tracking_uri('file:./mlruns'); c=mlflow.MlflowClient(); print([m.name for m in c.search_registered_models()])"
```

## DVC local storage
- This demo uses local DVC storage only.
- Default remote in `.dvc/config` points to `.dvc/local_remote_store`.
- Use these commands:
```bash
uv run dvc push
uv run dvc pull
```

## Config
- Update runtime settings in `params.yaml`:
  - data paths and target/time columns
  - processed-data and preprocessing artifact paths
  - preprocessing chunking: `data.preprocessing_chunk_size`
  - train split/CV settings
  - `max_train_rows` (`null` = full dataset)
  - MLflow experiment name/URI/model registry options

## Notes
- In this environment, `lightgbm` is skipped due to missing system library (`libgomp.so.1`).
- `xgboost` and `catboost` run successfully; if `xgboost` is unavailable, a `random_forest` fallback is used.
- Smart preprocessing script supports chunked writes to reduce memory spikes during full-data processing.
