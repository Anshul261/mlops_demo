MLOps pipeline for IEEE-CIS Fraud Detection using DVC + MLflow.

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
  - train split/CV settings
  - `max_train_rows` (for faster local runs)
  - MLflow experiment name/URI/model registry options

## Notes
- In this environment, `lightgbm` is skipped due to missing system library (`libgomp.so.1`).
- `xgboost` and `catboost` run successfully; if `xgboost` is unavailable, a `random_forest` fallback is used.
