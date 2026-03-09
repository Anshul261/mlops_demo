# Phase 1

This document captures what was completed before Phase 2 updates.

## Objective completed in Phase 1
- Set up a reproducible baseline MLOps workflow for IEEE-CIS Fraud Detection.
- Use the local Kaggle zip dataset and ignore `sample_submission.csv`.
- Add DVC orchestration and MLflow experiment tracking.

## Implemented components
- Data preparation script: `src/data/prepare_ieee_data.py`
  - Extracts `train_transaction.csv`, `train_identity.csv`, `test_transaction.csv`, `test_identity.csv`.
  - Creates merged datasets:
    - `data/ieee_fraud_detection/train_merged.parquet`
    - `data/ieee_fraud_detection/test_merged.parquet`
  - Writes data summary file.

- Training and tracking script: `src/train/train_ieee_models.py`
  - Trained baseline gradient-boosted models.
  - Logged CV/test metrics to MLflow.
  - Wrote model artifacts and leaderboard outputs.

- DVC pipeline
  - `prepare_data` stage for extraction and merging.
  - `train_models` stage for model training and report generation.

## Configuration and dependency updates
- Central config in `params.yaml`.
- Python dependencies updated in `pyproject.toml` / `uv.lock`.

## Run status at end of Phase 1
- `uv run dvc repro` successfully produced data and model artifacts.
- MLflow tracking used file-based backend (`file:./mlruns`).
- Environment note: LightGBM was skipped due to missing `libgomp.so.1`, while XGBoost and CatBoost ran.
