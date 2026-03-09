# Phase 2

This phase extends Phase 1 with local DVC storage, smart preprocessing as tracked data lineage, and MLflow model lifecycle aliases.

## What changed

## 1) DVC local storage for self-contained demo
- Configured default DVC remote to local filesystem path:
  - `.dvc/config` sets `core.remote = localstore`
  - `remote "localstore".url = .dvc/local_remote_store`
- Created `.dvc/local_remote_store/` for local cache-like remote behavior.

## 2) Smart preprocessing promoted to DVC stage
- Added preprocessing stage script: `src/data/smart_preprocess_ieee.py`
- New DVC stage: `smart_preprocess`
  - Inputs: merged train/test parquet files
  - Outputs:
    - `data/ieee_fraud_detection/processed_train.parquet`
    - `data/ieee_fraud_detection/processed_test.parquet`
    - `data/ieee_fraud_detection/preprocessing/smart_preprocessor.joblib`
    - `data/ieee_fraud_detection/preprocessing/feature_groups.json`
  - Metrics:
    - `data/ieee_fraud_detection/preprocessing/preprocessing_summary.json`

This ensures preprocessing transformations are versioned and reproducible as part of data lineage.

## 3) Training consumes processed data
- Updated training script: `src/train/train_ieee_models.py`
  - Reads `processed_train.parquet` instead of raw merged train parquet.
  - Keeps temporal/random split and CV logic.
  - Trains available models (XGBoost/CatBoost/LightGBM if available, fallback to RandomForest).

## 4) MLflow model registry lifecycle
- Added model registration per run to MLflow registered model:
  - Name from `params.yaml`: `mlflow.registered_model_name`
- Added alias updates:
  - Every trained model version can be assigned to `testing`.
  - Best model version is set to `testing` at end of run.
  - Optional auto-promotion to `production` controlled by:
    - `mlflow.promote_best_to_production`

## 5) Config and docs updates
- Extended `params.yaml` with processed-data and registry settings.
- Updated `dvc.yaml` to include `smart_preprocess` stage and dependency wiring.
- Updated main `README.md` with Phase 2 usage and behavior.

## Expected workflow
```bash
uv run dvc repro
uv run dvc push
```

This runs data extraction -> smart preprocessing -> model training, then stores DVC objects in local DVC storage.
