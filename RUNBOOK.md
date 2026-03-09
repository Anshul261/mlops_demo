# Runbook

## One-time setup
```bash
uv sync
```

## Execute pipeline (stage-by-stage)
```bash
uv run dvc repro prepare_data
uv run dvc repro smart_preprocess
uv run dvc repro train_models
```

## DVC local storage sync
```bash
uv run dvc push
uv run dvc pull
```

## MLflow UI and registry check
```bash
uv run mlflow ui --backend-store-uri file:./mlruns
uv run python -c "import mlflow; mlflow.set_tracking_uri('file:./mlruns'); c=mlflow.MlflowClient(); print([m.name for m in c.search_registered_models()])"
```

## Files and why they matter
- `params.yaml`: runtime config (data paths, chunking, train/CV, MLflow registry settings)
- `dvc.yaml`: pipeline definition (`prepare_data` -> `smart_preprocess` -> `train_models`)
- `dvc.lock`: frozen hashes/versions for reproducibility
- `src/data/prepare_ieee_data.py`: unzip + merge Kaggle train/test files
- `src/data/smart_preprocess_ieee.py`: smart feature preprocessing + chunked processed parquet output
- `src/train/train_ieee_models.py`: model training, metrics, MLflow logging, registry alias updates
- `.dvc/config`: local-only DVC remote (`.dvc/local_remote_store`)
- `artifacts/reports/metrics.json`: final metrics + best model summary
- `artifacts/reports/leaderboard.csv`: model comparison table
- `PHASE_1_README.md`: what was completed in Phase 1
- `PHASE_2_README.md`: Phase 2 upgrades and design choices

## Troubleshooting
- **OOM / process killed (`exited with -9`)**: run one DVC stage at a time; reduce `data.preprocessing_chunk_size` in `params.yaml`; close other memory-heavy processes.
- **LightGBM not loading (`libgomp.so.1` missing)**: expected in this environment; training continues with XGBoost/CatBoost (or RandomForest fallback).
- **No MLflow registered models**: ensure `uv run dvc repro train_models` completes successfully; then re-run the registry check command.
- **DVC artifacts not found after cleanup**: run `uv run dvc pull` to restore tracked outputs from local DVC storage.
- **Pipeline changed but stage not rerun**: force rerun with `uv run dvc repro -f <stage_name>`.
