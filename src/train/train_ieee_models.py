import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, train_test_split


def try_import_models() -> dict:
    available = {}

    try:
        import xgboost as xgb  # pylint: disable=import-outside-toplevel

        available["xgboost"] = xgb.XGBClassifier
    except Exception as exc:  # noqa: BLE001
        print(f"Skipping xgboost: {exc}")

    try:
        import lightgbm as lgb  # pylint: disable=import-outside-toplevel

        available["lightgbm"] = lgb.LGBMClassifier
    except Exception as exc:  # noqa: BLE001
        print(f"Skipping lightgbm: {exc}")

    try:
        from catboost import CatBoostClassifier  # pylint: disable=import-outside-toplevel

        available["catboost"] = CatBoostClassifier
    except Exception as exc:  # noqa: BLE001
        print(f"Skipping catboost: {exc}")

    return available


def metric_dict(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = 0.5

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": roc_auc,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def promote_model_aliases(model_name: str, version: str, set_production: bool) -> None:
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(model_name, "testing", version)
    if set_production:
        client.set_registered_model_alias(model_name, "production", version)


def run_training(cfg: dict[str, Any]) -> None:
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    mlflow_cfg = cfg["mlflow"]

    output_dir = Path(train_cfg["output_dir"])
    models_dir = output_dir / "models"
    reports_dir = output_dir / "reports"
    plots_dir = output_dir / "plots"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(data_cfg["processed_train_path"])
    target_col = data_cfg["target_column"]
    if target_col not in train_df.columns:
        raise ValueError(
            f"Missing target column '{target_col}' in processed train data"
        )

    max_train_rows = train_cfg.get("max_train_rows")
    if max_train_rows and max_train_rows < len(train_df):
        train_df = train_df.tail(max_train_rows).reset_index(drop=True)

    x = train_df.drop(columns=[target_col])
    y = train_df[target_col]

    if train_cfg["time_based_split"]:
        split_idx = int(len(train_df) * (1 - train_cfg["test_size"]))
        x_train, x_test = x.iloc[:split_idx], x.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=train_cfg["test_size"],
            random_state=train_cfg["random_state"],
            stratify=y,
        )

    x_train_t = x_train.to_numpy()
    x_test_t = x_test.to_numpy()

    model_classes = try_import_models()
    models = {}
    if "xgboost" in model_classes:
        models["xgboost"] = model_classes["xgboost"](
            random_state=train_cfg["random_state"], n_jobs=-1
        )
    if "lightgbm" in model_classes:
        models["lightgbm"] = model_classes["lightgbm"](
            random_state=train_cfg["random_state"], n_jobs=-1, verbosity=-1
        )
    if "catboost" in model_classes:
        models["catboost"] = model_classes["catboost"](
            random_seed=train_cfg["random_state"], thread_count=-1, verbose=False
        )
    if "xgboost" not in models:
        models["random_forest"] = RandomForestClassifier(
            n_estimators=300,
            random_state=train_cfg["random_state"],
            n_jobs=-1,
            class_weight="balanced_subsample",
        )

    if not models:
        raise RuntimeError("No models available for training.")

    if train_cfg["time_based_split"]:
        splitter = TimeSeriesSplit(n_splits=train_cfg["cv_folds"])
        split_indices = list(splitter.split(x_train_t))
    else:
        splitter = StratifiedKFold(
            n_splits=train_cfg["cv_folds"],
            shuffle=True,
            random_state=train_cfg["random_state"],
        )
        split_indices = list(splitter.split(x_train_t, y_train))

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    registered_model_name = mlflow_cfg["registered_model_name"]
    all_results = {}
    roc_data = {}

    for model_name, model in models.items():
        cv_metrics = {
            "accuracy": [],
            "roc_auc": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
        }

        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(
                {
                    "model_name": model_name,
                    "test_size": train_cfg["test_size"],
                    "cv_folds": train_cfg["cv_folds"],
                    "random_state": train_cfg["random_state"],
                    "time_based_split": train_cfg["time_based_split"],
                    "target_column": data_cfg["target_column"],
                    "processed_train_path": data_cfg["processed_train_path"],
                }
            )

            for train_idx, val_idx in split_indices:
                x_fold_train, x_fold_val = x_train_t[train_idx], x_train_t[val_idx]
                y_fold_train, y_fold_val = (
                    y_train.iloc[train_idx],
                    y_train.iloc[val_idx],
                )

                model_fold = model.__class__(**model.get_params())
                model_fold.fit(x_fold_train, y_fold_train)
                y_pred = model_fold.predict(x_fold_val)
                y_prob = model_fold.predict_proba(x_fold_val)[:, 1]
                fold_metrics = metric_dict(y_fold_val, y_pred, y_prob)
                for m_name, m_value in fold_metrics.items():
                    cv_metrics[m_name].append(m_value)

            model.fit(x_train_t, y_train)
            y_test_pred = model.predict(x_test_t)
            y_test_prob = model.predict_proba(x_test_t)[:, 1]
            test_metrics = metric_dict(y_test, y_test_pred, y_test_prob)

            cv_means = {k: float(np.mean(v)) for k, v in cv_metrics.items()}
            cv_stds = {k: float(np.std(v)) for k, v in cv_metrics.items()}

            for k, v in cv_means.items():
                mlflow.log_metric(f"cv_{k}_mean", v)
            for k, v in cv_stds.items():
                mlflow.log_metric(f"cv_{k}_std", v)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)

            model_path = models_dir / f"{model_name}.joblib"
            joblib.dump(model, model_path)
            mlflow.log_artifact(str(model_path), artifact_path="models")
            mlflow_artifact_name = f"model_{model_name}"
            mlflow.sklearn.log_model(model, artifact_path=mlflow_artifact_name)

            model_uri = (
                f"runs:/{mlflow.active_run().info.run_id}/{mlflow_artifact_name}"
            )
            mv = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
            mlflow.log_param("registered_model_name", registered_model_name)
            mlflow.log_param("registered_model_version", mv.version)
            promote_model_aliases(registered_model_name, str(mv.version), False)

            all_results[model_name] = {
                "cv_means": cv_means,
                "cv_stds": cv_stds,
                "test_metrics": test_metrics,
                "model_path": str(model_path),
                "registered_model": registered_model_name,
                "registered_model_version": str(mv.version),
            }

            fpr, tpr, _ = roc_curve(y_test, y_test_prob)
            roc_data[model_name] = {
                "fpr": fpr,
                "tpr": tpr,
                "auc": test_metrics["roc_auc"],
            }

    best_model = max(
        all_results.keys(), key=lambda m: all_results[m]["cv_means"]["roc_auc"]
    )
    best_version = all_results[best_model]["registered_model_version"]
    promote_model_aliases(
        registered_model_name,
        best_version,
        bool(mlflow_cfg.get("promote_best_to_production", False)),
    )

    results_payload = {
        "dataset": "ieee_fraud_detection",
        "target": data_cfg["target_column"],
        "best_model": best_model,
        "registered_model": registered_model_name,
        "aliases": {
            "testing": best_version,
            "production": best_version
            if mlflow_cfg.get("promote_best_to_production", False)
            else None,
        },
        "results": all_results,
    }

    with (reports_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    leaderboard = pd.DataFrame(
        [
            {
                "model": model_name,
                "cv_auc": vals["cv_means"]["roc_auc"],
                "test_auc": vals["test_metrics"]["roc_auc"],
                "cv_f1": vals["cv_means"]["f1_score"],
                "test_f1": vals["test_metrics"]["f1_score"],
                "registered_version": vals["registered_model_version"],
            }
            for model_name, vals in all_results.items()
        ]
    ).sort_values("cv_auc", ascending=False)
    leaderboard.to_csv(reports_dir / "leaderboard.csv", index=False)

    plt.figure(figsize=(10, 6))
    for model_name, values in roc_data.items():
        plt.plot(
            values["fpr"],
            values["tpr"],
            label=f"{model_name} (AUC={values['auc']:.4f})",
        )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves on Holdout Test Set")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(plots_dir / "roc_curves.png", dpi=250)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train baseline IEEE fraud models with MLflow tracking"
    )
    parser.add_argument(
        "--params", type=Path, default=Path("params.yaml"), help="Path to params yaml"
    )
    args = parser.parse_args()

    with args.params.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_training(cfg)


if __name__ == "__main__":
    main()
