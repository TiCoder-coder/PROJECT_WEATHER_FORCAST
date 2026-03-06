"""
optuna_twostage.py — Optuna hyperparameter tuning for TwoStage model.
Uses the same pipeline as train.py but searches over model parameters.

Usage:
    python scripts/optuna_twostage.py --n_trials 80
"""
import os, sys, json, argparse
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "WeatherForcast.settings")
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import django; django.setup()

import numpy as np
import optuna
from optuna.samplers import TPESampler

# Reuse train.py pipeline
from Weather_Forcast_App.Machine_learning_model.trainning.train import (
    _load_config, run_training, _save_json,
)

BASE_CONFIG = {
    "data": {
        "folder_key": "cleaned_merge",
        "filename": "cleaned_merge_merged_vrain_data_20260216_121532.csv",
    },
    "target_column": "rain_total",
    "group_by": None,
    "skip_schema_validation": True,
    "auto_detect_data_type": True,
    "split": {"test_size": 0.15, "valid_size": 0.15, "shuffle": True, "sort_by_time": False},
    "features": {},
    "feature_selection": {"enabled": True, "max_features": 200},
    "polynomial_features": {"enabled": True, "top_k_corr": 10, "degree": 2},
    "transform_target": {"log1p": True},
    "transform": {
        "missing_strategy": "median",
        "scaler_type": "robust",
        "encoding_type": "label",
        "handle_outliers": True,
        "outlier_method": "iqr",
    },
    "model": {"type": "two_stage", "params": {}},
}


def objective(trial: optuna.Trial) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "rain_threshold": trial.suggest_float("rain_threshold", 0.05, 0.3),
        "predict_threshold": trial.suggest_float("predict_threshold", 0.2, 0.5),
        "random_state": 42,
    }

    # Regressor objective: choose between huber, regression, fair
    reg_obj = trial.suggest_categorical("reg_objective", ["huber", "regression", "fair"])
    params["reg_params"] = {"objective": reg_obj}
    if reg_obj == "huber":
        params["reg_params"]["alpha"] = trial.suggest_float("huber_alpha", 0.5, 0.99)

    # Tweedie power only used if we ever switch back to tweedie
    params["tweedie_variance_power"] = 1.5

    config = {**BASE_CONFIG, "model": {"type": "two_stage", "params": params}}

    try:
        info = run_training(config)
        # Read saved metrics
        metrics_path = ROOT / "Weather_Forcast_App" / "Machine_learning_artifacts" / "latest" / "Metrics.json"
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)

        test_r2 = metrics["test"]["R2"]
        valid_r2 = metrics["valid"]["R2"]
        train_r2 = metrics["train"]["R2"]

        # Penalise overfitting: optimise test R² but penalise large train-test gap
        overfit_gap = max(0, train_r2 - test_r2)
        score = test_r2 - 0.3 * overfit_gap

        trial.set_user_attr("test_r2", test_r2)
        trial.set_user_attr("valid_r2", valid_r2)
        trial.set_user_attr("train_r2", train_r2)
        trial.set_user_attr("test_rmse", metrics["test"]["RMSE"])
        trial.set_user_attr("rain_det", metrics["test"].get("Rain_Detection_Accuracy", 0))

        return score
    except Exception as e:
        print(f"  [OPTUNA] Trial failed: {e}")
        return -999.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=50)
    args = parser.parse_args()

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Seed with our best known config (v4)
    study.enqueue_trial({
        "n_estimators": 1000, "learning_rate": 0.03, "max_depth": 7,
        "num_leaves": 63, "min_child_samples": 30, "subsample": 0.8,
        "colsample_bytree": 0.7, "reg_alpha": 0.5, "reg_lambda": 3.0,
        "rain_threshold": 0.1, "predict_threshold": 0.35,
        "reg_objective": "huber", "huber_alpha": 0.9,
    })

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    print("\n" + "=" * 80)
    print("OPTUNA TUNING COMPLETE")
    print(f"Best score: {study.best_value:.4f}")
    print(f"Best params: {json.dumps(study.best_params, indent=2)}")

    best = study.best_trial
    print(f"\nTest R²:   {best.user_attrs.get('test_r2', 'N/A')}")
    print(f"Valid R²:  {best.user_attrs.get('valid_r2', 'N/A')}")
    print(f"Train R²:  {best.user_attrs.get('train_r2', 'N/A')}")
    print(f"Test RMSE: {best.user_attrs.get('test_rmse', 'N/A')}")
    print(f"Rain Det:  {best.user_attrs.get('rain_det', 'N/A')}")

    # Save best config
    best_params = dict(study.best_params)
    reg_obj = best_params.pop("reg_objective")
    huber_alpha = best_params.pop("huber_alpha", None)
    best_params["reg_params"] = {"objective": reg_obj}
    if reg_obj == "huber" and huber_alpha is not None:
        best_params["reg_params"]["alpha"] = huber_alpha
    best_params["tweedie_variance_power"] = 1.5
    best_params["random_state"] = 42

    best_config = {**BASE_CONFIG, "model": {"type": "two_stage", "params": best_params}}
    out_path = ROOT / "config" / "train_config_optuna_best.json"
    _save_json(out_path, best_config)
    print(f"\nBest config saved to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
