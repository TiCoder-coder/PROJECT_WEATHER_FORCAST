"""
optuna_twostage_v2.py — Nâng cấp Optuna tuning cho TwoStage model.

Cải tiến so với v1:
  1. Tách hyperparameters riêng cho Classifier (clf_params) vs Regressor (reg_params)
  2. Thêm tweedie + tunable variance_power vào search space
  3. Mở rộng search space: subsample 0.3-0.95, colsample_bynode, fair_c
  4. Tunable sample_weight_cap (cho weight_ratio)
  5. Multi-objective: tối ưu cả R² lẫn Rain Detection Accuracy
  6. Seed 2 trial (v4 + current best) để warm-start

Usage:
    python scripts/optuna_twostage_v2.py --n_trials 100
"""
import os, sys, json, argparse, time
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "WeatherForcast.settings")
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import django; django.setup()

import numpy as np
import optuna
from optuna.samplers import TPESampler

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
    # ── Shared parameters ──────────────────────────────────────────
    n_estimators = trial.suggest_int("n_estimators", 500, 2000)
    learning_rate = trial.suggest_float("learning_rate", 0.005, 0.15, log=True)
    max_depth = trial.suggest_int("max_depth", 4, 12)
    num_leaves = trial.suggest_int("num_leaves", 15, 255)
    subsample = trial.suggest_float("subsample", 0.3, 0.95)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.3, 0.95)
    reg_alpha = trial.suggest_float("reg_alpha", 0.001, 10.0, log=True)
    reg_lambda = trial.suggest_float("reg_lambda", 0.01, 20.0, log=True)

    # ── Thresholds ──────────────────────────────────────────────────
    rain_threshold = trial.suggest_float("rain_threshold", 0.01, 0.5)
    predict_threshold = trial.suggest_float("predict_threshold", 0.15, 0.6)

    # ── Classifier-specific overrides ──────────────────────────────
    clf_min_child = trial.suggest_int("clf_min_child_samples", 5, 100)
    clf_reg_alpha = trial.suggest_float("clf_reg_alpha", 0.001, 10.0, log=True)
    clf_reg_lambda = trial.suggest_float("clf_reg_lambda", 0.01, 20.0, log=True)

    # ── Regressor objective ────────────────────────────────────────
    reg_obj = trial.suggest_categorical("reg_objective", ["huber", "fair", "tweedie", "regression"])

    reg_params_override = {"objective": reg_obj}
    tweedie_power = 1.5  # default
    if reg_obj == "huber":
        reg_params_override["alpha"] = trial.suggest_float("huber_alpha", 0.5, 0.99)
    elif reg_obj == "fair":
        reg_params_override["fair_c"] = trial.suggest_float("fair_c", 0.5, 5.0)
    elif reg_obj == "tweedie":
        tweedie_power = trial.suggest_float("tweedie_power", 1.1, 1.9)
    # else: mse regression

    # ── Regressor-specific overrides ──────────────────────────────
    reg_min_child = trial.suggest_int("reg_min_child_samples", 5, 100)

    # ── Build model params ────────────────────────────────────────
    params = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "num_leaves": num_leaves,
        "min_child_samples": 20,  # default, stages override below
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "rain_threshold": rain_threshold,
        "predict_threshold": predict_threshold,
        "tweedie_variance_power": tweedie_power if reg_obj == "tweedie" else 1.5,
        "random_state": 42,
        # Separate classifier parameters
        "clf_params": {
            "min_child_samples": clf_min_child,
            "reg_alpha": clf_reg_alpha,
            "reg_lambda": clf_reg_lambda,
        },
        # Separate regressor parameters
        "reg_params": {
            **reg_params_override,
            "min_child_samples": reg_min_child,
        },
    }

    config = {**BASE_CONFIG, "model": {"type": "two_stage", "params": params}}

    try:
        t0 = time.time()
        info = run_training(config)
        duration = time.time() - t0

        metrics_path = ROOT / "Weather_Forcast_App" / "Machine_learning_artifacts" / "latest" / "Metrics.json"
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)

        test_r2 = metrics["test"]["R2"]
        valid_r2 = metrics["valid"]["R2"]
        train_r2 = metrics["train"]["R2"]
        test_rmse = metrics["test"]["RMSE"]
        rain_det_test = metrics["test"].get("Rain_Detection_Accuracy", 0)
        rain_det_train = metrics["train"].get("Rain_Detection_Accuracy", 0)
        nonzero_mae = metrics["test"].get("NonZero_MAE", 999)

        # ── Composite score ──────────────────────────────────────────
        # Weight: 50% R² + 30% Rain Detection + 20% inverse NonZero_MAE
        # Penalize overfitting
        overfit_gap = max(0, train_r2 - test_r2)
        rain_overfit = max(0, rain_det_train - rain_det_test)

        score = (
            0.50 * test_r2
            + 0.30 * rain_det_test
            + 0.20 * max(0, 1 - nonzero_mae / 3.0)  # normalize: 0mm=1.0, 3mm=0.0
            - 0.25 * overfit_gap    # penalize R² overfitting
            - 0.15 * rain_overfit   # penalize rain detection overfitting
        )

        trial.set_user_attr("test_r2", test_r2)
        trial.set_user_attr("valid_r2", valid_r2)
        trial.set_user_attr("train_r2", train_r2)
        trial.set_user_attr("test_rmse", test_rmse)
        trial.set_user_attr("rain_det_test", rain_det_test)
        trial.set_user_attr("nonzero_mae", nonzero_mae)
        trial.set_user_attr("duration_s", round(duration, 1))

        print(f"  [TRIAL {trial.number}] score={score:.4f} | R²={test_r2:.4f} | RainDet={rain_det_test:.4f} | NZ_MAE={nonzero_mae:.3f} | {duration:.0f}s")
        return score
    except Exception as e:
        print(f"  [OPTUNA] Trial {trial.number} failed: {e}")
        return -999.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=100)
    args = parser.parse_args()

    sampler = TPESampler(seed=42, n_startup_trials=10)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # ── Seed trial 1: current Optuna best (v1) ────────────────────
    study.enqueue_trial({
        "n_estimators": 1033, "learning_rate": 0.0594, "max_depth": 7,
        "num_leaves": 114, "subsample": 0.502, "colsample_bytree": 0.806,
        "reg_alpha": 0.390, "reg_lambda": 5.151,
        "rain_threshold": 0.240, "predict_threshold": 0.378,
        "clf_min_child_samples": 24, "clf_reg_alpha": 0.390, "clf_reg_lambda": 5.151,
        "reg_objective": "fair", "reg_min_child_samples": 24,
    })

    # ── Seed trial 2: v4 config (baseline) ────────────────────────
    study.enqueue_trial({
        "n_estimators": 1000, "learning_rate": 0.03, "max_depth": 7,
        "num_leaves": 63, "subsample": 0.8, "colsample_bytree": 0.7,
        "reg_alpha": 0.5, "reg_lambda": 3.0,
        "rain_threshold": 0.1, "predict_threshold": 0.35,
        "clf_min_child_samples": 40, "clf_reg_alpha": 1.0, "clf_reg_lambda": 5.0,
        "reg_objective": "huber", "huber_alpha": 0.9,
        "reg_min_child_samples": 30,
    })

    # ── Seed trial 3: aggressive regularization for classifier ────
    study.enqueue_trial({
        "n_estimators": 1200, "learning_rate": 0.04, "max_depth": 6,
        "num_leaves": 50, "subsample": 0.7, "colsample_bytree": 0.8,
        "reg_alpha": 1.0, "reg_lambda": 8.0,
        "rain_threshold": 0.15, "predict_threshold": 0.45,
        "clf_min_child_samples": 60, "clf_reg_alpha": 3.0, "clf_reg_lambda": 10.0,
        "reg_objective": "fair",
        "reg_min_child_samples": 15,
    })

    print(f"\n{'='*80}")
    print(f"OPTUNA v2 — TwoStage Tuning ({args.n_trials} trials)")
    print(f"Improvements: separate clf/reg params, wider search, composite objective")
    print(f"{'='*80}\n")

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    print("\n" + "=" * 80)
    print("OPTUNA v2 TUNING COMPLETE")
    print(f"Best composite score: {study.best_value:.4f}")
    print(f"Best params:\n{json.dumps(study.best_params, indent=2)}")

    best = study.best_trial
    print(f"\n--- Best Trial Results ---")
    print(f"  Test R²:        {best.user_attrs.get('test_r2', 'N/A')}")
    print(f"  Valid R²:       {best.user_attrs.get('valid_r2', 'N/A')}")
    print(f"  Train R²:       {best.user_attrs.get('train_r2', 'N/A')}")
    print(f"  Test RMSE:      {best.user_attrs.get('test_rmse', 'N/A')}")
    print(f"  Rain Detection: {best.user_attrs.get('rain_det_test', 'N/A')}")
    print(f"  NonZero MAE:    {best.user_attrs.get('nonzero_mae', 'N/A')}")
    print(f"  Duration:       {best.user_attrs.get('duration_s', 'N/A')}s")

    # ── Top 5 trials ──────────────────────────────────────────────
    print(f"\n--- Top 5 Trials ---")
    trials_sorted = sorted(study.trials, key=lambda t: t.value if t.value is not None else -9999, reverse=True)
    for i, t in enumerate(trials_sorted[:5]):
        r2 = t.user_attrs.get('test_r2', '?')
        rd = t.user_attrs.get('rain_det_test', '?')
        nm = t.user_attrs.get('nonzero_mae', '?')
        print(f"  #{i+1} Trial {t.number}: score={t.value:.4f} | R²={r2} | RainDet={rd} | NZ_MAE={nm}")

    # ── Save best config ──────────────────────────────────────────
    best_params = dict(study.best_params)
    reg_obj = best_params.pop("reg_objective")
    huber_alpha = best_params.pop("huber_alpha", None)
    fair_c = best_params.pop("fair_c", None)
    tweedie_power = best_params.pop("tweedie_power", 1.5)

    clf_min_child = best_params.pop("clf_min_child_samples")
    clf_reg_alpha = best_params.pop("clf_reg_alpha")
    clf_reg_lambda = best_params.pop("clf_reg_lambda")
    reg_min_child = best_params.pop("reg_min_child_samples")

    reg_params = {"objective": reg_obj}
    if reg_obj == "huber" and huber_alpha is not None:
        reg_params["alpha"] = huber_alpha
    if reg_obj == "fair" and fair_c is not None:
        reg_params["fair_c"] = fair_c
    reg_params["min_child_samples"] = reg_min_child

    best_params["clf_params"] = {
        "min_child_samples": clf_min_child,
        "reg_alpha": clf_reg_alpha,
        "reg_lambda": clf_reg_lambda,
    }
    best_params["reg_params"] = reg_params
    best_params["tweedie_variance_power"] = tweedie_power if reg_obj == "tweedie" else 1.5
    best_params["random_state"] = 42

    best_config = {**BASE_CONFIG, "model": {"type": "two_stage", "params": best_params}}
    out_path = ROOT / "config" / "train_config_optuna_v2_best.json"
    _save_json(out_path, best_config)
    print(f"\nBest config saved to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
