"""
tune_optuna.py — Hyperparameter Tuning với Optuna cho WeatherTwoStageModel

Cách chạy:
    python3 tune_optuna.py                        # dùng config mặc định, 100 trials
    python3 tune_optuna.py --trials 200           # 200 trials
    python3 tune_optuna.py --metric r2            # tối ưu theo R² (mặc định)
    python3 tune_optuna.py --metric mae           # tối ưu theo MAE (minimize)
    python3 tune_optuna.py --metric rain_acc      # tối ưu theo Rain Detection Accuracy

Sau khi hoàn thành, best params được lưu vào:
    config/best_params_twostage.json

Để áp dụng best params, copy chúng vào phần "model.params" trong train_config.json
và chạy lại python3 -m Weather_Forcast_App.Machine_learning_model.trainning.train

Lưu ý:
    - Script này load dữ liệu từ Dataset_after_split/ (đã split sẵn)
    - Nếu chưa có split, chạy train.py trước để tạo split
    - Dùng TimeSeriesSplit để tránh data leakage
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Path setup ─────────────────────────────────────────────────────────────── #
THIS_FILE   = Path(__file__).resolve()
TRAIN_DIR   = THIS_FILE.parent                                 # trainning/
ML_ROOT     = TRAIN_DIR.parent                                 # Machine_learning_model/
APP_ROOT    = ML_ROOT.parent                                   # Weather_Forcast_App/
PROJECT_ROOT = APP_ROOT.parent                                 # project root

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Django setup (required before importing any model that uses Django ORM) ── #
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "WeatherForcast.settings")
import django
django.setup()

SPLIT_DIR   = ML_ROOT / "Dataset_after_split" / "Dataset_merge"
CONFIG_DIR  = ML_ROOT / "config"
ARTIFACTS   = APP_ROOT / "Machine_learning_artifacts" / "latest"


# ── Helpers ────────────────────────────────────────────────────────────────── #

def _load_split_csv(name: str) -> pd.DataFrame:
    """Load train/valid/test CSV từ Dataset_after_split/Dataset_merge/."""
    # Tìm file theo pattern merge_<name>.csv
    patterns = [f"merge_{name}.csv", f"*{name}*.csv"]
    for pattern in patterns:
        found = list(SPLIT_DIR.glob(pattern))
        if found:
            df = pd.read_csv(found[0], encoding="utf-8-sig")
            print(f"  [DATA] Loaded {name}: {df.shape}  ← {found[0].name}")
            return df
    raise FileNotFoundError(
        f"Không tìm thấy file split '{name}' trong {SPLIT_DIR}\n"
        f"Hãy chạy train.py trước để tạo split."
    )


def _load_transform_pipeline():
    """Load WeatherTransformPipeline đã fit từ artifacts/latest/."""
    pipeline_path = ARTIFACTS / "Transform_pipeline.pkl"
    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy Transform_pipeline.pkl tại {pipeline_path}\n"
            f"Hãy chạy train.py trước."
        )
    from Weather_Forcast_App.Machine_learning_model.features.Transformers import WeatherTransformPipeline
    pipeline = WeatherTransformPipeline.load(pipeline_path)
    print(f"  [PIPELINE] Loaded WeatherTransformPipeline from {pipeline_path}")
    return pipeline


def _load_feature_list() -> Dict[str, Any]:
    fl_path = ARTIFACTS / "Feature_list.json"
    if not fl_path.exists():
        raise FileNotFoundError(f"Không tìm thấy Feature_list.json tại {fl_path}")
    with open(fl_path, encoding="utf-8") as f:
        return json.load(f)


def _prepare_data():
    """
    Load train/valid splits, build engineered features, then transform với
    pipeline đã fit.  Mirrors the feature engineering steps in train.py so
    that the 50-column pipeline receives correctly shaped input.

    Returns (X_train, y_train, X_val, y_val)
    """
    from Weather_Forcast_App.Machine_learning_model.features.Build_transfer import WeatherFeatureBuilder

    feature_info  = _load_feature_list()
    target_col    = feature_info.get("target_column", "rain_total")
    all_features  = feature_info.get("all_feature_columns", [])
    applied_log   = feature_info.get("applied_log_target", False)
    removed_static = set(feature_info.get("removed_static_features", []))
    removed_const  = set(feature_info.get("removed_constant_features", []))

    train_info_path = ARTIFACTS / "Train_info.json"
    if train_info_path.exists():
        with open(train_info_path) as f:
            ti = json.load(f)
        applied_log = ti.get("target_transform", {}).get("log1p_applied", applied_log)

    config_path = CONFIG_DIR / "train_config.json"
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    poly_cfg = cfg.get("polynomial_features", {})
    poly_enabled = poly_cfg.get("enabled", False)
    poly_degree  = poly_cfg.get("degree", 2)
    poly_top_k   = poly_cfg.get("top_k_corr", 8)

    df_train = _load_split_csv("train")
    df_valid = _load_split_csv("valid")
    pipeline = _load_transform_pipeline()

    def _extract(df: pd.DataFrame) -> tuple:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        y = df[target_col].copy()

        # ── Step 1: feature builder — use same config as train.py ────── #
        # Pass the features config so lag/rolling/diff are disabled for
        # cross-sectional data (same as what train.py detected at runtime).
        feature_cfg = cfg.get("features", {})
        builder = WeatherFeatureBuilder(config=feature_cfg if feature_cfg else None)
        df_built = builder.build_all_features(df.copy(), target_column=target_col)

        # ── Step 2: drop static / constant columns ─────────────────── #
        drop_cols = (removed_static | removed_const) & set(df_built.columns)
        if drop_cols:
            df_built = df_built.drop(columns=list(drop_cols))

        # ── Step 3: polynomial features (replicate train.py logic) ─── #
        if poly_enabled:
            from Weather_Forcast_App.Machine_learning_model.trainning.train import _add_polynomial_features
            y_for_poly = df_built[target_col] if target_col in df_built.columns else y
            df_built, _ = _add_polynomial_features(
                df_built, y_for_poly, top_k=poly_top_k, degree=poly_degree
            )

        # ── Step 4: keep only the 50 features the pipeline knows ───── #
        feat_cols = [c for c in all_features if c in df_built.columns]
        missing   = [c for c in all_features if c not in df_built.columns]
        if missing:
            # Fill any missing engineered columns with 0 so pipeline doesn't crash
            for mc in missing:
                df_built[mc] = 0.0
            feat_cols = list(all_features)

        X_raw = df_built[feat_cols].copy()

        # ── Step 5: transform (pipeline already fit) ───────────────── #
        X_t = pipeline.transform(X_raw)
        if isinstance(X_t, np.ndarray):
            X_t = pd.DataFrame(X_t, columns=feat_cols[:X_t.shape[1]], index=X_raw.index)

        if applied_log:
            y = np.log1p(y.clip(lower=0))

        return X_t.values.astype(np.float32), y.fillna(0).values.astype(np.float32)

    print("\n[OPTUNA] Preparing data (with feature engineering)...")
    X_train, y_train = _extract(df_train)
    X_val,   y_val   = _extract(df_valid)
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")
    return X_train, y_train, X_val, y_val


# ── Optuna objective ───────────────────────────────────────────────────────── #

def make_objective(X_train, y_train, X_val, y_val, metric: str = "r2"):
    """
    Tạo Optuna objective function.
    metric: "r2" (maximize), "mae" (minimize), "rain_acc" (maximize)
    """
    from Weather_Forcast_App.Machine_learning_model.Models.TwoStage_Model import WeatherTwoStageModel

    def objective(trial):
        params = {
            "n_estimators":          trial.suggest_int("n_estimators", 200, 2000, step=100),
            "learning_rate":         trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth":             trial.suggest_int("max_depth", 4, 12),
            "num_leaves":            trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples":     trial.suggest_int("min_child_samples", 5, 50),
            "subsample":             trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":      trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":             trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":            trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.0, 2.0),
            "rain_threshold":        trial.suggest_float("rain_threshold", 0.05, 0.5),
            "predict_threshold":     trial.suggest_float("predict_threshold", 0.1, 0.6),
            "random_state": 42,
            "verbose": False,
        }

        try:
            model = WeatherTwoStageModel(**params)
            result = model.train(
                X_train, y_train,
                X_val=X_val, y_val=y_val,
                val_size=0,
                scale_features=False,
                verbose=False,
            )

            if not result.success or result.metrics is None:
                return float("nan")

            m = result.metrics
            if metric == "r2":
                return m.get("R2", float("-inf"))
            elif metric == "mae":
                return m.get("MAE", float("inf"))
            elif metric == "rain_acc":
                return m.get("Rain_Detection_Accuracy", float("-inf"))
            elif metric == "rmse":
                return m.get("RMSE", float("inf"))
            else:
                return m.get("R2", float("-inf"))

        except Exception as e:
            return float("nan")

    return objective


def _direction(metric: str) -> str:
    return "minimize" if metric in ("mae", "rmse") else "maximize"


# ── Main ───────────────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning cho WeatherTwoStageModel")
    parser.add_argument("--trials",  type=int, default=100,  help="Số Optuna trials (default: 100)")
    parser.add_argument("--metric",  type=str, default="r2", choices=["r2", "mae", "rmse", "rain_acc"],
                        help="Metric tối ưu (default: r2)")
    parser.add_argument("--timeout", type=int, default=0,    help="Timeout giây (0 = không giới hạn)")
    parser.add_argument("--output",  type=str, default=None, help="File output JSON (default: config/best_params_twostage.json)")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else CONFIG_DIR / "best_params_twostage.json"

    print("=" * 60)
    print("  Optuna Tuning — WeatherTwoStageModel")
    print(f"  Trials   : {args.trials}")
    print(f"  Metric   : {args.metric} ({_direction(args.metric)})")
    print(f"  Output   : {output_path}")
    print("=" * 60)

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ── Load data ──────────────────────────────────────────────────── #
    X_train, y_train, X_val, y_val = _prepare_data()

    # ── Create & run study ─────────────────────────────────────────── #
    sampler   = optuna.samplers.TPESampler(seed=42)
    pruner    = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
    direction = _direction(args.metric)

    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        study_name=f"twostage_{args.metric}",
    )

    objective = make_objective(X_train, y_train, X_val, y_val, metric=args.metric)

    t0 = time.time()
    study.optimize(
        objective,
        n_trials=args.trials,
        timeout=args.timeout if args.timeout > 0 else None,
        show_progress_bar=True,
    )
    elapsed = time.time() - t0

    # ── Results ────────────────────────────────────────────────────── #
    best = study.best_trial
    print("\n" + "=" * 60)
    print(f"  Best Trial  : #{best.number}")
    print(f"  Best {args.metric.upper():8s}: {best.value:.6f}")
    print(f"  Total time  : {elapsed:.1f}s  ({elapsed/max(len(study.trials),1):.1f}s/trial)")
    print("  Best Params :")
    for k, v in best.params.items():
        print(f"    {k:30s}: {v}")
    print("=" * 60)

    # ── Save ───────────────────────────────────────────────────────── #
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_json = {
        "optimized_metric": args.metric,
        "best_value": best.value,
        "n_trials": len(study.trials),
        "elapsed_seconds": round(elapsed, 1),
        "best_params": best.params,
        "usage": (
            "Copy nội dung 'best_params' vào 'model.params' trong "
            "Weather_Forcast_App/Machine_learning_model/config/train_config.json "
            "rồi chạy lại training."
        ),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Best params saved → {output_path}")


if __name__ == "__main__":
    main()
