#!/usr/bin/env python3
"""Run diagnostics after training: baseline, top errors, per-station RMSE.
Usage: python3 scripts/run_diagnostics.py
Generates: debug_top50_errors.csv, debug_worst_stations.csv and prints summary to stdout.
"""
import json
import joblib
import pathlib
import sys
import traceback
from typing import Any

import numpy as np
import pandas as pd
import time
import os


ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "Weather_Forcast_App" / "Machine_learning_artifacts" / "latest"
# Ensure project root is on sys.path so unpickling can import local package modules
import sys as _sys
if str(ROOT) not in _sys.path:
    _sys.path.insert(0, str(ROOT))


def load_info() -> Any:
    return json.loads((ART / "Train_info.json").read_text(encoding="utf-8"))


def _normalize_feature_names(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return [str(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return None


def _get_feature_names_from_estimator(estimator: Any) -> list[str] | None:
    if hasattr(estimator, "get_booster"):
        try:
            booster = estimator.get_booster()
            names = getattr(booster, "feature_names", None)
            normalized = _normalize_feature_names(names)
            if normalized:
                return normalized
        except Exception:
            pass

    for attr in (
        "feature_names",
        "feature_name_",
        "feature_names_",
        "feature_names_in_",
        "feature_names_out_",
        "feature_name_out",
        "columns_",
    ):
        value = getattr(estimator, attr, None)
        normalized = _normalize_feature_names(value)
        if normalized:
            return normalized

    return None


def _infer_model_feature_names(model: Any) -> list[str] | None:
    names = _get_feature_names_from_estimator(model)
    if names:
        return names

    if hasattr(model, "get_base_models"):
        for base in model.get_base_models():
            base_estimator = getattr(base, "model", base)
            names = _get_feature_names_from_estimator(base_estimator)
            if names:
                return names

    return None


def main():
    try:
        info = load_info()
    except Exception as e:
        print("[ERROR] Cannot read Train_info.json:", e)
        sys.exit(1)

    test_path = info.get("split_saved_paths", {}).get("test")
    if not test_path:
        print("[ERROR] test path not found in Train_info.json")
        sys.exit(1)

    print(f"[INFO] Loading test file: {test_path}")
    # Allow reading a sample for faster diagnostics via env DIAG_NROWS
    nrows = int(os.environ.get("DIAG_NROWS", "0"))
    if nrows and nrows > 0:
        print(f"[INFO] Reading only first {nrows} rows from test for faster run (DIAG_NROWS={nrows})")
        df = pd.read_csv(test_path, nrows=nrows)
    else:
        df = pd.read_csv(test_path)
    target_col = info.get("target_column", "rain_total")
    if target_col not in df.columns:
        print(f"[ERROR] target column '{target_col}' not present in test dataframe columns")
        print("Columns:", df.columns.tolist())
        sys.exit(1)

    y = df[target_col].values
    baseline = np.mean(y)
    baseline_rmse = float(np.sqrt(((y - baseline) ** 2).mean()))

    metrics = json.loads((ART / "Metrics.json").read_text(encoding="utf-8"))
    model_rmse = metrics.get("test", {}).get("RMSE")

    print("[SUMMARY]")
    print(f" Baseline(mean) RMSE: {baseline_rmse:.4f}")
    print(f" Model test RMSE:      {model_rmse}")

    # Load artifacts
    print("[INFO] Loading model and pipeline...")
    model = joblib.load(ART / "Model.pkl")
    pipeline = None
    try:
        pipeline = joblib.load(ART / "Transform_pipeline.pkl")
    except Exception:
        # Try class loader fallback
        from Weather_Forcast_App.Machine_learning_model.features.Transformers import WeatherTransformPipeline
        pipeline = WeatherTransformPipeline.load(ART / "Transform_pipeline.pkl")

    # Some saved artifacts may be dicts containing metadata; try to extract an object with
    # a .transform method (common keys: 'pipeline', 'transformer', 'pipe', 'transform_pipeline').
    if isinstance(pipeline, dict):
        print("[DEBUG] Loaded pipeline is a dict; attempting to extract transform object from keys...")
        # If the dict follows our custom save format it may contain 'steps' (list of transformer instances)
        if "steps" in pipeline and isinstance(pipeline["steps"], list):
            steps = pipeline["steps"]

            class _TransformWrapper:
                def __init__(self, steps):
                    self.steps = steps

                def transform(self, X):
                    Xt = X
                    for s in self.steps:
                        if hasattr(s, "transform"):
                            Xt = s.transform(Xt)
                        elif hasattr(s, "fit_transform"):
                            # fallback if transformer only implements fit_transform
                            Xt = s.fit_transform(Xt)
                        else:
                            raise RuntimeError(f"Step {s} has no transform method")
                    return Xt

            pipeline = _TransformWrapper(steps)
            print("[DEBUG] Wrapped 'steps' list into a transformable pipeline wrapper")
        else:
            # common keys to try
            for key in ("pipeline", "transformer", "pipe", "transform_pipeline"):
                if key in pipeline and hasattr(pipeline[key], "transform"):
                    pipeline = pipeline[key]
                    print(f"[DEBUG] Extracted pipeline from key: {key}")
                    break
            else:
                # search values for first object with transform
                found = False
                for k, v in pipeline.items():
                    if hasattr(v, "transform"):
                        pipeline = v
                        print(f"[DEBUG] Extracted pipeline from dict value key: {k}")
                        found = True
                        break
                if not found:
                    print("[ERROR] Could not find transformable object inside Transform_pipeline.pkl dict.\n" \
                          "Please inspect the pickle contents or re-save the transform pipeline as a single object.")
                    sys.exit(1)

    # Build features using builder
    from Weather_Forcast_App.Machine_learning_model.features.Build_transfer import WeatherFeatureBuilder

    builder = WeatherFeatureBuilder()
    print("[INFO] Building features for test set (this may take a while)...")
    t0 = time.time()
    df_feat = builder.build_all_features(df, target_column=target_col, group_by=info.get("group_by"))
    t1 = time.time()
    print(f"[INFO] Feature building done in {t1-t0:.1f}s")

    # Align expected features
    feat = json.loads((ART / "Feature_list.json").read_text(encoding="utf-8"))
    expected = feat.get("all_feature_columns", [])
    for c in expected:
        if c not in df_feat.columns:
            df_feat[c] = 0

    X = df_feat[expected]
    print(f"[INFO] Transforming features (n_rows={len(X)}, n_cols={len(X.columns)})")
    t0 = time.time()
    X_t = pipeline.transform(X)
    t1 = time.time()
    print(f"[INFO] Transform done in {t1-t0:.1f}s")

    print("[INFO] Predicting...")
    # Ensure X_t has dtypes acceptable for model.predict (xgboost needs numeric/bool/category)
    # Convert object (str) columns: try numeric -> datetime -> categorical codes
    obj_cols = X_t.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        print(f"[DEBUG] Converting object columns before predict: {obj_cols}")
        for c in obj_cols:
            # try numeric
            coerced = pd.to_numeric(X_t[c], errors="coerce")
            if coerced.notna().any():
                X_t[c] = coerced
                continue
            # try datetime -> epoch seconds
            try:
                dt = pd.to_datetime(X_t[c], errors="coerce")
                if dt.notna().any():
                    X_t[c] = dt.astype('int64') // 10**9
                    continue
            except Exception:
                pass
            # fallback: categorical codes
            X_t[c] = X_t[c].astype('category').cat.codes

    # Align X_t columns to the features used at training time
    model_feature_names = _infer_model_feature_names(model)
    if model_feature_names is None:
        model_feature_names = expected

    # If X_t is numpy array, try to convert to DataFrame using pipeline or feature metadata
    if isinstance(X_t, (list, tuple)) or (hasattr(X_t, "ndim") and getattr(X_t, "ndim") == 2 and not hasattr(X_t, "columns")):
        # attempt to build DataFrame
        cols = None
        if isinstance(pipeline, dict) and "feature_names" in pipeline:
            cols = pipeline["feature_names"]
        elif hasattr(pipeline, "feature_names_in_"):
            cols = list(pipeline.feature_names_in_)
        elif hasattr(pipeline, "steps"):
            # not reliable, skip
            cols = None
        if cols is None:
            cols = model_feature_names
        X_t = pd.DataFrame(X_t, columns=cols)

    # Now ensure only model_feature_names present and in same order
    if isinstance(X_t, pd.DataFrame):
        missing = [c for c in model_feature_names if c not in X_t.columns]
        extra = [c for c in X_t.columns if c not in model_feature_names]
        if missing:
            print(f"[WARN] training data did not have the following fields: {missing}")
            for c in missing:
                X_t[c] = 0
        if extra:
            print(f"[DEBUG] Dropping extra columns not expected by model: {extra}")
            X_t = X_t.drop(columns=extra)
        # Reorder
        X_t = X_t[model_feature_names]

    pred = model.predict(X_t)
    # some wrappers return object
    try:
        pred = getattr(pred, "predictions", pred)
    except Exception:
        pass
    pred = np.array(pred).reshape(-1)

    df_out = df.copy()
    df_out["y_true"] = y
    df_out["y_pred"] = pred
    df_out["abs_err"] = (df_out["y_true"] - df_out["y_pred"]).abs()

    top50 = df_out.sort_values("abs_err", ascending=False).head(50)
    top50_path = ROOT / "debug_top50_errors.csv"
    top50.to_csv(top50_path, index=False)
    print(f"[OK] Saved top 50 errors to: {top50_path}")

    # Per-station RMSE
    if "station_id" in df_out.columns:
        import numpy as _np

        grp = df_out.groupby("station_id").agg(n=(target_col, "size"), rmse=(lambda x: _np.sqrt(((x - df_out.loc[x.index, 'y_pred']) ** 2).mean())))
        # Above agg may be unstable; compute properly
        def station_rmse(sub):
            return float(np.sqrt(((sub[target_col] - sub['y_pred']) ** 2).mean()))

        grp2 = df_out.groupby("station_id").apply(station_rmse).rename("rmse").reset_index()
        grp2 = grp2.sort_values("rmse", ascending=False)
        stations_path = ROOT / "debug_worst_stations.csv"
        grp2.to_csv(stations_path, index=False)
        print(f"[OK] Saved per-station RMSE to: {stations_path}")
    else:
        print("[WARN] 'station_id' not in dataframe; skipping per-station RMSE")

    print("[DONE]")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(2)
