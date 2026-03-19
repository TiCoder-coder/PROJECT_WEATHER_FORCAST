from __future__ import annotations
import sys
from pathlib import Path

# =============================================================================
# TRAIN STACKING ENSEMBLE — Tổng chỉ huy riêng cho StackingEnsemble
# =============================================================================
"""
train_stacking_ensemble.py

File training chuyên biệt cho StackingEnsemble (Super Learner 2 tầng).

Flow chuẩn:
    1) Load config (.json)
    2) Load data (Loader.py)
    3) Validate schema (Schema.py)
    4) Split train/valid/test (Split.py) + lưu ra Dataset_after_split/
    5) Build features (Build_transfer.py) + clean + feature selection
    6) Transform pipeline (Transformers.py) — fit on train only
    7) Train StackingEnsemble:
           Stage 6: Verify base models
           Stage 7: OOF Classification (TimeSeriesSplit)
           Stage 8: OOF Regression rainy-only (TimeSeriesSplit)
           Stage 9: Refit tất cả base models trên full train
    8) Evaluate metrics (classify + regression rainy-only + end-to-end)
    9) Overfit/Underfit diagnostics
   10) Save artifacts:
           - stacking_ensemble_<timestamp>.joblib
           - Transform_pipeline.pkl
           - Feature_list.json
           - Metrics.json
           - Train_info.json

Chạy:
    python -m Weather_Forcast_App.Machine_learning_model.trainning.train_stacking_ensemble \
        --config config/train_config.json

    hoặc trực tiếp:
    python .../train_stacking_ensemble.py --config config/train_config.json

Config mẫu (thêm/override trong train_config.json):
    {
        "stacking": {
            "n_splits": 5,
            "predict_threshold": 0.4,
            "verbose": true
        }
    }
    Các key khác hoàn toàn tương đương với train_ensemble_average.py:
        data, split, features, transform, target_column, group_by,
        forecast_horizon, feature_selection, polynomial_features, ...
"""

THIS_FILE = Path(__file__).resolve()

# ── Tìm project root (thư mục chứa Weather_Forcast_App) ──
project_root = THIS_FILE
for _ in range(5):
    if (project_root / "Weather_Forcast_App").exists():
        break
    project_root = project_root.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Dùng _load_df_via_loader từ tuning.py (đã có sẵn, không cần copy)
from Weather_Forcast_App.Machine_learning_model.trainning.tuning import _load_df_via_loader

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# IMPORTS CÁC MODULE NỘI BỘ
# =============================================================================
from Weather_Forcast_App.Machine_learning_model.data.Loader import DataLoader
from Weather_Forcast_App.Machine_learning_model.data.Schema import validate_weather_dataframe
from Weather_Forcast_App.Machine_learning_model.data.Split import SplitConfig, split_dataframe

from Weather_Forcast_App.Machine_learning_model.features.Build_transfer import WeatherFeatureBuilder
from Weather_Forcast_App.Machine_learning_model.features.Transformers import WeatherTransformPipeline

from Weather_Forcast_App.Machine_learning_model.evaluation.metrics import (
    calculate_all_metrics,
    RAIN_THRESHOLD,
)

from Weather_Forcast_App.Machine_learning_model.Models.Ensemble_Stacking_Model import StackingEnsemble


# =============================================================================
# TIỆN ÍCH
# =============================================================================

def _now_tag() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    ext = path.suffix.lower()
    if ext == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if ext in [".yml", ".yaml"]:
        try:
            import yaml
            return yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as e:
            raise RuntimeError("Config is YAML but PyYAML not installed.") from e
    raise ValueError(f"Unsupported config extension: {ext}")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    import enum
    def _default(obj):
        if isinstance(obj, enum.Enum):
            return obj.value
        if hasattr(obj, "__dataclass_fields__"):
            from dataclasses import asdict as dc_asdict
            try:
                return dc_asdict(obj)
            except Exception:
                pass
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)
    _ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=_default), encoding="utf-8")


def _save_split_csvs(
    out_root: Path,
    split_name: str,
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
) -> Dict[str, str]:
    _ensure_dir(out_root)
    train_path = out_root / f"{split_name}_train.csv"
    valid_path = out_root / f"{split_name}_valid.csv"
    test_path  = out_root / f"{split_name}_test.csv"
    df_train.to_csv(train_path, index=False, encoding="utf-8-sig")
    df_valid.to_csv(valid_path, index=False, encoding="utf-8-sig")
    df_test.to_csv(test_path,  index=False, encoding="utf-8-sig")
    return {
        "train": str(train_path),
        "valid": str(valid_path),
        "test":  str(test_path),
    }


# =============================================================================
# DIAGNOSTICS: loại bỏ dòng dữ liệu sai từ lần chạy diagnostics trước
# =============================================================================

def _drop_known_bad_rows(df: pd.DataFrame, root_path: Path) -> pd.DataFrame:
    diagnostics_path = root_path / "debug_top50_errors.csv"
    if not diagnostics_path.exists():
        return df
    try:
        bad_df = pd.read_csv(diagnostics_path)
    except Exception as e:
        print(f"  [DIAGNOSTICS] Warning: could not read {diagnostics_path}: {e}")
        return df
    for col in ("y_true", "y_pred", "abs_err"):
        if col in bad_df.columns:
            bad_df = bad_df.drop(columns=[col])
    common_cols = [c for c in bad_df.columns if c in df.columns]
    if not common_cols:
        return df
    bad_keys = set(
        bad_df[common_cols].apply(lambda row: "|".join(map(str, row)), axis=1)
    )
    mask = df[common_cols].apply(lambda row: "|".join(map(str, row)), axis=1).isin(bad_keys)
    n_dropped = int(mask.sum())
    if n_dropped > 0:
        print(f"  [DIAGNOSTICS] Removed {n_dropped} known bad rows "
              f"(source: {diagnostics_path.name})")
    return df[~mask].reset_index(drop=True)


# =============================================================================
# SCHEMA VALIDATION
# =============================================================================

def _validate_schema_keep_valid_rows(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    total_before = len(df)
    schemas = validate_weather_dataframe(df)
    valid_records = [s.to_flat_dict() for s in schemas]
    valid_df = pd.DataFrame(valid_records)
    return valid_df, {
        "rows_before":  int(total_before),
        "rows_after":   int(len(valid_df)),
        "rows_dropped": int(total_before - len(valid_df)),
    }


# =============================================================================
# FEATURE HELPERS (copy pattern từ train_ensemble_average.py)
# =============================================================================

_STATIC_COL_KEYWORDS = [
    'location_vi_do', 'location_kinh_do', 'location_ma_tram',
    'location_tinh_thanh_pho', 'location_huyen',
    'vi_do', 'kinh_do', 'latitude', 'longitude',
]


def _is_static_derived_feature(col_name: str) -> bool:
    col_lower = col_name.lower()
    temporal_suffixes = ['_lag_', '_rolling_', '_diff_', '_pct_change_']
    for kw in _STATIC_COL_KEYWORDS:
        if kw in col_lower and any(suf in col_lower for suf in temporal_suffixes):
            return True
    return False


def _remove_static_derived_features(
    X: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    remove_cols = [col for col in X.columns if _is_static_derived_feature(col)]
    if remove_cols:
        print(f"  [FEATURE CLEAN] Removed {len(remove_cols)} static-derived features "
              f"(lat/lon lag/rolling/diff)")
    return X.drop(columns=remove_cols, errors='ignore'), remove_cols


def _remove_constant_features(
    X: pd.DataFrame, threshold: float = 0.001
) -> Tuple[pd.DataFrame, List[str]]:
    remove_cols = []
    for col in X.columns:
        if X[col].nunique() <= 1:
            remove_cols.append(col)
        elif X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            col_std = X[col].std()
            if col_std == 0 or (col_std is not None and np.isnan(col_std)):
                remove_cols.append(col)
    if remove_cols:
        print(f"  [FEATURE CLEAN] Removed {len(remove_cols)} constant features")
    return X.drop(columns=remove_cols, errors='ignore'), remove_cols


def _add_polynomial_features(
    X: pd.DataFrame, y: pd.Series, top_k: int = 8, degree: int = 2,
) -> Tuple[pd.DataFrame, List[str]]:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return X, []
    correlations = X[numeric_cols].corrwith(y).abs().dropna().sort_values(ascending=False)
    top_cols = correlations.head(top_k).index.tolist()
    new_features: Dict[str, Any] = {}
    new_names: List[str] = []
    for i, col_a in enumerate(top_cols):
        fname = f"{col_a}_sq"
        new_features[fname] = X[col_a] ** 2
        new_names.append(fname)
        for col_b in top_cols[i + 1:]:
            fname = f"{col_a}_x_{col_b}"
            new_features[fname] = X[col_a] * X[col_b]
            new_names.append(fname)
    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)
        print(f"  [POLY FEATURES] Added {len(new_names)} polynomial features "
              f"from top-{top_k} correlated columns")
    return X, new_names


def _select_features_by_importance(
    X_train: pd.DataFrame, y_train: pd.Series,
    max_features: int = 50,
    min_importance: float = 0.0,
) -> List[str]:
    """SHAP-based feature selection (fallback: LightGBM split importance)."""
    from lightgbm import LGBMRegressor
    y_fit = np.log1p(np.abs(y_train.values.astype(float)))
    selector = LGBMRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, verbose=-1,
        random_state=42, n_jobs=-1,
    )
    selector.fit(X_train, y_fit)
    try:
        import shap
        n_sample = min(1000, len(X_train))
        X_sample = X_train.iloc[:n_sample]
        explainer = shap.TreeExplainer(selector)
        shap_vals = explainer.shap_values(X_sample)
        if isinstance(shap_vals, list):
            shap_vals = np.abs(np.array(shap_vals)).mean(axis=0)
        importances = np.abs(shap_vals).mean(axis=0)
        feature_imp = sorted(zip(X_train.columns, importances), key=lambda x: x[1], reverse=True)
        selected = [name for name, imp in feature_imp if imp > min_importance][:max_features]
        if len(selected) < min(10, len(X_train.columns)):
            selected = [name for name, _ in feature_imp[:max_features]]
        print(f"  [FEATURE SELECT SHAP] {len(selected)}/{len(X_train.columns)} features kept")
        return selected
    except Exception as shap_err:
        print(f"  [FEATURE SELECT] SHAP unavailable ({shap_err}), fallback to LGB importance")
    try:
        importances = selector.feature_importances_
        feature_imp = sorted(zip(X_train.columns, importances), key=lambda x: x[1], reverse=True)
        selected = [name for name, imp in feature_imp if imp > min_importance][:max_features]
        if len(selected) < min(10, len(X_train.columns)):
            selected = [name for name, _ in feature_imp[:max_features]]
        print(f"  [FEATURE SELECT LGB] {len(selected)}/{len(X_train.columns)} features kept")
        return selected
    except Exception as e:
        print(f"  [FEATURE SELECT] Cannot run feature selection: {e}")
        return X_train.columns.tolist()


def _detect_data_type(df: pd.DataFrame) -> str:
    time_cols = [c for c in df.columns
                 if 'time' in c.lower() or 'date' in c.lower() or 'stamp' in c.lower()]
    if not time_cols:
        return 'cross_sectional'
    ts = pd.to_datetime(df[time_cols[0]], errors='coerce')
    n_unique_ts = ts.dropna().nunique()
    n_rows = len(df)
    ts_ratio = n_unique_ts / max(n_rows, 1)
    if n_unique_ts <= 5 or ts_ratio < 0.01:
        return 'cross_sectional'
    elif ts_ratio > 0.3:
        return 'time_series'
    return 'mixed'


def _build_features_for_split(
    builder: WeatherFeatureBuilder,
    df: pd.DataFrame,
    target_col: str,
    group_by: Optional[str] = None,
    forecast_horizon: int = 0,
    leaked_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in dataframe.")
    df_feat = builder.build_all_features(df, target_column=target_col, group_by=group_by)
    if forecast_horizon > 0:
        if group_by and group_by in df_feat.columns:
            df_feat[target_col] = df_feat.groupby(group_by)[target_col].shift(-forecast_horizon)
        else:
            df_feat[target_col] = df_feat[target_col].shift(-forecast_horizon)
        df_feat = df_feat.dropna(subset=[target_col])
        print(f"  [FORECAST] Shifted target by -{forecast_horizon} rows. "
              f"Remaining: {len(df_feat)} rows")
    y = df_feat[target_col].copy()
    X = df_feat.drop(columns=[target_col])
    if leaked_columns:
        cols_to_drop = [c for c in leaked_columns if c in X.columns]
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)
            print(f"  [FORECAST] Removed {len(cols_to_drop)} leaked columns")
    return X, y


# =============================================================================
# OVERFIT / UNDERFIT DIAGNOSTICS
# =============================================================================

def _detect_overfit_underfit(
    metrics_dict: Dict[str, Any], tolerance: float = 0.10
) -> Tuple[str, str]:
    """
    Ưu tiên:
        1. p_rain classification metrics (Recall, F1)
        2. rain_mm end-to-end MAE gap
        3. R² gap (fallback, unreliable với zero-inflated)
    """
    train = metrics_dict.get("train", {})
    valid = metrics_dict.get("valid", {})
    test  = metrics_dict.get("test",  {})

    # --- 1. Dùng Rain_F1 (từ StackingEnsemble.evaluate) ---
    f1_train = train.get("cls_f1") or train.get("Rain_F1")
    f1_valid = valid.get("cls_f1") or valid.get("Rain_F1")
    if f1_train is not None and f1_valid is not None:
        gap = f1_train - f1_valid
        tri = (f"Train F1={f1_train:.3f} / Valid F1={f1_valid:.3f}"
               + (f" / Test F1={test.get('cls_f1') or test.get('Rain_F1', 'N/A')}"
                  if test else ""))
        if gap > tolerance:
            return "overfit", f"F1 Train({f1_train:.3f}) > Valid({f1_valid:.3f}) by {gap:.3f} — {tri}"
        elif gap < -tolerance:
            return "underfit", f"F1 Valid({f1_valid:.3f}) > Train({f1_train:.3f}) — {tri}"
        else:
            return "good", f"F1 consistent across splits — {tri}"

    # --- 2. Dùng Rain_Detection_Accuracy ---
    acc_train = train.get("Rain_Detection_Accuracy")
    acc_valid = valid.get("Rain_Detection_Accuracy")
    if acc_train is not None and acc_valid is not None:
        gap = acc_train - acc_valid
        if gap > tolerance:
            return "overfit", f"RainAcc Train({acc_train:.3f}) > Valid({acc_valid:.3f}) by {gap:.3f}"
        elif gap < -tolerance:
            return "underfit", f"RainAcc Valid({acc_valid:.3f}) > Train({acc_train:.3f})"
        else:
            return "good", f"RainAcc consistent — Train={acc_train:.3f} / Valid={acc_valid:.3f}"

    # --- 3. R² fallback ---
    r2_train = train.get("R2") or train.get("e2e_r2")
    r2_valid = valid.get("R2") or valid.get("e2e_r2")
    if r2_train is not None and r2_valid is not None:
        gap = r2_train - r2_valid
        if gap > tolerance:
            return "overfit", f"R² Train({r2_train:.3f}) - Valid({r2_valid:.3f}) = {gap:.3f}"
        elif gap < -tolerance:
            return "underfit", f"R² Valid({r2_valid:.3f}) > Train({r2_train:.3f})"
        else:
            return "good", f"R² gap small ({gap:.3f})"

    return "unknown", "Insufficient metrics for overfit/underfit detection."


# =============================================================================
# MAIN TRAINING FLOW
# =============================================================================

def run_stacking_training(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hàm training chính, trả về train_info dict.
    """

    # ── Resolve paths ────────────────────────────────────────────────────────
    ml_model_root = THIS_FILE.parents[1]           # Machine_learning_model
    app_root      = THIS_FILE.parents[2]           # Weather_Forcast_App

    dataset_after_split_merge     = ml_model_root / "Dataset_after_split" / "Dataset_merge"
    dataset_after_split_not_merge = ml_model_root / "Dataset_after_split" / "Dataset_not_merge"
    artifacts_latest              = app_root / "Machine_learning_artifacts" / "stacking_ensemble" / "latest"
    _ensure_dir(artifacts_latest)

    # ── Đọc config ───────────────────────────────────────────────────────────
    data_cfg      = config.get("data") or {}
    feature_cfg   = config.get("features", {})
    split_cfg     = config.get("split", {})
    transform_cfg = config.get("transform", {})
    stacking_cfg  = config.get("stacking", {})    # ← phần riêng cho StackingEnsemble

    target_col = config.get("target_column", "luong_mua_hien_tai")
    group_by   = config.get("group_by")

    folder_key = data_cfg.get("folder_key")
    filename   = data_cfg.get("filename")
    if not folder_key:
        raise ValueError("Config missing: data.folder_key")
    if not filename:
        raise ValueError("Config missing: data.filename")

    # ─────────────────────────────────────────────────────────────────────────
    # BƯỚC 1: Load data
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[STEP 1] LOAD DATA")
    print("=" * 70)

    df_raw = _load_df_via_loader(app_root, folder_key, filename)
    df_raw = _drop_known_bad_rows(df_raw, project_root)
    print(f"  Loaded {len(df_raw)} rows, {len(df_raw.columns)} columns")

    if len(df_raw) == 0:
        raise RuntimeError("After loading: no rows. Check input data.")

    # ─────────────────────────────────────────────────────────────────────────
    # BƯỚC 2: Schema validation
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[STEP 2] SCHEMA VALIDATION")
    print("=" * 70)

    skip_schema = config.get("skip_schema_validation", False)
    if skip_schema:
        df_valid = df_raw.copy()
        # Rename raw columns nếu cần
        rename_map = {
            'station_id':   'location_station_id',
            'station_name': 'location_station_name',
            'province':     'location_province',
            'district':     'location_district',
            'latitude':     'location_latitude',
            'longitude':    'location_longitude',
        }
        for old, new in rename_map.items():
            if old in df_valid.columns and new not in df_valid.columns:
                df_valid = df_valid.rename(columns={old: new})
        for drop_col in ['status']:
            if drop_col in df_valid.columns:
                df_valid = df_valid.drop(columns=[drop_col])
        for tc in ['timestamp', 'data_time']:
            if tc in df_valid.columns:
                df_valid[tc] = pd.to_datetime(df_valid[tc], errors='coerce')
                if df_valid[tc].isna().any():
                    default_ts = df_valid[tc].dropna().mode()
                    if len(default_ts) > 0:
                        df_valid[tc] = df_valid[tc].fillna(default_ts.iloc[0])
        schema_report = {
            "rows_before": len(df_raw), "rows_after": len(df_valid),
            "rows_dropped": 0, "note": "schema_validation_skipped",
        }
        print(f"  [SCHEMA] Skipped. Keeping {len(df_valid)} rows.")
    else:
        df_valid, schema_report = _validate_schema_keep_valid_rows(df_raw)
        print(f"  [SCHEMA] {schema_report['rows_before']} → {schema_report['rows_after']} rows "
              f"({schema_report['rows_dropped']} dropped)")

    if len(df_valid) == 0:
        raise RuntimeError("After schema validation: no valid rows. Check data & schema rules.")

    # ─────────────────────────────────────────────────────────────────────────
    # BƯỚC 3: Split train / valid / test
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[STEP 3] SPLIT TRAIN / VALID / TEST")
    print("=" * 70)

    test_ratio  = float(split_cfg.get("test_size",  split_cfg.get("test_ratio",  0.1)))
    val_ratio   = float(split_cfg.get("valid_size", split_cfg.get("val_ratio",   0.1)))
    train_ratio = 1.0 - test_ratio - val_ratio

    split_config = SplitConfig(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        shuffle=bool(split_cfg.get("shuffle", False)),         # time-series: NO shuffle
        sort_by_time_if_possible=bool(split_cfg.get("sort_by_time", True)),
    )

    df_train, df_valid_split, df_test = split_dataframe(df_valid, split_config)

    if "merge" in folder_key.lower():
        split_out_dir = dataset_after_split_merge
        split_name = "merge"
    else:
        split_out_dir = dataset_after_split_not_merge
        split_name = "not_merge"

    split_paths = _save_split_csvs(
        out_root=split_out_dir, split_name=split_name,
        df_train=df_train, df_valid=df_valid_split, df_test=df_test,
    )
    print(f"  Train={len(df_train)}, Valid={len(df_valid_split)}, Test={len(df_test)}")
    print(f"  Saved to {split_out_dir}")

    # ─────────────────────────────────────────────────────────────────────────
    # BƯỚC 4: Build features + clean + optional feature selection
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[STEP 4] BUILD FEATURES")
    print("=" * 70)

    # Auto-detect time-series vs cross-sectional
    auto_detect = config.get("auto_detect_data_type", False)
    detected_data_type = _detect_data_type(df_valid) if auto_detect else "unknown"
    if detected_data_type == 'cross_sectional':
        print(f"  [DATA TYPE] Cross-sectional — disabling lag/rolling/diff features")
        if feature_cfg is None:
            feature_cfg = {}
        feature_cfg['lag_features']        = False
        feature_cfg['rolling_features']    = False
        feature_cfg['difference_features'] = False

    builder = WeatherFeatureBuilder(config=feature_cfg or None)

    forecast_horizon: int = int(config.get("forecast_horizon", 0))
    leaked_columns: List[str] = []
    if forecast_horizon > 0:
        print(f"  [FORECAST] horizon = {forecast_horizon} steps ahead")
        leaked_columns = config.get("leaked_columns", [
            "rain_current", "rain_avg", "rain_max", "rain_min",
        ])

    X_train_raw, y_train = _build_features_for_split(
        builder, df_train, target_col, group_by, forecast_horizon, leaked_columns)
    X_valid_raw, y_valid = _build_features_for_split(
        builder, df_valid_split, target_col, group_by, forecast_horizon, leaked_columns)
    X_test_raw,  y_test  = _build_features_for_split(
        builder, df_test, target_col, group_by, forecast_horizon, leaked_columns)

    # Loại constant features
    X_train_raw, removed_const = _remove_constant_features(X_train_raw)
    if removed_const:
        X_valid_raw = X_valid_raw.drop(columns=removed_const, errors='ignore')
        X_test_raw  = X_test_raw.drop(columns=removed_const,  errors='ignore')

    # Loại static-derived features (lat/lon lag/rolling/diff)
    X_train_raw, removed_static = _remove_static_derived_features(X_train_raw)
    if removed_static:
        X_valid_raw = X_valid_raw.drop(columns=removed_static, errors='ignore')
        X_test_raw  = X_test_raw.drop(columns=removed_static,  errors='ignore')

    # Optional: polynomial features
    poly_cfg = config.get("polynomial_features", {})
    if poly_cfg.get("enabled", False):
        X_train_raw, poly_names = _add_polynomial_features(
            X_train_raw, y_train,
            top_k=poly_cfg.get("top_k_corr", 8),
            degree=poly_cfg.get("degree", 2),
        )
        for fname in poly_names:
            if '_sq' in fname:
                base_col = fname.replace('_sq', '')
                if base_col in X_valid_raw.columns:
                    X_valid_raw[fname] = X_valid_raw[base_col] ** 2
                    X_test_raw[fname]  = X_test_raw[base_col]  ** 2
            elif '_x_' in fname:
                parts = fname.split('_x_')
                if (len(parts) == 2
                        and parts[0] in X_valid_raw.columns
                        and parts[1] in X_valid_raw.columns):
                    X_valid_raw[fname] = X_valid_raw[parts[0]] * X_valid_raw[parts[1]]
                    X_test_raw[fname]  = X_test_raw[parts[0]]  * X_test_raw[parts[1]]
    else:
        poly_names: List[str] = []

    # Optional: feature selection via SHAP
    enable_feature_selection = config.get("feature_selection", {}).get("enabled", False)
    max_features              = config.get("feature_selection", {}).get("max_features", 0)
    if enable_feature_selection and max_features > 0 and len(X_train_raw.columns) > max_features:
        selected_features = _select_features_by_importance(
            X_train_raw.select_dtypes(include=[np.number]).fillna(0),
            y_train.fillna(0),
            max_features=max_features,
        )
        X_train_raw = X_train_raw[[c for c in selected_features if c in X_train_raw.columns]]
        X_valid_raw = X_valid_raw[[c for c in selected_features if c in X_valid_raw.columns]]
        X_test_raw  = X_test_raw[[c for c in selected_features  if c in X_test_raw.columns]]

    # Loại bỏ cột identifier / datetime
    _non_feature_cols = {
        'timestamp', 'data_time', 'data_quality',
        'location_station_id', 'location_station_name',
        'location_province', 'location_district',
    }
    created_feature_names = builder.get_feature_names()
    all_feature_columns   = [c for c in X_train_raw.columns if c not in _non_feature_cols]

    _drop_existing = [c for c in _non_feature_cols if c in X_train_raw.columns]
    if _drop_existing:
        X_train_raw = X_train_raw.drop(columns=_drop_existing)
        X_valid_raw = X_valid_raw.drop(columns=[c for c in _drop_existing if c in X_valid_raw.columns])
        X_test_raw  = X_test_raw.drop(columns=[c for c in _drop_existing  if c in X_test_raw.columns])
        print(f"  [FEATURES] Dropped non-feature columns: {_drop_existing}")

    print(f"  [FEATURES] Final count: {len(all_feature_columns)} features  "
          f"(train shape: {X_train_raw.shape})")

    # ─────────────────────────────────────────────────────────────────────────
    # BƯỚC 5: WeatherTransformPipeline (scale / encode / impute)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[STEP 5] TRANSFORM PIPELINE")
    print("=" * 70)

    pipeline = WeatherTransformPipeline(
        missing_strategy=transform_cfg.get("missing_strategy", "median"),
        scaler_type=transform_cfg.get("scaler_type", "standard"),
        encoding_type=transform_cfg.get("encoding_type", "label"),
        handle_outliers=bool(transform_cfg.get("handle_outliers", True)),
        outlier_method=transform_cfg.get("outlier_method", "iqr"),
    )

    # fit ONLY on train
    X_train_t = pipeline.fit_transform(X_train_raw, None)
    X_valid_t = pipeline.transform(X_valid_raw)
    X_test_t  = pipeline.transform(X_test_raw)

    pipeline_path = artifacts_latest / "Transform_pipeline.pkl"
    pipeline.save(pipeline_path)
    print(f"  Pipeline saved to {pipeline_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # BƯỚC 6: Feature_list.json (align columns khi predict)
    # ─────────────────────────────────────────────────────────────────────────
    zero_ratio = float((y_train == 0).mean()) if len(y_train) > 0 else 0.0

    feature_list_path = artifacts_latest / "Feature_list.json"
    _save_json(feature_list_path, {
        "created_features":            created_feature_names,
        "all_feature_columns":         all_feature_columns,
        "target_column":               target_col,
        "forecast_horizon":            forecast_horizon,
        "generated_at":                _now_tag(),
        "group_by":                    group_by,
        "removed_static_features":     removed_static,
        "removed_constant_features":   removed_const,
        "detected_data_type":          detected_data_type,
        "feature_selection_enabled":   enable_feature_selection,
        "polynomial_features_added":   poly_cfg.get("enabled", False),
        "model_type":                  "stacking_ensemble",
        "note": (
            "all_feature_columns is the list of X columns after build + "
            "feature selection (for prediction alignment)."
        ),
    })
    print(f"  Feature_list.json saved | {len(all_feature_columns)} features")

    # ─────────────────────────────────────────────────────────────────────────
    # BƯỚC 7: TRAIN StackingEnsemble (Stages 6–9)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[STEP 7] TRAIN StackingEnsemble  (Stages 6 → 7 → 8 → 9)")
    print("=" * 70)

    stacking = StackingEnsemble(
        n_splits=int(stacking_cfg.get("n_splits", 5)),
        predict_threshold=float(stacking_cfg.get("predict_threshold", 0.4)),
        rain_threshold=float(stacking_cfg.get("rain_threshold", RAIN_THRESHOLD)),
        seed=int(stacking_cfg.get("seed", 42)),
        cls_params=stacking_cfg.get("cls_params"),     # None → dùng default
        reg_params=stacking_cfg.get("reg_params"),
        meta_cls_params=stacking_cfg.get("meta_cls_params"),
        meta_reg_params=stacking_cfg.get("meta_reg_params"),
        verbose=bool(stacking_cfg.get("verbose", True)),
    )

    stacking_result = stacking.fit(
        X_train_t, y_train.values,
        X_val=X_valid_t, y_val=y_valid.values,
    )

    if not stacking_result.success:
        raise RuntimeError(f"StackingEnsemble.fit() failed: {stacking_result.message}")

    print(f"\n{stacking.summary()}")

    # ─────────────────────────────────────────────────────────────────────────
    # BƯỚC 8: Evaluate — 3 nhóm metrics cho mỗi split
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[STEP 8] EVALUATE")
    print("=" * 70)

    def _eval_split(X, y_true_orig, split_name: str) -> Dict[str, Any]:
        """
        StackingEnsemble.evaluate() trả về dict gồm:
            classification: {precision, recall, f1, roc_auc, pr_auc, ...}
            regression_rainy: {mae, rmse, r2, ...}  (chỉ trên mẫu rainy)
            end_to_end: {mae, rmse, r2, ...}         (toàn bộ dự đoán mm)
        Ta flatten thành 1 dict để lưu cùng cấu trúc với train_ensemble_average.
        """
        raw = stacking.evaluate(X, y_true_orig, dataset_name=split_name)
        flat: Dict[str, Any] = {}

        # classification prefix: cls_*
        for k, v in raw.get("classification", {}).items():
            flat[f"cls_{k}"] = v

        # regression rainy prefix: reg_rainy_*
        for k, v in raw.get("regression_rainy", {}).items():
            flat[f"reg_rainy_{k}"] = v

        # end-to-end prefix: e2e_*
        for k, v in raw.get("end_to_end", {}).items():
            flat[f"e2e_{k}"] = v

        # Thêm Rain_Detection_Accuracy (end-to-end classify has_rain)
        y_pred_mm  = stacking.predict(X)
        y_arr      = np.asarray(y_true_orig)
        pred_rain  = (np.asarray(y_pred_mm) > RAIN_THRESHOLD).astype(int)
        act_rain   = (y_arr > RAIN_THRESHOLD).astype(int)
        flat["Rain_Detection_Accuracy"] = float((pred_rain == act_rain).mean())

        # Thêm NonZero metrics (rainy samples end-to-end)
        nz_mask = y_arr > 0
        if nz_mask.sum() > 10:
            flat["NonZero_MAE"]   = float(np.mean(np.abs(y_arr[nz_mask] - np.asarray(y_pred_mm)[nz_mask])))
            flat["NonZero_RMSE"]  = float(np.sqrt(np.mean((y_arr[nz_mask] - np.asarray(y_pred_mm)[nz_mask]) ** 2)))
            flat["NonZero_count"] = int(nz_mask.sum())

        # Cho phép calculate_all_metrics bổ sung thêm metrics chuẩn
        try:
            extra = calculate_all_metrics(
                y_arr, np.asarray(y_pred_mm),
                n_features=X.shape[1] if hasattr(X, "shape") else None,
                include_weather_metrics=True,
            )
            for k, v in extra.items():
                if k not in flat:
                    flat[k] = v
        except Exception:
            pass

        return flat

    metrics: Dict[str, Any] = {
        "generated_at":   _now_tag(),
        "model_type":     "stacking_ensemble",
        "training_time_seconds": round(stacking_result.training_time, 2),
        "n_cls_oof_samples": stacking_result.n_cls_oof_samples,
        "n_reg_oof_samples": stacking_result.n_reg_oof_samples,
    }

    metrics["train"] = _eval_split(X_train_t, y_train.values, "train")
    metrics["valid"] = _eval_split(X_valid_t, y_valid.values, "valid")
    metrics["test"]  = _eval_split(X_test_t,  y_test.values,  "test")

    # stage_metrics từ OOF (giai đoạn 6-9)
    metrics["stage_metrics"] = stacking_result.stage_metrics

    # ─────────────────────────────────────────────────────────────────────────
    # BƯỚC 9: Overfit/Underfit + Model quality diagnostics
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[STEP 9] DIAGNOSTICS")
    print("=" * 70)

    overfit_status, overfit_details = _detect_overfit_underfit(metrics)

    # Model quality dựa trên cls_f1 (phân loại mưa là việc chính)
    f1_vals = [
        metrics.get(s, {}).get("cls_f1")
        for s in ("train", "valid", "test")
    ]
    f1_vals = [v for v in f1_vals if v is not None]
    avg_f1  = sum(f1_vals) / len(f1_vals) if f1_vals else 0.0

    if avg_f1 >= 0.70:
        model_quality  = "excellent"
    elif avg_f1 >= 0.45:
        model_quality  = "good"
    elif avg_f1 >= 0.25:
        model_quality  = "fair"
    else:
        model_quality  = "poor"
    quality_detail = f"Rain cls_f1 avg={avg_f1:.3f}"

    metrics["diagnostics"] = {
        "overfit_status":               overfit_status,
        "overfit_details":              overfit_details,
        "model_quality":                model_quality,
        "quality_detail":               quality_detail,
        "target_zero_ratio":            zero_ratio,
        "detected_data_type":           detected_data_type,
        "n_features_after_selection":   len(all_feature_columns),
        "n_static_features_removed":    len(removed_static),
        "n_constant_features_removed":  len(removed_const),
        "polynomial_features_added":    poly_cfg.get("enabled", False),
        "predict_threshold":            stacking.predict_threshold,
        "n_splits":                     stacking.n_splits,
    }

    print(f"  Overfit status : {overfit_status}")
    print(f"  Details        : {overfit_details}")
    print(f"  Model quality  : {model_quality} ({quality_detail})")

    _save_json(artifacts_latest / "Metrics.json", metrics)
    print(f"  Metrics.json saved")

    # ─────────────────────────────────────────────────────────────────────────
    # BƯỚC 10: Save model artifact
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[STEP 10] SAVE ARTIFACTS")
    print("=" * 70)

    # StackingEnsemble.save() ghi vào ml_models/ với timestamp trong tên file
    # Ta cũng copy (symlink) sang artifacts_latest/Model.pkl để predict pipeline
    # dùng được cùng cấu trúc với các model khác.
    model_saved_path = stacking.save()                       # → ml_models/stacking_ensemble_<ts>.joblib
    model_artifact_path = artifacts_latest / "Model.pkl"
    import joblib
    joblib.dump(stacking, model_artifact_path)               # copy vào latest/

    print(f"  Stacking model : {model_saved_path}")
    print(f"  Model.pkl (latest): {model_artifact_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # BƯỚC 11: Schema-Aware Model Bank
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[STEP 11] SCHEMA-AWARE MODEL BANK")
    print("=" * 70)

    schema_bank_dir  = artifacts_latest / "schema_model_bank"
    schema_bank_path: Optional[str] = None
    schema_bank_routing: Optional[dict] = None

    try:
        from Weather_Forcast_App.Machine_learning_model.Models.Schema_Selector import (
            FoldSchemaModelBank,
        )

        bank = FoldSchemaModelBank.from_stacking(
            stacking         = stacking,
            y_train_mm       = y_train.values,
            feature_names    = all_feature_columns,
            predict_threshold = float(stacking_cfg.get("predict_threshold", stacking.predict_threshold)),
            rain_threshold   = float(stacking_cfg.get("rain_threshold", stacking.rain_threshold)),
            verbose          = True,
        )
        bank.analyze()

        # Cập nhật season centroids từ X_train nếu có month_sin/month_cos
        try:
            bank.update_season_centroids(X_train_t, all_feature_columns)
        except Exception as _ce:
            print(f"  [schema bank] update_season_centroids skipped: {_ce}")

        bank.save(schema_bank_dir)
        schema_bank_path    = str(schema_bank_dir)
        schema_bank_routing = bank.get_routing_config()
        print(bank.summary())
        print(f"  Schema bank saved → {schema_bank_dir}")

    except Exception as _e:
        print(f"  ⚠ Schema bank failed (non-critical): {_e}")

    # ─────────────────────────────────────────────────────────────────────────
    # BƯỚC 12: Train_info.json
    # ─────────────────────────────────────────────────────────────────────────
    train_info = {
        "trained_at":    _now_tag(),
        "model_type":    "stacking_ensemble",
        "project_root":  str(project_root),
        "ml_model_root": str(ml_model_root),
        "input": {
            "folder_key": folder_key,
            "filename":   filename,
        },
        "schema_report":   schema_report,
        "split_config":    asdict(split_config),
        "split_saved_paths": split_paths,
        "target_column":   target_col,
        "group_by":        group_by,
        "forecast_horizon": forecast_horizon,
        "leaked_columns_removed": leaked_columns if forecast_horizon > 0 else [],
        "feature_info": {
            "n_created_features":         int(len(created_feature_names)),
            "n_total_feature_columns":    int(len(all_feature_columns)),
            "n_static_features_removed":  int(len(removed_static)),
            "n_constant_features_removed": int(len(removed_const)),
            "feature_selection_enabled":  enable_feature_selection,
            "detected_data_type":         detected_data_type,
            "polynomial_features_added":  poly_cfg.get("enabled", False),
        },
        "target_info": {
            "zero_ratio":     zero_ratio,
            "rain_threshold": float(RAIN_THRESHOLD),
            "note": "StackingEnsemble handles log1p internally. No external log transform applied.",
        },
        "transform": {
            "pipeline_path": str(pipeline_path),
            "pipeline_info": pipeline.get_pipeline_info(),
        },
        "stacking_config": {
            "n_splits":           stacking.n_splits,
            "predict_threshold":  stacking.predict_threshold,
            "rain_threshold":     stacking.rain_threshold,
            "seed":               stacking.seed,
            "n_cls_oof_samples":  stacking_result.n_cls_oof_samples,
            "n_reg_oof_samples":  stacking_result.n_reg_oof_samples,
            "cls_model_names":    stacking.cls_model_names,
            "reg_model_names":    stacking.reg_model_names,
        },
        "artifacts": {
            "model_joblib":      str(model_saved_path),
            "model_pkl":         str(model_artifact_path),
            "pipeline":          str(pipeline_path),
            "feature_list":      str(feature_list_path),
            "metrics":           str(artifacts_latest / "Metrics.json"),
            "train_info":        str(artifacts_latest / "Train_info.json"),
            "schema_model_bank": schema_bank_path,
        },
        "schema_bank_routing":   schema_bank_routing,
        "diagnostics":           metrics["diagnostics"],
        "feature_builder_config": feature_cfg,
    }

    _save_json(artifacts_latest / "Train_info.json", train_info)
    print(f"  Train_info.json saved")

    return train_info


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Weather Forecast — Train StackingEnsemble (Super Learner)"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to train config (.json/.yml)",
    )
    args  = parser.parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    config   = _load_config(cfg_path)

    info = run_stacking_training(config)

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[OK] STACKING ENSEMBLE TRAINING DONE")
    artifacts = info.get("artifacts", {})
    print(f"  Model (joblib) : {artifacts.get('model_joblib', 'N/A')}")
    print(f"  Model.pkl      : {artifacts.get('model_pkl', 'N/A')}")
    print(f"  Pipeline       : {artifacts.get('pipeline', 'N/A')}")
    print(f"  Metrics        : {artifacts.get('metrics', 'N/A')}")
    print(f"  Train info     : {artifacts.get('train_info', 'N/A')}")

    diag = info.get("diagnostics", {})
    print("-" * 70)
    print(f"  Overfit status : {diag.get('overfit_status', 'N/A')}")
    try:
        print(f"  Details        : {diag.get('overfit_details', '')}")
    except UnicodeEncodeError:
        details = (diag.get("overfit_details", "") or "").encode("ascii", "replace").decode()
        print(f"  Details        : {details}")
    print(f"  Model quality  : {diag.get('model_quality', 'N/A')} — {diag.get('quality_detail', '')}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
