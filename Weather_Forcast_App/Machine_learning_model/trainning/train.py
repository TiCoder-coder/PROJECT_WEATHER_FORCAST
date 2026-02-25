

from __future__ import annotations
import sys
from pathlib import Path
# /media/voanhnhat/SDD_OUTSIDE5/PROJECT_WEATHER_FORECAST/Weather_Forcast_App/Machine_learning_model/trainning/train.py
# ----------------------------- TRAIN "TỔNG CHỈ HUY" -----------------------------------------------------------
"""
train.py - Tổng chỉ huy quá trình training

Flow chuẩn:
    1) Đọc config (json/yaml)
    2) Load data (Loader.py)
    3) Validate schema (Schema.py)
    4) Split train/valid/test (Split.py) + lưu ra Dataset_after_split/...
    5) Build features (Build_transfer.py)
    6) Transform pipeline thống nhất train/predict (Transformers.py)
    7) Train model (RandomForest/XGBoost/LightGBM/CatBoost wrappers)
    8) Evaluate metrics
    9) Save artifacts:
        - Model.pkl
        - Transform_pipeline.pkl
        - Feature_list.json
        - Metrics.json
        - Train_info.json

Chạy gợi ý:
    python -m Weather_Forcast_App.Machine_learning_model.trainning.train --config config/train_config.json

Hoặc chạy trực tiếp:
    python /.../Weather_Forcast_App/Machine_learning_model/trainning/train.py --config config/train_config.json
"""
# ======================================================================================
# (1) FIX IMPORT PATH: chạy trực tiếp file vẫn import được Weather_Forcast_App.*
# ======================================================================================
THIS_FILE = Path(__file__).resolve()
# .../Weather_Forcast_App/Machine_learning_model/trainning/train.py
# project_root = .../PROJECT_WEATHER_FORECAST
project_root = THIS_FILE
# đi lên tới thư mục chứa Weather_Forcast_App
for _ in range(5):
    if (project_root / "Weather_Forcast_App").exists():
        break
    project_root = project_root.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import _load_df_via_loader (after sys.path is set)
from Weather_Forcast_App.Machine_learning_model.trainning.tuning import _load_df_via_loader

import argparse
import json
import sys
import importlib

# Fix: Import _load_df_via_loader from tuning.py
import joblib
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import numpy as np

# ======================================================================================
# (1) FIX IMPORT PATH: chạy trực tiếp file vẫn import được Weather_Forcast_App.*
# ======================================================================================
THIS_FILE = Path(__file__).resolve()
# .../Weather_Forcast_App/Machine_learning_model/trainning/train.py
# project_root = .../PROJECT_WEATHER_FORECAST
project_root = THIS_FILE
# đi lên tới thư mục chứa Weather_Forcast_App
for _ in range(5):
    if (project_root / "Weather_Forcast_App").exists():
        break
    project_root = project_root.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ======================================================================================
# (2) IMPORT CÁC MODULE BẠN ĐÃ CÓ
# ======================================================================================
from Weather_Forcast_App.Machine_learning_model.data.Loader import DataLoader
from Weather_Forcast_App.Machine_learning_model.data.Schema import validate_weather_dataframe
from Weather_Forcast_App.Machine_learning_model.data.Split import SplitConfig, split_dataframe

from Weather_Forcast_App.Machine_learning_model.features.Build_transfer import WeatherFeatureBuilder
from Weather_Forcast_App.Machine_learning_model.features.Transformers import WeatherTransformPipeline

# Evaluation metrics - sử dụng module đã có thay vì import trực tiếp từ sklearn
from Weather_Forcast_App.Machine_learning_model.evaluation.metrics import calculate_all_metrics

# Model wrappers - sử dụng dict để giảm code
MODEL_REGISTRY = {
    "rf": "Weather_Forcast_App.Machine_learning_model.Models.Random_Forest_Model.WeatherRandomForest",
    "random_forest": "Weather_Forcast_App.Machine_learning_model.Models.Random_Forest_Model.WeatherRandomForest",
    "randomforest": "Weather_Forcast_App.Machine_learning_model.Models.Random_Forest_Model.WeatherRandomForest",
    "xgb": "Weather_Forcast_App.Machine_learning_model.Models.XGBoost_Model.WeatherXGBoost",
    "xgboost": "Weather_Forcast_App.Machine_learning_model.Models.XGBoost_Model.WeatherXGBoost",
    "lgbm": "Weather_Forcast_App.Machine_learning_model.Models.LightGBM_Model.WeatherLightGBM",
    "lightgbm": "Weather_Forcast_App.Machine_learning_model.Models.LightGBM_Model.WeatherLightGBM",
    "cat": "Weather_Forcast_App.Machine_learning_model.Models.CatBoost_Model.WeatherCatBoost",
    "catboost": "Weather_Forcast_App.Machine_learning_model.Models.CatBoost_Model.WeatherCatBoost",
}


# ======================================================================================
# (3) TIỆN ÍCH: load config
# ======================================================================================
def _load_config(path: Path) -> Dict[str, Any]:
    """
    Load config từ .json hoặc .yaml/.yml.
    - Nếu bạn chưa dùng yaml thì cứ xài json là OK.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    ext = path.suffix.lower()
    if ext == ".json":
        return json.loads(path.read_text(encoding="utf-8"))

    if ext in [".yml", ".yaml"]:
        # Không ép bạn cài pyyaml, nhưng nếu có thì đọc được
        try:
            import yaml  # type: ignore
            return yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as e:
            raise RuntimeError("Config is YAML but PyYAML not installed. Install pyyaml or use JSON.") from e

    raise ValueError(f"Unsupported config extension: {ext}")

def _now_tag() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ======================================================================================
# (4) SAVE HELPERS
# ======================================================================================
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    import enum
    def default(obj):
        if isinstance(obj, enum.Enum):
            return obj.value
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        try:
            from dataclasses import asdict as dc_asdict
            if hasattr(obj, "__dataclass_fields__"):
                return dc_asdict(obj)
        except Exception:
            pass
        return str(obj)
    _ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=default), encoding="utf-8")


def _save_split_csvs(
    out_root: Path,
    split_name: str,
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
) -> Dict[str, str]:
    """
    Lưu 3 file CSV train/valid/test vào Dataset_after_split/<split_name>/
    """
    _ensure_dir(out_root)
    train_path = out_root / f"{split_name}_train.csv"
    valid_path = out_root / f"{split_name}_valid.csv"
    test_path = out_root / f"{split_name}_test.csv"

    df_train.to_csv(train_path, index=False, encoding="utf-8-sig")
    df_valid.to_csv(valid_path, index=False, encoding="utf-8-sig")
    df_test.to_csv(test_path, index=False, encoding="utf-8-sig")

    return {
        "train": str(train_path),
        "valid": str(valid_path),
        "test": str(test_path),
    }


# ======================================================================================
# (5) VALIDATE SCHEMA
# ======================================================================================
def _validate_schema_keep_valid_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Dùng validate_weather_dataframe(df) của Schema.py.
    - Hàm này trả về list WeatherDataSchema hợp lệ
    - Ta rebuild lại DataFrame "valid_df" từ list đó để đảm bảo schema sạch.

    Trả về:
        valid_df, report
    """
    total_before = len(df)

    # validate_weather_dataframe trả về list schema objects hợp lệ
    schemas = validate_weather_dataframe(df)

    # build lại df từ schema
    valid_records = [s.to_flat_dict() for s in schemas]
    valid_df = pd.DataFrame(valid_records)

    report = {
        "rows_before": int(total_before),
        "rows_after": int(len(valid_df)),
        "rows_dropped": int(total_before - len(valid_df)),
    }
    return valid_df, report


# ======================================================================================
# (6) BUILD FEATURES + TRANSFORM
# ======================================================================================

# --- Các cột static KHÔNG nên tạo lag/rolling/diff (chỉ tạo noise) ---
_STATIC_COL_KEYWORDS = [
    'location_vi_do', 'location_kinh_do', 'location_ma_tram',
    'location_tinh_thanh_pho', 'location_huyen',
    'vi_do', 'kinh_do', 'latitude', 'longitude',
]


def _is_static_derived_feature(col_name: str) -> bool:
    """Kiểm tra xem feature có phải là lag/rolling/diff trên cột static không."""
    col_lower = col_name.lower()
    temporal_suffixes = ['_lag_', '_rolling_', '_diff_', '_pct_change_']
    for kw in _STATIC_COL_KEYWORDS:
        if kw in col_lower and any(suf in col_lower for suf in temporal_suffixes):
            return True
    return False


def _remove_static_derived_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Loại bỏ features tạo từ cột static (lat/lon lag, rolling, diff).
    Các features này chỉ tạo noise vì lat/lon không đổi theo thời gian.
    
    Returns:
        (X_cleaned, removed_columns)
    """
    remove_cols = [col for col in X.columns if _is_static_derived_feature(col)]
    if remove_cols:
        print(f"  [FEATURE CLEAN] Removed {len(remove_cols)} static-derived features (lat/lon lag/rolling/diff)")
    return X.drop(columns=remove_cols, errors='ignore'), remove_cols


def _select_features_by_importance(
    X_train: pd.DataFrame, y_train: pd.Series,
    max_features: int = 150,
    min_importance: float = 0.0,
) -> List[str]:
    """
    Dùng LightGBM nhanh để chọn top features theo importance.
    Giúp giảm chiều dữ liệu, tránh underfitting do nhiều features noise.
    
    Returns:
        list tên features đã chọn (sorted by importance desc)
    """
    try:
        from lightgbm import LGBMRegressor
        selector = LGBMRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            verbose=-1, random_state=42, n_jobs=-1,
        )
        selector.fit(X_train, y_train)
        importances = selector.feature_importances_
        feature_imp = sorted(
            zip(X_train.columns, importances),
            key=lambda x: x[1], reverse=True
        )
        # Chọn top max_features, bỏ qua min_importance nếu nó loại hết
        selected = [name for name, imp in feature_imp if imp > min_importance][:max_features]
        # Fallback: nếu quá ít features được chọn, lấy top max_features bất kể importance
        if len(selected) < min(20, len(X_train.columns)):
            selected = [name for name, _ in feature_imp[:max_features]]
        print(f"  [FEATURE SELECT] Selected {len(selected)}/{len(X_train.columns)} features (top importance)")
        return selected
    except Exception as e:
        print(f"  [FEATURE SELECT] Cannot run feature selection: {e}")
        return X_train.columns.tolist()


def _build_features_for_split(
    builder: WeatherFeatureBuilder,
    df: pd.DataFrame,
    target_col: str,
    group_by: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build features từ raw df.
    - Giữ target y riêng.
    - Trả về: X_df, y_series
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe columns.")

    # Build features (builder tự sort/time/lag/rolling/...)
    df_feat = builder.build_all_features(df, target_column=target_col, group_by=group_by)

    # tách y ra khỏi X
    y = df_feat[target_col].copy()
    X = df_feat.drop(columns=[target_col])

    return X, y


# ======================================================================================
# (7) MODEL FACTORY - Đơn giản hóa bằng dynamic import
# ======================================================================================
def _create_model(model_type: str, model_config: Dict[str, Any]):
    """
    Tạo instance model wrapper theo config.
    Sử dụng dynamic import để giảm số dòng code.
    """
    model_type = (model_type or "").lower().strip()
    if model_type == "ensemble":
        base_models = model_config.get("base_models", [])
        from Weather_Forcast_App.Machine_learning_model.Models.Ensemble_Model import WeatherEnsembleModel
        return WeatherEnsembleModel(base_models=base_models, model_registry=MODEL_REGISTRY)
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model_type='{model_type}'. Use: {list(set(MODEL_REGISTRY.keys()))}")
    module_path, class_name = MODEL_REGISTRY[model_type].rsplit(".", 1)
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    return model_class(**model_config)


# ======================================================================================
# (8) MAIN TRAIN FLOW
# ======================================================================================
def run_training(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hàm chạy training, trả về Train_info dict (để bạn log/print/ghi file).
    """

    # --------------------------
    # Resolve paths
    # --------------------------
    # Root: .../Weather_Forcast_App/Machine_learning_model/trainning/train.py
    ml_model_root = THIS_FILE.parents[1]  # Machine_learning_model
    app_root = THIS_FILE.parents[2]       # Weather_Forcast_App

    dataset_after_split_root = ml_model_root / "Dataset_after_split"
    dataset_after_split_merge = dataset_after_split_root / "Dataset_merge"
    dataset_after_split_not_merge = dataset_after_split_root / "Dataset_not_merge"

    artifacts_latest = app_root / "Machine_learning_artifacts" / "latest"
    _ensure_dir(artifacts_latest)

    # --------------------------
    # (1) Read config
    # --------------------------
    data_cfg = config.get("data") or {}
    
    feature_cfg = config.get("features", {})
    split_cfg = config.get("split", {})
    model_cfg = config.get("model", {})
    transform_cfg = config.get("transform", {})

    target_col = config.get("target_column", "luong_mua_hien_tai")
    group_by = config.get("group_by")  # ví dụ: "location_ma_tram" hoặc None

    # --------------------------
    # (2) Load data (Loader.py)
    # --------------------------
    folder_key = data_cfg.get("folder_key")
    filename = data_cfg.get("filename")

    if not folder_key:
        raise ValueError("Config missing: data.folder_key")
    if not filename:
        raise ValueError("Config missing: data.filename")

    # --------------------------
    # (3) Validate schema (Schema.py)
    # --------------------------
    df_raw = _load_df_via_loader(app_root, folder_key, filename)
    file_info = None  # tuning._load_df_via_loader does not return file_info

    if len(df_raw) == 0:
        raise RuntimeError("After loading data: no rows left. Check input data.")

    skip_schema = config.get("skip_schema_validation", False)
    if skip_schema:
        # Skip strict schema validation — use raw data with basic cleanup
        df_valid = df_raw.copy()
        # Rename raw columns to expected names if needed
        rename_map = {
            'station_id': 'location_station_id',
            'station_name': 'location_station_name',
            'province': 'location_province',
            'district': 'location_district',
            'latitude': 'location_latitude',
            'longitude': 'location_longitude',
        }
        for old, new in rename_map.items():
            if old in df_valid.columns and new not in df_valid.columns:
                df_valid = df_valid.rename(columns={old: new})
        # Drop columns not needed for ML
        for drop_col in ['status']:
            if drop_col in df_valid.columns:
                df_valid = df_valid.drop(columns=[drop_col])
        # Fill NaN timestamps with a default (to allow time feature building)
        for tc in ['timestamp', 'data_time']:
            if tc in df_valid.columns:
                df_valid[tc] = pd.to_datetime(df_valid[tc], errors='coerce')
                if df_valid[tc].isna().any():
                    default_ts = df_valid[tc].dropna().mode()
                    if len(default_ts) > 0:
                        df_valid[tc] = df_valid[tc].fillna(default_ts.iloc[0])
        schema_report = {"rows_before": len(df_raw), "rows_after": len(df_valid),
                         "rows_dropped": 0, "note": "schema_validation_skipped"}
        print(f"  [SCHEMA] Skipped strict validation. Keeping {len(df_valid)} rows (was {len(df_raw)}).")
        # Warn if features lack variation
        num_cols = df_valid.select_dtypes(include='number').columns
        n_const = sum(1 for c in num_cols if df_valid[c].nunique() <= 1)
        if n_const > len(num_cols) * 0.5:
            print(f"  [WARNING] {n_const}/{len(num_cols)} numeric columns are constant! Data may be a single-snapshot.")
    else:
        df_valid, schema_report = _validate_schema_keep_valid_rows(df_raw)

    if len(df_valid) == 0:
        raise RuntimeError("After schema validation: no valid rows left. Check input data & schema rules.")

    # --------------------------
    # (4) Split train/valid/test (Split.py) + save to Dataset_after_split/...
    # --------------------------
    # Chuyển đổi config sang format của SplitConfig (Split.py)
    test_ratio = float(split_cfg.get("test_size", split_cfg.get("test_ratio", 0.1)))
    val_ratio = float(split_cfg.get("valid_size", split_cfg.get("val_ratio", 0.1)))
    train_ratio = 1.0 - test_ratio - val_ratio
    
    split_config = SplitConfig(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        shuffle=bool(split_cfg.get("shuffle", False)),  # Time series: KHÔNG shuffle
        sort_by_time_if_possible=bool(split_cfg.get("sort_by_time", True)),
    )

    df_train, df_valid_split, df_test = split_dataframe(df_valid, split_config)

    # Xác định lưu vào Dataset_merge hay Dataset_not_merge theo folder_key
    if "merge" in folder_key.lower():
        split_out_dir = dataset_after_split_merge
        split_name = "merge"
    else:
        split_out_dir = dataset_after_split_not_merge
        split_name = "not_merge"

    split_paths = _save_split_csvs(
        out_root=split_out_dir,
        split_name=split_name,
        df_train=df_train,
        df_valid=df_valid_split,
        df_test=df_test,
    )

    # --------------------------
    # (5) Build features (Build_transfer.py)
    # --------------------------
    builder = WeatherFeatureBuilder(config=feature_cfg or None)

    X_train_raw, y_train = _build_features_for_split(builder, df_train, target_col=target_col, group_by=group_by)
    X_valid_raw, y_valid = _build_features_for_split(builder, df_valid_split, target_col=target_col, group_by=group_by)
    X_test_raw, y_test = _build_features_for_split(builder, df_test, target_col=target_col, group_by=group_by)

    # --- ANTI-UNDERFIT: Loại bỏ features noise từ cột static (lat/lon lag/rolling/diff) ---
    X_train_raw, removed_static = _remove_static_derived_features(X_train_raw)
    if removed_static:
        X_valid_raw = X_valid_raw.drop(columns=removed_static, errors='ignore')
        X_test_raw = X_test_raw.drop(columns=removed_static, errors='ignore')

    # --- ANTI-UNDERFIT: Feature selection bằng importance (giảm noise) ---
    enable_feature_selection = config.get("feature_selection", {}).get("enabled", True)
    max_features = config.get("feature_selection", {}).get("max_features", 150)
    
    if enable_feature_selection and len(X_train_raw.columns) > max_features:
        # Tạm dùng y_train gốc để select features (chưa transform target)
        selected_features = _select_features_by_importance(
            X_train_raw.select_dtypes(include=[np.number]).fillna(0),
            y_train.fillna(0),
            max_features=max_features,
        )
        X_train_raw = X_train_raw[[c for c in selected_features if c in X_train_raw.columns]]
        X_valid_raw = X_valid_raw[[c for c in selected_features if c in X_valid_raw.columns]]
        X_test_raw = X_test_raw[[c for c in selected_features if c in X_test_raw.columns]]

    # feature list: list các feature mới + toàn bộ cột output (sau build + selection)
    created_feature_names = builder.get_feature_names()
    all_feature_columns = X_train_raw.columns.tolist()

    # Save Feature_list.json (để predict giữ đúng columns)
    feature_list_path = artifacts_latest / "Feature_list.json"
    _save_json(feature_list_path, {
        "created_features": created_feature_names,
        "all_feature_columns": all_feature_columns,
        "target_column": target_col,
        "generated_at": _now_tag(),
        "group_by": group_by,
        "removed_static_features": removed_static,
        "feature_selection_enabled": enable_feature_selection,
        "note": "all_feature_columns là danh sách cột X sau build + feature selection (để predict align đúng cột)."
    })

    # --------------------------
    # (5b) ANTI-UNDERFIT: Log1p target transformation cho zero-inflated targets
    # --------------------------
    # Rainfall data thường là zero-inflated (rất nhiều giá trị 0 hoặc gần 0).
    # Log1p giúp model học tốt hơn vì nén phạm vi giá trị lớn.
    use_log_target = config.get("transform_target", {}).get("log1p", True)
    target_is_rain = any(kw in target_col.lower() for kw in ['mua', 'rain', 'precipitation'])
    
    # Chỉ tự động bật log1p nếu target liên quan đến mưa VÀ có nhiều giá trị 0
    zero_ratio = (y_train == 0).mean() if len(y_train) > 0 else 0
    if use_log_target and target_is_rain and zero_ratio > 0.3:
        print(f"  [TARGET TRANSFORM] Applied log1p for '{target_col}' (zero_ratio={zero_ratio:.2%})")
        y_train_model = np.log1p(y_train.clip(lower=0))
        y_valid_model = np.log1p(y_valid.clip(lower=0))
        y_test_model = np.log1p(y_test.clip(lower=0))
        applied_log_target = True
    else:
        y_train_model = y_train
        y_valid_model = y_valid
        y_test_model = y_test
        applied_log_target = False

    # --------------------------
    # (6) Transform pipeline thống nhất train/predict (Transformers.py)
    # --------------------------
    pipeline = WeatherTransformPipeline(
        missing_strategy=transform_cfg.get("missing_strategy", "median"),
        scaler_type=transform_cfg.get("scaler_type", "standard"),
        encoding_type=transform_cfg.get("encoding_type", "label"),
        handle_outliers=bool(transform_cfg.get("handle_outliers", True)),
        outlier_method=transform_cfg.get("outlier_method", "iqr"),
    )

    # Fit ONLY on train, rồi transform valid/test
    X_train = pipeline.fit_transform(X_train_raw, y_train_model if transform_cfg.get("pass_y_to_transform", False) else None)
    X_valid_t = pipeline.transform(X_valid_raw)
    X_test_t = pipeline.transform(X_test_raw)

    # Save pipeline
    pipeline_path = artifacts_latest / "Transform_pipeline.pkl"
    pipeline.save(pipeline_path)

    # --------------------------
    # (7) Train model (dùng y đã log1p nếu áp dụng)
    # --------------------------
    model_type = model_cfg.get("type", "random_forest")
    model_params = model_cfg.get("params", {})

    model = _create_model(model_type=model_type, model_config=model_params)

    # --- ANTI-UNDERFIT: Tạo sample_weight cho zero-inflated target ---
    # Upweight non-zero samples để model chú ý hơn vào rain events
    sample_weight = None
    if applied_log_target and zero_ratio > 0.5:
        # Non-zero samples nhận weight cao hơn tỉ lệ nghịch với tần suất
        weight_ratio = min(zero_ratio / (1 - zero_ratio + 1e-8), 10.0)  # cap tại 10x
        sample_weight = np.where(y_train > 0, weight_ratio, 1.0)
        print(f"  [SAMPLE WEIGHT] Applied sample_weight: non-zero={weight_ratio:.2f}x, zero=1.0x")

    # Wrapper thường có model.train(X, y, ...) - truyền X_val, y_val cho early stopping
    if hasattr(model, "train"):
        train_kwargs = {}
        # Truyền validation data cho early stopping (nếu wrapper hỗ trợ)
        train_kwargs["X_val"] = X_valid_t
        train_kwargs["y_val"] = y_valid_model
        train_kwargs["val_size"] = 0  # Đã có val riêng, không cần split thêm
        # Truyền sample_weight nếu model wrapper hỗ trợ
        if sample_weight is not None:
            try:
                import inspect
                sig = inspect.signature(model.train)
                if "sample_weight" in sig.parameters:
                    train_kwargs["sample_weight"] = sample_weight
            except Exception:
                pass
        try:
            model.train(X_train, y_train_model, **train_kwargs)
        except TypeError:
            # Fallback nếu wrapper không hỗ trợ kwargs đó
            model.train(X_train, y_train_model)
    elif hasattr(model, "fit"):
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        try:
            model.fit(X_train, y_train_model, **fit_kwargs)
        except TypeError:
            model.fit(X_train, y_train_model)
    else:
        raise RuntimeError(f"Model wrapper '{type(model).__name__}' has no train() or fit().")

    # --------------------------
    # (8) Evaluate metrics - Sử dụng module metrics.py
    # --------------------------
    metrics: Dict[str, Any] = {
        "generated_at": _now_tag(),
        "model_type": model_type,
        "applied_log_target": applied_log_target,
    }

    def _evaluate_set(X, y_original, y_transformed):
        """
        Helper để evaluate một dataset.
        - Nếu dùng log1p target: predict rồi expm1 trước khi so sánh với y_original.
        - Luôn đánh giá trên scale gốc để metrics có ý nghĩa thực tế.
        """
        y_pred_raw = model.predict(X)
        if hasattr(y_pred_raw, "predictions"):
            y_pred_raw = y_pred_raw.predictions
        y_pred_raw = np.array(y_pred_raw)
        
        # Inverse transform nếu dùng log1p
        if applied_log_target:
            y_pred = np.expm1(y_pred_raw).clip(min=0)
        else:
            y_pred = y_pred_raw
        
        y_actual = np.array(y_original)
        result = calculate_all_metrics(y_actual, y_pred, n_features=X.shape[1])
        
        # Thêm metrics riêng cho non-zero values (quan trọng cho rain prediction)
        non_zero_mask = y_actual > 0
        if non_zero_mask.sum() > 10:
            result["NonZero_MAE"] = float(np.mean(np.abs(y_actual[non_zero_mask] - y_pred[non_zero_mask])))
            result["NonZero_RMSE"] = float(np.sqrt(np.mean((y_actual[non_zero_mask] - y_pred[non_zero_mask])**2)))
            result["NonZero_count"] = int(non_zero_mask.sum())
        
        # Thêm rain detection accuracy (classify: has rain or not)
        pred_has_rain = (y_pred > 0.1).astype(int)
        actual_has_rain = (y_actual > 0.1).astype(int)
        if len(y_actual) > 0:
            result["Rain_Detection_Accuracy"] = float((pred_has_rain == actual_has_rain).mean())
        
        return result

    metrics["train"] = _evaluate_set(X_train, y_train, y_train_model)
    metrics["valid"] = _evaluate_set(X_valid_t, y_valid, y_valid_model)
    metrics["test"] = _evaluate_set(X_test_t, y_test, y_test_model)


    # --- Overfit/Underfit & Accuracy Diagnostics ---
    def detect_overfit_underfit(metrics_dict, tolerance=0.05):
        """
        Detect overfit/underfit based on train/valid metrics.
        For regression: use RMSE or MAE. For classification: use Rain_Accuracy if present.
        Returns: (status, details)
        """
        train = metrics_dict.get("train", {})
        valid = metrics_dict.get("valid", {})
        # Prefer RMSE, fallback to MAE
        metric_name = None
        for m in ["RMSE", "MAE", "Rain_Accuracy"]:
            if m in train and m in valid:
                metric_name = m
                break
        if not metric_name:
            return ("unknown", "Insufficient metrics for overfit/underfit detection.")
        train_score = train[metric_name]
        valid_score = valid[metric_name]
        # For accuracy, higher is better; for errors, lower is better
        if metric_name == "Rain_Accuracy":
            diff = train_score - valid_score
            if diff > tolerance:
                return ("overfit", f"Train accuracy ({train_score:.3f}) > Valid accuracy ({valid_score:.3f}) by {diff:.3f}")
            elif diff < -tolerance:
                return ("underfit", f"Valid accuracy ({valid_score:.3f}) > Train accuracy ({train_score:.3f}) by {-diff:.3f}")
            else:
                return ("good", f"Train/Valid accuracy are similar (diff={diff:.3f})")
        else:
            diff = valid_score - train_score
            if diff > tolerance * train_score:
                return ("overfit", f"Valid error ({valid_score:.3f}) > Train error ({train_score:.3f}) by {diff:.3f}")
            elif diff < -tolerance * train_score:
                return ("underfit", f"Train error ({train_score:.3f}) > Valid error ({valid_score:.3f}) by {-diff:.3f}")
            else:
                return ("good", f"Train/Valid errors are similar (diff={diff:.3f})")

    overfit_status, overfit_details = detect_overfit_underfit(metrics)
    
    # Kiểm tra thêm R² score - nếu âm tức model tệ hơn dự đoán mean
    r2_train = metrics.get("train", {}).get("R2", None)
    r2_valid = metrics.get("valid", {}).get("R2", None)
    r2_test = metrics.get("test", {}).get("R2", None)
    
    underfit_hints = []
    if r2_valid is not None and r2_valid < 0:
        underfit_hints.append(f"Valid R2={r2_valid:.3f} (negative = worse than mean baseline)")
    if r2_test is not None and r2_test < 0:
        underfit_hints.append(f"Test R2={r2_test:.3f} (negative = worse than mean baseline)")
    if r2_train is not None and r2_train < 0.3:
        underfit_hints.append(f"Train R2={r2_train:.3f} (too low, model not learning patterns)")
    
    if underfit_hints:
        overfit_status = "underfit"
        overfit_details += " | " + " | ".join(underfit_hints)
    
    metrics["diagnostics"] = {
        "overfit_status": overfit_status,
        "overfit_details": overfit_details,
        "applied_log_target": applied_log_target,
        "n_features_after_selection": len(all_feature_columns),
        "n_static_features_removed": len(removed_static),
        "target_zero_ratio": float(zero_ratio),
    }

    # Print accuracy if available
    accuracy_msg = ""
    if "Rain_Accuracy" in metrics["test"]:
        acc = metrics["test"]["Rain_Accuracy"]
        accuracy_msg = f"Test Rain_Accuracy: {acc:.4f} ({acc*100:.2f}%)"
    elif "RMSE" in metrics["test"]:
        rmse = metrics["test"]["RMSE"]
        accuracy_msg = f"Test RMSE: {rmse:.4f}"
    elif "MAE" in metrics["test"]:
        mae = metrics["test"]["MAE"]
        accuracy_msg = f"Test MAE: {mae:.4f}"

    metrics_path = artifacts_latest / "Metrics.json"
    _save_json(metrics_path, metrics)

    # --------------------------
    # (9) Save model artifact
    # --------------------------
    model_path = artifacts_latest / "Model.pkl"
    if hasattr(model, "save"):
        model.save(model_path)
    else:
        joblib.dump(model, model_path)

    # --------------------------
    # (10) Save train info
    # --------------------------
    train_info = {
        "trained_at": _now_tag(),
        "project_root": str(project_root),
        "ml_model_root": str(ml_model_root),
        "input": {
            "folder_key": folder_key,
            "filename": filename,
            "file_info": getattr(file_info, "__dict__", str(file_info)),
        },
        "schema_report": schema_report,
        "split_config": asdict(split_config),
        "split_saved_paths": split_paths,
        "target_column": target_col,
        "group_by": group_by,
        "feature_info": {
            "n_created_features": int(len(created_feature_names)),
            "n_total_feature_columns": int(len(all_feature_columns)),
            "n_static_features_removed": int(len(removed_static)),
            "feature_selection_enabled": enable_feature_selection,
        },
        "target_transform": {
            "log1p_applied": applied_log_target,
            "target_zero_ratio": float(zero_ratio),
        },
        "transform": {
            "pipeline_path": str(pipeline_path),
            "pipeline_info": pipeline.get_pipeline_info(),
        },
        "model": {
            "type": model_type,
            "params": model_params,
            "model_path": str(model_path),
        },
        "artifacts": {
            "feature_list": str(feature_list_path),
            "metrics": str(metrics_path),
        },
        "note": "Artifacts saved to Weather_Forcast_App/Machine_learning_artifacts/latest",
    }

    train_info_path = artifacts_latest / "Train_info.json"
    _save_json(train_info_path, train_info)

    return train_info


# ======================================================================================
# CLI
# ======================================================================================
def main():
    parser = argparse.ArgumentParser(description="Weather Forecast - Training Orchestrator")
    parser.add_argument("--config", type=str, required=True, help="Path to train config (.json/.yml)")
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    config = _load_config(cfg_path)

    info = run_training(config)

    # print nhanh cho bạn nhìn log
    print("=" * 80)
    print("[OK] TRAIN DONE")
    print("Artifacts:", (Path(info["model"]["model_path"]).parent))
    print("Model:", info["model"]["model_path"])
    print("Pipeline:", info["transform"]["pipeline_path"])
    print("Metrics:", info["artifacts"]["metrics"])
    print("Train info:", str(Path(info["model"]["model_path"]).parent / "Train_info.json"))
    # Print diagnostics if available
    try:
        import json
        with open(info["artifacts"]["metrics"], "r", encoding="utf-8") as f:
            metrics = json.load(f)
        diag = metrics.get("diagnostics", {})
        print("-" * 80)
        print(f"Overfit/Underfit status: {diag.get('overfit_status', 'N/A')}")
        # Convert to ASCII-safe for Windows console
        details_str = diag.get('overfit_details', '')
        try:
            print(f"Details: {details_str}")
        except UnicodeEncodeError:
            print(f"Details: {details_str.encode('ascii', 'replace').decode()}")
        if "test" in metrics:
            test_metrics = metrics["test"]
            if "Rain_Accuracy" in test_metrics:
                acc = test_metrics["Rain_Accuracy"]
                print(f"Test Rain_Accuracy: {acc:.4f} ({acc*100:.2f}%)")
            elif "RMSE" in test_metrics:
                rmse = test_metrics["RMSE"]
                print(f"Test RMSE: {rmse:.4f}")
            elif "MAE" in test_metrics:
                mae = test_metrics["MAE"]
                print(f"Test MAE: {mae:.4f}")
        print("-" * 80)
    except Exception as e:
        print(f"[Diagnostics] Could not print overfit/underfit/accuracy: {e}")
    print("=" * 80)


if __name__ == "__main__":
    main()