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

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
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
from Weather_Forcast_App.Machine_learning_model.data.Schema import validate_weather_dataframe, WeatherDataSchema
from Weather_Forcast_App.Machine_learning_model.data.Split import SplitConfig, split_dataframe

from Weather_Forcast_App.Machine_learning_model.features.Build_transfer import WeatherFeatureBuilder
from Weather_Forcast_App.Machine_learning_model.features.Transformers import WeatherTransformPipeline

# Model wrappers (bạn nói đã code rồi)
# LƯU Ý: tên class phải khớp file bạn đang dùng trong project
from Weather_Forcast_App.Machine_learning_model.Models.Random_Forest_Model import WeatherRandomForest
from Weather_Forcast_App.Machine_learning_model.Models.XGBoost_Model import WeatherXGBoost
from Weather_Forcast_App.Machine_learning_model.Models.LightGBM_Model import WeatherLightGBM
from Weather_Forcast_App.Machine_learning_model.Models.CatBoost_Model import WeatherCatBoost


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
        except Exception as e:
            raise RuntimeError("Config is YAML but PyYAML not installed. Install pyyaml or use JSON.") from e
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    raise ValueError(f"Unsupported config extension: {ext}. Use .json or .yml/.yaml")

def _load_df_via_loader(app_root: Path, folder_key: str, filename: str) -> tuple[pd.DataFrame, object]:
    loader = DataLoader(base_path=str(app_root))  # ✅ base_path trỏ về Weather_Forcast_App
    result = loader.load_all(folder_key, filename)

    if not result.is_success or result.data is None:
        raise FileNotFoundError(f"Cannot load data: folder_key={folder_key}, filename={filename}. Error: {result.message}")

    if not isinstance(result.data, pd.DataFrame):
        raise ValueError(f"Loaded data is not a DataFrame. Got: {type(result.data)}")

    return result.data, result.file_info

def _now_tag() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ======================================================================================
# (4) SAVE HELPERS
# ======================================================================================
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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
# (7) MODEL FACTORY
# ======================================================================================
def _create_model(model_type: str, model_config: Dict[str, Any]):
    """
    Tạo instance model wrapper theo config.
    model_type: random_forest | xgboost | lightgbm | catboost
    """
    model_type = (model_type or "").lower().strip()

    if model_type in ["rf", "random_forest", "randomforest"]:
        return WeatherRandomForest(**model_config)

    if model_type in ["xgb", "xgboost"]:
        return WeatherXGBoost(**model_config)

    if model_type in ["lgbm", "lightgbm"]:
        return WeatherLightGBM(**model_config)

    if model_type in ["cat", "catboost"]:
        return WeatherCatBoost(**model_config)

    raise ValueError(f"Unsupported model_type='{model_type}'. Use: random_forest/xgboost/lightgbm/catboost")


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
    app_root = Path(__file__).resolve()   # Weather_Forcast_App
    app_root = app_root.parents[3]

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
    loader = DataLoader(base_path=str(ml_model_root))

    folder_key = data_cfg.get("folder_key")
    filename = data_cfg.get("filename")

    if not folder_key:
        raise ValueError("Config missing: data.folder_key")
    if not filename:
        raise ValueError("Config missing: data.filename")

    # --------------------------
    # (3) Validate schema (Schema.py)
    # --------------------------
    df_raw, file_info = _load_df_via_loader(app_root, folder_key, filename)

    if len(df_raw) == 0:
        raise RuntimeError("After loading data: no rows left. Check input data.")

    df_valid, schema_report = _validate_schema_keep_valid_rows(df_raw)

    if len(df_valid) == 0:
        raise RuntimeError("After schema validation: no valid rows left. Check input data & schema rules.")

    # --------------------------
    # (4) Split train/valid/test (Split.py) + save to Dataset_after_split/...
    # --------------------------
    split_config = SplitConfig(
        test_size=float(split_cfg.get("test_size", 0.1)),
        valid_size=float(split_cfg.get("valid_size", 0.1)),
        random_state=int(split_cfg.get("random_state", 42)),
        shuffle=bool(split_cfg.get("shuffle", True)),
        stratify_col=split_cfg.get("stratify_col"),  # thường để None (vì regression)
        time_col=split_cfg.get("time_col"),          # nếu time-series: set cột thời gian
        sort_by_time=bool(split_cfg.get("sort_by_time", False)),
        group_col=split_cfg.get("group_col"),        # nếu split theo trạm
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

    # feature list: list các feature mới + toàn bộ cột output (sau build)
    # - builder.get_feature_names() là "features tạo thêm"
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
        "note": "all_feature_columns là danh sách cột X sau build_all_features (để predict align đúng cột)."
    })

    # --------------------------
    # (6) Transform pipeline thống nhất train/predict (Transformers.py)
    # --------------------------
    pipeline = WeatherTransformPipeline(
        missing_strategy=transform_cfg.get("missing_strategy", "median"),
        scaler_type=transform_cfg.get("scaler_type", "standard"),
        encoding_type=transform_cfg.get("encoding_type", "label"),
        handle_outliers=bool(transform_cfg.get("handle_outliers", False)),
        outlier_method=transform_cfg.get("outlier_method", "iqr"),
    )

    # Fit ONLY on train, rồi transform valid/test
    X_train = pipeline.fit_transform(X_train_raw, y_train if transform_cfg.get("pass_y_to_transform", False) else None)
    X_valid_t = pipeline.transform(X_valid_raw)
    X_test_t = pipeline.transform(X_test_raw)

    # Save pipeline
    pipeline_path = artifacts_latest / "Transform_pipeline.pkl"
    pipeline.save(pipeline_path)

    # --------------------------
    # (7) Train model
    # --------------------------
    model_type = model_cfg.get("type", "random_forest")
    model_params = model_cfg.get("params", {})

    model = _create_model(model_type=model_type, model_config=model_params)

    # Wrapper của bạn thường có model.train(X, y) (nhiều file bạn làm kiểu vậy)
    # Nếu wrapper bạn khác, bạn sửa đúng method name ở đây.
    if hasattr(model, "train"):
        model.train(X_train, y_train)
    elif hasattr(model, "fit"):
        model.fit(X_train, y_train)
    else:
        raise RuntimeError(f"Model wrapper '{type(model).__name__}' has no train() or fit().")

    # --------------------------
    # (8) Evaluate metrics
    # --------------------------
    metrics: Dict[str, Any] = {"generated_at": _now_tag(), "model_type": model_type}

    # Ưu tiên wrapper.evaluate() nếu có
    if hasattr(model, "evaluate"):
        metrics["train"] = model.evaluate(X_train, y_train)
        metrics["valid"] = model.evaluate(X_valid_t, y_valid)
        metrics["test"] = model.evaluate(X_test_t, y_test)
    else:
        # fallback tự tính (nếu wrapper không có)
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        y_pred = model.predict(X_test_t)
        metrics["test"] = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2": float(r2_score(y_test, y_pred)),
        }

    metrics_path = artifacts_latest / "Metrics.json"
    _save_json(metrics_path, metrics)

    # --------------------------
    # (9) Save model artifact
    # --------------------------
    model_path = artifacts_latest / "Model.pkl"
    if hasattr(model, "save"):
        model.save(model_path)
    else:
        # fallback joblib dump
        import joblib
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
    print("✅ TRAIN DONE")
    print("Artifacts:", (Path(info["model"]["model_path"]).parent))
    print("Model:", info["model"]["model_path"])
    print("Pipeline:", info["transform"]["pipeline_path"])
    print("Metrics:", info["artifacts"]["metrics"])
    print("Train info:", str(Path(info["model"]["model_path"]).parent / "Train_info.json"))
    print("=" * 80)


if __name__ == "__main__":
    main()
