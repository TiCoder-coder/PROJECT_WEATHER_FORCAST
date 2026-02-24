

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
# (2) IMPORT CÁC MODULE BẠN ĐÃ CÓ
# ======================================================================================
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
    cfg = dict(model_config or {})

    if model_type == "ensemble":
        base_models = cfg.get("base_models", [])
        from Weather_Forcast_App.Machine_learning_model.Models.Ensemble_Model import WeatherEnsembleModel
        return WeatherEnsembleModel(base_models=base_models, model_registry=MODEL_REGISTRY)
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model_type='{model_type}'. Use: {list(set(MODEL_REGISTRY.keys()))}")

    module_path, class_name = MODEL_REGISTRY[model_type].rsplit(".", 1)
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)

    # task_type mặc định cho training pipeline hiện tại là regression
    task_type = str(cfg.pop("task_type", "regression")).lower()

    # Các wrapper có signature khác nhau:
    # - XGBoost/LightGBM: nhận `params=...`
    # - RandomForest/CatBoost: nhận trực tiếp kwargs
    if model_type in {"xgb", "xgboost", "lgbm", "lightgbm"}:
        return model_class(task_type=task_type, params=cfg)

    if model_type in {"rf", "random_forest", "randomforest", "cat", "catboost"}:
        return model_class(task_type=task_type, **cfg)

    return model_class(**cfg)


def _train_model(model, model_type: str, X_train, y_train, X_valid, y_valid):
    """
    Train model theo đúng signature từng wrapper, ưu tiên dùng validation set
    khi wrapper hỗ trợ để tránh split lặp không cần thiết.
    """
    mtype = (model_type or "").lower().strip()

    def _ensure_training_success(train_output):
        # Nhiều wrapper trả TrainingResult(success=...) thay vì raise exception
        # -> ép fail-fast để không tiếp tục với model train lỗi.
        if hasattr(train_output, "success") and not bool(getattr(train_output, "success")):
            msg = getattr(train_output, "message", "Unknown training error")
            raise RuntimeError(f"Model training failed: {msg}")
        return train_output

    if hasattr(model, "train"):
        # XGBoost / LightGBM wrappers hỗ trợ truyền validation trực tiếp
        if mtype in {"xgb", "xgboost", "lgbm", "lightgbm"}:
            try:
                out = model.train(
                    X_train,
                    y_train,
                    X_val=X_valid,
                    y_val=y_valid,
                    val_size=0.0,
                    shuffle=False,
                    verbose=False,
                )
                return _ensure_training_success(out)
            except TypeError:
                pass

        # RandomForest/CatBoost wrappers hiện chưa nhận X_val/y_val trực tiếp,
        # nên để wrapper tự split nội bộ với default validation_split hợp lệ.
        if mtype in {"rf", "random_forest", "randomforest"}:
            try:
                out = model.train(X_train, y_train, verbose=False)
                return _ensure_training_success(out)
            except TypeError:
                pass
        if mtype in {"cat", "catboost"}:
            try:
                out = model.train(X_train, y_train, verbose=False)
                return _ensure_training_success(out)
            except TypeError:
                pass

        out = model.train(X_train, y_train)
        return _ensure_training_success(out)

    if hasattr(model, "fit"):
        return model.fit(X_train, y_train)

    raise RuntimeError(f"Model wrapper '{type(model).__name__}' has no train() or fit().")


def _normalize_metric_keys(metric_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chuẩn hóa key metric giữa các wrapper để phần diagnostics/summary dùng ổn định.
    """
    if not isinstance(metric_dict, dict):
        return metric_dict

    out = dict(metric_dict)
    alias = {
        "rmse": "RMSE",
        "mae": "MAE",
        "mse": "MSE",
        "r2": "R2",
        "r2_score": "R2",
        "accuracy": "Accuracy",
        "rain_accuracy": "Rain_Accuracy",
        "f1_score": "F1",
        "precision": "Precision",
        "recall": "Recall",
    }

    for key, value in list(metric_dict.items()):
        k = str(key).lower()
        mapped = alias.get(k)
        if mapped and mapped not in out:
            out[mapped] = value
    return out


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
        shuffle=bool(split_cfg.get("shuffle", False)),
        sort_by_time_if_possible=bool(split_cfg.get("sort_by_time", True)),
    )

    df_train, df_valid_split, df_test = split_dataframe(df_valid, split_config)

    # Xác định lưu vào Dataset_merge hay Dataset_not_merge theo folder_key
    folder_key_lc = folder_key.lower()
    if any(token in folder_key_lc for token in ["output", "raw", "not_merge", "not-merge"]):
        split_out_dir = dataset_after_split_not_merge
        split_name = "not_merge"
    elif "merge" in folder_key_lc:
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
    _train_model(model, model_type=model_type, X_train=X_train, y_train=y_train, X_valid=X_valid_t, y_valid=y_valid)

    # --------------------------
    # (8) Evaluate metrics - Sử dụng module metrics.py
    # --------------------------
    metrics: Dict[str, Any] = {"generated_at": _now_tag(), "model_type": model_type}

    def _evaluate_set(X, y):
        """Helper để evaluate một dataset."""
        if hasattr(model, "evaluate"):
            return _normalize_metric_keys(model.evaluate(X, y))
        # Fallback: sử dụng metrics.py
        y_pred = model.predict(X)
        if hasattr(y_pred, "predictions"):  # Handle PredictionResult dataclass
            y_pred = y_pred.predictions
        return _normalize_metric_keys(calculate_all_metrics(np.array(y), np.array(y_pred), n_features=X.shape[1]))

    metrics["train"] = _evaluate_set(X_train, y_train)
    metrics["valid"] = _evaluate_set(X_valid_t, y_valid)
    metrics["test"] = _evaluate_set(X_test_t, y_test)


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
        for m in ["RMSE", "MAE", "Accuracy", "Rain_Accuracy"]:
            if m in train and m in valid:
                metric_name = m
                break
        if not metric_name:
            return ("unknown", "Insufficient metrics for overfit/underfit detection.")
        train_score = train[metric_name]
        valid_score = valid[metric_name]
        # For accuracy, higher is better; for errors, lower is better
        if metric_name in {"Rain_Accuracy", "Accuracy"}:
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
    metrics["diagnostics"] = {
        "overfit_status": overfit_status,
        "overfit_details": overfit_details,
    }

    # Print accuracy if available
    accuracy_msg = ""
    if "Rain_Accuracy" in metrics["test"]:
        acc = metrics["test"]["Rain_Accuracy"]
        accuracy_msg = f"Test Rain_Accuracy: {acc:.4f} ({acc*100:.2f}%)"
    elif "Accuracy" in metrics["test"]:
        acc = metrics["test"]["Accuracy"]
        accuracy_msg = f"Test Accuracy: {acc:.4f} ({acc*100:.2f}%)"
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
        try:
            model.save(model_path)
        except Exception:
            # Một số wrapper (vd CatBoost) có định dạng save riêng (.cbm)
            # nên fallback về joblib để chuẩn hóa artifact đầu ra.
            joblib.dump(model, model_path)
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
    # Print diagnostics if available
    try:
        import json
        with open(info["artifacts"]["metrics"], "r") as f:
            metrics = json.load(f)
        diag = metrics.get("diagnostics", {})
        print("-" * 80)
        print(f"Overfit/Underfit status: {diag.get('overfit_status', 'N/A')}")
        print(f"Details: {diag.get('overfit_details', '')}")
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
