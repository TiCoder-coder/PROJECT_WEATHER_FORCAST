"""
View_Train.py – Giao diện huấn luyện mô hình Machine Learning
================================================================
Cung cấp:
  1) train_view          – GET  : Trang chính cấu hình & bấm Train
  2) train_start_view    – POST : Khởi chạy training (background thread)
  3) train_tail_view     – GET  : Polling logs/progress (JSON)
  4) train_configs_view  – GET  : Liệt kê dataset & config có sẵn
  5) train_artifacts_view – GET : Xem kết quả artifacts (Metrics, Train_info)
"""

from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
import sys
import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from Weather_Forcast_App.paths import (
    APP_ROOT, PROJECT_ROOT,
    DATA_CRAWL_DIR, DATA_MERGE_DIR,
    DATA_CLEAN_ROOT, DATA_CLEAN_MERGE_DIR, DATA_CLEAN_NOT_MERGE_DIR,
    ML_MODEL_ROOT, ML_ARTIFACTS_LATEST,
)

# ────────────────────────────────────────────────────────────────
# In-memory job store  (giống View_Clear.py)
# ────────────────────────────────────────────────────────────────
_JOBS: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _push(job_id: str, line: str) -> None:
    with _LOCK:
        if job_id in _JOBS:
            _JOBS[job_id]["logs"].append(f"[{_now()}] {line}")


def _set_progress(job_id: str, pct: int, step: str = "") -> None:
    with _LOCK:
        if job_id in _JOBS:
            _JOBS[job_id]["progress"] = pct
            if step:
                _JOBS[job_id]["step"] = step


# ────────────────────────────────────────────────────────────────
# Helpers: liệt kê datasets có thể train
# ────────────────────────────────────────────────────────────────
_FOLDER_MAP = {
    "data_crawl":       DATA_CRAWL_DIR,
    "data_merge":       DATA_MERGE_DIR,
    "cleaned_merge":    DATA_CLEAN_MERGE_DIR,
    "cleaned_not_merge": DATA_CLEAN_NOT_MERGE_DIR,
}

_SUPPORTED_EXT = {".csv", ".xlsx", ".xls"}


def _scan_datasets() -> List[Dict[str, str]]:
    """Scan tất cả thư mục data, trả về list {folder_key, filename, size, mtime}."""
    results: List[Dict[str, str]] = []
    for key, folder in _FOLDER_MAP.items():
        if not folder.exists():
            continue
        for p in sorted(folder.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True):
            if p.is_file() and p.suffix.lower() in _SUPPORTED_EXT:
                st = p.stat()
                results.append({
                    "folder_key": key,
                    "filename": p.name,
                    "size_mb": f"{st.st_size / 1024 / 1024:.2f}",
                    "mtime": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M"),
                })
    return results


def _load_existing_config() -> Dict[str, Any]:
    """Load config mẫu nếu có."""
    cfg_path = ML_MODEL_ROOT / "config" / "train_config.json"
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    # fallback
    cfg_path2 = PROJECT_ROOT / "config" / "train_config.json"
    if cfg_path2.exists():
        try:
            return json.loads(cfg_path2.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _load_artifacts() -> Dict[str, Any]:
    """Load thông tin artifacts gần nhất."""
    result: Dict[str, Any] = {}
    metrics_path = ML_ARTIFACTS_LATEST / "Metrics.json"
    train_info_path = ML_ARTIFACTS_LATEST / "Train_info.json"
    feature_list_path = ML_ARTIFACTS_LATEST / "Feature_list.json"

    for key, path in [("metrics", metrics_path), ("train_info", train_info_path),
                       ("feature_list", feature_list_path)]:
        if path.exists():
            try:
                result[key] = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                result[key] = None
        else:
            result[key] = None

    # Check model file
    model_path = ML_ARTIFACTS_LATEST / "Model.pkl"
    result["model_exists"] = model_path.exists()
    if model_path.exists():
        result["model_size_mb"] = f"{model_path.stat().st_size / 1024 / 1024:.2f}"
    else:
        result["model_size_mb"] = "0"

    return result


# ────────────────────────────────────────────────────────────────
# Background worker: chạy training
# ────────────────────────────────────────────────────────────────
def _training_worker(job_id: str, config: Dict[str, Any]) -> None:
    """Chạy training trong background thread, redirect stdout để capture logs."""
    old_stdout = sys.stdout
    try:
        _push(job_id, "🚀 Bắt đầu quá trình huấn luyện...")
        _set_progress(job_id, 5, "Khởi tạo")

        # Redirect stdout để capture print từ train.py
        sys.stdout = _LogCapture(job_id)

        _push(job_id, f"📂 Dataset: {config.get('data', {}).get('filename', 'N/A')}")
        _push(job_id, f"🤖 Model: {config.get('model', {}).get('type', 'N/A')}")
        _set_progress(job_id, 10, "Loading training module")

        # Import & run training
        from Weather_Forcast_App.Machine_learning_model.trainning.train import run_training

        _set_progress(job_id, 15, "Đang huấn luyện...")

        train_info = run_training(config)

        _set_progress(job_id, 95, "Lưu kết quả")

        # Restore stdout trước khi push log cuối
        sys.stdout = old_stdout

        _push(job_id, "✅ Huấn luyện hoàn tất!")

        # Trích xuất metrics
        metrics_path = ML_ARTIFACTS_LATEST / "Metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            test_metrics = metrics.get("test", {})
            r2 = test_metrics.get("R2", "N/A")
            rmse = test_metrics.get("RMSE", "N/A")
            mae = test_metrics.get("MAE", "N/A")
            rain_acc = test_metrics.get("Rain_Detection_Accuracy", "N/A")
            diag = metrics.get("diagnostics", {})

            _push(job_id, f"📊 Test R²: {r2}")
            _push(job_id, f"📊 Test RMSE: {rmse}")
            _push(job_id, f"📊 Test MAE: {mae}")
            if rain_acc != "N/A":
                _push(job_id, f"🌧️ Rain Detection Accuracy: {rain_acc}")
            _push(job_id, f"🔍 Overfit Status: {diag.get('overfit_status', 'N/A')}")

        _set_progress(job_id, 100, "Hoàn tất")
        with _LOCK:
            _JOBS[job_id]["status"] = "done"
            _JOBS[job_id]["result"] = "success"

    except Exception as e:
        _push(job_id, f"❌ Lỗi: {str(e)}")
        _push(job_id, traceback.format_exc())
        _set_progress(job_id, 100, "Lỗi")
        with _LOCK:
            _JOBS[job_id]["status"] = "error"
            _JOBS[job_id]["result"] = str(e)
    finally:
        # Luôn restore stdout dù thành công hay lỗi
        sys.stdout = old_stdout


class _LogCapture(io.TextIOBase):
    """Redirect print() output → job logs."""

    def __init__(self, job_id: str):
        self.job_id = job_id

    def write(self, s: str) -> int:
        if s and s.strip():
            _push(self.job_id, s.strip())
        return len(s) if s else 0

    def flush(self):
        pass

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False


# ────────────────────────────────────────────────────────────────
# VIEWS
# ────────────────────────────────────────────────────────────────

@require_http_methods(["GET"])
def train_view(request):
    """Trang chính huấn luyện model."""
    datasets = _scan_datasets()
    default_config = _load_existing_config()
    artifacts = _load_artifacts()

    # Lấy trạng thái job đang chạy (nếu có)
    active_job = None
    with _LOCK:
        for jid, jdata in _JOBS.items():
            if jdata.get("status") == "running":
                active_job = {"job_id": jid, **jdata}
                break

    # Serialize datasets thành JSON string cho JS
    datasets_json = json.dumps(datasets, ensure_ascii=False)

    context = {
        "datasets": datasets,
        "datasets_json": datasets_json,
        "default_config": json.dumps(default_config, ensure_ascii=False, indent=2),
        "default_config_dict": default_config,
        "artifacts": artifacts,
        "active_job": active_job,
        "model_types": [
            {"value": "xgboost", "label": "XGBoost"},
            {"value": "random_forest", "label": "Random Forest"},
            {"value": "lightgbm", "label": "LightGBM"},
            {"value": "catboost", "label": "CatBoost"},
            {"value": "ensemble", "label": "Ensemble (Voting)"},
        ],
    }
    return render(request, "weather/HTML_Train.html", context)


@require_http_methods(["POST"])
def train_start_view(request):
    """Khởi chạy training job."""
    # Kiểm tra có job nào đang chạy không
    with _LOCK:
        for jid, jdata in _JOBS.items():
            if jdata.get("status") == "running":
                return JsonResponse({
                    "ok": False,
                    "error": f"Đang có job huấn luyện chạy (ID: {jid}). Vui lòng chờ hoàn tất."
                }, status=409)

    try:
        body = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"ok": False, "error": "Invalid JSON body"}, status=400)

    # Build config
    folder_key = body.get("folder_key", "")
    filename = body.get("filename", "")
    model_type = body.get("model_type", "xgboost")
    target_column = body.get("target_column", "rain_total")
    test_size = float(body.get("test_size", 0.15))
    valid_size = float(body.get("valid_size", 0.15))
    use_default_config = body.get("use_default_config", False)

    if use_default_config:
        config = _load_existing_config()
        # Override dataset nếu user chọn
        if folder_key and filename:
            config["data"] = {"folder_key": folder_key, "filename": filename}
    else:
        config = {
            "data": {"folder_key": folder_key, "filename": filename},
            "skip_schema_validation": True,
            "target_column": target_column,
            "features": {
                "input_columns": [],
                "lag_features": False,
                "rolling_features": False,
                "difference_features": False,
                "time_features": True,
                "location_features": True,
                "interaction_features": True,
            },
            "auto_detect_data_type": True,
            "polynomial_features": {"enabled": True, "degree": 2, "top_k_corr": 8},
            "feature_selection": {"enabled": False},
            "model": {
                "type": model_type,
                "params": body.get("model_params", {}),
            },
            "split": {
                "test_size": test_size,
                "valid_size": valid_size,
                "shuffle": True,
                "sort_by_time": False,
            },
            "cv_folds": int(body.get("cv_folds", 5)),
            "random_state": 42,
            "metric": "rmse",
        }

        # Nếu ensemble, dùng config mặc định cho base_models
        if model_type == "ensemble":
            default_cfg = _load_existing_config()
            ensemble_params = default_cfg.get("model", {}).get("params", {})
            if ensemble_params:
                config["model"]["params"] = ensemble_params

    # Validate
    if not config.get("data", {}).get("folder_key"):
        return JsonResponse({"ok": False, "error": "Chưa chọn thư mục dataset"}, status=400)
    if not config.get("data", {}).get("filename"):
        return JsonResponse({"ok": False, "error": "Chưa chọn file dataset"}, status=400)

    # Tạo job
    job_id = str(uuid.uuid4())[:8]
    with _LOCK:
        _JOBS[job_id] = {
            "status": "running",
            "logs": [],
            "progress": 0,
            "step": "Khởi tạo",
            "started_at": _now(),
            "config": config,
            "result": None,
        }

    # Spawn thread
    t = threading.Thread(target=_training_worker, args=(job_id, config), daemon=True)
    t.start()

    return JsonResponse({"ok": True, "job_id": job_id})


@require_http_methods(["GET"])
def train_tail_view(request):
    """Polling endpoint – trả về logs & progress."""
    job_id = request.GET.get("job_id", "")
    try:
        after = int(request.GET.get("after", 0))
    except (ValueError, TypeError):
        after = 0

    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return JsonResponse({"ok": False, "error": "Job not found"}, status=404)

        logs = job["logs"][after:]
        return JsonResponse({
            "ok": True,
            "logs": logs,
            "total": len(job["logs"]),
            "progress": job.get("progress", 0),
            "step": job.get("step", ""),
            "status": job.get("status", "running"),
            "result": job.get("result"),
        })


@require_http_methods(["GET"])
def train_configs_view(request):
    """API: trả về danh sách datasets + config mặc định."""
    datasets = _scan_datasets()
    config = _load_existing_config()
    return JsonResponse({
        "ok": True,
        "datasets": datasets,
        "default_config": config,
    })


@require_http_methods(["GET"])
def train_artifacts_view(request):
    """API: trả về kết quả artifacts mới nhất."""
    artifacts = _load_artifacts()
    return JsonResponse({
        "ok": True,
        "artifacts": artifacts,
    })
