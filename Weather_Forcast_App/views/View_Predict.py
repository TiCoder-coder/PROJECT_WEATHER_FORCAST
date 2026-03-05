"""
View_Predict.py – Giao diện dự báo thời tiết (Inference / Prediction)
=====================================================================
Cung cấp:
  1) predict_view          – GET  : Trang chính dự báo
  2) predict_run_view      – POST : Chạy prediction (background thread)
  3) predict_tail_view     – GET  : Polling logs/progress (JSON)
  4) predict_manual_view   – POST : Dự báo từ form nhập tay (single row)
  5) predict_model_info_view – GET : Thông tin model hiện tại
"""

from __future__ import annotations

import json
import threading
import traceback
import uuid
import sys
import io
import csv
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt

from Weather_Forcast_App.paths import (
    APP_ROOT, PROJECT_ROOT,
    DATA_CRAWL_DIR, DATA_MERGE_DIR,
    DATA_CLEAN_ROOT, DATA_CLEAN_MERGE_DIR, DATA_CLEAN_NOT_MERGE_DIR,
    ML_MODEL_ROOT, ML_ARTIFACTS_LATEST,
)

# ────────────────────────────────────────────────────────────────
# In-memory job store
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
# Helpers
# ────────────────────────────────────────────────────────────────
_FOLDER_MAP = {
    "data_crawl":        DATA_CRAWL_DIR,
    "data_merge":        DATA_MERGE_DIR,
    "cleaned_merge":     DATA_CLEAN_MERGE_DIR,
    "cleaned_not_merge": DATA_CLEAN_NOT_MERGE_DIR,
}

_SUPPORTED_EXT = {".csv", ".xlsx", ".xls"}


def _scan_datasets() -> List[Dict[str, str]]:
    """Scan tất cả thư mục data, trả về list dict."""
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


def _load_model_info() -> Dict[str, Any]:
    """Load thông tin model hiện tại từ artifacts."""
    info: Dict[str, Any] = {
        "loaded": False,
        "model_type": "",
        "target_column": "",
        "n_features": 0,
        "trained_at": "",
        "model_size_mb": "0",
    }

    train_info_path = ML_ARTIFACTS_LATEST / "Train_info.json"
    if train_info_path.exists():
        try:
            ti = json.loads(train_info_path.read_text(encoding="utf-8"))
            info["trained_at"] = ti.get("trained_at", "")
            info["model_type"] = ti.get("model", {}).get("type", "")
            info["target_column"] = ti.get("target_column", "rain_total")
            info["predict_threshold"] = (
                ti.get("model", {}).get("params", {}).get("predict_threshold", 0.5)
            )
        except Exception:
            pass

    feature_path = ML_ARTIFACTS_LATEST / "Feature_list.json"
    if feature_path.exists():
        try:
            fl = json.loads(feature_path.read_text(encoding="utf-8"))
            info["n_features"] = len(fl.get("all_feature_columns", []))
        except Exception:
            pass

    metrics_path = ML_ARTIFACTS_LATEST / "Metrics.json"
    if metrics_path.exists():
        try:
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            info["test_r2"] = m.get("test", {}).get("R2")
            info["test_rmse"] = m.get("test", {}).get("RMSE")
            info["test_rain_acc"] = m.get("test", {}).get("Rain_Detection_Accuracy")
            info["diagnostics"] = m.get("diagnostics", {})
            info["diag_status"] = m.get("diagnostics", {}).get("overfit_status", "")
            info["diag_details"] = m.get("diagnostics", {}).get("overfit_details", "")
        except Exception:
            pass

    model_path = ML_ARTIFACTS_LATEST / "Model.pkl"
    if model_path.exists():
        info["loaded"] = True
        info["model_size_mb"] = f"{model_path.stat().st_size / 1024 / 1024:.2f}"

    return info


def _load_recent_predictions() -> List[Dict[str, Any]]:
    """Load kết quả dự báo gần nhất (nếu có)."""
    pred_path = ML_MODEL_ROOT / "WeatherForcast" / "predictions.csv"
    if not pred_path.exists():
        return []
    try:
        df = pd.read_csv(pred_path, nrows=50)
        rows = []
        for _, row in df.iterrows():
            rows.append({
                "station_name": str(row.get("station_name", "")),
                "province": str(row.get("province", "")),
                "district": str(row.get("district", "")),
                "rain_total": row.get("rain_total", ""),
                "status": str(row.get("status", "")),
                "timestamp": str(row.get("timestamp", "")),
                "y_pred": round(float(row.get("y_pred", 0)), 4) if pd.notna(row.get("y_pred")) else "",
            })
        return rows
    except Exception:
        return []


# ────────────────────────────────────────────────────────────────
# Stdout capture (giống View_Train.py)
# ────────────────────────────────────────────────────────────────
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
# Background worker: chạy prediction
# ────────────────────────────────────────────────────────────────
def _prediction_worker(job_id: str, file_path: str, nrows: Optional[int] = None) -> None:
    """Chạy prediction trong background thread."""
    old_stdout = sys.stdout
    try:
        _push(job_id, "🔮 Bắt đầu quá trình dự báo...")
        _set_progress(job_id, 5, "Khởi tạo")

        sys.stdout = _LogCapture(job_id)

        _push(job_id, f"📂 File: {Path(file_path).name}")
        _set_progress(job_id, 10, "Đọc dữ liệu")

        # Đọc file
        fp = Path(file_path)
        if fp.suffix.lower() == ".csv":
            df = pd.read_csv(fp, nrows=nrows) if nrows else pd.read_csv(fp)
        else:
            df = pd.read_excel(fp, nrows=nrows) if nrows else pd.read_excel(fp)

        _push(job_id, f"📊 Đọc được {len(df)} dòng, {len(df.columns)} cột")
        _set_progress(job_id, 20, "Load model")

        # Load predictor
        from Weather_Forcast_App.Machine_learning_model.interface.predictor import WeatherPredictor
        predictor = WeatherPredictor.from_artifacts(str(ML_ARTIFACTS_LATEST))

        _push(job_id, f"🤖 Model: {type(predictor.model).__name__}")
        _push(job_id, f"🎯 Target: {predictor.target_column}")
        _push(job_id, f"📐 Features: {len(predictor.feature_columns)}")
        _set_progress(job_id, 40, "Đang dự báo...")

        # Chạy prediction
        result = predictor.predict(df)
        predictions = result["predictions"]
        pred_time = result["prediction_time"]

        sys.stdout = old_stdout

        _push(job_id, f"✅ Dự báo xong! {len(predictions)} kết quả trong {pred_time:.2f}s")
        _set_progress(job_id, 80, "Lưu kết quả")

        # Gắn y_pred vào DataFrame gốc
        df["y_pred"] = predictions

        # Lưu file kết quả
        output_dir = ML_MODEL_ROOT / "WeatherForcast"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "predictions.csv"
        df.to_csv(output_path, index=False)
        _push(job_id, f"💾 Đã lưu: {output_path.name}")

        # Tính statistics
        pred_arr = np.array(predictions, dtype=float)
        _push(job_id, f"📊 Mean prediction: {np.mean(pred_arr):.4f}")
        _push(job_id, f"📊 Std prediction: {np.std(pred_arr):.4f}")
        _push(job_id, f"📊 Min: {np.min(pred_arr):.4f}, Max: {np.max(pred_arr):.4f}")

        # Nếu có cột target thực tế, tính RMSE
        target_col = predictor.target_column
        if target_col and target_col in df.columns:
            y_true = df[target_col].dropna()
            y_pred_aligned = pred_arr[:len(y_true)]
            if len(y_true) > 0:
                rmse = np.sqrt(np.mean((y_true.values - y_pred_aligned) ** 2))
                _push(job_id, f"📊 RMSE (so với actual): {rmse:.4f}")

        # Lấy top rows cho preview
        preview_rows = []
        preview_df = df.head(20)
        cols_to_show = []
        for c in ["station_name", "province", "district", "timestamp", "rain_total", "status", "y_pred"]:
            if c in preview_df.columns:
                cols_to_show.append(c)
        if not cols_to_show:
            cols_to_show = list(preview_df.columns[:7])

        for _, row in preview_df.iterrows():
            r = {}
            for c in cols_to_show:
                val = row[c]
                if isinstance(val, float):
                    r[c] = round(val, 4) if pd.notna(val) else ""
                else:
                    r[c] = str(val) if pd.notna(val) else ""
            preview_rows.append(r)

        _set_progress(job_id, 100, "Hoàn tất")
        with _LOCK:
            _JOBS[job_id]["status"] = "done"
            _JOBS[job_id]["result"] = "success"
            _JOBS[job_id]["preview"] = preview_rows
            _JOBS[job_id]["preview_columns"] = cols_to_show
            _JOBS[job_id]["stats"] = {
                "n_samples": len(predictions),
                "mean": round(float(np.mean(pred_arr)), 4),
                "std": round(float(np.std(pred_arr)), 4),
                "min": round(float(np.min(pred_arr)), 4),
                "max": round(float(np.max(pred_arr)), 4),
                "prediction_time": round(pred_time, 2),
            }

    except Exception as e:
        _push(job_id, f"❌ Lỗi: {str(e)}")
        _push(job_id, traceback.format_exc())
        _set_progress(job_id, 100, "Lỗi")
        with _LOCK:
            _JOBS[job_id]["status"] = "error"
            _JOBS[job_id]["result"] = str(e)
    finally:
        sys.stdout = old_stdout


# ────────────────────────────────────────────────────────────────
# VIEWS
# ────────────────────────────────────────────────────────────────

@require_http_methods(["GET"])
def predict_view(request):
    """Trang chính dự báo."""
    datasets = _scan_datasets()
    model_info = _load_model_info()
    recent_preds = _load_recent_predictions()

    # Lấy trạng thái job đang chạy (nếu có)
    active_job = None
    with _LOCK:
        for jid, jdata in _JOBS.items():
            if jdata.get("status") == "running":
                active_job = {"job_id": jid, **jdata}
                break

    datasets_json = json.dumps(datasets, ensure_ascii=False)
    recent_preds_json = json.dumps(recent_preds, ensure_ascii=False)

    context = {
        "datasets": datasets,
        "datasets_json": datasets_json,
        "model_info": model_info,
        "recent_preds": recent_preds,
        "recent_preds_json": recent_preds_json,
        "active_job": active_job,
    }
    return render(request, "weather/HTML_Predict.html", context)


@require_http_methods(["POST"])
def predict_run_view(request):
    """Khởi chạy prediction job từ dataset có sẵn hoặc file upload."""

    # Kiểm tra job đang chạy
    with _LOCK:
        for jid, jdata in _JOBS.items():
            if jdata.get("status") == "running":
                return JsonResponse({
                    "ok": False,
                    "error": f"Đang có job dự báo chạy (ID: {jid}). Vui lòng chờ."
                }, status=409)

    # Xử lý file upload
    uploaded_file = request.FILES.get("file")
    folder_key = request.POST.get("folder_key", "")
    filename = request.POST.get("filename", "")
    nrows_str = request.POST.get("nrows", "")

    nrows = None
    if nrows_str and nrows_str.strip():
        try:
            nrows = int(nrows_str)
            if nrows <= 0:
                nrows = None
        except ValueError:
            pass

    file_path = None

    if uploaded_file:
        # Lưu file tạm
        suffix = Path(uploaded_file.name).suffix
        if suffix.lower() not in _SUPPORTED_EXT:
            return JsonResponse({"ok": False, "error": f"Định dạng không hỗ trợ: {suffix}"}, status=400)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(DATA_CRAWL_DIR))
        for chunk in uploaded_file.chunks():
            tmp.write(chunk)
        tmp.close()
        file_path = tmp.name
    elif folder_key and filename:
        folder = _FOLDER_MAP.get(folder_key)
        if not folder:
            return JsonResponse({"ok": False, "error": "Thư mục không hợp lệ"}, status=400)
        fp = folder / filename
        if not fp.exists():
            return JsonResponse({"ok": False, "error": f"File không tồn tại: {filename}"}, status=400)
        file_path = str(fp)
    else:
        return JsonResponse({"ok": False, "error": "Chưa chọn file hoặc dataset"}, status=400)

    # Kiểm tra model
    model_path = ML_ARTIFACTS_LATEST / "Model.pkl"
    if not model_path.exists():
        return JsonResponse({
            "ok": False,
            "error": "Chưa có model đã train. Vui lòng huấn luyện trước."
        }, status=400)

    # Tạo job
    job_id = str(uuid.uuid4())[:8]
    with _LOCK:
        _JOBS[job_id] = {
            "status": "running",
            "logs": [],
            "progress": 0,
            "step": "Khởi tạo",
            "started_at": _now(),
            "file_path": file_path,
            "result": None,
            "preview": [],
            "preview_columns": [],
            "stats": {},
        }

    t = threading.Thread(target=_prediction_worker, args=(job_id, file_path, nrows), daemon=True)
    t.start()

    return JsonResponse({"ok": True, "job_id": job_id})


@require_http_methods(["GET"])
def predict_tail_view(request):
    """Polling endpoint – trả về logs, progress & preview."""
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
        resp = {
            "ok": True,
            "logs": logs,
            "total": len(job["logs"]),
            "progress": job.get("progress", 0),
            "step": job.get("step", ""),
            "status": job.get("status", "running"),
            "result": job.get("result"),
        }

        # Nếu job xong, gửi luôn preview data
        if job.get("status") in ("done", "error"):
            resp["preview"] = job.get("preview", [])
            resp["preview_columns"] = job.get("preview_columns", [])
            resp["stats"] = job.get("stats", {})

        return JsonResponse(resp)


@require_http_methods(["POST"])
def predict_manual_view(request):
    """Dự báo từ form nhập tay (single row hoặc few rows)."""
    try:
        body = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"ok": False, "error": "Invalid JSON body"}, status=400)

    rows_data = body.get("rows", [])
    if not rows_data:
        return JsonResponse({"ok": False, "error": "Không có dữ liệu đầu vào"}, status=400)

    # Kiểm tra model
    model_path = ML_ARTIFACTS_LATEST / "Model.pkl"
    if not model_path.exists():
        return JsonResponse({
            "ok": False,
            "error": "Chưa có model đã train. Vui lòng huấn luyện trước."
        }, status=400)

    try:
        df = pd.DataFrame(rows_data)

        # Convert numeric columns
        numeric_cols = [
            "temperature_current", "temperature_max", "temperature_min", "temperature_avg",
            "humidity_current", "humidity_max", "humidity_min", "humidity_avg",
            "pressure_current", "pressure_max", "pressure_min", "pressure_avg",
            "wind_speed_current", "wind_speed_max", "wind_speed_min", "wind_speed_avg",
            "wind_direction_current", "wind_direction_avg",
            "rain_current", "rain_max", "rain_min", "rain_avg",
            "cloud_cover_current", "cloud_cover_max", "cloud_cover_min", "cloud_cover_avg",
            "visibility_current", "visibility_max", "visibility_min", "visibility_avg",
            "thunder_probability", "latitude", "longitude",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        from Weather_Forcast_App.Machine_learning_model.interface.predictor import WeatherPredictor
        predictor = WeatherPredictor.from_artifacts(str(ML_ARTIFACTS_LATEST))

        result = predictor.predict(df)
        predictions = result["predictions"]

        # Build response
        response_rows = []
        for i, (_, row) in enumerate(df.iterrows()):
            pred_val = float(predictions[i]) if i < len(predictions) else 0
            r = {
                "station_name": str(row.get("station_name", "Thủ công")),
                "province": str(row.get("province", "")),
                "y_pred": round(pred_val, 4),
                "rain_status": "Mưa" if pred_val > 0.5 else "Không mưa",
            }
            response_rows.append(r)

        pred_arr = np.array([r["y_pred"] for r in response_rows])

        return JsonResponse({
            "ok": True,
            "predictions": response_rows,
            "stats": {
                "n_samples": len(response_rows),
                "mean": round(float(np.mean(pred_arr)), 4),
                "std": round(float(np.std(pred_arr)), 4),
                "min": round(float(np.min(pred_arr)), 4),
                "max": round(float(np.max(pred_arr)), 4),
                "prediction_time": round(result["prediction_time"], 3),
            }
        })

    except Exception as e:
        return JsonResponse({
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }, status=500)


@require_http_methods(["GET"])
def predict_model_info_view(request):
    """API: trả về thông tin model hiện tại."""
    info = _load_model_info()
    return JsonResponse({"ok": True, "model_info": info})
