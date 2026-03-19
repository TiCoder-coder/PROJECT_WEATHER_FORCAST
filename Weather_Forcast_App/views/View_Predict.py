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
from datetime import datetime, timedelta
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
    ML_MODEL_ROOT, ML_ARTIFACTS_LATEST, ML_STACKING_ARTIFACTS,
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


def _aggregate_predictions_by_location(df: pd.DataFrame, cols_to_show: List[str]) -> List[Dict[str, Any]]:
    """Nhóm kết quả dự báo theo địa điểm (station_name + province + district).

    Mỗi địa điểm chỉ hiển thị 1 dòng tổng hợp:
      - y_pred: giá trị trung bình
      - status: trạng thái phổ biến nhất (mode)
      - forecast_for: thời gian dự báo mới nhất
      - rain_total: giá trị trung bình (nếu có)
      - n_rows: số dòng gốc
    """
    group_cols = [c for c in ["station_name", "province", "district"] if c in df.columns]
    if not group_cols:
        # Không có cột location → trả toàn bộ rows (không group được)
        rows = []
        for _, row in df.iterrows():
            r = {}
            for c in cols_to_show:
                val = row.get(c)
                if isinstance(val, float):
                    r[c] = round(val, 4) if pd.notna(val) else ""
                else:
                    r[c] = str(val) if pd.notna(val) else ""
            rows.append(r)
        return rows

    agg_rows = []
    for group_key, group_df in df.groupby(group_cols, sort=True):
        r: Dict[str, Any] = {}
        # Đặt các cột group
        if isinstance(group_key, str):
            group_key = (group_key,)
        for i, gc in enumerate(group_cols):
            r[gc] = str(group_key[i]) if pd.notna(group_key[i]) else ""

        # Aggregate các cột metric
        for c in cols_to_show:
            if c in group_cols:
                continue
            if c == "y_pred":
                vals = group_df[c].dropna()
                r[c] = round(float(vals.mean()), 4) if len(vals) > 0 else ""
            elif c == "rain_total" and c in group_df.columns:
                vals = pd.to_numeric(group_df[c], errors="coerce").dropna()
                r[c] = round(float(vals.mean()), 4) if len(vals) > 0 else ""
            elif c == "status" and c in group_df.columns:
                r[c] = group_df[c].mode().iloc[0] if len(group_df[c].mode()) > 0 else ""
            elif c == "forecast_for" and c in group_df.columns:
                r[c] = str(group_df[c].iloc[-1]) if len(group_df) > 0 else ""
            elif c in group_df.columns:
                r[c] = str(group_df[c].iloc[-1]) if len(group_df) > 0 and pd.notna(group_df[c].iloc[-1]) else ""

        # Thêm số dòng gốc
        r["n_rows"] = len(group_df)
        agg_rows.append(r)

    return agg_rows


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
    """Load thông tin cả hai model (stacking + ensemble)."""
    info: Dict[str, Any] = {
        "loaded": False,
        "model_type": "",
        "target_column": "",
        "n_features": 0,
        "trained_at": "",
        "model_size_mb": "0",
        "stacking_available": False,
        "ensemble_available": False,
    }

    # --- Stacking model ---
    stacking_model_path = ML_STACKING_ARTIFACTS / "Model.pkl"
    if stacking_model_path.exists():
        info["stacking_available"] = True
        info["loaded"] = True
        info["stacking_size_mb"] = f"{stacking_model_path.stat().st_size / 1024 / 1024:.2f}"
        stacking_ti = ML_STACKING_ARTIFACTS / "Train_info.json"
        if stacking_ti.exists():
            try:
                ti = json.loads(stacking_ti.read_text(encoding="utf-8"))
                info["stacking_trained_at"] = ti.get("trained_at", "")
                info["stacking_model_type"] = ti.get("model_type", "stacking_ensemble")
            except Exception:
                pass
        stacking_metrics = ML_STACKING_ARTIFACTS / "Metrics.json"
        if stacking_metrics.exists():
            try:
                m = json.loads(stacking_metrics.read_text(encoding="utf-8"))
                info["stacking_test_r2"] = m.get("test", {}).get("R2")
                info["stacking_test_rmse"] = m.get("test", {}).get("RMSE")
                info["stacking_test_rain_acc"] = m.get("test", {}).get("Rain_Detection_Accuracy")
            except Exception:
                pass

    # --- Ensemble Average model ---
    ensemble_model_path = ML_ARTIFACTS_LATEST / "Model.pkl"
    if ensemble_model_path.exists():
        info["ensemble_available"] = True
        info["loaded"] = True
        info["model_size_mb"] = f"{ensemble_model_path.stat().st_size / 1024 / 1024:.2f}"
        info["model_exists"] = True

    train_info_path = ML_ARTIFACTS_LATEST / "Train_info.json"
    if train_info_path.exists():
        try:
            ti = json.loads(train_info_path.read_text(encoding="utf-8"))
            info["trained_at"] = ti.get("trained_at", "")
            info["model_type"] = ti.get("model_type", ti.get("model", {}).get("type", ""))
            info["target_column"] = ti.get("target_column", "rain_total")
            info["forecast_horizon"] = ti.get("forecast_horizon", 0)
            info["predict_threshold"] = (
                ti.get("stacking_config", {}).get("predict_threshold")
                or ti.get("model", {}).get("params", {}).get("predict_threshold", 0.5)
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

    return info


def _load_recent_predictions() -> List[Dict[str, Any]]:
    """Load toàn bộ kết quả dự báo (nếu có), nhóm theo địa điểm."""
    pred_path = ML_MODEL_ROOT / "WeatherForcast" / "predictions.csv"
    if not pred_path.exists():
        return []
    try:
        df = pd.read_csv(pred_path)

        # Normalize column names — support both old and new naming
        rename_map = {}
        if "location_station_name" in df.columns and "station_name" not in df.columns:
            rename_map["location_station_name"] = "station_name"
        if "location_province" in df.columns and "province" not in df.columns:
            rename_map["location_province"] = "province"
        if "location_district" in df.columns and "district" not in df.columns:
            rename_map["location_district"] = "district"
        if "timestamp" in df.columns and "forecast_for" not in df.columns:
            rename_map["timestamp"] = "forecast_for"
        if rename_map:
            df = df.rename(columns=rename_map)

        cols_to_show = ["station_name", "province", "district", "rain_total", "status", "y_pred", "n_rows", "forecast_for"]
        cols_to_show = [c for c in cols_to_show if c in df.columns or c == "n_rows"]

        agg_rows = _aggregate_predictions_by_location(df, cols_to_show)

        # Đổi key forecast_for → timestamp cho recent predictions template
        for r in agg_rows:
            if "forecast_for" in r:
                r["timestamp"] = r.pop("forecast_for")
        return agg_rows
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
# Predictor loader helper
# ────────────────────────────────────────────────────────────────
def _select_predictor(model_type: str = "auto"):
    """Load và trả về predictor dựa trên model_type.
    - 'stacking'        → WeatherPredictorStacking (bắt buộc dùng stacking)
    - 'ensemble_average'→ WeatherPredictor (bắt buộc dùng ensemble average)
    - 'auto' (default)  → thử stacking trước, fallback ensemble average
    """
    if model_type == "stacking":
        from Weather_Forcast_App.Machine_learning_model.interface.predictor_by_stacking_ensemble import WeatherPredictorStacking
        return WeatherPredictorStacking.from_artifacts(str(ML_STACKING_ARTIFACTS))
    elif model_type == "ensemble_average":
        from Weather_Forcast_App.Machine_learning_model.interface.predictor_by_ensemble_average import WeatherPredictor
        return WeatherPredictor.from_artifacts(str(ML_ARTIFACTS_LATEST))
    else:  # auto
        try:
            from Weather_Forcast_App.Machine_learning_model.interface.predictor_by_stacking_ensemble import WeatherPredictorStacking
            return WeatherPredictorStacking.from_artifacts(str(ML_STACKING_ARTIFACTS))
        except Exception:
            from Weather_Forcast_App.Machine_learning_model.interface.predictor_by_ensemble_average import WeatherPredictor
            return WeatherPredictor.from_artifacts(str(ML_ARTIFACTS_LATEST))


def _model_available(model_type: str = "auto") -> bool:
    """Kiểm tra model có sẵn cho model_type."""
    if model_type == "stacking":
        return (ML_STACKING_ARTIFACTS / "Model.pkl").exists()
    elif model_type == "ensemble_average":
        return (ML_ARTIFACTS_LATEST / "Model.pkl").exists()
    else:
        return (ML_STACKING_ARTIFACTS / "Model.pkl").exists() or (ML_ARTIFACTS_LATEST / "Model.pkl").exists()


# ────────────────────────────────────────────────────────────────
# Background worker: chạy prediction
# ────────────────────────────────────────────────────────────────
def _prediction_worker(job_id: str, file_path: str, nrows: Optional[int] = None, model_type: str = "auto") -> None:
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

        # Rename cột cho khớp với training features (nếu data từ crawler)
        _rename_map = {
            'station_id': 'location_station_id',
            'station_name': 'location_station_name',
            'province': 'location_province',
            'district': 'location_district',
            'latitude': 'location_latitude',
            'longitude': 'location_longitude',
        }
        df = df.rename(columns={old: new for old, new in _rename_map.items()
                                if old in df.columns and new not in df.columns})

        _set_progress(job_id, 20, "Load model")

        # Load predictor dựa theo model_type
        predictor = _select_predictor(model_type)

        _push(job_id, f"🤖 Model: {type(predictor.model).__name__}")
        _push(job_id, f"🎯 Target: {predictor.target_column}")
        _push(job_id, f"📐 Features: {len(predictor.feature_columns)}")
        _set_progress(job_id, 40, "Đang build features & dự báo...")

        # Lấy forecast_horizon từ model
        forecast_horizon = predictor.train_info.get("forecast_horizon", 0)
        if forecast_horizon > 0:
            _push(job_id, f"🔮 Forecast mode: dự báo trước {forecast_horizon} bước")

        # Tạm restore stdout trước khi predict (để log nội bộ không trộn vào nhật ký)
        sys.stdout = old_stdout

        # Chạy prediction
        try:
            result = predictor.predict(df)
            predictions = result["predictions"]
            pred_time = result["prediction_time"]
        except Exception as e:
            _push(job_id, f"❌ Lỗi khi dự báo: {str(e)}")
            _push(job_id, traceback.format_exc())
            raise

        _push(job_id, f"✅ Dự báo xong! {len(predictions)} kết quả trong {pred_time:.2f}s")
        _set_progress(job_id, 80, "Lưu kết quả")

        # Gắn y_pred vào DataFrame gốc
        df["y_pred"] = predictions

        # Cập nhật timestamp → thời điểm dự báo (tương lai nếu có forecast_horizon)
        now = datetime.now()
        if forecast_horizon > 0:
            forecast_dt = now + timedelta(hours=forecast_horizon)
            forecast_str = forecast_dt.strftime("%Y-%m-%d %H:%M:%S")
            df["forecast_for"] = forecast_str
            _push(job_id, f"🔮 Dự báo cho: {forecast_str} (hiện tại + {forecast_horizon}h)")
        else:
            df["forecast_for"] = now.strftime("%Y-%m-%d %H:%M:%S")

        # Rename location_ columns back to user-friendly names
        _rename_back = {
            'location_station_id': 'station_id',
            'location_station_name': 'station_name',
            'location_province': 'province',
            'location_district': 'district',
            'location_latitude': 'latitude',
            'location_longitude': 'longitude',
        }
        df = df.rename(columns={old: new for old, new in _rename_back.items()
                                if old in df.columns and new not in df.columns})

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
            mask = df[target_col].notna()
            y_true = df.loc[mask, target_col].values
            y_pred_aligned = pred_arr[mask.values]
            if len(y_true) > 0:
                rmse = np.sqrt(np.mean((y_true - y_pred_aligned) ** 2))
                _push(job_id, f"📊 RMSE (so với actual): {rmse:.4f}")

        # Lấy rows cho preview — nhóm theo địa điểm để không trùng lặp
        cols_to_show = []
        for c in ["station_name", "province", "district", "rain_total", "status", "y_pred", "n_rows", "forecast_for"]:
            if c in df.columns or c == "n_rows":
                cols_to_show.append(c)
        if not cols_to_show:
            cols_to_show = list(df.columns[:7])

        preview_rows = _aggregate_predictions_by_location(df, cols_to_show)
        _push(job_id, f"📍 Nhóm theo địa điểm: {len(preview_rows)} vị trí (từ {len(df)} dòng gốc)")

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
                "forecast_horizon_hours": forecast_horizon,
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
        "stacking_available": model_info.get("stacking_available", False),
        "ensemble_available": model_info.get("ensemble_available", False),
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
    model_type = request.POST.get("model_type", "auto")
    if not _model_available(model_type):
        model_label = {"stacking": "Stacking Ensemble", "ensemble_average": "Ensemble Average"}.get(model_type, "bất kỳ model")
        return JsonResponse({
            "ok": False,
            "error": f"Chưa có model {model_label} đã train. Vui lòng huấn luyện trước."
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

    t = threading.Thread(target=_prediction_worker, args=(job_id, file_path, nrows, model_type), daemon=True)
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
    model_type = body.get("model_type", "auto")
    if not _model_available(model_type):
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

        # Load predictor dựa theo model_type
        predictor = _select_predictor(model_type)

        result = predictor.predict(df)
        predictions = result["predictions"]
        forecast_horizon = result.get("forecast_horizon", 0)

        # Tính thời điểm dự báo (tương lai)
        now = datetime.now()
        forecast_dt = now + timedelta(hours=forecast_horizon) if forecast_horizon > 0 else now
        forecast_str = forecast_dt.strftime("%Y-%m-%d %H:%M:%S")

        # Dùng has_rain từ model (stacking) nếu có; fallback về ngưỡng 0.5mm
        has_rain_arr = result.get("has_rain")

        # Build response
        response_rows = []
        for i, (_, row) in enumerate(df.iterrows()):
            pred_val = float(predictions[i]) if i < len(predictions) else 0
            if has_rain_arr is not None:
                is_rain = bool(has_rain_arr[i]) if i < len(has_rain_arr) else (pred_val > 0.5)
            else:
                is_rain = pred_val > 0.5
            r = {
                "station_name": str(row.get("station_name", "Thủ công")),
                "province": str(row.get("province", "")),
                "y_pred": round(pred_val, 4),
                "rain_status": "Mưa" if is_rain else "Không mưa",
                "forecast_for": forecast_str,
            }
            response_rows.append(r)

        pred_arr = np.array([r["y_pred"] for r in response_rows])

        return JsonResponse({
            "ok": True,
            "predictions": response_rows,
            "forecast_info": {
                "forecast_horizon_hours": forecast_horizon,
                "data_at": now.strftime("%Y-%m-%d %H:%M:%S"),
                "forecast_for": forecast_str,
                "description": f"Dự báo lượng mưa sau {forecast_horizon} giờ tới" if forecast_horizon > 0 else "Dự báo tại thời điểm hiện tại",
            },
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


# ────────────────────────────────────────────────────────────────
# FORECAST NOW: Crawl dữ liệu mới → Predict ngay lập tức
# ────────────────────────────────────────────────────────────────

def _forecast_now_worker(job_id: str, model_type: str = "auto") -> None:
    """Crawl fresh data từ API → chạy prediction ngay."""
    old_stdout = sys.stdout
    try:
        _push(job_id, "🌐 Bắt đầu thu thập dữ liệu thời tiết mới nhất...")
        _set_progress(job_id, 5, "Khởi tạo crawler")

        sys.stdout = _LogCapture(job_id)

        # Import crawler & locations
        from Weather_Forcast_App.scripts.Crawl_data_by_API import (
            VietnamWeatherDataCrawler,
            vietnam_locations,
        )

        crawler = VietnamWeatherDataCrawler()
        locations = vietnam_locations

        _push(job_id, f"📍 Tổng số trạm: {len(locations)}")
        _set_progress(job_id, 10, "Đang crawl dữ liệu...")

        # Crawl với delay nhỏ hơn để nhanh hơn
        weather_data = crawler.crawl_all_locations(locations, delay=0.5)

        if not weather_data:
            _push(job_id, "❌ Không crawl được dữ liệu nào!")
            _set_progress(job_id, 100, "Lỗi")
            with _LOCK:
                _JOBS[job_id]["status"] = "error"
                _JOBS[job_id]["result"] = "Không crawl được dữ liệu"
            return

        _push(job_id, f"✅ Crawl xong: {len(weather_data)} trạm")
        _set_progress(job_id, 50, "Chuẩn bị dữ liệu")

        # Tạo DataFrame từ dữ liệu vừa crawl
        df = pd.DataFrame(weather_data)

        # Load predictor dựa theo model_type
        predictor = _select_predictor(model_type)
        forecast_horizon = predictor.train_info.get("forecast_horizon", 0)
        now = datetime.now()

        # Rename cột cho khớp với training features
        _rename_map = {
            'station_id': 'location_station_id',
            'station_name': 'location_station_name',
            'province': 'location_province',
            'district': 'location_district',
            'latitude': 'location_latitude',
            'longitude': 'location_longitude',
        }
        df = df.rename(columns={old: new for old, new in _rename_map.items()
                                if old in df.columns and new not in df.columns})

        # GIỮ NGUYÊN timestamp = thời điểm hiện tại cho feature builder
        # (model train với features tại t → predict target tại t + forecast_horizon)
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")
        df["timestamp"] = now_str

        _push(job_id, f"📊 DataFrame: {len(df)} dòng, {len(df.columns)} cột")
        _push(job_id, f"🕒 Dữ liệu thời tiết tại: {now_str}")
        _set_progress(job_id, 60, "Load model")

        _push(job_id, f"🤖 Model: {type(predictor.model).__name__}")
        _push(job_id, f"🎯 Target: {predictor.target_column}")
        _push(job_id, f"📐 Features: {len(predictor.feature_columns)}")
        _set_progress(job_id, 70, "Đang build features & dự báo...")

        # Tạm restore stdout trước khi predict (để log nội bộ không trộn vào nhật ký)
        sys.stdout = old_stdout

        # Chạy prediction (features tại thời điểm hiện tại → dự báo tương lai)
        try:
            result = predictor.predict(df)
            predictions = result["predictions"]
            pred_time = result["prediction_time"]
        except Exception as e:
            _push(job_id, f"❌ Lỗi khi dự báo: {str(e)}")
            _push(job_id, traceback.format_exc())
            raise

        # Sau khi predict xong, cập nhật timestamp → thời điểm dự báo (tương lai)
        forecast_dt = now + timedelta(hours=forecast_horizon) if forecast_horizon > 0 else now
        forecast_str = forecast_dt.strftime("%Y-%m-%d %H:%M:%S")

        _push(job_id, f"✅ Dự báo xong! {len(predictions)} kết quả trong {pred_time:.2f}s")
        if forecast_horizon > 0:
            _push(job_id, f"🔮 Kết quả dự báo cho: {forecast_str} (hiện tại + {forecast_horizon}h)")
        _set_progress(job_id, 85, "Lưu kết quả")

        # Gắn y_pred vào DataFrame
        df["y_pred"] = predictions
        df["timestamp"] = forecast_str  # Hiển thị thời điểm tương lai trong output
        df["forecast_for"] = forecast_str  # Thời điểm dự báo
        df["data_collected_at"] = now_str  # Thời điểm thu thập dữ liệu

        # Thêm cột status — dùng has_rain từ model (stacking) nếu có; fallback ngưỡng 0.5mm
        has_rain_arr = result.get("has_rain")
        if has_rain_arr is not None:
            df["status"] = np.where(np.asarray(has_rain_arr, dtype=bool), "Mưa", "Không mưa")
        else:
            df["status"] = np.where(np.array(predictions) > 0.5, "Mưa", "Không mưa")

        # Rename location_ columns back to user-friendly names for CSV & display
        _rename_back = {
            'location_station_id': 'station_id',
            'location_station_name': 'station_name',
            'location_province': 'province',
            'location_district': 'district',
            'location_latitude': 'latitude',
            'location_longitude': 'longitude',
        }
        df = df.rename(columns={old: new for old, new in _rename_back.items()
                                if old in df.columns and new not in df.columns})

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

        # Lấy rows cho preview — nhóm theo địa điểm để không trùng lặp
        cols_to_show = []
        for c in ["station_name", "province", "district", "rain_total", "status", "y_pred", "n_rows", "forecast_for"]:
            if c in df.columns or c == "n_rows":
                cols_to_show.append(c)
        if not cols_to_show:
            cols_to_show = list(df.columns[:7])

        preview_rows = _aggregate_predictions_by_location(df, cols_to_show)
        _push(job_id, f"📍 Nhóm theo địa điểm: {len(preview_rows)} vị trí (từ {len(df)} dòng gốc)")

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
                "forecast_horizon_hours": forecast_horizon,
                "data_collected_at": now_str,
                "forecast_for": forecast_str,
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


@csrf_exempt
@require_http_methods(["POST"])
def predict_forecast_now_view(request):
    """Crawl dữ liệu mới nhất từ API → chạy prediction ngay lập tức."""

    # Kiểm tra job đang chạy
    with _LOCK:
        for jid, jdata in _JOBS.items():
            if jdata.get("status") == "running":
                return JsonResponse({
                    "ok": False,
                    "error": f"Đang có job chạy (ID: {jid}). Vui lòng chờ."
                }, status=409)

    # Kiểm tra model
    try:
        body = json.loads(request.body.decode("utf-8")) if request.body else {}
    except Exception:
        body = {}
    model_type = body.get("model_type", "auto")

    if not _model_available(model_type):
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
            "file_path": "(crawl mới)",
            "result": None,
            "preview": [],
            "preview_columns": [],
            "stats": {},
        }

    t = threading.Thread(target=_forecast_now_worker, args=(job_id, model_type), daemon=True)
    t.start()

    return JsonResponse({"ok": True, "job_id": job_id})
