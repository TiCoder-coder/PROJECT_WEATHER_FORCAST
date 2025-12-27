import os
import sys
import glob
import threading
import subprocess
from datetime import datetime
from collections import deque

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods


_STATE = {
    "is_running": False,
    "logs": deque(maxlen=2000),
    "last_returncode": None,
    "last_started_at": None,
    "last_finished_at": None,
}

_STATE_LOCK = threading.Lock()


def _append_log(line: str):
    line = (line or "").rstrip("\n")
    if not line:
        return
    with _STATE_LOCK:
        _STATE["logs"].append(line)


def _get_latest_file(folder: str, patterns):
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(folder, p)))
    if not files:
        return None
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files[0]


def _run_script_background(script_path: str, output_dir: str, extra_args=None):
    extra_args = extra_args or []

    with _STATE_LOCK:
        _STATE["is_running"] = True
        _STATE["last_returncode"] = None
        _STATE["last_started_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _STATE["last_finished_at"] = None
        _STATE["logs"].clear()

    os.makedirs(output_dir, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CRAWL_MODE"] = "once"

    cmd = [sys.executable, "-u", script_path] + extra_args
    _append_log("[INFO] CMD = " + " ".join(cmd))

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(settings.BASE_DIR),
            env=env,
        )

        for line in proc.stdout:
            _append_log(line)

        rc = proc.wait()
        _append_log(f"[INFO] Script finished with returncode={rc}")

        with _STATE_LOCK:
            _STATE["last_returncode"] = rc
            _STATE["last_finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    except Exception as e:
        _append_log(f"[ERROR] Exception: {type(e).__name__}: {e}")

    finally:
        with _STATE_LOCK:
            _STATE["is_running"] = False


@require_http_methods(["GET", "POST"])
def crawl_api_weather_view(request):
    """
    Trang crawl thời tiết bằng API.
    - GET  : render giao diện + thông tin lần crawl gần nhất
    - POST : action=start -> chạy script nền (subprocess) và TRẢ JSON nếu là AJAX
    """
    script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "Crawl_data_by_API.py")
    script_path = os.path.abspath(script_path)

    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    def compute_last_output_info():
        """
        Trả về (last_file_name, last_file_size_mb, last_run_time_str)
        dựa trên file mới nhất trong output_dir.
        """
        try:
            patterns = ("*.xlsx", "*.csv")
            files = []
            for p in patterns:
                files.extend(glob.glob(os.path.join(output_dir, p)))

            if not files:
                return None, None, None

            latest = max(files, key=os.path.getmtime)
            last_file_name = os.path.basename(latest)
            size_mb = round(os.path.getsize(latest) / (1024 * 1024), 2)

            ts = datetime.fromtimestamp(os.path.getmtime(latest))
            last_run_time = ts.strftime("%Y-%m-%d %H:%M:%S")
            return last_file_name, size_mb, last_run_time
        except Exception:
            return None, None, None

    last_file_name, last_file_size_mb, last_run_time = compute_last_output_info()

    with _STATE_LOCK:
        is_running = _STATE["is_running"]
        logs_snapshot = list(_STATE["logs"])[-300:]

    context = {
        "is_running": is_running,
        "logs": logs_snapshot,
        "last_csv_name": last_file_name,
        "last_csv_size_mb": last_file_size_mb,
        "csv_size_mb": last_file_size_mb,
        "last_crawl_time": last_run_time,
    }

    is_ajax = request.headers.get("x-requested-with") == "XMLHttpRequest"

    # ---------------- GET ----------------
    if request.method == "GET":
        return render(request, "weather/HTML_Crawl_data_by_API.html", context)

    # ---------------- POST ----------------
    action = request.POST.get("action", "").strip()
    mode = request.POST.get("mode", "full").strip()
    verbose = request.POST.get("verbose") in ("on", "1", "true", "True")

    if action != "start":
        if is_ajax:
            return JsonResponse({"ok": False, "error": "Invalid action"}, status=400)
        _append_log(f"[WARN] Invalid action: {action}")
        return render(request, "weather/HTML_Crawl_data_by_API.html", context)

    with _STATE_LOCK:
        if _STATE["is_running"]:
            _append_log("[WARN] Job is already running. Ignored.")
            if is_ajax:
                return JsonResponse({"ok": False, "error": "Job is already running"}, status=409)
            context["is_running"] = True
            context["logs"] = list(_STATE["logs"])[-300:]
            return render(request, "weather/HTML_Crawl_data_by_API.html", context)

    if not os.path.exists(script_path):
        msg = f"[ERROR] Không tìm thấy script: {script_path}"
        _append_log(msg)
        if is_ajax:
            return JsonResponse({"ok": False, "error": msg}, status=404)
        context["logs"] = list(_STATE["logs"])[-300:]
        return render(request, "weather/HTML_Crawl_data_by_API.html", context)

    extra_args = []

    try:
        t = threading.Thread(
            target=_run_script_background,
            kwargs=dict(
                script_path=script_path,
                output_dir=output_dir,
                extra_args=extra_args,
            ),
            daemon=True,
        )
        t.start()
        _append_log(f"[INFO] Started background job (mode={mode}, verbose={verbose})")
    except Exception as e:
        _append_log(f"[ERROR] Không thể start job: {e}")
        if is_ajax:
            return JsonResponse({"ok": False, "error": str(e)}, status=500)
        context["logs"] = list(_STATE["logs"])[-300:]
        return render(request, "weather/HTML_Crawl_data_by_API.html", context)

    if is_ajax:
        with _STATE_LOCK:
            return JsonResponse(
                {
                    "ok": True,
                    "is_running": _STATE["is_running"],
                    "logs": list(_STATE["logs"])[-300:],
                }
            )

    with _STATE_LOCK:
        context["is_running"] = _STATE["is_running"]
        context["logs"] = list(_STATE["logs"])[-300:]
    return render(request, "weather/HTML_Crawl_data_by_API.html", context)

@require_http_methods(["GET"])
def api_weather_logs_view(request):
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))

    last_file_name = last_size_mb = last_run_time = None
    try:
        patterns = ("*.xlsx", "*.csv")
        files = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(output_dir, p)))
        if files:
            latest = max(files, key=os.path.getmtime)
            last_file_name = os.path.basename(latest)
            last_size_mb = round(os.path.getsize(latest) / (1024 * 1024), 2)
            last_run_time = datetime.fromtimestamp(os.path.getmtime(latest)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass

    with _STATE_LOCK:
        data = {
            "is_running": _STATE["is_running"],
            "logs": list(_STATE["logs"])[-300:],
            "last_returncode": _STATE["last_returncode"],
            "last_started_at": _STATE["last_started_at"],
            "last_finished_at": _STATE["last_finished_at"],
            "last_csv_name": last_file_name,
            "csv_size_mb": last_size_mb,
            "last_crawl_time": last_run_time,
        }
    return JsonResponse(data)

@require_http_methods(["GET"])
def crawl_vrain_html_view(request):
    return render(request, "weather/coming_soon_vrain_html.html")