import os
import sys
import threading
import subprocess
from pathlib import Path
from datetime import datetime

from django.http import JsonResponse, HttpResponseNotAllowed
from django.shortcuts import render


APP_ROOT = Path(__file__).resolve().parents[1]  # Weather_Forcast_App/
SCRIPT_PATH = APP_ROOT / "scripts" / "Crawl_data_from_Vrain_by_API.py"
OUTPUT_DIR = APP_ROOT / "output"


_STATE = {
    "is_running": False,
    "logs": [],
    "last_returncode": None,
    "last_crawl_time": None,
    "last_file": None,
    "last_size_mb": None,
}

_LOG_LIMIT = 2500


def _push_log(line: str):
    line = (line or "").rstrip("\n")
    if not line:
        return
    _STATE["logs"].append(line)
    if len(_STATE["logs"]) > _LOG_LIMIT:
        _STATE["logs"] = _STATE["logs"][-_LOG_LIMIT:]


def _scan_latest_output():
    """
    Tìm file output mới nhất trong output/.
    Ưu tiên các đuôi hay dùng: xlsx/csv/xls
    """
    if not OUTPUT_DIR.exists():
        return None, None, None

    exts = {".xlsx", ".csv", ".xls"}
    files = [p for p in OUTPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not files:
        return None, None, None

    latest = max(files, key=lambda p: p.stat().st_mtime)
    size_mb = round(latest.stat().st_size / (1024 * 1024), 2)
    mtime_str = datetime.fromtimestamp(latest.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return latest.name, size_mb, mtime_str


def _run_script_worker():
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        _push_log("========== START VRAIN API CRAWL ==========")
        _push_log(f"Script: {SCRIPT_PATH}")
        _push_log(f"Output dir: {OUTPUT_DIR}")

        if not SCRIPT_PATH.exists():
            _push_log("[ERROR] Script không tồn tại!")
            _STATE["last_returncode"] = -1
            return

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        proc = subprocess.Popen(
            [sys.executable, "-u", str(SCRIPT_PATH)],
            cwd=str(APP_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        for line in proc.stdout:
            _push_log(line)

        rc = proc.wait()
        _STATE["last_returncode"] = rc
        _STATE["last_crawl_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        last_file, last_size_mb, _ = _scan_latest_output()
        _STATE["last_file"] = last_file
        _STATE["last_size_mb"] = last_size_mb

        _push_log("========== DONE ==========")
        _push_log(f"Return code: {rc}")
        if last_file:
            _push_log(f"Latest output: {last_file} ({last_size_mb} MB)")
        else:
            _push_log("No output file detected in output/.")

    except Exception as e:
        _STATE["last_returncode"] = -1
        _push_log(f"[EXCEPTION] {repr(e)}")
    finally:
        _STATE["is_running"] = False


# ========= Views =========

def crawl_vrain_api_view(request):
    last_file, last_size_mb, last_time_from_file = _scan_latest_output()

    context = {
        "is_running": _STATE["is_running"],
        "last_returncode": _STATE["last_returncode"],
        "last_crawl_time": _STATE["last_crawl_time"] or last_time_from_file,
        "last_file": last_file or _STATE["last_file"],
        "last_size_mb": last_size_mb or _STATE["last_size_mb"],
    }
    return render(request, "weather/HTML_Crawl_data_from_Vrain_by_API.html", context)


def crawl_vrain_api_start_view(request):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])

    if _STATE["is_running"]:
        return JsonResponse({"ok": False, "message": "Job đang chạy rồi."}, status=409)

    _STATE["is_running"] = True
    _STATE["logs"] = []
    _STATE["last_returncode"] = None

    t = threading.Thread(target=_run_script_worker, daemon=True)
    t.start()

    return JsonResponse({"ok": True, "message": "Started"})


def crawl_vrain_api_tail_view(request):
    since = request.GET.get("since", "0")
    try:
        since_i = max(0, int(since))
    except:
        since_i = 0

    logs = _STATE["logs"]
    new_lines = logs[since_i:]

    return JsonResponse({
        "ok": True,
        "is_running": _STATE["is_running"],
        "next_since": len(logs),
        "lines": new_lines,
        "last_returncode": _STATE["last_returncode"],
        "last_crawl_time": _STATE["last_crawl_time"],
        "last_file": _STATE["last_file"],
        "last_size_mb": _STATE["last_size_mb"],
    })
