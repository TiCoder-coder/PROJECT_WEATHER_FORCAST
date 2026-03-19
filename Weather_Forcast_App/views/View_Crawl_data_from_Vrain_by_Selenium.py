from pathlib import Path
from datetime import datetime

from django.http import JsonResponse, HttpResponseNotAllowed
from django.shortcuts import render

from Weather_Forcast_App.paths import DATA_CRAWL_DIR
from Weather_Forcast_App.crawl_queue import get_queue, JobStatus

# ============================================================
# CẤU HÌNH ĐƯỜNG DẪN
# ============================================================
APP_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = APP_ROOT / "scripts" / "Crawl_data_from_Vrain_by_Selenium.py"
OUTPUT_DIR = DATA_CRAWL_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_LABEL = "Vrain Selenium"


def _scan_latest_output():
    """Tìm file output mới nhất trong OUTPUT_DIR."""
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


# ============================================================
# crawl_vrain_selenium_view: VIEW GET RENDER UI
# ============================================================
def crawl_vrain_selenium_view(request):
    last_file, last_size_mb, last_time_from_file = _scan_latest_output()
    q = get_queue()
    job = q.latest_job_for(_LABEL)

    context = {
        "is_running": job.status == JobStatus.RUNNING if job else False,
        "last_returncode": job.returncode if job else None,
        "last_crawl_time": (job.finished_at if job else None) or last_time_from_file,
        "last_file": (job.last_file if job else None) or last_file,
        "last_size_mb": (job.last_size_mb if job else None) or last_size_mb,
        "current_job_id": job.job_id if job else None,
    }
    return render(request, "weather/HTML_Crawl_data_from_Vrain_by_Selenium.html", context)


# ============================================================
# crawl_vrain_selenium_start_view: ENDPOINT START JOB (POST)
# ============================================================
def crawl_vrain_selenium_start_view(request):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])

    q = get_queue()
    job = q.enqueue(
        script_path=SCRIPT_PATH,
        output_dir=OUTPUT_DIR,
        label=_LABEL,
    )
    pos = q.queue_position(job.job_id)
    msg = "Job đang chạy" if pos == 0 else f"Job đã vào hàng đợi (vị trí #{pos})"
    return JsonResponse({
        "ok": True,
        "job_id": job.job_id,
        "queue_position": pos,
        "status": job.status,
        "message": msg,
    })


# ============================================================
# crawl_vrain_selenium_tail_view: ENDPOINT POLLING LOG (GET)
# Selenium JS dùng param "offset" thay vì "since" — cả hai đều hỗ trợ.
# ============================================================
def crawl_vrain_selenium_tail_view(request):
    job_id = request.GET.get("job_id", "")
    # hỗ trợ cả "offset" (js cũ) lẫn "since" (js mới)
    since = int(request.GET.get("since", request.GET.get("offset", 0)) or 0)

    q = get_queue()
    if not job_id:
        job = q.latest_job_for(_LABEL)
        if job:
            job_id = job.job_id

    data = q.get_status_dict(job_id, since)
    data["ok"] = True
    # giữ compat với JS cũ dùng "offset" + "done"
    data["offset"] = data.get("next_since", since)
    data["done"] = not data.get("is_running", False) and not data.get("is_queued", False)
    return JsonResponse(data)