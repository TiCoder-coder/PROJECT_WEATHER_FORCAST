import os
import glob
from datetime import datetime

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from Weather_Forcast_App.paths import DATA_CRAWL_DIR, SCRIPT_CRAWL_BY_API
from Weather_Forcast_App.crawl_queue import get_queue, JobStatus

_LABEL = "OpenMeteo API"


def _scan_latest_output():
    """Tìm file output mới nhất trong DATA_CRAWL_DIR."""
    output_dir = str(DATA_CRAWL_DIR)
    try:
        files = []
        for p in ("*.xlsx", "*.csv"):
            files.extend(glob.glob(os.path.join(output_dir, p)))
        if not files:
            return None, None, None
        latest = max(files, key=os.path.getmtime)
        name = os.path.basename(latest)
        size_mb = round(os.path.getsize(latest) / (1024 * 1024), 2)
        mtime_str = datetime.fromtimestamp(os.path.getmtime(latest)).strftime("%Y-%m-%d %H:%M:%S")
        return name, size_mb, mtime_str
    except Exception:
        return None, None, None


# ============================================================
# crawl_api_weather_view: VIEW TRANG CRAWL THỜI TIẾT BẰNG API
# ============================================================
@require_http_methods(["GET", "POST"])
def crawl_api_weather_view(request):
    os.makedirs(str(DATA_CRAWL_DIR), exist_ok=True)
    is_ajax = request.headers.get("x-requested-with") == "XMLHttpRequest"

    # ---- GET: render giao diện ----
    if request.method == "GET":
        last_file, last_size_mb, last_time_from_file = _scan_latest_output()
        q = get_queue()
        job = q.latest_job_for(_LABEL)
        context = {
            "is_running": job.status == JobStatus.RUNNING if job else False,
            "last_returncode": job.returncode if job else None,
            "last_crawl_time": (job.finished_at if job else None) or last_time_from_file,
            "last_csv_name": (job.last_file if job else None) or last_file,
            "csv_size_mb": (job.last_size_mb if job else None) or last_size_mb,
            "last_csv_size_mb": (job.last_size_mb if job else None) or last_size_mb,
            "current_job_id": job.job_id if job else None,
        }
        return render(request, "weather/HTML_Crawl_data_by_API.html", context)

    # ---- POST: action=start ----
    action = request.POST.get("action", "").strip()
    if action != "start":
        if is_ajax:
            return JsonResponse({"ok": False, "error": "Invalid action"}, status=400)
        return render(request, "weather/HTML_Crawl_data_by_API.html", {})

    q = get_queue()
    job = q.enqueue(script_path=SCRIPT_CRAWL_BY_API, output_dir=DATA_CRAWL_DIR, label=_LABEL)
    pos = q.queue_position(job.job_id)

    if is_ajax:
        return JsonResponse({
            "ok": True,
            "job_id": job.job_id,
            "queue_position": pos,
            "is_running": job.status == JobStatus.RUNNING,
            "is_queued": job.status == JobStatus.QUEUED,
        })

    last_file, last_size_mb, last_time_from_file = _scan_latest_output()
    context = {
        "is_running": job.status == JobStatus.RUNNING,
        "last_returncode": None,
        "last_crawl_time": last_time_from_file,
        "last_csv_name": last_file,
        "csv_size_mb": last_size_mb,
        "last_csv_size_mb": last_size_mb,
        "current_job_id": job.job_id,
    }
    return render(request, "weather/HTML_Crawl_data_by_API.html", context)


# ============================================================
# api_weather_logs_view: ENDPOINT TRẢ LOGS/STATE CHO FRONTEND POLLING
# ============================================================
@require_http_methods(["GET"])
def api_weather_logs_view(request):
    job_id = request.GET.get("job_id", "")
    since = int(request.GET.get("since", 0) or 0)

    q = get_queue()
    if not job_id:
        job = q.latest_job_for(_LABEL)
        if job:
            job_id = job.job_id

    data = q.get_status_dict(job_id, since)

    # compat fields cho JS cũ
    job = q.get_job(job_id) if job_id else None
    data["last_csv_name"] = data.get("last_file")
    data["csv_size_mb"] = data.get("last_size_mb")
    data["last_started_at"] = job.started_at if job else None
    data["last_finished_at"] = data.get("last_crawl_time")

    return JsonResponse(data)

