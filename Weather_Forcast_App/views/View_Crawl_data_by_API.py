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


# ============================================================
# _STATE: TRẠNG THÁI CHẠY JOB CRAWL (IN-MEMORY)
# ============================================================
# Đây là "global state" lưu trong RAM của process Django để:
# - biết job hiện có đang chạy không (is_running)
# - lưu log realtime để frontend polling
# - lưu returncode lần chạy gần nhất + thời gian bắt đầu/kết thúc
#
# Giải thích từng field:
# - is_running:
#   + True  => đang có job chạy nền
#   + False => không có job nào chạy
#
# - logs:
#   + deque(maxlen=2000): hàng đợi 2 chiều, tự động giới hạn 2000 dòng cuối
#   + ưu điểm: append nhanh, tự cắt bớt log cũ -> tránh tràn RAM
#
# - last_returncode:
#   + return code của subprocess lần chạy gần nhất
#   + thường: 0 là OK, khác 0 là lỗi
#
# - last_started_at / last_finished_at:
#   + timestamp string để UI hiển thị "lần chạy gần nhất"
_STATE = {
    "is_running": False,
    "logs": deque(maxlen=2000),
    "last_returncode": None,
    "last_started_at": None,
    "last_finished_at": None,
}

# ============================================================
# _STATE_LOCK: LOCK ĐỒNG BỘ THREAD
# ============================================================
# Vì:
# - Django view thread (request) có thể đọc/ghi _STATE
# - background thread (_run_script_background) cũng đọc/ghi _STATE
# => cần Lock để tránh race condition (đọc/ghi chồng nhau, logs bị lỗi)
_STATE_LOCK = threading.Lock()


# ============================================================
# _append_log(line): THÊM 1 DÒNG LOG VÀO _STATE["logs"]
# ============================================================
# - strip newline
# - bỏ dòng rỗng
# - dùng lock để thread-safe
def _append_log(line: str):
    line = (line or "").rstrip("\n")
    if not line:
        return
    with _STATE_LOCK:
        _STATE["logs"].append(line)


# ============================================================
# _get_latest_file(folder, patterns): LẤY FILE MỚI NHẤT THEO PATTERN
# ============================================================
# Mục đích:
# - tìm file output mới nhất trong folder (dựa vào mtime)
# - patterns: danh sách pattern glob, ví dụ: ("*.xlsx","*.csv")
#
# Cách làm:
# - glob tất cả file match pattern
# - sort theo os.path.getmtime (mtime) giảm dần
# - trả về file đầu tiên (mới nhất)
def _get_latest_file(folder: str, patterns):
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(folder, p)))
    if not files:
        return None
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files[0]


# ============================================================
# _run_script_background(script_path, output_dir, extra_args=None):
# CHẠY SCRIPT CRAWL BẰNG SUBPROCESS Ở BACKGROUND THREAD
# ============================================================
# Vai trò:
# - được gọi trong thread daemon để không block request
# - spawn subprocess chạy file python (Crawl_data_by_API.py)
# - stream stdout/stderr theo dòng và đưa vào _STATE["logs"]
#
# Tham số:
# - script_path: đường dẫn tuyệt đối tới script python cần chạy
# - output_dir: thư mục output (tạo nếu chưa có)
# - extra_args: list args bổ sung cho script (nếu có)
def _run_script_background(script_path: str, output_dir: str, extra_args=None):
    extra_args = extra_args or []  # nếu None thì thay bằng list rỗng

    # ------------------------------------------------------------
    # Reset state khi bắt đầu job
    # ------------------------------------------------------------
    with _STATE_LOCK:
        _STATE["is_running"] = True
        _STATE["last_returncode"] = None
        _STATE["last_started_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _STATE["last_finished_at"] = None
        _STATE["logs"].clear()  # clear logs cũ để UI hiển thị log của job mới

    # đảm bảo output_dir tồn tại
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Chuẩn bị environment cho subprocess
    # ------------------------------------------------------------
    env = os.environ.copy()

    # PYTHONUNBUFFERED=1:
    # - ép Python subprocess flush stdout/stderr ngay lập tức
    # - giúp log realtime (không bị delay do buffer)
    env["PYTHONUNBUFFERED"] = "1"

    # CRAWL_MODE="once":
    # - biến môi trường tuỳ bạn dùng trong script để chạy 1 lần rồi thoát
    env["CRAWL_MODE"] = "once"

    # ------------------------------------------------------------
    # Build command chạy script
    # ------------------------------------------------------------
    # sys.executable:
    # - đường dẫn python hiện tại (đúng venv đang chạy Django)
    # "-u":
    # - unbuffered mode (cũng giúp log realtime)
    cmd = [sys.executable, "-u", script_path] + extra_args

    # log command để debug
    _append_log("[INFO] CMD = " + " ".join(cmd))

    try:
        # --------------------------------------------------------
        # subprocess.Popen: chạy script nền
        # --------------------------------------------------------
        # stdout=PIPE:
        # - lấy stdout để đọc từng dòng log
        #
        # stderr=STDOUT:
        # - gộp stderr vào stdout => đọc 1 stream duy nhất
        #
        # text=True:
        # - đọc dạng string (không phải bytes)
        #
        # bufsize=1:
        # - line-buffered (kết hợp với -u và PYTHONUNBUFFERED)
        #
        # cwd=str(settings.BASE_DIR):
        # - đặt working directory cho script
        # - đảm bảo script dùng relative path sẽ đúng theo BASE_DIR
        #
        # env=env:
        # - truyền env đã chỉnh
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(settings.BASE_DIR),
            env=env,
        )

        # --------------------------------------------------------
        # Đọc stdout theo dòng (realtime)
        # --------------------------------------------------------
        for line in proc.stdout:
            _append_log(line)

        # wait process kết thúc và lấy returncode
        rc = proc.wait()
        _append_log(f"[INFO] Script finished with returncode={rc}")

        # cập nhật state kết thúc
        with _STATE_LOCK:
            _STATE["last_returncode"] = rc
            _STATE["last_finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    except Exception as e:
        # nếu xảy ra exception khi chạy subprocess
        _append_log(f"[ERROR] Exception: {type(e).__name__}: {e}")

    finally:
        # đảm bảo trạng thái is_running luôn được set False khi xong
        with _STATE_LOCK:
            _STATE["is_running"] = False


# ============================================================
# crawl_api_weather_view: VIEW TRANG CRAWL THỜI TIẾT BẰNG API
# ============================================================
# Method hỗ trợ: GET, POST
#
# - GET:
#   + render template HTML_Crawl_data_by_API.html
#   + kèm context: is_running, logs snapshot, last file info...
#
# - POST:
#   + nếu action=start => spawn thread chạy _run_script_background
#   + nếu request là AJAX => trả JSON (để frontend update realtime)
@require_http_methods(["GET", "POST"])
def crawl_api_weather_view(request):
    """
    Trang crawl thời tiết bằng API.
    - GET  : render giao diện + thông tin lần crawl gần nhất
    - POST : action=start -> chạy script nền (subprocess) và TRẢ JSON nếu là AJAX
    """
    # ------------------------------------------------------------
    # Xác định đường dẫn script crawl
    # ------------------------------------------------------------
    # __file__ ở đây là file views.py (hoặc file chứa code này)
    # ".." đi lên 1 thư mục, rồi vào scripts/Crawl_data_by_API.py
    script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "Crawl_data_by_API.py")
    script_path = os.path.abspath(script_path)

    # ------------------------------------------------------------
    # Xác định output_dir cho script
    # ------------------------------------------------------------
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)  # đảm bảo thư mục tồn tại

    # ------------------------------------------------------------
    # compute_last_output_info(): lấy thông tin file output mới nhất
    # ------------------------------------------------------------
    # Trả về:
    # - last_file_name: tên file mới nhất
    # - last_file_size_mb: dung lượng MB
    # - last_run_time: thời gian mtime (được coi như thời gian crawl gần nhất)
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

            # latest: file có mtime lớn nhất => file mới nhất
            latest = max(files, key=os.path.getmtime)
            last_file_name = os.path.basename(latest)
            size_mb = round(os.path.getsize(latest) / (1024 * 1024), 2)

            ts = datetime.fromtimestamp(os.path.getmtime(latest))
            last_run_time = ts.strftime("%Y-%m-%d %H:%M:%S")
            return last_file_name, size_mb, last_run_time
        except Exception:
            # nếu có lỗi bất kỳ -> trả None cho an toàn
            return None, None, None

    # lấy info output mới nhất để hiển thị lên UI
    last_file_name, last_file_size_mb, last_run_time = compute_last_output_info()

    # ------------------------------------------------------------
    # Snapshot trạng thái chạy + logs để render UI
    # ------------------------------------------------------------
    with _STATE_LOCK:
        is_running = _STATE["is_running"]
        logs_snapshot = list(_STATE["logs"])[-300:]  # chỉ lấy 300 dòng cuối cho nhẹ

    # context truyền qua template
    context = {
        "is_running": is_running,
        "logs": logs_snapshot,
        "last_csv_name": last_file_name,
        "last_csv_size_mb": last_file_size_mb,
        # csv_size_mb có vẻ dùng cho JS update UI (giữ lại đồng bộ template)
        "csv_size_mb": last_file_size_mb,
        "last_crawl_time": last_run_time,
    }

    # ------------------------------------------------------------
    # Detect AJAX request
    # ------------------------------------------------------------
    # Frontend fetch thường set header:
    # "X-Requested-With": "XMLHttpRequest"
    is_ajax = request.headers.get("x-requested-with") == "XMLHttpRequest"

    # ---------------- GET ----------------
    if request.method == "GET":
        # render trang UI crawl
        return render(request, "weather/HTML_Crawl_data_by_API.html", context)

    # ---------------- POST ----------------
    # action: người dùng gửi action=start để bắt đầu
    action = request.POST.get("action", "").strip()

    # mode/verbose: biến bạn chuẩn bị sẵn (hiện tại chưa đưa vào extra_args)
    mode = request.POST.get("mode", "full").strip()
    verbose = request.POST.get("verbose") in ("on", "1", "true", "True")

    # Nếu action != start => reject
    if action != "start":
        if is_ajax:
            return JsonResponse({"ok": False, "error": "Invalid action"}, status=400)
        _append_log(f"[WARN] Invalid action: {action}")
        return render(request, "weather/HTML_Crawl_data_by_API.html", context)

    # ------------------------------------------------------------
    # Check nếu job đang chạy -> không cho start job mới
    # ------------------------------------------------------------
    with _STATE_LOCK:
        if _STATE["is_running"]:
            _append_log("[WARN] Job is already running. Ignored.")
            if is_ajax:
                return JsonResponse({"ok": False, "error": "Job is already running"}, status=409)
            context["is_running"] = True
            context["logs"] = list(_STATE["logs"])[-300:]
            return render(request, "weather/HTML_Crawl_data_by_API.html", context)

    # ------------------------------------------------------------
    # Check script tồn tại
    # ------------------------------------------------------------
    if not os.path.exists(script_path):
        msg = f"[ERROR] Không tìm thấy script: {script_path}"
        _append_log(msg)
        if is_ajax:
            return JsonResponse({"ok": False, "error": msg}, status=404)
        context["logs"] = list(_STATE["logs"])[-300:]
        return render(request, "weather/HTML_Crawl_data_by_API.html", context)

    # extra_args:
    # - list args truyền vào script (hiện đang rỗng)
    # - bạn có thể map mode/verbose vào extra_args trong tương lai
    # (nhưng bạn yêu cầu không đổi code)
    extra_args = []

    # ------------------------------------------------------------
    # Start background thread chạy subprocess
    # ------------------------------------------------------------
    try:
        t = threading.Thread(
            target=_run_script_background,
            kwargs=dict(
                script_path=script_path,
                output_dir=output_dir,
                extra_args=extra_args,
            ),
            daemon=True,  # daemon để không giữ process khi server shutdown
        )
        t.start()
        _append_log(f"[INFO] Started background job (mode={mode}, verbose={verbose})")
    except Exception as e:
        _append_log(f"[ERROR] Không thể start job: {e}")
        if is_ajax:
            return JsonResponse({"ok": False, "error": str(e)}, status=500)
        context["logs"] = list(_STATE["logs"])[-300:]
        return render(request, "weather/HTML_Crawl_data_by_API.html", context)

    # ------------------------------------------------------------
    # Nếu AJAX => trả JSON để frontend biết job đã start
    # ------------------------------------------------------------
    if is_ajax:
        with _STATE_LOCK:
            return JsonResponse(
                {
                    "ok": True,
                    "is_running": _STATE["is_running"],
                    "logs": list(_STATE["logs"])[-300:],
                }
            )

    # ------------------------------------------------------------
    # Nếu không AJAX => render lại template với context mới
    # ------------------------------------------------------------
    with _STATE_LOCK:
        context["is_running"] = _STATE["is_running"]
        context["logs"] = list(_STATE["logs"])[-300:]
    return render(request, "weather/HTML_Crawl_data_by_API.html", context)


# ============================================================
# api_weather_logs_view: ENDPOINT TRẢ LOGS/STATE CHO FRONTEND POLLING
# ============================================================
# Method: GET
#
# Trả về:
# - is_running, logs (300 dòng cuối)
# - last_returncode, last_started_at, last_finished_at
# - thông tin file output mới nhất: last_csv_name, csv_size_mb, last_crawl_time
@require_http_methods(["GET"])
def api_weather_logs_view(request):
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))

    # ------------------------------------------------------------
    # Lấy thông tin file output mới nhất (nếu có)
    # ------------------------------------------------------------
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
        pass  # nuốt lỗi, vẫn trả logs/state bình thường

    # ------------------------------------------------------------
    # Snapshot state thread-safe
    # ------------------------------------------------------------
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


# ============================================================
# crawl_vrain_html_view: TRANG COMING SOON (VRAIN HTML)
# ============================================================
# - Hiện tại chỉ render 1 trang thông báo "coming soon"
# - Sau này bạn sẽ thay bằng trang crawl vrain html thực tế
@require_http_methods(["GET"])
def crawl_vrain_html_view(request):
    return render(request, "weather/coming_soon_vrain_html.html")
