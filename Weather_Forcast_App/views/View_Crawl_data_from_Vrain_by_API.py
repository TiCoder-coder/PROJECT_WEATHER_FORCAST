import os
import sys
import threading
import subprocess
from pathlib import Path
from datetime import datetime

from django.http import JsonResponse, HttpResponseNotAllowed
from django.shortcuts import render


# ============================================================
# CẤU HÌNH ĐƯỜNG DẪN CỐ ĐỊNH CHO APP (ROOT / SCRIPT / OUTPUT)
# ============================================================
# APP_ROOT:
# - Thư mục gốc của app (tính từ file views hiện tại).
# - Path(__file__).resolve(): lấy path tuyệt đối của file đang chạy.
# - .parents[1]: đi lên 2 cấp thư mục để ra root (tuỳ cấu trúc dự án).
APP_ROOT = Path(__file__).resolve().parents[1]

# SCRIPT_PATH:
# - Đường dẫn đến script crawl dữ liệu mưa Vrain qua API.
# - Script này sẽ được chạy bằng subprocess ở background thread.
SCRIPT_PATH = APP_ROOT / "scripts" / "Crawl_data_from_Vrain_by_API.py"

# OUTPUT_DIR:
# - Thư mục output nơi script sẽ xuất file dataset (xlsx/csv/xls).
OUTPUT_DIR = APP_ROOT / "output"


# ============================================================
# _STATE: TRẠNG THÁI JOB (LƯU TRONG RAM / IN-MEMORY)
# ============================================================
# Dùng để:
# - UI biết job có đang chạy không (is_running)
# - UI lấy log realtime (logs)
# - hiển thị thông tin lần chạy gần nhất (returncode, time, file, size)
#
# Giải thích:
# - is_running:
#   + True: đang có job crawl chạy nền
#   + False: không có job nào đang chạy
#
# - logs:
#   + list các dòng log (string)
#   + endpoint tail sẽ trả các dòng mới (polling theo since)
#
# - last_returncode:
#   + mã thoát của subprocess lần chạy gần nhất
#   + 0 thường là OK; khác 0 thường là có lỗi
#
# - last_crawl_time:
#   + thời điểm kết thúc job (string), dùng hiển thị "lần crawl gần nhất"
#
# - last_file / last_size_mb:
#   + file output mới nhất & dung lượng (MB) để hiển thị trên UI
_STATE = {
    "is_running": False,
    "logs": [],
    "last_returncode": None,
    "last_crawl_time": None,
    "last_file": None,
    "last_size_mb": None,
}

# Thread Lock để tránh race condition khi đọc/ghi _STATE từ nhiều thread
_STATE_LOCK = threading.Lock()

# _LOG_LIMIT:
# - Giới hạn số dòng log lưu trong RAM để tránh phình bộ nhớ.
# - Nếu vượt giới hạn, chỉ giữ lại _LOG_LIMIT dòng cuối cùng (mới nhất).
_LOG_LIMIT = 2500


# ============================================================
# _push_log(line): THÊM 1 DÒNG LOG VÀO _STATE["logs"]
# ============================================================
# Mục tiêu:
# - Worker subprocess sẽ in log ra stdout/stderr
# - Ta đọc từng dòng và lưu lại để frontend polling hiển thị realtime
#
# Cách làm:
# - rstrip newline cuối dòng
# - bỏ qua dòng rỗng
# - append vào list logs
# - nếu vượt _LOG_LIMIT: cắt giữ lại phần cuối
def _push_log(line: str):
    line = (line or "").rstrip("\n")
    if not line:
        return
    with _STATE_LOCK:
        _STATE["logs"].append(line)
        if len(_STATE["logs"]) > _LOG_LIMIT:
            _STATE["logs"] = _STATE["logs"][-_LOG_LIMIT:]


# ============================================================
# _scan_latest_output(): QUÉT THƯ MỤC output/ ĐỂ LẤY FILE MỚI NHẤT
# ============================================================
# Dùng cho:
# - Trang UI (GET) hiển thị dataset mới nhất (tên file, size, thời gian)
# - Sau khi job chạy xong, update last_file và last_size_mb
#
# Logic:
# - nếu OUTPUT_DIR không tồn tại => return None
# - chỉ xét file có đuôi thuộc exts (.xlsx/.csv/.xls)
# - chọn file có st_mtime lớn nhất (mới nhất)
# - trả về:
#   + latest.name  : tên file
#   + size_mb      : dung lượng MB
#   + mtime_str    : mtime dạng string để hiển thị UI
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


# ============================================================
# _run_script_worker(): WORKER CHẠY TRONG THREAD BACKGROUND
# ============================================================
# Vai trò:
# - tạo OUTPUT_DIR nếu chưa có
# - chạy script bằng subprocess (python -u ...) để log realtime
# - đọc stdout theo dòng và push log vào _STATE["logs"]
# - cập nhật trạng thái sau khi xong: returncode, last_crawl_time, last_file, last_size_mb
#
# Lưu ý:
# - Không dùng lock => dựa vào việc start_view đã chặn chạy song song.
# - Nếu bạn chạy nhiều worker đồng thời thì nên thêm Lock (nhưng bạn bảo không đổi code).
def _run_script_worker():
    try:
        # đảm bảo output dir tồn tại trước khi chạy (tránh script ghi file lỗi vì thiếu folder)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # log header để UI phân tách từng lần chạy
        _push_log("========== START VRAIN API CRAWL ==========")
        _push_log(f"Script: {SCRIPT_PATH}")
        _push_log(f"Output dir: {OUTPUT_DIR}")

        # nếu script không tồn tại -> log lỗi + set returncode = -1
        if not SCRIPT_PATH.exists():
            _push_log("[ERROR] Script không tồn tại!")
            _STATE["last_returncode"] = -1
            return

        # --------------------------------------------------------
        # env: cấu hình để log realtime (unbuffered)
        # --------------------------------------------------------
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # PYTHONUNBUFFERED=1:
        # - ép Python flush stdout/stderr ngay
        # - giúp logs hiển thị realtime thay vì bị buffer

        # --------------------------------------------------------
        # subprocess.Popen: chạy script crawl
        # --------------------------------------------------------
        # [sys.executable, "-u", str(SCRIPT_PATH)]:
        # - sys.executable: python đang chạy Django (đúng môi trường venv)
        # - "-u": unbuffered mode (kết hợp PYTHONUNBUFFERED càng chắc realtime)
        #
        # cwd=str(APP_ROOT):
        # - đặt working directory = root app
        # - để script dùng relative path ổn định
        #
        # stdout=PIPE:
        # - bắt stdout để đọc theo dòng
        #
        # stderr=STDOUT:
        # - gộp stderr chung vào stdout => log lỗi cũng đi cùng log thường
        #
        # text=True:
        # - đọc stream thành string thay vì bytes
        #
        # bufsize=1:
        # - line-buffered (khi text=True) => đọc được theo dòng
        #
        # env=env:
        # - truyền môi trường có PYTHONUNBUFFERED
        proc = subprocess.Popen(
            [sys.executable, "-u", str(SCRIPT_PATH)],
            cwd=str(APP_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        # --------------------------------------------------------
        # Stream log realtime: đọc từng dòng từ stdout
        # --------------------------------------------------------
        for line in proc.stdout:
            _push_log(line)

        # đợi process kết thúc và lấy return code
        rc = proc.wait()
        _STATE["last_returncode"] = rc

        # lưu thời điểm "kết thúc crawl"
        _STATE["last_crawl_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # quét file output mới nhất để cập nhật lên UI
        last_file, last_size_mb, _ = _scan_latest_output()
        _STATE["last_file"] = last_file
        _STATE["last_size_mb"] = last_size_mb

        # log footer
        _push_log("========== DONE ==========")
        _push_log(f"Return code: {rc}")
        if last_file:
            _push_log(f"Latest output: {last_file} ({last_size_mb} MB)")
        else:
            _push_log("No output file detected in output/.")

    except Exception as e:
        # nếu có exception trong worker:
        # - set returncode = -1
        # - log exception dạng repr để debug (giữ cả type + message)
        _STATE["last_returncode"] = -1
        _push_log(f"[EXCEPTION] {repr(e)}")
    finally:
        # kết thúc worker => set is_running = False để UI/endpoint biết job đã xong
        _STATE["is_running"] = False


# ============================================================
# crawl_vrain_api_view: VIEW GET RENDER GIAO DIỆN CRAWL VRAIN API
# ============================================================
# - Quét output để lấy file mới nhất (nếu có)
# - Tạo context gồm:
#   + is_running, last_returncode
#   + last_crawl_time: ưu tiên _STATE (lần chạy gần nhất), nếu None thì dùng mtime file
#   + last_file/last_size_mb: ưu tiên kết quả scan mới nhất, fallback sang _STATE
# - Render template HTML_Crawl_data_from_Vrain_by_API.html
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


# ============================================================
# crawl_vrain_api_start_view: ENDPOINT START JOB (POST)
# ============================================================
# - Chỉ cho phép POST (nếu không => HttpResponseNotAllowed)
# - Nếu job đang chạy => trả 409 Conflict
# - Reset state logs + last_returncode
# - Spawn thread daemon chạy _run_script_worker
# - Trả JSON ok để frontend biết đã start
def crawl_vrain_api_start_view(request):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])

    with _STATE_LOCK:
        if _STATE["is_running"]:
            return JsonResponse({"ok": False, "message": "Job đang chạy rồi."}, status=409)
        _STATE["is_running"] = True
        _STATE["logs"] = []
        _STATE["last_returncode"] = None

    t = threading.Thread(target=_run_script_worker, daemon=True)
    t.start()

    return JsonResponse({"ok": True, "message": "Started"})


# ============================================================
# crawl_vrain_api_tail_view: ENDPOINT POLLING LOG REALTIME (GET)
# ============================================================
# Frontend sẽ gọi định kỳ để lấy log mới.
#
# Query param:
# - since: index log client đã nhận
#
# Logic:
# - parse since -> since_i
# - logs = _STATE["logs"]
# - new_lines = logs[since_i:] => phần log mới
#
# Response:
# - next_since: len(logs) => client dùng cho lần gọi tiếp theo
# - is_running: job còn chạy không
# - last_*: metadata lần crawl gần nhất
def crawl_vrain_api_tail_view(request):
    since = request.GET.get("since", "0")
    try:
        since_i = max(0, int(since))
    except:
        since_i = 0

    with _STATE_LOCK:
        logs = list(_STATE["logs"])
        data = {
            "ok": True,
            "is_running": _STATE["is_running"],
            "next_since": len(logs),
            "lines": logs[since_i:],
            "last_returncode": _STATE["last_returncode"],
            "last_crawl_time": _STATE["last_crawl_time"],
            "last_file": _STATE["last_file"],
            "last_size_mb": _STATE["last_size_mb"],
        }

    return JsonResponse(data)
