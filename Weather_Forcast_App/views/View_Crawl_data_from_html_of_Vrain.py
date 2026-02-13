import os
import sys
import threading
import subprocess
from pathlib import Path
from datetime import datetime

from django.http import JsonResponse, HttpResponseNotAllowed
from django.shortcuts import render


# ============================================================
# CẤU HÌNH ĐƯỜNG DẪN (APP ROOT / SCRIPT / OUTPUT)
# ============================================================
# APP_ROOT:
# - thư mục gốc của app (tính từ file hiện tại)
# - Path(__file__).resolve(): lấy đường dẫn tuyệt đối của file python đang chạy
# - .parents[1]: đi lên 2 cấp thư mục (tuỳ cấu trúc project) để ra root của app
APP_ROOT = Path(__file__).resolve().parents[1]

# SCRIPT_PATH:
# - đường dẫn tới script crawler HTML của Vrain
# - script này sẽ được chạy bằng subprocess trong background thread
SCRIPT_PATH = APP_ROOT / "scripts" / "Crawl_data_from_html_of_Vrain.py"

# OUTPUT_DIR:
# - thư mục output nơi script sẽ xuất file (.xlsx/.csv)
OUTPUT_DIR = Path("/media/voanhnhat/SDD_OUTSIDE5/PROJECT_WEATHER_FORECAST/data/data_crawl")


# ============================================================
# _STATE: TRẠNG THÁI CHẠY JOB (IN-MEMORY)
# ============================================================
# Lưu trạng thái job trong RAM để:
# - view GET render UI hiển thị info lần chạy gần nhất
# - endpoint tail trả log realtime khi frontend polling
#
# Giải thích từng field:
# - is_running:
#   + True  => job đang chạy
#   + False => job không chạy
#
# - logs:
#   + list log lines (string) để UI hiển thị realtime
#   + bị giới hạn bởi _LOG_LIMIT để tránh tràn RAM
#
# - last_returncode:
#   + return code lần chạy gần nhất của subprocess
#   + 0 thường là OK, khác 0 là lỗi
#
# - last_crawl_time:
#   + thời điểm job kết thúc (hoặc lần crawl gần nhất)
#
# - last_file / last_size_mb:
#   + tên file output mới nhất và dung lượng MB (để hiển thị trên UI)
_STATE = {
    "is_running": False,
    "logs": [],
    "last_returncode": None,
    "last_crawl_time": None,
    "last_file": None,
    "last_size_mb": None,
}

# _LOG_LIMIT:
# - giới hạn số dòng log lưu giữ
# - nếu vượt -> chỉ giữ lại N dòng cuối (mới nhất)
_LOG_LIMIT = 2500


# ============================================================
# _push_log(line): THÊM LOG VÀO _STATE["logs"] VÀ GIỚI HẠN DUNG LƯỢNG
# ============================================================
# Mục tiêu:
# - Worker subprocess sẽ in log ra stdout
# - Ta đọc từng dòng và đẩy vào logs để frontend lấy realtime
#
# Cơ chế:
# - strip newline cuối dòng
# - bỏ dòng rỗng
# - append vào list
# - nếu logs vượt _LOG_LIMIT => cắt giữ lại phần cuối (mới nhất)
def _push_log(line: str):
    line = (line or "").rstrip("\n")
    if not line:
        return
    _STATE["logs"].append(line)
    if len(_STATE["logs"]) > _LOG_LIMIT:
        _STATE["logs"] = _STATE["logs"][-_LOG_LIMIT:]


# ============================================================
# _scan_latest_output(): TÌM FILE OUTPUT MỚI NHẤT TRONG output/
# ============================================================
# Vai trò:
# - sau khi crawler chạy xong, ta muốn biết file nào vừa được tạo ra gần nhất
# - UI cũng cần hiển thị "latest output file" và dung lượng
#
# Logic:
# - nếu OUTPUT_DIR không tồn tại => return (None, None, None)
# - chỉ xét các file có đuôi trong exts (.xlsx .csv .xls)
# - chọn file có mtime lớn nhất (mới nhất)
# - trả về:
#   (latest.name, size_mb, mtime_str)
def _scan_latest_output():
    """
    Tìm file output mới nhất trong output/.
    Ưu tiên .xlsx, .csv (bạn có thể thêm đuôi khác nếu cần).
    """
    if not OUTPUT_DIR.exists():
        return None, None, None

    # chỉ chấp nhận các file theo extension này
    exts = {".xlsx", ".csv", ".xls"}

    # iter dir và filter file hợp lệ
    files = [p for p in OUTPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not files:
        return None, None, None

    # latest: file có mtime lớn nhất => file mới nhất
    latest = max(files, key=lambda p: p.stat().st_mtime)

    # size_mb: bytes -> MB
    size_mb = round(latest.stat().st_size / (1024 * 1024), 2)

    # mtime_str: format mtime để hiển thị UI
    mtime_str = datetime.fromtimestamp(latest.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return latest.name, size_mb, mtime_str


# ============================================================
# _run_script_worker(): HÀM CHẠY TRONG THREAD BACKGROUND
# ============================================================
# Vai trò:
# - chạy script Crawl_data_from_html_of_Vrain.py bằng subprocess
# - đọc stdout (gộp stderr vào stdout) và ghi log realtime
# - cập nhật _STATE sau khi xong: returncode, last_crawl_time, last_file, last_size_mb
#
# Lưu ý:
# - Đây là in-memory state, không có lock => nếu gọi song song nhiều request
#   có thể race condition, nhưng bạn đã chặn is_running ở start_view.
def _run_script_worker():
    try:
        # log header
        _push_log("========== START VRAIN HTML CRAWL ==========")
        _push_log(f"Script: {SCRIPT_PATH}")
        _push_log(f"Output dir: {OUTPUT_DIR}")

        # --------------------------------------------------------
        # check script tồn tại
        # --------------------------------------------------------
        if not SCRIPT_PATH.exists():
            _push_log("[ERROR] Script không tồn tại!")
            _STATE["last_returncode"] = -1
            return

        # --------------------------------------------------------
        # chạy subprocess
        # --------------------------------------------------------
        # [sys.executable, str(SCRIPT_PATH)]:
        # - sys.executable: python đang chạy Django (đúng venv)
        # - chạy file script bằng python
        #
        # cwd=str(APP_ROOT):
        # - đặt working directory = APP_ROOT
        # - đảm bảo script dùng relative path sẽ đúng
        #
        # stdout=PIPE + stderr=STDOUT:
        # - gom toàn bộ output thành 1 stream để đọc theo dòng
        #
        # text=True:
        # - đọc dạng string
        #
        # bufsize=1:
        # - line buffered (hữu ích cho realtime log)
        proc = subprocess.Popen(
            [sys.executable, str(SCRIPT_PATH)],
            cwd=str(APP_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # --------------------------------------------------------
        # đọc log realtime từ subprocess stdout
        # --------------------------------------------------------
        for line in proc.stdout:
            _push_log(line)

        # đợi process kết thúc và lấy returncode
        rc = proc.wait()
        _STATE["last_returncode"] = rc

        # last_crawl_time:
        # - thời điểm job kết thúc (ghi vào state)
        _STATE["last_crawl_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # --------------------------------------------------------
        # Sau khi xong, quét file output mới nhất để hiển thị UI
        # --------------------------------------------------------
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
        # nếu có exception:
        # - đặt returncode = -1 (quy ước lỗi)
        # - ghi log exception dạng repr để dễ debug
        _STATE["last_returncode"] = -1
        _push_log(f"[EXCEPTION] {repr(e)}")
    finally:
        # luôn luôn set is_running False khi kết thúc
        _STATE["is_running"] = False


# ============================================================
# crawl_vrain_html_view: VIEW GET RENDER UI TRANG CRAWL VRAIN HTML
# ============================================================
# - đọc latest output info từ thư mục output/
# - build context dựa trên _STATE + info từ file
# - render template HTML_Crawl_data_from_html_of_Vrain.html
def crawl_vrain_html_view(request):
    # scan latest output để lấy thông tin "lần chạy gần nhất"
    last_file, last_size_mb, last_time_from_file = _scan_latest_output()

    # context:
    # - is_running: job có đang chạy không
    # - last_returncode: returncode lần chạy gần nhất
    # - last_crawl_time: ưu tiên _STATE["last_crawl_time"], nếu None thì dùng mtime file
    # - last_file/last_size_mb: ưu tiên file scan, nếu không có thì dùng state cũ
    context = {
        "is_running": _STATE["is_running"],
        "last_returncode": _STATE["last_returncode"],
        "last_crawl_time": _STATE["last_crawl_time"] or last_time_from_file,
        "last_file": last_file or _STATE["last_file"],
        "last_size_mb": last_size_mb or _STATE["last_size_mb"],
    }

    # render UI
    return render(request, "weather/HTML_Crawl_data_from_html_of_Vrain.html", context)


# ============================================================
# crawl_vrain_html_start_view: ENDPOINT START JOB (POST)
# ============================================================
# - chỉ chấp nhận POST
# - nếu job đang chạy -> 409
# - reset state logs/returncode
# - spawn thread chạy _run_script_worker()
# - trả JSON ok
def crawl_vrain_html_start_view(request):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])

    # nếu job đang chạy thì không start job mới
    if _STATE["is_running"]:
        return JsonResponse({"ok": False, "message": "Job đang chạy rồi."}, status=409)

    # set trạng thái bắt đầu chạy
    _STATE["is_running"] = True
    _STATE["logs"] = []              # reset log
    _STATE["last_returncode"] = None # reset return code (chưa có kết quả)

    # tạo thread daemon chạy worker
    t = threading.Thread(target=_run_script_worker, daemon=True)
    t.start()

    return JsonResponse({"ok": True, "message": "Started"})


# ============================================================
# crawl_vrain_html_tail_view: ENDPOINT POLLING LOG REALTIME (GET)
# ============================================================
# Frontend sẽ gọi liên tục để lấy log mới.
#
# Query param:
# - since: index log client đã nhận (vd: 0, 10, 25...)
#
# Logic:
# - parse since -> since_i
# - logs = _STATE["logs"]
# - new_lines = logs[since_i:]
# - next_since = len(logs) (để lần sau client gửi since=next_since)
#
# Response:
# - lines: log mới
# - is_running: job còn chạy không
# - last_*: metadata lần chạy gần nhất
def crawl_vrain_html_tail_view(request):
    """
    Frontend gọi polling để lấy log mới.
    Query param:
      - since: index log đã có (int)
    """
    since = request.GET.get("since", "0")
    try:
        since_i = max(0, int(since))
    except:
        since_i = 0

    # logs hiện tại trong state
    logs = _STATE["logs"]

    # chỉ lấy các dòng log mới kể từ since_i
    new_lines = logs[since_i:]

    # trả JSON cho frontend:
    # - next_since = len(logs) để frontend cập nhật offset
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
