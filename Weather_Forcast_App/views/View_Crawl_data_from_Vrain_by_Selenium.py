import os
import sys
import threading
import subprocess
from pathlib import Path
from datetime import datetime
import uuid

from django.http import JsonResponse, HttpResponseNotAllowed
from django.shortcuts import render


# ============================================================
# CẤU HÌNH PATH TOÀN CỤC (ROOT / SCRIPT / OUTPUT)
# ============================================================
# APP_ROOT:
# - thư mục gốc của app (tính từ file python hiện tại)
# - Path(__file__).resolve(): lấy đường dẫn tuyệt đối của file
# - .parents[1]: đi lên 2 cấp thư mục (phụ thuộc cấu trúc project)
APP_ROOT = Path(__file__).resolve().parents[1]

# SCRIPT_PATH:
# - script crawl Vrain bằng Selenium
# - sẽ được chạy bằng subprocess trong thread background
SCRIPT_PATH = APP_ROOT / "scripts" / "Crawl_data_from_Vrain_by_Selenium.py"

# OUTPUT_DIR:
# - thư mục output nơi script selenium sẽ xuất file (xlsx/csv/xls)
# Dynamic path: tự tính từ vị trí project, không hardcode Linux path
from Weather_Forcast_App.paths import DATA_CRAWL_DIR
OUTPUT_DIR = DATA_CRAWL_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# _STATE: TRẠNG THÁI JOB (LƯU IN-MEMORY)
# ============================================================
# Dùng để:
# - kiểm soát job có đang chạy không (is_running)
# - lưu job_id hiện tại (phân biệt từng lần chạy)
# - lưu logs realtime cho frontend polling
# - lưu metadata lần chạy gần nhất: returncode, crawl time, file, size
#
# Giải thích field:
# - is_running:
#   + True  => đang chạy crawl selenium
#   + False => job đã xong / chưa chạy
#
# - job_id:
#   + id duy nhất cho mỗi lần start (uuid4 hex)
#   + frontend có thể dùng để đảm bảo đang xem đúng job
#
# - logs:
#   + list các dòng log (string) đọc từ stdout/stderr của subprocess
#   + endpoint tail sẽ trả incremental theo offset
#
# - last_returncode:
#   + return code lần chạy gần nhất (0 OK, !=0 lỗi)
#
# - last_crawl_time:
#   + thời điểm job kết thúc (string)
#
# - last_file / last_size_mb:
#   + file output mới nhất và dung lượng MB để hiển thị UI
_STATE = {
    "is_running": False,
    "job_id": None,
    "logs": [],
    "last_returncode": None,
    "last_crawl_time": None,
    "last_file": None,
    "last_size_mb": None,
}

# Thread Lock để tránh race condition khi đọc/ghi _STATE từ nhiều thread
_STATE_LOCK = threading.Lock()

# _LOG_LIMIT:
# - giới hạn số dòng log lưu trong RAM để tránh tràn bộ nhớ
# - nếu vượt giới hạn, chỉ giữ lại N dòng cuối cùng
_LOG_LIMIT = 3000


# ============================================================
# _push_log(line): THÊM LOG VÀO _STATE["logs"] (CÓ GIỚI HẠN)
# ============================================================
# Cơ chế:
# - strip newline cuối dòng
# - bỏ qua dòng rỗng
# - append vào logs
# - nếu logs vượt _LOG_LIMIT => giữ lại phần cuối (mới nhất)
def _push_log(line: str):
    line = (line or "").rstrip("\n")
    if not line:
        return
    with _STATE_LOCK:
        _STATE["logs"].append(line)
        if len(_STATE["logs"]) > _LOG_LIMIT:
            _STATE["logs"] = _STATE["logs"][-_LOG_LIMIT:]


# ============================================================
# _scan_latest_output(): QUÉT output/ LẤY FILE MỚI NHẤT
# ============================================================
# Dùng để:
# - UI hiển thị dataset mới nhất
# - sau khi job xong: cập nhật _STATE["last_file"], _STATE["last_size_mb"]
#
# Logic:
# - nếu OUTPUT_DIR chưa tồn tại => return None
# - chỉ xét file có extension trong exts
# - chọn file có mtime lớn nhất (mới nhất)
# - trả về:
#   (latest.name, size_mb, mtime_str)
def _scan_latest_output():
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
# _run_script_worker(job_id): WORKER CHẠY SUBPROCESS SELENIUM
# ============================================================
# Vai trò:
# - chạy script selenium bằng subprocess trong background thread
# - đọc stdout/stderr theo dòng và push vào logs
# - update _STATE sau khi xong: returncode, crawl time, last file, size
#
# Tham số:
# - job_id: id của lần chạy hiện tại (được set ở start_view)
#
# Lưu ý:
# - job_id hiện được truyền vào nhưng trong code chưa dùng để filter logs;
#   bạn giữ nó để frontend biết job hiện tại là job nào (tránh nhầm).
def _run_script_worker(job_id: str):
    try:
        # log header để phân tách log của từng lần chạy
        _push_log("========== START VRAIN SELENIUM CRAWL ==========")
        _push_log(f"Script: {SCRIPT_PATH}")
        _push_log(f"Output dir: {OUTPUT_DIR}")

        # check script tồn tại trước khi chạy
        if not SCRIPT_PATH.exists():
            _push_log("[ERROR] Script không tồn tại!")
            _STATE["last_returncode"] = -1
            return

        # --------------------------------------------------------
        # env: cấu hình để python subprocess flush log realtime
        # --------------------------------------------------------
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # PYTHONUNBUFFERED=1:
        # - hạn chế việc stdout bị buffer => log realtime rõ hơn

        # --------------------------------------------------------
        # subprocess.Popen: chạy script selenium
        # --------------------------------------------------------
        # [sys.executable, "-u", str(SCRIPT_PATH)]:
        # - sys.executable: python đang chạy server (đúng venv)
        # - "-u": unbuffered (đẩy log realtime)
        #
        # cwd=str(APP_ROOT):
        # - đặt working directory root app
        #
        # stdout=PIPE, stderr=STDOUT:
        # - gom stderr vào stdout để đọc 1 stream duy nhất
        #
        # text=True:
        # - đọc output thành string
        #
        # bufsize=1:
        # - line buffered (khi text=True)
        #
        # env=env:
        # - truyền env đã set PYTHONUNBUFFERED
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
        # Đọc stdout theo dòng (realtime)
        # --------------------------------------------------------
        # proc.stdout có thể None trong vài trường hợp, nên code check if proc.stdout:
        if proc.stdout:
          for line in proc.stdout:
              _push_log(line)

        # đợi process kết thúc và lấy returncode
        rc = proc.wait()
        _STATE["last_returncode"] = rc

        # last_crawl_time: thời điểm job kết thúc (string)
        _STATE["last_crawl_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # quét file output mới nhất để cập nhật lên UI
        last_file, last_size_mb, _ = _scan_latest_output()
        _STATE["last_file"] = last_file
        _STATE["last_size_mb"] = last_size_mb

        # log footer + thông tin output nếu có
        _push_log("========== DONE ==========")
        _push_log(f"Return code: {rc}")
        if last_file:
            _push_log(f"Latest output: {last_file} ({last_size_mb} MB)")
        else:
            _push_log("No output file detected in output/.")
    except Exception as e:
        # nếu xảy ra exception:
        # - gán returncode = -1 (quy ước lỗi)
        # - log exception dạng repr để dễ debug
        _STATE["last_returncode"] = -1
        _push_log(f"[EXCEPTION] {repr(e)}")
    finally:
        # dù thành công hay lỗi, kết thúc worker -> set is_running False
        _STATE["is_running"] = False


# ============================================================
# crawl_vrain_selenium_view: VIEW GET RENDER UI SELENIUM
# ============================================================
# - Quét output lấy file mới nhất
# - build context:
#   + is_running, last_returncode
#   + last_crawl_time: ưu tiên _STATE, fallback sang mtime file
#   + last_file/last_size_mb: ưu tiên scan mới nhất, fallback state
# - render template HTML_Crawl_data_from_Vrain_by_Selenium.html
def crawl_vrain_selenium_view(request):
    last_file, last_size_mb, last_time_from_file = _scan_latest_output()
    context = {
        "is_running": _STATE["is_running"],
        "last_returncode": _STATE["last_returncode"],
        "last_crawl_time": _STATE["last_crawl_time"] or last_time_from_file,
        "last_file": last_file or _STATE["last_file"],
        "last_size_mb": last_size_mb or _STATE["last_size_mb"],
    }
    return render(request, "weather/HTML_Crawl_data_from_Vrain_by_Selenium.html", context)


# ============================================================
# crawl_vrain_selenium_start_view: ENDPOINT START JOB (POST)
# ============================================================
# - chỉ cho phép POST
# - nếu job đang chạy => 409
# - tạo job_id mới (uuid)
# - reset state logs / returncode
# - spawn thread daemon chạy _run_script_worker(job_id)
# - trả JSON: ok + job_id
def crawl_vrain_selenium_start_view(request):
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])

    with _STATE_LOCK:
        if _STATE["is_running"]:
            return JsonResponse({"ok": False, "message": "Job đang chạy rồi."}, status=409)
        job_id = uuid.uuid4().hex
        _STATE["job_id"] = job_id
        _STATE["is_running"] = True
        _STATE["logs"] = []
        _STATE["last_returncode"] = None

    t = threading.Thread(target=_run_script_worker, args=(job_id,), daemon=True)
    t.start()

    return JsonResponse({"ok": True, "job_id": job_id})


# ============================================================
# crawl_vrain_selenium_tail_view: ENDPOINT POLLING LOG (GET)
# ============================================================
# Frontend gọi liên tục để lấy log mới.
#
# Query param:
# - offset: số dòng log client đã có (int)
#
# Logic:
# - parse offset -> offset_i
# - logs = _STATE["logs"]
# - new_lines = logs[offset_i:] => phần log mới
#
# Response:
# - job_id: id của job hiện tại
# - lines: log mới
# - offset: len(logs) => offset mới để frontend dùng lần gọi sau
# - done: job đã xong chưa (done = not is_running)
# - is_running + metadata lần chạy gần nhất
def crawl_vrain_selenium_tail_view(request):
    offset = request.GET.get("offset", "0")
    try:
        offset_i = max(0, int(offset))
    except:
        offset_i = 0

    with _STATE_LOCK:
        logs = list(_STATE["logs"])
        data = {
            "ok": True,
            "job_id": _STATE["job_id"],
            "lines": logs[offset_i:],
            "offset": len(logs),
            "done": (not _STATE["is_running"]),
            "is_running": _STATE["is_running"],
            "last_returncode": _STATE["last_returncode"],
            "last_crawl_time": _STATE["last_crawl_time"],
            "last_file": _STATE["last_file"],
            "last_size_mb": _STATE["last_size_mb"],
        }

    return JsonResponse(data)
