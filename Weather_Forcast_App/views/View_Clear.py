import json
import threading
import uuid
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

from django.http import JsonResponse, HttpResponseNotAllowed, Http404
from django.views.decorators.http import require_http_methods


# ============================================================
# CẤU HÌNH ĐƯỜNG DẪN THƯ MỤC (PATHS)
# ============================================================
# APP_ROOT:
# - Thư mục gốc của app (tính từ file hiện tại)
# - Path(__file__).resolve(): đường dẫn tuyệt đối tới file python đang chạy
# - .parents[1]: đi lên 2 cấp (tuỳ cấu trúc project) để ra "root" của app

# Dynamic path: tự tính từ vị trí project, không hardcode Linux path
from Weather_Forcast_App.paths import (
    DATA_MERGE_DIR, DATA_CRAWL_DIR, DATA_CLEAN_ROOT,
    DATA_CLEAN_MERGE_DIR, DATA_CLEAN_NOT_MERGE_DIR,
)
MERGE_DIR = DATA_MERGE_DIR
OUTPUT_DIR = DATA_CRAWL_DIR
CLEANED_ROOT = DATA_CLEAN_ROOT
CLEANED_MERGE_DIR = DATA_CLEAN_MERGE_DIR
CLEANED_RAW_DIR = DATA_CLEAN_NOT_MERGE_DIR

# ALLOWED_EXTS:
# - chỉ cho phép clean các file có đuôi trong danh sách (để an toàn)
ALLOWED_EXTS = {".csv", ".xlsx", ".xls"}

# LOG_LIMIT:
# - giới hạn số dòng log lưu trong RAM cho mỗi job
# - giúp tránh tràn bộ nhớ nếu job chạy lâu và log quá nhiều
LOG_LIMIT = 4000


# ============================================================
# JOB STORE (IN-MEMORY) + LOCK ĐỒNG BỘ LUỒNG
# ============================================================
# _JOBS:
# - dict lưu toàn bộ job đang chạy/đã chạy
# - key: job_id
# - value: object chứa logs, progress, result, error...
#
# Lưu ý:
# - Đây là in-memory store => nếu restart server thì mất hết job
# - Nếu chạy multi-process (gunicorn workers) thì mỗi process sẽ có _JOBS riêng
_JOBS = {}

# _JOBS_LOCK:
# - threading.Lock để đồng bộ khi nhiều thread cùng đọc/ghi _JOBS
# - cần thiết vì bạn spawn thread (_worker) để chạy background
_JOBS_LOCK = threading.Lock()


# ============================================================
# _now(): LẤY THỜI GIAN HIỆN TẠI DẠNG STRING
# ============================================================
# - Trả về chuỗi timestamp để lưu started_at / finished_at cho job
def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# _push(job_id, line): THÊM 1 DÒNG LOG VÀO JOB
# ============================================================
# Mục đích:
# - Worker thread sẽ liên tục gọi _push để ghi log
# - Frontend sẽ poll logs qua clean_data_tail_view
#
# Cơ chế:
# - strip newline, bỏ dòng rỗng
# - dùng _JOBS_LOCK để thread-safe
# - cắt log nếu vượt LOG_LIMIT (giữ lại phần mới nhất)
def _push(job_id: str, line: str):
    line = (line or "").rstrip("\n")  # bỏ \n cuối dòng (nếu có)
    if not line:
        return  # không lưu dòng rỗng

    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return  # job không tồn tại (có thể đã bị xoá/không tạo)

        job["logs"].append(line)  # append log mới

        # Nếu logs vượt giới hạn -> cắt giữ lại LOG_LIMIT dòng cuối
        if len(job["logs"]) > LOG_LIMIT:
            job["logs"] = job["logs"][-LOG_LIMIT:]


# ============================================================
# _set_progress(job_id, pct, step): CẬP NHẬT TIẾN TRÌNH JOB
# ============================================================
# pct:
# - phần trăm tiến độ (0..100)
# step:
# - mô tả bước hiện tại ("Đọc dữ liệu", "Clean dữ liệu", ...)
#
# clamp pct:
# - đảm bảo pct luôn nằm trong 0..100 để UI không bị sai
def _set_progress(job_id: str, pct: int, step: str):
    pct = max(0, min(100, int(pct)))  # clamp 0..100
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return
        job["progress"] = {"pct": pct, "step": step}


# ============================================================
# _safe_pick_file(base_dir, filename): CHỌN FILE AN TOÀN
# ============================================================
# Mục đích:
# - Tránh path traversal (vd: filename = "../../etc/passwd")
# - Chỉ cho phép chọn file nằm trong base_dir
# - Chỉ cho phép file tồn tại và đúng loại đuôi (ALLOWED_EXTS)
#
# Cách làm:
# - resolve() để chuẩn hoá đường dẫn tuyệt đối
# - kiểm tra base_dir có nằm trong parents của p hay không
# - nếu không -> raise Http404("Invalid path")
def _safe_pick_file(base_dir: Path, filename: str) -> Path:
    base_dir = base_dir.resolve()
    p = (base_dir / filename).resolve()

    # Nếu base_dir không phải "cha" của p => file đã thoát khỏi base_dir => nguy hiểm
    if base_dir not in p.parents:
        raise Http404("Invalid path")

    # File phải tồn tại và là file
    if not p.exists() or not p.is_file():
        raise Http404("File not found")

    # Chỉ cho phép đúng đuôi file
    if p.suffix.lower() not in ALLOWED_EXTS:
        raise Http404("Unsupported file type")

    return p


# ============================================================
# _scan_files(directory): QUÉT FILE TRONG THƯ MỤC VÀ TRẢ METADATA
# ============================================================
# Mục tiêu:
# - trả danh sách file cho UI (dropdown/list)
# - sắp xếp theo mới nhất trước (mtime giảm dần)
# - mỗi item gồm: name, size_mb, mtime, ext
def _scan_files(directory: Path):
    # đảm bảo thư mục tồn tại
    directory.mkdir(parents=True, exist_ok=True)

    # patterns: danh sách pattern file cần quét
    patterns = ["*.xlsx", "*.xls", "*.csv"]
    files = []
    for pat in patterns:
        files.extend(list(directory.glob(pat)))  # glob theo pattern

    # sort theo thời gian sửa đổi (mtime) giảm dần => file mới nhất lên đầu
    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)

    items = []
    for p in files:
        st = p.stat()
        items.append({
            "name": p.name,
            "size_mb": round(st.st_size / (1024 * 1024), 2),  # bytes -> MB
            "mtime": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "ext": p.suffix.lower(),
        })
    return items


# ============================================================
# _load_df(file_path, job_id): ĐỌC FILE THÀNH DATAFRAME
# ============================================================
# - Ghi log "[INFO] Đang đọc file..."
# - Nếu excel -> pd.read_excel
# - Nếu csv -> pd.read_csv (utf-8-sig để tương thích file có BOM)
# - low_memory=False để pandas đọc ổn định dtype (tránh warning chunk)
def _load_df(file_path: Path, job_id: str) -> pd.DataFrame:
    _push(job_id, f"[INFO] Đang đọc file: {file_path.name}")
    if file_path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)
    # csv
    return pd.read_csv(file_path, encoding="utf-8-sig", low_memory=False)


# ============================================================
# _clean_dataframe(df, job_id): CLEAN DATA “AN TOÀN”
# ============================================================
# Đây là core cleaning logic (khá tổng quát cho nhiều nguồn file):
# - trim strings
# - convert numeric nếu đủ điều kiện
# - parse datetime cho các cột có tên gợi ý thời gian
# - fill missing:
#   + numeric -> median (nếu median nan thì =0)
#   + object  -> mode (nếu không có mode thì "unknown")
# - drop duplicates
#
# Đồng thời:
# - log tiến trình bằng _push
# - update progress bằng _set_progress
# - trả report (thống kê trước/sau) để UI hiển thị
def _clean_dataframe(df: pd.DataFrame, job_id: str):
    """
    Clean “an toàn” cho nhiều nguồn:
    - trim string
    - convert numeric nếu có thể
    - parse datetime nếu tên cột gợi ý
    - fill missing: numeric->median, text->mode/"unknown"
    - drop duplicates
    """
    report = {}

    # Log shape ban đầu
    _push(job_id, f"[INFO] Shape ban đầu: rows={len(df)} cols={df.shape[1]}")
    report["rows_before"] = int(len(df))
    report["cols"] = int(df.shape[1])

    # -------------------------
    # 1) Profiling missing/duplicates
    # -------------------------
    _set_progress(job_id, 15, "Profiling missing/duplicates")

    # missing_before:
    # - tổng số ô bị NaN trong toàn dataframe
    missing_before = int(df.isna().sum().sum())

    # dup_before:
    # - số dòng duplicate hoàn toàn (tất cả cột giống nhau)
    dup_before = int(df.duplicated().sum())

    report["missing_before"] = missing_before
    report["duplicates_before"] = dup_before

    _push(job_id, f"[INFO] Missing trước: {missing_before}")
    _push(job_id, f"[INFO] Duplicate trước: {dup_before}")

    # -------------------------
    # 2) Chuẩn hoá string & numeric
    # -------------------------
    _set_progress(job_id, 30, "Chuẩn hoá string & numeric")

    # Lấy danh sách cột kiểu object (thường là string/mixed)
    obj_cols = df.select_dtypes(include=["object"]).columns

    # Với mỗi cột object:
    # - ép về str để strip (cắt khoảng trắng)
    # - replace các string "nan"/"None"/"" thành NaN thực sự
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})

    # Cố convert object -> numeric nếu đủ “tín hiệu”
    # - replace "," -> "." để parse số kiểu "12,5"
    # - errors="coerce": không parse được thì NaN
    # Điều kiện để chấp nhận convert:
    # - số lượng giá trị parse được (notna) >= max(5, 60% số non-null)
    # -> tránh convert nhầm cột text thành numeric khi dữ liệu lẫn lộn
    for c in obj_cols:
        s = pd.to_numeric(df[c].str.replace(",", ".", regex=False), errors="coerce")
        if s.notna().sum() >= max(5, int(0.6 * len(df[c].dropna()))):
            df[c] = s

    # -------------------------
    # 3) Parse datetime (nếu có)
    # -------------------------
    _set_progress(job_id, 45, "Parse datetime (nếu có)")

    # candidates:
    # - các cột có tên gợi ý liên quan thời gian:
    #   "time", "date", "timestamp", "ngày", "giờ", "cap nhat", "cập nhật"
    candidates = []
    for c in df.columns:
        name = str(c).lower()
        if any(k in name for k in ["time", "date", "timestamp", "ngày", "giờ", "cap nhat", "cập nhật"]):
            candidates.append(c)

    # Với mỗi cột candidate:
    # - pd.to_datetime(errors="coerce") để parse an toàn
    # - dayfirst=True: ưu tiên kiểu ngày Việt Nam (dd/mm/yyyy)
    # Điều kiện chấp nhận:
    # - parsed.notna() >= max(5, 30% tổng dòng)
    # -> tránh parse nhầm cột không phải datetime
    for c in candidates:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce", dayfirst=True, utc=False)
            if parsed.notna().sum() >= max(5, int(0.3 * len(df))):
                df[c] = parsed
                _push(job_id, f"[INFO] Parsed datetime column: {c}")
        except Exception:
            pass  # nuốt lỗi parse để không làm crash toàn pipeline

    # -------------------------
    # 4) Impute missing
    # -------------------------
    _set_progress(job_id, 60, "Impute missing")

    # num_cols: các cột số (np.number)
    num_cols = df.select_dtypes(include=[np.number]).columns

    # obj_cols: các cột object (text)
    obj_cols = df.select_dtypes(include=["object"]).columns

    # Fill missing numeric bằng median (robust hơn mean khi có outliers)
    for c in num_cols:
        med = df[c].median(skipna=True)
        if pd.isna(med):
            med = 0  # nếu median không tính được -> fallback 0
        df[c] = df[c].fillna(med)

    # Fill missing text bằng mode (giá trị xuất hiện nhiều nhất)
    # - nếu mode không có -> fallback "unknown"
    for c in obj_cols:
        mode = None
        try:
            mode = df[c].mode(dropna=True)
            mode = mode.iloc[0] if not mode.empty else None
        except Exception:
            mode = None
        df[c] = df[c].fillna(mode if mode is not None else "unknown")

    # -------------------------
    # 5) Drop duplicates
    # -------------------------
    _set_progress(job_id, 75, "Drop duplicates")
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    report["duplicates_removed"] = int(removed)
    _push(job_id, f"[INFO] Removed duplicates: {removed}")

    # -------------------------
    # 6) Final check
    # -------------------------
    _set_progress(job_id, 90, "Final check")
    missing_after = int(df.isna().sum().sum())
    report["missing_after"] = missing_after
    report["rows_after"] = int(len(df))
    _push(job_id, f"[INFO] Missing sau: {missing_after}")
    _push(job_id, f"[INFO] Shape sau: rows={len(df)} cols={df.shape[1]}")

    # Done cleaning
    _set_progress(job_id, 95, "Done cleaning")
    return df, report


# ============================================================
# _worker(job_id, source, filename): THREAD WORKER CHẠY NỀN
# ============================================================
# Mục đích:
# - Khi user bấm "start cleaning", view sẽ tạo job_id và spawn thread này
# - Thread sẽ:
#   1) xác định thư mục input/output theo source (merge/output)
#   2) chọn file cần clean (nếu filename không truyền -> lấy file mới nhất)
#   3) đọc file vào df
#   4) clean df
#   5) ghi file cleaned_...csv
#   6) cập nhật trạng thái job trong _JOBS (done/result/progress/error)
def _worker(job_id: str, source: str, filename: str | None):
    try:
        _push(job_id, "========== START CLEAN ==========")
        _push(job_id, f"[INFO] source={source}")
        _set_progress(job_id, 5, "Chuẩn bị thư mục")

        # Đảm bảo các thư mục tồn tại trước khi làm
        MERGE_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        CLEANED_MERGE_DIR.mkdir(parents=True, exist_ok=True)
        CLEANED_RAW_DIR.mkdir(parents=True, exist_ok=True)

        # --------------------------------------------------------
        # Chọn input dir + output dir theo source
        # --------------------------------------------------------
        if source == "merge":
            in_dir = MERGE_DIR
            out_dir = CLEANED_MERGE_DIR
            out_folder_key = "cleaned_merge"  # key dùng để build URL view/download
        elif source == "output":
            in_dir = OUTPUT_DIR
            out_dir = CLEANED_RAW_DIR
            out_folder_key = "cleaned_raw"
        else:
            # Source không hợp lệ => raise
            raise ValueError("Invalid source")

        # --------------------------------------------------------
        # Nếu không truyền filename: auto chọn file mới nhất
        # --------------------------------------------------------
        if not filename:
            items = _scan_files(in_dir)  # list file đã sort mới nhất trước
            if not items:
                raise FileNotFoundError("Không có file nào trong thư mục nguồn.")
            filename = items[0]["name"]  # file mới nhất

        # Validate & pick file an toàn
        file_path = _safe_pick_file(in_dir, filename)

        # --------------------------------------------------------
        # Đọc dữ liệu
        # --------------------------------------------------------
        _set_progress(job_id, 10, "Đọc dữ liệu")
        df = _load_df(file_path, job_id)

        # --------------------------------------------------------
        # Clean dữ liệu
        # --------------------------------------------------------
        _set_progress(job_id, 20, "Clean dữ liệu")
        df2, report = _clean_dataframe(df, job_id)

        # --------------------------------------------------------
        # Ghi file output CSV
        # --------------------------------------------------------
        _set_progress(job_id, 96, "Ghi file output")

        # ts: timestamp để tên file output không bị trùng
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # stem: tên file không có đuôi
        stem = file_path.stem

        # out_name: tên file cleaned (có chứa source + stem + ts)
        out_name = f"cleaned_{source}_{stem}_{ts}.csv"

        # out_path: đường dẫn đầy đủ
        out_path = out_dir / out_name

        # to_csv: encode utf-8-sig để Excel dễ mở tiếng Việt (BOM)
        df2.to_csv(out_path, index=False, encoding="utf-8-sig")

        # tính size MB để hiển thị UI
        size_mb = round(out_path.stat().st_size / (1024 * 1024), 2)
        _push(job_id, f"[INFO] Saved: {out_name} ({size_mb} MB)")
        _push(job_id, "========== DONE ==========")

        # --------------------------------------------------------
        # Cập nhật job state trong _JOBS: done/result/progress...
        # --------------------------------------------------------
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
            if job:
                job["done"] = True
                job["error"] = None
                job["finished_at"] = _now()
                job["result"] = {
                    "source": source,
                    "input_file": file_path.name,
                    "output_file": out_name,
                    "output_folder_key": out_folder_key,
                    "size_mb": size_mb,
                    "report": report,
                    # Các URL để UI mở xem/tải file cleaned
                    "view_url": f"/datasets/view/{out_folder_key}/{out_name}/",
                    "download_url": f"/datasets/download/{out_folder_key}/{out_name}/",
                }
                # progress 100% khi hoàn thành
                job["progress"] = {"pct": 100, "step": "Hoàn thành"}

    except Exception as e:
        # Nếu có lỗi ở bất kỳ bước nào:
        # - ghi log error
        # - đánh dấu job done=true nhưng error != None
        _push(job_id, f"[ERROR] {type(e).__name__}: {e}")
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
            if job:
                job["done"] = True
                job["error"] = f"{type(e).__name__}: {e}"
                job["finished_at"] = _now()
                job["progress"] = {"pct": 100, "step": "Lỗi"}


# ============================================================
# clean_files_list_view: API TRẢ DANH SÁCH FILE CHO UI
# ============================================================
# Method: GET
# Query param: source=merge|output
# - merge  -> scan MERGE_DIR
# - output -> scan OUTPUT_DIR
# Response:
# - ok=True + files: danh sách metadata file
@require_http_methods(["GET"])
def clean_files_list_view(request):
    source = (request.GET.get("source") or "").strip().lower()
    if source == "merge":
        items = _scan_files(MERGE_DIR)
    elif source == "output":
        items = _scan_files(OUTPUT_DIR)
    else:
        return JsonResponse({"ok": False, "message": "source phải là merge/output"}, status=400)

    return JsonResponse({"ok": True, "source": source, "files": items})


# ============================================================
# clean_data_start_view: API START CLEAN JOB (TẠO THREAD)
# ============================================================
# Method: POST
# Body có thể là:
# - JSON: {"source": "...", "filename": "..."}
# - Form-data: source=...&filename=...
#
# Luồng:
# 1) đọc source/filename tuỳ content-type
# 2) validate source
# 3) tạo job_id mới (uuid)
# 4) tạo entry trong _JOBS
# 5) spawn thread chạy _worker
# 6) trả JSON {ok:true, job_id:...}
@require_http_methods(["POST"])
def clean_data_start_view(request):
    source = None
    filename = None

    # Lấy content-type để quyết định parse JSON hay form
    ct = (request.headers.get("content-type") or "").lower()

    # Nếu request gửi JSON
    if "application/json" in ct:
        try:
            payload = json.loads(request.body.decode("utf-8"))
        except Exception:
            payload = {}
        source = (payload.get("source") or "").strip().lower()
        filename = (payload.get("filename") or "").strip() or None
    else:
        # Nếu không phải JSON thì đọc từ request.POST (form-data)
        source = (request.POST.get("source") or "").strip().lower()
        filename = (request.POST.get("filename") or "").strip() or None

    # Validate source
    if source not in ("merge", "output"):
        return JsonResponse({"ok": False, "message": "source phải là merge hoặc output"}, status=400)

    # Tạo job_id dạng hex (uuid4)
    job_id = uuid.uuid4().hex

    # Khởi tạo job trong _JOBS
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "started_at": _now(),
            "finished_at": None,
            "done": False,  # chưa xong
            "error": None,  # chưa có lỗi
            "logs": [],     # list log lines
            "progress": {"pct": 0, "step": "Khởi tạo"},
            "result": None, # kết quả cuối sẽ được worker set
        }

    # Tạo thread chạy _worker ở background
    # daemon=True: nếu process server tắt thì thread cũng tắt theo
    t = threading.Thread(target=_worker, args=(job_id, source, filename), daemon=True)
    t.start()

    # Trả về job_id để frontend poll logs/progress
    return JsonResponse({"ok": True, "job_id": job_id})


# ============================================================
# clean_data_tail_view: API TAIL LOGS/PROGRESS CỦA JOB
# ============================================================
# Method: GET
# Query:
# - job_id: id của job cần lấy logs
# - since: offset (index) trong list logs
#
# Response:
# - lines: logs mới kể từ since
# - next_since: offset mới cho lần poll tiếp theo
# - done/error/progress/result: trạng thái job hiện tại
@require_http_methods(["GET"])
def clean_data_tail_view(request):
    # Lấy job_id từ query string
    job_id = (request.GET.get("job_id") or "").strip()

    # since: offset log client đã nhận (default "0")
    since = request.GET.get("since", "0")
    try:
        since_i = max(0, int(since))
    except Exception:
        since_i = 0

    # Lấy job từ _JOBS (thread-safe)
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)

    # Nếu job_id không tồn tại -> 404
    if not job:
        return JsonResponse({"ok": False, "message": "job_id không tồn tại"}, status=404)

    # logs: list toàn bộ logs hiện có
    logs = job["logs"]

    # new_lines: chỉ lấy phần logs từ offset since_i trở đi
    new_lines = logs[since_i:]

    # next_since: offset mới = since cũ + số dòng mới vừa lấy
    next_since = since_i + len(new_lines)

    # Trả JSON cho frontend:
    # - done/error để biết job xong hay lỗi
    # - progress để hiển thị progress bar
    # - result để hiển thị link view/download khi xong
    return JsonResponse({
        "ok": True,
        "job_id": job_id,
        "done": job["done"],
        "error": job["error"],
        "progress": job["progress"],
        "started_at": job["started_at"],
        "finished_at": job["finished_at"],
        "lines": new_lines,
        "next_since": next_since,
        "result": job["result"],
    })


# ============================================================
# ALIAS: clean_data_view trỏ về clean_data_start_view
# ============================================================
# Ý nghĩa:
# - Có thể trước đó bạn dùng tên view là clean_data_view
# - Giờ bạn tách ra start_view/tail_view rõ ràng
# - Nhưng vẫn muốn giữ tên cũ (đỡ sửa urls.py/template/frontend)
clean_data_view = clean_data_start_view
