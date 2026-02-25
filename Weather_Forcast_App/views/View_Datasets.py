import mimetypes
from pathlib import Path
from datetime import datetime
import pandas as pd
import json
from django.conf import settings
from django.http import FileResponse, Http404, HttpResponse
from django.shortcuts import render
from django.utils.html import escape
from datetime import datetime
from django.utils import timezone as dj_tz

# ============================================================
# MỤC TIÊU FILE NÀY
# ============================================================
# File này là "Datasets module" trong Django:
# - Liệt kê danh sách dataset ở nhiều thư mục:
#   + output/         : file crawl raw từ các pipeline
#   + Merge_data/     : file đã merge
#   + cleaned_data/   : file đã làm sạch (có thể chia nhiều nhánh)
# - Cho phép:
#   + xem preview dataset (CSV/Excel render bảng HTML + phân trang)
#   + tải xuống dataset (download)
#   + xử lý an toàn path (ngăn path traversal)
#
# Ngoài ra:
# - Hỗ trợ xem dạng AJAX (frontend load thêm trang theo page)
# - Có hiển thị thời gian local theo timezone Django


# ============================================================
# _base_dir(): XÁC ĐỊNH THƯ MỤC GỐC Weather_Forcast_App
# ============================================================
# Trả về thư mục Weather_Forcast_App để các hàm khác build đúng path.
#
# Logic:
# 1) base = settings.BASE_DIR (thường là root project Django)
# 2) nếu BASE_DIR đã trỏ đúng vào Weather_Forcast_App => return luôn
# 3) nếu không: thử base/Weather_Forcast_App tồn tại => dùng nó
# 4) nếu vẫn không có: cảnh báo và fallback về BASE_DIR
#
# Mục đích:
# - Dự án của bạn có lúc chạy từ nhiều nơi (manage.py khác thư mục),
#   nên cần "tự dò" thư mục đúng để tránh lỗi không tìm thấy folder.
def _base_dir() -> Path:
    """
    Trả về thư mục Weather_Forcast_App
    """
    base = Path(settings.BASE_DIR)
    
    # Nếu BASE_DIR đã chính là thư mục Weather_Forcast_App
    if base.name == "Weather_Forcast_App":
        return base
    
    # Nếu BASE_DIR là root dự án, thì Weather_Forcast_App nằm dưới nó
    weather_app_dir = base / "Weather_Forcast_App"
    if weather_app_dir.exists():
        return weather_app_dir
    
    # Nếu không tìm thấy, fallback: dùng BASE_DIR để khỏi crash
    print(f"WARNING: Could not find Weather_Forcast_App directory. Using BASE_DIR: {base}")
    return base


# ============================================================
# CÁC HÀM TRẢ VỀ THƯ MỤC CHUẨN (OUTPUT / MERGE / CLEANED)
# ============================================================

def _output_dir() -> Path:
    """Trả về thư mục output"""
    # Dynamic path: tự tính từ vị trí project, không hardcode
    from Weather_Forcast_App.paths import DATA_CRAWL_DIR
    DATA_CRAWL_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_CRAWL_DIR


def _merged_dir() -> Path:
    """Trả về thư mục Merge_data"""
    # Dynamic path: tự tính từ vị trí project, không hardcode
    from Weather_Forcast_App.paths import DATA_MERGE_DIR
    DATA_MERGE_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_MERGE_DIR


def _cleaned_dir() -> Path:
    """Trả về thư mục cleaned_data"""
    # cleaned_data/ chứa file clean sau xử lý missing/duplicate/...
    return _base_dir() / "cleaned_data"


# ============================================================
# CÁC NHÁNH THƯ MỤC CLEANED (MERGE / RAW)
# ============================================================
# Lưu ý: trong code của bạn có 2 lần định nghĩa _cleaned_merge_dir và _cleaned_raw_dir
# - Python sẽ lấy định nghĩa ở dưới cùng (định nghĩa sau sẽ ghi đè định nghĩa trước)
# - Bạn yêu cầu "không đổi code" nên chỉ ghi chú để bạn hiểu hành vi này.

def _cleaned_merge_dir():
    # cleaned_data/Clean_Data_For_File_Merge
    # - chứa file clean cho nguồn "merged"
    return _cleaned_dir() / "Clean_Data_For_File_Merge"

def _cleaned_not_merge_dir() -> Path:
    """cleaned_data/Clean_Data_For_File_Not_Merge"""
    # cleaned_data/Clean_Data_For_File_Not_Merge
    # - chứa file clean cho nguồn "output" (raw)
    return _cleaned_dir() / "Clean_Data_For_File_Not_Merge"

def _cleaned_raw_dir():
    # Ở đây _cleaned_raw_dir trả về cùng path với _cleaned_not_merge_dir
    # => "raw" nghĩa là dữ liệu output chưa merge nhưng đã clean
    return _cleaned_dir() / "Clean_Data_For_File_Not_Merge"


# ============================================================
# _folder_to_dir(folder): MAP "KEY" -> THƯ MỤC THỰC TẾ
# ============================================================
# Khi user truy cập URL dạng:
#   /datasets/view/<folder>/<filename>/
# hoặc:
#   /datasets/download/<folder>/<filename>/
#
# folder sẽ là 1 key (ví dụ: "output", "merged", "cleaned_merge"...)
# Hàm này map key đó về đúng Path.
#
# Nếu key không nằm trong mapping => return None (sau đó raise 404).
def _folder_to_dir(folder: str) -> Path | None:
    key = (folder or "").strip().lower()

    mapping = {
        "output": _output_dir(),
        "merged": _merged_dir(),
        "cleaned": _cleaned_dir(),
        "cleaned_merge": _cleaned_merge_dir(),
        "cleaned_raw": _cleaned_raw_dir(),
    }
    return mapping.get(key)


# ============================================================
# _safe_join(base_dir, filename): CHỐNG PATH TRAVERSAL
# ============================================================
# Mục đích bảo mật:
# - Nếu user truyền filename kiểu: "../../etc/passwd"
#   thì resolve() sẽ trỏ ra ngoài base_dir -> phải chặn.
#
# Logic:
# - base = base_dir.resolve(): lấy path chuẩn tuyệt đối
# - p = (base/filename).resolve(): join rồi resolve để loại bỏ ../
# - nếu base không nằm trong parents của p và p != base => invalid path
# - kiểm tra tồn tại + là file
# - trả về path hợp lệ
def _safe_join(base_dir: Path, filename: str) -> Path:
    """
    Kiểm tra và trả về đường dẫn file an toàn
    """
    base = base_dir.resolve()
    p = (base / filename).resolve()
    
    # Nếu p "chạy ra ngoài" base => reject
    if base not in p.parents and p != base:
        raise Http404("Invalid path")
    
    # Không tồn tại hoặc không phải file => 404
    if not p.exists() or not p.is_file():
        raise Http404("File not found")
    
    return p


# ============================================================
# _get_files_info(folder_path, folder_key): LẤY THÔNG TIN FILES
# ============================================================
# Trả về list dict gồm metadata cho các file dataset.
#
# Chỉ lấy file có đuôi: .csv, .xlsx, .xls
#
# Mỗi item gồm:
# - name: tên file
# - mtime: thời gian sửa đổi cuối (string theo timezone Django)
# - mtime_ts: timestamp float (dùng sort)
# - size_mb: dung lượng MB
# - ext: extension lower
# - folder: key folder (output/merged/cleaned_merge/...)
#
# Lưu ý timezone:
# - dùng dj_tz.get_current_timezone() và dj_tz.localtime()
# - giúp hiển thị đúng timezone bạn cấu hình trong Django.
def _get_files_info(folder_path: Path, folder_key: str | None = None):
    if not folder_path.exists():
        return []

    items = []
    for path in folder_path.iterdir():
        if path.is_file() and path.suffix.lower() in [".csv", ".xlsx", ".xls"]:
            st = path.stat()

            # Convert st_mtime -> datetime theo timezone hiện tại của Django
            dt = datetime.fromtimestamp(st.st_mtime, tz=dj_tz.get_current_timezone())
            mtime_str = dj_tz.localtime(dt).strftime("%Y-%m-%d %H:%M:%S")

            item = {
                "name": path.name,
                "mtime": mtime_str,                          # chuỗi thời gian hiển thị
                "mtime_ts": st.st_mtime,                     # timestamp dùng sort
                "size_mb": round(st.st_size / (1024 * 1024), 2),
                "ext": path.suffix.lower(),
                "folder": folder_key,
            }
            # Nếu có folder_key thì set rõ (dù đã set ở trên)
            if folder_key:
                item["folder"] = folder_key

            items.append(item)

    # sort file mới nhất lên đầu
    items.sort(key=lambda x: x.get("mtime_ts", 0), reverse=True)
    return items


# ============================================================
# _tag(items, folder_key): GẮN THÊM folder KEY CHO TỪNG ITEM
# ============================================================
# Hàm tiện ích: trả về list mới, mỗi dict được merge thêm folder=folder_key
# (Trong code hiện tại bạn không dùng trực tiếp hàm này nhiều, nhưng để sẵn.)
def _tag(items, folder_key: str):
    return [dict(x, folder=folder_key) for x in items]


# ============================================================
# datasets_view: TRANG LIỆT KÊ DATASETS (OUTPUT / MERGED / CLEANED)
# ============================================================
# - đảm bảo các thư mục tồn tại (mkdir)
# - lấy danh sách file của từng thư mục bằng _get_files_info
# - gom cleaned items từ:
#   + cleaned_data/ (root)
#   + cleaned_data/Clean_Data_For_File_Merge
#   + cleaned_data/Clean_Data_For_File_Not_Merge
# - sort cleaned_items theo mtime_ts giảm dần
# - xác định latest của từng nhóm để hiển thị "mới nhất"
# - đọc query param tab để đổi UI (recent/process)
# - render template weather/Datasets.html
def datasets_view(request):
    # đảm bảo các folder tồn tại để tránh lỗi UI khi chưa có file
    _output_dir().mkdir(parents=True, exist_ok=True)
    _merged_dir().mkdir(parents=True, exist_ok=True)
    cleaned_dir = _cleaned_dir()
    cleaned_merge_dir = _cleaned_merge_dir()
    cleaned_raw_dir = _cleaned_not_merge_dir()

    # lấy file list ở output + merged
    output_items = _get_files_info(_output_dir(), "output")
    merged_items = _get_files_info(_merged_dir(), "merged")

    # lấy file list ở cleaned (root + merge + raw)
    cleaned_root_items  = _get_files_info(_cleaned_dir(), "cleaned")
    cleaned_merge_items = _get_files_info(_cleaned_merge_dir(), "cleaned_merge")
    cleaned_raw_items   = _get_files_info(_cleaned_raw_dir(), "cleaned_raw")

    # gộp tất cả cleaned vào 1 list rồi sort theo thời gian mới nhất
    cleaned_items = sorted(
        cleaned_root_items + cleaned_merge_items + cleaned_raw_items,
        key=lambda x: x.get("mtime_ts", 0),
        reverse=True
    )

    # lấy "file mới nhất" của mỗi nhóm để hiển thị nổi bật
    latest_output = output_items[0] if output_items else None
    latest_merged = merged_items[0] if merged_items else None
    latest_cleaned = cleaned_items[0] if cleaned_items else None

    latest_cleaned_merge = cleaned_merge_items[0] if cleaned_merge_items else None
    latest_cleaned_raw = cleaned_raw_items[0] if cleaned_raw_items else None

    # tab UI: recent hoặc process (mặc định recent)
    tab = (request.GET.get("tab") or "recent").lower()
    if tab not in ("recent", "process"):
        tab = "recent"

    # context đẩy sang template Datasets.html
    context = {
        "output_items": output_items,
        "merged_items": merged_items,
        "cleaned_items": cleaned_items,

        "latest_output": latest_output,
        "latest_merged": latest_merged,
        "latest_cleaned": latest_cleaned,

        "cleaned_merge_items": cleaned_merge_items,
        "cleaned_raw_items": cleaned_raw_items,
        "latest_cleaned_merge": latest_cleaned_merge,
        "latest_cleaned_raw": latest_cleaned_raw,
        "active_tab": tab,
    }

    # set lại active_tab (trùng lặp nhưng vẫn OK)
    context["active_tab"] = tab
    return render(request, "weather/Datasets.html", context)


# ============================================================
# dataset_download_view: DOWNLOAD FILE (ATTACHMENT)
# ============================================================
# URL dạng: /datasets/download/<folder>/<filename>/
#
# Bước:
# 1) map folder -> base_dir bằng _folder_to_dir
# 2) _safe_join để đảm bảo filename hợp lệ và không bị ../
# 3) guess content-type để browser hiểu loại file
# 4) FileResponse(open(...)) stream file về client
# 5) set Content-Disposition = attachment => bắt browser download
def dataset_download_view(request, folder: str, filename: str):
    base_dir = _folder_to_dir(folder)
    if not base_dir:
        raise Http404("Invalid folder")

    p = _safe_join(base_dir, filename)
    content_type, _ = mimetypes.guess_type(str(p))
    resp = FileResponse(open(p, "rb"), content_type=content_type or "application/octet-stream")
    resp["Content-Disposition"] = f'attachment; filename="{p.name}"'
    return resp


# ============================================================
# _get_file_type(filename): XÁC ĐỊNH LOẠI FILE THEO EXTENSION
# ============================================================
# Dùng trong dataset_view_view để quyết định:
# - csv/excel => đọc bằng pandas rồi render bảng
# - json/txt  => đọc text và render raw content
def _get_file_type(filename: str) -> str:
    """Xác định loại file từ extension"""
    ext = Path(filename).suffix.lower()
    if ext in ['.csv']:
        return 'csv'
    elif ext in ['.xlsx', '.xls']:
        return 'excel'
    elif ext == '.json':
        return 'json'
    else:
        return 'txt'


# ============================================================
# dataset_view_view: PREVIEW FILE (TABLE / TEXT) + HỖ TRỢ AJAX
# ============================================================
# URL dạng: /datasets/view/<folder>/<filename>/
#
# Hành vi:
# - Nếu CSV/Excel:
#   + đọc theo trang (page) với rows_per_page=10000
#   + nếu AJAX: trả JSON (records) để frontend append/load more
#   + nếu không AJAX: render HTML table (df.to_html) trong template preview
#
# - Nếu không phải CSV/Excel:
#   + cố gắng đọc text UTF-8
#   + nếu lỗi decode: đọc binary rồi decode ignore
#   + nếu vẫn lỗi: trả FileResponse inline
#
# Lưu ý:
# - với CSV: khi AJAX, dùng skiprows để "bỏ qua" phần đã đọc
# - với Excel: code đang đọc full df_full khi AJAX để slice (tốn RAM nếu file lớn)
#   (Bạn bảo không đổi code nên mình chỉ ghi chú để bạn hiểu.)
def dataset_view_view(request, folder: str, filename: str):
    base_dir = _folder_to_dir(folder)
    if not base_dir:
        raise Http404("Invalid folder")

    # lấy path an toàn
    p = _safe_join(base_dir, filename)

    # xác định loại file
    file_type = _get_file_type(filename)

    # check request có phải AJAX không (frontend fetch)
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    
    # ========================================================
    # CASE 1: CSV / EXCEL => đọc bằng pandas và render bảng
    # ========================================================
    if file_type in ['csv', 'excel']:
        try:
            # page hiện tại (mặc định 1)
            page = int(request.GET.get('page', 1))

            # số dòng mỗi trang (10k) để tránh load file quá lớn 1 lần
            rows_per_page = 10000

            # tính range index cần hiển thị
            start_row = (page - 1) * rows_per_page
            end_row = start_row + rows_per_page
            
            # ---------------- CSV ----------------
            if file_type == 'csv':
                if is_ajax:
                    # AJAX: chỉ đọc phần cần (skiprows từ 1..start_row vì row 0 là header)
                    df = pd.read_csv(p, encoding='utf-8', skiprows=range(1, start_row), nrows=rows_per_page)

                    # total_rows: đếm số dòng bằng cách đọc file text
                    # - trừ 1 vì dòng header
                    total_rows = 0
                    with open(p, 'r', encoding='utf-8') as f:
                        total_rows = sum(1 for line in f) - 1
                else:
                    # non-AJAX: đọc trang đầu tiên
                    df = pd.read_csv(p, encoding='utf-8', nrows=rows_per_page)

                    # total_rows: đếm tổng số dòng trong file
                    total_rows = 0
                    with open(p, 'r', encoding='utf-8') as f:
                        total_rows = sum(1 for line in f) - 1

            # ---------------- EXCEL ----------------
            else:
                if is_ajax:
                    # AJAX: đang đọc toàn bộ file Excel (df_full) rồi cắt slice
                    # - cách này dễ code nhưng nặng RAM nếu file lớn
                    df_full = pd.read_excel(p, engine='openpyxl')
                    total_rows = len(df_full)
                    df = df_full.iloc[start_row:end_row]
                else:
                    # non-AJAX: đọc 10k dòng đầu (nrows) để render nhanh
                    df = pd.read_excel(p, engine='openpyxl', nrows=rows_per_page)

                    # đồng thời đọc full để lấy total_rows
                    df_full = pd.read_excel(p, engine='openpyxl')
                    total_rows = len(df_full)
            
            # ----------------------------------------------------
            # Nếu AJAX => trả JSON (records) cho frontend
            # ----------------------------------------------------
            if is_ajax:
                data = {
                    'data': df.fillna('').to_dict('records'),  # convert DataFrame thành list dict
                    'page': page,                               # trang hiện tại
                    'total_rows': total_rows,                   # tổng số dòng trong file
                    'has_more': end_row < total_rows            # còn trang tiếp theo không
                }
                return HttpResponse(json.dumps(data, default=str), content_type='application/json')
            
            # ----------------------------------------------------
            # Non-AJAX => render HTML table (df.to_html)
            # ----------------------------------------------------
            html_table = df.fillna('').to_html(
                classes='table table-striped table-bordered',   # class bootstrap để đẹp hơn
                index=False,                                     # không show index pandas
                float_format=lambda x: '{:.2f}'.format(x) if isinstance(x, float) else str(x)
            )
            
            context = {
                'filename': filename,
                'folder': folder,
                'file_type': file_type,
                'file_size_kb': p.stat().st_size / 1024,
                'total_rows': total_rows,
                'rows_per_page': rows_per_page,
                'showing_rows': min(rows_per_page, total_rows),
                'html_table': html_table,
                'referer_url': request.META.get('HTTP_REFERER', '/'),
                'download_url': request.path.replace('/view/', '/download/'),
                'is_table': True,
            }
            
            return render(request, 'weather/Dataset_preview.html', context)
            
        except Exception as e:
            # Nếu lỗi đọc file CSV/Excel -> render trang Error.html
            error_context = {
                'error_title': 'Lỗi khi đọc file',
                'error_message': f'Không thể đọc file {escape(filename)}',
                'error_detail': str(e),
                'back_url': request.META.get('HTTP_REFERER', '/'),
            }
            return render(request, 'weather/Error.html', error_context, status=500)
    
    # ========================================================
    # CASE 2: FILE TEXT/JSON/TXT => đọc nội dung và render raw
    # ========================================================
    else:
        try:
            # đọc text UTF-8 bình thường
            with open(p, 'r', encoding='utf-8') as f:
                content = f.read()
            
            context = {
                'filename': filename,
                'folder': folder,
                'file_type': file_type,
                'file_size_kb': p.stat().st_size / 1024,
                'content': content,
                'referer_url': request.META.get('HTTP_REFERER', '/'),
                'download_url': request.path.replace('/view/', '/download/'),
                'is_table': False,
            }
            
            return render(request, 'weather/Dataset_preview.html', context)
            
        except UnicodeDecodeError:
            # Nếu file không decode được utf-8 (bị encoding lạ/binary)
            try:
                # đọc binary rồi decode ignore (bỏ ký tự lỗi)
                with open(p, 'rb') as f:
                    content = f.read().decode('utf-8', errors='ignore')
                
                context = {
                    'filename': filename,
                    'folder': folder,
                    'file_type': file_type,
                    'file_size_kb': p.stat().st_size / 1024,
                    'content': content,
                    'referer_url': request.META.get('HTTP_REFERER', '/'),
                    'download_url': request.path.replace('/view/', '/download/'),
                    'is_table': False,
                }
                
                return render(request, 'weather/Dataset_preview.html', context)
                
            except Exception as e:
                # Nếu vẫn lỗi: trả file inline (browser tự quyết định mở/download)
                content_type, _ = mimetypes.guess_type(str(p))
                resp = FileResponse(open(p, "rb"), content_type=content_type or "application/octet-stream")
                resp["Content-Disposition"] = f'inline; filename="{p.name}"'
                return resp
                
        except Exception as e:
            # Lỗi khác khi đọc file text -> render Error.html
            error_context = {
                'error_title': 'Lỗi khi đọc file',
                'error_message': f'Không thể đọc file {escape(filename)}',
                'error_detail': str(e),
                'back_url': request.META.get('HTTP_REFERER', '/'),
            }
            return render(request, 'weather/Error.html', error_context, status=500)
        

# ============================================================
# CHÚ Ý: ĐỊNH NGHĨA LẠI HÀM (OVERRIDE) Ở CUỐI FILE
# ============================================================
# Ở cuối code bạn định nghĩa lại _cleaned_merge_dir và _cleaned_raw_dir.
# Trong Python:
# - function định nghĩa sau sẽ ghi đè function cùng tên trước đó.
# Vì vậy hai hàm dưới đây mới là phiên bản "có hiệu lực" cuối cùng.

def _cleaned_merge_dir() -> Path:
    # cleaned_data/Clean_Data_For_File_Merge
    return _cleaned_dir() / "Clean_Data_For_File_Merge"

def _cleaned_raw_dir() -> Path:
    # cleaned_data/Clean_Data_For_File_Not_Merge
    return _cleaned_dir() / "Clean_Data_For_File_Not_Merge"
