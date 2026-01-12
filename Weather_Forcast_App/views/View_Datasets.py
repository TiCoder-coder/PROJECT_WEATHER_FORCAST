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

def _base_dir() -> Path:
    """
    Trả về thư mục Weather_Forcast_App
    """
    base = Path(settings.BASE_DIR)
    
    if base.name == "Weather_Forcast_App":
        return base
    
    weather_app_dir = base / "Weather_Forcast_App"
    if weather_app_dir.exists():
        return weather_app_dir
    
    print(f"WARNING: Could not find Weather_Forcast_App directory. Using BASE_DIR: {base}")
    return base


def _output_dir() -> Path:
    """Trả về thư mục output"""
    return _base_dir() / "output"


def _merged_dir() -> Path:
    """Trả về thư mục Merge_data"""
    return _base_dir() / "Merge_data"


def _cleaned_dir() -> Path:
    """Trả về thư mục cleaned_data"""
    return _base_dir() / "cleaned_data"
def _cleaned_merge_dir():
    return _cleaned_dir() / "Clean_Data_For_File_Merge"
def _cleaned_not_merge_dir() -> Path:
    """cleaned_data/Clean_Data_For_File_Not_Merge"""
    return _cleaned_dir() / "Clean_Data_For_File_Not_Merge"
def _cleaned_raw_dir():
    return _cleaned_dir() / "Clean_Data_For_File_Not_Merge"

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


def _safe_join(base_dir: Path, filename: str) -> Path:
    """
    Kiểm tra và trả về đường dẫn file an toàn
    """
    base = base_dir.resolve()
    p = (base / filename).resolve()
    
    if base not in p.parents and p != base:
        raise Http404("Invalid path")
    
    if not p.exists() or not p.is_file():
        raise Http404("File not found")
    
    return p

def _get_files_info(folder_path: Path, folder_key: str | None = None):
    if not folder_path.exists():
        return []

    items = []
    for path in folder_path.iterdir():
        if path.is_file() and path.suffix.lower() in [".csv", ".xlsx", ".xls"]:
            st = path.stat()

            dt = datetime.fromtimestamp(st.st_mtime, tz=dj_tz.get_current_timezone())
            mtime_str = dj_tz.localtime(dt).strftime("%Y-%m-%d %H:%M:%S")

            item = {
                "name": path.name,
                "mtime": mtime_str,
                "mtime_ts": st.st_mtime,
                "size_mb": round(st.st_size / (1024 * 1024), 2),
                "ext": path.suffix.lower(),
                "folder": folder_key,
            }
            if folder_key:
                item["folder"] = folder_key

            items.append(item)

    items.sort(key=lambda x: x.get("mtime_ts", 0), reverse=True)
    return items



def _tag(items, folder_key: str):
    return [dict(x, folder=folder_key) for x in items]

def datasets_view(request):
    _output_dir().mkdir(parents=True, exist_ok=True)
    _merged_dir().mkdir(parents=True, exist_ok=True)
    cleaned_dir = _cleaned_dir()
    cleaned_merge_dir = _cleaned_merge_dir()
    cleaned_raw_dir = _cleaned_not_merge_dir()


    output_items = _get_files_info(_output_dir(), "output")
    merged_items = _get_files_info(_merged_dir(), "merged")

    cleaned_root_items  = _get_files_info(_cleaned_dir(), "cleaned")
    cleaned_merge_items = _get_files_info(_cleaned_merge_dir(), "cleaned_merge")
    cleaned_raw_items   = _get_files_info(_cleaned_raw_dir(), "cleaned_raw")

    cleaned_items = sorted(
        cleaned_root_items + cleaned_merge_items + cleaned_raw_items,
        key=lambda x: x.get("mtime_ts", 0),
        reverse=True
    )


    latest_output = output_items[0] if output_items else None
    latest_merged = merged_items[0] if merged_items else None
    latest_cleaned = cleaned_items[0] if cleaned_items else None

    latest_cleaned_merge = cleaned_merge_items[0] if cleaned_merge_items else None
    latest_cleaned_raw = cleaned_raw_items[0] if cleaned_raw_items else None

    tab = (request.GET.get("tab") or "recent").lower()
    if tab not in ("recent", "process"):
        tab = "recent"


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

    context["active_tab"] = tab
    return render(request, "weather/Datasets.html", context)



def dataset_download_view(request, folder: str, filename: str):
    base_dir = _folder_to_dir(folder)
    if not base_dir:
        raise Http404("Invalid folder")

    p = _safe_join(base_dir, filename)
    content_type, _ = mimetypes.guess_type(str(p))
    resp = FileResponse(open(p, "rb"), content_type=content_type or "application/octet-stream")
    resp["Content-Disposition"] = f'attachment; filename="{p.name}"'
    return resp
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


def dataset_view_view(request, folder: str, filename: str):
    base_dir = _folder_to_dir(folder)
    if not base_dir:
        raise Http404("Invalid folder")

    p = _safe_join(base_dir, filename)

    file_type = _get_file_type(filename)

    
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    
    if file_type in ['csv', 'excel']:
        try:
            page = int(request.GET.get('page', 1))
            rows_per_page = 100
            start_row = (page - 1) * rows_per_page
            end_row = start_row + rows_per_page
            
            if file_type == 'csv':
                if is_ajax:
                    df = pd.read_csv(p, encoding='utf-8', skiprows=range(1, start_row), nrows=rows_per_page)
                    total_rows = 0
                    with open(p, 'r', encoding='utf-8') as f:
                        total_rows = sum(1 for line in f) - 1
                else:
                    df = pd.read_csv(p, encoding='utf-8', nrows=rows_per_page)
                    total_rows = 0
                    with open(p, 'r', encoding='utf-8') as f:
                        total_rows = sum(1 for line in f) - 1
            else:
                if is_ajax:
                    df_full = pd.read_excel(p, engine='openpyxl')
                    total_rows = len(df_full)
                    df = df_full.iloc[start_row:end_row]
                else:
                    df = pd.read_excel(p, engine='openpyxl', nrows=rows_per_page)
                    df_full = pd.read_excel(p, engine='openpyxl')
                    total_rows = len(df_full)
            
            if is_ajax:
                data = {
                    'data': df.fillna('').to_dict('records'),
                    'page': page,
                    'total_rows': total_rows,
                    'has_more': end_row < total_rows
                }
                return HttpResponse(json.dumps(data, default=str), content_type='application/json')
            
            html_table = df.fillna('').to_html(
                classes='table table-striped table-bordered',
                index=False,
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
            
            return render(request, 'weather/dataset_preview.html', context)
            
        except Exception as e:
            error_context = {
                'error_title': 'Lỗi khi đọc file',
                'error_message': f'Không thể đọc file {escape(filename)}',
                'error_detail': str(e),
                'back_url': request.META.get('HTTP_REFERER', '/'),
            }
            return render(request, 'weather/error.html', error_context, status=500)
    
    else:
        try:
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
            
            return render(request, 'weather/dataset_preview.html', context)
            
        except UnicodeDecodeError:
            try:
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
                
                return render(request, 'weather/dataset_preview.html', context)
                
            except Exception as e:
                content_type, _ = mimetypes.guess_type(str(p))
                resp = FileResponse(open(p, "rb"), content_type=content_type or "application/octet-stream")
                resp["Content-Disposition"] = f'inline; filename="{p.name}"'
                return resp
                
        except Exception as e:
            error_context = {
                'error_title': 'Lỗi khi đọc file',
                'error_message': f'Không thể đọc file {escape(filename)}',
                'error_detail': str(e),
                'back_url': request.META.get('HTTP_REFERER', '/'),
            }
            return render(request, 'weather/error.html', error_context, status=500)
        
def _cleaned_merge_dir() -> Path:
    return _cleaned_dir() / "Clean_Data_For_File_Merge"

def _cleaned_raw_dir() -> Path:
    return _cleaned_dir() / "Clean_Data_For_File_Not_Merge"
