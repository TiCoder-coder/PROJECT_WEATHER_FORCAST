import mimetypes
from pathlib import Path
from datetime import datetime
import pandas as pd
from django.conf import settings
from django.http import FileResponse, Http404, HttpResponse
from django.shortcuts import render, redirect
from django.utils.html import escape
import json


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


def _merge_dir() -> Path:
    """Trả về thư mục Merge_data"""
    return _base_dir() / "Merge_data"


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


def _get_files_info(directory: Path) -> list:
    """
    Lấy thông tin các file trong thư mục
    """
    print(f"DEBUG: Checking directory: {directory}")
    print(f"DEBUG: Directory exists: {directory.exists()}")
    
    if not directory.exists():
        print(f"DEBUG: Directory does not exist, creating: {directory}")
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"ERROR: Could not create directory: {e}")
        return []
    
    patterns = ["*.xlsx", "*.csv", "*.json", "*.txt"]
    files = []
    for pat in patterns:
        found = list(directory.glob(pat))
        print(f"DEBUG: Pattern '{pat}' found {len(found)} files")
        files.extend(found)

    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    print(f"DEBUG: Total files found: {len(files)}")

    items = []
    for p in files:
        st = p.stat()
        items.append({
            "name": p.name,
            "size_mb": round(st.st_size / (1024 * 1024), 2),
            "mtime": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        })

    return items


def datasets_view(request):
    """
    View hiển thị danh sách file từ cả thư mục output và Merge_data
    """
    base = _base_dir()
    print(f"DEBUG: BASE_DIR from settings: {settings.BASE_DIR}")
    print(f"DEBUG: Calculated base_dir: {base}")
    
    output_dir = _output_dir()
    merge_dir = _merge_dir()
    
    print(f"DEBUG: Output dir: {output_dir}")
    print(f"DEBUG: Merge dir: {merge_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    merge_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Scanning OUTPUT directory ===")
    output_items = _get_files_info(output_dir)
    latest_output = output_items[0] if output_items else None

    print(f"\n=== Scanning MERGE_DATA directory ===")
    merged_items = _get_files_info(merge_dir)
    latest_merged = merged_items[0] if merged_items else None

    print(f"\nDEBUG: Returning {len(output_items)} output files and {len(merged_items)} merged files")

    return render(request, "weather/Datasets.html", {
        "output_items": output_items,
        "latest_output": latest_output,
        "merged_items": merged_items,
        "latest_merged": latest_merged,
    })


def dataset_download_view(request, folder: str, filename: str):
    """
    View tải xuống file từ thư mục output hoặc Merge_data
    folder: 'output' hoặc 'merged'
    """
    if folder == "output":
        base_dir = _output_dir()
    elif folder == "merged":
        base_dir = _merge_dir()
    else:
        raise Http404("Invalid folder")
    
    p = _safe_join(base_dir, filename)
    content_type, _ = mimetypes.guess_type(str(p))
    resp = FileResponse(open(p, "rb"), content_type=content_type or "application/octet-stream")
    resp["Content-Disposition"] = f'attachment; filename="{p.name}"'
    return resp


def dataset_view_view(request, folder: str, filename: str):
    """
    View xem trước file từ thư mục output hoặc Merge_data
    Tối ưu: Chỉ hiển thị 100 dòng đầu tiên và tải thêm khi cần
    """
    if folder == "output":
        base_dir = _output_dir()
    elif folder == "merged":
        base_dir = _merge_dir()
    else:
        raise Http404("Invalid folder")
    
    p = _safe_join(base_dir, filename)
    
    # Kiểm tra nếu là request AJAX để lấy thêm dữ liệu
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    page = int(request.GET.get('page', 1))
    rows_per_page = 100
    start_row = (page - 1) * rows_per_page
    end_row = start_row + rows_per_page
    
    # Kiểm tra loại file
    if p.suffix.lower() in ['.xlsx', '.xls', '.csv']:
        try:
            # Đọc file với chunk để tối ưu
            if p.suffix.lower() == '.csv':
                # Đọc file CSV với chunk
                if is_ajax:
                    # Chỉ đọc phần cần thiết cho AJAX request
                    df = pd.read_csv(p, encoding='utf-8', skiprows=range(1, start_row), nrows=rows_per_page)
                    total_rows = 0
                    with open(p, 'r', encoding='utf-8') as f:
                        total_rows = sum(1 for line in f) - 1  # Trừ header
                else:
                    # Đọc 100 dòng đầu cho trang đầu
                    df = pd.read_csv(p, encoding='utf-8', nrows=rows_per_page)
                    total_rows = 0
                    with open(p, 'r', encoding='utf-8') as f:
                        total_rows = sum(1 for line in f) - 1
            else:
                # Đọc file Excel với engine openpyxl
                if is_ajax:
                    # Đọc toàn bộ để lấy tổng số dòng, nhưng chỉ lấy phần cần thiết
                    df_full = pd.read_excel(p, engine='openpyxl')
                    total_rows = len(df_full)
                    df = df_full.iloc[start_row:end_row]
                else:
                    # Chỉ đọc 100 dòng đầu
                    df = pd.read_excel(p, engine='openpyxl', nrows=rows_per_page)
                    df_full = pd.read_excel(p, engine='openpyxl')
                    total_rows = len(df_full)
            
            # Nếu là AJAX request, trả về JSON
            if is_ajax:
                # Chuyển DataFrame thành dictionary
                data = {
                    'data': df.fillna('').to_dict('records'),
                    'page': page,
                    'total_rows': total_rows,
                    'has_more': end_row < total_rows
                }
                return HttpResponse(json.dumps(data, default=str), content_type='application/json')
            
            # Chuyển DataFrame thành HTML table (chỉ 100 dòng đầu)
            html_table = df.fillna('').to_html(
                classes='table table-striped table-bordered',
                index=False,
                float_format=lambda x: '{:.2f}'.format(x) if isinstance(x, float) else str(x)
            )
            
            # Tạo trang HTML với lazy loading (đã loại bỏ nút đóng cửa sổ)
            html_content = f"""
            <!DOCTYPE html>
            <html lang="vi">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Xem: {escape(filename)}</title>
                <style>
                    body {{
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                        margin: 0;
                        padding: 20px;
                        background: linear-gradient(135deg, #0a192f 0%, #1a365d 100%);
                        color: #e2e8f0;
                        min-height: 100vh;
                    }}
                    .container {{
                        background: rgba(22, 32, 45, 0.9);
                        backdrop-filter: blur(10px);
                        padding: 30px;
                        border-radius: 16px;
                        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.3);
                        max-width: 95%;
                        margin: 0 auto;
                    }}
                    h1 {{
                        color: #38b2ac;
                        border-bottom: 2px solid #38b2ac;
                        padding-bottom: 15px;
                        margin-top: 0;
                    }}
                    .file-info {{
                        background: rgba(30, 41, 59, 0.7);
                        padding: 20px;
                        border-radius: 12px;
                        margin-bottom: 25px;
                        border-left: 5px solid #38b2ac;
                    }}
                    .stats {{
                        color: #a0aec0;
                        margin: 15px 0;
                        font-style: italic;
                    }}
                    .loading {{
                        display: none;
                        text-align: center;
                        padding: 20px;
                        color: #38b2ac;
                    }}
                    .load-more {{
                        text-align: center;
                        margin: 30px 0;
                    }}
                    .btn {{
                        background: linear-gradient(135deg, #4299e1, #3182ce);
                        color: white;
                        border: none;
                        padding: 12px 30px;
                        border-radius: 30px;
                        font-weight: 700;
                        cursor: pointer;
                        transition: all 0.3s ease;
                    }}
                    .btn:hover {{
                        background: linear-gradient(135deg, #3182ce, #2c5282);
                        transform: translateY(-2px);
                    }}
                    .btn:disabled {{
                        opacity: 0.6;
                        cursor: not-allowed;
                    }}
                    .back-btn {{
                        display: inline-block;
                        margin: 10px 10px 10px 0;
                        padding: 10px 25px;
                        background: rgba(113, 128, 150, 0.3);
                        color: #cbd5e0;
                        text-decoration: none;
                        border-radius: 25px;
                        border: 1px solid rgba(113, 128, 150, 0.5);
                        transition: all 0.3s ease;
                    }}
                    .back-btn:hover {{
                        background: rgba(113, 128, 150, 0.5);
                        transform: translateY(-2px);
                    }}
                    table {{
                        width: 100%;
                        border-collapse: separate;
                        border-spacing: 0;
                        margin: 25px 0;
                        background: rgba(30, 41, 59, 0.5);
                        border-radius: 12px;
                        overflow: hidden;
                    }}
                    thead {{
                        background: linear-gradient(135deg, rgba(26, 54, 93, 0.9), rgba(30, 64, 175, 0.8));
                    }}
                    th {{
                        padding: 18px;
                        text-align: left;
                        font-weight: 700;
                        color: #e2e8f0;
                        border-bottom: 2px solid #38b2ac;
                    }}
                    td {{
                        padding: 15px;
                        border-bottom: 1px solid rgba(56, 178, 172, 0.1);
                        color: #cbd5e0;
                    }}
                    tr:hover {{
                        background: rgba(56, 178, 172, 0.1);
                    }}
                    .pagination {{
                        display: flex;
                        justify-content: center;
                        gap: 10px;
                        margin: 25px 0;
                    }}
                    .page-btn {{
                        padding: 8px 16px;
                        border-radius: 20px;
                        background: rgba(56, 178, 172, 0.2);
                        color: #38b2ac;
                        border: 1px solid rgba(56, 178, 172, 0.3);
                        cursor: pointer;
                    }}
                    .page-btn.active {{
                        background: linear-gradient(135deg, #38b2ac, #4299e1);
                        color: white;
                    }}
                    .table-container {{
                        max-height: 70vh;
                        overflow: auto;
                        border-radius: 12px;
                    }}
                    ::-webkit-scrollbar {{
                        width: 10px;
                    }}
                    ::-webkit-scrollbar-track {{
                        background: rgba(22, 32, 45, 0.5);
                    }}
                    ::-webkit-scrollbar-thumb {{
                        background: rgba(56, 178, 172, 0.5);
                        border-radius: 5px;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>📄 Xem: {escape(filename)}</h1>
                    
                    <div class="file-info">
                        <strong>📁 Thư mục:</strong> {folder}<br>
                        <strong>📏 Kích thước:</strong> {p.stat().st_size / 1024:.2f} KB<br>
                        <strong>📊 Tổng số dòng:</strong> <span id="totalRows">{total_rows}</span><br>
                        <strong>👁️ Đang hiển thị:</strong> <span id="showingRows">{min(rows_per_page, total_rows)}</span> dòng đầu
                    </div>
                    
                    <div class="table-container" id="tableContainer">
                        {html_table}
                    </div>
                    
                    <div class="loading" id="loading">
                        ⏳ Đang tải thêm dữ liệu...
                    </div>
                    
                    <div class="load-more" id="loadMoreContainer">
                        <button class="btn" id="loadMoreBtn" onclick="loadMoreData()">
                            ⬇️ Tải thêm dữ liệu
                        </button>
                    </div>
                    
                    <div class="pagination" id="pagination">
                        <!-- Pagination sẽ được tạo bởi JavaScript -->
                    </div>
                    
                    <div style="margin-top: 30px;">
                        <a href="{request.META.get('HTTP_REFERER', '/')}" class="back-btn">📋 Quay lại danh sách</a>
                        <a href="{request.path.replace('/view/', '/download/')}" class="back-btn">⬇️ Tải toàn bộ file</a>
                    </div>
                </div>
                
                <script>
                    let currentPage = 1;
                    const rowsPerPage = {rows_per_page};
                    let totalRows = {total_rows};
                    let totalPages = Math.ceil(totalRows / rowsPerPage);
                    
                    // Tạo phân trang
                    function createPagination() {{
                        const pagination = document.getElementById('pagination');
                        pagination.innerHTML = '';
                        
                        // Nút đầu
                        if (currentPage > 1) {{
                            const btn = document.createElement('button');
                            btn.className = 'page-btn';
                            btn.innerHTML = '⏮️ Đầu';
                            btn.onclick = () => loadPage(1);
                            pagination.appendChild(btn);
                        }}
                        
                        // Các nút số trang
                        const startPage = Math.max(1, currentPage - 2);
                        const endPage = Math.min(totalPages, currentPage + 2);
                        
                        for (let i = startPage; i <= endPage; i++) {{
                            const btn = document.createElement('button');
                            btn.className = 'page-btn' + (i === currentPage ? ' active' : '');
                            btn.innerHTML = i;
                            btn.onclick = () => loadPage(i);
                            pagination.appendChild(btn);
                        }}
                        
                        // Nút cuối
                        if (currentPage < totalPages) {{
                            const btn = document.createElement('button');
                            btn.className = 'page-btn';
                            btn.innerHTML = '⏭️ Cuối';
                            btn.onclick = () => loadPage(totalPages);
                            pagination.appendChild(btn);
                        }}
                    }}
                    
                    // Tải thêm dữ liệu
                    async function loadMoreData() {{
                        currentPage++;
                        await loadPage(currentPage);
                    }}
                    
                    // Tải trang cụ thể
                    async function loadPage(page) {{
                        const loading = document.getElementById('loading');
                        const loadMoreBtn = document.getElementById('loadMoreBtn');
                        const tableContainer = document.getElementById('tableContainer');
                        
                        loading.style.display = 'block';
                        loadMoreBtn.disabled = true;
                        
                        try {{
                            const response = await fetch(`{request.path}?page=${{page}}`, {{
                                headers: {{
                                    'X-Requested-With': 'XMLHttpRequest'
                                }}
                            }});
                            
                            const data = await response.json();
                            
                            if (data.data.length > 0) {{
                                // Tạo bảng mới từ dữ liệu
                                let tableHtml = '<table class="table table-striped table-bordered"><thead><tr>';
                                
                                // Tạo header từ keys của object đầu tiên
                                const firstRow = data.data[0];
                                for (const key in firstRow) {{
                                    tableHtml += `<th>${{key}}</th>`;
                                }}
                                tableHtml += '</tr></thead><tbody>';
                                
                                // Thêm các dòng dữ liệu
                                data.data.forEach(row => {{
                                    tableHtml += '<tr>';
                                    for (const key in row) {{
                                        tableHtml += `<td>${{row[key]}}</td>`;
                                    }}
                                    tableHtml += '</tr>';
                                }});
                                
                                tableHtml += '</tbody></table>';
                                
                                // Nếu là trang đầu, thay thế bảng cũ
                                if (page === 1) {{
                                    tableContainer.innerHTML = tableHtml;
                                }} else {{
                                    // Nếu là trang tiếp theo, thêm vào cuối bảng hiện tại
                                    const currentTable = tableContainer.querySelector('table');
                                    if (currentTable) {{
                                        const tbody = currentTable.querySelector('tbody');
                                        const newRows = new DOMParser().parseFromString(tableHtml, 'text/html')
                                            .querySelector('tbody').innerHTML;
                                        tbody.innerHTML += newRows;
                                    }}
                                }}
                                
                                // Cập nhật thông tin
                                document.getElementById('showingRows').textContent = 
                                    Math.min(page * rowsPerPage, totalRows);
                                currentPage = page;
                                
                                // Ẩn nút tải thêm nếu đã hiển thị hết
                                if (page * rowsPerPage >= totalRows) {{
                                    document.getElementById('loadMoreContainer').style.display = 'none';
                                }}
                            }}
                            
                            createPagination();
                            
                        }} catch (error) {{
                            console.error('Error loading data:', error);
                            alert('Lỗi khi tải dữ liệu: ' + error.message);
                        }} finally {{
                            loading.style.display = 'none';
                            loadMoreBtn.disabled = false;
                        }}
                    }}
                    
                    // Khởi tạo phân trang
                    document.addEventListener('DOMContentLoaded', function() {{
                        createPagination();
                        
                        // Ẩn nút tải thêm nếu không còn dữ liệu
                        if (rowsPerPage >= totalRows) {{
                            document.getElementById('loadMoreContainer').style.display = 'none';
                        }}
                    }});
                </script>
            </body>
            </html>
            """
            
            return HttpResponse(html_content, content_type='text/html; charset=utf-8')
            
        except Exception as e:
            # Nếu có lỗi khi đọc file, trả về thông báo lỗi
            return HttpResponse(f"""
                <!DOCTYPE html>
                <html lang="vi">
                <head>
                    <meta charset="UTF-8">
                    <title>Lỗi</title>
                    <style>
                        body {{
                            font-family: Arial, sans-serif;
                            padding: 40px;
                            background: #f0f2f5;
                        }}
                        .error-container {{
                            background: white;
                            padding: 30px;
                            border-radius: 10px;
                            max-width: 600px;
                            margin: 0 auto;
                            box-shadow: 0 0 20px rgba(0,0,0,0.1);
                            text-align: center;
                        }}
                        .back-btn {{
                            display: inline-block;
                            margin-top: 20px;
                            padding: 10px 20px;
                            background: #3498db;
                            color: white;
                            text-decoration: none;
                            border-radius: 5px;
                        }}
                    </style>
                </head>
                <body>
                    <div class="error-container">
                        <h2 style="color: #e74c3c;">❌ Lỗi khi đọc file</h2>
                        <p>Không thể đọc file {escape(filename)}</p>
                        <p><strong>Chi tiết lỗi:</strong> {escape(str(e))}</p>
                        <a href="{request.META.get('HTTP_REFERER', '/')}" class="back-btn">📋 Quay lại danh sách</a>
                    </div>
                </body>
                </html>
            """, status=500)
    
    else:
        # Với các loại file khác (txt, json, etc.), hiển thị nội dung trực tiếp
        try:
            with open(p, 'r', encoding='utf-8') as f:
                content = f.read()
            
            html_content = f"""
            <!DOCTYPE html>
            <html lang="vi">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Xem: {escape(filename)}</title>
                <style>
                    body {{ 
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                        margin: 0;
                        padding: 20px;
                        background: linear-gradient(135deg, #0a192f 0%, #1a365d 100%);
                        color: #e2e8f0;
                        min-height: 100vh;
                    }}
                    .container {{ 
                        background: rgba(22, 32, 45, 0.9);
                        backdrop-filter: blur(10px);
                        padding: 30px;
                        border-radius: 16px;
                        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.3);
                        max-height: 80vh;
                        overflow: auto;
                    }}
                    .file-header {{
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        margin-bottom: 20px;
                    }}
                    .back-btn {{
                        display: inline-block;
                        padding: 10px 25px;
                        background: rgba(113, 128, 150, 0.3);
                        color: #cbd5e0;
                        text-decoration: none;
                        border-radius: 25px;
                        border: 1px solid rgba(113, 128, 150, 0.5);
                        transition: all 0.3s ease;
                    }}
                    .back-btn:hover {{
                        background: rgba(113, 128, 150, 0.5);
                        transform: translateY(-2px);
                    }}
                    pre {{
                        white-space: pre-wrap;
                        word-wrap: break-word;
                        font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
                        line-height: 1.5;
                    }}
                    ::-webkit-scrollbar {{
                        width: 10px;
                    }}
                    ::-webkit-scrollbar-track {{
                        background: rgba(22, 32, 45, 0.5);
                    }}
                    ::-webkit-scrollbar-thumb {{
                        background: rgba(56, 178, 172, 0.5);
                        border-radius: 5px;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="file-header">
                        <h1 style="margin: 0; color: #38b2ac;">📄 {escape(filename)}</h1>
                        <a href="{request.META.get('HTTP_REFERER', '/')}" class="back-btn">📋 Quay lại</a>
                    </div>
                    <pre>{escape(content)}</pre>
                </div>
            </body>
            </html>
            """
            return HttpResponse(html_content, content_type='text/html; charset=utf-8')
            
        except:
            # Nếu không đọc được dưới dạng text, trả về file để download
            content_type, _ = mimetypes.guess_type(str(p))
            resp = FileResponse(open(p, "rb"), content_type=content_type or "application/octet-stream")
            resp["Content-Disposition"] = f'inline; filename="{p.name}"'
            return resp
