"""
View xử lý gộp dữ liệu từ các file Excel trong thư mục output
vào file merged_weather_data.xlsx trong thư mục Merge_data
"""

import subprocess
import json
from pathlib import Path
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt


@require_http_methods(["POST"])
@csrf_exempt
def merge_data_view(request):
    """
    View xử lý merge dữ liệu từ các file .xlsx trong thư mục output
    bằng cách chạy script Merge_xlsx.py
    """
    try:
        # Đường dẫn tới script Merge_xlsx.py
        script_path = Path(r"D:\PROJRCT_WEATHER_FORCAST\Weather_Forcast_App\scripts\Merge_xlsx.py")
        
        # Kiểm tra script có tồn tại không
        if not script_path.exists():
            return JsonResponse({
                "success": False,
                "message": f"Không tìm thấy file Merge_xlsx.py tại: {script_path}"
            }, status=404)
        
        # Chạy script Merge_xlsx.py
        # Working directory là thư mục chứa script
        script_dir = script_path.parent
        
        # Thiết lập environment variables để xử lý Unicode
        import os
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(
            ["python", str(script_path)],
            capture_output=True,
            text=True,
            timeout=300,  # Timeout 5 phút
            cwd=str(script_dir),  # Chạy từ thư mục scripts
            env=env,  # Thêm environment để xử lý UTF-8
            encoding='utf-8',  # Đảm bảo output được decode đúng
            errors='replace'  # Thay thế ký tự lỗi thay vì crash
        )
        
        # Kiểm tra kết quả
        if result.returncode == 0:
            # Đọc output để lấy thông tin chi tiết
            output_lines = result.stdout.strip().split('\n')
            
            # Tìm thông tin về số file đã merge
            new_files_count = 0
            total_rows = 0
            
            for line in output_lines:
                if "Số file mới chưa merge:" in line:
                    try:
                        new_files_count = int(line.split(":")[-1].strip())
                    except:
                        pass
                elif "Tổng số dòng sau khi merge:" in line:
                    try:
                        total_rows = int(line.split(":")[-1].strip())
                    except:
                        pass
            
            # Tạo thông báo chi tiết
            if new_files_count > 0:
                message = f"✅ Đã gộp thành công {new_files_count} file mới! Tổng: {total_rows} dòng dữ liệu."
            else:
                message = "ℹ️ Không có file mới để gộp. Tất cả dữ liệu đã được merge trước đó."
            
            return JsonResponse({
                "success": True,
                "message": message,
                "details": {
                    "new_files": new_files_count,
                    "total_rows": total_rows,
                    "output": result.stdout
                }
            })
        else:
            # Lỗi khi chạy script
            error_message = result.stderr if result.stderr else "Lỗi không xác định"
            return JsonResponse({
                "success": False,
                "message": f"Lỗi khi gộp dữ liệu: {error_message}",
                "details": {
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            }, status=500)
            
    except subprocess.TimeoutExpired:
        return JsonResponse({
            "success": False,
            "message": "⏱️ Quá trình gộp dữ liệu mất quá nhiều thời gian (timeout 5 phút)"
        }, status=408)
        
    except Exception as e:
        return JsonResponse({
            "success": False,
            "message": f"❌ Lỗi hệ thống: {str(e)}"
        }, status=500)