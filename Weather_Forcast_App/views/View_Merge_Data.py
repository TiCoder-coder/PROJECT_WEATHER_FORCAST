import os
import subprocess
from pathlib import Path
from datetime import datetime
import sys

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.urls import reverse


# ============================================================
# _latest_file_info(dir_path)
# ============================================================
# Mục đích:
# - Quét trong 1 thư mục (dir_path) và tìm file "mới nhất" (mtime lớn nhất)
# - Chỉ xét các file dữ liệu có đuôi .csv hoặc .xlsx
# - Trả về metadata để frontend hiển thị (tên file, dung lượng, thời gian sửa đổi)
#
# Input:
# - dir_path: đường dẫn thư mục (string)
#
# Output:
# - None: nếu thư mục không tồn tại hoặc không có file phù hợp
# - dict: nếu có file phù hợp, ví dụ:
#   {
#     "name": "merged_vrain_data.xlsx",
#     "size_mb": 12.34,
#     "mtime": "2026-02-05 20:15:00",
#   }
def _latest_file_info(dir_path: str):
    # Convert string -> Path để thao tác filesystem gọn hơn
    p = Path(dir_path)

    # Nếu thư mục không tồn tại => không có gì để quét
    if not p.exists():
        return None

    # Lấy danh sách file trong thư mục (chỉ lấy file + đúng extension)
    files = [
        f for f in p.iterdir()
        if f.is_file() and f.suffix.lower() in {".csv", ".xlsx"}
    ]

    # Nếu không có file phù hợp => return None
    if not files:
        return None

    # Chọn file mới nhất dựa trên thời gian sửa đổi (st_mtime)
    latest = max(files, key=lambda x: x.stat().st_mtime)

    # Lấy stat để đọc size và mtime nhanh (tránh gọi nhiều lần)
    st = latest.stat()

    # Build dict metadata trả về
    return {
        "name": latest.name,  # tên file (không gồm path)
        "size_mb": round(st.st_size / (1024 * 1024), 2),  # bytes -> MB, làm tròn 2 chữ số
        "mtime": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),  # mtime -> chuỗi
    }


# ============================================================
# @csrf_exempt
# ============================================================
# Decorator này tắt kiểm tra CSRF cho view.
# Thường dùng khi:
# - bạn gọi endpoint bằng fetch/AJAX mà chưa set CSRF token đúng
# - hoặc endpoint nội bộ (internal tool) bạn muốn đơn giản hóa
#
# Lưu ý bảo mật (để bạn hiểu):
# - CSRF exempt có thể làm endpoint dễ bị gọi trái phép từ website khác
# - Chỉ nên dùng nếu endpoint có cơ chế bảo vệ khác hoặc chỉ dùng nội bộ
@csrf_exempt
def merge_data_view(request):
    # ============================================================
    # 1) Kiểm tra method: chỉ cho POST
    # ============================================================
    # Endpoint này là "bấm nút merge" từ frontend -> cần POST
    if request.method != "POST":
        return JsonResponse({"success": False, "message": "Method not allowed."}, status=405)

    try:
        # ============================================================
        # 2) Xác định các đường dẫn quan trọng
        # ============================================================
        # BASE_DIR: thư mục cha của file views hiện tại (tùy cấu trúc project)
        # - __file__ là path file python hiện tại
        # - abspath -> lấy đường dẫn tuyệt đối
        # - dirname 2 lần -> đi lên 2 cấp thư mục
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # script_path: đường dẫn script merge excel của bạn
        # - scripts/Merge_xlsx.py là script thực hiện merge
        script_path = os.path.join(BASE_DIR, "scripts", "Merge_xlsx.py")

        # python_exe: đường dẫn python interpreter đang chạy server Django
        # - đảm bảo chạy script bằng đúng môi trường/venv hiện tại
        python_exe = sys.executable

        # output_dir: thư mục chứa file crawl raw
        # - bạn dùng để đếm số file mới sinh ra (new_files_count)
        output_dir = os.path.join(BASE_DIR, "output")

        # merge_dir: thư mục chứa file merge đầu ra
        # - _latest_file_info(merge_dir) sẽ lấy file merge mới nhất
        merge_dir = os.path.join(BASE_DIR, "Merge_data")

        # ============================================================
        # 3) Snapshot file trước khi merge (để đếm file mới)
        # ============================================================
        # before_files:
        # - nếu output_dir tồn tại -> listdir lấy danh sách tên file
        # - nếu không tồn tại -> set rỗng
        before_files = set(os.listdir(output_dir)) if os.path.exists(output_dir) else set()

        # ============================================================
        # 4) Chạy script merge bằng subprocess
        # ============================================================
        # subprocess.run:
        # - chạy lệnh: python_exe script_path
        # - capture_output=True => gom stdout + stderr vào result
        # - text=True => stdout/stderr dạng string (không phải bytes)
        # - check=False => không raise exception nếu returncode != 0
        #
        # Vì bạn muốn:
        # - tự kiểm tra result.returncode để trả JSON phù hợp
        result = subprocess.run(
            [python_exe, script_path],
            capture_output=True,
            text=True,
            check=False
        )

        # ============================================================
        # 5) Snapshot file sau khi merge -> tính số file mới
        # ============================================================
        after_files = set(os.listdir(output_dir)) if os.path.exists(output_dir) else set()

        # new_files_count:
        # - after_files - before_files => tập file xuất hiện mới trong output_dir
        # - len(...) => số lượng file mới
        #
        # Lưu ý:
        # - script merge thường ghi file vào Merge_data (không phải output)
        # - nhưng ở đây bạn đếm "output_dir" có file mới không (phục vụ dashboard/log)
        new_files_count = len(after_files - before_files)

        # ============================================================
        # 6) Lấy thông tin file merge mới nhất trong Merge_data
        # ============================================================
        latest_merged = _latest_file_info(merge_dir)

        # Nếu có file merge mới nhất:
        # - gắn thêm folder key để frontend biết thuộc thư mục nào
        # - tạo URL view/download dựa trên named routes trong urls.py
        if latest_merged:
            folder_key = "merged"
            latest_merged["folder"] = folder_key

            # reverse("weather:dataset_view", args=[folder_key, filename])
            # -> tạo url preview dataset (xem trong browser)
            latest_merged["view_url"] = reverse(
                "weather:dataset_view",
                args=[folder_key, latest_merged["name"]]
            )

            # reverse("weather:dataset_download", args=[folder_key, filename])
            # -> tạo url download dataset (tải file)
            latest_merged["download_url"] = reverse(
                "weather:dataset_download",
                args=[folder_key, latest_merged["name"]]
            )

        # ============================================================
        # 7) Nếu script lỗi (returncode != 0) -> trả JSON thất bại
        # ============================================================
        # returncode:
        # - 0  : chạy OK
        # - !=0: lỗi (có thể do exception trong script, thiếu file, thiếu thư mục, ...)
        if result.returncode != 0:
            return JsonResponse({
                "success": False,
                "message": "Gộp dữ liệu thất bại!",
                "stderr": result.stderr,          # log lỗi từ script (quan trọng để debug)
                "new_files_count": new_files_count,
                "latest_merged": latest_merged    # vẫn trả để frontend biết file merge hiện tại (nếu có)
            })

        # ============================================================
        # 8) Thành công -> trả JSON OK
        # ============================================================
        return JsonResponse({
            "success": True,
            "message": "Gộp dữ liệu thành công!",
            "new_files_count": new_files_count,
            "latest_merged": latest_merged
        })

    except Exception as e:
        # ============================================================
        # 9) Catch-all exception: lỗi bất ngờ trong view
        # ============================================================
        # Ví dụ:
        # - permission error khi listdir/open folder
        # - reverse lỗi do thiếu route
        # - script_path sai, ...
        return JsonResponse({"success": False, "message": f"Lỗi: {str(e)}"})
