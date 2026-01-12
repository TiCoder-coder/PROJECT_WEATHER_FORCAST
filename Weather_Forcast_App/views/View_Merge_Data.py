import os
import subprocess
from pathlib import Path
from datetime import datetime

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.urls import reverse


def _latest_file_info(dir_path: str):
    p = Path(dir_path)
    if not p.exists():
        return None

    files = [
        f for f in p.iterdir()
        if f.is_file() and f.suffix.lower() in {".csv", ".xlsx"}
    ]
    if not files:
        return None

    latest = max(files, key=lambda x: x.stat().st_mtime)
    st = latest.stat()
    return {
        "name": latest.name,
        "size_mb": round(st.st_size / (1024 * 1024), 2),
        "mtime": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    }


@csrf_exempt
def merge_data_view(request):
    if request.method != "POST":
        return JsonResponse({"success": False, "message": "Method not allowed."}, status=405)

    try:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script_path = os.path.join(BASE_DIR, "scripts", "merge_data.py")
        output_dir = os.path.join(BASE_DIR, "output")
        merge_dir = os.path.join(BASE_DIR, "Merge_data")  # <<< QUAN TRỌNG

        before_files = set(os.listdir(output_dir)) if os.path.exists(output_dir) else set()

        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            check=False
        )

        after_files = set(os.listdir(output_dir)) if os.path.exists(output_dir) else set()
        new_files_count = len(after_files - before_files)

        # Lấy file merged mới nhất trong Merge_data
        latest_merged = _latest_file_info(merge_dir)
        if latest_merged:
            folder_key = "merged"
            latest_merged["folder"] = folder_key
            latest_merged["view_url"] = reverse("weather:dataset_view", args=[folder_key, latest_merged["name"]])
            latest_merged["download_url"] = reverse("weather:dataset_download", args=[folder_key, latest_merged["name"]])

        if result.returncode != 0:
            return JsonResponse({
                "success": False,
                "message": "Gộp dữ liệu thất bại!",
                "stderr": result.stderr,
                "new_files_count": new_files_count,
                "latest_merged": latest_merged
            })

        return JsonResponse({
            "success": True,
            "message": "Gộp dữ liệu thành công!",
            "new_files_count": new_files_count,
            "latest_merged": latest_merged
        })

    except Exception as e:
        return JsonResponse({"success": False, "message": f"Lỗi: {str(e)}"})
