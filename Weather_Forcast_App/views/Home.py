from django.shortcuts import render
from pathlib import Path
from datetime import datetime
from django.conf import settings

def _fmt_dt(ts: float) -> str:
    # ts: timestamp seconds
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

def home_view(request):
    """
    Trang Home chính của hệ thống Weather Forecast.
    Chỉ lo phần giao diện, các số liệu dùng giá trị mặc định.
    Sau này bạn muốn thì bổ sung logic đọc file output.
    """

    # 1) Lấy path output (ưu tiên BASE_DIR/output nếu bạn để đúng project)
    # Nếu bạn đang cố định output ở Weather_Forcast_App/output thì sửa lại cho đúng cấu trúc bạn đang dùng.
    output_dir = Path(settings.BASE_DIR) / "Weather_Forcast_App" / "output"
    if not output_dir.exists():
        # fallback: nếu bạn đang để output ngay trong BASE_DIR/output
        output_dir = Path(settings.BASE_DIR) / "output"

    # 2) Lấy danh sách file
    files = [p for p in output_dir.glob("*") if p.is_file()]

    # 3) File mới nhất
    latest_file = max(files, key=lambda p: p.stat().st_mtime) if files else None

    # 4) Đếm lượt chạy theo rule tên file (bạn chỉnh prefix cho đúng tên file của bạn)
    def count_by_prefix(prefixes):
        c = 0
        for p in files:
            name = p.name.lower()
            if any(name.startswith(x) for x in prefixes):
                c += 1
        return c

    total_api_runs = count_by_prefix(["vietnam_weather_data", "api_weather", "openweather"])
    total_vrain_runs = count_by_prefix(["bao_cao_mua", "vrain"])
    total_image_runs = count_by_prefix(["image", "camera"])

    context = {
        "latest_dataset_name": latest_file.name if latest_file else "Chưa có dataset",
        "latest_dataset_time": _fmt_dt(latest_file.stat().st_mtime) if latest_file else "—",
        "total_api_runs": total_api_runs,
        "total_vrain_runs": total_vrain_runs,
        "total_image_runs": total_image_runs,
    }
    return render(request, "weather/Home.html", context)
