from django.shortcuts import render
from pathlib import Path
from datetime import datetime
from django.conf import settings

# ============================================================
# _fmt_dt(ts): HÀM FORMAT THỜI GIAN (timestamp -> string)
# ============================================================
# Mục đích:
# - Các file trong hệ thống có thuộc tính thời gian chỉnh sửa (mtime) dạng timestamp (float)
#   ví dụ: 1700000000.123
# - Bạn muốn hiển thị thời gian này ra UI theo format dễ đọc: "YYYY-MM-DD HH:MM:SS"
#
# ts: float
# - là UNIX timestamp tính theo giây (seconds since epoch)
#
# datetime.fromtimestamp(ts):
# - chuyển UNIX timestamp thành datetime theo timezone local của server
#
# strftime("%Y-%m-%d %H:%M:%S"):
# - format datetime thành chuỗi.
def _fmt_dt(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# home_view(request): VIEW TRANG HOME (Dashboard) CHO WEATHER APP
# ============================================================
# Vai trò:
# - Đây là Django view render template Home.html
# - Nó sẽ:
#   1) Tìm thư mục output chứa dataset được crawl/tạo ra
#   2) Lấy danh sách file dataset
#   3) Xác định file mới nhất (latest dataset)
#   4) Đếm số lần chạy theo nhóm dữ liệu dựa vào prefix tên file
#   5) Lấy info đăng nhập từ session (custom auth)
#   6) Truyền context sang template để hiển thị UI
def home_view(request):
    # ============================================================
    # 1) XÁC ĐỊNH ĐƯỜNG DẪN THƯ MỤC OUTPUT
    # ============================================================
    # Ưu tiên:
    # - settings.BASE_DIR/Weather_Forcast_App/output
    # Nếu không tồn tại:
    # - fallback sang settings.BASE_DIR/output
    #
    # Lý do fallback:
    # - Có thể bạn chạy app ở nhiều môi trường / cấu trúc project khác nhau
    # - Hoặc trước đó output nằm ở root, sau này chuyển vào Weather_Forcast_App/output
    # Sử dụng thư mục data_crawl tuyệt đối
    output_dir = Path("/media/voanhnhat/SDD_OUTSIDE5/PROJECT_WEATHER_FORECAST/data/data_crawl")
    if not output_dir.exists():
        output_dir = Path("/media/voanhnhat/SDD_OUTSIDE5/PROJECT_WEATHER_FORECAST/data/data_crawl")

    # ============================================================
    # 2) LẤY TẤT CẢ FILE TRONG THƯ MỤC OUTPUT
    # ============================================================
    # output_dir.glob("*"):
    # - lấy tất cả entry trong thư mục (file + folder)
    #
    # p.is_file():
    # - chỉ giữ lại những entry là file (loại bỏ folder)
    files = [p for p in output_dir.glob("*") if p.is_file()]

    # ============================================================
    # 3) TÌM FILE DATASET MỚI NHẤT (LATEST)
    # ============================================================
    # p.stat().st_mtime:
    # - thời gian "last modified time" (mtime) của file
    #
    # max(..., key=...):
    # - lấy file có mtime lớn nhất => file mới nhất
    #
    # Nếu files rỗng => latest_file = None
    latest_file = max(files, key=lambda p: p.stat().st_mtime) if files else None

    # ============================================================
    # 4) HÀM ĐẾM FILE THEO PREFIX
    # ============================================================
    # count_by_prefix(prefixes):
    # - prefixes: list các prefix có thể match
    # - duyệt tất cả files:
    #   + chuyển tên file về lowercase để so sánh không phân biệt hoa/thường
    #   + nếu file name bắt đầu bằng bất kỳ prefix nào trong prefixes => count++
    #
    # Mục đích:
    # - "đếm số lần chạy" pipeline dựa vào quy ước đặt tên file output
    # - ví dụ:
    #   vietnam_weather_data...  => pipeline API weather
    #   vrain... hoặc bao_cao_mua => pipeline vrain
    def count_by_prefix(prefixes):
        c = 0
        for p in files:
            name = p.name.lower()
            if any(name.startswith(x) for x in prefixes):
                c += 1
        return c

    # ============================================================
    # 5) ĐẾM SỐ FILE THEO NHÓM PIPELINE
    # ============================================================
    # Các nhóm này là "heuristic" dựa theo prefix tên file.
    # Bạn dùng nó như một số liệu thống kê trên Home:
    #
    # total_api_runs:
    # - các file bắt đầu bởi:
    #   "vietnam_weather_data" / "api_weather" / "openweather"
    # => giả định đây là các output từ crawler API thời tiết.
    total_api_runs = count_by_prefix(["vietnam_weather_data", "api_weather", "openweather"])

    # total_vrain_runs:
    # - các file bắt đầu bởi:
    #   "bao_cao_mua" / "vrain"
    # => giả định đây là output của crawler Vrain.
    total_vrain_runs = count_by_prefix(["bao_cao_mua", "vrain"])

    # total_image_runs:
    # - các file bắt đầu bởi:
    #   "image" / "camera"
    # => giả định output từ pipeline ảnh/camera (nếu có).
    total_image_runs = count_by_prefix(["image", "camera"])

    # ============================================================
    # 6) LẤY THÔNG TIN ĐĂNG NHẬP TỪ SESSION (CUSTOM AUTH)
    # ============================================================
    # request.session:
    # - nơi Django lưu session data (tuỳ backend session: database/cache/redis)
    #
    # profile:
    # - bạn lưu thông tin user (tên, email, role...) dưới key "profile"
    #
    # is_logged_in:
    # - bạn check đăng nhập dựa trên "access_token" trong session
    # - nếu access_token tồn tại => coi như user đã login
    profile = request.session.get("profile")
    is_logged_in = request.session.get("access_token") is not None

    # ============================================================
    # 7) TẠO CONTEXT TRUYỀN QUA TEMPLATE Home.html
    # ============================================================
    # latest_dataset_name:
    # - nếu có latest_file => dùng latest_file.name
    # - nếu không => "Chưa có dataset"
    #
    # latest_dataset_time:
    # - nếu có latest_file => lấy st_mtime và format bằng _fmt_dt
    # - nếu không => "—"
    #
    # total_*:
    # - số lượng file tương ứng nhóm pipeline
    #
    # profile, is_logged_in:
    # - template có thể dùng để:
    #   + hiển thị avatar/tên user
    #   + hiện nút login/logout
    #   + hiện nút "Quick Crawl" cho user đăng nhập
    context = {
        "latest_dataset_name": latest_file.name if latest_file else "Chưa có dataset",
        "latest_dataset_time": _fmt_dt(latest_file.stat().st_mtime) if latest_file else "—",
        "total_api_runs": total_api_runs,
        "total_vrain_runs": total_vrain_runs,
        "total_image_runs": total_image_runs,
        "profile": profile,
        "is_logged_in": is_logged_in,
    }

    # ============================================================
    # 8) RENDER TEMPLATE
    # ============================================================
    # render(request, "weather/Home.html", context):
    # - trả HttpResponse bằng cách render template HTML
    # - context sẽ được template engine (Django templates) dùng để hiển thị dữ liệu
    return render(request, "weather/Home.html", context)
