"""
URL Configuration - WeatherForcast (CẤP ROOT PROJECT)
====================================================

Đây là file cấu hình URL chính (root URLconf) của project Django.

Mục đích:
    - Định tuyến URL từ ngôn ngữ natural sang các view/handler
    - Tổng hợp các URL pattern từ các app con
    - Cấu hình static files và media files trong DEBUG mode

Cấu trúc URL của project:
    http://localhost:8000/admin/          -> Django Admin
    http://localhost:8000/                -> Tất cả URL của Weather_Forcast_App
    (các endpoint cụ thể được define trong Weather_Forcast_App/urls.py)

Lưu ý:
    - Khi thêm app mới, hãy include() URLconf của app thay vì định nghĩa trực tiếp ở đây
    - Giúp code tổ chức hơn và dễ bảo trì
"""

from django.contrib import admin
from django.urls import path, include
from Weather_Forcast_App.views import View_Crawl_data_by_API
from django.conf import settings
from django.conf.urls.static import static

# ============================================================
# ĐỊNH TUYẾN URL CHÍNH (ROOT URLPATTERNS)
# ============================================================
urlpatterns = [
    # Admin panel: http://localhost:8000/admin/
    # - Django admin hỗ trợ bằng sẵn
    # - Cho phép quản lý models, users, permissions
    # - Cần superuser để đăng nhập
    path("admin/", admin.site.urls),
    
    # Include tất cả URL của app Weather_Forcast_App
    # - File định nghĩa: Weather_Forcast_App/urls.py
    # - Các URL sẽ được include ngay ở root (không có prefix)
    # - Ví dụ: /crawl-api-weather/, /auth/login/, /datasets/, v.v.
    path("", include("Weather_Forcast_App.urls")),
]

# ============================================================
# CẤU HÌNH STATIC FILES + MEDIA FILES (CHỈ DÙNG KHI DEBUG=True)
# ============================================================
# Trong production:
# - Django KHÔNG phục vụ static files (sử dụng web server như Nginx)
# - Chỉ khi DEBUG=True (development), Django mới tự phục vụ
#
# settings.STATIC_URL: URL để truy cập static (default: /static/)
# settings.STATIC_ROOT: đường dẫn thực tế của thư mục static trên disk
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)