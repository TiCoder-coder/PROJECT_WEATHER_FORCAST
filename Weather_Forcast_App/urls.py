"""
URL Configuration - Weather_Forcast_App (APP)
==============================================

Đây là file định tuyến URL cho toàn bộ app Weather_Forcast_App.

Mục đích:
    - Định tuyến các endpoint vào các view tương ứng
    - Tổ chức code rõ ràng: [endpoint] -> [view] -> [logic]
    - Hỗ trợ cả GET (hiển thị HTML form) và POST (xử lý dữ liệu)

Cấu trúc chính:
    1. Dashboard (home view)
    2. Authentication (login, register, logout, password reset, OTP verification)
    3. Data Crawling (crawl từ 3 nguồn Vrain: HTML, API, Selenium)
    4. Data Management (datasets, merge, clean)

Mô tả từng endpoint được viết dưới đây.
"""

from django.urls import path

# ============================================================
# IMPORT CÁC VIEWS
# ============================================================
# Views được tổ chức theo từng chức năng để dễ quản lý

from .views.Home import home_view

# ============================================================
# CRAWL DATA VIEWS (TỪ 3 NGUỒN KHÁC NHAU)
# ============================================================
from .views.View_Crawl_data_by_API import crawl_api_weather_view, api_weather_logs_view
from .views.View_Crawl_data_from_html_of_Vrain import (
    crawl_vrain_html_view,
    crawl_vrain_html_start_view,
    crawl_vrain_html_tail_view,
)
from .views.View_Crawl_data_from_Vrain_by_API import (
    crawl_vrain_api_view,
    crawl_vrain_api_start_view,
    crawl_vrain_api_tail_view,
)
from .views.View_Crawl_data_from_Vrain_by_Selenium import (
    crawl_vrain_selenium_view,
    crawl_vrain_selenium_start_view,
    crawl_vrain_selenium_tail_view,
)

# ============================================================
# DATA MANAGEMENT VIEWS (XỨLÝ, GHÉP, LÀMSẠCH DỮ LIỆU)
# ============================================================
from .views.View_Datasets import datasets_view, dataset_download_view, dataset_view_view
from .views.View_Merge_Data import merge_data_view
from .views.View_Clear import (
    clean_data_view,
    clean_files_list_view,
    clean_data_tail_view,
)

# ============================================================
# AUTHENTICATION VIEWS (ĐĂNG NHẬP, ĐĂNG KÝ, QUẢN LÝ TÀI KHOẢN)
# ============================================================
from .views.View_login import (
    login_view, register_view, logout_view, profile_view,
    forgot_password_view, password_reset_sent_view,
    reset_password_view, password_reset_complete_view,
    forgot_password_otp_view, verify_otp_view, reset_password_otp_view,
    verify_email_register_view, resend_email_otp_view, cancel_register_view,
)

# ============================================================
# APP NAMESPACE (TÍNH NĂNG REVERSE URL)
# ============================================================
# app_name = "weather" cho phép dùng:
#   reverse("weather:home") -> "/weather/"  (hoặc "")
#   reverse("weather:login") -> "/weather/auth/login/"
#
# Namespace giúp tránh xung đột tên URL giữa các app
app_name = "weather"

# ============================================================
# URL PATTERNS (ĐỊNH TUYẾN CHÍNH)
# ============================================================
urlpatterns = [
    # ============================================================
    # 1. DASHBOARD / HOME (TRANG CHỦ)
    # ============================================================
    # Method: GET
    # Giải thích:
    #   - Trang chủ hiển thị dashboard chính
    #   - Thống kê dữ liệu, biểu đồ, menu chức năng
    #   - Thường không yêu cầu đăng nhập (public page)
    path("", home_view, name="home"),
    
    
    # ============================================================
    # 2. DATA CRAWLING - OPENWEATHER API
    # ============================================================
    # 2.1) MAIN VIEW (Hiển thị form + thông tin job chạy gần nhất)
    # Method: GET (form), POST (start job)
    # Giải thích:
    #   - Giao diện crawl dữ liệu từ OpenWeather API
    #   - POST với action="start" -> chạy script background
    path("crawl-api-weather/", crawl_api_weather_view, name="crawl_api_weather"),
    path("crawl-by-api/", crawl_api_weather_view, name="crawl_by_api"),
    
    # 2.2) LOGS ENDPOINT (Frontend polling để lấy log real-time)
    # Method: GET
    # Giải thích:
    #   - Endpoint JSON để frontend poll logs
    #   - Backend trả dòng log mới nhất
    #   - Hỗ trợ pagination (since, limit)
    path("crawl-api-weather/logs/", api_weather_logs_view, name="crawl_api_weather_logs"),
    
    
    # ============================================================
    # 3. DATA CRAWLING - VRAIN (VRAIN.VN WEATHER STATION)
    # ============================================================
    # 3.1) HTML PARSING (Lấy dữ liệu bằng Selenium từ HTML VRAIN)
    # GET: form view để HĐ bấm start
    # POST: start job
    # LOGS: endpoint JSON cho polling
    path("crawl-vrain-html/", crawl_vrain_html_view, name="crawl_vrain_html"),
    path("crawl-vrain-html/start/", crawl_vrain_html_start_view, name="crawl_vrain_html_start"),
    path("crawl-vrain-html/tail/", crawl_vrain_html_tail_view, name="crawl_vrain_html_tail"),
    
    # 3.2) API METHOD (Lấy dữ liệu qua API của VRAIN)
    path("crawl-vrain-api/", crawl_vrain_api_view, name="crawl_vrain_api"),
    path("crawl-vrain-api/start/", crawl_vrain_api_start_view, name="crawl_vrain_api_start"),
    path("crawl-vrain-api/tail/", crawl_vrain_api_tail_view, name="crawl_vrain_api_tail"),
    
    # 3.3) SELENIUM METHOD (Dùng trình duyệt fake để crawl)
    path("crawl-vrain-selenium/", crawl_vrain_selenium_view, name="crawl_vrain_selenium"),
    path("crawl-vrain-selenium/start/", crawl_vrain_selenium_start_view, name="crawl_vrain_selenium_start"),
    path("crawl-vrain-selenium/tail/", crawl_vrain_selenium_tail_view, name="crawl_vrain_selenium_tail"),
    
    
    # ============================================================
    # 4. DATA MANAGEMENT - XEM & TẢI DỮ LIỆU
    # ============================================================
    # 4.1) DANH SÁCH DATASETS
    # Method: GET
    # Giải thích:
    #   - Trang hiển thị danh sách file datasets (output/, merged/, cleaned/)
    #   - Có thể chọn file để xem/tải
    path("datasets/", datasets_view, name="datasets"),
    
    # 4.2) XEM DATASET (preview data trong bảng)
    # Method: GET
    # Tham số URL:
    #   - folder: "output", "merged", "cleaned", "cleaned_merge", "cleaned_raw"
    #   - filename: tên file (csv, xlsx, ...)
    # Giải thích:
    #   - Hiển thị preview dữ liệu với pagination
    #   - Cho phép scroll, filter, sắp xếp
    path("datasets/view/<str:folder>/<str:filename>/", dataset_view_view, name="dataset_view"),
    
    # 4.3) TẢI DATASET
    # Method: GET
    # Giải thích:
    #   - Trigger download file (CSV, Excel, v.v.)
    #   - Frontend thường dùng <a href="...download..."> hoặc fetch()
    path("datasets/download/<str:folder>/<str:filename>/", dataset_download_view, name="dataset_download"),
    
    
    # ============================================================
    # 5. DATA PROCESSING - GHÉP FILE
    # ============================================================
    # Method: POST
    # Giải thích:
    #   - Endpoint để ghép (merge) nhiều file Excel/CSV
    #   - Script Merge_xlsx.py sẽ chạy bằng subprocess
    #   - Trả JSON kết quả (như số file merge, thông tin file output)
    path("datasets/merge/", merge_data_view, name="merge_data"),
    
    
    # ============================================================
    # 6. DATA PROCESSING - LÀM SẠCH DỮ LIỆU
    # ============================================================
    # 6.1) MAIN VIEW (Form analyze/clean)
    # Method: GET (form), POST (start job)
    # Giải thích:
    #   - Gửi JSON {filename, file_type, action}
    #   - action="analyze": phân tích missing data, tạo heatmap
    #   - action="clean": làm sạch + xuất file + tạo report
    path("datasets/clean/", clean_data_view, name="clean_data"),
    
    # 6.2) DANH SÁCH FILE ĐÃ CLEAN
    # Method: GET
    # Giải thích:
    #   - Endpoint JSON trả danh sách file đã clean
    #   - Hiển thị tên file, size, thời gian
    path("datasets/clean/list/", clean_files_list_view, name="clean_list"),
    
    # 6.3) LOG REALTIME KHI CLEANING
    # Method: GET
    # Giải thích:
    #   - Endpoint JSON cho frontend polling
    #   - Backend trả các dòng log mới nhất
    path("datasets/clean/tail/", clean_data_tail_view, name="clean_tail"),
    
    
    # ============================================================
    # 7. AUTHENTICATION - ĐĂNG NHẬP / ĐĂNG KÝ / TÀI KHOẢN
    # ============================================================
    # 7.1) ĐĂNG NHẬP
    # Method: GET (form), POST (authenticate)
    path("auth/login/", login_view, name="login"),
    
    # 7.2) ĐĂNG KÝ
    # Method: GET (form), POST (create account)
    path("auth/register/", register_view, name="register"),
    
    # 7.3) ĐĂNG XUẤT
    # Method: POST
    path("auth/logout/", logout_view, name="logout"),
    
    # 7.4) HỒ SƠ CÁ NHÂN
    # Method: GET (xem), POST/PATCH (sửa)
    path("auth/profile/", profile_view, name="profile"),
    
    # 7.5) QUÊN MẬT KHẨU (LUỒNG EMAIL WITH OTP)
    # Method: GET (form), POST (yêu cầu reset)
    # - Người dùng nhập email
    # - Backend gửi OTP qua email
    path("auth/forgot-password/", forgot_password_view, name="forgot_password"),
    
    # 7.6) THÔNG BÁO ĐÃ GỬI EMAIL
    # Method: GET
    # - Trang thông báo "Email đã gửi, vui lòng kiểm tra hộp thư"
    path("auth/password-reset-sent/", password_reset_sent_view, name="password_reset_sent"),
    
    # 7.7) NHẬP OTP ĐẶT LẠI MẬT KHẨU (LUỒNG VỚI OTP)
    # Method: POST
    path("auth/forgot-password-otp/", forgot_password_otp_view, name="forgot_password_otp"),
    
    # 7.8) VERIFY OTP (kiểm tra mã OTP)
    # Method: POST
    path("auth/verify-otp/", verify_otp_view, name="verify_otp"),
    
    # 7.9) RESET MẬT KHẨU SAU KHI VERIFY OTP
    # Method: POST
    # - Người dùng nhập mật khẩu mới
    path("auth/reset-password-otp/", reset_password_otp_view, name="reset_password_otp"),
    
    # 7.10) VERIFY EMAIL KHI ĐĂNG KÝ
    # Method: POST
    # - Người dùng nhập OTP từ email để kích hoạt tài khoản
    path("auth/verify-email-register/", verify_email_register_view, name="verify_email_register"),
    
    # 7.11) GỬI LẠI EMAIL OTP (nếu user không nhận được hoặc hết hạn)
    # Method: POST
    path("auth/resend-email-otp/", resend_email_otp_view, name="resend_email_otp"),
    
    # 7.12) CÓ THỂHỦY ĐĂNG KÝ (trong quá trình chưa hoàn thành)
    # Method: POST
    path("auth/cancel-register/", cancel_register_view, name="cancel_register"),

]