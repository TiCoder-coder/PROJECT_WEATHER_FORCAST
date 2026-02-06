"""
Django Settings - WeatherForcast Project
=========================================

Đây là file cấu hình chính (settings) của dự án Django.

Mục đích:
    - Cấu hình database, authentication, static files
    - Cấu hình email, JWT tokens, session/cookies
    - Cấu hình apps, middleware, templates
    - Quản lý biến môi trường (secrets, keys, v.v.)

Tổ chức file:
    - Load environment variables từ .env
    - Cấu hình Django (apps, middleware, templates, databases)
    - Cấu hình authentication (JWT, sessions, cookies)
    - Cấu hình email (SMTP)
    - Cấu hình static files

Lưu ý:
    - Không commit .env vào git (chứa secrets)
    - DEBUG = False nên Deployment
    - ALLOWED_HOSTS cần cấu hình đầy đủ cho production
"""

from pathlib import Path
from decouple import config
from dotenv import load_dotenv
import os

# ============================================================
# 1. ĐƯỜNG DẪN & BASE_DIR
# ============================================================
# BASE_DIR: đường dẫn gốc của project
# - Được tính bằng:
#   + __file__ = settings.py
#   + .resolve().parent.parent = đi lên 2 cấp thư mục
# - Dùng để định vị các thư mục con (static, media, templates, ...)
BASE_DIR = Path(__file__).resolve().parent.parent

# ============================================================
# 2. LOAD ENVIRONMENT VARIABLES (.env)
# ============================================================
# load_dotenv(path) sẽ đọc file .env và đẩy các biến vào os.environ
# Ở đó, config("KEY") sẽ lấy giá trị từ .env hoặc biến hệ thống
# 
# Lợi ích:
# - Không hardcode secrets trong source code
# - Dễ dàng thay đổi cấu hình mà không sửa code
# - Bảo mật hơn khi push lên git
load_dotenv(os.path.join(BASE_DIR, ".env"))
print("Loaded .env from:", os.path.join(BASE_DIR, ".env"))

# ============================================================
# 3. SECURITY - KEYS & SECRETS
# ============================================================
# SECRET_KEY: khóa bí mật cho Django (dùng để mã hóa session, CSRF tokens, v.v.)
# - KHÔNG nên để mặc định, phải cấu hình trong .env
# - Nếu bị leak => cần thay đổi ngay
SECRET_KEY = config("SECRET_KEY")

# ============================================================
# 4. DATABASE - MONGODB
# ============================================================
# MONGO_URI: connection string kết nối MongoDB
# - Format: mongodb://user:pass@host:port/?replicaSet=rs0&...
# - Có thể kết nối 1 node hoặc replica set
# - project dùng MongoDB thay vì SQLite/PostgreSQL
MONGO_URI = config("MONGO_URI", default="")

# DB_NAME: tên database trong MongoDB
# - Mặc định là "Login" (chứa collection logins, password_reset_otps, v.v.)
# - Có thể thay đổi để tách dev/prod
DB_NAME = config("DB_NAME", default="Login")

# ============================================================
# 5. JWT AUTHENTICATION (JSON Web Tokens)
# ============================================================
# JWT_SECRET: khóa bí mật để sign/verify JWT tokens
# - Dùng để xác thực API requests (thay cho session cookies)
# - Client sẽ gửi: Authorization: Bearer <token>
JWT_SECRET = config("JWT_SECRET", default="")

# JWT_ALGORITHM: thuật toán ký token (HS256, RS256, v.v.)
# - HS256 (HMAC-SHA256) là mặc định, đơn giản, phù hợp dự án nhỏ-vừa
# - RS256 (RSA) phức tạp hơn nhưng bảo mật hơn (để production lớn)
JWT_ALGORITHM = config("JWT_ALGORITHM", default="HS256")

# ACCESS_TOKEN_EXPIRE_HOURS: thời gian hết hạn access token (giờ)
# - Mặc định: 3 giờ
# - Token hết hạn => user cần refresh lại
ACCESS_TOKEN_EXPIRE_HOURS = config("ACCESS_TOKEN_EXPIRE_HOURS", cast=int, default=3)

# REFRESH_TOKEN_EXPIRE_DAYS: thời gian hết hạn refresh token (ngày)
# - Mặc định: 1 ngày
# - Refresh token dùng để lấy access token mới mà không cần đăng nhập lại
REFRESH_TOKEN_EXPIRE_DAYS = config("REFRESH_TOKEN_EXPIRE_DAYS", cast=int, default=1)

# JWT_ISSUER: tổ chức phát hành token (claim iss)
# - Giúp verify token từ đúng server, chặn token từ server khác
JWT_ISSUER = config("JWT_ISSUER", default="weather_api")

# JWT_AUDIENCE: đối tượng nhận token (claim aud)
# - Giúp verify token dùng cho đúng mục đích
JWT_AUDIENCE = config("JWT_AUDIENCE", default="weather_web")

# ============================================================
# 6. SESSION & COOKIE SECURITY
# ============================================================
# SESSION_COOKIE_HTTPONLY: chỉ gửi cookie qua HTTP (không cho JS truy cập)
# - Chống XSS (JavaScript injection không thể đọc cookie)
# - Nên luôn = True cho production
SESSION_COOKIE_HTTPONLY = True

# CSRF_COOKIE_HTTPONLY: CSRF token có thể truy cập từ JS không?
# - False: frontend JS có thể đọc CSRF token để gửi cùng request
# - True: CSRF token chỉ gửi qua HTTP header (backend tự quản lý)
# - Project này dùng JS fetch, vì vậy để False để frontend lấy token dễ dàng
CSRF_COOKIE_HTTPONLY = False

# SESSION_COOKIE_SAMESITE: chính sách same-site cho session cookie
# - "Lax": gửi khi click link, nhưng không gửi khi cross-origin request bình thường
# - "Strict": chỉ gửi cùng site (bảo mật hơn nhưng khắt khe hơn)
# - "None": luôn gửi (yêu cầu SECURE=True)
# - "Lax" là cân bằng tốt giữa bảo mật và UX
SESSION_COOKIE_SAMESITE = "Lax"

# CSRF_COOKIE_SAMESITE: same-site cho CSRF token
CSRF_COOKIE_SAMESITE = "Lax"

# ============================================================
# 7. DEBUG MODE
# ============================================================
# DEBUG = True: chế độ phát triển
# - Hiển thị error page chi tiết (rất hữu ích khi debug)
# - Reloader tự động khi code thay đổi
# - Không cache templates
# ⚠️ PHẢI = False khi deploy (lộ sensitive info)
DEBUG = True

# ============================================================
# 8. ALLOWED HOSTS
# ============================================================
# Danh sách các hostname/IP được phép truy cập
# - Chỉ host trong danh sách này mới được Django chấp nhận
# - Chặn Host Header Injection attacks
# - Thêm domain thực tế khi deploy:
#   ["yourdomain.com", "www.yourdomain.com", "api.yourdomain.com"]
ALLOWED_HOSTS = ["127.0.0.1", "localhost"]

# ============================================================
# 9. INSTALLED APPS
# ============================================================
# Danh sách các app được cài đặt (Django apps + custom apps)
INSTALLED_APPS = [
    # Django built-in apps
    "django.contrib.admin",              # Admin panel
    "django.contrib.auth",               # Authentication framework (User model, permissions)
    "django.contrib.contenttypes",       # Content type framework (model metadata)
    "django.contrib.sessions",           # Session framework
    "django.contrib.messages",           # Message framework (gishow messages tạm thời)
    "django.contrib.staticfiles",        # Static files management
    
    # Custom app
    "Weather_Forcast_App",               # App chính của project (views, models, migrations)
]

# ============================================================
# 10. MIDDLEWARE
# ============================================================
# Middleware là một lớp xử lý request/response
# - Chạy tuần tự khi request đến, rồi chạy ngược lại khi response đi
# Thứ tự có ý nghĩa!
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",       # Thêm headers bảo mật
    "django.contrib.sessions.middleware.SessionMiddleware", # Quản lý sessions
    "django.middleware.common.CommonMiddleware",           # Normalize URL, gzip response
    "django.middleware.csrf.CsrfViewMiddleware",          # Bảo vệ CSRF attacks
    "django.contrib.auth.middleware.AuthenticationMiddleware", # Gắn user vào request
    "django.contrib.messages.middleware.MessageMiddleware",   # Message framework
    "django.middleware.clickjacking.XFrameOptionsMiddleware",  # Chặn clickjacking (X-Frame-Options)
]

# ============================================================
# 11. URL CONFIGURATION
# ============================================================
# ROOT_URLCONF: file chứa urlpatterns chính
# - Django sẽ import file này để routing URL
ROOT_URLCONF = "WeatherForcast.urls"

# ============================================================
# 12. TEMPLATES
# ============================================================
# Cấu hình template engine (mặc định là Django's built-in)
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        
        # DIRS: thư mục chứa templates (tìm kiếm trước)
        # [] = không có thư mục global, tìm trong từng app
        "DIRS": [],
        
        # APP_DIRS: tìm templates trong thư mục templates/ của mỗi app
        # True = tìm trong Weather_Forcast_App/templates/
        "APP_DIRS": True,
        
        # OPTIONS: cấu hình template engine
        "OPTIONS": {
            # context_processors: hàm tự động thêm biến vào context khi render
            "context_processors": [
                "django.template.context_processors.request",  # request object
                "django.contrib.auth.context_processors.auth", # user object
                "django.contrib.messages.context_processors.messages", # messages
            ],
        },
    },
]

# ============================================================
# 13. WSGI APPLICATION
# ============================================================
# WSGI_APPLICATION: entry point cho WSGI server (Gunicorn, uWSGI, v.v.)
# - Production server sẽ gọi hàm application() từ file này
WSGI_APPLICATION = "WeatherForcast.wsgi.application"

# ============================================================
# 14. DATABASE CONFIGURATION
# ============================================================
# Django mặc định dùng SQLite (test app tạo sẵn)
# Project dùng MongoDB qua wrapper, nhưng vẫn giữ SQLite cho Django admin models
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

# ============================================================
# 15. PASSWORD VALIDATION
# ============================================================
# Validators kiểm tra mật khẩu khi user.set_password()
# Giảm khả năng bị brute force/guess
AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
        # Chặn mật khẩu quá giống username/email/...
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
        # Mặc định: tối thiểu 8 ký tự
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
        # Chặn mật khẩu phổ biến (password123, qwerty, v.v.)
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
        # Chặn mật khẩu chỉ toàn số
    },
]

# ============================================================
# 16. LOCALIZATION
# ============================================================
# LANGUAGE_CODE: ngôn ngữ mặc định (cho Django messages, forms, v.v.)
LANGUAGE_CODE = "en-us"

# TIME_ZONE: timezone cho datetime
# - "Asia/Ho_Chi_Minh" = UTC+7 (Việt Nam)
# - Tất cả datetime được save ở UTC trong DB
# - Lúc hiển thị, Django sẽ convert sang TIME_ZONE này
TIME_ZONE = "Asia/Ho_Chi_Minh"

# USE_I18N: bật internationalization
USE_I18N = True

# USE_TZ: bật timezone aware datetime
# - True: datetime.now() sẽ có timezone info
# - False: naive datetime (không có timezone)
USE_TZ = True

# ============================================================
# 17. STATIC FILES
# ============================================================
# STATIC_URL: URL base cho static files (CSS, JS, images)
# - Truy cập: http://localhost:8000/static/style.css
STATIC_URL = "static/"

# STATICFILES_DIRS: thư mục chứa static files (tìm kiếm ngoài app dirs)
# - Django sẽ tìm trong danh sách này + trong static/ của mỗi app
# - Weather_Forcast_App/static/ chứa CSS, JS, images của project
STATICFILES_DIRS = [
    BASE_DIR / "Weather_Forcast_App" / "static",
]

# ============================================================
# 18. MODEL CONFIGURATION
# ============================================================
# DEFAULT_AUTO_FIELD: kiểu field auto-increment mặc định cho model
# - "django.db.models.BigAutoField" = int64 (tối đa ~9 tỷ bản ghi)
# - "django.db.models.AutoField" = int32 (cũ hơn)
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ============================================================
# 19. EMAIL CONFIGURATION (SMTP)
# ============================================================
# Cấu hình gửi email (OTP, password reset, notifications, v.v.)

# EMAIL_BACKEND: backend gửi email
# - "django.core.mail.backends.smtp.EmailBackend" = SMTP thật
# - "django.core.mail.backends.console.EmailBackend" = in ra console (test)
EMAIL_BACKEND = "django.core.mail.backends.smtp.EmailBackend"

# EMAIL_HOST: SMTP server (ví dụ: gmail, outlook, custom server)
EMAIL_HOST = config("EMAIL_HOST", default="")

# EMAIL_PORT: port SMTP
# - 587 = STARTTLS (mã hóa TLS)
# - 465 = SSL (mã hóa từ đầu)
# - 25 = không mã hóa (cũ, thường spam)
EMAIL_PORT = config("EMAIL_PORT", cast=int, default=587)

# EMAIL_HOST_USER: username/email tài khoản SMTP
EMAIL_HOST_USER = config("EMAIL_HOST_USER", default="")

# EMAIL_HOST_PASSWORD: password tài khoản SMTP
# ⚠️ Không hardcode! Lấy từ .env
EMAIL_HOST_PASSWORD = config("EMAIL_HOST_PASSWORD", default="")

# EMAIL_USE_TLS: dùng TLS khi kết nối (True nên dùng)
# - True: STARTTLS protocol
# - False: SSL hoặc không mã hóa
EMAIL_USE_TLS = config("EMAIL_USE_TLS", cast=bool, default=True)

# DEFAULT_FROM_EMAIL: email "người gửi" mặc định
# - Khi gửi email từ Django, nếu không chỉ định from_email thì dùng cái này
# - Thường = EMAIL_HOST_USER (hoặc "noreply@yourdomain.com")
DEFAULT_FROM_EMAIL = config("DEFAULT_FROM_EMAIL", default=EMAIL_HOST_USER)

# ============================================================
# 20. OTP & PASSWORD RESET CONFIGURATION
# ============================================================
# PASSWORD_RESET_OTP_EXPIRE_SECONDS: thời gian hết hạn OTP (giây)
# - Mặc định: 600 giây = 10 phút
# - User có 10 phút để enter OTP trước khi phải yêu cầu gửi lại
PASSWORD_RESET_OTP_EXPIRE_SECONDS = config("PASSWORD_RESET_OTP_EXPIRE_SECONDS", cast=int, default=600)

# PASSWORD_RESET_OTP_MAX_ATTEMPTS: số lần nhập SAI OTP tối đa
# - Mặc định: 5 lần
# - Sau 5 lần SAI, account có thể bị lock tạm thời
PASSWORD_RESET_OTP_MAX_ATTEMPTS = config("PASSWORD_RESET_OTP_MAX_ATTEMPTS", cast=int, default=5)