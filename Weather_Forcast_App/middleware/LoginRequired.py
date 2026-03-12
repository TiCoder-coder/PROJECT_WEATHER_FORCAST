"""
LoginRequiredMiddleware
=======================
Middleware bảo vệ tất cả các route — chỉ cho phép truy cập khi đã đăng nhập.

Cách hoạt động:
- Nếu path nằm trong PUBLIC_PATHS (login, register, static...) → cho qua không kiểm tra.
- Nếu user chưa đăng nhập (không có access_token trong session):
    + Request AJAX / JSON API  → trả về JSON 401 {"error": "...", "redirect": "/auth/login/"}
    + Request HTML thông thường → redirect về trang login kèm ?next=<path hiện tại>
"""

import re
from django.http import JsonResponse
from django.shortcuts import redirect
from django.urls import reverse

# ============================================================
# DANH SÁCH CÁC PATH KHÔNG CẦN ĐĂNG NHẬP (PUBLIC)
# ============================================================
# Mỗi phần tử là regex được match với request.path_info
_PUBLIC_PATTERNS = [
    r"^/$",                                  # Trang chủ (home)
    r"^/auth/login/",                        # Đăng nhập
    r"^/auth/register/",                     # Đăng ký
    r"^/auth/logout/",                       # Đăng xuất
    r"^/auth/forgot-password/",             # Quên mật khẩu
    r"^/auth/password-reset-sent/",         # Thông báo gửi email
    r"^/auth/forgot-password-otp/",         # Nhập OTP reset
    r"^/auth/verify-otp/",                  # Verify OTP
    r"^/auth/reset-password-otp/",          # Đặt lại mật khẩu qua OTP
    r"^/auth/verify-email-register/",       # Verify email đăng ký
    r"^/auth/resend-email-otp/",            # Gửi lại OTP
    r"^/auth/reset-password/",              # Reset mật khẩu qua token
    r"^/auth/password-reset-complete/",     # Hoàn tất reset
    r"^/auth/cancel-register/",             # Hủy đăng ký
    r"^/admin/",                             # Django admin
    r"^/static/",                            # Static files
    r"^/media/",                             # Media files
]

_compiled = [re.compile(p) for p in _PUBLIC_PATTERNS]


def _is_public(path: str) -> bool:
    return any(p.match(path) for p in _compiled)


def _is_ajax(request) -> bool:
    """Phát hiện request AJAX / API (JSON) theo nhiều dấu hiệu."""
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return True
    accept = request.headers.get("Accept", "")
    if "application/json" in accept:
        return True
    content_type = request.headers.get("Content-Type", "")
    if "application/json" in content_type:
        return True
    # Các endpoint API nội bộ luôn trả JSON
    ajax_suffixes = (
        "/start/", "/tail/", "/logs/",
        "/merge/", "/clean/", "/list/",
        "/run/", "/manual/", "/model-info/",
        "/forecast-now/", "/configs/",
        "/artifacts/", "/tune/start/", "/tune/tail/",
    )
    return any(request.path_info.endswith(s) for s in ajax_suffixes)


class LoginRequiredMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if not _is_public(request.path_info):
            is_logged_in = bool(request.session.get("access_token"))

            if not is_logged_in:
                login_url = reverse("weather:login")

                if _is_ajax(request):
                    return JsonResponse(
                        {
                            "error": "Bạn cần đăng nhập để sử dụng chức năng này.",
                            "redirect": login_url,
                        },
                        status=401,
                    )

                # Lưu path hiện tại để sau khi login redirect về đúng trang
                next_url = request.get_full_path()
                return redirect(f"{login_url}?next={next_url}")

        return self.get_response(request)
