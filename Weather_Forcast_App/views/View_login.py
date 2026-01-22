from django.shortcuts import render, redirect
from django.contrib import messages
from django.views.decorators.http import require_http_methods
from django.urls import reverse
from datetime import datetime
from Weather_Forcast_App.scripts.Login_services import ManagerService
from Weather_Forcast_App.scripts.Email_validator import EmailValidator, EmailValidationError
from Weather_Forcast_App.middleware.Jwt_handler import create_access_token
from bson import ObjectId

SESSION_RESET_EMAIL = "reset_email"
SESSION_RESET_OTP_OK = "reset_otp_ok"
SESSION_RESET_OTP = "reset_otp"

# Session keys cho đăng ký với xác thực email
SESSION_REGISTER_DATA = "register_data"
SESSION_REGISTER_EMAIL_VERIFIED = "register_email_verified"

def _make_json_safe(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]
    return obj


def _extract_error_message(exception):
    """
    Trích xuất thông báo lỗi từ exception, xử lý cả ValidationError của DRF
    """
    from rest_framework.exceptions import ValidationError, PermissionDenied
    
    if isinstance(exception, (ValidationError, PermissionDenied)):
        detail = exception.detail
        # Nếu là list
        if isinstance(detail, list):
            # Lấy phần tử đầu tiên và convert sang string
            if len(detail) > 0:
                return str(detail[0])
            return "Có lỗi xảy ra"
        # Nếu là dict
        if isinstance(detail, dict):
            # Lấy lỗi đầu tiên
            for key, value in detail.items():
                if isinstance(value, list) and len(value) > 0:
                    return str(value[0])
                return str(value)
            return "Có lỗi xảy ra"
        # Nếu là string hoặc ErrorDetail
        return str(detail)
    
    # Exception thông thường
    return str(exception)

class SessionUser:
    def __init__(self, data: dict):
        self.username = data.get("userName") or data.get("username")
        self.email = data.get("email")
        full_name = data.get("name") or ""
        parts = full_name.split(" ", 1)
        self.first_name = parts[0] if parts else ""
        self.last_name = parts[1] if len(parts) > 1 else ""
        self.date_joined = data.get("createdAt")
        self.last_login = data.get("last_login")

    def get_full_name(self):
        return (self.first_name + " " + self.last_name).strip()


def _require_session_login(request):
    profile = request.session.get("profile")
    token = request.session.get("access_token")
    if not profile or not token:
        return None
    return profile


@require_http_methods(["GET", "POST"])
def login_view(request):
    if request.method == "GET":
        # Nếu đã đăng nhập thì chuyển về profile
        if request.session.get("access_token"):
            return redirect("weather:profile")
        return render(request, "weather/auth/Login.html")

    identifier = request.POST.get("username", "").strip()
    password = request.POST.get("password", "")

    if not identifier:
        messages.error(request, "⚠️ Vui lòng nhập tên đăng nhập hoặc email.")
        return redirect("weather:login")
    
    if not password:
        messages.error(request, "⚠️ Vui lòng nhập mật khẩu.")
        return redirect("weather:login")

    try:
        manager = ManagerService.authenticate(identifier, password)

        token = create_access_token({
            "manager_id": manager["_id"],
            "role": manager.get("role", "guest"),
        })

        request.session["access_token"] = token
        request.session["profile"] = _make_json_safe(manager)

        messages.success(request, f"✅ Đăng nhập thành công! Chào mừng {manager.get('name', manager.get('userName'))}!")
        return redirect("weather:home")

    except Exception as e:
        error_msg = _extract_error_message(e)
        messages.error(request, f"❌ {error_msg}")
        return redirect("weather:login")


@require_http_methods(["GET", "POST"])
def register_view(request):
    if request.method == "GET":
        # Nếu đã đăng nhập thì chuyển về profile
        if request.session.get("access_token"):
            return redirect("weather:profile")
        return render(request, "weather/auth/Register.html")

    # Lấy dữ liệu từ form (đúng theo Register.html hiện tại)
    first_name = request.POST.get("first_name", "").strip()
    last_name = request.POST.get("last_name", "").strip()
    name = f"{first_name} {last_name}".strip()

    userName = request.POST.get("username", "").strip()
    email = request.POST.get("email", "").strip().lower()  # Normalize email
    password = request.POST.get("password", "")
    confirm_password = request.POST.get("confirm_password", "")

    # Validate cơ bản phía server
    if not first_name:
        messages.error(request, "⚠️ Vui lòng nhập Họ của bạn.")
        return render(request, "weather/auth/Register.html", {
            "form_data": {"first_name": first_name, "last_name": last_name, "username": userName, "email": email}
        })
    
    if not last_name:
        messages.error(request, "⚠️ Vui lòng nhập Tên của bạn.")
        return render(request, "weather/auth/Register.html", {
            "form_data": {"first_name": first_name, "last_name": last_name, "username": userName, "email": email}
        })

    if not userName:
        messages.error(request, "⚠️ Vui lòng nhập tên đăng nhập.")
        return render(request, "weather/auth/Register.html", {
            "form_data": {"first_name": first_name, "last_name": last_name, "username": userName, "email": email}
        })
    
    if not email:
        messages.error(request, "⚠️ Vui lòng nhập địa chỉ email.")
        return render(request, "weather/auth/Register.html", {
            "form_data": {"first_name": first_name, "last_name": last_name, "username": userName, "email": email}
        })
    
    if not password:
        messages.error(request, "⚠️ Vui lòng nhập mật khẩu.")
        return render(request, "weather/auth/Register.html", {
            "form_data": {"first_name": first_name, "last_name": last_name, "username": userName, "email": email}
        })

    if password != confirm_password:
        messages.error(request, "⚠️ Mật khẩu xác nhận không khớp. Vui lòng nhập lại.")
        return render(request, "weather/auth/Register.html", {
            "form_data": {"first_name": first_name, "last_name": last_name, "username": userName, "email": email}
        })

    # Validate email tồn tại thực sự (kiểm tra MX records và disposable)
    try:
        email_validation = EmailValidator.validate_email_exists(email)
        if not email_validation['valid']:
            messages.error(request, ', '.join(email_validation['errors']))
            return render(request, "weather/auth/Register.html", {
                "form_data": {"first_name": first_name, "last_name": last_name, "username": userName, "email": email}
            })
    except Exception as e:
        messages.error(request, f"Lỗi kiểm tra email: {str(e)}")
        return render(request, "weather/auth/Register.html", {
            "form_data": {"first_name": first_name, "last_name": last_name, "username": userName, "email": email}
        })

    # Kiểm tra username và email đã tồn tại chưa
    from Weather_Forcast_App.Repositories.Login_repositories import LoginRepository
    if LoginRepository.find_by_username(userName):
        messages.error(request, f"❌ Tên đăng nhập '{userName}' đã được sử dụng. Vui lòng chọn tên khác.")
        return render(request, "weather/auth/Register.html", {
            "form_data": {"first_name": first_name, "last_name": last_name, "username": "", "email": email}
        })
    if LoginRepository.find_by_username_or_email(email):
        messages.error(request, f"❌ Email '{email}' đã được đăng ký. Vui lòng sử dụng email khác hoặc đăng nhập nếu đã có tài khoản.")
        return render(request, "weather/auth/Register.html", {
            "form_data": {"first_name": first_name, "last_name": last_name, "username": userName, "email": ""}
        })

    # Kiểm tra độ mạnh mật khẩu
    if not ManagerService.check_password_strength(password):
        errors = ManagerService.get_password_strength_errors(password)
        messages.error(request, "⚠️ Mật khẩu chưa đủ mạnh: " + ", ".join(errors))
        return render(request, "weather/auth/Register.html", {
            "form_data": {"first_name": first_name, "last_name": last_name, "username": userName, "email": email}
        })

    # Lưu thông tin đăng ký vào session và gửi OTP xác thực email
    try:
        # Gửi OTP xác thực email
        EmailValidator.send_verification_otp(email, name)
        
        # Lưu thông tin đăng ký vào session
        request.session[SESSION_REGISTER_DATA] = {
            "name": name,
            "first_name": first_name,
            "last_name": last_name,
            "userName": userName,
            "email": email,
            "password": password,
            "role": "staff",
        }
        request.session[SESSION_REGISTER_EMAIL_VERIFIED] = False
        
        messages.success(request, f"📧 Mã OTP đã được gửi đến {email}. Vui lòng kiểm tra hộp thư (bao gồm cả thư mục Spam) để xác thực.")
        return redirect("weather:verify_email_register")
        
    except EmailValidationError as e:
        messages.error(request, f"❌ {str(e)}")
        return render(request, "weather/auth/Register.html", {
            "form_data": {"first_name": first_name, "last_name": last_name, "username": userName, "email": email}
        })
    except Exception as e:
        messages.error(request, f"❌ Không thể gửi email xác thực. Vui lòng thử lại sau. Chi tiết: {str(e)}")
        return render(request, "weather/auth/Register.html", {
            "form_data": {"first_name": first_name, "last_name": last_name, "username": userName, "email": email}
        })



@require_http_methods(["GET"])
def logout_view(request):
    request.session.flush()
    messages.info(request, "Bạn đã đăng xuất.")
    return redirect("weather:login")


@require_http_methods(["GET", "POST"])
def profile_view(request):
    profile = _require_session_login(request)
    if not profile:
        messages.warning(request, "Bạn cần đăng nhập.")
        return redirect("weather:login")

    user_obj = SessionUser(profile)
    
    if request.method == "GET":
        return render(request, "weather/auth/Profile.html", {"user": user_obj, "profile": profile})
    
    # POST - Update profile
    name = request.POST.get("name", "").strip()
    email = request.POST.get("email", "").strip().lower()
    
    if not name:
        messages.error(request, "Họ tên không được để trống.")
        return render(request, "weather/auth/Profile.html", {"user": user_obj, "profile": profile})
    
    try:
        from Weather_Forcast_App.Repositories.Login_repositories import LoginRepository
        from datetime import datetime, timezone
        
        user_id = profile.get("_id")
        
        # Chuẩn bị dữ liệu update
        update_data = {
            "name": name,
            "updatedAt": datetime.now(timezone.utc)
        }
        
        # Chỉ update email nếu thay đổi
        old_email = profile.get("email", "")
        if email and email != old_email:
            # Kiểm tra email đã tồn tại chưa
            existing = LoginRepository.find_by_username_or_email(email)
            if existing and str(existing.get("_id")) != str(user_id):
                messages.error(request, "Email đã được sử dụng bởi tài khoản khác.")
                return render(request, "weather/auth/Profile.html", {"user": user_obj, "profile": profile})
            update_data["email"] = email
        
        # Update trong database
        LoginRepository.update_by_id(user_id, update_data)
        
        # Cập nhật session
        profile["name"] = name
        if email:
            profile["email"] = email
        request.session["profile"] = _make_json_safe(profile)
        
        messages.success(request, "✅ Cập nhật thông tin thành công!")
        return redirect("weather:profile")
        
    except Exception as e:
        error_msg = _extract_error_message(e)
        messages.error(request, f"❌ Lỗi cập nhật: {error_msg}")
        return render(request, "weather/auth/Profile.html", {"user": user_obj, "profile": profile})


@require_http_methods(["GET", "POST"])
def forgot_password_view(request):
    if request.method == "GET":
        return render(request, "weather/auth/Forgot_password.html")

    identifier = request.POST.get("email", "").strip()  # form đang đặt name="email"
    try:
        token = ManagerService.generate_token(identifier)

        reset_link = request.build_absolute_uri(
            reverse("weather:reset_password", kwargs={"token": token})
        )

        # Dev mode: in ra console để test (sau này thay bằng gửi email)
        print("========== RESET LINK (DEV) ==========")
        print(reset_link)
        print("======================================")

        request.session["last_reset_link"] = reset_link
        messages.success(request, "Nếu tài khoản tồn tại, link reset đã được tạo (dev: xem terminal).")
        return redirect("weather:password_reset_sent")

    except Exception as e:
        error_msg = _extract_error_message(e)
        messages.error(request, f"❌ {error_msg}")
        return redirect("weather:forgot_password")


@require_http_methods(["GET"])
def password_reset_sent_view(request):
    # Bạn có thể show link dev trong template nếu muốn
    return render(request, "weather/auth/Password_reset_sent.html")


@require_http_methods(["GET", "POST"])
def reset_password_view(request, token: str):
    # Kiểm tra token hợp lệ để set validlink cho template
    try:
        ManagerService.verify_reset_token(token)
        validlink = True
    except Exception as e:
        validlink = False
        messages.error(request, str(e))

    if request.method == "GET":
        return render(request, "weather/auth/Reset_password.html", {"validlink": validlink})

    if not validlink:
        return render(request, "weather/auth/Reset_password.html", {"validlink": False})

    new_password = request.POST.get("new_password1", "")
    confirm_password = request.POST.get("new_password2", "")

    if new_password != confirm_password:
        messages.error(request, "Mật khẩu xác nhận không khớp.")
        return render(request, "weather/auth/Reset_password.html", {"validlink": True})

    try:
        ManagerService.reset_password_with_token(token, new_password)
        messages.success(request, "✅ Đặt lại mật khẩu thành công!")
        return redirect("weather:password_reset_complete")

    except Exception as e:
        error_msg = _extract_error_message(e)
        messages.error(request, f"❌ {error_msg}")
        return render(request, "weather/auth/Reset_password.html", {"validlink": True})


@require_http_methods(["GET"])
def password_reset_complete_view(request):
    return render(request, "weather/auth/Password_reset_complete.html")
@require_http_methods(["GET", "POST"])
def forgot_password_otp_view(request):
    if request.method == "GET":
        return render(request, "weather/auth/Forgot_password.html")

    email = request.POST.get("email", "").strip().lower()
    if not email:
        messages.error(request, "Vui lòng nhập email.")
        return redirect("weather:forgot_password_otp")

    try:
        # Gửi OTP và kiểm tra email tồn tại
        result = ManagerService.send_reset_otp(email)
        
        if not result["email_exists"]:
            messages.error(request, "❌ Email này chưa được đăng ký trong hệ thống. Vui lòng kiểm tra lại hoặc đăng ký tài khoản mới.")
            return redirect("weather:forgot_password_otp")
        
        if result["success"]:
            request.session[SESSION_RESET_EMAIL] = email
            messages.success(request, f"📧 {result['message']}. Vui lòng kiểm tra hộp thư (bao gồm cả Spam).")
            return redirect("weather:verify_otp")
        else:
            messages.error(request, f"❌ {result['message']}")
            return redirect("weather:forgot_password_otp")

    except Exception as e:
        error_msg = _extract_error_message(e)
        messages.error(request, f"❌ Gửi OTP thất bại: {error_msg}")
        return redirect("weather:forgot_password_otp")



@require_http_methods(["GET", "POST"])
def verify_otp_view(request):
    email = request.session.get(SESSION_RESET_EMAIL)
    if not email:
        messages.warning(request, "Vui lòng nhập email để nhận OTP trước.")
        return redirect("weather:forgot_password_otp")

    if request.method == "GET":
        return render(request, "weather/auth/Verify_otp.html", {"email": email})

    otp = request.POST.get("otp", "").strip()
    if not otp:
        messages.error(request, "Vui lòng nhập OTP.")
        return redirect("weather:verify_otp")

    try:
        ManagerService.verify_reset_otp(email, otp)

        request.session[SESSION_RESET_OTP_OK] = True
        request.session[SESSION_RESET_OTP] = otp  # giữ tạm để bước reset dùng
        messages.success(request, "✅ OTP hợp lệ. Bạn có thể đặt mật khẩu mới.")
        return redirect("weather:reset_password_otp")

    except Exception as e:
        error_msg = _extract_error_message(e)
        messages.error(request, f"❌ {error_msg}")
        return redirect("weather:verify_otp")


@require_http_methods(["GET", "POST"])
def reset_password_otp_view(request):
    email = request.session.get(SESSION_RESET_EMAIL)
    otp_ok = request.session.get(SESSION_RESET_OTP_OK)
    otp = request.session.get(SESSION_RESET_OTP)

    if not email or not otp_ok or not otp:
        messages.warning(request, "Phiên đặt lại mật khẩu không hợp lệ. Vui lòng làm lại.")
        return redirect("weather:forgot_password_otp")

    if request.method == "GET":
        return render(request, "weather/auth/Reset_password_otp.html")

    new_password = request.POST.get("new_password", "")
    confirm_password = request.POST.get("confirm_password", "")

    if new_password != confirm_password:
        messages.error(request, "Mật khẩu xác nhận không khớp.")
        return redirect("weather:reset_password_otp")

    try:
        ManagerService.reset_password_with_otp(email, otp, new_password)

        # clear session
        request.session.pop(SESSION_RESET_EMAIL, None)
        request.session.pop(SESSION_RESET_OTP_OK, None)
        request.session.pop(SESSION_RESET_OTP, None)

        messages.success(request, "✅ Đổi mật khẩu thành công! Hãy đăng nhập lại.")
        return redirect("weather:login")

    except Exception as e:
        error_msg = _extract_error_message(e)
        messages.error(request, f"❌ {error_msg}")
        return redirect("weather:reset_password_otp")


# ============== XÁC THỰC EMAIL KHI ĐĂNG KÝ ==============

@require_http_methods(["GET", "POST"])
def verify_email_register_view(request):
    """
    Xác thực email OTP khi đăng ký tài khoản mới
    """
    register_data = request.session.get(SESSION_REGISTER_DATA)
    
    if not register_data:
        messages.warning(request, "Vui lòng điền thông tin đăng ký trước.")
        return redirect("weather:register")
    
    email = register_data.get("email", "")
    
    if request.method == "GET":
        return render(request, "weather/auth/Verify_email_register.html", {
            "email": email,
            "name": register_data.get("name", "")
        })
    
    otp = request.POST.get("otp", "").strip()
    
    if not otp:
        messages.error(request, "Vui lòng nhập mã OTP.")
        return redirect("weather:verify_email_register")
    
    try:
        # Xác thực OTP
        EmailValidator.verify_email_otp(email, otp)
        
        # Đánh dấu email đã được xác thực
        request.session[SESSION_REGISTER_EMAIL_VERIFIED] = True
        
        # Tiến hành đăng ký tài khoản
        ManagerService.register_public(register_data, skip_email_verification=True)
        
        # Tự động đăng nhập sau khi đăng ký thành công
        try:
            manager = ManagerService.authenticate(register_data["userName"], register_data["password"])
            token = create_access_token({
                "manager_id": manager["_id"],
                "role": manager.get("role", "guest"),
            })
            request.session["access_token"] = token
            request.session["profile"] = _make_json_safe(manager)
            
            # Xóa dữ liệu đăng ký khỏi session
            request.session.pop(SESSION_REGISTER_DATA, None)
            request.session.pop(SESSION_REGISTER_EMAIL_VERIFIED, None)
            
            messages.success(request, f"🎉 Chào mừng {register_data.get('name', '')}! Tài khoản đã được tạo thành công.")
            return redirect("weather:home")
        except Exception as login_err:
            # Xóa dữ liệu đăng ký khỏi session
            request.session.pop(SESSION_REGISTER_DATA, None)
            request.session.pop(SESSION_REGISTER_EMAIL_VERIFIED, None)
            
            messages.success(request, "🎉 Tạo tài khoản thành công! Hãy đăng nhập để sử dụng.")
            return redirect("weather:login")
            
    except EmailValidationError as e:
        messages.error(request, f"❌ {str(e)}")
        return redirect("weather:verify_email_register")
    except Exception as e:
        error_msg = _extract_error_message(e)
        messages.error(request, f"❌ {error_msg}")
        return redirect("weather:verify_email_register")


@require_http_methods(["POST"])
def resend_email_otp_view(request):
    """
    Gửi lại OTP xác thực email đăng ký
    """
    register_data = request.session.get(SESSION_REGISTER_DATA)
    
    if not register_data:
        messages.warning(request, "Phiên đăng ký đã hết hạn. Vui lòng đăng ký lại.")
        return redirect("weather:register")
    
    email = register_data.get("email", "")
    name = register_data.get("name", "")
    
    try:
        EmailValidator.send_verification_otp(email, name)
        messages.success(request, f"📧 Mã OTP mới đã được gửi đến {email}.")
    except EmailValidationError as e:
        messages.error(request, f"❌ {str(e)}")
    except Exception as e:
        error_msg = _extract_error_message(e)
        messages.error(request, f"❌ Lỗi gửi email: {error_msg}")
    
    return redirect("weather:verify_email_register")


@require_http_methods(["GET"])
def cancel_register_view(request):
    """
    Hủy quá trình đăng ký và xóa session
    """
    request.session.pop(SESSION_REGISTER_DATA, None)
    request.session.pop(SESSION_REGISTER_EMAIL_VERIFIED, None)
    messages.info(request, "Đã hủy quá trình đăng ký.")
    return redirect("weather:register")