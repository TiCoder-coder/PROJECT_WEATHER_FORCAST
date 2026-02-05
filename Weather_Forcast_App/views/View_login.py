# ============================================================
# AUTH VIEWS (Login / Register / Profile / Forgot-Reset Password)
# ============================================================
# File này đang đóng vai trò "Controller/View" cho luồng xác thực (auth) trong Django:
# - Đăng nhập (login_view)
# - Đăng ký + xác thực email bằng OTP (register_view + verify_email_register_view)
# - Đăng xuất (logout_view)
# - Profile (profile_view)
# - Quên mật khẩu: 2 kiểu
#   1) Reset bằng token/link (forgot_password_view + reset_password_view)
#   2) Reset bằng OTP (forgot_password_otp_view + verify_otp_view + reset_password_otp_view)
#
# Bạn đang dùng "custom auth" dựa trên session:
# - request.session["access_token"] : JWT token (tự tạo) để đánh dấu đã login
# - request.session["profile"]      : thông tin user (manager) dạng dict
#
# Ngoài ra còn dùng session để lưu trạng thái OTP cho các bước verify/reset.
# ============================================================

from django.shortcuts import render, redirect
from django.contrib import messages
from django.views.decorators.http import require_http_methods
from django.urls import reverse
from datetime import datetime

# Service layer: nơi xử lý nghiệp vụ login/register/reset password...
from Weather_Forcast_App.scripts.Login_services import ManagerService

# EmailValidator: lớp kiểm tra email tồn tại + gửi/verify OTP
from Weather_Forcast_App.scripts.Email_validator import EmailValidator, EmailValidationError

# JWT handler: tạo access token (custom) để lưu vào session
from Weather_Forcast_App.middleware.Jwt_handler import create_access_token

# MongoDB ObjectId (BSON) - thường dùng khi thao tác ID trong MongoDB
# (Trong file này import nhưng hiện tại chưa thấy dùng trực tiếp ở code dưới.)
from bson import ObjectId


# ============================================================
# SESSION KEY CONSTANTS
# ============================================================
# Các biến hằng này để tránh "magic string" rải rác trong code
# => giảm sai chính tả, dễ refactor, dễ debug
SESSION_RESET_EMAIL = "reset_email"          # email dùng cho flow reset bằng OTP
SESSION_RESET_OTP_OK = "reset_otp_ok"        # cờ đánh dấu OTP đã verify đúng
SESSION_RESET_OTP = "reset_otp"              # OTP đã nhập và verify thành công

SESSION_REGISTER_DATA = "register_data"              # lưu tạm thông tin đăng ký trước khi verify email OTP
SESSION_REGISTER_EMAIL_VERIFIED = "register_email_verified"  # cờ đánh dấu đã verify email đăng ký


# ============================================================
# _make_json_safe(obj)
# ============================================================
# Mục đích:
# - Session của Django phải serialize được dữ liệu (thường JSON-like).
# - Một số kiểu dữ liệu (ví dụ datetime) không serialize JSON mặc định được.
# Hàm này sẽ "chuyển đổi" đệ quy:
# - datetime -> isoformat string
# - dict/list -> duyệt từng phần tử và chuyển đổi tương tự
# - còn lại -> giữ nguyên
def _make_json_safe(obj):
    # Nếu là datetime => đổi sang chuỗi ISO 8601
    if isinstance(obj, datetime):
        return obj.isoformat()

    # Nếu là dict => duyệt từng key-value và xử lý đệ quy
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}

    # Nếu là list => map từng phần tử
    if isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]

    # Các kiểu khác (str/int/float/bool/None/...) giữ nguyên
    return obj


# ============================================================
# _extract_error_message(exception)
# ============================================================
# Mục đích:
# - Khi gọi các service/repository/serializer có thể ném exception phức tạp
# - Đặc biệt DRF ValidationError thường có cấu trúc detail dạng list/dict
# Hàm này cố gắng "rút ra" message dễ hiểu nhất để show lên UI (messages.error)
def _extract_error_message(exception):
    """
    Trích xuất thông báo lỗi từ exception, xử lý cả ValidationError của DRF
    """
    # Import cục bộ để tránh phụ thuộc quá sớm (và tránh circular import nếu có)
    from rest_framework.exceptions import ValidationError, PermissionDenied
    
    # Nếu exception là ValidationError hoặc PermissionDenied của DRF
    if isinstance(exception, (ValidationError, PermissionDenied)):
        detail = exception.detail  # detail có thể là list/dict/string
        
        # Case 1: detail là list => thường dạng ["message 1", "message 2", ...]
        if isinstance(detail, list):
            if len(detail) > 0:
                return str(detail[0])  # ưu tiên lấy message đầu
            return "Có lỗi xảy ra"
        
        # Case 2: detail là dict => thường dạng {"field": ["msg"], ...}
        if isinstance(detail, dict):
            for key, value in detail.items():
                # Nếu value là list và có phần tử => lấy phần tử đầu
                if isinstance(value, list) and len(value) > 0:
                    return str(value[0])
                # Nếu value không phải list => convert trực tiếp
                return str(value)
            return "Có lỗi xảy ra"
        
        # Case 3: detail là string/other => stringify
        return str(detail)
    
    # Nếu không phải DRF error => stringify exception bình thường
    return str(exception)


# ============================================================
# SessionUser
# ============================================================
# Mục đích:
# - Chuyển dict "profile" trong session thành object dễ dùng trong template
# - Tách "name" thành first_name / last_name cho view/template
class SessionUser:
    def __init__(self, data: dict):
        # Hỗ trợ 2 kiểu key: userName hoặc username (phòng khi backend trả khác nhau)
        self.username = data.get("userName") or data.get("username")

        # Email
        self.email = data.get("email")

        # Full name có thể nằm ở key "name"
        full_name = data.get("name") or ""

        # split(" ", 1): tách tối đa 2 phần (first + phần còn lại)
        parts = full_name.split(" ", 1)
        self.first_name = parts[0] if parts else ""
        self.last_name = parts[1] if len(parts) > 1 else ""

        # Các field thời gian (có thể là datetime/iso string tùy _make_json_safe)
        self.date_joined = data.get("createdAt")
        self.last_login = data.get("last_login")

    # Helper cho template: hiện full name
    def get_full_name(self):
        return (self.first_name + " " + self.last_name).strip()


# ============================================================
# _require_session_login(request)
# ============================================================
# Mục đích:
# - Check user đã login chưa dựa vào session
# - Điều kiện: có "profile" và có "access_token"
# - Nếu chưa login => return None
def _require_session_login(request):
    profile = request.session.get("profile")
    token = request.session.get("access_token")
    if not profile or not token:
        return None
    return profile


# ============================================================
# login_view
# ============================================================
# Route: GET -> render trang login
#        POST -> validate input -> authenticate -> set session -> redirect
@require_http_methods(["GET", "POST"])
def login_view(request):
    # ---------------------------
    # GET: nếu đã login rồi thì chuyển sang profile, không cần login nữa
    # ---------------------------
    if request.method == "GET":
        if request.session.get("access_token"):
            return redirect("weather:profile")
        return render(request, "weather/auth/Login.html")

    # ---------------------------
    # POST: lấy dữ liệu từ form
    # identifier: có thể là username hoặc email (tùy service authenticate hỗ trợ)
    # ---------------------------
    identifier = request.POST.get("username", "").strip()
    password = request.POST.get("password", "")

    # Validate input cơ bản trước khi gọi service
    if not identifier:
        messages.error(request, "⚠️ Vui lòng nhập tên đăng nhập hoặc email.")
        return redirect("weather:login")
    
    if not password:
        messages.error(request, "⚠️ Vui lòng nhập mật khẩu.")
        return redirect("weather:login")

    try:
        # Gọi service authenticate: nếu đúng trả về dict manager/user
        manager = ManagerService.authenticate(identifier, password)

        # Tạo JWT access token (custom)
        # payload: manager_id + role
        token = create_access_token({
            "manager_id": manager["_id"],
            "role": manager.get("role", "guest"),
        })

        # Lưu token và profile vào session để duy trì login
        request.session["access_token"] = token
        request.session["profile"] = _make_json_safe(manager)

        # remember_me: nếu user tick => session sống 14 ngày
        # nếu không tick => session expire khi đóng trình duyệt
        remember_me = request.POST.get("remember_me")
        if remember_me:
            request.session.set_expiry(60 * 60 * 24 * 14)
        else:
            request.session.set_expiry(0)

        # Thông báo thành công
        messages.success(request, f"✅ Đăng nhập thành công! Chào mừng {manager.get('name', manager.get('userName'))}!")
        return redirect("weather:home")

    except Exception as e:
        # Bắt mọi lỗi authenticate và show message gọn
        error_msg = _extract_error_message(e)
        messages.error(request, f"❌ {error_msg}")
        return redirect("weather:login")


# ============================================================
# register_view
# ============================================================
# Flow:
# GET  -> render form register
# POST -> validate form -> validate email tồn tại -> check trùng username/email
#      -> check password strength
#      -> gửi OTP về email
#      -> lưu register_data vào session
#      -> redirect sang verify_email_register_view
@require_http_methods(["GET", "POST"])
def register_view(request):
    # GET: nếu đã login => không cho đăng ký, redirect profile
    if request.method == "GET":
        if request.session.get("access_token"):
            return redirect("weather:profile")
        return render(request, "weather/auth/Register.html")

    # ---------------------------
    # POST: lấy dữ liệu form
    # ---------------------------
    first_name = request.POST.get("first_name", "").strip()
    last_name = request.POST.get("last_name", "").strip()
    name = f"{first_name} {last_name}".strip()

    userName = request.POST.get("username", "").strip()

    # Normalize email: strip + lower (giảm case-sensitive và lỗi nhập)
    email = request.POST.get("email", "").strip().lower()

    password = request.POST.get("password", "")
    confirm_password = request.POST.get("confirm_password", "")

    # ---------------------------
    # Validate từng field và render lại form kèm form_data
    # form_data giúp template fill lại input user đã nhập (không phải nhập lại từ đầu)
    # ---------------------------
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

    # Check confirm password khớp
    if password != confirm_password:
        messages.error(request, "⚠️ Mật khẩu xác nhận không khớp. Vui lòng nhập lại.")
        return render(request, "weather/auth/Register.html", {
            "form_data": {"first_name": first_name, "last_name": last_name, "username": userName, "email": email}
        })

    # ---------------------------
    # Validate email tồn tại (theo logic EmailValidator)
    # - Có thể gọi API verify email hoặc rule riêng
    # - Nếu invalid: show errors list
    # ---------------------------
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

    # ---------------------------
    # Check trùng username/email trong DB
    # ---------------------------
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

    # ---------------------------
    # Check password strength theo rule của ManagerService
    # ---------------------------
    if not ManagerService.check_password_strength(password):
        errors = ManagerService.get_password_strength_errors(password)
        messages.error(request, "⚠️ Mật khẩu chưa đủ mạnh: " + ", ".join(errors))
        return render(request, "weather/auth/Register.html", {
            "form_data": {"first_name": first_name, "last_name": last_name, "username": userName, "email": email}
        })

    # ---------------------------
    # Gửi OTP xác thực email đăng ký
    # - Nếu gửi được: lưu dữ liệu đăng ký vào session và redirect tới verify OTP
    # ---------------------------
    try:
        EmailValidator.send_verification_otp(email, name)
        
        # Lưu data đăng ký vào session để bước verify OTP dùng lại
        # NOTE/TODO: Bạn đang lưu name/username/email/role... nhưng KHÔNG thấy lưu password.
        #            Trong verify_email_register_view phía dưới, bạn gọi:
        #            ManagerService.authenticate(register_data["userName"], register_data["password"])
        #            => Có nguy cơ KeyError vì register_data không có "password".
        #            (Chỉ comment để bạn debug, KHÔNG đổi code theo yêu cầu của bạn)
        request.session[SESSION_REGISTER_DATA] = {
            "name": name,
            "first_name": first_name,
            "last_name": last_name,
            "userName": userName,
            "email": email,
            "role": "staff",
        }

        # Cờ đánh dấu chưa verify
        request.session[SESSION_REGISTER_EMAIL_VERIFIED] = False
        
        messages.success(request, f"📧 Mã OTP đã được gửi đến {email}. Vui lòng kiểm tra hộp thư (bao gồm cả thư mục Spam) để xác thực.")
        return redirect("weather:verify_email_register")
        
    except EmailValidationError as e:
        # EmailValidationError: lỗi nghiệp vụ/email provider
        messages.error(request, f"❌ {str(e)}")
        return render(request, "weather/auth/Register.html", {
            "form_data": {"first_name": first_name, "last_name": last_name, "username": userName, "email": email}
        })
    except Exception as e:
        # Lỗi bất ngờ khi gửi mail
        messages.error(request, f"❌ Không thể gửi email xác thực. Vui lòng thử lại sau. Chi tiết: {str(e)}")
        return render(request, "weather/auth/Register.html", {
            "form_data": {"first_name": first_name, "last_name": last_name, "username": userName, "email": email}
        })


# ============================================================
# logout_view
# ============================================================
# - flush(): xóa toàn bộ session (access_token, profile, otp flags, ...)
# - Sau đó redirect về trang login
@require_http_methods(["GET"])
def logout_view(request):
    request.session.flush()
    messages.info(request, "Bạn đã đăng xuất.")
    return redirect("weather:login")


# ============================================================
# profile_view
# ============================================================
# GET:
# - lấy profile từ session -> wrap thành SessionUser -> render Profile.html
# POST:
# - update name/email trong DB (qua LoginRepository)
# - update lại session profile để UI đồng bộ
@require_http_methods(["GET", "POST"])
def profile_view(request):
    # Bắt buộc login: nếu không có session => redirect login
    profile = _require_session_login(request)
    if not profile:
        messages.warning(request, "Bạn cần đăng nhập.")
        return redirect("weather:login")

    # Convert dict -> object để template dùng dễ hơn
    user_obj = SessionUser(profile)
    
    # GET: render profile
    if request.method == "GET":
        return render(request, "weather/auth/Profile.html", {"user": user_obj, "profile": profile})
    
    # POST: lấy data update
    name = request.POST.get("name", "").strip()
    email = request.POST.get("email", "").strip().lower()
    
    if not name:
        messages.error(request, "Họ tên không được để trống.")
        return render(request, "weather/auth/Profile.html", {"user": user_obj, "profile": profile})
    
    try:
        # Repository để update DB
        from Weather_Forcast_App.Repositories.Login_repositories import LoginRepository
        from datetime import datetime, timezone
        
        # user_id lấy từ session profile
        user_id = profile.get("_id")
        
        # payload update: name + updatedAt
        update_data = {
            "name": name,
            "updatedAt": datetime.now(timezone.utc)
        }
        
        # Nếu email thay đổi -> check trùng với tài khoản khác
        old_email = profile.get("email", "")
        if email and email != old_email:
            existing = LoginRepository.find_by_username_or_email(email)
            if existing and str(existing.get("_id")) != str(user_id):
                messages.error(request, "Email đã được sử dụng bởi tài khoản khác.")
                return render(request, "weather/auth/Profile.html", {"user": user_obj, "profile": profile})
            update_data["email"] = email
        
        # Update DB
        LoginRepository.update_by_id(user_id, update_data)
        
        # Update lại session profile để UI reflect ngay
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


# ============================================================
# forgot_password_view (Reset bằng TOKEN/LINK)
# ============================================================
# Flow:
# POST email -> ManagerService.generate_token(email) -> build link -> (dev) print
# => user vào link => reset_password_view(token)
@require_http_methods(["GET", "POST"])
def forgot_password_view(request):
    if request.method == "GET":
        return render(request, "weather/auth/Forgot_password.html")

    # form đặt name="email" nhưng biến bạn đặt identifier cho linh hoạt
    identifier = request.POST.get("email", "").strip()
    try:
        # Service tạo token reset (có thể lưu DB/Redis, hoặc JWT tùy bạn implement)
        token = ManagerService.generate_token(identifier)

        # Tạo reset link tuyệt đối (domain + path)
        reset_link = request.build_absolute_uri(
            reverse("weather:reset_password", kwargs={"token": token})
        )

        # Debug dev mode: in link ra terminal
        print("========== RESET LINK (DEV) ==========")
        print(reset_link)
        print("======================================")

        # Lưu session để trang khác có thể hiển thị/trace (dev)
        request.session["last_reset_link"] = reset_link
        messages.success(request, "Nếu tài khoản tồn tại, link reset đã được tạo (dev: xem terminal).")
        return redirect("weather:password_reset_sent")

    except Exception as e:
        error_msg = _extract_error_message(e)
        messages.error(request, f"❌ {error_msg}")
        return redirect("weather:forgot_password")


# Trang thông báo "đã gửi" (hoặc đã tạo link)
@require_http_methods(["GET"])
def password_reset_sent_view(request):
    return render(request, "weather/auth/Password_reset_sent.html")


# ============================================================
# reset_password_view(token)
# ============================================================
# GET:
# - verify token -> render form reset (validlink True/False)
# POST:
# - validate new_password1/new_password2
# - gọi service reset_password_with_token
@require_http_methods(["GET", "POST"])
def reset_password_view(request, token: str):
    try:
        # verify token có hợp lệ không
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
        # Update password theo token
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


# ============================================================
# forgot_password_otp_view (Reset bằng OTP)
# ============================================================
# POST email -> ManagerService.send_reset_otp(email)
# - nếu email chưa tồn tại => báo lỗi
# - nếu gửi OTP ok => lưu SESSION_RESET_EMAIL và redirect verify_otp_view
@require_http_methods(["GET", "POST"])
def forgot_password_otp_view(request):
    if request.method == "GET":
        return render(request, "weather/auth/Forgot_password.html")

    email = request.POST.get("email", "").strip().lower()
    if not email:
        messages.error(request, "Vui lòng nhập email.")
        return redirect("weather:forgot_password_otp")

    try:
        result = ManagerService.send_reset_otp(email)
        
        # email_exists: flag service trả về cho biết email có tồn tại trong DB không
        if not result["email_exists"]:
            messages.error(request, "❌ Email này chưa được đăng ký trong hệ thống. Vui lòng kiểm tra lại hoặc đăng ký tài khoản mới.")
            return redirect("weather:forgot_password_otp")
        
        # success: gửi OTP ok hay fail
        if result["success"]:
            # Lưu email vào session để bước verify OTP dùng lại
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


# ============================================================
# verify_otp_view
# ============================================================
# GET: render trang nhập OTP
# POST: verify OTP -> set SESSION_RESET_OTP_OK + SESSION_RESET_OTP -> redirect reset_password_otp_view
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
        # Service verify OTP
        ManagerService.verify_reset_otp(email, otp)

        # Đánh dấu OTP đã OK + lưu OTP vào session để reset bước sau dùng
        request.session[SESSION_RESET_OTP_OK] = True
        request.session[SESSION_RESET_OTP] = otp
        messages.success(request, "✅ OTP hợp lệ. Bạn có thể đặt mật khẩu mới.")
        return redirect("weather:reset_password_otp")

    except Exception as e:
        error_msg = _extract_error_message(e)
        messages.error(request, f"❌ {error_msg}")
        return redirect("weather:verify_otp")


# ============================================================
# reset_password_otp_view
# ============================================================
# GET: render form nhập mật khẩu mới
# POST: validate -> ManagerService.reset_password_with_otp(email, otp, new_password)
#      -> clear session flags -> redirect login
@require_http_methods(["GET", "POST"])
def reset_password_otp_view(request):
    email = request.session.get(SESSION_RESET_EMAIL)
    otp_ok = request.session.get(SESSION_RESET_OTP_OK)
    otp = request.session.get(SESSION_RESET_OTP)

    # Nếu session thiếu bất kỳ phần nào => flow không hợp lệ => quay lại bước đầu
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
        # Reset password bằng OTP
        ManagerService.reset_password_with_otp(email, otp, new_password)

        # Xóa session của flow reset OTP để tránh reuse
        request.session.pop(SESSION_RESET_EMAIL, None)
        request.session.pop(SESSION_RESET_OTP_OK, None)
        request.session.pop(SESSION_RESET_OTP, None)

        messages.success(request, "✅ Đổi mật khẩu thành công! Hãy đăng nhập lại.")
        return redirect("weather:login")

    except Exception as e:
        error_msg = _extract_error_message(e)
        messages.error(request, f"❌ {error_msg}")
        return redirect("weather:reset_password_otp")


# ============================================================
# verify_email_register_view
# ============================================================
# Flow đăng ký + OTP email:
# - register_view gửi OTP và lưu SESSION_REGISTER_DATA
# - verify_email_register_view:
#    GET  -> render nhập OTP
#    POST -> verify OTP -> register_public(...) -> auto login -> redirect home
@require_http_methods(["GET", "POST"])
def verify_email_register_view(request):
    """
    Xác thực email OTP khi đăng ký tài khoản mới
    """
    register_data = request.session.get(SESSION_REGISTER_DATA)
    
    # Nếu không có register_data => user chưa qua bước register hoặc session hết hạn
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
        # Verify OTP email đăng ký
        EmailValidator.verify_email_otp(email, otp)
        
        # Đánh dấu email verified
        request.session[SESSION_REGISTER_EMAIL_VERIFIED] = True
        
        # Tạo tài khoản chính thức trong DB
        # skip_email_verification=True vì bạn vừa verify OTP xong
        ManagerService.register_public(register_data, skip_email_verification=True)
        
        # Sau khi tạo account, bạn thử auto-login:
        try:
            # NOTE/TODO: chỗ này dùng register_data["password"] có thể lỗi nếu register_data không chứa password
            manager = ManagerService.authenticate(register_data["userName"], register_data["password"])
            token = create_access_token({
                "manager_id": manager["_id"],
                "role": manager.get("role", "guest"),
            })
            request.session["access_token"] = token
            request.session["profile"] = _make_json_safe(manager)
            
            # Xóa session tạm của register flow
            request.session.pop(SESSION_REGISTER_DATA, None)
            request.session.pop(SESSION_REGISTER_EMAIL_VERIFIED, None)
            
            messages.success(request, f"🎉 Chào mừng {register_data.get('name', '')}! Tài khoản đã được tạo thành công.")
            return redirect("weather:home")
        except Exception as login_err:
            # Nếu auto-login fail => vẫn coi như tạo thành công, yêu cầu user login thủ công
            request.session.pop(SESSION_REGISTER_DATA, None)
            request.session.pop(SESSION_REGISTER_EMAIL_VERIFIED, None)
            
            messages.success(request, "🎉 Tạo tài khoản thành công! Hãy đăng nhập để sử dụng.")
            return redirect("weather:login")
            
    except EmailValidationError as e:
        # OTP sai/hết hạn/invalid theo logic EmailValidator
        messages.error(request, f"❌ {str(e)}")
        return redirect("weather:verify_email_register")
    except Exception as e:
        error_msg = _extract_error_message(e)
        messages.error(request, f"❌ {error_msg}")
        return redirect("weather:verify_email_register")


# ============================================================
# resend_email_otp_view
# ============================================================
# - Gửi lại OTP mới cho email đăng ký (lấy từ SESSION_REGISTER_DATA)
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


# ============================================================
# cancel_register_view
# ============================================================
# - Hủy đăng ký: xóa session register flow và quay lại register page
@require_http_methods(["GET"])
def cancel_register_view(request):
    """
    Hủy quá trình đăng ký và xóa session
    """
    request.session.pop(SESSION_REGISTER_DATA, None)
    request.session.pop(SESSION_REGISTER_EMAIL_VERIFIED, None)
    messages.info(request, "Đã hủy quá trình đăng ký.")
    return redirect("weather:register")
