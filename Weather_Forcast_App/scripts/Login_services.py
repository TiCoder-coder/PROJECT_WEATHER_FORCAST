import jwt
from datetime import datetime, timezone, timedelta
from django.contrib.auth.hashers import make_password, check_password
from django.core.signing import TimestampSigner, BadSignature, SignatureExpired
from rest_framework.exceptions import ValidationError, PermissionDenied
from decouple import config
from Weather_Forcast_App.Repositories.Login_repositories import LoginRepository
from bson import ObjectId
from Weather_Forcast_App.middleware.Auth import require_manager, require_manager_or_admin
import random, secrets, hashlib
from django.conf import settings
from pymongo import ASCENDING, DESCENDING
from Weather_Forcast_App.db_connection import get_database, create_index_safe, transaction

def convert_objectid(obj):

    if isinstance(obj, list):
        return [convert_objectid(o) for o in obj]
    

    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            if isinstance(v, ObjectId):
                new_obj[k] = str(v)
            else:
                new_obj[k] = convert_objectid(v)
        return new_obj
    return obj


MAX_FAILED_ATTEMPTS = int(config("MAX_FAILED_ATTEMPS", default=5))
PASSWORD_PEPPER = config("PASSWORD_PEPPER", default=None)
RESET_TOKEN_SALT = config("RESET_TOKEN_SALT", default="reset_secret")
RESET_TOKEN_EXPIRY_SECONDS = int(config("RESET_TOKEN_EXPIRY_SECONDS", default=3600))
RESET_OTP_RESEND_MIN_SECONDS = int(config("RESET_OTP_RESEND_MIN_SECONDS", default=60))
RESET_OTP_MAX_PER_HOUR = int(config("RESET_OTP_MAX_PER_HOUR", default=5))

signer = TimestampSigner(key=config("SECRET_KEY"))
db = get_database()

managers = db["logins"]
password_reset_otps = db["password_reset_otps"]

create_index_safe(password_reset_otps, "expiresAt", expireAfterSeconds=0)
create_index_safe(password_reset_otps, [("email", ASCENDING), ("createdAt", DESCENDING)])

def _apply_pepper(raw_password: str) -> str:
    return raw_password + PASSWORD_PEPPER if PASSWORD_PEPPER else raw_password

class ManagerService:

    @staticmethod
    def check_password_strength(password: str) -> bool:
        """
        Kiểm tra mật khẩu có đủ mạnh không:
        - Tối thiểu 8 ký tự
        - Có ít nhất 1 chữ thường
        - Có ít nhất 1 chữ hoa  
        - Có ít nhất 1 số
        - Có ít nhất 1 ký tự đặc biệt (!@#$%^&*()-_+=)
        """
        if len(password) < 8:
            return False
        has_lower = any(c.islower() for c in password)                                       # Phai co ki tu thuong
        has_upper = any(c.isupper() for c in password)                                       # Phai co ki tu in hoa
        has_digit = any(c.isdigit() for c in password)                                       # Phai co so
        has_special = any(c in "!@#$%^&*()-_+=" for c in password)                           # Phai co ki tu dac biet
        return has_lower and has_upper and has_digit and has_special
    
    @staticmethod
    def get_password_strength_errors(password: str) -> list:
        """Trả về danh sách các yêu cầu chưa đạt"""
        errors = []
        if len(password) < 8:
            errors.append("Tối thiểu 8 ký tự")
        if not any(c.islower() for c in password):
            errors.append("Cần có chữ thường (a-z)")
        if not any(c.isupper() for c in password):
            errors.append("Cần có chữ in hoa (A-Z)")
        if not any(c.isdigit() for c in password):
            errors.append("Cần có chữ số (0-9)")
        if not any(c in "!@#$%^&*()-_+=" for c in password):
            errors.append("Cần có ký tự đặc biệt (!@#$%^&*()-_+=)")
        return errors


    @staticmethod
    def _increase_failed_attempt(manager):
        
        failed = manager.get("failed_attempts", 0) + 1
        
        lock_until = None
        
        if failed >= MAX_FAILED_ATTEMPTS:
            lock_until = datetime.now(timezone.utc) + timedelta(minutes=5)                  # Thoi gian locj la thoi gian hien tai + 5 phut
        
        LoginRepository.update_by_id(manager["_id"], {
            "failed_attempts": failed,
            "lock_until": lock_until
        })

    @staticmethod
    def generate_token(identifier: str) -> str:
        
        manager = LoginRepository.find_by_username_or_email(identifier)
        if not manager:
            raise ValidationError("Manager not found for token generation")
        
        payload = f"manager:{manager['_id']}"
        signed = signer.sign(payload + "|" + RESET_TOKEN_SALT)
        return signed
    
    @staticmethod
    def reset_password_with_token(token: str, new_password: str):
        
        try:
            
            signed_value = signer.unsign(token, max_age=RESET_TOKEN_EXPIRY_SECONDS)
        except (SignatureExpired, BadSignature):
            raise ValidationError("Token invalid or expired")

        payload, salt = signed_value.split("|", 1)
        
        if salt != RESET_TOKEN_SALT:
            raise ValidationError("Invalid token salt")

        manager_id = payload.split(":")[1]
        
        manager = LoginRepository.find_by_id(manager_id)
        if not manager:
            raise ValidationError("Manager not found")

        if not ManagerService.check_password_strength(new_password):
            raise ValidationError("Weak new password")

        hashed_pw = make_password(_apply_pepper(new_password))
        LoginRepository.update_by_id(manager_id, {
            "password": hashed_pw,
            "failed_attempts": 0,
            "lock_until": None,
            "updatedAt": datetime.now(timezone.utc)
        })
        return {"Reset": True}

    @staticmethod
    def create_manager(user, name, userName, password, email, role="staff"):
        with transaction() as session:
            require_manager_or_admin(user)

            if LoginRepository.find_by_username(userName):
                raise ValidationError("Username already exists")
            if LoginRepository.find_by_username_or_email(email):
                raise ValidationError("Email already exists")
            if not ManagerService.check_password_strength(password):
                raise ValidationError("Weak password")

            hashed_pw = make_password(_apply_pepper(password))
            normalized_role = str(role).capitalize() if role.lower() in ["manager", "staff", "admin"] else role

            data = {
                "name": name,
                "userName": userName,
                "password": hashed_pw,
                "email": email,
                "role": normalized_role,
                "is_active": True,
                "failed_attempts": 0,
                "lock_until": None,
                "createdAt": datetime.now(timezone.utc),
                "updatedAt": datetime.now(timezone.utc)
            }
            LoginRepository.insert_one(data, session=session)
            return {"Created": True}

    @staticmethod
    def review_all_managers(user):
        require_manager_or_admin(user)

        managers = LoginRepository.find_all()
        for m in managers:
            m["_id"] = str(m["_id"])
            m.pop("password", None)
        return managers

    @staticmethod
    def review_manager_by_id(user, idManager):
        require_manager_or_admin(user)

        manager = LoginRepository.find_by_id(idManager)
        if not manager:
            raise ValidationError(f"No manager found with id: {idManager}")
        manager["_id"] = str(manager["_id"])
        manager.pop("password", None)
        return manager


    @staticmethod
    def update_manager(user, idManager, validated_data):
        with transaction() as session:
            require_manager_or_admin(user)

            update_fields = {
                key: value for key, value in validated_data.items()
                if key in ["name", "email", "role", "is_active"]
            }
            update_fields["updatedAt"] = datetime.now(timezone.utc)
            result = LoginRepository.update_by_id(idManager, update_fields, session=session)
            if result.matched_count == 0:
                raise ValidationError(f"No manager found with id: {idManager}")
            return {"Updated": True}

    @staticmethod
    def delete_manager(user, idManager):
        with transaction() as session:
            require_manager_or_admin(user)

            result = LoginRepository.delete_by_id(idManager, session=session)
            if result.deleted_count == 0:
                raise ValidationError(f"No manager found with id: {idManager}")
            return {"Deleted": True}

    @staticmethod
    def authenticate(identifier: str, password: str):
        """
        Cho phép đăng nhập bằng username HOẶC email.
        identifier: có thể là username hoặc email
        """
        manager = LoginRepository.find_by_username_or_email(identifier)
        if not manager:
            raise ValidationError("Tên đăng nhập hoặc email không tồn tại")

        if manager.get("lock_until") and manager["lock_until"] > datetime.now(timezone.utc):
            remaining = (manager["lock_until"] - datetime.now(timezone.utc)).seconds // 60
            raise PermissionDenied(f"Tài khoản đang bị khóa. Vui lòng thử lại sau {remaining} phút.")

        if not manager.get("is_active", True):
            raise PermissionDenied("Tài khoản đã bị vô hiệu hóa. Vui lòng liên hệ admin.")
        
        if not check_password(_apply_pepper(password), manager["password"]):
            ManagerService._increase_failed_attempt(manager)
            remaining_attempts = MAX_FAILED_ATTEMPTS - manager.get("failed_attempts", 0) - 1
            if remaining_attempts > 0:
                raise PermissionDenied(f"Mật khẩu không đúng. Còn {remaining_attempts} lần thử.")
            else:
                raise PermissionDenied("Mật khẩu không đúng. Tài khoản sẽ bị khóa tạm thời.")
        
        LoginRepository.update_by_id(manager["_id"], {
            "failed_attempts": 0,
            "last_login": datetime.now(timezone.utc),
            "lock_until": None, 
            "updatedAt": datetime.now(timezone.utc) 
        })

        manager["_id"] = str(manager["_id"])
        
        manager.pop("password", None)
        for k in ("createdAt", "updatedAt", "last_login", "lock_until"):
            v = manager.get(k)
            if isinstance(v, datetime):
                manager[k] = v.isoformat()

        return manager

    @staticmethod
    def verify_reset_token(token: str) -> dict:
        try:
            signed_value = signer.unsign(token, max_age=RESET_TOKEN_EXPIRY_SECONDS)

            payload, salt = signed_value.split("|", 1)
            if salt != RESET_TOKEN_SALT:
                raise Exception("Invalid token salt")

            if not payload.startswith("manager:"):
                raise Exception("Invalid token payload")

            manager_id = payload.split(":", 1)[1]
            return {"manager_id": manager_id}

        except SignatureExpired:
            raise Exception("Link reset đã hết hạn.")
        except BadSignature:
            raise Exception("Link reset không hợp lệ.")


    @staticmethod
    def register_public(data: dict, skip_email_verification: bool = False):
        """
        Cho phép đăng ký từ web (public). Nếu bạn muốn chặn public,
        thì xoá endpoint register hoặc yêu cầu admin tạo tài khoản.
        
        Args:
            data: Dữ liệu đăng ký
            skip_email_verification: Bỏ qua xác thực email (mặc định False)
        """
        import re
        from Weather_Forcast_App.scripts.Email_validator import EmailValidator, EmailValidationError
        
        name = (data.get("name") or "").strip()
        userName = (data.get("userName") or "").strip()
        password = data.get("password") or ""
        email = (data.get("email") or "").strip().lower()  # Normalize email
        role = (data.get("role") or "staff").strip()

        if not name or not userName or not password or not email:
            raise Exception("Thiếu thông tin đăng ký (name, username, email, password).")

        if len(userName) < 3 or len(userName) > 30:
            raise Exception("Tên đăng nhập phải từ 3-30 ký tự.")
        if not re.match(r'^[a-zA-Z0-9_]+$', userName):
            raise Exception("Tên đăng nhập chỉ được chứa chữ cái, số và dấu gạch dưới (_).")

        email_validation = EmailValidator.validate_email_exists(email)
        if not email_validation['valid']:
            raise Exception(', '.join(email_validation['errors']))
        
        if not skip_email_verification:
            if not EmailValidator.is_email_verified(email, within_seconds=3600):
                raise Exception("Email chưa được xác thực. Vui lòng xác thực email trước khi đăng ký.")

        if not ManagerService.check_password_strength(password):
            errors = ManagerService.get_password_strength_errors(password)
            raise Exception("Mật khẩu yếu: " + ", ".join(errors))

        if LoginRepository.find_by_username(userName):
            raise Exception("Tên đăng nhập đã tồn tại.")
        if LoginRepository.find_by_username_or_email(email):
            raise Exception("Email đã được sử dụng.")

        hashed_password = make_password(_apply_pepper(password))

        doc = {
            "name": name,
            "userName": userName,
            "password": hashed_password,
            "email": email,
            "role": role,
            "is_active": True,
            "failed_attempts": 0,
            "lock_until": None,
            "last_login": None,
            "createdAt": datetime.now(timezone.utc),
            "updatedAt": datetime.now(timezone.utc),
        }

        LoginRepository.insert_one(doc)
        return True
    
    @staticmethod
    def _hash_otp(otp: str, salt: str) -> str:
        raw = f"{otp}:{salt}:{settings.SECRET_KEY}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def send_reset_otp(email: str, check_exists: bool = True) -> dict:
        from Weather_Forcast_App.scripts.email_templates import generate_otp, send_otp_email

        email = (email or "").strip().lower()
        if not email:
            raise ValueError("Vui lòng nhập email.")

        now = datetime.now(timezone.utc)

        user = managers.find_one({"email": email, "is_active": True})
        if not user:
            return {"success": False, "email_exists": False, "message": "Email chưa được đăng ký trong hệ thống."}

        recent = password_reset_otps.find_one(
            {"email": email, "createdAt": {"$gte": now - timedelta(seconds=RESET_OTP_RESEND_MIN_SECONDS)}},
            sort=[("createdAt", DESCENDING)]
        )
        if recent:
            waited = int((now - recent["createdAt"]).total_seconds())
            remain = max(1, RESET_OTP_RESEND_MIN_SECONDS - waited)
            return {"success": False, "email_exists": True, "message": f"Bạn gửi OTP quá nhanh. Vui lòng thử lại sau {remain} giây."}

        count_1h = password_reset_otps.count_documents(
            {"email": email, "createdAt": {"$gte": now - timedelta(hours=1)}}
        )
        if count_1h >= RESET_OTP_MAX_PER_HOUR:
            return {"success": False, "email_exists": True, "message": "Bạn đã yêu cầu OTP quá nhiều lần trong 1 giờ. Vui lòng thử lại sau."}

        otp = generate_otp()
        salt = secrets.token_hex(8)
        expires_at = now + timedelta(seconds=settings.PASSWORD_RESET_OTP_EXPIRE_SECONDS)

        otp_doc = {
            "email": email,
            "otpHash": ManagerService._hash_otp(otp, salt),
            "salt": salt,
            "attempts": 0,
            "used": False,
            "createdAt": now,
            "expiresAt": expires_at,
        }

        inserted_id = None
        with transaction() as session:
            password_reset_otps.update_many(
                {"email": email, "used": False},
                {"$set": {"used": True}},
                session=session
            )
            res = password_reset_otps.insert_one(otp_doc, session=session)
            inserted_id = res.inserted_id

        user_name = user.get("name", user.get("userName", ""))
        try:
            send_otp_email(
                email=email,
                name=user_name,
                otp=otp,
                purpose="đặt lại mật khẩu",
                expire_minutes=settings.PASSWORD_RESET_OTP_EXPIRE_SECONDS // 60
            )
        except Exception:
            password_reset_otps.update_one(
                {"_id": inserted_id},
                {"$set": {"used": True, "usedAt": datetime.now(timezone.utc)}}
            )
            return {"success": False, "email_exists": True, "message": "Gửi email OTP thất bại. Vui lòng thử lại."}

        return {"success": True, "email_exists": True, "message": f"Mã OTP đã được gửi đến {email}"}



    @staticmethod
    def verify_reset_otp(email: str, otp: str) -> None:
        email = (email or "").strip().lower()
        otp = (otp or "").strip()
        if not email or not otp:
            raise ValueError("Thiếu email hoặc OTP.")

        now = datetime.now(timezone.utc)

        rec = password_reset_otps.find({"email": email, "used": False, "expiresAt": {"$gt": now}})\
                                .sort("createdAt", -1)\
                                .limit(1)
        rec = next(rec, None)
        if not rec:
            raise ValueError("OTP không hợp lệ hoặc đã hết hạn.")

        if rec["attempts"] >= settings.PASSWORD_RESET_OTP_MAX_ATTEMPTS:
            raise ValueError("Bạn đã nhập sai quá nhiều lần. Hãy yêu cầu OTP mới.")

        expected = ManagerService._hash_otp(otp, rec["salt"])
        if expected != rec["otpHash"]:
            password_reset_otps.update_one({"_id": rec["_id"]}, {"$inc": {"attempts": 1}})
            raise ValueError("OTP không đúng.")

        password_reset_otps.update_one({"_id": rec["_id"]}, {"$set": {"verifiedAt": now}})

    @staticmethod
    def reset_password_with_otp(email: str, otp: str, new_password: str) -> None:
        email = (email or "").strip().lower()
        otp = (otp or "").strip()

        if not ManagerService.check_password_strength(new_password):
            raise ValueError("Mật khẩu yếu (>=8, có hoa/thường/số/ký tự đặc biệt).")

        now = datetime.now(timezone.utc)
        hashed = make_password(_apply_pepper(new_password))

        with transaction() as session:
            rec = password_reset_otps.find_one(
                {"email": email, "used": False, "expiresAt": {"$gt": now}},
                sort=[("createdAt", DESCENDING)],
                session=session
            )
            if not rec:
                raise ValueError("OTP không hợp lệ hoặc đã hết hạn.")

            if rec.get("attempts", 0) >= settings.PASSWORD_RESET_OTP_MAX_ATTEMPTS:
                raise ValueError("Bạn đã nhập sai quá nhiều lần. Hãy yêu cầu OTP mới.")

            expected = ManagerService._hash_otp(otp, rec["salt"])
            if expected != rec["otpHash"]:
                password_reset_otps.update_one(
                    {"_id": rec["_id"]},
                    {"$inc": {"attempts": 1}},
                    session=session
                )
                raise ValueError("OTP không đúng.")

            user = managers.find_one({"email": email, "is_active": True}, session=session)
            if not user:
                raise ValueError("Không tìm thấy tài khoản.")

            managers.update_one(
                {"_id": user["_id"]},
                {"$set": {"password": hashed, "updatedAt": now}},
                session=session
            )

            password_reset_otps.update_many(
                {"email": email, "used": False},
                {"$set": {"used": True, "usedAt": now}},
                session=session
            )


