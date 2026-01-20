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
from django.core.mail import send_mail
from django.conf import settings
from pymongo import MongoClient, ASCENDING, DESCENDING
# Dinh nghia mot ham dung de convert cac object -> str
def convert_objectid(obj):
    # Neu la list: duyet qua cac phan tu torng list va chuyen doi object -> str
    if isinstance(obj, list):
        return [convert_objectid(o) for o in obj]
    
    # Neu la dict: duyet qua cac phan tu va xu li cho tung key, value
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            if isinstance(v, ObjectId):
                new_obj[k] = str(v)
            else:
                new_obj[k] = convert_objectid(v)
        return new_obj
    return obj


# SECURITY CONFIG
MAX_FAILED_ATTEMPTS = int(config("MAX_FAILED_ATTEMPS", default=5))
PASSWORD_PEPPER = config("PASSWORD_PEPPER", default=None)
RESET_TOKEN_SALT = config("RESET_TOKEN_SALT", default="reset_secret")
RESET_TOKEN_EXPIRY_SECONDS = int(config("RESET_TOKEN_EXPIRY_SECONDS", default=3600))
# signer = TimestampSigner(config("SECRET_KEY"))
signer = TimestampSigner(key=config("SECRET_KEY"))
client = MongoClient(config("MONGO_URI"))
db = client[config("DB_NAME")]

managers = db["logins"]  # trùng với LoginRepository đang dùng
password_reset_otps = db["password_reset_otps"]

password_reset_otps.create_index("expiresAt", expireAfterSeconds=0)
password_reset_otps.create_index([("email", ASCENDING), ("createdAt", DESCENDING)])

def _apply_pepper(raw_password: str) -> str:
    return raw_password + PASSWORD_PEPPER if PASSWORD_PEPPER else raw_password


# --------------------------------------------- CLASS KHONG TRUC TIEP TUONG TAC VOI DATABASE MA SE THONG QUA REPOSITORY DE TUONG TAC ----------------------------------------------------------------------- 
# ------------------------------------------------------------------VA PHAN QUYEN SU DUNG CHO STAFF HAY MANAGER VA ADMIN------------------------------------------------------------------------------------    

class ManagerService:
    

    # DINH NGHIA MOT HAM DUNG DE KIEM TRA XEM PASSWORD TAO RA CO MANH KHONG---------------------------------------------------------------------------------------------------------------------------------
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


    # DINH NGHI MOT HAM DUNG DE KIEM TRA SO LAN DANG NHAP THAT BAI --- NEU LON HON SO LAN CHO PHEP THI ---> KHOA TAI KHOAN TRONG 1 KHOANG THOI GIAN --------------------------------------------------------
    @staticmethod
    def _increase_failed_attempt(manager):
        
        # Sau moi lan loi thi cong them 1
        failed = manager.get("failed_attempts", 0) + 1
        
        # Chua lock neu chua vuot qua so lan quy dinh
        lock_until = None
        
        # Lock
        if failed >= MAX_FAILED_ATTEMPTS:
            lock_until = datetime.now(timezone.utc) + timedelta(minutes=5)                  # Thoi gian locj la thoi gian hien tai + 5 phut
        
        # Cap nhap xupng database manager bi loi bao nhieu lan va lock den khi nao
        LoginRepository.update_by_id(manager["_id"], {
            "failed_attempts": failed,
            "lock_until": lock_until
        })

    # MA HOA TOKEN TRUOC KHI DUA RA ---- TOKEN NAY SE DUOC NOI VOI MOT CHUOI KHAC --- NEU DE CHO NGUOI KHAC CO TOKEN THI CUNG KHONG THE TRUY CAP VI KHONG CO CHUOI SALT
    @staticmethod
    def generate_token(identifier: str) -> str:
        
        # Tim kiem thong tin manager
        manager = LoginRepository.find_by_username_or_email(identifier)
        if not manager:
            raise ValidationError("Manager not found for token generation")
        
        # Neu tim thay manager thi se dua thong tin do vao token
        payload = f"manager:{manager['_id']}"
        signed = signer.sign(payload + "|" + RESET_TOKEN_SALT)
        return signed
    
    # HAM DUNG DE RESET PASSWORD KHI CAN--------------------------------------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def reset_password_with_token(token: str, new_password: str):
        
        # Giai ma token
        try:
            
            # Giai ma token va kiem tra xem token con han su dung hay khong
            signed_value = signer.unsign(token, max_age=RESET_TOKEN_EXPIRY_SECONDS)
        except (SignatureExpired, BadSignature):
            raise ValidationError("Token invalid or expired")

        # Tach paylooad va chuoi salt ra dung de kiem tra xem co dung thong tin khong
        payload, salt = signed_value.split("|", 1)
        
        # Neu khac chuoi salt da duoc dinh nghia trong .env thi bao loi
        if salt != RESET_TOKEN_SALT:
            raise ValidationError("Invalid token salt")

        # Kiem tra id nguoi dung
        manager_id = payload.split(":")[1]
        
        # Kiem tra xem id do co phai la cua manager hay khong
        manager = LoginRepository.find_by_id(manager_id)
        if not manager:
            raise ValidationError("Manager not found")

        # Neu dung thi kiem tra xem mat khau muon dat lai co dung chuan hay khong
        if not ManagerService.check_password_strength(new_password):
            raise ValidationError("Weak new password")

        # Neu ok thi lay password moi dem di hash va cap nhap
        hashed_pw = make_password(_apply_pepper(new_password))
        LoginRepository.update_by_id(manager_id, {
            "password": hashed_pw,
            "failed_attempts": 0,
            "lock_until": None,
            "updatedAt": datetime.now(timezone.utc)
        })
        return {"Reset": True}

    # CREATE MANAGER -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def create_manager(user, name, userName, password, email, role="staff"):
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
        LoginRepository.insert_one(data)
        return {"Created": True}

    
    # REVIEW ALL MANAGER --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def review_all_managers(user):
        require_manager_or_admin(user)

        managers = LoginRepository.find_all()
        for m in managers:
            m["_id"] = str(m["_id"])
            m.pop("password", None)
        return managers

    
    # REVIEW MANAGER BY ID ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def review_manager_by_id(user, idManager):
        require_manager_or_admin(user)

        manager = LoginRepository.find_by_id(idManager)
        if not manager:
            raise ValidationError(f"No manager found with id: {idManager}")
        manager["_id"] = str(manager["_id"])
        manager.pop("password", None)
        return manager


    # UPDATE MANAGER ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def update_manager(user, idManager, validated_data):
        require_manager_or_admin(user)

        update_fields = {
            key: value for key, value in validated_data.items()
            if key in ["name", "email", "role", "is_active"]
        }
        update_fields["updatedAt"] = datetime.now(timezone.utc)
        result = LoginRepository.update_by_id(idManager, update_fields)
        if result.matched_count == 0:
            raise ValidationError(f"No manager found with id: {idManager}")
        return {"Updated": True}


    # DELETE MANAGER -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def delete_manager(user, idManager):
        require_manager_or_admin(user)

        result = LoginRepository.delete_by_id(idManager)
        if result.deleted_count == 0:
            raise ValidationError(f"No manager found with id: {idManager}")
        return {"Deleted": True}

    # AUTHENTICATE: DUNG DE KIEM TRA DANH NHAP ----------------------------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def authenticate(identifier: str, password: str):
        """
        Cho phép đăng nhập bằng username HOẶC email.
        identifier: có thể là username hoặc email
        """
        # Kiem tra manager co ton tai khong (tim theo username hoac email)
        manager = LoginRepository.find_by_username_or_email(identifier)
        if not manager:
            raise ValidationError("Tên đăng nhập hoặc email không tồn tại")

        # Kiem tra tai khoan manager do co bi khoa tai khoan khong
        if manager.get("lock_until") and manager["lock_until"] > datetime.now(timezone.utc):
            remaining = (manager["lock_until"] - datetime.now(timezone.utc)).seconds // 60
            raise PermissionDenied(f"Tài khoản đang bị khóa. Vui lòng thử lại sau {remaining} phút.")

        # Kiem tra tai khoan co active khong
        if not manager.get("is_active", True):
            raise PermissionDenied("Tài khoản đã bị vô hiệu hóa. Vui lòng liên hệ admin.")
        
        # Kiem tra xem co dung password khong
        if not check_password(_apply_pepper(password), manager["password"]):        # Vi password da duoc hash nen phai ket hop voi _apply_peper moi kiem tra duoc
            ManagerService._increase_failed_attempt(manager)
            remaining_attempts = MAX_FAILED_ATTEMPTS - manager.get("failed_attempts", 0) - 1
            if remaining_attempts > 0:
                raise PermissionDenied(f"Mật khẩu không đúng. Còn {remaining_attempts} lần thử.")
            else:
                raise PermissionDenied("Mật khẩu không đúng. Tài khoản sẽ bị khóa tạm thời.")
        
        # Neu dang nhap thanh cong thi cap nhap lai cac thong tin
        LoginRepository.update_by_id(manager["_id"], {
            "failed_attempts": 0,                                                   # Cap nhap lai so lan that bai la o
            "last_login": datetime.now(timezone.utc),                               # Cap nhap lai lan dang nhap cuoi cung
            "lock_until": None,                                                     # Cap nhap lai khong lock
            "updatedAt": datetime.now(timezone.utc)                                 # Cap nhap lai ngay update
        })

        # Chuyen doi id sang string va tra ve thong tin
        manager["_id"] = str(manager["_id"])
        
        manager.pop("password", None)                                               # Xoa di mat khau truoc khi in ra de bao mat
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
    def register_public(data: dict):
        """
        Cho phép đăng ký từ web (public). Nếu bạn muốn chặn public,
        thì xoá endpoint register hoặc yêu cầu admin tạo tài khoản.
        """
        import re
        
        name = (data.get("name") or "").strip()
        userName = (data.get("userName") or "").strip()
        password = data.get("password") or ""
        email = (data.get("email") or "").strip().lower()  # Normalize email
        role = (data.get("role") or "staff").strip()

        # Validate required fields
        if not name or not userName or not password or not email:
            raise Exception("Thiếu thông tin đăng ký (name, username, email, password).")

        # Validate username format (3-30 ký tự, chỉ chữ, số, underscore)
        if len(userName) < 3 or len(userName) > 30:
            raise Exception("Tên đăng nhập phải từ 3-30 ký tự.")
        if not re.match(r'^[a-zA-Z0-9_]+$', userName):
            raise Exception("Tên đăng nhập chỉ được chứa chữ cái, số và dấu gạch dưới (_).")

        # Validate email format
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            raise Exception("Email không đúng định dạng.")

        # Validate password strength
        if not ManagerService.check_password_strength(password):
            errors = ManagerService.get_password_strength_errors(password)
            raise Exception("Mật khẩu yếu: " + ", ".join(errors))

        # Check duplicates
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
    def send_reset_otp(email: str) -> None:
        email = (email or "").strip().lower()
        if not email:
            raise ValueError("Vui lòng nhập email.")

        user = managers.find_one({"email": email, "is_active": True})
        if not user:
            return  # không leak thông tin

        # vô hiệu hoá OTP cũ (optional nhưng nên)
        password_reset_otps.update_many({"email": email, "used": False}, {"$set": {"used": True}})

        otp = f"{random.randint(0, 999999):06d}"
        salt = secrets.token_hex(8)

        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=settings.PASSWORD_RESET_OTP_EXPIRE_SECONDS)

        password_reset_otps.insert_one({
            "email": email,
            "otpHash": ManagerService._hash_otp(otp, salt),
            "salt": salt,
            "attempts": 0,
            "used": False,
            "createdAt": now,
            "expiresAt": expires_at,
        })

        subject = "VN Weather Hub - Mã OTP đặt lại mật khẩu"
        message = (
            f"Mã OTP của bạn là: {otp}\n\n"
            f"Mã có hiệu lực trong {settings.PASSWORD_RESET_OTP_EXPIRE_SECONDS // 60} phút."
        )
        send_mail(subject, message, settings.DEFAULT_FROM_EMAIL, [email], fail_silently=False)


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
        if not ManagerService.check_password_strength(new_password):
            raise ValueError("Mật khẩu yếu (>=8, có hoa/thường/số/ký tự đặc biệt).")

        ManagerService.verify_reset_otp(email, otp)

        user = managers.find_one({"email": email, "is_active": True})
        if not user:
            raise ValueError("Không tìm thấy tài khoản.")

        hashed = make_password(_apply_pepper(new_password))
        now = datetime.now(timezone.utc)

        managers.update_one({"_id": user["_id"]}, {"$set": {"password": hashed, "updatedAt": now}})

        password_reset_otps.update_many({"email": email, "used": False}, {"$set": {"used": True, "usedAt": now}})


