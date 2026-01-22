"""
Email Validator Service
Xác thực email có tồn tại thực sự hay không thông qua:
1. Kiểm tra cú pháp email
2. Kiểm tra DNS MX records
3. Gửi OTP xác thực qua email
"""

import re
import dns.resolver
import hashlib
import secrets
import random
from datetime import datetime, timezone, timedelta
from django.conf import settings
from pymongo import ASCENDING, DESCENDING
from decouple import config
from Weather_Forcast_App.db_connection import get_database, create_index_safe

# MongoDB connection
db = get_database()

# Collection lưu OTP xác thực email đăng ký
email_verification_otps = db["email_verification_otps"]

# Tạo index tự động xóa OTP hết hạn
create_index_safe(email_verification_otps, "expiresAt", expireAfterSeconds=0)
create_index_safe(email_verification_otps, [("email", ASCENDING), ("createdAt", DESCENDING)])

# Config
EMAIL_OTP_EXPIRE_SECONDS = int(config("PASSWORD_RESET_OTP_EXPIRE_SECONDS", default=600))  # 10 phút
EMAIL_OTP_MAX_ATTEMPTS = int(config("PASSWORD_RESET_OTP_MAX_ATTEMPTS", default=5))


class EmailValidationError(Exception):
    """Custom exception cho lỗi xác thực email"""
    pass


class EmailValidator:
    """
    Service xác thực email bao gồm:
    - Kiểm tra cú pháp
    - Kiểm tra DNS MX records
    - Gửi OTP xác thực
    """
    
    # Danh sách domain email phổ biến (whitelist cho phép bỏ qua MX check)
    TRUSTED_DOMAINS = {
        'gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 
        'icloud.com', 'protonmail.com', 'mail.com', 'zoho.com',
        'yandex.com', 'aol.com', 'live.com', 'msn.com',
        'edu.vn', 'fpt.edu.vn', 'hust.edu.vn', 'vnu.edu.vn'
    }
    
    # Danh sách domain email tạm thời/disposable (blacklist)
    DISPOSABLE_DOMAINS = {
        'tempmail.com', 'guerrillamail.com', 'mailinator.com',
        '10minutemail.com', 'throwaway.email', 'temp-mail.org',
        'fakeinbox.com', 'trashmail.com', 'maildrop.cc',
        'yopmail.com', 'sharklasers.com', 'getairmail.com',
        'dispostable.com', 'mailnesia.com', 'emailondeck.com'
    }

    @staticmethod
    def validate_email_format(email: str) -> dict:
        """
        Kiểm tra cú pháp email hợp lệ
        Trả về dict với thông tin chi tiết về lỗi
        """
        result = {
            'valid': False,
            'error': None
        }
        
        if not email:
            result['error'] = 'Vui lòng nhập địa chỉ email'
            return result
        
        email = email.strip().lower()
        
        # Kiểm tra có chứa @
        if '@' not in email:
            result['error'] = 'Email phải chứa ký tự @'
            return result
        
        local_part, domain = email.rsplit('@', 1)
        
        # Kiểm tra local part (phần trước @)
        if not local_part:
            result['error'] = 'Thiếu tên người dùng trước ký tự @'
            return result
        
        # Kiểm tra domain (phần sau @)
        if not domain:
            result['error'] = 'Thiếu tên miền sau ký tự @'
            return result
        
        if '.' not in domain:
            result['error'] = 'Tên miền email không hợp lệ (thiếu phần mở rộng như .com, .vn)'
            return result
        
        # Kiểm tra ký tự đặc biệt/unicode trong email
        import unicodedata
        for char in email:
            if ord(char) > 127:  # Ký tự không phải ASCII
                result['error'] = f'Email không được chứa ký tự có dấu hoặc ký tự đặc biệt ({char}). Chỉ sử dụng chữ cái a-z, số 0-9 và các ký tự . _ % + -'
                return result
        
        # Pattern kiểm tra email chuẩn
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if not re.match(email_pattern, email):
            # Xác định lỗi cụ thể
            if not re.match(r'^[a-zA-Z0-9._%+-]+$', local_part):
                result['error'] = 'Tên người dùng email chỉ được chứa chữ cái (a-z), số (0-9) và các ký tự . _ % + -'
            elif not re.match(r'^[a-zA-Z0-9.-]+$', domain.rsplit('.', 1)[0]):
                result['error'] = 'Tên miền email không hợp lệ'
            else:
                result['error'] = 'Định dạng email không hợp lệ'
            return result
        
        # Local part không được bắt đầu/kết thúc bằng dấu chấm
        if local_part.startswith('.') or local_part.endswith('.'):
            result['error'] = 'Tên người dùng email không được bắt đầu hoặc kết thúc bằng dấu chấm (.)'
            return result
        
        # Không được có 2 dấu chấm liên tiếp
        if '..' in email:
            result['error'] = 'Email không được chứa hai dấu chấm liên tiếp (..)'
            return result
        
        result['valid'] = True
        return result

    @staticmethod
    def get_email_domain(email: str) -> str:
        """Lấy domain từ email"""
        if '@' not in email:
            return ''
        return email.rsplit('@', 1)[1].lower()

    @staticmethod
    def is_disposable_email(email: str) -> bool:
        """
        Kiểm tra xem email có phải từ domain tạm thời/disposable không
        """
        domain = EmailValidator.get_email_domain(email)
        return domain in EmailValidator.DISPOSABLE_DOMAINS

    @staticmethod
    def check_mx_records(email: str) -> dict:
        """
        Kiểm tra DNS MX records của domain email
        Trả về dict với thông tin:
        - valid: True/False
        - mx_records: list các MX records
        - message: thông báo
        """
        result = {
            'valid': False,
            'mx_records': [],
            'message': ''
        }
        
        domain = EmailValidator.get_email_domain(email)
        
        if not domain:
            result['message'] = 'Email không hợp lệ'
            return result
        
        # Nếu domain trong trusted list, bỏ qua MX check
        if domain in EmailValidator.TRUSTED_DOMAINS:
            result['valid'] = True
            result['message'] = 'Domain tin cậy'
            return result
        
        try:
            # Query MX records
            mx_records = dns.resolver.resolve(domain, 'MX')
            
            for mx in mx_records:
                result['mx_records'].append({
                    'priority': mx.preference,
                    'host': str(mx.exchange).rstrip('.')
                })
            
            if result['mx_records']:
                result['valid'] = True
                result['message'] = f'Tìm thấy {len(result["mx_records"])} MX records'
            else:
                result['message'] = 'Không tìm thấy MX records'
                
        except dns.resolver.NXDOMAIN:
            result['message'] = 'Domain không tồn tại'
        except dns.resolver.NoAnswer:
            result['message'] = 'Domain không có MX records'
        except dns.resolver.NoNameservers:
            result['message'] = 'Không thể truy vấn DNS'
        except dns.exception.Timeout:
            result['message'] = 'Timeout khi truy vấn DNS'
            # Cho phép tiếp tục nếu timeout (có thể do mạng)
            result['valid'] = True
        except Exception as e:
            result['message'] = f'Lỗi kiểm tra DNS: {str(e)}'
            # Cho phép tiếp tục nếu có lỗi không xác định
            result['valid'] = True
            
        return result

    @staticmethod
    def validate_email_exists(email: str) -> dict:
        """
        Xác thực email tồn tại thực sự
        Kết hợp kiểm tra cú pháp, disposable check, và MX records
        """
        email = (email or '').strip().lower()
        
        result = {
            'valid': False,
            'email': email,
            'errors': []
        }
        
        # 1. Kiểm tra cú pháp chi tiết
        format_result = EmailValidator.validate_email_format(email)
        if not format_result['valid']:
            result['errors'].append(format_result['error'])
            return result
        
        # 2. Kiểm tra disposable email
        if EmailValidator.is_disposable_email(email):
            result['errors'].append('❌ Không chấp nhận email tạm thời (tempmail, guerrillamail, mailinator...). Vui lòng sử dụng email thật.')
            return result
        
        # 3. Kiểm tra MX records
        mx_result = EmailValidator.check_mx_records(email)
        if not mx_result['valid']:
            domain = EmailValidator.get_email_domain(email)
            result['errors'].append(f'❌ Tên miền email "{domain}" không tồn tại hoặc không thể nhận email. Vui lòng kiểm tra lại.')
            return result
        
        result['valid'] = True
        return result

    @staticmethod
    def _hash_otp(otp: str, salt: str) -> str:
        """Hash OTP với salt và secret key"""
        raw = f"{otp}:{salt}:{settings.SECRET_KEY}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def send_verification_otp(email: str, name: str = '') -> None:
        """
        Gửi OTP xác thực email
        
        Args:
            email: Địa chỉ email cần xác thực
            name: Tên người dùng (optional)
        
        Raises:
            EmailValidationError: Nếu email không hợp lệ
        """
        from Weather_Forcast_App.scripts.email_templates import generate_otp, send_otp_email
        
        email = (email or '').strip().lower()
        
        # Validate email trước khi gửi
        validation = EmailValidator.validate_email_exists(email)
        if not validation['valid']:
            raise EmailValidationError(', '.join(validation['errors']))
        
        # Vô hiệu hóa OTP cũ
        email_verification_otps.update_many(
            {"email": email, "used": False}, 
            {"$set": {"used": True}}
        )
        
        # Tạo OTP mới (5 số)
        otp = generate_otp()
        salt = secrets.token_hex(8)
        
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=EMAIL_OTP_EXPIRE_SECONDS)
        
        # Lưu vào database
        email_verification_otps.insert_one({
            "email": email,
            "otpHash": EmailValidator._hash_otp(otp, salt),
            "salt": salt,
            "attempts": 0,
            "used": False,
            "createdAt": now,
            "expiresAt": expires_at,
        })
        
        # Gửi email với template đẹp
        try:
            send_otp_email(
                email=email,
                name=name,
                otp=otp,
                purpose="đăng ký",
                expire_minutes=EMAIL_OTP_EXPIRE_SECONDS // 60
            )
        except Exception as e:
            raise EmailValidationError(f"Không thể gửi email xác thực: {str(e)}")

    @staticmethod
    def verify_email_otp(email: str, otp: str) -> bool:
        """
        Xác thực OTP email
        
        Args:
            email: Địa chỉ email
            otp: Mã OTP người dùng nhập
            
        Returns:
            True nếu OTP hợp lệ
            
        Raises:
            EmailValidationError: Nếu OTP không hợp lệ
        """
        email = (email or '').strip().lower()
        otp = (otp or '').strip()
        
        if not email or not otp:
            raise EmailValidationError("Vui lòng nhập đầy đủ email và mã OTP")
        
        now = datetime.now(timezone.utc)
        
        # Tìm OTP record mới nhất chưa sử dụng và chưa hết hạn
        rec = email_verification_otps.find({
            "email": email, 
            "used": False, 
            "expiresAt": {"$gt": now}
        }).sort("createdAt", -1).limit(1)
        
        rec = next(rec, None)
        
        if not rec:
            raise EmailValidationError("Mã OTP không hợp lệ hoặc đã hết hạn. Vui lòng yêu cầu mã mới.")
        
        # Kiểm tra số lần thử
        if rec["attempts"] >= EMAIL_OTP_MAX_ATTEMPTS:
            raise EmailValidationError("Bạn đã nhập sai quá nhiều lần. Vui lòng yêu cầu mã OTP mới.")
        
        # Kiểm tra OTP
        expected = EmailValidator._hash_otp(otp, rec["salt"])
        if expected != rec["otpHash"]:
            # Tăng số lần thử sai
            email_verification_otps.update_one(
                {"_id": rec["_id"]}, 
                {"$inc": {"attempts": 1}}
            )
            remaining = EMAIL_OTP_MAX_ATTEMPTS - rec["attempts"] - 1
            raise EmailValidationError(f"Mã OTP không đúng. Còn {remaining} lần thử.")
        
        # Đánh dấu OTP đã được verify
        email_verification_otps.update_one(
            {"_id": rec["_id"]}, 
            {"$set": {"verifiedAt": now, "used": True}}
        )
        
        return True

    @staticmethod
    def is_email_verified(email: str, within_seconds: int = 3600) -> bool:
        """
        Kiểm tra xem email đã được xác thực trong khoảng thời gian gần đây chưa
        
        Args:
            email: Địa chỉ email
            within_seconds: Thời gian tính từ hiện tại (mặc định 1 giờ)
            
        Returns:
            True nếu email đã được xác thực
        """
        email = (email or '').strip().lower()
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=within_seconds)
        
        rec = email_verification_otps.find_one({
            "email": email,
            "verifiedAt": {"$exists": True, "$gte": cutoff}
        })
        
        return rec is not None
