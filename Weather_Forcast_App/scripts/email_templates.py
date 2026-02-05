"""
Email Templates cho VN Weather Hub
Gửi email OTP - Hỗ trợ nhiều provider (Resend, SMTP)

Giải thích tổng quan:
- File này có 3 phần chính:
  (1) Load ENV + cấu hình provider Resend
  (2) Sinh OTP an toàn bằng secrets
  (3) Tạo template email (plain text + HTML) và gửi email theo thứ tự ưu tiên:
      - Resend API (nếu có RESEND_API_KEY)
      - SMTP của Django (nếu có EMAIL_HOST_PASSWORD)
      - Console (dev mode / fallback)
"""
import os
import secrets
from dotenv import load_dotenv

# ============================================================
# LOAD BIẾN MÔI TRƯỜNG (.env)
# ============================================================
# - load_dotenv() sẽ đọc file .env (nếu có) và đưa các biến vào os.environ
# - Giúp bạn cấu hình key/email mà không hardcode trong source
load_dotenv()

# ============================================================
# CẤU HÌNH RESEND (Provider 1)
# ============================================================
# - RESEND_API_KEY: API key để gọi Resend gửi email
# - RESEND_FROM_EMAIL: email người gửi (from). Mặc định dùng onboarding@resend.dev
#   (thường khi dùng domain riêng bạn sẽ đổi lại để tránh bị chặn)
RESEND_API_KEY = os.getenv("RESEND_API_KEY")
RESEND_FROM_EMAIL = os.getenv("RESEND_FROM_EMAIL", "onboarding@resend.dev")


def generate_otp() -> str:
    """Tạo mã OTP 5 số an toàn (dùng secrets thay vì random)

    Giải thích:
    - secrets: module dành cho mục đích bảo mật (cryptographically strong)
      tốt hơn random trong trường hợp OTP/token.
    - secrets.randbelow(100000) -> số nguyên 0..99999
    - f"{...:05d}" -> format thành chuỗi luôn đủ 5 chữ số (vd: 00042, 93812)
    """
    return f"{secrets.randbelow(100000):05d}"


def get_otp_email_template(
    name: str,
    otp: str,
    purpose: str = "xác thực",
    expire_minutes: int = 10
) -> tuple:
    """
    Tạo template email OTP cá nhân hóa
    
    Args:
        name: Tên người dùng (để chào theo tên, có thể rỗng)
        otp: Mã OTP (chuỗi 5 số)
        purpose: Mục đích (xác thực / đặt lại mật khẩu / đăng ký)
                 (Ở code của bạn: có nhánh 'đăng ký' và nhánh còn lại coi như reset pass)
        expire_minutes: Thời gian hết hạn (phút)
    
    Returns:
        tuple: (subject, plain_message, html_message)
        - subject: tiêu đề email
        - plain_message: nội dung text thuần (fallback cho client không hỗ trợ HTML)
        - html_message: nội dung HTML (đẹp, có style)
    """
    
    # ============================================================
    # LỜI CHÀO (Greeting) THEO TÊN
    # ============================================================
    # - Nếu có name -> "Xin chào {name}!"
    # - Nếu name rỗng/None -> "Xin chào bạn!"
    greeting = f"Xin chào {name}!" if name else "Xin chào bạn!"
    
    # ============================================================
    # XÁC ĐỊNH NỘI DUNG THEO purpose
    # ============================================================
    # - purpose == "đăng ký": email xác thực đăng ký
    # - else: mặc định coi là OTP đặt lại mật khẩu
    #
    # Các biến tạo ra:
    # - subject: tiêu đề email
    # - action_text: dùng để đưa vào phần cảnh báo ("Nếu bạn không yêu cầu ...")
    # - intro: đoạn mở đầu nội dung
    if purpose == "đăng ký":
        subject = "🌦️ VN Weather Hub - Xác thực email đăng ký"
        action_text = "đăng ký tài khoản"
        intro = "Cảm ơn bạn đã đăng ký tài khoản tại VN Weather Hub!"
    else:
        subject = "🔐 VN Weather Hub - Mã OTP đặt lại mật khẩu"
        action_text = "đặt lại mật khẩu"
        intro = "Bạn đã yêu cầu đặt lại mật khẩu cho tài khoản VN Weather Hub."
    
    # ============================================================
    # PLAIN TEXT VERSION (Text thuần)
    # ============================================================
    # - Dùng f-string để chèn greeting, intro, otp, expire_minutes, action_text
    # - Format khung bằng ký tự line để đọc dễ trong email text-only
    plain_message = f"""{greeting}

{intro}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   MÃ XÁC THỰC CỦA BẠN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        🔑 {otp}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⏱️ Mã có hiệu lực trong {expire_minutes} phút.

⚠️ Lưu ý bảo mật:
• Không chia sẻ mã này với bất kỳ ai
• VN Weather Hub sẽ không bao giờ yêu cầu mã OTP qua điện thoại
• Nếu bạn không yêu cầu {action_text}, vui lòng bỏ qua email này

━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Trân trọng,
🌦️ VN Weather Hub Team

© 2026 VN Weather Hub. All rights reserved.
"""

    # ============================================================
    # HTML VERSION (đẹp hơn)
    # ============================================================
    # - HTML có inline-style để tương thích tốt trong email clients (Gmail, Outlook...)
    # - Có layout bằng <table> vì email client thường không hỗ trợ flex/grid đầy đủ
    # - OTP được đặt trong "OTP Box" nổi bật
    # - Có notice bảo mật dạng khung vàng
    #
    # Lưu ý: Các biến chèn vào HTML:
    # - {greeting}, {intro}, {otp}, {expire_minutes}, {action_text}
    html_message = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <!-- viewport giúp hiển thị tốt trên mobile -->
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<!-- body nền tối, font phổ biến để email client render ổn -->
<body style="margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #0f172a;">
    <!-- Table wrapper: email layout an toàn -->
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background-color: #0f172a; padding: 40px 20px;">
        <tr>
            <td align="center">
                <!-- Container chính, giới hạn max 500px -->
                <table role="presentation" width="100%" max-width="500" cellspacing="0" cellpadding="0" style="max-width: 500px; background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 16px; border: 1px solid #334155; overflow: hidden;">
                    
                    <!-- Header -->
                    <tr>
                        <td style="background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); padding: 30px; text-align: center;">
                            <h1 style="margin: 0; color: #ffffff; font-size: 24px; font-weight: 600;">
                                🌦️ VN Weather Hub
                            </h1>
                        </td>
                    </tr>
                    
                    <!-- Content -->
                    <tr>
                        <td style="padding: 40px 30px;">
                            <!-- Greeting -->
                            <h2 style="margin: 0 0 20px 0; color: #f1f5f9; font-size: 20px; font-weight: 600;">
                                {greeting}
                            </h2>
                            
                            <p style="margin: 0 0 25px 0; color: #94a3b8; font-size: 15px; line-height: 1.6;">
                                {intro}
                            </p>
                            
                            <!-- OTP Box -->
                            <div style="background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%); border: 2px solid #3b82f6; border-radius: 12px; padding: 25px; text-align: center; margin: 25px 0;">
                                <p style="margin: 0 0 10px 0; color: #94a3b8; font-size: 13px; text-transform: uppercase; letter-spacing: 2px;">
                                    Mã xác thực của bạn
                                </p>
                                <!-- OTP hiển thị to, font monospace + letter-spacing -->
                                <div style="font-size: 36px; font-weight: 700; color: #3b82f6; letter-spacing: 8px; font-family: 'Courier New', monospace;">
                                    {otp}
                                </div>
                                <p style="margin: 15px 0 0 0; color: #f59e0b; font-size: 13px;">
                                    ⏱️ Có hiệu lực trong {expire_minutes} phút
                                </p>
                            </div>
                            
                            <!-- Security Notice -->
                            <div style="background-color: rgba(245, 158, 11, 0.1); border-left: 3px solid #f59e0b; padding: 15px; border-radius: 0 8px 8px 0; margin-top: 25px;">
                                <p style="margin: 0; color: #fbbf24; font-size: 13px; font-weight: 600;">
                                    ⚠️ Lưu ý bảo mật:
                                </p>
                                <ul style="margin: 10px 0 0 0; padding-left: 20px; color: #94a3b8; font-size: 13px; line-height: 1.8;">
                                    <li>Không chia sẻ mã này với bất kỳ ai</li>
                                    <li>VN Weather Hub không bao giờ yêu cầu mã OTP qua điện thoại</li>
                                    <li>Nếu bạn không yêu cầu {action_text}, hãy bỏ qua email này</li>
                                </ul>
                            </div>
                        </td>
                    </tr>
                    
                    <!-- Footer -->
                    <tr>
                        <td style="background-color: #0f172a; padding: 25px 30px; border-top: 1px solid #334155; text-align: center;">
                            <p style="margin: 0 0 10px 0; color: #64748b; font-size: 13px;">
                                Trân trọng,<br>
                                <strong style="color: #94a3b8;">🌦️ VN Weather Hub Team</strong>
                            </p>
                            <p style="margin: 0; color: #475569; font-size: 11px;">
                                © 2026 VN Weather Hub. All rights reserved.
                            </p>
                        </td>
                    </tr>
                    
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
"""
    
    # Trả về bộ 3: subject + text + html để hàm send_otp_email dùng
    return subject, plain_message, html_message


def send_otp_email(
    email: str,
    name: str,
    otp: str,
    purpose: str = "xác thực",
    expire_minutes: int = 10
) -> dict:
    """
    Gửi email OTP - Hỗ trợ nhiều phương thức:
    1. Resend API (nếu có RESEND_API_KEY)
    2. Django SMTP (nếu có EMAIL_HOST_PASSWORD)
    3. Console (in ra terminal - development mode)
    
    Args:
        email: Địa chỉ email người nhận
        name: Tên người dùng
        otp: Mã OTP
        purpose: Mục đích gửi
        expire_minutes: Thời gian hết hạn
    
    Returns:
        dict: Kết quả gửi email
        - success: True/False
        - provider: resend/smtp/console
        - result: dữ liệu trả về từ provider (nếu có)
        - otp: (chỉ trong console mode) để debug
    """
    import requests
    from django.conf import settings
    
    # ============================================================
    # TẠO NỘI DUNG EMAIL (subject + plain + html)
    # ============================================================
    subject, plain_message, html_message = get_otp_email_template(
        name=name,
        otp=otp,
        purpose=purpose,
        expire_minutes=expire_minutes
    )
    
    # ============================================================
    # CHECK PROVIDER KHẢ DỤNG
    # ============================================================
    # has_smtp:
    # - Nếu Django settings có EMAIL_HOST_PASSWORD => khả năng SMTP được cấu hình
    # - (Đây là cách check "có password" chứ chưa chắc cấu hình đúng 100%)
    has_smtp = bool(getattr(settings, 'EMAIL_HOST_PASSWORD', None))

    # has_resend:
    # - Chỉ cần RESEND_API_KEY tồn tại là coi như có thể dùng Resend
    has_resend = bool(RESEND_API_KEY)
    
    # ============================================================
    # DEVELOPMENT MODE: KHÔNG CÓ SMTP VÀ CŨNG KHÔNG CÓ RESEND
    # ============================================================
    # - Trong giai đoạn dev/test, bạn có thể chưa set key / chưa set SMTP
    # - Khi đó in OTP ra console để test luồng xác thực
    if not has_smtp and not has_resend:
        print("\n" + "="*60)
        print("📧 [DEVELOPMENT MODE] - OTP sẽ được in ra console")
        print("="*60)
        print(f"📮 Email: {email}")
        print(f"👤 Tên: {name}")
        print(f"🎯 Mục đích: {purpose}")
        print(f"🔑 MÃ OTP: {otp}")
        print(f"⏱️ Hết hạn sau: {expire_minutes} phút")
        print("="*60 + "\n")
        return {"success": True, "provider": "console", "otp": otp}
    
    # ============================================================
    # PROVIDER 1: RESEND API (ƯU TIÊN NẾU CÓ KEY)
    # ============================================================
    # - Nếu có RESEND_API_KEY, thử gửi qua Resend trước
    # - Nếu Resend lỗi và có SMTP -> fallback sang SMTP
    # - Nếu Resend lỗi và không có SMTP -> fallback console
    if RESEND_API_KEY:
        try:
            # Gọi API gửi email của Resend
            response = requests.post(
                "https://api.resend.com/emails",
                headers={
                    "Authorization": f"Bearer {RESEND_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    # From hiển thị: "VN Weather Hub <email@domain>"
                    "from": f"VN Weather Hub <{RESEND_FROM_EMAIL}>",
                    # Resend yêu cầu list người nhận
                    "to": [email],
                    "subject": subject,
                    "html": html_message,   # HTML content
                    "text": plain_message   # Text fallback
                },
                timeout=30  # timeout để tránh treo request quá lâu
            )
            
            # Resend thường trả 200 nếu thành công
            if response.status_code == 200:
                result = response.json()
                print(f"[EMAIL] Sent to {email} via Resend API, id: {result.get('id')}")
                return {"success": True, "provider": "resend", "result": result}
            else:
                # Nếu không phải 200: parse lỗi từ response JSON
                error_data = response.json()
                error_msg = error_data.get("message", "Unknown error")
                print(f"[EMAIL] Resend API error: {error_msg}")

                # Một số lỗi thường gặp của Resend:
                # - domain chưa verify
                # - bị 403 forbidden
                # Khi đó bạn chủ động fallback SMTP nếu có
                if "verify a domain" in error_msg.lower() or response.status_code == 403:
                    print("[EMAIL] Falling back to SMTP...")
                else:
                    # Nếu không có SMTP để fallback -> in ra console để test
                    if not has_smtp:
                        print(f"\n[FALLBACK] OTP cho {email}: {otp}\n")
                        return {"success": True, "provider": "console", "otp": otp}
                    # Nếu có SMTP thì raise để đi vào flow fallback phía dưới
                    raise Exception(f"Resend API error: {error_msg}")
        except requests.exceptions.RequestException as e:
            # RequestException: lỗi mạng, timeout, DNS...
            print(f"[EMAIL] Resend API request error: {e}")
            # Nếu không có SMTP -> fallback console ngay
            if not has_smtp:
                print(f"\n[FALLBACK] OTP cho {email}: {otp}\n")
                return {"success": True, "provider": "console", "otp": otp}
            # Nếu có SMTP -> thông báo và để chạy xuống phần SMTP
            print("[EMAIL] Falling back to SMTP...")
    
    # ============================================================
    # PROVIDER 2: SMTP (Django send_mail)
    # ============================================================
    # - Chỉ chạy nếu has_smtp == True
    # - send_mail sẽ dùng cấu hình trong settings.py:
    #   EMAIL_HOST, EMAIL_PORT, EMAIL_HOST_USER, EMAIL_HOST_PASSWORD,
    #   EMAIL_USE_TLS/SSL, DEFAULT_FROM_EMAIL, ...
    if has_smtp:
        from django.core.mail import send_mail
        
        try:
            # send_mail trả về số email gửi thành công (thường 1 nếu OK)
            result = send_mail(
                subject=subject,
                message=plain_message,              # text version
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[email],
                html_message=html_message,          # html version
                fail_silently=False                 # False để ném exception nếu lỗi SMTP
            )
            print(f"[EMAIL] Sent to {email} via SMTP, result: {result}")
            return {"success": True, "provider": "smtp", "result": result}
        except Exception as e:
            # Nếu SMTP lỗi:
            # - in lỗi để debug
            # - fallback console để vẫn test được OTP flow
            print(f"[EMAIL] SMTP failed: {e}")
            print(f"\n[FALLBACK] OTP cho {email}: {otp}\n")
            return {"success": True, "provider": "console", "otp": otp}
    
    # ============================================================
    # FALLBACK CUỐI CÙNG (khi không gửi được provider nào)
    # ============================================================
    # - Trường hợp này thường hiếm vì phía trên đã handle đa số tình huống
    print(f"\n[FALLBACK] OTP cho {email}: {otp}\n")
    return {"success": True, "provider": "console", "otp": otp}  # (giữ nguyên theo yêu cầu: chỉ thêm chú thích, không đổi logic)
