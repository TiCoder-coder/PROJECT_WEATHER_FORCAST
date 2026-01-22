"""
Email Templates cho VN Weather Hub
Gửi email OTP - Hỗ trợ nhiều provider (Resend, SMTP)
"""
import os
import secrets
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Resend API config
RESEND_API_KEY = os.getenv("RESEND_API_KEY")
RESEND_FROM_EMAIL = os.getenv("RESEND_FROM_EMAIL", "onboarding@resend.dev")


def generate_otp() -> str:
    """Tạo mã OTP 5 số an toàn (dùng secrets thay vì random)"""
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
        name: Tên người dùng
        otp: Mã OTP
        purpose: Mục đích (xác thực / đặt lại mật khẩu)
        expire_minutes: Thời gian hết hạn (phút)
    
    Returns:
        tuple: (subject, plain_message, html_message)
    """
    
    greeting = f"Xin chào {name}!" if name else "Xin chào bạn!"
    
    if purpose == "đăng ký":
        subject = "🌦️ VN Weather Hub - Xác thực email đăng ký"
        action_text = "đăng ký tài khoản"
        intro = "Cảm ơn bạn đã đăng ký tài khoản tại VN Weather Hub!"
    else:
        subject = "🔐 VN Weather Hub - Mã OTP đặt lại mật khẩu"
        action_text = "đặt lại mật khẩu"
        intro = "Bạn đã yêu cầu đặt lại mật khẩu cho tài khoản VN Weather Hub."
    
    # Plain text version
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

    # HTML version (đẹp hơn)
    html_message = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #0f172a;">
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background-color: #0f172a; padding: 40px 20px;">
        <tr>
            <td align="center">
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
    """
    import requests
    from django.conf import settings
    
    subject, plain_message, html_message = get_otp_email_template(
        name=name,
        otp=otp,
        purpose=purpose,
        expire_minutes=expire_minutes
    )
    
    # Kiểm tra xem có cấu hình email không
    has_smtp = bool(getattr(settings, 'EMAIL_HOST_PASSWORD', None))
    has_resend = bool(RESEND_API_KEY)
    
    # Nếu không có cấu hình email nào -> In ra console (development mode)
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
    
    # Thử Resend API trước
    if RESEND_API_KEY:
        try:
            response = requests.post(
                "https://api.resend.com/emails",
                headers={
                    "Authorization": f"Bearer {RESEND_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "from": f"VN Weather Hub <{RESEND_FROM_EMAIL}>",
                    "to": [email],
                    "subject": subject,
                    "html": html_message,
                    "text": plain_message
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"[EMAIL] Sent to {email} via Resend API, id: {result.get('id')}")
                return {"success": True, "provider": "resend", "result": result}
            else:
                error_data = response.json()
                error_msg = error_data.get("message", "Unknown error")
                print(f"[EMAIL] Resend API error: {error_msg}")
                # Nếu lỗi là do chưa verify domain, fallback sang SMTP
                if "verify a domain" in error_msg.lower() or response.status_code == 403:
                    print("[EMAIL] Falling back to SMTP...")
                else:
                    # Nếu không có SMTP, in ra console
                    if not has_smtp:
                        print(f"\n[FALLBACK] OTP cho {email}: {otp}\n")
                        return {"success": True, "provider": "console", "otp": otp}
                    raise Exception(f"Resend API error: {error_msg}")
        except requests.exceptions.RequestException as e:
            print(f"[EMAIL] Resend API request error: {e}")
            if not has_smtp:
                print(f"\n[FALLBACK] OTP cho {email}: {otp}\n")
                return {"success": True, "provider": "console", "otp": otp}
            print("[EMAIL] Falling back to SMTP...")
    
    # Fallback: Dùng Django SMTP
    if has_smtp:
        from django.core.mail import send_mail
        
        try:
            result = send_mail(
                subject=subject,
                message=plain_message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[email],
                html_message=html_message,
                fail_silently=False
            )
            print(f"[EMAIL] Sent to {email} via SMTP, result: {result}")
            return {"success": True, "provider": "smtp", "result": result}
        except Exception as e:
            print(f"[EMAIL] SMTP failed: {e}")
            # Fallback cuối cùng: in ra console
            print(f"\n[FALLBACK] OTP cho {email}: {otp}\n")
            return {"success": True, "provider": "console", "otp": otp}
    
    # Fallback cuối: console
    print(f"\n[FALLBACK] OTP cho {email}: {otp}\n")
    return {"success": True, "provider": "console", "otp": otp}
