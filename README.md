---

# 🌦️ Weather_Forcast_App — Weather Data Pipeline & Dashboard

<b>Django</b> app để <b>crawl</b> dữ liệu thời tiết → <b>gộp (merge)</b> → <b>làm sạch (clean)</b> → <b>xem trước / tải về</b> dataset (CSV/Excel/JSON/TXT) với giao diện “glass + weather effects”.

<br/>

<img alt="Python" src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white">
<img alt="Django" src="https://img.shields.io/badge/Django-3.x-092E20?logo=django&logoColor=white">
<img alt="Pandas" src="https://img.shields.io/badge/Pandas-data-150458?logo=pandas&logoColor=white">
<img alt="UI" src="https://img.shields.io/badge/UI-Glassmorphism-7C3AED">
<img alt="Datasets" src="https://img.shields.io/badge/Datasets-Preview%20%26%20Download-0EA5E9">

<br/>
<sub>🔗 Merge workflow • 🧹 Clean wizard • 📄 Dataset preview • 🌧️ Weather effects • 📦 Download</sub>

</div>

---
![Picture](https://nub.news/api/image/681000/article.png)
---

## 📌 Mục lục
<details open>
<summary><b>📚 Mục lục</b></summary>

- [1. Tổng quan](#1-tổng-quan)
- [2. Các luồng dữ liệu trong project](#2-các-luồng-dữ-liệu-trong-project)
- [3. Tính năng nổi bật](#3-tính-năng-nổi-bật)
- [4. Cấu trúc thư mục dữ liệu](#4-cấu-trúc-thư-mục-dữ-liệu)
- [5. Giao diện chính](#5-giao-diện-chính)
- [6. Routes / Endpoints](#6-routes--endpoints)
- [7. Mapping “folder key” (rất quan trọng)](#7-mapping-folder-key-rất-quan-trọng)
- [8. Dataset Preview (CSV/Excel/JSON/TXT)](#8-dataset-preview-csvexceljsontxt)
- [9. Clean Wizard](#9-clean-wizard)
- [10. Merge result modal](#10-merge-result-modal)
- [11. Cài đặt & chạy](#11-cài-đặt--chạy)
- [12. Lỗi thường gặp & cách xử lý](#12-lỗi-thường-gặp--cách-xử-lý)
- [13. Roadmap](#13-roadmap)
- [14. Ghi chú nguồn dữ liệu](#14-ghi-chú-nguồn-dữ-liệu)

</details>

---

## 1. 🌤️ Tổng quan

**Weather_Forcast_App** là một hệ thống **Django** tập trung vào **pipeline dữ liệu thời tiết end-to-end**  
*(thu thập → lưu trữ → xử lý → hiển thị)* và **dashboard web** giúp người dùng thao tác dữ liệu trực quan mà không cần mở file thủ công.

### 🎯 Mục tiêu chính

- 🧪 **Xử lý dữ liệu**
  - Crawl / Merge / Clean theo luồng rõ ràng
  - Có log
  - Có phân loại thư mục theo từng nhóm dữ liệu
- 🖥️ **Trải nghiệm người dùng**
  - Xem trước (preview) dataset trực tiếp trên web
  - Tải file nhanh theo từng nhóm (download)

---

## 🧱 Kiến trúc tổng thể (Multi-layer)

Hệ thống được chia thành **3 layer chính** (dễ mở rộng / dễ bảo trì):

### 🎨 1) Presentation Layer (UI / Templates / Static)

- Giao diện người dùng Django Template:
  - 🏠 `Home.html` — Trang tổng quan
  - 📚 `Datasets.html` — Danh sách dataset theo nhóm
  - 👀 `dataset_preview.html` — Xem trước nội dung file (table/text)
- CSS/JS trong `static/weather/...` để:
  - ✅ UI đẹp, responsive
  - ⚡ Hiệu ứng thời tiết (mây, mưa, sấm…)
  - 🧭 Modal/Overlay cho **Merge** & **Clean Wizard**

---

### 🧩 2) Application Layer (Views / Routing)

- Các view trong `Weather_Forcast_App/views/...` đóng vai trò **controller**:
  - 🏠 `Home.py` — Điều hướng và hiển thị tổng quan
  - 📦 `View_Datasets.py` — List dataset theo thư mục + Preview/Download
  - 🔗 `View_Merge_Data.py` — API/Endpoint gộp dữ liệu (merge)
  - 🧼 `View_Clear.py` — API/Endpoint làm sạch dữ liệu (clean)
  - 🌧️ Các view crawl: Selenium / API / HTML parsing từ **Vrain** & **OpenWeather**
- `urls.py` định nghĩa route:
  - 👀 Xem file: `dataset_view`
  - ⬇️ Tải file: `dataset_download`
  - 🔗 Merge: `merge_data`
  - 🧼 Clean wizard: `clean_list`, `clean_data`, `clean_tail`...

---

### ⚙️ 3) Data/Processing Layer (Scripts + Storage)

- Các script xử lý trong `Weather_Forcast_App/scripts/...` là “engine” chạy thật:
  - 🌐 Crawl data (API / Selenium / HTML)
  - 🔗 Merge nhiều file → 1 dataset chung
  - 🧼 Clean data: chuẩn hóa, xử lý thiếu, bỏ trùng, format...
- Dữ liệu đầu ra/đầu vào được quản lý theo **thư mục chuẩn** (theo nhóm raw/merged/cleaned)

---

## 🗃️ Hệ dữ liệu & định dạng file

Project dùng **nhiều loại storage** (tùy mục đích):

### ✅ 1) Database (SQL / SQLite)

- 🗄️ `db.sqlite3` — DB mặc định của Django (dev)
- 🧊 `vietnam_weather.db` — DB riêng cho dữ liệu thời tiết (tuỳ bạn dùng cho lưu record/summary)

### ✅ 2) File-based datasets (CSV / XLSX / JSON / TXT)

- 📄 **CSV** — nhẹ, dễ xử lý, phù hợp Pandas/ML
- 📊 **XLSX** — phù hợp báo cáo, nhiều sheet, dễ đọc cho người dùng
- 🧾 **JSON/TXT** — phục vụ preview/log/định dạng khác

---

## 🧭 Những tính năng người dùng có thể làm trên web

### 👁️ Duyệt dataset theo nhóm thư mục

- 📦 `output/` — dữ liệu thô (raw) sau crawl *(chưa merge)*
- 🔗 `Merge_data/` — dữ liệu đã gộp *(merged)*
- 🧼 `cleaned_data/` — dữ liệu đã làm sạch *(cleaned)*
  - 🧩 `Clean_Data_For_File_Merge/` — clean từ dữ liệu **đã merge**
  - 📦 `Clean_Data_For_File_Not_Merge/` — clean từ dữ liệu **raw/output**

### 🔍 Preview trực tiếp trên web

- 📊 CSV/XLSX: hiển thị dạng bảng + phân trang/pagination
- 🧾 JSON/TXT: hiển thị dạng text/preformatted
- ✅ Mở nhanh “xem ngay” mà không cần download

### ⬇️ Download file

- Tải trực tiếp dataset theo từng nhóm (raw/merged/cleaned)

### 🔗 Merge data (raw → merged)

- Bấm nút **Merge** → hệ thống gộp dữ liệu → lưu vào `Merge_data/`
- ✅ Có thể hiển thị file mới nhất + cho **Xem/Tải ngay** sau khi merge (modal)

### 🧼 Clean data (2 nhánh)

- 🧩 Clean từ file đã merge → output vào `Clean_Data_For_File_Merge/`
- 📦 Clean từ file chưa merge → output vào `Clean_Data_For_File_Not_Merge/`
- ✅ Có wizard: chọn nguồn → chọn file → xem tiến trình → xem/tải kết quả

---

## 2. Các luồng dữ liệu trong project

```
flowchart LR
  A[Crawl modules\n(API / HTML / Selenium)] --> B[output/\nRaw datasets]
  B -->|Merge| C[Merge_data/\nMerged datasets]
  C -->|Clean (merge source)| D[cleaned_data/Clean_Data_For_File_Merge/\nCleaned merged]
  B -->|Clean (output source)| E[cleaned_data/Clean_Data_For_File_Not_Merge/\nCleaned raw]
  C --> F[Datasets page]
  D --> F
  E --> F
  F --> G[Dataset Preview\n/view/...]
  F --> H[Download\n/download/...]
```

---

## 3. Tính năng nổi bật

### 📁 Duyệt dataset theo nhóm
- **DỮ LIỆU ĐÃ GỘP**: đọc từ thư mục `Merge_data/`
- **DỮ LIỆU THÔ (OUTPUT)**: đọc từ thư mục `output/`
- **DỮ LIỆU ĐÃ LÀM SẠCH**: đọc từ `cleaned_data/…` (gồm 2 nhánh)

### 👀 Xem trước (Preview)
- CSV/Excel → render bảng, hỗ trợ **pagination / tải thêm**
- JSON → **syntax highlight**
- TXT → hiển thị text trong khung scroll

### ⬇️ Tải về (Download)
- Download theo đúng folder key + filename, có kiểm tra an toàn (chỉ cho phép file trong thư mục hợp lệ)

### 🔗 Merge
- Nút **🔗 GỘP DỮ LIỆU** (ở section “Dữ liệu thô”)
- Backend chạy merge, trả JSON (success/message + thông tin file mới)
- Frontend có thể mở **Merge Result Modal** để người dùng:
  - xem tên file mới, dung lượng, thời gian
  - bấm **👀 XEM / ⬇️ TẢI**
  - bấm **✕** để đóng và quay lại

### 🧹 Clean Wizard (UI 3 bước)
1) Chọn nguồn:
   - `merge` (làm sạch từ file đã merge)
   - `output` (làm sạch từ file thô)
2) Chọn file (có search)
3) Theo dõi tiến trình + log + report và nút xem/tải kết quả

### 🌧️ Weather UI Effects
- Background layers: mây / gió / mưa / sấm chớp (CSS + JS random flash)

---

## 🔐 Hệ thống Xác thực (Authentication System)

Hệ thống xác thực bảo mật đầy đủ với **đăng nhập**, **đăng ký** và **quên mật khẩu** qua OTP email.

### 📋 Tổng quan tính năng

| Tính năng            | Mô tả                                           |
|----------------------|-------------------------------------------------|
| 🔑 **Đăng nhập**     | Hỗ trợ đăng nhập bằng username HOẶC email       |
| 📝 **Đăng ký**       | Xác thực email qua OTP trước khi tạo tài khoản  |
| 🔄 **Quên mật khẩu** | Reset password qua OTP gửi đến email            |          
| 🛡️ **Bảo mật**       | Mật khẩu mạnh, khóa tài khoản khi sai nhiều lần |
| 📧 **Email**         | Hỗ trợ Gmail SMTP, Resend API, Console mode     |

---

### 🔑 Đăng nhập (Login)

**Route:** `/login/`

#### Luồng hoạt động:
```
Người dùng nhập username/email + password
        ↓
Kiểm tra tài khoản tồn tại (find by username OR email)
        ↓
Kiểm tra tài khoản có bị khóa không
        ↓
Kiểm tra tài khoản có active không
        ↓
Xác thực mật khẩu (với pepper + hash)
        ↓
Tạo JWT token + Lưu session
        ↓
✅ Chuyển về trang Home
```

#### Tính năng bảo mật:
| Tính năng               | Chi tiết                                                        |
|-------------------------|-----------------------------------------------------------------|
| **Đăng nhập linh hoạt** | Có thể dùng username hoặc email                                 |
| **Pepper password**     | Thêm chuỗi bí mật trước khi hash                                |
| **Khóa tài khoản**      | Sau **5 lần** sai → khóa **5 phút**                             |
| **Đếm lần sai**         | Hiển thị số lần thử còn lại                                     |
| **JWT Token**           | Tạo token với role và manager_id                                |

#### Cấu trúc session sau đăng nhập:
```python
request.session["access_token"] = jwt_token
request.session["profile"] = {
    "_id": "...",
    "name": "Võ Anh Nhật",
    "userName": "nhat123",
    "email": "nhat@gmail.com",
    "role": "Staff",
    "last_login": "2026-01-22T10:00:00"
}
```

---

### 📝 Đăng ký (Register)

**Route:** `/register/` → `/verify-email-register/`

#### Luồng hoạt động (2 bước):
```
📋 BƯỚC 1: Nhập thông tin
├── Họ + Tên
├── Username (3-30 ký tự, chữ/số/underscore)
├── Email
├── Mật khẩu + Xác nhận mật khẩu
        ↓
🔍 Validation:
├── Kiểm tra email hợp lệ (cú pháp + MX records)
├── Kiểm tra email không phải disposable (tempmail, mailinator...)
├── Kiểm tra username chưa tồn tại
├── Kiểm tra email chưa đăng ký
├── Kiểm tra độ mạnh mật khẩu
        ↓
📧 Gửi OTP 5 số đến email
        ↓
💾 Lưu thông tin đăng ký vào session (chưa tạo account)
        ↓

📧 BƯỚC 2: Xác thực OTP
├── Nhập mã OTP từ email
├── Có thể gửi lại OTP
├── Có thể hủy đăng ký
        ↓
✅ Xác thực OTP thành công
        ↓
👤 Tạo tài khoản trong database
        ↓
🔑 Tự động đăng nhập
        ↓
🏠 Chuyển về trang Home
```

#### Yêu cầu mật khẩu mạnh:
```
✅ Tối thiểu 8 ký tự
✅ Có ít nhất 1 chữ thường (a-z)
✅ Có ít nhất 1 chữ IN HOA (A-Z)
✅ Có ít nhất 1 chữ số (0-9)
✅ Có ít nhất 1 ký tự đặc biệt (!@#$%^&*()-_+=)
```

#### Validation Email:
| Kiểm tra       | Mô tả                                       |
|----------------|---------------------------------------------|
| **Cú pháp**    | Đúng định dạng email@domain.com             |
| **Unicode**    | Không chấp nhận ký tự có dấu                |
| **MX Records** | Kiểm tra domain có thể nhận email           |
| **Disposable** | Chặn tempmail, guerrillamail, mailinator... |
| **Trusted**    | Bỏ qua MX check cho gmail.com, yahoo.com... |

#### Templates liên quan:
- `Register.html` — Form đăng ký
- `Verify_email_register.html` — Nhập OTP xác thực

---

### 🔄 Quên mật khẩu (Forgot Password)

**Route:** `/forgot-password/` → `/verify-otp/` → `/reset-password-otp/`

#### Luồng hoạt động (3 bước):
```
📧 BƯỚC 1: Nhập email
├── Nhập email đã đăng ký
        ↓
🔍 Kiểm tra email tồn tại trong hệ thống
        ↓
📧 Gửi OTP 5 số đến email
        ↓

🔢 BƯỚC 2: Xác thực OTP
├── Nhập mã OTP (5 số)
├── Tối đa 5 lần thử sai
├── Có thể gửi lại OTP
        ↓

🔐 BƯỚC 3: Đặt mật khẩu mới
├── Nhập mật khẩu mới (phải đủ mạnh)
├── Xác nhận mật khẩu
        ↓
✅ Cập nhật mật khẩu thành công
        ↓
🔑 Chuyển về trang đăng nhập
```

#### Bảo mật OTP:
| Tính năng      | Chi tiết                                                         |
|----------------|------------------------------------------------------------------|
| **Mã OTP**     | 5 số, tạo bằng `secrets.randbelow()` (an toàn hơn `random`)      |
| **Hash OTP**   | Lưu hash SHA-256 (otp + salt + secret_key), không lưu plain text |
| **Thời hạn**   | Hết hạn sau **10 phút** (TTL index tự động xóa)                  |
| **Số lần thử** | Tối đa **5 lần** sai, sau đó phải yêu cầu OTP mới                |
| **OTP cũ**     | Tự động vô hiệu hóa OTP cũ khi tạo mới                           |

#### Templates liên quan:
- `Forgot_password.html` — Nhập email
- `Verify_otp.html` — Nhập OTP
- `Reset_password_otp.html` — Đặt mật khẩu mới

---

### 📧 Hệ thống Email OTP

#### Cấu hình gửi email (thứ tự ưu tiên):
```
1️⃣ Gmail SMTP (khuyến nghị - ổn định nhất)
    ↓ nếu không có config
2️⃣ Resend API (nếu có RESEND_API_KEY)
    ↓ nếu không có config
3️⃣ Console Mode (in OTP ra terminal - development)
```

#### Cấu hình trong `.env`:
```env
SECRET_KEY=django-insecure-4$t0@wnk+#qu19m66%a90(d10z69tr$-ei@u_pf_%#m5it@=t+
MONGO_URI=mongodb://localhost:27110/Login?directConnection=true
DB_HOST=mongodb+srv://voanhnhat1612:<Nhat@16122006>@cluster0.9xeejj9.mongodb.net/
DB_NAME=Login

DB_USER=Ti-coder
DB_PASSWORD=Nhat@16122006
DB_PORT=27017
DB_ADMIN_EMAIL=voanhnhat1612@gmail.com
DB_AUTH_SOURCE=admin

DB_AUTH_MECHANISM=SCRAM_SHA-1
MAX_FAILED_ATTEMPS=5
LOCKOUT_SECOND=600
RESET_TOKEN_SALT=manager-reset-salt
RESET_TOKEN_EXPIRY_SECONDS=3600
SECRET_KEY=O4qvkC2lzeVn70eOD7qajoMHbZhsV3MPYL2WI8bDhG19pFp1g17_VPQw54bJ0kIzSX9uP49-4mZGXrplf_I6Rg
PASSWORD_PEPPER=yPTp0tlNjhhCmktx_FInwo0bLcu2aquaT3BLVMJaQqw
JWT_SECRET=MHGtW9YsZcP1O04ScNbiOTVMPS-DCS_NKeenFBzaWXzR2Fk7_3xxnT2vubAMIuXNVybtBsCYifEYHxVW6fRnEQ
JWT_ALGORITHM=HS256
JWT_ACCESS_TTL=900
JWT_REFRESH_TTL=604800

USER_NAME_ADMIN=VoAnhNhat
ADMIN_PASSWORD=Nhat@16122006
ADMIN_EMAIL=voanhnhat@zoo.com

ACCESS_TOKEN_EXPIRE_HOURS=3
REFRESH_TOKEN_EXPIRE_DAYS=1
JWT_ISSUER=weather_api
JWT_AUDIENCE=weather_web

# Gmail SMTP - Gui email truc tiep vao Gmail
EMAIL_BACKEND=django.core.mail.backends.smtp.EmailBackend
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_HOST_USER=vohuynhanhtuan0512@gmail.com
EMAIL_HOST_PASSWORD=hsvoefxqomrtrnms
EMAIL_USE_TLS=True
DEFAULT_FROM_EMAIL=VN Weather Hub <vohuynhanhtuan0512@gmail.com>

PASSWORD_RESET_OTP_EXPIRE_SECONDS=600
PASSWORD_RESET_OTP_MAX_ATTEMPTS=5

# Resend API
RESEND_API_KEY=re_hTC5WBm1_4dy31Hk5FEontVHBfMADFEBY
RESEND_FROM_EMAIL=onboarding@resend.dev
```

#### Hướng dẫn lấy Gmail App Password:
1. Vào [Google Account](https://myaccount.google.com/)
2. **Security** → **2-Step Verification** (bật nếu chưa có)
3. **Security** → **App passwords**
4. Tạo app password cho "Mail" + "Windows Computer"
5. Copy mã 16 ký tự vào `EMAIL_HOST_PASSWORD`

### 👤 Hồ sơ cá nhân (Profile)

**Route:** `/profile/`

#### Tính năng:
- Xem thông tin tài khoản (tên, email, username, role)
- Cập nhật họ tên
- Cập nhật email (kiểm tra trùng lặp)
- Xem thời gian đăng ký và đăng nhập cuối

---

### 🔒 Bảo mật hệ thống

#### Password Security:
```python
# Pepper: thêm chuỗi bí mật trước khi hash
hashed = make_password(password + PASSWORD_PEPPER)

# Kiểm tra mật khẩu
check_password(input + PASSWORD_PEPPER, hashed)
```

#### JWT Token:
```python
token = create_access_token({
    "manager_id": "abc123",
    "role": "Staff"
})
```

#### Khóa tài khoản:
```python
if failed_attempts >= 5:
    lock_until = now + 5 phút
    # Tài khoản tạm khóa
```

---

### 🗄️ MongoDB Collections

#### Collection: `logins`
```javascript
{
    "_id": ObjectId("..."),
    "name": "Võ Anh Nhật",
    "userName": "nhat123",
    "email": "nhat@gmail.com",
    "password": "pbkdf2_sha256$...",  // Hashed with pepper
    "role": "Staff",                   // Staff | Manager | Admin
    "is_active": true,
    "failed_attempts": 0,
    "lock_until": null,
    "last_login": ISODate("..."),
    "createdAt": ISODate("..."),
    "updatedAt": ISODate("...")
}
```

#### Collection: `email_verification_otps`
```javascript
{
    "_id": ObjectId("..."),
    "email": "nhat@gmail.com",
    "otpHash": "sha256...",           // Không lưu plain OTP
    "salt": "random_hex",
    "attempts": 0,
    "used": false,
    "createdAt": ISODate("..."),
    "expiresAt": ISODate("..."),      // TTL index tự động xóa
    "verifiedAt": ISODate("...")      // Khi xác thực thành công
}
```

#### Collection: `password_reset_otps`
```javascript
{
    "_id": ObjectId("..."),
    "email": "nhat@gmail.com",
    "otpHash": "sha256...",
    "salt": "random_hex",
    "attempts": 0,
    "used": false,
    "createdAt": ISODate("..."),
    "expiresAt": ISODate("..."),      // TTL index tự động xóa
    "verifiedAt": ISODate("...")
}
```

---

### 📁 Cấu trúc chính của project

```
├── 📁 Weather_Forcast_App
│   ├── 📁 Enums
│   │   ├── 🐍 Enums.py
│   │   └── 🐍 __init__.py
│   ├── 📁 Machine_learning_artifacts
│   │   └── 📁 latest
│   │       ├── ⚙️ Feature_list.json         // Danh sách các cột feature model dùng. Đảm bảo input predict đúng schema, tránh lỗi thiếu/sai thứ tự cột. Là “hợp đồng” giữa features/ và models/
│   │       ├── ⚙️ Metrics.json              // Chỉ số đánh giá lần train gần nhất. Dùng show web, báo cáo, so sánh model
│   │       ├── 📄 Model.pkl                 // Model đã train (pickle/joblib). App chỉ cần load file này để predict, không cần train lại
│   │       └── ⚙️ Train_info.json           // Thông tin cấu hình train: dataset, thời gian, split, thuật toán, hyperparams
│   ├── 📁 Machine_learning_model
│   │   ├── 📁 config
│   │   │   └── ⚙️ default.yaml              // File cấu hình trung tâm: path dataset, target, horizon, model type, params, split rules
│   │   ├── 📁 data
│   │   │   ├── 🐍 Loader.py                 // Load dataset (csv/xlsx) vào DataFrame, xử lý datetime, sort, missing cơ bản
│   │   │   ├── 🐍 Schema.py                 // Định nghĩa “luật dữ liệu”: cột bắt buộc, kiểu dữ liệu, giá trị hợp lệ. Sai báo lỗi rõ
│   │   │   └── 🐍 Split.py                  // Chia train/valid/test. Time series: split theo thời gian, không random
│   │   ├── 📁 evaluation
│   │   │   ├── 🐍 metrics.py                // Định nghĩa các metric: MAE, RMSE, MAPE, R2… Dùng chung cho mọi model
│   │   │   └── 🐍 report.py                 // Xuất báo cáo: bảng so sánh model, lưu biểu đồ, file report csv/json
│   │   ├── 📁 features
│   │   │   ├── 🐍 Build_transfer.py         // Xây features từ raw data: lag, rolling, time features, location features
│   │   │   └── 🐍 Transformers.py           // Module transformer: scaler, encoder, missing, pipeline transform cho train & predict
│   │   ├── 📁 interface
│   │   │   └── 🐍 predictor.py              // Cổng dự báo: load Model.pkl, Feature_list.json, nhận input, build features, predict
│   │   ├── 📁 Models
│   │   │   ├── 🐍 Base_model.py             // Interface chuẩn cho mọi model: fit, predict, save, load, get_params
│   │   │   ├── 🐍 CatBoost.py
│   │   │   ├── 🐍 LightGBM_Model.py
│   │   │   ├── 🐍 Random_Forest_Model.py
│   │   │   └── 🐍 XGBoost_Model.py
│   │   ├── 📁 trainning
│   │   │   ├── 🐍 train.py                  // Tổng chỉ huy train: đọc config, load data, validate, split, build features, train, evaluate, save artifacts
│   │   │   └── 🐍 tuning.py                 // Hyperparameter tuning: grid search, random search, optuna. Output: params tốt nhất
│   │   └── ⚙️ .gitkeep
│   ├── 📁 Merge_data
│   │   ├── 📄 merged_files_log.txt
│   │   └── 📄 merged_vrain_data.xlsx
│   ├── 📁 Models
│   │   ├── 🐍 Login.py
│   │   └── 🐍 __init__.py
│   ├── 📁 Repositories
│   │   ├── 🐍 Login_repositories.py
│   │   └── 🐍 __init__.py
│   ├── 📁 Seriallizer
│   │   └── 📁 Login
│   │       ├── 🐍 Base_login.py
│   │       ├── 🐍 Create_login.py
│   │       ├── 🐍 Update_login.py
│   │       └── 🐍 __init__.py
│   ├── 📁 TEST
│   │   └── ⚙️ .gitkeep
│   ├── 📁 cleaned_data
│   │   ├── 📁 Clean_Data_For_File_Merge
│   │   │   └── 📄 cleaned_merge_merged_vrain_data_20260124_192207.csv
│   │   └── 📁 Clean_Data_For_File_Not_Merge
│   │       ├── 📄 cleaned_output_Bao_cao_20260124_191737_20260124_192237.csv
│   │       ├── 📄 cleaned_output_Bao_cao_20260124_191946_20260124_192226.csv
│   │       └── 📄 cleaned_output_Bao_cao_20260124_191959_20260124_192219.csv
│   ├── 📁 logs
│   │   └── ⚙️ .gitkeep
│   ├── 📁 management
│   │   ├── 📁 commands
│   │   │   ├── 🐍 __init__.py
│   │   │   └── 🐍 insert_first_data.py
│   │   └── 🐍 __init__.py
│   ├── 📁 middleware
│   │   ├── 🐍 Auth.py
│   │   ├── 🐍 Authentication.py
│   │   ├── 🐍 Jwt_handler.py
│   │   └── 🐍 __init__.py
│   ├── 📁 migrations
│   │   └── 🐍 __init__.py
│   ├── 📁 output
│   │   ├── 📄 Bao_cao_20260124_191737.xlsx
│   │   ├── 📄 Bao_cao_20260124_191946.xlsx
│   │   └── 📄 Bao_cao_20260124_191959.csv
│   ├── 📁 runtime
│   │   └── 📁 logs
│   │       └── ⚙️ .gitkeep
│   ├── 📁 scripts
│   │   ├── 🐍 Cleardata.py
│   │   ├── 🐍 Crawl_data_by_API.py
│   │   ├── 🐍 Crawl_data_from_Vrain_by_API.py
│   │   ├── 🐍 Crawl_data_from_Vrain_by_Selenium.py
│   │   ├── 🐍 Crawl_data_from_html_of_Vrain.py
│   │   ├── 🐍 Email_validator.py
│   │   ├── 🐍 Login_services.py
│   │   ├── 🐍 Merge_xlsx.py
│   │   ├── 🐍 __init__.py
│   │   └── 🐍 email_templates.py
│   ├── 📁 static
│   │   └── 📁 weather
│   │       ├── 📁 css
│   │       ├── 📁 img
│   │       └── 📁 js
│   ├── 📁 templates
│   │   └── 📁 weather
│   │       ├── 📁 auth
│   │       ├── 🌐 Dataset_preview.html
│   │       ├── 🌐 Datasets.html
│   │       ├── 🌐 Error.html
│   │       ├── 🌐 HTML_Crawl_data_by_API.html
│   │       ├── 🌐 HTML_Crawl_data_from_Vrain_by_API.html
│   │       ├── 🌐 HTML_Crawl_data_from_Vrain_by_Selenium.html
│   │       ├── 🌐 HTML_Crawl_data_from_html_of_Vrain.html
│   │       ├── 🌐 Home.html
│   │       └── 🌐 Sidebar_nav.html
│   ├── 📁 views
│   │   ├── 🐍 Home.py
│   │   ├── 🐍 View_Clear.py
│   │   ├── 🐍 View_Crawl_data_by_API.py
│   │   ├── 🐍 View_Crawl_data_from_Vrain_by_API.py
│   │   ├── 🐍 View_Crawl_data_from_Vrain_by_Selenium.py
│   │   ├── 🐍 View_Crawl_data_from_html_of_Vrain.py
│   │   ├── 🐍 View_Datasets.py
│   │   ├── 🐍 View_Merge_Data.py
│   │   ├── 🐍 View_login.py
│   │   └── 🐍 __init__.py
│   ├── 🐍 __init__.py
│   ├── 🐍 admin.py
│   ├── 🐍 apps.py
│   ├── 🐍 db_connection.py
│   ├── 🐍 models.py
│   └── 🐍 urls.py
├── 📁 WeatherForcast
│   ├── 🐍 __init__.py
│   ├── 🐍 asgi.py
│   ├── 🐍 settings.py
│   ├── 🐍 urls.py
│   └── 🐍 wsgi.py
```

---
#### 📝 Lưu ý thực tế đã tối ưu và refactor:
- Đã chuẩn hóa toàn bộ đường dẫn artifacts ML về `Machine_learning_artifacts/latest` (không còn hardcode rải rác, chỉ dùng 1 nơi duy nhất cho export/load model, pipeline, metrics, train_info).
- Các module ML (train.py, LightGBM_Model.py, predictor.py) đã tách biệt, mỗi file 1 nhiệm vụ rõ ràng, gọi lẫn nhau qua interface chuẩn, không còn code thừa, không có file rác.
- Nếu muốn mở rộng/thay đổi cấu trúc thư mục, chỉ cần sửa 1 nơi (config hoặc biến LATEST_ARTIFACTS_DIR), không phải sửa nhiều file.
- Đã kiểm tra và loại bỏ hoàn toàn file thừa, file không dùng trong artifacts.
- Đề xuất cấu trúc cây thư mục rõ ràng, tách biệt backend, ML, scripts, data, static, template, artifacts, giúp bảo trì và mở rộng dễ dàng.
        ├── 📁 Merge_data
        │   ├── 📄 merged_files_log.txt
        │   └── 📄 merged_vrain_data.xlsx
        ├── 📁 Models
        │   ├── 🐍 Login.py
        │   └── 🐍 __init__.py
        ├── 📁 Repositories
        │   ├── 🐍 Login_repositories.py
        │   └── 🐍 __init__.py
        ├── 📁 Seriallizer
        │   └── 📁 Login
        │       ├── 🐍 Base_login.py
        │       ├── 🐍 Create_login.py
        │       ├── 🐍 Update_login.py
        │       └── 🐍 __init__.py
        ├── 📁 TEST
        │   └── ⚙️ .gitkeep
        ├── 📁 cleaned_data
        │   ├── 📁 Clean_Data_For_File_Merge
        │   │   └── 📄 cleaned_merge_merged_vrain_data_20260124_192207.csv
        │   └── 📁 Clean_Data_For_File_Not_Merge
        │       ├── 📄 cleaned_output_Bao_cao_20260124_191737_20260124_192237.csv
        │       ├── 📄 cleaned_output_Bao_cao_20260124_191946_20260124_192226.csv
        │       └── 📄 cleaned_output_Bao_cao_20260124_191959_20260124_192219.csv
        ├── 📁 logs
        │   └── ⚙️ .gitkeep
        ├── 📁 management
        │   ├── 📁 commands
        │   │   ├── 🐍 __init__.py
        │   │   └── 🐍 insert_first_data.py
        │   └── 🐍 __init__.py
        ├── 📁 middleware
        │   ├── 🐍 Auth.py
        │   ├── 🐍 Authentication.py
        │   ├── 🐍 Jwt_handler.py
        │   └── 🐍 __init__.py
        ├── 📁 migrations
        │   └── 🐍 __init__.py
        ├── 📁 output
        │   ├── 📄 Bao_cao_20260124_191737.xlsx
        │   ├── 📄 Bao_cao_20260124_191946.xlsx
        │   └── 📄 Bao_cao_20260124_191959.csv
        ├── 📁 runtime
        │   └── 📁 logs
        │       └── ⚙️ .gitkeep
        ├── 📁 scripts
        │   ├── 🐍 Cleardata.py
        │   ├── 🐍 Crawl_data_by_API.py
        │   ├── 🐍 Crawl_data_from_Vrain_by_API.py
        │   ├── 🐍 Crawl_data_from_Vrain_by_Selenium.py
        │   ├── 🐍 Crawl_data_from_html_of_Vrain.py
        │   ├── 🐍 Email_validator.py
        │   ├── 🐍 Login_services.py
        │   ├── 🐍 Merge_xlsx.py
        │   ├── 🐍 __init__.py
        │   └── 🐍 email_templates.py
        ├── 📁 static
        │   └── 📁 weather
        │       ├── 📁 css
        │       │   ├── 🎨 Auth.css
        │       │   ├── 🎨 CSS_Crawl_data_by_API.css
        │       │   ├── 🎨 CSS_Crawl_data_from_Vrain_by_API.css
        │       │   ├── 🎨 CSS_Crawl_data_from_Vrain_by_Selenium.css
        │       │   ├── 🎨 CSS_Crawl_data_from_html_of_Vrain.css
        │       │   ├── 🎨 Dataset_preview.css
        │       │   ├── 🎨 Datasets.css
        │       │   ├── 🎨 Home.css
        │       │   └── 🎨 Sidebar.css
        │       ├── 📁 img
        │       │   ├── 📁 icons
        │       │   │   └── ⚙️ .gitkeep
        │       │   └── 📁 ui
        │       │       ├── 🖼️ Home.png
        │       │       ├── 🖼️ Weather.png
        │       │       ├── 🖼️ cloud.png
        │       │       ├── 🖼️ earth_texture.png
        │       │       ├── 🖼️ sun.png
        │       │       ├── 🖼️ thunder.png
        │       │       ├── 🖼️ tree.png
        │       │       └── 🖼️ water.png
        │       └── 📁 js
        │           ├── 📄 Home.js
        │           ├── 📄 JS_Crawl_data_by_API.js
        │           ├── 📄 JS_Crawl_data_from_Vrain_by_API.js
        │           ├── 📄 JS_Crawl_data_from_Vrain_by_Selenium.js
        │           └── 📄 JS_Crawl_data_from_html_of_Vrain.js
        ├── 📁 templates
        │   └── 📁 weather
        │       ├── 📁 auth
        │       │   ├── 🌐 Forgot_password.html
        │       │   ├── 🌐 Login.html
        │       │   ├── 🌐 Password_reset_complete.html
        │       │   ├── 🌐 Password_reset_sent.html
        │       │   ├── 🌐 Profile.html
        │       │   ├── 🌐 Register.html
        │       │   ├── 🌐 Reset_password.html
        │       │   ├── 🌐 Reset_password_otp.html
        │       │   ├── 🌐 Verify_email_register.html
        │       │   └── 🌐 Verify_otp.html
        │       ├── 🌐 Dataset_preview.html
        │       ├── 🌐 Datasets.html
        │       ├── 🌐 Error.html
        │       ├── 🌐 HTML_Crawl_data_by_API.html
        │       ├── 🌐 HTML_Crawl_data_from_Vrain_by_API.html
        │       ├── 🌐 HTML_Crawl_data_from_Vrain_by_Selenium.html
        │       ├── 🌐 HTML_Crawl_data_from_html_of_Vrain.html
        │       ├── 🌐 Home.html
        │       └── 🌐 Sidebar_nav.html
        ├── 📁 views
        │   ├── 🐍 Home.py
        │   ├── 🐍 View_Clear.py
        │   ├── 🐍 View_Crawl_data_by_API.py
        │   ├── 🐍 View_Crawl_data_from_Vrain_by_API.py
        │   ├── 🐍 View_Crawl_data_from_Vrain_by_Selenium.py
        │   ├── 🐍 View_Crawl_data_from_html_of_Vrain.py
        │   ├── 🐍 View_Datasets.py
        │   ├── 🐍 View_Merge_Data.py
        │   ├── 🐍 View_login.py
        │   └── 🐍 __init__.py
        ├── 🐍 __init__.py
        ├── 🐍 admin.py
        ├── 🐍 apps.py
        ├── 🐍 db_connection.py
        ├── 🐍 models.py
        └── 🐍 urls.py
├── 📁 WeatherForcast
        ├── 🐍 __init__.py
        ├── 🐍 asgi.py
        ├── 🐍 settings.py
        ├── 🐍 urls.py
        └── 🐍 wsgi.py
```

---

### 🚀 API Routes (Authentication)

| Method   | Route                     | Mô tả                 |
|----------|---------------------------|-----------------------|
| GET/POST | `/login/`                 | Đăng nhập             |
| GET/POST | `/register/`              | Đăng ký (bước 1)      |
| GET/POST | `/verify-email-register/` | Xác thực OTP đăng ký  |
| POST     | `/resend-email-otp/`      | Gửi lại OTP đăng ký   |
| GET      | `/cancel-register/`       | Hủy đăng ký           |
| GET      | `/logout/`                | Đăng xuất             |
| GET/POST | `/profile/`               | Hồ sơ cá nhân         |
| GET/POST | `/forgot-password/`       | Quên mật khẩu (bước 1)|
| GET/POST | `/verify-otp/`            | Xác thực OTP (bước 2) |
| GET/POST | `/reset-password-otp/`    | Đặt MK mới (bước 3)   |

---

### 🧪 Development Mode (Console Email)

Khi **không cấu hình email** (không có `EMAIL_HOST_PASSWORD` và `RESEND_API_KEY`), OTP sẽ được in ra terminal:

```
============================================================
📧 [DEVELOPMENT MODE] - OTP sẽ được in ra console
============================================================
📮 Email: test@example.com
👤 Tên: Test User
🎯 Mục đích: đăng ký
🔑 MÃ OTP: 12345
⏱️ Hết hạn sau: 10 phút
============================================================
```

> 💡 **Tip:** Mode này rất hữu ích khi phát triển local hoặc cho bạn bè clone repo test thử mà không cần cấu hình email.

---

## 4. Cấu trúc thư mục dữ liệu

```

📦 vietnam_weather.db
   └─ (DB dữ liệu thời tiết riêng của project – tùy bạn dùng/commit; thường nên ignore nếu là dữ liệu lớn)

⚙️ Dockerfile
   └─ (Build image để chạy project bằng Docker)

⚙️ requirements.txt
   └─ (Danh sách thư viện Python cần cài)

📦 manage.py
   └─ (Entry-point của Django: runserver, migrate, collectstatic, …)

📁 venv/
   └─ (Môi trường ảo Python – ❌ KHÔNG nên đưa lên Git)
      ├─ 📁 bin/ (activate, pip, python, …)
      ├─ 📁 lib/
      └─ 📁 include/

📁 WeatherForcast/                       🧩 (Django project config – “root project”)
   ├─ ⚙️ settings.py                     (Cấu hình Django: INSTALLED_APPS, DB, STATIC, …)
   ├─ ⚙️ urls.py                         (Router tổng: include app urls)
   ├─ ⚙️ asgi.py / wsgi.py               (Serve production / ASGI-WGI entry)
   └─ 📁 __pycache__/                    (cache – ignore)

📁 Weather_Forcast_App/                  🧩 (Django app chính của hệ thống)
   ├─ 📦 apps.py / admin.py / models.py  (App config, admin, models nếu có)
   ├─ ⚙️ urls.py                         (Router của app: datasets, crawl, merge, clean, …)
   ├─ 📁 views/                          🧠 (Controller/Views theo từng chức năng)
   │  ├─ 🧩 Home.py                       (View trang Home)
   │  ├─ 🧩 View_Datasets.py              (Danh sách datasets + view/download + list/clean UI)
   │  ├─ 🧩 View_Merge_Data.py            (Gộp dữ liệu)
   │  ├─ 🧩 View_Clear.py                 (Làm sạch dữ liệu)
   │  ├─ 🧩 View_Crawl_data_by_API.py
   │  ├─ 🧩 View_Crawl_data_from_Vrain_by_API.py
   │  ├─ 🧩 View_Crawl_data_from_Vrain_by_Selenium.py
   │  └─ 🧩 View_Crawl_data_from_html_of_Vrain.py
   │
   ├─ 📁 scripts/                         ⚙️ (Script xử lý dữ liệu – “engine”)
   │  ├─ 🧩 Crawl_data_by_API.py           (Crawl thời tiết bằng API)
   │  ├─ 🧩 Crawl_data_from_Vrain_by_API.py
   │  ├─ 🧩 Crawl_data_from_Vrain_by_Selenium.py
   │  ├─ 🧩 Crawl_data_from_html_of_Vrain.py
   │  ├─ 🧩 Merge_xlsx.py                  (Gộp file xlsx/csv thành dataset chung)
   │  └─ 🧩 Cleardata.py                   (Làm sạch/chuẩn hóa data sau crawl/merge)
   │
   ├─ 🎨 templates/
   │  └─ 🎨 weather/
   │     ├─ 📄 Home.html                   (UI trang Home)
   │     ├─ 📄 Datasets.html               (UI trang Datasets: merged/cleaned/output + modal)
   │     └─ 📄 dataset_preview.html         (UI preview bảng/JSON/text + phân trang/lazy load)
   │
   ├─ 🎨 static/
   │  └─ 🎨 weather/
   │     ├─ 🎨 css/                        (Home.css, Datasets.css, dataset_preview.css, …)
   │     ├─ 🧠 js/                         (Home.js nếu có)
   │     └─ 🖼️ images/                     (nếu bạn có asset)
   │
   ├─ 🗃️ output/                           (Dữ liệu thô sau crawl – “chưa xử lý/hoặc chưa merge”)
   │  ├─ 📦 vietnam_weather_data_YYYYMMDD_HHMMSS.xlsx   (pattern nhiều file)
   │  ├─ 📦 vrain_comprehensive_data_YYYYMMDD_HHMMSS.xlsx
   │  ├─ 📦 luong_mua_thong_ke_selenium_YYYYMMDD_HHMMSS.csv
   │  └─ 📦 Bao_cao_mua_YYYYMMDD_HHMMSS.xlsx
   │
   ├─ 🗃️ Merge_data/                       (Dữ liệu đã gộp – “merge_data”)
   │  ├─ 📦 merged_vrain_data.xlsx
   │  ├─ 📦 merged_weather_data.xlsx
   │  ├─ 📦 merged_vietnam_weather_data.xlsx
   │  ├─ 🧾 merged_files_log.txt
   │  └─ 🧾 merged_vietnam_files_log.txt
   │
   ├─ 🗃️ cleaned_data/                      (Dữ liệu sau làm sạch)
   │  ├─ 🗃️ Clean_Data_For_File_Merge/       (Clean output của nhóm “đã merge”)
   │  └─ 🗃️ Clean_Data_For_File_Not_Merge/   (Clean output của nhóm “chưa merge/output”)
   │
   ├─ 🧾 logs/                               (Log tổng – tùy bạn ghi gì)
   ├─ 🧾 runtime/logs/                        (Log runtime khi chạy job/clean/merge nếu bạn dùng)
   ├─ 🧠 ml_models/                           (Nơi để model/weights/artefact ML – nếu có training)
   ├─ 🧩 services/                            (Business services – nếu bạn tách service layer)
   ├─ 🧪 TEST/                                (Test/nháp thử)
   ├─ 📁 migrations/                          (Migration Django)
   ├─ 📁 __pycache__/                         (cache – ignore)
   └─ 📦 vietnam_weather.db                   (DB bản sao/DB phụ trong app – cân nhắc ignore)

```

---

## 5. Giao diện chính

### 📚 Trang Datasets
- Template: `templates/weather/Datasets.html`
- CSS: `static/weather/css/Datasets.css`
- Các khối chính:
  - Merge datasets (list + “mới nhất”)
  - Clean wizard + cleaned list
  - Output datasets (raw list) + nút merge

### 📄 Trang Dataset Preview
- Template: `templates/weather/dataset_preview.html`
- CSS: `static/weather/css/dataset_preview.css`
- Hiển thị:
  - Header file + loại file + info (folder/size/rows…)
  - Table hoặc text + pagination/load more

---

## 6. Routes / Endpoints

> Dưới đây là những route **đang xuất hiện trong project** (tham chiếu theo tên reverse trong template + list URL pattern từng hiển thị trong debug 404).

### 6.1. Pages
- `home` → trang chủ
- `datasets/` → danh sách dataset (name: `datasets`)
- `datasets/view/<folder>/<filename>/` → xem file (name: `dataset_view`)
- `datasets/download/<folder>/<filename>/` → tải file (name: `dataset_download`)

### 6.2. Crawl modules (đã có trong urls)
- `crawl-api-weather/` (+ logs)
- `crawl-vrain-html/` (+ start/tail)
- `crawl-vrain-api/` (+ start/tail)
- `crawl-vrain-selenium/` (+ start/tail)

> Mỗi nhóm crawl thường có **start/tail** để chạy nền + đọc log tiến trình.

### 6.3. Merge / Clean (được gọi từ template)
- `weather:merge_data` (POST) → chạy gộp dữ liệu
- `weather:clean_list` (GET) → lấy danh sách file theo `source=merge|output` (cho Clean Wizard)
- `weather:clean_data` (POST) → start clean job → trả `job_id`
- `weather:clean_tail` (GET) → poll tiến trình/log/report theo `job_id`

---

## 7. Mapping “folder key”

**dataset_view / dataset_download** nhận 2 tham số: `folder` + `filename`.

Trong `View_Datasets.py`, folder key được map như sau:

| Folder key | Trỏ tới thư mục thực tế |
|---|---|
| `output` | `Weather_Forcast_App/output/` |
| `merged` | `Weather_Forcast_App/Merge_data/` |
| `cleaned` | `Weather_Forcast_App/cleaned_data/` (root) |
| `cleaned_merge` | `Weather_Forcast_App/cleaned_data/Clean_Data_For_File_Merge/` |
| `cleaned_raw` | `Weather_Forcast_App/cleaned_data/Clean_Data_For_File_Not_Merge/` |

---

## 8. Dataset Preview (CSV/Excel/JSON/TXT)

### 8.1. CSV/Excel (table mode)
- `rows_per_page = 100`
- Query param: `?page=N`
- Nếu request là AJAX (`X-Requested-With: XMLHttpRequest`) → trả JSON để frontend render nhanh

### 8.2. JSON (text + highlight)
- Template có script parse JSON và highlight:
  - key / string / number / boolean / null

### 8.3. TXT
- Render plain text trong `<pre>`

---

## 9. Clean Wizard
Clean Wizard trong `Datasets.html` gồm 3 step:

1) **Chọn nguồn** (`merge` hoặc `output`)  
2) **Chọn file** (list có search)  
3) **Chạy job + theo dõi** (poll `clean_tail`)  
   - progress bar
   - log
   - report (rows/missing/duplicates/size)
   - nút xem/tải output file

---

## 10. Merge result modal

Đề xuất hành vi sau khi merge xong:
- Backend trả JSON gồm `latest_merged`:
  - `name`, `size_mb`, `mtime`
  - `view_url`, `download_url`
- Frontend mở modal:
  - bấm xem/tải ngay
  - bấm ✕/ESC để đóng + reload cập nhật danh sách

---

## 11. Cài đặt & chạy

### 11.1. Yêu cầu
- Python 3.x
- Django 3.x
- pandas
- openpyx3

### 11.2 Cấu hình gpu
## 🐧 Linux (Ubuntu/Debian)

### ✅ A) Cài LightGBM CPU (khuyên dùng)
```bash
python -m pip install -U pip setuptools wheel
pip install lightgbm
```

### 🚀 B) Build LightGBM GPU (OpenCL) (NVIDIA/AMD)

#### 1) Cài dependencies
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake python3-dev \
  ocl-icd-opencl-dev opencl-headers clinfo \
  libboost-dev libboost-filesystem-dev libboost-system-dev
```

#### 2) Kiểm tra OpenCL
```bash
clinfo | head -n 40
```
Nếu thấy `Platform` / `Device` (VD: NVIDIA / AMD) là OK.

#### 3) Clone & build LightGBM GPU
```bash
cd ~
git clone --recursive https://github.com/microsoft/LightGBM.git
cd LightGBM

rm -rf build
cmake -B build -S . -DUSE_GPU=1
cmake --build build -j"$(nproc)"
```

#### 4) Cài Python package vào đúng venv (**KHÔNG dùng sudo**)
```bash
python -m pip install -U pip setuptools wheel
./build-python.sh install --precompile
```

---

## 🪟 Windows

### ✅ A) Cài LightGBM CPU (đơn giản nhất)
```powershell
python -m pip install -U pip setuptools wheel
pip install lightgbm
```

### 🚀 B) Build LightGBM GPU (OpenCL) (nâng cao)
> Chỉ nên làm nếu bạn thực sự cần GPU.

Cần cài:
- 🧰 Visual Studio (Desktop development with C++)
- 🧱 CMake
- ⚡ OpenCL runtime / SDK (NVIDIA/AMD/Intel)

Sau đó build theo hướng dẫn chính thức của LightGBM (GPU/OpenCL) tùy theo runtime GPU của bạn.

---

## 🍎 macOS (Intel / Apple Silicon)

### ✅ A) Cài LightGBM CPU (khuyên dùng)
```bash
python -m pip install -U pip setuptools wheel
pip install lightgbm
```

### ⚠️ GPU trên macOS
LightGBM GPU mode dùng **OpenCL** (OpenCL trên macOS đã bị deprecate), nên **không khuyến nghị** build GPU trên macOS.  
Nếu cần tăng tốc: dùng CPU multi-core (`n_jobs=-1`) hoặc chuyển sang Linux/Windows có OpenCL runtime.

---

## 🧪 Verify Installation

Chạy trong Python:
```python
import lightgbm as lgb
print("LightGBM version:", lgb.__version__)
```

---

## ⚙️ Bật GPU trong code (chỉ khi bạn build GPU thành công)

Trong project, nếu bạn có flag `use_gpu=True`, thường sẽ map sang tham số:
- `device_type="gpu"`

Ví dụ:
```python
params = {
  "device_type": "gpu",   # chỉ dùng khi GPU build OK
  "n_jobs": -1,
  "random_state": 42,
}
```

> 💡 Nếu bật `use_gpu=True` mà máy/chưa build GPU đúng chuẩn, bạn có thể thấy warning / fallback / hoặc lỗi — lúc đó để `use_gpu=False` sẽ chạy ổn định trên CPU.

---

## 🧠 Ghi chú nhanh (seed / random_state)

- `random_state: 42` 🎲 là **seed** cho các thao tác ngẫu nhiên (split dữ liệu, sampling, v.v.)
- Mục tiêu: chạy lại nhiều lần sẽ ra kết quả **ổn định hơn** (reproducible).
- Không bắt buộc phải là 42 — chỉ là con số “quen dùng” trong ví dụ.

---

## 🆘 Troubleshooting (lỗi hay gặp)

### 1) `Could NOT find OpenCL ...`
➡️ Cài OpenCL dev headers:
```bash
sudo apt-get install -y ocl-icd-opencl-dev opencl-headers clinfo
```

### 2) `Could NOT find Boost ... filesystem system`
➡️ Cài Boost:
```bash
sudo apt-get install -y libboost-dev libboost-filesystem-dev libboost-system-dev
```

### 3) Đang dùng venv mà lại cài bằng `sudo`
⚠️ Tránh `sudo pip ...` vì sẽ cài vào system Python, dễ lệch môi trường.  
✅ Hãy activate venv rồi chạy:
```bash
python -m pip install lightgbm
```

---


### Cấu hình docker transaction
- Hướng dẫn setting docker để chạy (Setting transaction mongodb)

#### ✅ 1) Kiểm tra Docker trước (dọn tài nguyên nếu bị chiếm port / trùng container)

- Xem container đang chạy: `docker ps`
- Xem tất cả container: `docker ps -a`
- Xoá container (nếu cần): `docker rm -f <container_id_or_name>`
- Xem images: `docker images`
- Xoá images (nếu cần): `docker rmi <image_id>`
- Xem network: `docker network ls`
- Xoá network (nếu cần): `docker network rm <network_name>`

#### ✅ 2) Tạo network riêng cho Mongo Replica Set

```bash
docker network create mongoNet
```

#### ✅ 3) Pull MongoDB image (nếu chưa có)

```bash
docker pull mongo:latest
```

#### ✅ 4) Tạo 3 container chạy chung Replica Set (mongoRepSet)

```bash
docker run -d --name r0 --net mongoNet -p 27108:27017 mongo:latest mongod --replSet mongoRepSet --bind_ip_all --port 27017
docker run -d --name r1 --net mongoNet -p 27109:27017 mongo:latest mongod --replSet mongoRepSet --bind_ip_all --port 27017
docker run -d --name r2 --net mongoNet -p 27110:27017 mongo:latest mongod --replSet mongoRepSet --bind_ip_all --port 27017
```

- Lí do tạo ra 3 container (3 node) là vì replica set thường là 3 nốt để node primary mà hỏng thì cũng còn 2 node secondary vẫn sẽ chạy được, không làm hỏng chương trình.

#### ✅ 5) Initiate Replica Set (chạy trong r0)

- Setting r0 sẽ là primary còn lại là secondary

```bash
docker exec -it r0 mongosh --eval '
rs.initiate({
  _id: "mongoRepSet",
  members: [
    { _id: 0, host: "r0:27017" },
    { _id: 1, host: "r1:27017" },
    { _id: 2, host: "r2:27017" }
  ]
})
'
```

#### ✅ 6) Kiểm tra trạng thái Replica Set

```bash
docker exec -it r0 mongosh --eval 'rs.status().members.map(m=>({name:m.name,stateStr:m.stateStr}))'
```

#### ✅ 7) Vào shell của node primary (r0)

```bash
docker exec -it r0 mongosh
```

- Check trạng thái:

```bash
rs.status()
```

#### ✅ 8) Test ghi database (primary ghi được, secondary sẽ báo lỗi)

Trong `r0`:

```bash
use Login
db.Login.insert({name: "test"})
db.Login.find()
```

Vào `r1` hoặc `r2` và thử insert sẽ thấy báo lỗi (do secondary không cho ghi).

---

### 11.3. Cấu hình env
SECRET_KEY = "..."
MONGO_URI=mongodb://localhost:27108/Login?directConnection=true

### 11.4. Chạy nhanh
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt

python manage.py migrate
python manage.py runserver

Check train: python3 Weather_Forcast_App/Machine_learning_model/trainning/train.py --config Weather_Forcast_App/Machine_learning_model/config/train_config.json
```

---

## 12. Lỗi thường gặp & cách xử lý

### 12.1. 👀 XEM / ⬇️ TẢI bị 404 “File not found”
**Nguyên nhân:** truyền sai folder key (không khớp mapping mục 7).  
**Fix:** dùng đúng key (`output`, `merged`, `cleaned_merge`, `cleaned_raw`, …) hoặc dùng `f.folder`.

### 12.2. “📅 MỚI NHẤT” đúng nhưng list bên dưới không đổi
**Nguyên nhân hay gặp:** template dùng nhầm biến hoặc list lấy từ nguồn khác.  
**Fix checklist:**
- “mới nhất” và list phải cùng nguồn (đều từ `Merge_data`, hoặc đều từ `cleaned_merge`…)
- check lại variable name (ví dụ `latest_merged` vs `latest_cleaned_merge`)
- đảm bảo merge thật sự tạo file trong đúng thư mục (`Merge_data`)

### 12.3. CSS không cập nhật
- File CSS trong template có `?v=...` để cache-busting  
- Nếu vẫn không thấy đổi: hard reload / clear cache

### 12.4. Lỗi docker chưa chạy
- Khởi động docker: docker start r0 r1 r2

---

## 13. Roadmap

<ul>
        <li>📈 <b>Dashboard ML models</b>: Xây dựng dashboard trực quan, biểu đồ dự báo, so sánh các model, export report.</li>
        <li>🔐 <b>Auth/Role</b>: Phân quyền thao tác pipeline (merge/clean/crawl), quản lý user, role, log hoạt động.</li>
        <li>✅ <b>Schema validation</b>: Kiểm tra schema trước khi merge/clean, cảnh báo lỗi, tự động sửa lỗi phổ biến.</li>
        <li>🚀 <b>Deploy</b>: Triển khai Docker/Railway, tích hợp CI/CD, lưu trữ dữ liệu trên S3/MinIO, backup tự động.</li>
        <li>🧠 <b>ML pipeline mở rộng</b>: Thêm module dự báo nâng cao, tuning tự động, tích hợp Optuna, XGBoost, LightGBM, CatBoost.</li>
        <li>🧩 <b>Service layer</b>: Tách biệt business logic, dễ bảo trì, mở rộng.</li>
        <li>🧪 <b>Test/Benchmark</b>: Bổ sung test, benchmark, validate pipeline, đảm bảo chất lượng.</li>
        <li>🌐 <b>API RESTful</b>: Mở rộng API cho frontend/mobile, tích hợp Swagger/OpenAPI.</li>
</ul>

---

## 14. Ghi chú nguồn dữ liệu
Nếu crawl dữ liệu từ bên thứ ba (OpenWeather / vrain / website thống kê…):
- Tôn trọng điều khoản sử dụng (Terms/ToS)
- Rate-limit crawl để tránh gây tải
- Ghi attribution nếu cần

---
👤 Maintainer / Profile Info
  
- 🧑‍💻 Maintainer: Võ Anh Nhật, Dư Quốc Việt, Trương Hoài Tú, Võ Huỳnh Anh Tuần
  
- 🎓 University: UTH
  
- 📧 Email: voanhnhat1612@gmmail.com, vohuynhanhtuan0512@gmail.com, hoaitu163@gmail.com, duviet720@gmail.com
  
- 📞 Phone: 0335052899
  
-  Last updated: 24/12/2006
---
<div align="center">
  <sub>Made with ☕ + ⛈️ — Weather Forecast Project</sub>
</div>

