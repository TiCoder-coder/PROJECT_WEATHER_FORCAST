# 📁 scripts

## Tổng quan
Thư mục này chứa các script Python phục vụ cho việc crawl dữ liệu, xử lý, gộp file, làm sạch, và các tiện ích backend liên quan đến dữ liệu thời tiết.

---

## 🕸️ Các script Crawl dữ liệu

### `Crawl_data_by_API.py` (~5742 dòng)
Crawl dữ liệu thời tiết từ **3 nguồn API** với fallback tự động:

| Thứ tự | Nguồn | Ghi chú |
|--------|-------|---------|
| 1 | **Open-Meteo** (số 1) | Miễn phí, archive mode 90 ngày, không cần key |
| 2 | **WeatherAPI** (fallback) | API key: `WEATHERAPI_KEY` env var |
| 3 | **OpenWeatherMap** (fallback) | API key: `OPENWEATHER_API_KEY` env var |
| 4 | **Statistical fallback** | `generate_vietnam_statistical_weather()` — dữ liệu tổng hợp từ khí hậu từng vùng VN |

**Chế độ chạy** (`CRAWL_MODE` env var):
- `"continuous"`: Chạy liên tục vòng lặp
- `"manual"`: Chạy một lần

**Output**:
- **42 cột** định nghĩa qua `weather_full_columns()`
- Đường dẫn: `data/data_crawl/` (tự động tính từ `__file__`)
- Định dạng Excel với format đẹp: freeze panes, auto-filter, borders, number format (qua `_format_weather_sheet()`)
- Một file Excel theo tỉnh + file tổng hợp tất cả các tỉnh

**Parallelization**: `ThreadPoolExecutor` cho phép crawl song song các tỉnh

---

### `Crawl_data_from_html_of_Vrain.py` (~416 dòng)
Crawl dữ liệu mưa từ **HTML tĩnh của trang Vrain** bằng Selenium + Regex.

**Phạm vi**: 5 tỉnh Đồng bằng Sông Cửu Long (ĐBSCL) cố định:
- Vĩnh Long, Đồng Tháp, An Giang, Cần Thơ, Cà Mau

**Quy trình**:
1. Lấy ngày/giờ từ trang chủ Vrain
2. Crawl song song 5 URL cố định bằng `ThreadPoolExecutor`
3. Parse HTML bằng Regex
4. Merge với dữ liệu cũ (nếu có), deduplicate, ghi đè file mới nhất

**Output**:
- `data/data_crawl/Bao_cao_YYYYMMDD_HHMMSS.csv` (file tổng hợp)
- `data/data_crawl/Bao_cao_vrain_DBSCL/<tỉnh>_YYYYMMDD.csv` + `.xlsx` (per-province)

> ℹ️ **Airflow scheduler** (`airflow/dags/weather_crawl_schedule.py`) duy nhất sử dụng script này.

---

### `Crawl_data_from_Vrain_by_API.py` (~2738 dòng)
Crawl dữ liệu mưa từ **REST API của Vrain** và lưu vào **SQLite**.

**Cơ sở dữ liệu SQLite** (`vietnam_weather.db`):
- Bảng: `provinces`, `stations`, `weather_data`, `vrain_data`

**Output**:
- SQLite database
- Excel output với styling chuyên nghiệp

**Parallelization**: Multi-threaded

---

### `Crawl_data_from_Vrain_by_Selenium.py` (~417 dòng)
Crawl dữ liệu mưa từ Vrain bằng **Selenium element parsing** (khác với html version dùng Regex).

**Phạm vi**: Cùng 5 tỉnh ĐBSCL như html version (`DBSCL_PROVINCES` dict)

**Output**:
- Per-province CSV + XLSX giống với html version

---

## 🧹 Dữ liệu và xử lý

### `Cleardata.py`
Làm sạch dữ liệu sau crawl/merge:
- Loại NaN, outliers, sai kiểu dữ liệu
- Chuẩn hóa định dạng datetime
- Lưu file `.clean_final.csv`

### `Merge_xlsx.py`
Gộp các file CSV/XLSX từ nhiều nguồn vào dataset chủ:
- Hỗ trợ merge file `.xlsx` và `.csv`
- Deduplicate các dòng trùng
- Lưu log vào `data/data_merge/merged_files_log.txt`

---

## 📬 Tiện ích Auth / Email

### `Email_validator.py`
Kiểm tra hợp lệ email (format, domain, MX record).

### `Login_services.py`
Xử lý đăng nhập, xác thực JWT, refresh token.

### `email_templates.py`
Template email gửi OTP, xác thực đăng ký, reset password.

---

## Cấu trúc thư mục

```
scripts/
├── 🐍 Cleardata.py                      # Làm sạch dữ liệu sau crawl/merge
├── 🐍 Crawl_data_by_API.py              # Crawl 3-nguồn API (Open-Meteo/WeatherAPI/OWM)
├── 🐍 Crawl_data_from_Vrain_by_API.py   # Crawl Vrain qua REST API + SQLite
├── 🐍 Crawl_data_from_Vrain_by_Selenium.py # Crawl Vrain bằng Selenium (element parsing)
├── 🐍 Crawl_data_from_html_of_Vrain.py  # Crawl Vrain bằng Selenium + Regex (dùng bởi Airflow)
├── 🐍 Email_validator.py                # Kiểm tra hợp lệ email
├── 🐍 Login_services.py                 # Dịch vụ đăng nhập/JWT
├── 🐍 Merge_xlsx.py                     # Gộp file XLSX/CSV
└── 🐍 email_templates.py               # Template email OTP/xác thực
```

---

## 🔄 Luồng dữ liệu tổng quan

```
Crawl_data_by_API.py              ─┐
Crawl_data_from_html_of_Vrain.py  ─┤
(Airflow scheduler)                │
Crawl_data_from_Vrain_by_API.py   ─┤──→  data/data_crawl/*.csv/.xlsx
Crawl_data_from_Vrain_by_Selenium ─┘
                                             ↓
                                      Merge_xlsx.py
                                             ↓
                                     data/data_merge/
                                             ↓
                                       Cleardata.py
                                             ↓
                            data/data_clean/*.clean_final.csv
                                             ↓
                                python manage.py train
                                             ↓
                      Machine_learning_artifacts/<model_type>/latest/
```

---

## 👤 Maintainer / Profile Info
- 🧑‍💻 Maintainer: Võ Anh Nhật, Dư Quốc Việt, Trương Hoài Tú, Võ Huỳnh Anh Tuần
- 🎓 University: UTH
- 📧 Email: voanhnhat1612@gmmail.com, vohuynhanhtuan0512@gmail.com, hoaitu163@gmail.com, duviet720@gmail.com
- 📞 Phone: 0335052899

---

## License
MIT License
