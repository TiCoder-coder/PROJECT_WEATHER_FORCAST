# 🇻🇳☁️ Vietnam Weather Forecast Crawler

Dự án **thu thập & lưu trữ dữ liệu thời tiết cho các tỉnh/thành phố Việt Nam** từ nhiều nguồn API miễn phí, sau đó **lưu vào SQLite** và **xuất file Excel kèm đánh giá chất lượng dữ liệu**.

![Picture](https://nub.news/api/image/681000/article.png)

---

## 🎯 Mục tiêu

- Thu thập dữ liệu thời tiết **real-time / gần real-time** cho hàng trăm địa điểm trên khắp Việt Nam.
- Kết hợp **nhiều nguồn API** (Open-Meteo, WeatherAPI, OpenWeatherMap, …) và đánh giá **chất lượng dữ liệu**.
- Lưu lại lịch sử dữ liệu để:
  - Phân tích xu hướng
  - Huấn luyện mô hình dự báo
  - Làm báo cáo / bài tập lớn / đồ án

---

## 🏗️ Kiến trúc tổng quan

Project hiện tại tập trung trong file:

- `Craw_data.py` – **file chính**, chứa:
  - Class `SQLiteManager`: quản lý database SQLite
  - Class `VietnamWeatherDataCrawler`: gọi API, xử lý dữ liệu, đánh giá chất lượng, lưu Excel/DB
  - Danh sách `vietnam_locations`: hơn 400+ địa điểm (tỉnh, huyện, thành phố trên cả nước)
  - Hàm `main()`: chạy **1 vòng crawl**
  - Hàm `run_continuously()`: chạy **vòng lặp vô hạn**, cứ 10 phút crawl 1 lần

---

## ⚙️ Công nghệ sử dụng

- **Ngôn ngữ:** Python 3.10
- **Thư viện chính:**
  - `requests` – gọi API HTTP
  - `pandas` – xử lý bảng dữ liệu
  - `numpy` – hỗ trợ tính toán, random
  - `openpyxl` – xuất Excel, style, định dạng
  - `sqlite3` – database nhúng (built-in của Python)
  - `logging` – log tiến trình, lỗi, chất lượng dữ liệu
  - `json`, `datetime`, `time`, `os`, `random` – thư viện chuẩn

---

## 🌐 Nguồn dữ liệu thời tiết

Crawler cố gắng lấy dữ liệu theo thứ tự:

1. **Open-Meteo API**  
2. **WeatherAPI** (cần API key)  
3. **OpenWeatherMap** (cần API key)  
4. **Fallback thống kê nội bộ**  
   - Nếu tất cả API ngoài đều lỗi, hệ thống sẽ sinh dữ liệu “giả lập có kiểm soát” dựa trên:
     - Vị trí (Bắc / Trung / Nam)
     - Tháng trong năm (mùa khô/mưa, mùa đông/hè…)
     - Giờ trong ngày (sáng / trưa / tối / đêm)

Mỗi bản ghi được gắn:

- `data_source`: openmeteo / weatherapi / openweather / statistical  
- `data_quality`: high / medium / low (dựa trên nguồn & logic tự đánh giá)

---

## 🗃️ Cấu trúc database SQLite

📁 File database mặc định: **`vietnam_weather.db`**

Các bảng chính:

### 1. `weather_stations` 🛰️

Lưu thông tin trạm/địa điểm:

- `station_id` – mã trạm (PRIMARY KEY)
- `station_name` – tên hiển thị
- `province` – tỉnh/thành
- `district` – quận/huyện
- `type` – loại trạm (city / district / …)
- `region` – vùng (Đông Bắc Bộ, Đồng bằng sông Cửu Long, …)
- `latitude`, `longitude` – toạ độ
- `created_date` – thời gian tạo

### 2. `weather_data` 🌦️

Lưu dữ liệu thời tiết theo thời gian, ví dụ:

- Thông tin chung:
  - `station_id`, `province`, `district`
  - `timestamp` – thời điểm đo
  - `data_source` – nguồn API
  - `data_quality` – high / medium / low
- Các chỉ số thời tiết chính:
  - Nhiệt độ: `temperature_current`, `temperature_max`, `temperature_min`, `temperature_avg`
  - Độ ẩm: `humidity_current`, `humidity_max`, `humidity_min`, `humidity_avg`
  - Áp suất: `pressure_current`, `pressure_max`, `pressure_min`, `pressure_avg`
  - Gió: `wind_speed_*`, `wind_direction_*`
  - Lượng mưa / mây / tầm nhìn / sấm sét…
- `error_reason` – lý do fallback nếu không gọi được API
- `created_date` – thời gian lưu vào DB

### 3. `data_quality_log` 📈

Lưu lại **báo cáo chất lượng** sau mỗi lần crawl:

- `run_timestamp` – thời gian chạy
- `data_type` – “weather”
- `total_records` – tổng số bản ghi
- `high_quality`, `medium_quality`, `low_quality`
- `high_percent`, `medium_percent`, `low_percent`

---

## 📊 File Excel đầu ra

Mỗi lần `main()` chạy thành công sẽ tạo một file:

- Thư mục: `output/`
- Tên file dạng:  
  `vietnam_weather_data_YYYYMMDD_HHMMSS.xlsx`

Trong file có ít nhất 2 sheet:

1. **`WeatherData`**  
   - Toàn bộ dữ liệu thời tiết đã crawl  
   - Mỗi dòng tương ứng với 1 trạm ở 1 thời điểm

2. **`DataQuality`**  
   - Tổng hợp chất lượng dữ liệu:
     - Số bản ghi high / medium / low
     - % từng loại
   - Có format màu sắc, in đậm, căn giữa để dễ đọc

---

## 🔧 Cài đặt & chạy

### 1️⃣ Tạo virtualenv (khuyến nghị)

  cd PROJECT_WEATHER_FORECAST

  python3 -m venv venv
  source venv/bin/activate     # Linux/macOS
  venv\Scripts\activate        # Windows

### 2️⃣ Chạy để crawl_data

- Đối với chạy thông thường: python PROJECT_WEATHER_FORECAST/Crawl_data_byAPI.py

- Đối với chạy bằng Docker để chạy ngầm:
    docker build -t vietnam-weather-crawler .

    docker run -d \
      --name weather_crawler \
      --restart=always \
      -e CRAWL_MODE=continuous \
      -e OPENWEATHER_API_KEY=.....THAY API OPENWEATHER....... \
      -e WEATHERAPI_KEY=..........THAY WEATHERAPI.......... \
      -v "$(pwd)/output:/app/output" \
      -v "$(pwd)/vietnam_weather.db:/app/vietnam_weather.db" \
      vietnam-weather-crawler

  ---
  👤 Maintainer / Profile Info
  
    - 🧑‍💻 Maintainer: Võ Anh Nhật, Dư Quốc Việt, Trương Hoài Tú, Võ Huỳnh Anh Tuần
  
    - 🎓 University: UTH
  
    - 📧 Email: voanhnhat1612@gmmail.com, vohuynhanhtuan0512@gmail.com, hoaitu163@gmail.com, duviet720@gmail.com
  
    - 📞 Phone: 0335052899
  
    -  Last updated: 24/12/2006