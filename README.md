<div align="center">

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
- openpyxl

### 11.2. Chạy nhanh
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt

python manage.py migrate
python manage.py runserver
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

---

## 13. Roadmap
- 📈 Dashboard biểu đồ dự báo (ML models)
- 🔐 Auth/Role cho thao tác pipeline (merge/clean/crawl)
- ✅ Schema validation trước khi merge/clean
- 🚀 Deploy (Docker/Railway) + storage (S3/MinIO)

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

