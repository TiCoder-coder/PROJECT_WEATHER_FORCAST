# CHANGELOG — Weather Forecast Project

> **Ngày**: 07/03/2026
> **Phiên làm việc**: Xây dựng pipeline notebook, sửa lỗi, kiểm tra toàn bộ dự án

---

## 📁 File mới tạo

### 1. `Weather_Forcast_App/Evaluate_accuracy/tremblingProcess.ipynb`
- **Mục đích**: Notebook tổng hợp toàn bộ pipeline ML — từ crawl dữ liệu đến dự báo.
- **Nội dung** (31 cells):
  - **Cell 1**: Tiêu đề notebook
  - **Cell 2**: Flowchart Mermaid (`graph TD`) mô tả toàn bộ pipeline 6 phase
  - **Cell 4**: Setup cell — định nghĩa `PROJECT_ROOT`, hàm `run_script()` chạy subprocess
  - **Phase 1** — Crawl dữ liệu: gọi `scripts/crawl_weather_data.py`
  - **Phase 2** — Merge CSV: gọi `scripts/merge_csv.py`
  - **Phase 3** — Làm sạch dữ liệu (4 bước):
    - Bước 1: ~~Đổi tên cột~~ → **BỎ QUA** (Cleardata đã làm)
    - Bước 2: Chạy `Cleardata.py` (clean chính)
    - Bước 3: ~~Clean thêm~~ → **BỎ QUA** (trùng lặp)
    - Bước 4: Clean NaN datetime (`clean_datetime_nan.py`)
  - **Phase 4** — Cấu hình & Train model: cập nhật `train_config.json`, gọi `manage.py train_model`
  - **Phase 5** — Dự báo: gọi `WeatherForcast.py`, output ra `data/data_forecast/forecast_results.csv`
  - **Phase 6** — Diagnostics: gọi `scripts/run_diagnostics.py`
- **Lưu ý kỹ thuật**:
  - Dùng `subprocess.run()` thay vì `!python` để tương thích mọi môi trường
  - Flowchart trải qua 4 lần thiết kế lại, phiên bản cuối dùng `graph TD` thuần ASCII

### 2. `data/data_forecast/` (thư mục mới)
- Chứa output dự báo: `forecast_results.csv` (13,431 dòng)

---

## ✏️ File đã chỉnh sửa

### 3. `Weather_Forcast_App/Machine_learning_model/Models/TwoStage_Model.py`
- **Vấn đề**: `UnicodeEncodeError` trên Windows do console encoding `cp1252` không hỗ trợ emoji và ký tự đặc biệt
- **Sửa đổi** (dòng ~255):
  | Trước | Sau |
  |-------|-----|
  | `🌧️  Two-Stage Model — Training` | `[RAIN] Two-Stage Model - Training` |
  | `──` (box-drawing) | `--` (ASCII) |
  | `R²` (superscript) | `R2` (plain) |

### 4. `Weather_Forcast_App/management/commands/train_model.py`
- **Vấn đề**: Cùng lỗi `UnicodeEncodeError` với ký tự `──` và `R²`
- **Sửa đổi**:
  | Dòng | Trước | Sau |
  |------|-------|-----|
  | ~116 | `── Metrics ──────` | `-- Metrics --` |
  | ~125 | `R²=` | `R2=` |
  | ~131 | `── Diagnostics ──────` | `-- Diagnostics --` |

### 5. `Weather_Forcast_App/Machine_learning_model/WeatherForcast/WeatherForcast.py`
- **Sửa đổi** (dòng ~635): Cập nhật default output path
  ```python
  # Trước
  default=ROOT / "forecast_results.csv"
  # Sau
  default=ROOT / "data" / "data_forecast" / "forecast_results.csv"
  ```

### 6. `Weather_Forcast_App/Machine_learning_model/config/train_config.json`
- **Sửa đổi**: Cập nhật tên file dữ liệu
  ```json
  // Trước
  "filename": "merged_vrain_data_cleaned_..."
  // Sau
  "filename": "merged_vrain_data_cleaned_20260307_223506.clean_final.csv"
  ```
- `target_column`: `rain_total`
- `model.type`: `two_stage`

### 7. `Weather_Forcast_App/scripts/Cleardata.py`
- **Vấn đề**: `TypeError` khi serialize `pandas.Timestamp` sang JSON
- **Sửa đổi**: Thêm xử lý convert `Timestamp` → string trong quá trình xuất JSON report

### 8. `Weather_Forcast_App/Machine_learning_model/trainning/train.py`
- **Vấn đề**: Import `shap` gây lỗi khi chưa cài
- **Sửa đổi**: Wrap import `shap` trong `try/except` để optional

### 9. `requirements.txt`
- **Thêm mới** (cuối file):
  ```
  seaborn==0.13.2
  scikit-learn==1.8.0
  shap==0.51.0
  xgboost==3.2.0
  catboost==1.2.10
  lightgbm==4.6.0
  ```

---

## 🔍 Kiểm tra toàn bộ dự án (Health Check)

### Django
- `python manage.py check` → **OK**, không có lỗi
- **44 URL patterns** đã đăng ký
- Middleware: `JWTAuthentication` hoạt động
- Model: `LoginModel` (MongoDB)

### Machine Learning
- **5 model types**: RandomForest, XGBoost, LightGBM, CatBoost, TwoStage
- Training kết quả:
  - R2 = **0.70** (test set)
  - Rain Detection Accuracy = **93.7%**
  - Thời gian train: ~2 giây
- Dự báo: **13,431 dòng**, RMSE = 3.61

### Cấu trúc thư mục `data/`
```
data/
├── data_crawl/       → 12 file CSV gốc
├── data_merge/       → 2 file merged + log
├── data_clean/
│   └── data_merge_clean/  → 21 file đã clean
└── data_forecast/    → forecast_results.csv (MỚI)
```

### Artifacts
```
Machine_learning_artifacts/latest/
├── Feature_list.json
├── Metrics.json
└── Train_info.json
```

---

## 📋 Tóm tắt

| Hạng mục | Số lượng |
|----------|----------|
| File mới tạo | 1 notebook + 1 thư mục |
| File chỉnh sửa | 7 file |
| Thư viện thêm mới | 6 packages |
| Lỗi Unicode đã sửa | 3 file (5 vị trí) |
| Đường dẫn đã cập nhật | 2 file |
| Pipeline phases | 6 (Crawl → Merge → Clean → Train → Predict → Diagnostics) |
