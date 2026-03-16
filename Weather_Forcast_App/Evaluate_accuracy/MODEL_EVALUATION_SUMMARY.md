# 🌧️ BÁO CÁO ĐÁNH GIÁ MÔ HÌNH DỰ BÁO LƯỢNG MƯA

> **Ngày cập nhật**: 2026-03-14  
> **Dữ liệu**: 94,128 bản ghi, 10,539 trạm (data crawl 2026-03-14)  
> **Notebook đánh giá**: `Weather_Forcast_App/Evaluate_accuracy/evaluate.ipynb`  
> **Artifacts**: `Weather_Forcast_App/Machine_learning_artifacts/latest/`

---

## 📋 Mục lục

1. [Tổng quan](#1-tổng-quan)
2. [Những gì đã làm](#2-những-gì-đã-làm)
3. [Kết quả hiện tại](#3-kết-quả-hiện-tại)
4. [Cần lưu ý](#4-cần-lưu-ý)
5. [Cần cải thiện](#5-cần-cải-thiện)
6. [Lịch sử thí nghiệm](#6-lịch-sử-thí-nghiệm)
7. [Cấu trúc file quan trọng](#7-cấu-trúc-file-quan-trọng)

---

## 1. Tổng quan

### Kiến trúc mô hình: WeatherEnsembleModel

```
Input (93 features) → Feature Engineering
     → 4 Regressors: XGBoost + LightGBM + CatBoost + RandomForest
     → Average(predictions)
     → Nếu avg ≥ rain_threshold (0.22mm) → báo mưa
     → Nếu avg <  rain_threshold (0.22mm) → output = 0.0
```

- **Dữ liệu**: 94,128 bản ghi, 10,539 trạm, 93 features (40 tạo mới + 53 raw)
- **Split**: 80% train (75,302) / 10% valid (9,412) / 10% test (9,414) — **chronological, sort_by_time=True**
- **Target**: `rain_total` — lượng mưa (mm), zero-inflated ~71.2% (train)
- **Transform**: log1p(target) + StandardScaler(features) + IQR Outlier Clip
- **feature_selection**: Tắt (`feature_selection_enabled=False`)

### Pipeline xử lý

```
MissingValueHandler → OutlierHandler (IQR)
→ CategoricalEncoder → WeatherScaler (StandardScaler)
→ log1p(target) → 4 Regressors → Average → inv_log1p(prediction)
```

---

## 2. Những gì đã làm

### Phase 1 (cũ): Xây dựng baseline với 7,322 mẫu
- Model v1–v3 với 7,322 mẫu — xem Lịch sử thí nghiệm

### Phase 2 (hiện tại): Scale lên 94,128 mẫu (2026-03-14)
- Crawl thêm dữ liệu: 94,128 bản ghi từ 10,539 trạm
- Cấu hình huấn luyện: `config/train_config.json` (default)
- Split chronological 80/10/10 (no shuffle)
- Optuna 25 trials (best_score=-Infinity → dùng initial params)
- **Kết quả**: Model.pkl 5.49 MB, Load 4.6s, RAM 108MB
- **Phát hiện**: Model overfit nặng, Rain Detection thực ra < 50% (precision)

### Đánh giá và bổ sung metrics (2026-03-14)
- Thêm MBE (Mean Bias Error) vào `compute_metrics()`
- Thêm Pearson r vào `compute_metrics()`
- Thêm cell CSI & Frequency Bias (4 ngưỡng mưa)
- Thêm cell đo độ nặng mô hình (disk, RAM, latency)
- Sửa cell kết luận (wrong architecture description, wrong config path)

---

## 3. Kết quả hiện tại

### 3.1 Dataset Statistics

| Thống kê | Train | Valid | Test |
|----------|-------|-------|------|
| Số mẫu | 75,302 (80%) | 9,412 (10%) | 9,414 (10%) |
| Zero% (rain=0) | 71.2% | 83.5% | 51.2% |
| Mean rain | 0.1804 mm | 0.0765 mm | 0.2677 mm |
| Split method | **chronological** (sort_by_time=True, no shuffle) ||||

### 3.2 Regression (Dự báo lượng mưa mm-scale)

| Metric | Train | Valid | Test | Gap (Test-Train) |
|--------|-------|-------|------|-----------------|
| R² | — | — | **-0.0673** | — |
| RMSE (mm) | — | — | **0.3638** | — |
| MAE (mm) | — | — | **0.3499** | — |
| sMAPE (%) | — | — | **129.42%** | — |
| **MBE (mm)** | — | — | *(run evaluate.ipynb)* | — |
| **Pearson r** | — | — | *(run evaluate.ipynb)* | — |

> ⚠️ **R² test = -0.0673**: Model dự báo lượng mưa tệ hơn đoán trung bình đơn giản.  
> sMAPE = 129% cho thấy sai số phần trăm rất lớn — model chủ yếu predict gần 0 cho mọi mẫu.

### 3.3 Classification (Mưa / Không mưa, ngưỡng 0.22mm)

| Metric | Test |
|--------|------|
| TN (đúng: không mưa) | **27** |
| FP (sai: báo mưa nhưng không mưa) | **4,922** |
| FN (sai: bỏ sót mưa) | **12** |
| TP (đúng: có mưa) | **4,429** |
| **Accuracy (Rain Detection)** | **47.45%** |
| **Precision** | **47.36%** |
| **Recall** | **99.73%** |
| **F1-score** | **64.23%** |
| **CSI (≥0.1mm)** | *(run evaluate.ipynb)* |
| **Frequency Bias** | *(run evaluate.ipynb)* |

> ⚠️ **Vấn đề**: Model gần như báo mưa cho MỌI mẫu (Recall=99.73%, Precision=47.4%).  
> Chỉ 27 mẫu TN trong 9,414 mẫu test — model không biết dự báo "không mưa".

### 3.4 CSI & Frequency Bias (4 ngưỡng mưa — chạy evaluate.ipynb để có giá trị)

| Ngưỡng | CSI | Frequency Bias | POD | FAR |
|--------|-----|---------------|-----|-----|
| ≥0.1mm (mưa nhẹ) | *(run)* | *(run)* | *(run)* | *(run)* |
| ≥2.5mm (mưa vừa) | *(run)* | *(run)* | *(run)* | *(run)* |
| ≥7.5mm (mưa to) | *(run)* | *(run)* | *(run)* | *(run)* |
| ≥25mm (mưa rất to) | *(run)* | *(run)* | *(run)* | *(run)* |

### 3.5 Model Size & Performance

| Metric | Giá trị |
|--------|---------|
| Model.pkl disk | **5.49 MB** |
| Total artifacts | **5.52 MB** |
| Load time | **4.606 s** |
| RAM (heap peak) | **108.1 MB** |
| XGBoost sub-model | 0.46 MB |
| LightGBM sub-model | 0.34 MB |
| CatBoost sub-model | 0.18 MB |
| RandomForest sub-model | **4.39 MB** |
| Inference batch=1 | ~88 ms (88,160 µs/row) |
| Inference batch=1000 | ~106 ms → 9,432 rows/s |

### 3.6 Trạng thái mô hình

| Trạng thái | Giải thích |
|------------|-----------|
| **overfit_status**: overfit | Train performance >> Test performance |
| **model_quality**: good | Cấu trúc model OK, cần thêm/chất lượng data |

---

## 4. Cần lưu ý

### 🔴 Quan trọng

1. **Model không phân biệt khô/mưa được**
   - Recall=99.73%, Precision=47.4% → gần như báo mưa cho tất cả
   - Chỉ 27/4949 mẫu không mưa bị phân loại đúng
   - **Nguyên nhân có thể**: Zero ratio quá cao (71% train vs 51% test), hoặc rain_threshold=0.22mm quá thấp

2. **R² test âm (-0.0673)**
   - Model dự báo lượng mưa kém hơn đoán trung bình (ỹ=0.27mm)
   - sMAPE=129% → sai số rất lớn
   - **Nguyên nhân có thể**: Optuna failed (best_score=-Infinity), 25 trials không đủ

3. **Optuna tuning thất bại**
   - `best_score=-Infinity` cho tất cả 4 models
   - 25 trials, nhưng final params dùng từ initial values không tối ưu
   - n_estimators=437, learning_rate=0.190 (học quá nhanh)

4. **Chronological split (no shuffle)**
   - 80% đầu (theo thời gian) = train, 10% giữa = valid, 10% cuối = test
   - Test set ở thời điểm mới nhất — phân phối có thể khác train
   - Zero% test (51.2%) khác hẳn zero% train (71.2%)

### 🟡 Lưu ý kỹ thuật

5. **feature_selection_enabled=False**
   - Không dùng SHAP để chọn feature
   - 93 features = 40 created + 53 raw (không qua lọc)

6. **StandardScaler** (không phải RobustScaler)
   - Sensitive hơn với outliers so với RobustScaler
   - Data đã qua IQR Outlier Handler trước scaler

7. **RandomForest chiếm 4.39/5.49 MB = 80% model size**
   - Nếu muốn giảm kích thước: giảm n_estimators của RF hoặc loại bỏ RF

---

## 5. Cần cải thiện

### Ưu tiên CAO

| # | Vấn đề | Giải pháp | Kỳ vọng |
|---|--------|-----------|---------|
| 1 | Precision quá thấp (47%) | Tăng rain_threshold từ 0.22 → 0.5-1.0mm | Precision > 70% |
| 2 | Optuna tuning thất bại | Debug objective function, chạy lại 100+ trials | Convergence bình thường |
| 3 | R² test âm | Kiểm tra data pipeline, target transform, và objective | R² > 0.5 |
| 4 | learning_rate=0.190 quá cao | Giảm learning_rate → 0.05-0.10 | Tránh overfit |

### Ưu tiên TRUNG BÌNH

| # | Vấn đề | Giải pháp | Kỳ vọng |
|---|--------|-----------|---------|
| 5 | Dữ liệu zero %  train≠test | Cross-validation hoặc stratified split | Metrics ổn định hơn |
| 6 | RandomForest 80% model size | Giảm n_estimators RF hoặc loại bỏ RF | Model.pkl < 2MB |
| 7 | 14 features (93→53 raw) chưa optimal | Bật feature_selection, thử SHAP | Loại bỏ noisy features |

### Dài hạn

| # | Vấn đề | Giải pháp | Ghi chú |
|---|--------|-----------|---------|
| 8 | Thiếu context thời gian | LSTM/Transformer nếu có sequential data | Cần restructure data |
| 9 | Thiếu dữ liệu ngoài | Tích hợp radar, vệ tinh, NWP | API cần license |
| 10 | Mưa to cực ít | Thu thập thêm hoặc augment | Cần verified data |

---

## 6. Lịch sử thí nghiệm

| Version | Dữ liệu | Config | Split | R² Test | Rain Det | Status |
|---------|---------|--------|-------|---------|----------|--------|
| v1 (Baseline) | 7,322 mẫu | 50 trials | 70/15/15 shuffle | ~74% | N/A | Archived |
| v2 (Optuna 100) | 7,322 mẫu | 100 trials | 70/15/15 shuffle | **81.4%** | 94.8% | Archived |
| v3-targeted | 7,322 mẫu | Huber+progressive | 70/15/15 shuffle | 77.1% | ~95% | Archived |
| **current** ✅ | **94,128 mẫu** | **train_config.json** | **80/10/10 chrono** | **-0.067** | **47.45%** | Active (needs tuning) |

> **Lưu ý**: Model hiện tại có kết quả kém hơn do Optuna tuning thất bại và dữ liệu thay đổi.  
> Cần debug và retrain để đạt kết quả tốt hơn.

---

## 7. Cấu trúc file quan trọng

```
PROJECT_WEATHER_FORCAST/
├── config/
│   └── train_config.json                    # ✅ Config hiện tại (default params)
│
├── Weather_Forcast_App/
│   ├── Evaluate_accuracy/
│   │   ├── evaluate.ipynb                   # ✅ Notebook đánh giá (44 cells)
│   │   └── MODEL_EVALUATION_SUMMARY.md      # ← File này
│   │
│   ├── Machine_learning_model/
│   │   ├── trainning/train.py               # Training pipeline
│   │   ├── Models/Ensemble_Model.py         # WeatherEnsembleModel (4 sub-models)
│   │   └── interface/weather_predictor.py   # WeatherPredictor (load + predict)
│   │
│   └── Machine_learning_artifacts/
│       └── latest/
│           ├── Model.pkl                    # 5.49 MB, 4 regressors
│           ├── Feature_list.json            # 93 features (40 created + 53 raw)
│           ├── Metrics.json                 # Saved metrics từ lần train cuối
│           └── Train_info.json              # Training metadata (split, config, etc.)
│
└── dynamic_measurement.ipynb                # Model size measurement (project root)
```

### Cách train lại model

```bash
python manage.py train --config config/train_config.json
```

### Cách evaluate

Mở `Weather_Forcast_App/Evaluate_accuracy/evaluate.ipynb` → Run All Cells

---

> **Lần cập nhật cuối**: 2026-03-14  
> **Thay đổi**: Cập nhật kết quả với 94,128 mẫu, WeatherEnsembleModel (4 sub-models), metrics thực tế, model size
