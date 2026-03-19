# 🌧️ BÁO CÁO ĐÁNH GIÁ MÔ HÌNH DỰ BÁO LƯỢNG MƯA

> **Ngày cập nhật**: 2026-03-19  
> **Dữ liệu**: 112,648 bản ghi từ `merged_vrain_data_cleaned_20260319_225913.clean_final.csv`  
> **Notebook đánh giá**: `Weather_Forcast_App/Evaluate_accuracy/evaluate.ipynb`  
> **Artifacts**: `Weather_Forcast_App/Machine_learning_artifacts/ensemble_average/latest/` và `stacking_ensemble/latest/`

---

## 📋 Mục lục

1. [Tổng quan](#1-tổng-quan)
2. [Những gì đã làm](#2-những-gì-đã-làm)
3. [Kết quả hiện tại — Ensemble Average](#3-kết-quả-hiện-tại--ensemble-average)
4. [Kết quả hiện tại — Stacking Ensemble](#4-kết-quả-hiện-tại--stacking-ensemble)
5. [So sánh 2 mô hình](#5-so-sánh-2-mô-hình)
6. [Cần lưu ý](#6-cần-lưu-ý)
7. [Lịch sử thí nghiệm](#7-lịch-sử-thí-nghiệm)
8. [Cấu trúc file quan trọng](#8-cấu-trúc-file-quan-trọng)

---

## 1. Tổng quan

### Kiến trúc mô hình 1: Ensemble Average (`WeatherEnsembleModel`)

```
Input (68 features) → log1p(target) → Feature Engineering
     → 4 Regressors: XGBoost + LightGBM + CatBoost + RandomForest
     → Average(predictions) → expm1(prediction)
     → Nếu avg ≥ rain_threshold (0.22mm) → báo mưa
     → Nếu avg <  rain_threshold (0.22mm) → output = 0.0
```

- **Split**: 80% train / 10% valid / 10% test — **chronological, sort_by_time=True**
- **Target**: `rain_total` (mm) — log1p transform áp dụng **bên ngoài**
- **Pipeline**: MissingValueHandler → OutlierHandler (IQR) → CategoricalEncoder → WeatherScaler (StandardScaler)
- **Thresholds**: `rain_threshold=0.22`, `predict_threshold=0.45`

### Kiến trúc mô hình 2: Stacking Ensemble (`WeatherStackingEnsembleModel`)

```
Input (68 features, log1p KHÔNG áp dụng bên ngoài)
    ↓
Stage 1: 8 Base Models (OOF, n_splits=8)
    ├── XGB_cls / RF_cls / CatBoost_cls / LightGBM_cls  → P(rain)
    └── XGB_reg / RF_reg / CatBoost_reg / LightGBM_reg  → amount (log1p space)
    ↓
Stage 2: 2 Meta-LightGBM (trained on OOF predictions)
    ├── meta_cls → P(rain) → threshold 0.4 → binary prediction
    └── meta_reg → rain amount (mm, expm1 applied)
    ↓
Schema Bank Routing (per rain_intensity × season)
→ Chọn base model phù hợp nhất theo ngữ cảnh
```

- **OOF samples**: 90,094 (cls) / 47,200 (reg)
- **Thresholds**: `predict_threshold=0.4`, `rain_threshold=0.1`, `seed=42`
- **log1p xử lý nội bộ** — không cần transform bên ngoài

---

## 2. Những gì đã làm

### Phase 1 (cũ): Baseline với 7,322 mẫu
- Model v1–v3 với 7,322 mẫu — xem Lịch sử thí nghiệm

### Phase 2 (2026-03-14): Scale lên 94,128 mẫu
- Crawl thêm dữ liệu: 94,128 bản ghi từ 10,539 trạm
- Phát hiện model overfit nặng, Rain Detection < 50% (precision)
- Thêm: MBE, Pearson r, CSI, Frequency Bias, model size metrics
- Kết quả: R² test = -0.0673, Rain Detection = 47.45%

### Phase 3 (2026-03-19): Retrain với 112,648 mẫu
- Dataset mới: `merged_vrain_data_cleaned_20260319_225913.clean_final.csv` (112,648 rows)
- Giảm features từ 93 → 68 (loại 18 constant features, feature selection)
- Train **Ensemble Average** → `ensemble_average/latest/`
- ROC-AUC và PR-AUC cell thêm vào evaluate.ipynb

### Phase 4 (2026-03-19): Stacking Ensemble — đã implement và train
- Thiết kế `WeatherStackingEnsembleModel` với 8 base + 2 meta-LightGBM
- OOF cross-validation (n_splits=8) cho calibrated predictions
- Schema bank routing theo rain_intensity (no_rain/light/moderate/heavy/very_heavy) × season (rainy/dry)
- Kết quả: **GOOD FIT** — F1 gap train-valid = 0.035 (so với 0.234 trước khi tuning)
- Artifacts lưu tại `stacking_ensemble/latest/`

---

## 3. Kết quả hiện tại — Ensemble Average

> **Trained**: 2026-03-19 23:00:28 | **Thời gian train**: 15.1s

### 3.1 Regression

| Metric | Train | Valid | Test |
|--------|-------|-------|------|
| **R²** | **0.9923** | 0.7447 | 0.5262 |
| **RMSE (mm)** | 0.3724 | 2.5576 | 3.0413 |

### 3.2 Classification (Rain Detection)

| Metric | Train | Valid | Test |
|--------|-------|-------|------|
| **Rain_F1** | **0.9390** | 0.8469 | 0.8380 |
| **Rain_Detection_Accuracy** | 0.9319 | 0.7723 | 0.7483 |
| **ROC_AUC** | 0.9976 | 0.8647 | 0.8410 |
| **PR_AUC** | 0.9973 | 0.9127 | 0.9159 |

### 3.3 Diagnostics

| Trạng thái | Chi tiết |
|------------|---------|
| **overfit_status: overfit** | RainAcc Train(0.932) > Valid(0.772) gap = **0.160** |

---

## 4. Kết quả hiện tại — Stacking Ensemble

> **Trained**: 2026-03-19 23:12:06 | **Thời gian train**: 48.82s

### 4.1 Regression

| Metric | Train | Valid | Test |
|--------|-------|-------|------|
| **R²** | 0.8194 | 0.7663 | **0.5587** |
| **RMSE (mm)** | 1.8013 | 2.4472 | **2.9350** |

### 4.2 Classification (Rain Detection)

| Metric | Train | Valid | Test |
|--------|-------|-------|------|
| **Rain_F1** | 0.8998 | 0.8647 | **0.8476** |
| **Rain_Detection_Accuracy** | 0.8884 | 0.8156 | **0.7874** |
| **ROC_AUC** | 0.8983 | 0.8170 | 0.7875 |
| **PR_AUC** | 0.8787 | 0.8603 | 0.8579 |

### 4.3 Diagnostics

| Trạng thái | Chi tiết |
|------------|---------|
| **overfit_status: good** | Rain_F1 gap train-valid = **0.035** ✅ |

---

## 5. So sánh 2 mô hình

| Metric (Test) | Ensemble Average | Stacking Ensemble | Winner |
|---------------|-----------------|-------------------|--------|
| R² | 0.5262 | **0.5587** | Stacking |
| RMSE (mm) | 3.0413 | **2.9350** | Stacking |
| Rain_F1 | 0.8380 | **0.8476** | Stacking |
| Rain_Detection_Acc | 0.7483 | **0.7874** | Stacking |
| ROC_AUC | **0.8410** | 0.7875 | Ensemble Avg |
| PR_AUC | **0.9159** | 0.8579 | Ensemble Avg |
| Thời gian train | **15.1s** | 48.82s | Ensemble Avg |
| Overfit status | ⚠️ overfit | ✅ **good** | Stacking |

> **Nhận xét**: Stacking Ensemble tốt hơn về độ chính xác dự báo (R², RMSE, Rain_F1, RainAcc) và FIT tốt hơn nhiều (F1 gap 0.035 vs 0.160). Ensemble Average có ROC/PR-AUC cao hơn nhưng dấu hiệu overfit rõ rệt. Stacking là model khuyến nghị cho production.

---

## 6. Cần lưu ý

### 🟡 Ensemble Average

1. **overfit_status = 'overfit'** — Gap RainAcc train-valid = 0.160 (vượt ngưỡng 0.15)
2. **Test R² = 0.526** — Chưa lý tưởng, còn room cải thiện với feature engineering nâng cao
3. **log1p applied externally** — Cần nhớ khi gọi predict, pipeline xử lý ngoài

### 🟢 Stacking Ensemble

1. **GOOD FIT** — F1 gap = 0.035, generalization tốt
2. **log1p xử lý nội bộ** — Không cần transform trước khi gọi predict
3. **Schema bank routing** — Có thể cần calibration thêm với từng vùng địa lý
4. **Train chậm hơn** — 48.82s vs 15.1s (do OOF cross-validation n_splits=8)

---

## 7. Lịch sử thí nghiệm

| Version | Dữ liệu | Config | Split | R² Test | Rain_F1 Test | Overfit | Status |
|---------|---------|--------|-------|---------|--------------|---------|--------|
| v1 (Baseline) | 7,322 mẫu | 50 trials | 70/15/15 shuffle | ~74% | N/A | ? | Archived |
| v2 (Optuna 100) | 7,322 mẫu | 100 trials | 70/15/15 shuffle | **81.4%** | 94.8% | ? | Archived |
| v3-targeted | 7,322 mẫu | Huber+progressive | 70/15/15 shuffle | 77.1% | ~95% | ? | Archived |
| v4 (94k rows) | 94,128 mẫu | train_config.json | 80/10/10 chrono | -0.067 | 64.23% | overfit | Archived |
| **v5 — Ensemble Average** ✅ | **112,648 mẫu** | train_config.json | 80/10/10 chrono | **0.526** | **0.838** | ⚠️ overfit | **Active** |
| **v6 — Stacking Ensemble** ✅ | **112,648 mẫu** | train_config.json (stacking) | 80/10/10 chrono | **0.559** | **0.848** | ✅ good | **Active** |

---

## 8. Cấu trúc file quan trọng

```
PROJECT_WEATHER_FORCAST/
├── config/
│   └── train_config.json                    # Config huấn luyện (ensemble + stacking)
│
├── Weather_Forcast_App/
│   ├── Evaluate_accuracy/
│   │   ├── evaluate.ipynb                   # Notebook đánh giá (44+ cells)
│   │   ├── tremblingProcess.ipynb           # Full ML pipeline (crawl→merge→clean→train→forecast)
│   │   └── MODEL_EVALUATION_SUMMARY.md      # ← File này
│   │
│   ├── Machine_learning_model/
│   │   ├── trainning/train.py               # Training pipeline
│   │   ├── Models/Ensemble_Average_Model.py # WeatherEnsembleModel (4 sub-models, soft voting)
│   │   ├── Models/Ensemble_Stacking_Model.py # WeatherStackingEnsembleModel (8 base + 2 meta)
│   │   └── interface/weather_predictor.py   # WeatherPredictor (load + predict)
│   │
│   └── Machine_learning_artifacts/
│       ├── ensemble_average/latest/
│       │   ├── Model.pkl                    # Ensemble Average model
│       │   ├── Feature_list.json            # 68 features
│       │   ├── Metrics.json                 # Metrics (overfit_status='overfit')
│       │   └── Train_info.json              # Training metadata
│       └── stacking_ensemble/latest/
│           ├── Model.pkl                    # Stacking Ensemble model
│           ├── Feature_list.json            # 68 features
│           ├── Metrics.json                 # Metrics (overfit_status='good')
│           └── Train_info.json              # n_splits=8, predict_threshold=0.4, OOF info
│
└── dynamic_measurement.ipynb                # Model size measurement
```

### Cách train lại model

```bash
# Ensemble Average
python manage.py train --config config/train_config.json --model ensemble

# Stacking Ensemble
python manage.py train --config config/train_config.json --model stacking
```

### Cách evaluate

Mở `Weather_Forcast_App/Evaluate_accuracy/evaluate.ipynb` → Run All Cells

---

> **Lần cập nhật cuối**: 2026-03-19  
> **Thay đổi**: Retrain với 112,648 mẫu; thêm Stacking Ensemble (GOOD FIT); cập nhật metrics thực tế cả 2 mô hình


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
