# 🌦️ Weather Forecast – ML Models (Flow theo Source Code)

Tài liệu này mô tả **luồng xử lý (pipeline)** của 4 thuật toán ML trong project dự báo thời tiết, dựa trên các wrapper trong:

`Weather_Forcast_App/Machine_learning_model/Models/`

✅ Models:
- 🌲 Random Forest (scikit-learn)
- 🐱 CatBoost
- 🚀 XGBoost
- 💡 LightGBM

✅ Test script:
- `Weather_Forcast_App/Machine_learning_model/TEST/test_ml_models.py`

---

## 0) Tư duy thiết kế chung (để UI/API gọi thống nhất)

Các model wrapper đều cố gắng chuẩn hóa theo cùng một “API cảm giác giống nhau”:

### A. Task type
- `classification`: dự đoán nhãn (Sunny/Cloudy/Rainy…)
- `regression`: dự đoán số (rain_mm, temperature…)

### B. Trạng thái model
- `UNTRAINED` → `TRAINED` (hoặc `FAILED` nếu train lỗi)

### C. Output chuẩn hoá
**TrainingResult** thường có:
- `success`: True/False
- `metrics`: dict (accuracy/f1… hoặc rmse/mae/r2…)
- `training_time`
- `n_samples`, `n_features`
- `feature_names`
- `feature_importances` (nếu model có)
- `best_iteration` (nếu có early stopping)
- `message`

**PredictionResult** thường có:
- `predictions`
- `probabilities` (chỉ classification + `return_proba=True`)
- `prediction_time`

👉 Mục tiêu: tầng API/UI không cần biết chi tiết từng library.

---

## 1) Pipeline chung (hầu hết model đều theo flow này)

1) **Init**
- trộn params: default + user params  
- set `task_type`, `random_state`, flags (GPU…)
- chuẩn bị metadata: schema feature, encoder, mapping category/datetime…

2) **Prepare Features (X)**
- đảm bảo X là DataFrame
- xử lý NaN/inf
- xử lý datetime (nếu có)
- xử lý categorical (one-hot hoặc native categorical)
- “đóng băng schema” lúc train để predict luôn đúng cột

3) **Prepare Target (y)**
- regression: ép float
- classification: LabelEncoder (string → int), lưu `classes_` để decode ngược

4) **Split train/val**
- time-series: ưu tiên `shuffle=False` để tránh leak theo thời gian
- classification tabular: có thể `shuffle=True` + `stratify=True`

5) **Train**
- fit + early stopping/log (nếu hỗ trợ)

6) **Evaluate**
- regression: RMSE/MAE/R2…
- classification: Accuracy/F1/Precision/Recall…

7) **Predict**
- preprocess X (fit=False) + align schema
- classification decode ngược về label gốc

8) **Save/Load / Export artifacts**
- lưu cả **model + preprocess metadata** để inference ổn định

---

# 🌲 2) Random Forest (`Random_Forest_Model.py`)

## 2.1 Điểm chính
- Dùng sklearn `RandomForestRegressor` / `RandomForestClassifier`
- Hỗ trợ:
  - train + evaluate
  - predict (+ predict_proba cho classification)
  - feature importance
  - cross validate
  - save/load (joblib)

## 2.2 Flow chi tiết

### A) Train
1. Nhận `X, y`
2. Chuẩn hoá X (DataFrame/ndarray → numeric)
3. Nếu classification: đảm bảo y đúng dạng (có thể encode)
4. Split train/val theo `validation_split`
5. Fit model
6. Evaluate (accuracy/f1 hoặc rmse/mae/r2…)
7. Trả `TrainingResult`

### B) Predict
1. Chuẩn hoá X giống lúc train
2. Regression: `predict`
3. Classification:
   - `predict`
   - nếu `return_proba=True` thì gọi `predict_proba`
4. Trả `PredictionResult`

### C) Cross-validation
- sklearn `cross_val_score` (accuracy hoặc r2)

### D) Save/Load
- save: joblib dump model + metadata
- load: khôi phục model + trạng thái `is_trained`

---

# 🐱 3) CatBoost (`CatBoost_Model.py`)

## 3.1 Điểm chính
- Mạnh khi có nhiều **categorical features** dạng string
- Không cần one-hot lớn như XGBoost
- Hỗ trợ:
  - cat_features theo tên cột hoặc index
  - Pool + eval_set + early stopping
  - predict_proba
  - cv + grid_search
  - save/load

## 3.2 Flow chi tiết

### A) Train
1. Chuẩn hoá X (DataFrame)
2. Chuẩn hoá y:
   - regression: float
   - classification: có thể encode / đảm bảo đúng dạng
3. Resolve `cat_features` → index phù hợp CatBoost
4. Tạo `Pool(X, y, cat_features=...)`
5. Nếu có validation: tạo `Pool` val
6. `model.fit(train_pool, eval_set=val_pool, ...)`
7. Evaluate → metrics
8. Trả `TrainingResult`

### B) Predict
- Tạo Pool từ X mới
- Classification:
  - `predict` + `predict_proba` (nếu cần)
- Regression:
  - `predict`

### C) CV + Tuning
- CV: `catboost.cv(...)`
- Tuning: `grid_search(...)`

### D) Save/Load
- save model (thường `.cbm`) + metadata (json) để load/predict ổn định

---

# 🚀 4) XGBoost (`XGBoost_Model.py`)

## 4.1 Điểm “nhạy version”
Bạn đã gặp lỗi kiểu:
- `XGBModel.fit() got an unexpected keyword argument 'eval_metric'`
- `... early_stopping_rounds ...`

👉 Vì **API `fit()` thay đổi theo phiên bản XGBoost** (đặc biệt bản bạn cài là 3.x).
=> Wrapper cần:
- set `eval_metric` trong params / set_params (hoặc callback đúng chuẩn),
- tránh nhét `eval_metric`, `early_stopping_rounds` bừa vào `.fit()`.

## 4.2 Flow xử lý (đúng chuẩn cho wrapper)

### A) Prepare Features (đặc trưng của XGBoost wrapper)
1. Convert X → DataFrame
2. Datetime → tách feature numeric (year/month/day/dow/hour/minute…)
3. Categorical → **one-hot** (`pd.get_dummies`)
4. Lưu `feature_names` lúc train
5. Khi predict: **align schema**
   - thiếu cột → thêm (0 hoặc NaN)
   - dư cột → drop
   - reorder đúng thứ tự train

### B) Prepare Target
- Regression: float
- Classification:
  - LabelEncoder string → int
  - lưu classes để decode dự đoán

### C) Train
1. Preprocess X (fit=True), preprocess y
2. Split train/val (val_size)
3. Init model:
   - regression: `XGBRegressor`
   - classification: `XGBClassifier` (binary/multiclass)
4. Set params tương thích version (eval_metric, early stopping)
5. Fit
6. Evaluate + TrainingResult

### D) Predict
- preprocess X (fit=False) + align schema
- classification:
  - predict_proba nếu cần
  - predict → decode về label gốc
- regression: predict float

### E) Save/Load
- joblib dump:
  - model
  - feature_names
  - label_encoder
  - mapping datetime/categorical config

---

# 💡 5) LightGBM (`LightGBM_Model.py`)

## 5.1 Điểm mạnh wrapper LightGBM của bạn
Wrapper này làm rất kỹ 3 thứ để predict “không lệch cột”:

### A) Schema freeze (`feature_names`)
- Sau khi preprocess xong lúc train, wrapper “đóng băng” danh sách cột
- Lúc predict/evaluate:
  - add missing cols
  - drop extra cols
  - reorder đúng schema

### B) Datetime feature extraction
Nếu có cột datetime:
- convert về datetime
- tách:
  - `*_year, *_month, *_day, *_dow, *_hour, *_minute`
- drop cột datetime gốc
- lưu mapping trong `_datetime_feature_map`

### C) Categorical ổn định bằng `category` + set_categories
- fit=True:
  - cast sang `category`
  - lưu categories list vào `_cat_categories`
- fit=False:
  - cast category
  - `set_categories(train_categories)` để unseen category → NaN (LightGBM xử lý được)

## 5.2 Flow chi tiết

### A) Train
1. `_prepare_features(X, fit=True)`:
   - replace inf → NaN
   - datetime → derived cols
   - categorical → category + lưu categories
   - set `feature_names`
2. `_prepare_target(y, fit=True)`:
   - regression float
   - classification LabelEncoder
3. Split train/val
4. `_init_model(n_classes)`:
   - classification multiclass set `num_class`
5. Fit:
   - dùng callbacks:
     - `lgb.early_stopping(...)`
     - `lgb.log_evaluation(...)`
   - fallback nếu version LightGBM không hỗ trợ callbacks
6. Evaluate + feature importance
7. TrainingResult

### B) Predict
1. `_prepare_features(X, fit=False)`:
   - align schema/categories/datetime
2. predict:
   - regression float
   - classification decode label
3. PredictionResult

### C) Export artifacts “latest”
Tạo folder:
`Weather_Forcast_App/Machine_learning_artifacts/latest/`
- `Model.pkl`
- `Feature_list.json`
- `Metrics.json`
- `Train_info.json`

=> để backend/service inference dùng ngay.

---

## 6) Gợi ý chuẩn cho dữ liệu thời tiết (time-series)
- Nếu data theo thời gian: **shuffle=False**
- Tránh leak: không shuffle trước khi split
- Nếu cần CV time-series: dùng `TimeSeriesSplit` (không dùng CV random)

---

## 7) Run test
```bash
python Weather_Forcast_App/Machine_learning_model/TEST/test_ml_models.py
