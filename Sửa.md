# Sửa.md — Tổng hợp các thay đổi (Changelog)

> Cập nhật: 2026-02-25

---

## 1. CatBoost_Model.py — Fix tạo model sai task_type trong Ensemble

**File:** `Weather_Forcast_App/Machine_learning_model/Models/CatBoost_Model.py`

**Vấn đề:** Khi Ensemble gọi `WeatherCatBoost(params=cfg)`, tham số `params` rơi vào `**kwargs` (là dict lồng nhau), dẫn đến:
- `task_type` mặc định = `'classification'` thay vì `'regression'`
- Tạo CatBoostClassifier thay vì CatBoostRegressor
- Hyperparams của user (iterations, depth, learning_rate) bị bỏ qua, dùng DEFAULT

**Lý do sửa:**
Khi Ensemble_Model gọi `_create_model()`, nó truyền config dict qua keyword `params=cfg`. Nhưng `WeatherCatBoost.__init__` chỉ có `**kwargs`, nên cả dict `{"n_estimators": 500, "max_depth": 10, "task_type": "regression", ...}` bị nhét vào kwargs dưới key `"params"` — trở thành dict lồng trong dict. Kết quả:
- `self.task_type` lấy từ `kwargs.get('task_type')` → `None` → default `'classification'`
- CatBoost tạo `CatBoostClassifier` cho bài toán regression → dự đoán sai hoàn toàn
- Tất cả hyperparams user cấu hình (iterations=500, depth=10, learning_rate=0.05) đều bị bỏ qua, dùng DEFAULT_PARAMS

Ngoài ra, user viết config theo cú pháp sklearn (`n_estimators`, `max_depth`, `random_state`) nhưng CatBoost API dùng tên khác (`iterations`, `depth`, `random_seed`). Không map sẽ bị CatBoost bỏ qua hoặc báo lỗi.

**Sửa:**
- Thêm tham số `params: Optional[Dict[str, Any]] = None` vào `__init__`
- Extract `task_type`, `loss_function` từ params dict nếu có
- Map tên params sklearn-style → CatBoost-native:
  - `n_estimators` → `iterations`
  - `max_depth` → `depth`
  - `random_state` → `random_seed`
  - `reg_alpha` → `l2_leaf_reg`
- Loại bỏ các keys CatBoost không hiểu (`reg_lambda`, `subsample`, `colsample_bytree`, …)

---

## 2. Random_Forest_Model.py — Fix tạo model sai task_type trong Ensemble

**File:** `Weather_Forcast_App/Machine_learning_model/Models/Random_Forest_Model.py`

**Vấn đề:** Tương tự CatBoost — Ensemble gọi `WeatherRandomForest(params=cfg)` nhưng class chỉ có `**kwargs`, dẫn đến:
- `task_type` mặc định = `'classification'` → tạo RandomForestClassifier cho bài toán regression
- Hyperparams user (n_estimators=800, max_depth=18) bị ngó lơ

**Lý do sửa:**
Giống hệt bug CatBoost — `WeatherRandomForest.__init__` chỉ có `**kwargs`, nên khi Ensemble truyền `params=cfg`, toàn bộ config dict bị gói thành `kwargs["params"] = {...}` thay vì được unpack. Hậu quả:
- `task_type` = None → default `'classification'` → tạo `RandomForestClassifier` cho bài toán dự đoán lượng mưa (regression)
- RandomForestClassifier predict ra class labels (0, 1, 2...) thay vì giá trị liên tục → metrics sai toàn bộ
- Params user (n_estimators=800, max_depth=18) hoàn toàn bị bỏ qua

Thêm nữa, user config chứa các key dành cho tree boosting (reg_alpha, reg_lambda, learning_rate, subsample) — RandomForest không hỗ trợ các tham số này, truyền vào sẽ gây `TypeError: unexpected keyword argument`.

**Sửa:**
- Thêm tham số `params: Optional[Dict[str, Any]] = None` vào `__init__`
- Extract `task_type` từ params dict
- Loại bỏ keys không hợp lệ cho sklearn RF (`reg_alpha`, `reg_lambda`, `learning_rate`, …)

---

## 3. Ensemble_Model.py — Xóa dead code

**File:** `Weather_Forcast_App/Machine_learning_model/Models/Ensemble_Model.py`

**Lý do sửa:**
Trong `_create_model()`, sau dòng `return model_class(**cfg)` còn có đoạn code xử lý `random_state` → `random_seed` cho CatBoost. Nhưng vì `return` đã kết thúc hàm, đoạn code này **không bao giờ chạy** — nó là dead code gây hiểu nhầm khi đọc (tưởng logic đang hoạt động nhưng thực tế bị skip). Xóa để code sạch hơn và tránh nhầm lẫn khi debug.

**Sửa:** Xóa đoạn code chết sau `return model_class(**cfg)` trong `_create_model()`.

---

## 4. Transformers.py — Fix OutlierHandler IQR=0 phá hủy features

**File:** `Weather_Forcast_App/Machine_learning_model/features/Transformers.py`

**Vấn đề:** Với dữ liệu zero-inflated (≥75% cùng giá trị), IQR=0 → clipping range = [Q1, Q3] = [value, value] → ALL values bị clip thành constant → model nhận toàn features hằng số.

**Lý do sửa:**
Dữ liệu thời tiết có nhiều cột zero-inflated — ví dụ `rain_total` có ~85% giá trị = 0. Khi tính IQR:
- Q1 (25th percentile) = 0, Q3 (75th percentile) = 0 → IQR = Q3 - Q1 = 0
- Clipping range = [Q1 - 1.5*0, Q3 + 1.5*0] = [0, 0]
- **Mọi giá trị** (kể cả mưa thật 5mm, 10mm, 20mm) đều bị clip về 0 → cột trở thành hằng số

Điều này xảy ra với **tất cả 150 features** sau feature engineering (vì data gốc đã bị schema validation drop hầu hết variation). CatBoost nhận 150 cột toàn hằng số → báo lỗi `"All features are either constant or ignored"` → training fail.

Fix bằng cách: khi IQR=0, dùng [min, max] của cột thay vì [Q1, Q3] — giữ nguyên range thực tế của dữ liệu, không clip bất kỳ giá trị nào.

**Sửa:** Trong `OutlierHandler.fit()`, khi `IQR == 0`, fallback dùng `[min, max]` thay vì `[Q1, Q3]`:
```python
if IQR == 0:
    lower = X[col].min()
    upper = X[col].max()
else:
    lower = Q1 - self.iqr_multiplier * IQR
    upper = Q3 + self.iqr_multiplier * IQR
```

---

## 5. Build_transfer.py — Fix bool config crash

**File:** `Weather_Forcast_App/Machine_learning_model/features/Build_transfer.py`

**Vấn đề:** Khi config có `"time_features": true` (bool thay vì dict), code gọi `.get()` trên bool → crash `'bool' object has no attribute 'get'`.

**Lý do sửa:**
User viết config dạng `"time_features": true` (bật feature, dùng default settings). Nhưng code `_get_time_config()` luôn gọi `config.get("hour", True)` — giả sử `config` là dict. Khi `config = True` (bool), Python gọi `True.get("hour")` → crash vì bool không có method `.get()`.

Tương tự cho cả 5 hàm config (lag, rolling, time, location, interaction). Bất kỳ feature nào user set `true` thay vì `{"key": value}` đều sẽ crash. Fix bằng cách kiểm tra `isinstance(config, bool)` — nếu là bool, trả về default config dict; nếu là dict, xử lý bình thường.

Ngoài ra, lag/rolling features cho cột static (latitude, longitude, station_id) tạo ra giá trị hằng số (vì lat/lon không đổi theo thời gian) — chỉ thêm noise cho model. Thêm filter `STATIC_COLUMN_KEYWORDS` để loại bỏ.

**Sửa:** Thêm `isinstance(config, bool)` check cho cả 5 feature config getters:
- `_get_lag_config()`
- `_get_rolling_config()`
- `_get_time_config()`
- `_get_location_config()`
- `_get_interaction_config()`

Thêm `STATIC_COLUMN_KEYWORDS` và `_get_numeric_weather_columns(exclude_static=True)` để lọc bỏ cột static (lat/lon/station) khỏi lag/rolling features.

---

## 6. train.py — Nhiều cải thiện anti-underfitting + fix encoding

**File:** `Weather_Forcast_App/Machine_learning_model/trainning/train.py`

### 6a. Skip schema validation (mới)
- Thêm config option `"skip_schema_validation": true`
- Khi bật: giữ nguyên tất cả rows từ raw data, rename columns cho phù hợp, fill NaN timestamps

**Lý do:** File CSV gốc có 7356 rows, trong đó 1699 rows chứa dữ liệu thời tiết đa dạng (temperature 16.6–33.5°C, humidity khác nhau) nhưng cột `timestamp` = NaN. Schema validation gọi `validate_weather_dataframe()` + `to_flat_dict()` để kiểm tra format → drop hết 1699 rows này vì thiếu timestamp. Còn lại 5657 rows đều từ cùng 1 snapshot (09:36–09:38 ngày 2026-02-16) → tất cả features gần như hằng số → model không thể học gì.

Skip schema giữ lại tất cả 7356 rows, dùng column rename đơn giản thay vì validation phức tạp → data có variation → model train được (R² từ -0.0002 lên 0.53)

### 6b. Feature selection (LightGBM-based)
- `_select_features_by_importance()`: Dùng LGBMRegressor chọn top features theo importance
- Fix: `min_importance=0.0` (từ `1e-5` — giá trị cũ filter hết features)
- Fallback: Nếu <20 features được chọn, lấy top `max_features` bất kể importance

**Lý do:** Feature engineering tạo ~256 features (lag, rolling, time, interaction). Nhiều features là noise. Dùng LightGBM quick-fit để đánh giá importance và chỉ giữ features hữu ích. Bug cũ: `min_importance=1e-5` filter hết vì nhiều features có importance=0 chính xác → trả về 0 features → XGBoost crash `"0 features supplied"`. Fix về 0.0 + fallback top N đảm bảo luôn có features để train

### 6c. Log1p target transformation
- Tự động detect zero-inflated target (khi `zero_ratio > 0.3` và target là rain)
- Apply `np.log1p(y)` trước training, `np.expm1(y_pred)` khi evaluate

**Lý do:** Target `rain_total` có phân bố cực kỳ lệch: ~85% = 0, phần còn lại từ 0.1 đến hàng chục mm. Nếu train trực tiếp, model tối ưu MSE bằng cách predict gần 0 cho mọi sample (vì đa số = 0). `log1p` nén range lớn (0–50mm) thành range nhỏ hơn (0–3.9), giúp model phân biệt giữa mưa nhẹ (0.5mm) và mưa to (20mm) tốt hơn

### 6d. Sample weighting
- Upweight non-zero rain samples (tỉ lệ nghịch với frequency, cap 10x)

**Lý do:** Với 85% samples có rain=0, model chỉ cần predict 0 cho mọi input là đã đạt MSE rất thấp (vì 85% đúng). Nhưng mục đích dự báo là phát hiện KHI NÀO có mưa. Sample weighting tăng "hình phạt" khi model predict sai cho samples có mưa thật (weight 5-10x), buộc model phải học patterns phân biệt mưa/không mưa thay vì lazy predict 0

### 6e. Static feature removal
- `_is_static_derived_feature()` + `_remove_static_derived_features()`
- Loại bỏ lag/rolling/diff features của cột static (lat/lon) — chỉ tạo noise

**Lý do:** Feature engineering tạo lag/rolling/diff cho TẤT CẢ cột numeric, kể cả `latitude`, `longitude`, `station_id`. Nhưng lat/lon của một trạm không đổi → `lag_1_latitude = latitude` luôn, `rolling_3_latitude = latitude` luôn, `diff_1_latitude = 0` luôn. Những features này **100% hằng số** hoặc hoàn toàn trùng cột gốc — chỉ thêm noise và tăng dimensionality vô ích, làm model overfit trên features vô nghĩa

### 6f. Fix UnicodeEncodeError trên Windows
- Thay tất cả print messages tiếng Việt → English ASCII-safe
- Fix emoji `✅` → `[OK]`
- Thêm `encoding="utf-8"` khi đọc Metrics.json
- Thêm try/except UnicodeEncodeError cho diagnostics output

**Lý do:** Windows PowerShell mặc định dùng encoding `cp1252` (Western European). Khi code print tiếng Việt ("Đang tải dữ liệu...") hoặc emoji (✅), Python gọi `sys.stdout.encode('cp1252')` → crash `UnicodeEncodeError: 'charmap' codec can't encode character`. Training chạy đến giữa chừng rồi crash mất kết quả. Fix bằng English ASCII thay vì phụ thuộc vào encoding của terminal

### 6g. Evaluation improvements
- `_evaluate_set()`: Inverse-transform predictions (expm1) trước khi tính metrics
- Thêm `NonZero_MAE`, `NonZero_RMSE`, `Rain_Detection_Accuracy`
- Detect overfit/underfit dựa trên R² score (âm = tệ hơn mean baseline)

**Lý do:** Vì target đã qua log1p transform, predictions cũng ở log-scale. Nếu tính MAE/RMSE trên log-scale → con số nhỏ nhưng vô nghĩa (không phải mm mưa thực tế). Cần `expm1` inverse transform về đơn vị gốc trước khi tính metrics.

Thêm `NonZero_MAE/RMSE` vì overall MAE bị "phình" bởi 85% samples rain=0 (dễ predict đúng). NonZero metrics chỉ đo trên samples có mưa thật — phản ánh đúng khả năng dự đoán lượng mưa. `Rain_Detection_Accuracy` đo % trường hợp model phát hiện đúng có/không mưa (binary threshold > 0.1mm)

---

## 7. Phát hiện vấn đề dữ liệu (DATA QUALITY)

**File dữ liệu:** `cleaned_merge_merged_vrain_data_20260216_121532.csv`

**Vấn đề nghiêm trọng:**
- File chỉ có **3 unique timestamps** (tất cả `2026-02-16 09:36~09:38`) — một snapshot duy nhất
- **5657/7356 rows** có temperature_current=27.9 (cùng giá trị cho tất cả trạm VRAIN)
- **1699 rows** có temperature khác nhau (16.6–33.5°C) nhưng **NaN timestamp** → bị Schema validation drop hết
- Sau schema validation: tất cả features trừ `rain_total` **hằng số** → model không thể học

**Giải pháp tạm:**
- `"skip_schema_validation": true` trong config giữ lại 7356 rows
- Kết quả tốt hơn: R² từ -0.0002 → **0.53** (RandomForest/CatBoost)

**Khuyến nghị:**
- Crawl thêm dữ liệu nhiều ngày/giờ khác nhau (time series thực sự)
- Hoặc dùng dataset có nhiều timestamps hơn để model học được patterns theo thời gian

---

## 8. Kết quả training cuối cùng (skip_schema_validation=true)

| Model | Val R² | Val RMSE | Ghi chú |
|-------|--------|----------|---------|
| XGBoost | — | — | Trained OK |
| RandomForest | 0.528 | 0.391 | Trained 2.06s |
| LightGBM | — | 0.862 | Early stopping |
| CatBoost | 0.531 | 0.389 | Trained 14.76s |
| **Ensemble** | — | **Test RMSE=1.79** | Mean of 4 models |

- Train R²: 0.099 — vẫn underfit do data chỉ có 1 time snapshot
- Cần thêm data (nhiều timestamps) để cải thiện

---

## 9. Các file đã chỉnh sửa (tóm tắt)

| File | Thay đổi |
|------|----------|
| `Models/CatBoost_Model.py` | Thêm `params=` dict, map sklearn params, fix task_type |
| `Models/Random_Forest_Model.py` | Thêm `params=` dict, fix task_type |
| `Models/Ensemble_Model.py` | Xóa dead code |
| `features/Transformers.py` | Fix IQR=0 edge case |
| `features/Build_transfer.py` | Fix bool config crash, static column filter |
| `trainning/train.py` | Skip schema, feature selection, log1p, sample weight, encoding fix |
| `config/train_config.json` | Thêm `skip_schema_validation`, điều chỉnh CatBoost params |
