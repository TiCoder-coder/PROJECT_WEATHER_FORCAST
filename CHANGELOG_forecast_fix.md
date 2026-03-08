# Sửa lỗi phần Dự báo thời tiết (Forecast) — 2026-03-08

## Tổng quan

Kiểm tra và sửa lỗi logic dự báo trong `Weather_Forcast_App/views/View_Predict.py` để đảm bảo model dự báo đúng **24 giờ tương lai** theo thiết kế.

---

## Xác nhận pipeline training đã đúng

- Config `forecast_horizon: 24` → khi train, target được shift `-24` bước (`df[target].shift(-24)`)
- Model học: **features tại thời điểm t** → **lượng mưa tại t + 24h**
- Các cột rò rỉ dữ liệu (`rain_current`, `rain_avg`, `rain_max`, `rain_min`) đã bị loại khi train

**File liên quan:** `Weather_Forcast_App/Machine_learning_model/trainning/train.py` (hàm `_build_features_for_split`)

---

## Lỗi #1: Timestamp bị gán sai TRƯỚC khi predict (BUG NGHIÊM TRỌNG)

**File:** `Weather_Forcast_App/views/View_Predict.py` — hàm `_forecast_now_worker`

**Trước khi sửa:**
```python
# Gán timestamp = now + 24h TRƯỚC khi predict
forecast_dt = now + timedelta(hours=forecast_horizon)
df["timestamp"] = forecast_dt.strftime(...)

# Feature builder tạo time features (hour, month, day_of_week)
# dựa trên timestamp TƯƠNG LAI → SAI!
result = predictor.predict(df)
```

**Vấn đề:** Feature builder tạo các time features (`hour`, `month`, `day_of_week`, `sin/cos encodings`) dựa trên cột `timestamp`. Khi gán timestamp thành thời gian tương lai trước khi predict, model nhận input sai — vì model đã train với features tại thời điểm hiện tại.

**Sau khi sửa:**
```python
# Giữ timestamp = thời điểm hiện tại cho feature builder
df["timestamp"] = now.strftime(...)

# Predict đúng: features tại t → dự báo rain tại t+24h
result = predictor.predict(df)

# SAU khi predict, gán thời điểm dự báo để hiển thị
forecast_dt = now + timedelta(hours=forecast_horizon)
df["forecast_for"] = forecast_dt.strftime(...)
df["data_collected_at"] = now.strftime(...)
```

---

## Lỗi #2: `predict_manual_view` thiếu thông tin forecast horizon

**File:** `Weather_Forcast_App/views/View_Predict.py` — hàm `predict_manual_view`

**Trước khi sửa:**
```python
return JsonResponse({
    "ok": True,
    "predictions": response_rows,
    "stats": { ... }
})
```

**Vấn đề:** Response không cho user biết kết quả dự báo là cho 24h tương lai hay thời điểm hiện tại.

**Sau khi sửa:**
```python
return JsonResponse({
    "ok": True,
    "predictions": response_rows,
    "forecast_info": {
        "forecast_horizon_hours": 24,
        "data_at": "2026-03-08 15:00:00",
        "forecast_for": "2026-03-09 15:00:00",
        "description": "Dự báo lượng mưa sau 24 giờ tới",
    },
    "stats": { ... }
})
```

Mỗi row trong `predictions` cũng được thêm trường `forecast_for`.

---

## Lỗi #3: Threshold phân loại mưa bị hardcode sai

**File:** `Weather_Forcast_App/views/View_Predict.py` — hàm `_forecast_now_worker` và `predict_manual_view`

**Trước khi sửa:**
```python
threshold = 0.5  # Hardcode
df["status"] = np.where(predictions > threshold, "Mưa", "Không mưa")
```

**Vấn đề:** Model `two_stage` đã được Optuna tuning với `predict_threshold = 0.449`. Dùng 0.5 làm mất một phần trường hợp mưa.

**Sau khi sửa:**
```python
predict_threshold = predictor.train_info.get("model", {}).get("params", {}).get("predict_threshold", 0.5)
df["status"] = np.where(predictions > predict_threshold, "Mưa", "Không mưa")
```

---

## Lỗi phụ: `_forecast_now_worker` dùng `rain_total` thực tế để gán status

**Trước khi sửa:**
```python
if "rain_total" in df.columns:
    df["status"] = df["rain_total"].apply(lambda x: "Mưa" if x > 0 else "Không mưa")
```

**Vấn đề:** Khi crawl dữ liệu mới, `rain_total` là lượng mưa **hiện tại** — không phải tương lai. Dùng nó để gán status cho dự báo tương lai là vô nghĩa.

**Sau khi sửa:** Status luôn dựa trên `y_pred` (giá trị dự báo).

---

## Cột mới trong output CSV

| Cột | Mô tả |
|-----|-------|
| `forecast_for` | Thời điểm mà kết quả dự báo áp dụng (now + 24h) |
| `data_collected_at` | Thời điểm thu thập dữ liệu gốc (chỉ trong forecast-now) |

---

## Cập nhật preview & stats

- Preview columns trong `_forecast_now_worker`: thay `timestamp`, `rain_total` → `forecast_for`, `data_collected_at`
- Preview columns trong `_prediction_worker`: thay `timestamp` → `forecast_for`
- Stats response thêm: `forecast_horizon_hours`, `data_collected_at`, `forecast_for`
- `_load_recent_predictions()`: ưu tiên đọc `forecast_for` thay vì `timestamp` để hiển thị đúng

---

## Tóm tắt file thay đổi

| File | Thay đổi |
|------|----------|
| `Weather_Forcast_App/views/View_Predict.py` | Sửa 4 lỗi ở 3 hàm: `_forecast_now_worker`, `_prediction_worker`, `predict_manual_view`, `_load_recent_predictions` |
