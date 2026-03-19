# 📦 Ensemble Average — Model Artifacts (latest)

> **Trained**: 2026-03-19 23:00:28  
> **Dataset**: `merged_vrain_data_cleaned_20260319_225913.clean_final.csv` (112,648 rows)  
> **Training time**: 15.1 seconds  
> **Features**: 68 sau feature selection (40 engineered, 18 constant removed, polynomial added)

---

## Kiến trúc mô hình

```
Input (68 features)
    ↓  log1p(rain_total)   ← Transform target bên ngoài
    ↓  MissingValueHandler
    ↓  OutlierHandler (IQR clip)
    ↓  CategoricalEncoder
    ↓  WeatherScaler (StandardScaler)
    ↓
    ├── XGBoostRegressor
    ├── LightGBMRegressor
    ├── CatBoostRegressor
    └── RandomForestRegressor
    ↓  Average(4 predictions)
    ↓  expm1(avg)  ← Inverse transform
    ↓
    if avg >= rain_threshold (0.22mm) → báo mưa
    else → output = 0.0
```

**Thresholds**: `rain_threshold=0.22 mm`, `predict_threshold=0.45`

---

## Metrics (test set)

| Metric | Train | Valid | Test |
|--------|-------|-------|------|
| **R²** | 0.9923 | 0.7447 | **0.5262** |
| **RMSE (mm)** | 0.3724 | 2.5576 | **3.0413** |
| **Rain_F1** | 0.9390 | 0.8469 | **0.8380** |
| **Rain_Det_Acc** | 0.9319 | 0.7723 | **0.7483** |
| **ROC_AUC** | 0.9976 | 0.8647 | **0.8410** |
| **PR_AUC** | 0.9973 | 0.9127 | **0.9159** |

**Split**: 80% train / 10% valid / 10% test — chronological (sort_by_time=True, no shuffle)

---

## Diagnostics

| Trạng thái | Chi tiết |
|------------|---------|
| **overfit_status**: ⚠️ overfit | RainAcc Train(0.932) > Valid(0.772) — gap = 0.160 |

---

## Nội dung thư mục

| File | Mô tả |
|------|-------|
| `Model.pkl` | WeatherEnsembleModel serialized (XGB+LGB+Cat+RF) |
| `Transform_pipeline.pkl` | Pipeline transform (MissingValue→Outlier→CatEnc→Scaler) |
| `Feature_list.json` | Danh sách 68 features đầu vào |
| `Metrics.json` | Metrics đầy đủ (train/valid/test + ROC/PR-AUC + diagnostics) |
| `Train_info.json` | Metadata: dataset, split ratios, thresholds, model_type, timestamp |

---

## Cách load và sử dụng

```python
from Weather_Forcast_App.Machine_learning_model.interface.weather_predictor import WeatherPredictor

predictor = WeatherPredictor(model_type="ensemble_average")
result = predictor.predict(input_df)
```

---

> **Lần cập nhật**: 2026-03-19  
> **Ghi chú**: Model có dấu hiệu overfit (gap RainAcc = 0.160). Xem xét dùng Stacking Ensemble cho production nếu cần calibration tốt hơn.
