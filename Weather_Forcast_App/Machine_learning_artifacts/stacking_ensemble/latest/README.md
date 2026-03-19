# 📦 Stacking Ensemble — Model Artifacts (latest)

> **Trained**: 2026-03-19 23:12:06  
> **Dataset**: `merged_vrain_data_cleaned_20260319_225913.clean_final.csv` (112,648 rows)  
> **Training time**: 48.82 seconds  
> **Features**: 68 sau feature selection  
> **OOF samples**: 90,094 (cls) / 47,200 (reg)

---

## Kiến trúc mô hình

```
Input (68 features)
  ← log1p KHÔNG áp dụng bên ngoài — model xử lý nội bộ
    ↓
Stage 1: 8 Base Models (OOF cross-validation, n_splits=8)
    ├── Classifiers (predict P(rain)):
    │   ├── XGB_cls   (subsample ROC-AUC: 0.9426)
    │   ├── RF_cls    (subsample ROC-AUC: 0.8767)
    │   ├── Cat_cls   (subsample ROC-AUC: 0.9263)
    │   └── LGBM_cls  (subsample ROC-AUC: 0.9590)
    └── Regressors (predict rain amount, log1p space):
        ├── XGB_reg   (subsample MAE log1p: 0.2137)
        ├── RF_reg    (subsample MAE log1p: 0.4004)
        ├── Cat_reg   (subsample MAE log1p: 0.3514)
        └── LGBM_reg  (subsample MAE log1p: 0.2955)
    ↓  OOF predictions → meta features
Stage 2: 2 Meta-LightGBM (trained on OOF)
    ├── meta_cls → P(rain) → threshold 0.4 → binary
    └── meta_reg → rain amount → expm1 → mm
    ↓
Schema Bank Routing
    → Chọn base model tốt nhất per rain_intensity × season
    → rain_intensity: no_rain / light / moderate / heavy / very_heavy
    → season: rainy / dry
```

**Thresholds**: `predict_threshold=0.4`, `rain_threshold=0.1 mm`, `seed=42`

---

## Metrics (test set)

| Metric | Train | Valid | Test |
|--------|-------|-------|------|
| **R²** | 0.8194 | 0.7663 | **0.5587** |
| **RMSE (mm)** | 1.8013 | 2.4472 | **2.9350** |
| **Rain_F1** | 0.8998 | 0.8647 | **0.8476** |
| **Rain_Det_Acc** | 0.8884 | 0.8156 | **0.7874** |
| **ROC_AUC** | 0.8983 | 0.8170 | **0.7875** |
| **PR_AUC** | 0.8787 | 0.8603 | **0.8579** |

**Split**: 80% train / 10% valid / 10% test — chronological (sort_by_time=True, no shuffle)

---

## Diagnostics

| Trạng thái | Chi tiết |
|------------|---------|
| **overfit_status**: ✅ good | Rain_F1 gap train-valid = 0.035 (ngưỡng 0.15) |

---

## Cấu hình Hyperparameters

```json
{
  "n_splits": 8,
  "predict_threshold": 0.4,
  "rain_threshold": 0.1,
  "seed": 42,
  "cls_params": {
    "xgb":  {"n_estimators": 250, "max_depth": 4, "min_child_weight": 15, "reg_alpha": 1.0, "reg_lambda": 3.0},
    "lgbm": {"n_estimators": 250, "max_depth": 4, "num_leaves": 20, "min_child_samples": 80},
    "cat":  {"iterations": 200, "depth": 4, "l2_leaf_reg": 15},
    "rf":   {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 20}
  },
  "meta_cls_params": {"n_estimators": 60, "num_leaves": 7, "min_child_samples": 100, "reg_alpha": 2.0, "reg_lambda": 5.0},
  "meta_reg_params":  {"n_estimators": 60, "num_leaves": 7, "min_child_samples": 30, "reg_alpha": 2.0, "reg_lambda": 5.0}
}
```

---

## Nội dung thư mục

| File | Mô tả |
|------|-------|
| `Model.pkl` | WeatherStackingEnsembleModel serialized (8 base + 2 meta) |
| `Transform_pipeline.pkl` | Pipeline transform |
| `Feature_list.json` | Danh sách 68 features đầu vào |
| `Metrics.json` | Metrics đầy đủ (train/valid/test + ROC/PR-AUC) |
| `Train_info.json` | Metadata: n_splits=8, OOF info, schema_bank config, timestamp |

---

## Cách load và sử dụng

```python
from Weather_Forcast_App.Machine_learning_model.interface.weather_predictor import WeatherPredictor

# QUAN TRỌNG: KHÔNG áp dụng log1p trước — model tự xử lý nội bộ
predictor = WeatherPredictor(model_type="stacking_ensemble")
result = predictor.predict(input_df)
```

---

> **Lần cập nhật**: 2026-03-19  
> **Ghi chú**: Model GOOD FIT (F1 gap=0.035). Khuyến nghị cho production với dataset hiện tại (112,648 rows).
