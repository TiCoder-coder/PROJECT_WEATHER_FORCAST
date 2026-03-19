# 📁 Machine_learning_artifacts

## Tổng quan
Thư mục này lưu trữ toàn bộ artifacts của các mô hình ML đã huấn luyện: model serialized, transform pipeline, feature list, metrics, và metadata huấn luyện. Mỗi loại mô hình được lưu trong thư mục riêng, cấu trúc `<model_type>/latest/`.

---

## Cấu trúc thư mục

```
Machine_learning_artifacts/
├── ensemble_average/
│   ├── latest/                    # Ensemble Average — phiên bản đang dùng
│   │   ├── Model.pkl              # WeatherEnsembleModel (XGB+LGB+Cat+RF, soft voting)
│   │   ├── Transform_pipeline.pkl # MissingValueHandler+OutlierHandler+CatEncoder+WeatherScaler
│   │   ├── Feature_list.json      # 68 features đầu vào
│   │   ├── Metrics.json           # Metrics thực tế (train/valid/test)
│   │   └── Train_info.json        # Metadata: dataset, split, thresholds, training time
│   └── README.md                  # Mô tả chi tiết Ensemble Average
│
└── stacking_ensemble/
    ├── latest/                    # Stacking Ensemble — phiên bản đang dùng
    │   ├── Model.pkl              # WeatherStackingEnsembleModel (8 base + 2 meta-LightGBM)
    │   ├── Transform_pipeline.pkl # Pipeline transform
    │   ├── Feature_list.json      # 68 features đầu vào
    │   ├── Metrics.json           # Metrics thực tế (train/valid/test)
    │   └── Train_info.json        # Metadata: n_splits=8, OOF info, schema_bank_routing
    └── README.md                  # Mô tả chi tiết Stacking Ensemble
```

---

## Mô hình đang hoạt động

### 1. Ensemble Average (`ensemble_average/latest/`)

> **Trained**: 2026-03-19 23:00:28 | Dataset: 112,648 rows | Features: 68 | Training time: 15.1s

| Metric | Train | Valid | Test |
|--------|-------|-------|------|
| R² | 0.9923 | 0.7447 | 0.5262 |
| RMSE (mm) | 0.3724 | 2.5576 | 3.0413 |
| Rain_F1 | 0.9390 | 0.8469 | 0.8380 |
| Rain_Det_Acc | 0.9319 | 0.7723 | 0.7483 |
| ROC_AUC | 0.9976 | 0.8647 | 0.8410 |
| PR_AUC | 0.9973 | 0.9127 | 0.9159 |

- **overfit_status**: ⚠️ `overfit` (RainAcc gap = 0.160)
- **log1p**: applied externally trước khi train
- **Thresholds**: `rain_threshold=0.22`, `predict_threshold=0.45`

### 2. Stacking Ensemble (`stacking_ensemble/latest/`)

> **Trained**: 2026-03-19 23:12:06 | Dataset: 112,648 rows | Features: 68 | Training time: 48.82s

| Metric | Train | Valid | Test |
|--------|-------|-------|------|
| R² | 0.8194 | 0.7663 | 0.5587 |
| RMSE (mm) | 1.8013 | 2.4472 | 2.9350 |
| Rain_F1 | 0.8998 | 0.8647 | 0.8476 |
| Rain_Det_Acc | 0.8884 | 0.8156 | 0.7874 |
| ROC_AUC | 0.8983 | 0.8170 | 0.7875 |
| PR_AUC | 0.8787 | 0.8603 | 0.8579 |

- **overfit_status**: ✅ `good` (F1 gap = 0.035)
- **log1p**: xử lý **nội bộ** — không transform bên ngoài
- **Thresholds**: `predict_threshold=0.4`, `rain_threshold=0.1`
- **Stage 1**: 8 base models (OOF, n_splits=8) — xgb/rf/cat/lgbm × cls+reg
- **Stage 2**: 2 meta-LightGBM (meta_cls + meta_reg) trained on OOF
- **Schema bank routing**: per rain_intensity (no_rain/light/moderate/heavy/very_heavy) × season (rainy/dry)

---

## Quy tắc sử dụng

- **Không commit** file `.pkl` lên git (file nhị phân lớn, xem `.gitignore`)
- **Load model**: Dùng `WeatherPredictor` trong `Machine_learning_model/interface/`
- **Train lại**: Chạy `python manage.py train --config config/train_config.json`
- **Artifacts path** tự động được quản lý bởi `Weather_Forcast_App/paths.py`

---

## 👤 Maintainer / Profile Info
- 🧑‍💻 Maintainer: Võ Anh Nhật, Dư Quốc Việt, Trương Hoài Tú, Võ Huỳnh Anh Tuần
- 🎓 University: UTH
- 📧 Email: voanhnhat1612@gmmail.com, vohuynhanhtuan0512@gmail.com, hoaitu163@gmail.com, duviet720@gmail.com
- 📞 Phone: 0335052899

---

## License
MIT License
