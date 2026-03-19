# 📁 Models

## Tổng quan
Thư mục này định nghĩa các model ML cho pipeline dự báo thời tiết: từng model riêng lẻ (XGBoost, LightGBM, CatBoost, RandomForest), ensemble models, và base class.

---

## Các model đang có

| File | Class | Mô tả |
|------|-------|-------|
| `Base_model.py` | `BaseModel` | Abstract base class — interface chung cho tất cả model |
| `XGBoost_Model.py` | `XGBoostModel` | Gradient boosting XGBoost |
| `LightGBM_Model.py` | `LightGBMModel` | Gradient boosting LightGBM |
| `CatBoost_Model.py` | `CatBoostModel` | Gradient boosting CatBoost |
| `Random_Forest_Model.py` | `RandomForestModel` | Random Forest (Scikit-learn) |
| `Ensemble_Average_Model.py` | `WeatherEnsembleModel` | Soft voting ensemble (XGB+LGB+Cat+RF) — production v1 |
| `Ensemble_Stacking_Model.py` | `WeatherStackingEnsembleModel` | 2-stage OOF stacking (8 base + 2 meta-LightGBM) — production v2 |
| `Ensemble_Stacking_LSTM_Meta.py` | `StackingLSTMMeta` | Variant stacking với LSTM meta-learner (experimental) |
| `TwoStage_Model.py` | `TwoStageModel` | Two-stage: classifier → regressor (legacy) |
| `Schema_Selector.py` | `SchemaSelector` | Schema bank routing by rain_intensity × season |
| `ml_types.py` | — | Type definitions và enums cho ML pipeline |

---

## Kiến trúc Ensemble

### Ensemble Average (`Ensemble_Average_Model.py`)
```
4 Regressors → Average → expm1 → rain decision (threshold 0.22mm)
```
- log1p applied externally
- overfit_status: ⚠️ overfit (F1 gap=0.160)

### Stacking Ensemble (`Ensemble_Stacking_Model.py`)
```
8 Base Models (OOF, n_splits=8) → 2 Meta-LightGBM → Schema Bank Routing
```
- log1p applied internally
- overfit_status: ✅ good (F1 gap=0.035)
- Khuyến nghị cho production

---

## Cấu trúc thư mục

```
Models/
├── 🐍 Base_model.py
├── 🐍 XGBoost_Model.py
├── 🐍 LightGBM_Model.py
├── 🐍 CatBoost_Model.py
├── 🐍 Random_Forest_Model.py
├── 🐍 Ensemble_Average_Model.py       # Production v1
├── 🐍 Ensemble_Stacking_Model.py      # Production v2 (GOOD FIT)
├── 🐍 Ensemble_Stacking_LSTM_Meta.py  # Experimental
├── 🐍 TwoStage_Model.py               # Legacy
├── 🐍 Schema_Selector.py              # Schema bank routing
├── 🐍 ml_types.py                     # Type definitions
├── 📄 ProcessingStream.md             # Mô tả luồng xử lý
└── 📁 knowledge/                      # Knowledge base cho model selection
```

---

## 👤 Maintainer / Profile Info
- 🧑‍💻 Maintainer: Võ Anh Nhật, Dư Quốc Việt, Trương Hoài Tú, Võ Huỳnh Anh Tuần
- 🎓 University: UTH
- 📧 Email: voanhnhat1612@gmmail.com, vohuynhanhtuan0512@gmail.com, hoaitu163@gmail.com, duviet720@gmail.com
- 📞 Phone: 0335052899

---

## License
MIT License
