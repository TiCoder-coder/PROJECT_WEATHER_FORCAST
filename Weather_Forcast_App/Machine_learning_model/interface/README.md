# 📁 interface

## Tổng quan
Thư mục này chứa các class interface chuẩn để load và sử dụng model đã huấn luyện cho việc dự báo thời tiết.

---

## Các file

| File | Class | Model type |
|------|-------|-----------|
| `predictor_by_ensemble_average.py` | `WeatherPredictorEnsembleAverage` | Ensemble Average (log1p external) |
| `predictor_by_stacking_ensemble.py` | `WeatherPredictorStackingEnsemble` | Stacking Ensemble (log1p internal) |

---

## Cách sử dụng

```python
# Ensemble Average
from Weather_Forcast_App.Machine_learning_model.interface.predictor_by_ensemble_average import WeatherPredictorEnsembleAverage

predictor = WeatherPredictorEnsembleAverage()
result = predictor.predict(input_df)

# Stacking Ensemble (KHUYẾN NGHỊ — GOOD FIT)
# KHÔNG áp dụng log1p trước — model tự xử lý nội bộ
from Weather_Forcast_App.Machine_learning_model.interface.predictor_by_stacking_ensemble import WeatherPredictorStackingEnsemble

predictor = WeatherPredictorStackingEnsemble()
result = predictor.predict(input_df)
```

---

## Artifacts path

Mỗi predictor tự động load từ:
- `ensemble_average`: `Machine_learning_artifacts/ensemble_average/latest/`
- `stacking_ensemble`: `Machine_learning_artifacts/stacking_ensemble/latest/`

Path được quản lý bởi `Weather_Forcast_App/paths.py`.

---

## 👤 Maintainer / Profile Info
- 🧑‍💻 Maintainer: Võ Anh Nhật, Dư Quốc Việt, Trương Hoài Tú, Võ Huỳnh Anh Tuần
- 🎓 University: UTH
- 📧 Email: voanhnhat1612@gmmail.com, vohuynhanhtuan0512@gmail.com, hoaitu163@gmail.com, duviet720@gmail.com
- 📞 Phone: 0335052899

---

## License
MIT License
