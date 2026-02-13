# 📁 Machine_learning_model

## Tổng quan
Thư mục này chứa toàn bộ mã nguồn, config, model, pipeline, và test cho các tác vụ machine learning (ML) dự báo thời tiết.

## Chức năng chung
- Xây dựng, huấn luyện, đánh giá các mô hình ML (XGBoost, LightGBM, CatBoost, RandomForest).
- Tuning hyperparameters, lưu artifacts, test pipeline.
- Tách biệt từng module: train, model, features, data, config, evaluation, interface.

## Cấu trúc thư mục
<ul>
  <li>trainning/: Huấn luyện, tuning, pipeline.</li>
  <li>Models/: Định nghĩa các model ML.</li>
  <li>features/: Xử lý đặc trưng, transformers.</li>
  <li>data/: Loader, schema, split.</li>
  <li>config/: Config YAML/JSON.</li>
  <li>evaluation/: Đánh giá mô hình.</li>
  <li>interface/: Interface chuẩn cho model/pipeline.</li>
  <li>TEST/: Test/benchmark pipeline.</li>
</ul>

## Chức năng riêng lẻ
- `trainning/`: Huấn luyện, tuning, pipeline.
- `Models/`: Định nghĩa model, base, knowledge.
- `features/`: Xử lý đặc trưng, transformers.
- `data/`: Loader, schema, split.
- `config/`: Config pipeline.
- `evaluation/`: Đánh giá, metrics.
- `interface/`: Interface chuẩn.
- `TEST/`: Test, benchmark.

---

## 👤 Maintainer / Profile Info
- 🧑‍💻 Maintainer: Võ Anh Nhật, Dư Quốc Việt, Trương Hoài Tú, Võ Huỳnh Anh Tuần
- 🎓 University: UTH
- 📧 Email: voanhnhat1612@gmmail.com, vohuynhanhtuan0512@gmail.com, hoaitu163@gmail.com, duviet720@gmail.com
- 📞 Phone: 0335052899

---

## License
MIT License
