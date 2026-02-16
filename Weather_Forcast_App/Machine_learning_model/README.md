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


---

## 🔬 Kỹ thuật & Phương pháp sử dụng trong các file tuning và training (flow)

### 1. Kỹ thuật Tuning Hyperparameters
- **GridSearchCV**: Duyệt toàn bộ các tổ hợp tham số, đảm bảo tìm ra bộ tốt nhất nhưng tốn thời gian.
- **RandomizedSearchCV**: Chọn ngẫu nhiên các tổ hợp tham số, nhanh hơn GridSearch, phù hợp khi search space lớn.
- **Optuna (TPE Sampler, Pruning)**: Tối ưu hóa thông minh, tự động dừng các trial kém, giúp tiết kiệm tài nguyên và thời gian. Hỗ trợ sampling, pruning, logging.
- **Bayesian Optimization**: Định nghĩa sẵn, có thể mở rộng thêm (chưa dùng).

### 2. Kỹ thuật Training Pipeline (Flow tổng)
- **Đọc config (JSON/YAML)**: Định nghĩa toàn bộ pipeline, tham số, đường dẫn, target, split, features.
- **Load data**: Sử dụng Loader.py để đọc file CSV/XLSX, xử lý datetime, missing, sort.
- **Validate schema**: Đảm bảo đúng định dạng, cột, kiểu dữ liệu.
- **Split train/valid/test**: Chia dữ liệu theo config, lưu ra thư mục riêng.
- **Build features**: Xây dựng đặc trưng từ raw data, tạo các biến mới, lag, interaction, location.
- **Transform pipeline**: Chuẩn hóa, encode, impute, đảm bảo train/predict dùng đúng pipeline.
- **Train model**: Wrapper cho các model (RandomForest, XGBoost, LightGBM, CatBoost), hỗ trợ early stopping, logging, save best iteration.
- **Evaluate metrics**: Đánh giá bằng RMSE, MAE, R2, Accuracy, F1, Precision, Recall tùy bài toán.
- **Save artifacts**: Lưu model, pipeline, feature list, metrics, train_info ra thư mục artifacts/latest.

### 3. Kỹ thuật Logging & Error Handling
- **Logging chuẩn**: Sử dụng logger theo module, ghi log quá trình training/tuning, lưu log file.
- **Error handling**: Kiểm tra type, validate schema, raise exception hợp lý, catch lỗi khi load data, train, tuning.

### 4. Kỹ thuật Interface & Inference
- **Interface chuẩn**: Đảm bảo predict/inference dùng đúng pipeline, features, artifacts như lúc train.
- **Test/Benchmark**: Có module TEST để kiểm thử, benchmark, validate pipeline.

### 5. Kỹ thuật lưu trữ & quản lý artifacts
- **Chuẩn hóa artifacts**: Lưu toàn bộ model, pipeline, metrics, train_info về `Machine_learning_artifacts/latest`.
- **Dễ dàng backup, mở rộng, load lại cho inference/API/UI.**

---

## 👤 Maintainer / Profile Info
- 🧑‍💻 Maintainer: Võ Anh Nhật, Dư Quốc Việt, Trương Hoài Tú, Võ Huỳnh Anh Tuần
- 🎓 University: UTH
- 📧 Email: voanhnhat1612@gmmail.com, vohuynhanhtuan0512@gmail.com, hoaitu163@gmail.com, duviet720@gmail.com
- 📞 Phone: 0335052899

---

## License
MIT License
