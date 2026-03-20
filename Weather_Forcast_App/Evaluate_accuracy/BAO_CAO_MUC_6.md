# 6. Kết quả thực nghiệm, đánh giá và so sánh mô hình

## 6.1. Tổng quan thực nghiệm

### 6.1.1. Mô tả tập dữ liệu

Nghiên cứu sử dụng dữ liệu thời tiết thực tế từ các trạm quan trắc khí tượng, được thu thập và xử lý thông qua hệ thống crawl dữ liệu tự động. Sau quá trình tiền xử lý (làm sạch, loại bỏ giá trị ngoại lai, xử lý giá trị thiếu, hợp nhất các nguồn dữ liệu), bộ dữ liệu gốc gồm **112.648 bản ghi** được tổ chức thành dạng chuỗi thời gian và chia tách theo tỷ lệ **80/10/10** (Train / Validation / Test):

| Tập dữ liệu | Số mẫu | Tỷ lệ |
|:---:|:---:|:---:|
| **Train** | 289.156 | 80% |
| **Validation** | 36.144 | 10% |
| **Test** | 36.145 | 10% |
| **Tổng cộng** | 361.445 | 100% |

Sau quá trình xây dựng đặc trưng (feature engineering) bao gồm tạo các biến trễ (lag features), biến trung bình trượt (rolling averages), và dịch chuyển mục tiêu theo tầm dự báo (forecast horizon shift), tập test thực tế dùng để đánh giá còn **36.121 mẫu** cho cả hai mô hình.

> **Lưu ý quan trọng:** Cả hai mô hình Ensemble Average và Stacking Ensemble đều được đánh giá trên **cùng một tập test** (36.121 mẫu), với cùng giá trị mục tiêu ($y_{true}$), đảm bảo tính công bằng và khách quan trong so sánh. Điều này đã được xác minh bằng cách kiểm tra `np.allclose(y_test_EA, y_test_Stacking) = True`.

### 6.1.2. Cấu hình thực nghiệm

| Thông số | Giá trị |
|:---|:---|
| **Biến mục tiêu** | `rain_total` — tổng lượng mưa (mm) |
| **Tầm dự báo** | 24 giờ (forecast horizon = 24) |
| **Ngưỡng phân loại mưa/không mưa** | 0.1 mm |
| **Định nghĩa mưa** | Lượng mưa ≥ 0.1 mm được tính là "có mưa" |

### 6.1.3. Kiến trúc hai mô hình

**a) Mô hình Ensemble Average (EA):**
- **Phương pháp:** Trung bình có trọng số (weighted average) kết quả dự đoán từ các mô hình nền.
- **Mô hình nền:** XGBRegressor, LGBMRegressor, CatBoostRegressor, RandomForestRegressor (4 mô hình).
- **Số đặc trưng đầu vào:** 68.
- **Nguyên lý:** Kết hợp dự đoán bằng trung bình — đơn giản, giảm phương sai.

**b) Mô hình Stacking Ensemble:**
- **Phương pháp:** Hai tầng (two-level stacking) với kỹ thuật Out-of-Fold (OOF) prediction.
- **Tầng 1 — Mô hình nền (Base models):**
  - *Nhánh phân loại (Classification):* XGBoost, Random Forest, CatBoost, LightGBM (4 mô hình).
  - *Nhánh hồi quy (Regression):* XGBoost, Random Forest, CatBoost, LightGBM (4 mô hình).
  - Tổng cộng: **8 mô hình nền.**
- **Tầng 2 — Meta-learner:**
  - *Meta-classifier:* LGBMClassifier — quyết định "mưa" hay "không mưa".
  - *Meta-regressor:* LGBMRegressor — dự đoán lượng mưa cụ thể.
- **Số đặc trưng đầu vào:** 71.
- **Cross-validation folds:** 8 (dùng để tạo OOF predictions).
- **Ngưỡng dự đoán (predict threshold):** 0.4.
- **Nguyên lý:** Meta-learner học cách kết hợp tối ưu từ các dự đoán OOF của tầng 1 — phức tạp hơn nhưng khai thác được điểm mạnh của từng mô hình nền.

---

## 6.2. Kết quả mô hình Ensemble Average (EA)

### 6.2.1. Đánh giá hồi quy — Dự đoán lượng mưa

Mô hình EA được đánh giá khả năng dự đoán chính xác giá trị lượng mưa (biến liên tục) trên tập test (36.121 mẫu):

| Chỉ số | Tập Train (289.156 mẫu) | Tập Test (36.121 mẫu) | Đánh giá |
|:---|:---:|:---:|:---|
| **R² (Hệ số xác định)** | 0.4522 | 0.3928 | Mô hình giải thích ~39.3% biến động lượng mưa |
| **RMSE (mm)** | 0.6438 | 0.7373 | Sai số trung bình ~0.74 mm |
| **MAE (mm)** | 0.4918 | 0.5705 | Sai số tuyệt đối trung bình ~0.57 mm |
| **MBE (mm)** | — | +0.1212 | Thiên lệch dương: mô hình hay dự đoán *cao hơn* thực tế |
| **Pearson r** | — | 0.6508 | Tương quan tuyến tính trung bình–khá |
| **sMAPE (%)** | — | 106.34 | Sai số phần trăm đối xứng rất cao |

**Phân tích trạng thái Overfit/Underfit:**
- R² Train = 0.4522 vs R² Test = 0.3928 → Chênh lệch (gap) = **0.4661** (theo hệ số tính toán từ notebook).
- RMSE Train = 0.6438 vs RMSE Test = 0.7373 → **Tăng 14.5%** khi chuyển từ Train sang Test.
- **Kết luận:** Mô hình ở trạng thái **Overfit** — hiệu suất giảm rõ rệt trên dữ liệu chưa từng thấy.

### 6.2.2. Đánh giá phân loại — Phát hiện mưa

Với ngưỡng 0.1 mm, mô hình EA được đánh giá khả năng phân loại "có mưa" vs "không mưa":

| Chỉ số | Giá trị | Đánh giá |
|:---|:---:|:---|
| **Accuracy** | **61.03%** | Thấp — chỉ phân loại đúng ~61% mẫu |
| **Precision** | 80.51% | Khá — khi dự đoán "mưa", đúng 80.5% |
| **Recall** | 50.01% | Rất thấp — chỉ phát hiện được 50% số lần mưa thực |
| **F1-Score** | 37.93% | Rất thấp — mất cân bằng nghiêm trọng giữa Precision và Recall |
| **CSI (Critical Success Index)** | 0.6103 | Trung bình |
| **ROC-AUC** | 0.8039 | Khá — phân biệt tốt khi xét trên toàn bộ ngưỡng xác suất |
| **PR-AUC** | 0.8606 | Khá tốt — hiệu suất Precision-Recall curve |
| **Rain Detection** | 61.03% | Thấp |

**Ma trận nhầm lẫn (Confusion Matrix):**

|  | Dự đoán: Không mưa | Dự đoán: Mưa |
|:---|:---:|:---:|
| **Thực tế: Không mưa** | TN = 4 | FP = 14.075 |
| **Thực tế: Mưa** | FN = 0 | TP = 22.042 |

**Phân tích ma trận nhầm lẫn:**
Mô hình EA có xu hướng **dự đoán hầu như mọi mẫu đều là "mưa"** (TN = 4 trên tổng 14.079 mẫu không mưa thực tế). Điều này dẫn đến:
- **FP rất cao (14.075):** Báo mưa giả (false alarm) ở gần như toàn bộ ngày khô.
- **FN = 0:** Không bỏ sót trường hợp mưa nào, nhưng đây là kết quả "ảo" do mô hình luôn báo mưa.
- Mô hình thiếu khả năng **phân biệt** ngày khô và ngày mưa ở mức ngưỡng hiện tại.

### 6.2.3. Đánh giá theo phân khúc lượng mưa

| Phân khúc | Số mẫu | MAE (mm) | RMSE (mm) | Rain Det Acc (%) | Nhận xét |
|:---|:---:|:---:|:---:|:---:|:---|
| Không mưa (0 mm) | 11.390 | 0.518 | 0.627 | **0.00%** | Hoàn toàn thất bại — hay báo mưa giả |
| Mưa nhẹ (0.1–2.5 mm) | 18.154 | 0.490 | 0.652 | **85.20%** | Xuất sắc — sai số rất thấp |
| Mưa vừa (2.5–7.5 mm) | 6.577 | 0.883 | 1.064 | **100.00%** | Phát hiện tốt — sai số chấp nhận được |
| Mưa to (7.5–25 mm) | 0 | — | — | — | Không có mẫu |
| Mưa rất to (> 25 mm) | 0 | — | — | — | Không có mẫu |

Mô hình EA dự báo **xuất sắc ở phân khúc mưa nhẹ** nhưng **hoàn toàn thất bại ở phân khúc không mưa** — điều này phản ánh trực tiếp tình trạng báo mưa giả (false alarm) gần tuyệt đối ở nhóm không mưa.

### 6.2.4. Đánh giá độ ổn định

| Tập dữ liệu | R² | RMSE | MAE | Rain Det Acc |
|:---|:---:|:---:|:---:|:---:|
| Train | 0.9923 | 0.3724 | 0.0819 | 93.19% |
| Valid | 0.7447 | 2.5576 | 0.9344 | 77.23% |
| Test | 0.5262 | 3.0413 | 1.0821 | 74.83% |

- **Độ lệch chuẩn R² qua các fold:** σ = 0.1904 → **Không ổn định** (biến động lớn giữa các fold).
- **R² trung bình cross-validation:** 0.7544 (nhưng phương sai cao).
- **Điểm tổng hợp (Overall Score):** 51.13/100 → **Xếp hạng D** — Mô hình chưa đạt yêu cầu.

---

## 6.3. Kết quả mô hình Stacking Ensemble

### 6.3.1. Đánh giá hồi quy — Dự đoán lượng mưa

| Chỉ số | Tập Train (289.132 mẫu) | Tập Test (36.121 mẫu) | Đánh giá |
|:---|:---:|:---:|:---|
| **R² (Hệ số xác định)** | 0.4669 | 0.3522 | Mô hình giải thích ~35.2% biến động lượng mưa |
| **RMSE (mm)** | 0.6350 | 0.7616 | Sai số trung bình ~0.76 mm |
| **MAE (mm)** | 0.4335 | 0.5595 | Sai số tuyệt đối trung bình ~0.56 mm |
| **MBE (mm)** | +0.0241 | −0.0628 | Thiên lệch nhẹ âm: dự đoán *thấp hơn* thực tế một chút |
| **Pearson r** | 0.6840 | 0.5980 | Tương quan tuyến tính trung bình |
| **sMAPE (%)** | 70.12 | 77.89 | Sai số phần trăm đối xứng cao nhưng thấp hơn EA |

**Phân tích trạng thái Overfit/Underfit:**
- R² Train = 0.4669 vs R² Test = 0.3522 → Chênh lệch gap = **0.2555**.
- RMSE ratio (Test/Train) = **1.45** — mức chấp nhận được.
- **Kết luận:** Mô hình ở trạng thái **Good Fit** — hiệu suất Train và Test tương đối nhất quán, không có dấu hiệu overfit nghiêm trọng.

### 6.3.2. Đánh giá phân loại — Phát hiện mưa

| Chỉ số | Giá trị | Đánh giá |
|:---|:---:|:---|
| **Accuracy** | **80.70%** | Tốt — phân loại đúng hơn 80% mẫu |
| **Precision** | 84.32% | Tốt — khi dự đoán "mưa", đúng 84.3% |
| **Recall** | 88.21% | Tốt — phát hiện được 88.2% số lần mưa thực |
| **F1-Score** | 86.22% | Tốt — cân bằng tốt giữa Precision và Recall |
| **CSI (Critical Success Index)** | 0.7578 | Tốt — chỉ số dự báo khắt khe đạt mức cao |
| **FAR (False Alarm Rate)** | 15.68% | Thấp — tỷ lệ báo mưa giả chỉ ~16% |
| **ROC-AUC** | 0.8133 | Tốt — phân biệt tốt giữa mưa và không mưa |
| **PR-AUC** | 0.8245 | Tốt |
| **Rain Detection** | 79.46% | Khá |

**Ma trận nhầm lẫn (Confusion Matrix):**

|  | Dự đoán: Không mưa | Dự đoán: Mưa |
|:---|:---:|:---:|
| **Thực tế: Không mưa** | TN = 7.334 | FP = 4.056 |
| **Thực tế: Mưa** | FN = 2.916 | TP = 21.815 |

**Phân tích ma trận nhầm lẫn:**
Mô hình Stacking cho thấy sự **phân biệt rõ ràng** giữa ngày mưa và ngày khô:
- **TN = 7.334:** Nhận diện đúng 52.1% ngày không mưa (7.334/14.079).
- **TP = 21.815:** Phát hiện đúng 98.97% ngày mưa thực sự (21.815/22.042).
- **FP = 4.056:** Tỷ lệ báo mưa giả thấp hơn EA đáng kể (4.056 vs 14.075).
- **FN = 2.916:** Bỏ sót 2.916 trường hợp mưa — chấp nhận được.

### 6.3.3. Tổng hợp kết quả Stacking

Mô hình Stacking Ensemble thể hiện hiệu suất **ổn định và cân bằng** trên cả hai nhiệm vụ hồi quy và phân loại. Kiến trúc hai tầng với meta-learner LightGBM cho phép mô hình khai thác hiệu quả điểm mạnh của từng mô hình nền, đặc biệt trong việc phân biệt ngày mưa/không mưa.

---

## 6.4. So sánh toàn diện hai mô hình

### 6.4.1. Bảng so sánh các chỉ số hồi quy

| Chỉ số | Ensemble Average | Stacking Ensemble | Chênh lệch | Mô hình tốt hơn |
|:---|:---:|:---:|:---:|:---:|
| **R²** | **0.3928** | 0.3522 | +0.0406 | **EA** ↑ |
| **RMSE (mm)** | **0.7373** | 0.7616 | −0.0243 | **EA** ↓ |
| **MAE (mm)** | 0.5705 | **0.5595** | −0.0110 | **Stacking** ↓ |
| **MBE (mm)** | +0.1212 | **−0.0628** | — | **Stacking** (gần 0 hơn) |
| **Pearson r** | **0.6508** | 0.5980 | +0.0528 | **EA** ↑ |
| **sMAPE (%)** | 106.34 | **77.89** | −28.45 | **Stacking** ↓ |

**Nhận xét hồi quy:** Mô hình EA có R², RMSE, và Pearson r tốt hơn — tức là khả năng giải thích biến động và mức tương quan cao hơn một chút. Tuy nhiên, Stacking có MAE tốt hơn (sai số tuyệt đối thấp hơn), thiên lệch hệ thống (MBE) nhỏ hơn đáng kể, và sMAPE thấp hơn rất nhiều. **Xét tổng thể hồi quy, hai mô hình tương đương nhau** với mỗi bên có ưu thế ở một số chỉ số khác nhau.

### 6.4.2. Bảng so sánh các chỉ số phân loại

| Chỉ số | Ensemble Average | Stacking Ensemble | Chênh lệch | Mô hình tốt hơn |
|:---|:---:|:---:|:---:|:---:|
| **Accuracy** | 61.03% | **80.70%** | +19.67 pp | **Stacking** ↑↑ |
| **Precision** | 80.51% | **84.32%** | +3.81 pp | **Stacking** ↑ |
| **Recall** | 50.01% | **88.21%** | +38.20 pp | **Stacking** ↑↑↑ |
| **F1-Score** | 37.93% | **86.22%** | +48.29 pp | **Stacking** ↑↑↑ |
| **CSI** | 0.6103 | **0.7578** | +0.1475 | **Stacking** ↑↑ |
| **ROC-AUC** | 0.8039 | **0.8133** | +0.0094 | **Stacking** ↑ |
| **PR-AUC** | **0.8606** | 0.8245 | −0.0361 | **EA** ↑ |
| **Rain Detection** | 61.03% | **79.46%** | +18.43 pp | **Stacking** ↑↑ |
| **FAR** | ≈100% | **15.68%** | — | **Stacking** ↓↓↓ |

> *pp = percentage point (điểm phần trăm)*

**Nhận xét phân loại:** Stacking Ensemble **vượt trội hoàn toàn** so với EA trong nhiệm vụ phân loại mưa/không mưa, với Accuracy cao hơn **19.67 điểm phần trăm**, F1-Score cao hơn **48.29 điểm phần trăm**, và Recall cao hơn **38.20 điểm phần trăm**. Đặc biệt, tỷ lệ báo mưa giả (FAR) của EA gần 100% — mô hình EA gần như luôn dự đoán "có mưa" cho mọi mẫu — trong khi FAR của Stacking chỉ 15.68%.

Điểm duy nhất EA tốt hơn là PR-AUC (0.8606 vs 0.8245), cho thấy bản chất xác suất của EA tốt hơn một chút trong xếp hạng mẫu, nhưng khi áp dụng ngưỡng quyết định, khả năng phân loại của EA kém rất xa Stacking.

### 6.4.3. So sánh ma trận nhầm lẫn

| Thành phần | Ensemble Average | Stacking Ensemble | Phân tích |
|:---|:---:|:---:|:---|
| **True Negative (TN)** | 4 | 7.334 | EA gần như không nhận diện được ngày không mưa |
| **False Positive (FP)** | 14.075 | 4.056 | EA báo mưa giả gấp **3.5 lần** Stacking |
| **False Negative (FN)** | 0 | 2.916 | EA không bỏ sót mưa nhưng do luôn báo mưa |
| **True Positive (TP)** | 22.042 | 21.815 | Tương đương — cả hai phát hiện mưa tốt |

### 6.4.4. So sánh tình trạng Overfit

| Tiêu chí | Ensemble Average | Stacking Ensemble |
|:---|:---:|:---:|
| **Trạng thái** | **Overfit** ⚠️ | **Good Fit** ✅ |
| **R² gap** | 0.4661 (cao) | 0.2555 (chấp nhận được) |
| **RMSE ratio (Test/Train)** | 1.145 | 1.451 |
| **Độ ổn định (R² std)** | 0.1904 (không ổn định) | — |
| **Overall Score** | 51.13/100 (Grade D) | — |

Mặc dù RMSE ratio của Stacking cao hơn, trạng thái tổng thể của Stacking vẫn được đánh giá là **Good Fit** vì sự chênh lệch giữa Train và Test nằm trong giới hạn chấp nhận và mô hình không có biểu hiện "học thuộc lòng" tập Train.

### 6.4.5. Bảng tổng hợp — Mô hình nào thắng?

| # | Chỉ số | EA | Stacking | Thắng |
|:---:|:---|:---:|:---:|:---:|
| 1 | R² | **0.3928** | 0.3522 | EA |
| 2 | RMSE (mm) | **0.7373** | 0.7616 | EA |
| 3 | MAE (mm) | 0.5705 | **0.5595** | Stacking |
| 4 | MBE (mm) | +0.1212 | **−0.0628** | Stacking |
| 5 | Pearson r | **0.6508** | 0.5980 | EA |
| 6 | sMAPE (%) | 106.34 | **77.89** | Stacking |
| 7 | Rain Detection (%) | 61.03 | **79.46** | Stacking |
| 8 | Accuracy (cls) | 61.03% | **80.70%** | Stacking |
| 9 | Precision (cls) | 80.51% | **84.32%** | Stacking |
| 10 | Recall (cls) | 50.01% | **88.21%** | Stacking |
| 11 | F1-Score (cls) | 37.93% | **86.22%** | Stacking |
| 12 | CSI | 0.6103 | **0.7578** | Stacking |
|  | **Tổng thắng** | **3** | **9** | **Stacking ✅** |

Trong tổng số 12 chỉ số đánh giá, **Stacking Ensemble thắng 9 chỉ số** (chiếm 75%), trong khi **Ensemble Average chỉ thắng 3 chỉ số** (chiếm 25%). Các chỉ số EA thắng tập trung ở nhóm hồi quy đo tương quan (R², RMSE, Pearson r), trong khi Stacking chiếm ưu thế tuyệt đối ở mọi chỉ số phân loại.

---

## 6.5. Phân tích chuyên sâu

### 6.5.1. Vì sao EA thất bại ở phân loại?

Mô hình Ensemble Average **chỉ có tầng hồi quy** — nó dự đoán lượng mưa liên tục rồi áp dụng ngưỡng 0.1mm để phân loại. Vấn đề nằm ở chỗ: khi các mô hình nền đều có xu hướng dự đoán giá trị dương nhỏ (do phần lớn dữ liệu có mưa), phép trung bình sẽ tạo ra giá trị > 0.1mm cho gần như mọi mẫu, dẫn đến **false alarm gần tuyệt đối** ở nhóm không mưa.

Ngược lại, Stacking Ensemble có **nhánh phân loại riêng biệt** (4 mô hình classifier) với meta-classifier LGBMClassifier chuyên quyết định mưa hay không mưa, sau đó mới dùng nhánh hồi quy để dự đoán lượng mưa. Cách tiếp cận hai nhánh này cho phép phân biệt hiệu quả hơn.

### 6.5.2. Vì sao EA tốt hơn ở một số chỉ số hồi quy?

EA có R² (0.3928 vs 0.3522) và RMSE (0.7373 vs 0.7616) tốt hơn một chút. Điều này giải thích bởi: phép trung bình có trọng số giảm phương sai dự đoán hiệu quả (theo lý thuyết ensemble), giúp các giá trị dự đoán "trơn" hơn và gần giá trị kỳ vọng hơn. Tuy nhiên, ưu thế này nhỏ (R² chênh 0.04, RMSE chênh 0.02mm) và không đủ bù đắp cho sự thất bại nghiêm trọng ở nhiệm vụ phân loại.

### 6.5.3. Thiên lệch hệ thống (Bias)

- **EA:** MBE = +0.1212 mm — mô hình **dự đoán cao hơn** thực tế trung bình 0.12mm. Điều này phù hợp với xu hướng "luôn báo mưa" đã phân tích.
- **Stacking:** MBE = −0.0628 mm — mô hình **dự đoán thấp hơn** thực tế trung bình 0.06mm. Thiên lệch nhỏ hơn gần **2 lần** so với EA, cho thấy Stacking có tính trung lập (unbiased) tốt hơn.

---

## 6.6. Kết luận và khuyến nghị

### 6.6.1. Kết luận

Qua quá trình đánh giá toàn diện trên cùng tập test gồm **36.121 mẫu** với **12 chỉ số đo lường** bao gồm cả nhóm hồi quy (R², RMSE, MAE, MBE, Pearson r, sMAPE) và nhóm phân loại (Accuracy, Precision, Recall, F1-Score, CSI, Rain Detection), nghiên cứu rút ra các kết luận sau:

1. **Mô hình Stacking Ensemble là lựa chọn tốt hơn** để đưa vào hệ thống dự báo thời tiết sản xuất (production), với Accuracy = 80.70%, F1-Score = 86.22%, CSI = 0.7578, và trạng thái Good Fit.

2. **Mô hình Ensemble Average không đáp ứng yêu cầu** ở nhiệm vụ phân loại, với Accuracy chỉ 61.03%, F1-Score = 37.93%, tỷ lệ báo mưa giả gần tuyệt đối, và trạng thái Overfit nghiêm trọng (Overall Score = 51.13/100, Grade D).

3. **Kiến trúc hai nhánh (classification + regression) của Stacking** cho thấy ưu thế rõ rệt so với kiến trúc đơn nhánh (chỉ regression) của EA trong bài toán dự báo mưa, vì nó cho phép xử lý riêng biệt hai nhiệm vụ: phát hiện mưa và ước lượng lượng mưa.

4. Mặc dù cả hai mô hình đều có R² ở mức trung bình (< 0.5), điều này phản ánh **bản chất khó dự đoán** của hiện tượng mưa — một biến thời tiết có tính ngẫu nhiên cao, chịu ảnh hưởng bởi nhiều yếu tố phi tuyến và chưa được thu thập đầy đủ.

### 6.6.2. Khuyến nghị

1. **Triển khai sản xuất:** Sử dụng mô hình **Stacking Ensemble** làm mô hình chính cho hệ thống dự báo thời tiết.

2. **Cải thiện mô hình EA:** Nếu muốn tiếp tục phát triển EA, cần:
   - Thêm nhánh phân loại riêng hoặc tối ưu hóa ngưỡng quyết định.
   - Áp dụng kỹ thuật regularization mạnh hơn để giảm overfit.
   - Cân nhắc loại bỏ hoặc giảm trọng số các mô hình nền có xu hướng overpredict.

3. **Nâng cao hiệu suất chung:**
   - Thu thập thêm dữ liệu (đặc biệt dữ liệu mưa to và mưa rất to — hiện thiếu mẫu).
   - Bổ sung các đặc trưng mới từ nguồn dữ liệu vệ tinh, radar thời tiết.
   - Nghiên cứu các kiến trúc deep learning (LSTM, Transformer) cho dữ liệu chuỗi thời gian.
   - Tối ưu hóa hyperparameter bằng Bayesian Optimization hoặc Optuna.

---

> **Verdict cuối cùng:** 🏆 **Stacking Ensemble** được chọn làm mô hình triển khai, với hiệu suất phân loại vượt trội (+19.67% Accuracy, +48.29% F1-Score) và trạng thái Good Fit ổn định.
