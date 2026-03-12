<div align="center">

# 🌿 LightGBM — Light Gradient Boosting Machine  
<sub><b>GBDT siêu nhanh cho dữ liệu dạng bảng (tabular)</b> — phân loại, hồi quy, ranking</sub>

<br/>

<img alt="LightGBM" src="https://img.shields.io/badge/Model-Gradient%20Boosting%20Trees-2ea44f?style=for-the-badge" />
<img alt="Use-case" src="https://img.shields.io/badge/Use--case-Tabular%20Data-blue?style=for-the-badge" />
<img alt="Focus" src="https://img.shields.io/badge/Focus-Speed%20%2B%20Memory-orange?style=for-the-badge" />

<br/><br/>

</div>

---

## 📚 Mục lục
- [1. LightGBM là gì?](#1-lightgbm-là-gì)
- [2. Tại sao LightGBM nhanh? (ý tưởng cốt lõi)](#2-tại-sao-lightgbm-nhanh-ý-tưởng-cốt-lõi)
- [3. LightGBM làm được gì? (bài toán & objective)](#3-lightgbm-làm-được-gì-bài-toán--objective)
- [4. Các đặc điểm nổi bật](#4-các-đặc-điểm-nổi-bật)
- [5. Tham số quan trọng (cheat sheet)](#5-tham-số-quan-trọng-cheat-sheet)
- [6. Quy trình train chuẩn (thực chiến)](#6-quy-trình-train-chuẩn-thực-chiến)
- [7. Categorical & Missing Values: làm đúng ngay từ đầu](#7-categorical--missing-values-làm-đúng-ngay-từ-đầu)
- [8. Ví dụ code nhanh (Python)](#8-ví-dụ-code-nhanh-python)
- [9. Diễn giải mô hình (interpretability)](#9-diễn-giải-mô-hình-interpretability)
- [10. Ưu & nhược điểm](#10-ưu--nhược-điểm)
- [11. So sánh LightGBM vs XGBoost](#11-so-sánh-lightgbm-vs-xgboost)
- [12. Những “bẫy” hay gặp & checklist debug](#12-những-bẫy-hay-gặp--checklist-debug)
- [13. Tài liệu tham khảo](#13-tài-liệu-tham-khảo)

---

## 1. LightGBM là gì?

**LightGBM (Light Gradient Boosting Machine)** là một framework thuộc họ **Gradient Boosting Decision Trees (GBDT)**.  Nó xây dựng mô hình bằng cách **cộng dồn nhiều cây quyết định** (decision trees) theo kiểu *boosting*: mỗi cây mới cố gắng **sửa lỗi** (giảm loss) mà các cây trước còn mắc phải.

> ✅ LightGBM nổi tiếng vì: **train nhanh**, **tốn ít RAM**, **scale tốt** cho dữ liệu lớn / nhiều feature, và vẫn cho chất lượng mô hình rất mạnh trên dữ liệu dạng bảng.

---

## 2. Tại sao LightGBM nhanh? (ý tưởng cốt lõi)

### 2.1 Histogram-based split (chia ngưỡng theo histogram)
- Thay vì thử mọi giá trị liên tục để tìm split tốt nhất, LightGBM **bucket hóa** giá trị feature thành các **bins** (histogram).  
➡️ Giảm rất mạnh số phép tính và giảm bộ nhớ.

**Hiệu quả thực tế:**
- Dữ liệu càng lớn → càng thấy lợi thế rõ.
- Có thể điều chỉnh `max_bin` để trade-off **nhanh** vs **chính xác**.

---

### 2.2 Leaf-wise (best-first) tree growth — điểm khác biệt lớn
Nhiều GBDT grow cây theo kiểu **level-wise** (tăng theo tầng, cây cân đối).  
LightGBM grow kiểu **leaf-wise**: **luôn split lá nào giảm loss nhiều nhất trước**.

**Ưu điểm:** thường cho accuracy tốt hơn với cùng số split.  
**Nhược điểm:** **dễ overfit** nếu không giới hạn độ phức tạp (vì cây có thể rất “sâu” ở một nhánh).

👉 Vì vậy trong tune, bạn gần như luôn phải để ý: `num_leaves`, `max_depth`, `min_data_in_leaf`.

✨ Một số chỉ số cần nhớ khi làm việc với LightGBM:

- **objective**: Tham số này dùng để xác định mục tiêu của bài toán mà bạn đang cố gắng giải quyết. Những giá trị thường gặp gồm có ‘binary’ (cho các bài toán phân loại chỉ có hai lớp), ‘multiclass’ (cho bài toán phân loại có nhiều hơn hai lớp), và ‘regression’ (cho các bài toán dự đoán giá trị liên tục). ---------- VN WEATHER HUUB ĐANG SỬ DỤNG "REGESTION" VÌ DỰ BÁO THỜI TIẾT.

- **metric**: Được sử dụng để chỉ định thước đo (chỉ số) bạn muốn dùng để đánh giá chất lượng của mô hình trong quá trình huấn luyện hoặc kiểm thử. Ví dụ, bạn có thể chọn ‘binary_error’ khi làm việc với bài toán phân loại nhị phân.

- **num_leaves**: Tham số này quy định số lượng lá tối đa cho mỗi cây quyết định được tạo ra. Nó là một yếu tố quan trọng ảnh hưởng trực tiếp đến mức độ phức tạp (complexity) của mô hình được huấn luyện.

- **learning_rate**: Còn gọi là tốc độ học, tham số này kiểm soát mức độ điều chỉnh trọng số của mô hình sau mỗi vòng lặp boosting, qua đó tác động đến tốc độ hội tụ của mô hình (học nhanh hay chậm).

- **feature_fraction**: Tham số này xác định tỷ lệ (phần trăm) các đặc trưng sẽ được lựa chọn ngẫu nhiên để sử dụng trong quá trình xây dựng mỗi cây quyết định riêng lẻ.

- **bagging_fraction** và **bagging_freq**: Bộ đôi tham số này cho phép bạn kích hoạt kỹ thuật bagging. bagging_fraction quy định tỷ lệ mẫu dữ liệu huấn luyện được chọn ngẫu nhiên (có lặp lại) cho mỗi cây, và bagging_freq xác định tần suất thực hiện bagging (ví dụ: thực hiện bagging sau mỗi k vòng lặp). Mục đích chính của bagging là giúp mô hình giảm thiểu hiện tượng quá khớp (overfitting).

---

### 2.3 GOSS — Gradient-based One-Side Sampling
**GOSS** lấy mẫu thông minh:
- giữ nhiều điểm có **gradient lớn** (điểm “khó”, ảnh hưởng lớn đến loss),
- sampling phần gradient nhỏ.

➡️ Mục tiêu: giảm dữ liệu cần tính split nhưng vẫn giữ “thông tin quan trọng” cho boosting.

---

### 2.4 EFB — Exclusive Feature Bundling
**EFB** gộp các feature “hiếm khi cùng khác 0” (thường gặp ở dữ liệu sparse / one-hot / text-like).  
➡️ Giảm số chiều hiệu quả → giảm chi phí tính histogram/split.

---

## 3. LightGBM làm được gì? (bài toán & objective)

LightGBM thường dùng cho:

- ✅ **Classification**: nhị phân (`binary`), đa lớp (`multiclass`)
- ✅ **Regression**: dự đoán số thực (`regression`, `regression_l1`, …)
- ✅ **Ranking**: xếp hạng (trong search/recommendation) (`lambdarank`)
- ✅ **Quantile / Poisson / Tweedie**… (dữ liệu đặc thù)

---

## 4. Các đặc điểm nổi bật

- ⚡ **Hiệu năng vượt trội**: train nhanh, ít RAM (đặc biệt với histogram + EFB).
- 🧱 **Xử lý dữ liệu lớn**: nhiều mẫu & nhiều feature.
- 🧠 **Đa năng**: phân loại + hồi quy + ranking.
- 🧵 **Hỗ trợ song song & phân tán**: multi-core CPU, training phân tán (tuỳ setup).
- 🎛️ **Linh hoạt tham số**: tune sâu để tối ưu theo từng bài toán.
- 🕳️ **Xử lý missing value tốt**: nhiều trường hợp không cần impute phức tạp.
- 🏷️ **Hỗ trợ categorical**: nếu khai báo đúng kiểu/cột categorical.

---

## 5. Tham số quan trọng (cheat sheet)

> Nếu phải nhớ **5 tham số quan trọng nhất**:  
> `learning_rate`, `n_estimators`, `num_leaves`, `min_data_in_leaf`, `feature_fraction/bagging_fraction`

### 5.1 Nhóm mục tiêu & metric
- `objective`: mục tiêu bài toán  
  - `binary`, `multiclass`, `regression`, …
- `metric`: thước đo trong training/validation  
  - `auc`, `binary_logloss`, `rmse`, `mae`, …

### 5.2 Nhóm số cây & tốc độ học
- `n_estimators` / `num_boost_round`: số cây
- `learning_rate`: tốc độ học  
  - nhỏ hơn → cần nhiều cây hơn, thường ổn định hơn

### 5.3 Nhóm độ phức tạp cây (cực quan trọng vì leaf-wise)
- `num_leaves`: số lá tối đa mỗi cây (**top 1**)  
- `max_depth`: giới hạn độ sâu (giảm overfit)
- `min_data_in_leaf` (aka `min_child_samples`): tối thiểu mẫu trong 1 lá  
- `min_sum_hessian_in_leaf`: làm lá ổn định hơn (nhất là dữ liệu nhiễu)

### 5.4 Nhóm sampling để giảm overfit
- `feature_fraction` (aka `colsample_bytree`): % feature mỗi cây
- `bagging_fraction` (aka `subsample`): % sample mỗi cây
- `bagging_freq` (aka `subsample_freq`): tần suất bagging

### 5.5 Nhóm regularization
- `lambda_l1` (aka `reg_alpha`)
- `lambda_l2` (aka `reg_lambda`)
- `min_gain_to_split`: gain tối thiểu mới được split

### 5.6 Nhóm histogram
- `max_bin`: số bins cho histogram (nhỏ hơn → nhanh hơn, có thể giảm accuracy)

---

## 6. Quy trình train chuẩn (thực chiến)

### 6.1 “Recipe” nhanh cho đa số bài tabular
1. Chia `train/valid` chuẩn (hoặc K-fold CV).  
2. Train với `early_stopping` để tìm số cây tối ưu.  
3. Tune theo thứ tự:
   - **(A) Tree complexity:** `num_leaves`, `max_depth`, `min_data_in_leaf`
   - **(B) Sampling:** `feature_fraction`, `bagging_fraction`, `bagging_freq`
   - **(C) Regularization + histogram:** `lambda_l1/l2`, `min_gain_to_split`, `max_bin`
4. Chốt lại bằng CV, rồi train full train với `best_iteration`.

### 6.2 Quy tắc tránh overfit (rất hay gặp)
- Nếu **train rất tốt, valid tệ**:
  - giảm `num_leaves`
  - tăng `min_data_in_leaf`
  - thêm sampling (`feature_fraction`, `bagging_fraction`)
  - tăng `lambda_l2` (hoặc `lambda_l1`)
  - giới hạn `max_depth`

---

## 7. Categorical & Missing Values: làm đúng ngay từ đầu

### 7.1 Missing values
- LightGBM thường xử lý missing trực tiếp khá tốt.
- Tuy nhiên, nếu missing mang ý nghĩa riêng, bạn có thể thêm feature `is_missing`.

### 7.2 Categorical — lỗi hay gặp nhất
✅ Nên:
- chuyển cột categorical về kiểu `category` (pandas) hoặc integer-coded,
- khai báo `categorical_feature`.

❌ Tránh:
- label-encode xong **nhưng quên set categorical** → model coi như biến số liên tục (split numeric) → sai bản chất.

> Với categorical có **cardinality rất cao**: cân nhắc gộp rare categories / hashing / target encoding (cẩn thận leakage).

---

## 8. Ví dụ code nhanh (Python)

### 8.1 Cài đặt
```bash
pip install lightgbm
# hoặc
conda install -c conda-forge lightgbm
```

### 8.2 Sklearn API — Classification (nhị phân)
```python
import lightgbm as lgb
from lightgbm import LGBMClassifier

model = LGBMClassifier(
    n_estimators=5000,
    learning_rate=0.05,
    num_leaves=64,
    max_depth=-1,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=0.0,
    random_state=42,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(stopping_rounds=200)]
)

print("best_iteration =", model.best_iteration_)
```

### 8.3 Regression
```python
from lightgbm import LGBMRegressor
import lightgbm as lgb

reg = LGBMRegressor(
    n_estimators=10000,
    learning_rate=0.03,
    num_leaves=64,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=42,
)

reg.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(300)]
)
```

### 8.4 Categorical đúng cách (pandas)
```python
cat_cols = ["city", "channel", "product_type"]

for c in cat_cols:
    X_train[c] = X_train[c].astype("category")
    X_valid[c] = X_valid[c].astype("category")

model = LGBMClassifier(
    n_estimators=5000,
    learning_rate=0.05,
    num_leaves=64,
    random_state=42,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    categorical_feature=cat_cols,
    callbacks=[lgb.early_stopping(200)]
)
```

### 8.5 Native LightGBM API (Dataset + train)
```python
import lightgbm as lgb

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 30,
}

bst = lgb.train(
    params,
    train_data,
    num_boost_round=5000,
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(200)]
)

print("best_iteration =", bst.best_iteration)
```

---

## 9. Diễn giải mô hình (interpretability)

### 9.1 Feature importance
- `gain`: tổng gain do feature tạo ra  
- `split`: số lần feature được dùng để split

Trong sklearn API:
```python
import pandas as pd

imp = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)
print(imp.head(20))
```

### 9.2 SHAP (giải thích theo từng dự đoán)
- SHAP rất hữu ích để hiểu “vì sao model dự đoán như vậy”
- Cẩn thận với dữ liệu lớn: SHAP có thể tốn thời gian

```python
# pip install shap
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_valid)
```

---

## 10. Ưu & nhược điểm

### ✅ Ưu điểm
- ⚡ **Nhanh & tiết kiệm bộ nhớ** (rất mạnh với data lớn): LightGBM nổi bật với khả năng xử lý nhanh chóng và hiệu quả các tập dữ liệu có dung lượng lớn. Nhờ áp dụng các thuật toán được tối ưu hóa, LightGBM yêu cầu ít bộ nhớ hơn so với khi sử dụng các thuật toán boosting thế hệ trước.
- 🧵 **Song song tốt** trên CPU: LightGBM có thể rút ngắn đáng kể thời gian huấn luyện mô hình thông qua việc khai thác hiệu quả khả năng tính toán song song.
- 🧠 **Chất lượng cao** trên tabular
- 🎛️ **Nhiều tham số** để tối ưu theo bài toán: : Người dùng có được sự linh hoạt cao do LightGBM cung cấp một loạt các tham số có thể được điều chỉnh để tối ưu hóa mô hình cho từng bài toán cụ thể.
- 🏷️ **Hỗ trợ categorical/missing** tốt nếu khai báo đúng

### ❗ Nhược điểm
- 🧩 **Dễ overfit** nếu `num_leaves` lớn, thiếu ràng buộc (do leaf-wise)
- 🛠️ **Cần tune tham số** để đạt hiệu quả tối đa
**==> Đòi hỏi tinh chỉnh tham số kỹ lưỡng: Để LightGBM hoạt động với hiệu quả cao nhất, việc điều chỉnh cẩn thận nhiều tham số cấu hình là một yêu cầu cần thiết.**
- 🧠 **Diễn giải không trực quan** với người mới (cần tools như SHAP): Mô hình do LightGBM tạo ra có thể khó hiểu và diễn giải, đặc biệt đối với những cá nhân mới bắt đầu tìm hiểu về lĩnh vực học máy.

---

## 11. So sánh LightGBM vs XGBoost

<table>
  <tr>
    <th>Tiêu chí</th>
    <th>LightGBM</th>
    <th>XGBoost</th>
  </tr>
  <tr>
    <td><b>Tốc độ train</b></td>
    <td>Thường <b>nhanh hơn</b> (histogram + EFB + leaf-wise)</td>
    <td>Nhanh, nhưng nhiều case chậm hơn LGBM</td>
  </tr>
  <tr>
    <td><b>RAM</b></td>
    <td>Thường <b>tối ưu hơn</b> trên dữ liệu lớn</td>
    <td>Ổn, nhưng có thể tốn hơn trên data rất lớn</td>
  </tr>
  <tr>
    <td><b>Chiến lược grow cây</b></td>
    <td><b>Leaf-wise</b> (best-first) → mạnh nhưng dễ overfit</td>
    <td>Thường level-wise / controlled → ổn định hơn</td>
  </tr>
  <tr>
    <td><b>Dữ liệu nhỏ</b></td>
    <td>Có thể overfit nếu tune chưa tốt</td>
    <td>Thường ổn định hơn</td>
  </tr>
  <tr>
    <td><b>Categorical</b></td>
    <td>Có hỗ trợ (cần khai báo đúng)</td>
    <td>Hỗ trợ nhưng thường phụ thuộc preprocessing (one-hot/encoding)</td>
  </tr>
</table>

**Tóm lại chọn gì?**
- Chọn **LightGBM** nếu bạn ưu tiên **tốc độ**, **scale**, data lớn/nhiều feature.
- Chọn **XGBoost** nếu bạn cần **tính ổn định** cao, dataset nhỏ-vừa, hoặc workflow tune đã quen.

> Gợi ý thêm: nếu bài toán có categorical “khó” (nhiều giá trị hiếm, high-cardinality), đôi khi **CatBoost** là lựa chọn rất đáng thử.

---

## 12. Những “bẫy” hay gặp & checklist debug

### 12.1 Bẫy phổ biến
- 🔥 **Leakage** (đặc biệt time-series): feature chứa thông tin tương lai.
- 🌿 `num_leaves` quá lớn, thiếu ràng buộc → overfit.
- 🧪 Valid split sai:  
  - time-series mà shuffle  
  - dữ liệu theo user mà split lẫn user giữa train/valid
- 🏷️ Categorical sai: quên `categorical_feature`.
- 🎯 Metric không đúng mục tiêu (AUC vs F1 vs logloss…).

### 12.2 Checklist nhanh khi “điểm tụt”
- [ ] Split đúng kiểu dữ liệu? (time/user/group)
- [ ] Early stopping có dùng chưa?
- [ ] `num_leaves` có quá lớn không?
- [ ] `min_data_in_leaf` có quá nhỏ không?
- [ ] Sampling (`feature_fraction`, `bagging_fraction`) đã bật?
- [ ] Categorical khai báo đúng?
- [ ] Có leakage trong feature engineering?

---

## 13. Tài liệu tham khảo

- LightGBM Paper (NIPS 2017): “LightGBM: A Highly Efficient Gradient Boosting Decision Tree”
- Official LightGBM docs: Parameters / Tuning / Advanced Topics
- XGBoost docs & paper

<details>

</details>

---

<div align="center">
</div>