# CatBoost - Thuật Toán Gradient Boosting Cho Dữ Liệu Phân Loại

## Mục Lục
1. [Giới thiệu tổng quan](#1-giới-thiệu-tổng-quan)
2. [CatBoost là gì?](#2-catboost-là-gì)
3. [Lịch sử phát triển](#3-lịch-sử-phát-triển)
4. [Cơ chế hoạt động chi tiết](#4-cơ-chế-hoạt-động-chi-tiết)
5. [Các tính năng nổi bật](#5-các-tính-năng-nổi-bật)
6. [Các tham số quan trọng](#6-các-tham-số-quan-trọng)
7. [So sánh với các thuật toán khác](#7-so-sánh-với-các-thuật-toán-khác)
8. [Ưu điểm và hạn chế](#8-ưu-điểm-và-hạn-chế)
9. [Ứng dụng thực tế](#9-ứng-dụng-thực-tế)
10. [Hướng dẫn cài đặt và sử dụng](#10-hướng-dẫn-cài-đặt-và-sử-dụng)
11. [Ví dụ thực hành](#11-ví-dụ-thực-hành)
12. [Best Practices](#12-best-practices)
13. [Tài liệu tham khảo](#13-tài-liệu-tham-khảo)

---

## 1. Giới Thiệu Tổng Quan

Trong lĩnh vực học máy (Machine Learning), việc chọn đúng thuật toán và công cụ để giải quyết các bài toán dữ liệu phức tạp là vô cùng quan trọng. **CatBoost** (Categorical Boosting) là một trong những thư viện học máy mạnh mẽ và hiệu quả nhất hiện nay, được thiết kế đặc biệt để xử lý dữ liệu có chứa các đặc trưng phân loại (categorical features) một cách tự động và hiệu quả.

CatBoost đặc biệt nổi bật trong việc:
- Xử lý dữ liệu dạng cây quyết định (decision trees)
- Tối ưu hóa cực kỳ hiệu quả cho các bài toán phân loại và hồi quy
- Giảm thiểu hiện tượng overfitting
- Không yêu cầu nhiều công sức tinh chỉnh tham số

---

## 2. CatBoost Là Gì?

### 2.1. Định nghĩa

**CatBoost** (viết tắt của **Cat**egorical **Boost**ing) là một thư viện học máy mã nguồn mở được xây dựng để giải quyết các vấn đề trong học máy bằng cách sử dụng các thuật toán boosting dựa trên cây quyết định (decision tree).

### 2.2. Đặc điểm chính

CatBoost là một thuật toán thuộc họ **Gradient Boosting**, nhưng được tích hợp thêm những cải tiến đáng kể liên quan đến:
- **Hiệu suất hoạt động**: Tối ưu hóa tốc độ huấn luyện
- **Khả năng tổng quát hóa**: Giảm thiểu overfitting
- **Xử lý dữ liệu phân loại**: Không cần mã hóa thủ công

### 2.3. Điểm khác biệt

Một điểm khác biệt quan trọng so với nhiều thư viện học máy khác là CatBoost thể hiện sức mạnh đặc biệt khi làm việc với **dữ liệu có đặc trưng dạng phân loại** (categorical data). Nhờ vậy, CatBoost trở thành một sự lựa chọn xuất sắc khi cần xử lý những bộ dữ liệu phức tạp chứa nhiều biến phân loại.

---

## 3. Lịch Sử Phát Triển

### 3.1. Nguồn gốc

CatBoost được phát triển bởi **Yandex** - một công ty công nghệ nổi tiếng của Nga, chuyên về công cụ tìm kiếm và các dịch vụ internet.

### 3.2. Mục tiêu phát triển

Yandex phát triển CatBoost nhằm:
- Cải thiện các hệ thống dự đoán nội bộ
- Xử lý hiệu quả dữ liệu có nhiều đặc trưng phân loại
- Tăng tốc độ huấn luyện mô hình
- Giảm thiểu công sức tinh chỉnh hyperparameters

### 3.3. Phát hành

CatBoost được công bố vào năm **2017** và nhanh chóng trở thành một trong những thư viện gradient boosting được ưa chuộng nhất trong cộng đồng machine learning.

---

## 4. Cơ Chế Hoạt Động Chi Tiết

### 4.1. Nguyên lý Gradient Boosting

CatBoost dựa trên nguyên lý **Gradient Boosting** - một kỹ thuật ensemble learning trong đó:

1. **Xây dựng tuần tự**: Các mô hình yếu (weak learners) được xây dựng tuần tự
2. **Học từ sai số**: Mỗi mô hình mới tập trung vào việc sửa chữa sai số của các mô hình trước
3. **Kết hợp kết quả**: Kết quả cuối cùng là tổng hợp có trọng số của tất cả các mô hình

### 4.2. Quá trình huấn luyện

Tại mỗi vòng lặp của thuật toán, CatBoost thực hiện:

```
Bước 1: Tính toán gradient âm của hàm mất mát đối với các dự đoán hiện tại
Bước 2: Sử dụng gradient này để cập nhật các dự đoán
Bước 3: Cộng một phiên bản đã được điều chỉnh của gradient vào các dự đoán hiện tại
Bước 4: Chọn yếu tố điều chỉnh bằng thuật toán line search nhằm tối thiểu hóa hàm mất mát
```

### 4.3. Tối ưu hóa dựa trên Gradient

Để xây dựng các cây quyết định, CatBoost sử dụng kỹ thuật **tối ưu hóa dựa trên gradient**:

- Các cây được điều chỉnh để phù hợp với gradient âm của hàm mất mát
- Giúp các cây tập trung vào các vùng không gian đặc trưng có ảnh hưởng lớn nhất đến hàm mất mát
- Từ đó mang lại các dự đoán chính xác hơn

### 4.4. Ordered Boosting

CatBoost giới thiệu một thuật toán mới gọi là **Ordered Boosting**:

```
┌─────────────────────────────────────────────────────────────┐
│                    ORDERED BOOSTING                         │
├─────────────────────────────────────────────────────────────┤
│ 1. Hoán đổi các đặc trưng theo một thứ tự cụ thể            │
│ 2. Tối ưu hóa hàm mục tiêu học                              │
│ 3. Giúp việc hội tụ nhanh hơn                               │
│ 4. Cải thiện độ chính xác, đặc biệt với bộ dữ liệu lớn      │
└─────────────────────────────────────────────────────────────┘
```

### 4.5. Xử lý đặc trưng phân loại

CatBoost sử dụng kỹ thuật **Target Statistics** để mã hóa categorical features:

```python
# Công thức tính Target Statistics
target_stat = (count_in_category * mean_target + prior * global_mean) / (count_in_category + prior)
```

Trong đó:
- `count_in_category`: Số lượng mẫu trong category
- `mean_target`: Giá trị target trung bình trong category
- `prior`: Tham số điều chỉnh (smoothing parameter)
- `global_mean`: Giá trị target trung bình toàn cục

### 4.6. Symmetric Trees

CatBoost sử dụng **Symmetric Decision Trees** (cây quyết định đối xứng):

```
                    Root
                   /    \
            Split_1      Split_1
             /  \         /  \
        Split_2 Split_2 Split_2 Split_2
         / \     / \     / \     / \
        L1 L2   L3 L4   L5 L6   L7 L8
```

Đặc điểm:
- Cùng một điều kiện split được áp dụng cho tất cả các nodes ở cùng level
- Tăng tốc độ inference
- Giảm overfitting

---

## 5. Các Tính Năng Nổi Bật

### 5.1. Khả năng xử lý dữ liệu phân loại ưu việt

| Thuật toán khác | CatBoost |
|-----------------|----------|
| Yêu cầu one-hot encoding | Xử lý trực tiếp categorical features |
| Yêu cầu label encoding | Không cần mã hóa thủ công |
| Có thể mất thông tin khi encoding | Bảo toàn thông tin category |
| Tốn thời gian tiền xử lý | Tiết kiệm thời gian |

### 5.2. Cơ chế chống Overfitting hiệu quả

CatBoost cung cấp nhiều cơ chế chống overfitting:

1. **Ordered Boosting**: Sử dụng permutation để tránh target leakage
2. **Ordered Target Statistics**: Tính toán target statistics theo thứ tự
3. **Random permutations**: Sử dụng nhiều permutation khác nhau
4. **Early Stopping**: Dừng huấn luyện khi không còn cải thiện

### 5.3. Tốc độ huấn luyện được tối ưu hóa

```
┌────────────────────────────────────────────────┐
│         TỐC ĐỘ HUẤN LUYỆN CATBOOST             │
├────────────────────────────────────────────────┤
│ ✓ Xử lý song song hiệu quả (parallel)          │
│ ✓ Phương pháp tối ưu hóa độc quyền             │
│ ✓ Hỗ trợ huấn luyện GPU                        │
│ ✓ Hỗ trợ nhiều GPU cùng lúc                    │
│ ✓ Symmetric trees cho inference nhanh          │
└────────────────────────────────────────────────┘
```

### 5.4. Ít yêu cầu tinh chỉnh tham số

- **Tham số mặc định tốt**: Các giá trị mặc định thường đã cho kết quả rất tốt
- **Auto-tuning**: Một số tham số được tự động điều chỉnh
- **Thân thiện người mới**: Không cần kiến thức sâu về hyperparameter tuning

### 5.5. Hỗ trợ đa dạng các loại bài toán

| Loại bài toán | Mô tả | Ví dụ |
|---------------|-------|-------|
| **Classification** | Phân loại nhị phân hoặc đa lớp | Spam detection, Image classification |
| **Regression** | Dự đoán giá trị số liên tục | Dự báo giá, Dự đoán doanh thu |
| **Ranking** | Xếp hạng items | Search ranking, Recommendation |
| **Multi-output** | Nhiều output cùng lúc | Multi-label classification |

---

## 6. Các Tham Số Quan Trọng

### 6.1. Tham số cơ bản

```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    iterations=1000,        # Số vòng lặp boosting (số cây)
    depth=6,                # Độ sâu tối đa của mỗi cây
    learning_rate=0.03,     # Tốc độ học
    loss_function='Logloss', # Hàm mất mát
    cat_features=[0, 2, 5], # Chỉ số các cột categorical
    verbose=100             # In log mỗi 100 iterations
)
```

### 6.2. Chi tiết các tham số

#### **iterations** (int, default=1000)
- Số vòng lặp boosting
- Tương ứng với số lượng cây quyết định
- Giá trị cao hơn → mô hình phức tạp hơn → có thể overfitting

```python
# Khuyến nghị
iterations = 500  # Cho bộ dữ liệu nhỏ
iterations = 1000 # Cho bộ dữ liệu trung bình
iterations = 3000 # Cho bộ dữ liệu lớn (kết hợp early stopping)
```

#### **depth** (int, default=6)
- Độ sâu tối đa của mỗi cây quyết định
- Giá trị cao → capture được quan hệ phức tạp → dễ overfitting

```python
# Khuyến nghị
depth = 4  # Cho bộ dữ liệu nhỏ, tránh overfitting
depth = 6  # Giá trị mặc định, cân bằng
depth = 10 # Cho bộ dữ liệu lớn với quan hệ phức tạp
```

#### **learning_rate** (float, default=0.03)
- Tốc độ học, kiểm soát mức độ đóng góp của mỗi cây
- Giá trị nhỏ → hội tụ chậm nhưng ổn định
- Giá trị lớn → hội tụ nhanh nhưng có thể bỏ qua optimal point

```python
# Mối quan hệ với iterations
learning_rate = 0.03  # iterations = 1000
learning_rate = 0.01  # iterations = 3000
learning_rate = 0.1   # iterations = 300
```

#### **loss_function** (string)
- Hàm mất mát để đánh giá và tối ưu hóa

| Bài toán | Loss Function | Mô tả |
|----------|---------------|-------|
| Binary Classification | `Logloss` | Log loss (cross-entropy) |
| Multi-class Classification | `MultiClass` | Multi-class cross-entropy |
| Regression | `RMSE` | Root Mean Squared Error |
| Regression | `MAE` | Mean Absolute Error |
| Ranking | `YetiRank` | Yandex ranking loss |

#### **cat_features** (list)
- Danh sách chỉ số hoặc tên các cột categorical
- CatBoost sẽ tự động xử lý các features này

```python
# Theo chỉ số
cat_features = [0, 2, 5]

# Theo tên cột
cat_features = ['gender', 'city', 'category']
```

### 6.3. Tham số nâng cao

```python
model = CatBoostClassifier(
    # Regularization
    l2_leaf_reg=3.0,           # L2 regularization
    random_strength=1.0,       # Random noise cho scores
    bagging_temperature=1.0,   # Bayesian bootstrap strength
    
    # Xử lý missing values
    nan_mode='Min',            # 'Min', 'Max', hoặc 'Forbidden'
    
    # Tối ưu hóa
    bootstrap_type='Bayesian', # 'Bayesian', 'Bernoulli', 'MVS'
    grow_policy='SymmetricTree', # 'SymmetricTree', 'Depthwise', 'Lossguide'
    
    # Early stopping
    early_stopping_rounds=50,  # Dừng nếu không cải thiện sau 50 rounds
    
    # GPU
    task_type='GPU',           # Sử dụng GPU
    devices='0:1',             # Sử dụng GPU 0 và 1
)
```

---

## 7. So Sánh Với Các Thuật Toán Khác

### 7.1. CatBoost vs XGBoost vs LightGBM

| Tiêu chí | CatBoost | XGBoost | LightGBM |
|----------|----------|---------|----------|
| **Xử lý Categorical** | Tự động, hiệu quả | Cần encoding thủ công | Hỗ trợ cơ bản |
| **Tốc độ huấn luyện** | Nhanh | Trung bình | Rất nhanh |
| **Chống Overfitting** | Rất tốt (Ordered Boosting) | Tốt | Tốt |
| **Độ chính xác** | Cao | Cao | Cao |
| **Hyperparameter tuning** | Ít cần thiết | Cần nhiều | Cần trung bình |
| **GPU support** | Tốt | Tốt | Tốt |
| **Cây quyết định** | Symmetric | Không giới hạn | Leaf-wise |

### 7.2. Khi nào nên dùng CatBoost?

✅ **Nên dùng CatBoost khi:**
- Dữ liệu có nhiều features categorical
- Cần kết quả tốt mà không muốn tune nhiều
- Quan tâm đến việc chống overfitting
- Muốn tiết kiệm thời gian tiền xử lý

❌ **Không nên dùng CatBoost khi:**
- Dữ liệu chỉ có features số (numerical)
- Cần mô hình nhẹ cho production
- Bộ dữ liệu rất nhỏ
- Cần giải thích chi tiết từng feature

---

## 8. Ưu Điểm Và Hạn Chế

### 8.1. Ưu điểm

```
┌─────────────────────────────────────────────────────────────┐
│                       ƯU ĐIỂM CATBOOST                       │
├─────────────────────────────────────────────────────────────┤
│ ★ Hiệu năng cao                                             │
│   - Độ chính xác tốt trên nhiều loại bài toán               │
│   - Đặc biệt mạnh với dữ liệu categorical                   │
│                                                             │
│ ★ Hạn chế Overfitting hiệu quả                              │
│   - Ordered Boosting giúp tổng quát hóa tốt                 │
│   - Ít cần early stopping thủ công                          │
│                                                             │
│ ★ Thân thiện với người dùng                                 │
│   - API đơn giản, dễ sử dụng                                │
│   - Tham số mặc định đã rất tốt                             │
│                                                             │
│ ★ Xử lý dữ liệu phân loại tự động                           │
│   - Không cần one-hot encoding                              │
│   - Tiết kiệm thời gian tiền xử lý                          │
│                                                             │
│ ★ Hỗ trợ GPU                                                │
│   - Tăng tốc huấn luyện đáng kể                             │
│   - Hỗ trợ multi-GPU                                        │
└─────────────────────────────────────────────────────────────┘
```

### 8.2. Hạn chế

```
┌─────────────────────────────────────────────────────────────┐
│                      HẠN CHẾ CATBOOST                        │
├─────────────────────────────────────────────────────────────┤
│ ✗ Yêu cầu tài nguyên tính toán                              │
│   - Bộ dữ liệu lớn + iterations cao = tốn RAM/CPU           │
│   - Cần GPU cho tốc độ tối ưu                               │
│                                                             │
│ ✗ Độ phức tạp về cơ chế hoạt động                           │
│   - Khó hiểu sâu các cơ chế bên trong                       │
│   - Yêu cầu kiến thức về gradient boosting                  │
│                                                             │
│ ✗ Model size                                                │
│   - File model có thể lớn                                   │
│   - Inference chậm hơn so với linear models                 │
│                                                             │
│ ✗ Không phù hợp cho streaming data                          │
│   - Cần retrain toàn bộ khi có dữ liệu mới                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. Ứng Dụng Thực Tế

### 9.1. Hệ thống đề xuất (Recommendation Systems)

```python
# Gợi ý sản phẩm, phim, âm nhạc dựa trên hành vi người dùng
features = ['user_id', 'item_category', 'time_of_day', 'device_type']
cat_features = ['item_category', 'device_type']
```

### 9.2. Phát hiện gian lận (Fraud Detection)

```python
# Phát hiện giao dịch gian lận trong thẻ tín dụng, bảo hiểm
features = ['transaction_amount', 'merchant_category', 'location', 'time']
cat_features = ['merchant_category', 'location']
```

### 9.3. Phân loại hình ảnh và văn bản

```python
# Phân loại spam/không spam, cảm xúc tích cực/tiêu cực
features = ['word_count', 'special_chars', 'sender_domain', 'subject_keywords']
cat_features = ['sender_domain']
```

### 9.4. Dự đoán khách hàng rời bỏ (Customer Churn)

```python
# Dự đoán khách hàng sẽ ngừng sử dụng dịch vụ
features = ['tenure', 'contract_type', 'payment_method', 'monthly_charges']
cat_features = ['contract_type', 'payment_method']
```

### 9.5. Chẩn đoán y tế (Medical Diagnosis)

```python
# Dự đoán khả năng mắc bệnh dựa trên triệu chứng và tiền sử
features = ['age', 'gender', 'symptoms', 'medical_history', 'test_results']
cat_features = ['gender', 'symptoms', 'medical_history']
```

### 9.6. Xử lý ngôn ngữ tự nhiên (NLP)

```python
# Phân tích sentiment, chatbot responses
features = ['text_length', 'word_embeddings', 'source_platform', 'language']
cat_features = ['source_platform', 'language']
```

### 9.7. Dự báo thời tiết (Weather Forecasting)

```python
# Dự đoán nhiệt độ, lượng mưa, điều kiện thời tiết
features = ['temperature', 'humidity', 'wind_speed', 'season', 'location', 'weather_type']
cat_features = ['season', 'location', 'weather_type']
```

### 9.8. Dự báo chuỗi thời gian (Time Series Forecasting)

```python
# Dự đoán giá cổ phiếu, lưu lượng giao thông
features = ['historical_values', 'day_of_week', 'month', 'holiday_indicator']
cat_features = ['day_of_week', 'month', 'holiday_indicator']
```

---

## 10. Hướng Dẫn Cài Đặt Và Sử Dụng

### 10.1. Cài đặt

#### Sử dụng pip (khuyến nghị)
```bash
pip install catboost
```

#### Sử dụng conda
```bash
conda install -c conda-forge catboost
```

#### Cài đặt với GPU support
```bash
pip install catboost-gpu
```

### 10.2. Kiểm tra cài đặt

```python
import catboost
print(f"CatBoost version: {catboost.__version__}")

# Kiểm tra GPU support
from catboost import CatBoostClassifier
model = CatBoostClassifier(task_type='GPU')
print("GPU support: Available")
```

### 10.3. Import cơ bản

```python
# Import các class chính
from catboost import CatBoostClassifier  # Cho bài toán phân loại
from catboost import CatBoostRegressor   # Cho bài toán hồi quy
from catboost import CatBoostRanker      # Cho bài toán ranking
from catboost import Pool                # Để tạo dataset

# Import utilities
from catboost import cv                  # Cross-validation
from catboost import sum_models          # Combine models
```

---

## 11. Ví Dụ Thực Hành

### 11.1. Bài toán phân loại nhị phân

```python
import catboost
from catboost import CatBoostClassifier, Pool
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Tải và chuẩn bị dữ liệu
data = load_breast_cancer()
X = data.data
y = data.target

# 2. Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Khởi tạo mô hình
model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function='Logloss',
    verbose=100,
    random_state=42
)

# 4. Huấn luyện
model.fit(X_train, y_train, eval_set=(X_test, y_test))

# 5. Dự đoán
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# 6. Đánh giá
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
```

### 11.2. Bài toán với Categorical Features

```python
import pandas as pd
from catboost import CatBoostClassifier, Pool

# 1. Tạo dữ liệu mẫu với categorical features
data = pd.DataFrame({
    'age': [25, 30, 35, 40, 45, 50, 55, 60],
    'gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'city': ['Hanoi', 'HCMC', 'Danang', 'Hanoi', 'HCMC', 'Danang', 'Hanoi', 'HCMC'],
    'income': [500, 800, 600, 900, 700, 1000, 800, 1200],
    'purchased': [0, 1, 0, 1, 0, 1, 1, 1]
})

# 2. Chuẩn bị features và target
X = data.drop('purchased', axis=1)
y = data['purchased']

# 3. Xác định categorical features
cat_features = ['gender', 'city']

# 4. Tạo Pool object (khuyến nghị cho categorical data)
train_pool = Pool(
    data=X,
    label=y,
    cat_features=cat_features
)

# 5. Huấn luyện
model = CatBoostClassifier(
    iterations=100,
    depth=4,
    learning_rate=0.1,
    verbose=False
)
model.fit(train_pool)

# 6. Dự đoán
predictions = model.predict(X)
print(f"Predictions: {predictions}")
```

### 11.3. Bài toán hồi quy

```python
from catboost import CatBoostRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 1. Tạo dữ liệu
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)

# 2. Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Khởi tạo và huấn luyện
model = CatBoostRegressor(
    iterations=500,
    depth=6,
    learning_rate=0.1,
    loss_function='RMSE',
    verbose=100
)
model.fit(X_train, y_train, eval_set=(X_test, y_test))

# 4. Dự đoán và đánh giá
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
```

### 11.4. Cross-Validation

```python
from catboost import CatBoostClassifier, Pool, cv
import pandas as pd

# Chuẩn bị dữ liệu
# ... (giả sử đã có X, y)

# Tạo Pool
pool = Pool(X, label=y, cat_features=cat_features)

# Định nghĩa parameters
params = {
    'iterations': 500,
    'depth': 6,
    'learning_rate': 0.05,
    'loss_function': 'Logloss',
    'verbose': False
}

# Chạy cross-validation
cv_results = cv(
    pool=pool,
    params=params,
    fold_count=5,
    shuffle=True,
    stratified=True,
    verbose=False
)

print(f"CV Results:\n{cv_results.tail()}")
print(f"Mean Test Accuracy: {1 - cv_results['test-Logloss-mean'].iloc[-1]:.4f}")
```

### 11.5. Feature Importance

```python
import matplotlib.pyplot as plt
import pandas as pd

# Sau khi huấn luyện model
feature_importance = model.get_feature_importance()
feature_names = model.feature_names_

# Tạo DataFrame để visualization
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

# Vẽ biểu đồ
plt.figure(figsize=(10, 8))
plt.barh(importance_df['feature'][:15], importance_df['importance'][:15])
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Top 15 Most Important Features')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

### 11.6. Hyperparameter Tuning với Grid Search

```python
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV

# Định nghĩa parameter grid
param_grid = {
    'iterations': [100, 300, 500],
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1]
}

# Khởi tạo model
model = CatBoostClassifier(
    loss_function='Logloss',
    verbose=False,
    random_state=42
)

# Grid search
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
```

### 11.7. Lưu và tải Model

```python
# Lưu model
model.save_model('catboost_model.cbm')

# Hoặc lưu dạng JSON (có thể đọc được)
model.save_model('catboost_model.json', format='json')

# Tải model
loaded_model = CatBoostClassifier()
loaded_model.load_model('catboost_model.cbm')

# Dự đoán với model đã tải
predictions = loaded_model.predict(X_test)
```

---

## 12. Best Practices

### 12.1. Tiền xử lý dữ liệu

```python
# ✅ Đúng: Để CatBoost xử lý categorical features
cat_features = ['gender', 'city', 'category']
model.fit(X, y, cat_features=cat_features)

# ❌ Sai: Không cần one-hot encoding
# X = pd.get_dummies(X)  # Không cần thiết với CatBoost
```

### 12.2. Xử lý Missing Values

```python
# CatBoost xử lý missing values tự động
model = CatBoostClassifier(
    nan_mode='Min'  # Hoặc 'Max', 'Forbidden'
)
```

### 12.3. Early Stopping

```python
model = CatBoostClassifier(
    iterations=3000,
    early_stopping_rounds=50,  # Dừng nếu không cải thiện sau 50 rounds
    verbose=100
)

model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    use_best_model=True  # Sử dụng model tốt nhất, không phải model cuối
)
```

### 12.4. Sử dụng GPU

```python
model = CatBoostClassifier(
    task_type='GPU',
    devices='0',  # GPU device ID
    gpu_ram_part=0.5  # Sử dụng 50% GPU RAM
)
```

### 12.5. Logging và Monitoring

```python
# Verbose modes
verbose = False  # Không in gì
verbose = True   # In mỗi iteration
verbose = 100    # In mỗi 100 iterations

# Custom logging
model = CatBoostClassifier(
    verbose=100,
    logging_level='Info'  # 'Silent', 'Verbose', 'Info', 'Debug'
)
```

### 12.6. Tips tối ưu hiệu năng

```
┌─────────────────────────────────────────────────────────────┐
│                  TIPS TỐI ƯU HIỆU NĂNG                       │
├─────────────────────────────────────────────────────────────┤
│ 1. Sử dụng Pool object cho large datasets                   │
│ 2. Bật GPU nếu có available                                 │
│ 3. Sử dụng early_stopping để tránh overfitting              │
│ 4. Giảm depth nếu overfitting                               │
│ 5. Tăng learning_rate + giảm iterations cho speed           │
│ 6. Sử dụng grow_policy='Lossguide' cho deep trees           │
│ 7. Caching: model.fit(..., save_snapshot=True)              │
└─────────────────────────────────────────────────────────────┘
```

---

## 13. Tài Liệu Tham Khảo

### 13.1. Official Documentation
- [CatBoost Official Documentation](https://catboost.ai/docs/)
- [CatBoost GitHub Repository](https://github.com/catboost/catboost)
- [CatBoost Tutorials](https://catboost.ai/docs/concepts/tutorials.html)

### 13.2. Research Papers
- Prokhorenkova, L., et al. (2018). "CatBoost: unbiased boosting with categorical features"
- Dorogush, A. V., et al. (2018). "CatBoost: gradient boosting with categorical features support"

### 13.3. Bài viết tham khảo
- [Interdata - CatBoost là gì?](https://interdata.vn/blog/catboost-la-gi/)
- [FUNiX - CatBoost: Một thư viện máy học để xử lý dữ liệu](https://funix.edu.vn/chia-se-kien-thuc/catboost-mot-thu-vien-may-hoc-de-xu-ly-du-lieu/)

### 13.4. API Reference

```python
# Các class chính
CatBoostClassifier  # Phân loại
CatBoostRegressor   # Hồi quy  
CatBoostRanker      # Ranking
Pool                # Dataset container

# Các hàm tiện ích
cv()                # Cross-validation
sum_models()        # Combine multiple models
to_regressor()      # Convert classifier to regressor
```

---

## Kết Luận

**CatBoost** là một thư viện mạnh mẽ và hiệu quả cho các bài toán học máy, đặc biệt là khi làm việc với dữ liệu phân loại. Với khả năng:

- ✅ Xử lý trực tiếp các đặc trưng phân loại
- ✅ Tốc độ huấn luyện nhanh
- ✅ Tính linh hoạt trong việc điều chỉnh tham số
- ✅ Cơ chế chống overfitting hiệu quả

CatBoost là lựa chọn tuyệt vời cho các chuyên gia dữ liệu và nhà nghiên cứu trong việc phát triển mô hình học máy. Đặc biệt trong bài toán **dự báo thời tiết**, CatBoost có thể được áp dụng hiệu quả để dự đoán:

- Nhiệt độ
- Lượng mưa
- Điều kiện thời tiết
- Các hiện tượng thời tiết cực đoan

---

*Tài liệu được tổng hợp và biên soạn cho dự án Weather Forecast App*

*Cập nhật: Tháng 1/2026*
