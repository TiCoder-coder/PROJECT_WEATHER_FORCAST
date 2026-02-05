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
13. [Bài tập giải tay và ứng dụng](#13-bài-tập-giải-tay-và-ứng-dụng)
14. [Tài liệu tham khảo](#14-tài-liệu-tham-khảo)

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

## 13. Bài Tập Giải Tay Và Ứng Dụng

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  📚 PHẦN BÀI TẬP THỰC HÀNH - CATBOOST                                        ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║  Mục tiêu: Hiểu sâu thuật toán CatBoost qua các bài tập giải tay chi tiết    ║
║  Nội dung: Target Statistics, Ordered Boosting, Gradient, Symmetric Trees    ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

### 📝 BÀI TẬP 1: Tính Target Statistics cho Categorical Features

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  🎯 MỤC TIÊU: Học cách mã hóa categorical feature thành số bằng Target      │
│               Statistics - kỹ thuật đặc trưng của CatBoost                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 📋 ĐỀ BÀI

Cho bộ dữ liệu dự đoán khách hàng mua hàng với categorical feature **"Thành phố"**:

```
┌───────┬────────────┬──────────────┐
│  Mẫu  │ Thành phố  │ Mua hàng (y) │
├───────┼────────────┼──────────────┤
│   1   │   Hanoi    │      1       │
│   2   │   HCMC     │      0       │
│   3   │   Hanoi    │      1       │
│   4   │   Danang   │      0       │
│   5   │   HCMC     │      1       │
│   6   │   Hanoi    │      0       │
│   7   │   Danang   │      1       │
│   8   │   HCMC     │      1       │
└───────┴────────────┴──────────────┘
```

**Yêu cầu:** Tính Target Statistics cho mỗi category với **prior = 1**

---

#### ✏️ LỜI GIẢI CHI TIẾT

##### 📌 BƯỚC 1: Xác định công thức

```
╔════════════════════════════════════════════════════════════════════════════╗
║                    CÔNG THỨC TARGET STATISTICS                              ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                             ║
║         count × mean_target + prior × global_mean                           ║
║   TS = ─────────────────────────────────────────────                        ║
║                    count + prior                                            ║
║                                                                             ║
║  Trong đó:                                                                  ║
║  • count      = số mẫu trong category                                       ║
║  • mean_target = trung bình y trong category                                ║
║  • prior      = tham số smoothing (cho trước = 1)                           ║
║  • global_mean = trung bình y toàn bộ dữ liệu                               ║
╚════════════════════════════════════════════════════════════════════════════╝
```

##### 📌 BƯỚC 2: Tính Global Mean

```
                    Tổng tất cả y
   global_mean = ─────────────────
                  Tổng số mẫu

                 1 + 0 + 1 + 0 + 1 + 0 + 1 + 1
              = ───────────────────────────────
                            8

                   5
              = ───── = 0.625
                   8
```

> 📍 **Kết quả:** global_mean = **0.625**

##### 📌 BƯỚC 3: Thống kê theo từng Category

```
┌────────────┬─────────────┬───────┬─────────┬──────────────┐
│  Category  │  Các mẫu    │ count │ sum(y)  │ mean_target  │
├────────────┼─────────────┼───────┼─────────┼──────────────┤
│   Hanoi    │  1, 3, 6    │   3   │ 1+1+0=2 │ 2/3 = 0.667  │
│   HCMC     │  2, 5, 8    │   3   │ 0+1+1=2 │ 2/3 = 0.667  │
│   Danang   │  4, 7       │   2   │ 0+1=1   │ 1/2 = 0.500  │
└────────────┴─────────────┴───────┴─────────┴──────────────┘
```

##### 📌 BƯỚC 4: Tính Target Statistics cho từng Category

**🔹 Tính cho HANOI:**
```
         count × mean_target + prior × global_mean
   TS = ────────────────────────────────────────────
                     count + prior

         3 × 0.667 + 1 × 0.625
      = ────────────────────────
               3 + 1

         2.001 + 0.625
      = ─────────────── 
              4

         2.626
      = ─────── = 0.6565
           4
```

**🔹 Tính cho HCMC:**
```
         3 × 0.667 + 1 × 0.625
   TS = ────────────────────────
                3 + 1

         2.001 + 0.625       2.626
      = ─────────────── = ─────── = 0.6565
              4               4
```

**🔹 Tính cho DANANG:**
```
         2 × 0.500 + 1 × 0.625
   TS = ────────────────────────
                2 + 1

         1.000 + 0.625       1.625
      = ─────────────── = ─────── = 0.5417
              3               3
```

##### 📌 BƯỚC 5: Kết quả cuối cùng

```
╔════════════════════════════════════════════════════════════════╗
║              🎯 KẾT QUẢ TARGET STATISTICS                       ║
╠════════════════╦═══════════════════════════════════════════════╣
║   Thành phố    ║     Target Statistics (giá trị số)            ║
╠════════════════╬═══════════════════════════════════════════════╣
║     Hanoi      ║              0.6565                           ║
║     HCMC       ║              0.6565                           ║
║     Danang     ║              0.5417                           ║
╚════════════════╩═══════════════════════════════════════════════╝
```

##### 💡 NHẬN XÉT QUAN TRỌNG

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ✅ Prior = 1 giúp "smoothing" - kéo các giá trị cực đoan về global mean    │
│  ✅ Category ít mẫu (Danang: 2 mẫu) bị ảnh hưởng bởi prior nhiều hơn        │
│  ✅ Giá trị TS dùng làm feature số thay cho categorical gốc                 │
│  ✅ Không cần One-Hot Encoding → giảm số chiều dữ liệu                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 📝 BÀI TẬP 2: Ordered Target Statistics (Tránh Target Leakage)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  🎯 MỤC TIÊU: Hiểu cách CatBoost tránh "rò rỉ thông tin" (target leakage)   │
│               bằng cách tính Target Statistics theo thứ tự                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 📋 ĐỀ BÀI

Sử dụng dữ liệu Bài 1, tính **Ordered Target Statistics** cho feature "Thành phố" theo thứ tự xuất hiện (prior = 1).

---

#### ✏️ LỜI GIẢI CHI TIẾT

##### 📌 BƯỚC 1: Hiểu nguyên lý Ordered Target Statistics

```
╔════════════════════════════════════════════════════════════════════════════╗
║              ⚠️ VẤN ĐỀ: TARGET LEAKAGE                                      ║
╠════════════════════════════════════════════════════════════════════════════╣
║  Khi tính TS cho mẫu i, nếu dùng thông tin của chính mẫu i                 ║
║  → Model "nhìn thấy" target trước khi dự đoán → GIAN LẬN!                  ║
╠════════════════════════════════════════════════════════════════════════════╣
║              ✅ GIẢI PHÁP: ORDERED TARGET STATISTICS                        ║
╠════════════════════════════════════════════════════════════════════════════╣
║  Chỉ dùng thông tin từ các mẫu TRƯỚC mẫu i (mẫu 1 → i-1)                   ║
║                                                                             ║
║         count_before × mean_before + prior × global_mean_before             ║
║  OTS = ──────────────────────────────────────────────────────────           ║
║                      count_before + prior                                   ║
╚════════════════════════════════════════════════════════════════════════════╝
```

##### 📌 BƯỚC 2: Tính tuần tự cho từng mẫu

**🔹 MẪU 1 (Hanoi, y=1):**
```
┌─────────────────────────────────────────────┐
│  Mẫu Hanoi trước đó: KHÔNG CÓ              │
│  → Dùng giá trị mặc định                    │
│  Ordered_TS(1) = 0.5                        │
└─────────────────────────────────────────────┘
```

**🔹 MẪU 2 (HCMC, y=0):**
```
┌─────────────────────────────────────────────┐
│  Mẫu HCMC trước đó: KHÔNG CÓ               │
│  → Dùng giá trị mặc định                    │
│  Ordered_TS(2) = 0.5                        │
└─────────────────────────────────────────────┘
```

**🔹 MẪU 3 (Hanoi, y=1):**
```
┌─────────────────────────────────────────────────────────────────┐
│  Mẫu Hanoi trước đó: Mẫu 1 (y=1)                               │
│  • count_before = 1                                             │
│  • mean_before = 1/1 = 1.0                                      │
│  • global_mean_before = (y₁+y₂)/2 = (1+0)/2 = 0.5               │
├─────────────────────────────────────────────────────────────────┤
│              1 × 1.0 + 1 × 0.5                                  │
│  OTS(3) = ─────────────────────── = 1.5/2 = 0.75                │
│                  1 + 1                                          │
└─────────────────────────────────────────────────────────────────┘
```

**🔹 MẪU 4 (Danang, y=0):**
```
┌─────────────────────────────────────────────┐
│  Mẫu Danang trước đó: KHÔNG CÓ             │
│  → Dùng giá trị mặc định                    │
│  Ordered_TS(4) = 0.5                        │
└─────────────────────────────────────────────┘
```

**🔹 MẪU 5 (HCMC, y=1):**
```
┌─────────────────────────────────────────────────────────────────┐
│  Mẫu HCMC trước đó: Mẫu 2 (y=0)                                │
│  • count_before = 1                                             │
│  • mean_before = 0/1 = 0.0                                      │
│  • global_mean_before = (1+0+1+0)/4 = 0.5                       │
├─────────────────────────────────────────────────────────────────┤
│              1 × 0.0 + 1 × 0.5                                  │
│  OTS(5) = ─────────────────────── = 0.5/2 = 0.25                │
│                  1 + 1                                          │
└─────────────────────────────────────────────────────────────────┘
```

**🔹 MẪU 6 (Hanoi, y=0):**
```
┌─────────────────────────────────────────────────────────────────┐
│  Mẫu Hanoi trước đó: Mẫu 1 (y=1), Mẫu 3 (y=1)                  │
│  • count_before = 2                                             │
│  • mean_before = (1+1)/2 = 1.0                                  │
│  • global_mean_before = (1+0+1+0+1)/5 = 0.6                     │
├─────────────────────────────────────────────────────────────────┤
│              2 × 1.0 + 1 × 0.6                                  │
│  OTS(6) = ─────────────────────── = 2.6/3 = 0.867               │
│                  2 + 1                                          │
└─────────────────────────────────────────────────────────────────┘
```

**🔹 MẪU 7 (Danang, y=1):**
```
┌─────────────────────────────────────────────────────────────────┐
│  Mẫu Danang trước đó: Mẫu 4 (y=0)                              │
│  • count_before = 1                                             │
│  • mean_before = 0/1 = 0.0                                      │
│  • global_mean_before = (1+0+1+0+1+0)/6 = 0.5                   │
├─────────────────────────────────────────────────────────────────┤
│              1 × 0.0 + 1 × 0.5                                  │
│  OTS(7) = ─────────────────────── = 0.5/2 = 0.25                │
│                  1 + 1                                          │
└─────────────────────────────────────────────────────────────────┘
```

**🔹 MẪU 8 (HCMC, y=1):**
```
┌─────────────────────────────────────────────────────────────────┐
│  Mẫu HCMC trước đó: Mẫu 2 (y=0), Mẫu 5 (y=1)                   │
│  • count_before = 2                                             │
│  • mean_before = (0+1)/2 = 0.5                                  │
│  • global_mean_before = (1+0+1+0+1+0+1)/7 = 4/7 ≈ 0.571         │
├─────────────────────────────────────────────────────────────────┤
│              2 × 0.5 + 1 × 0.571                                │
│  OTS(8) = ───────────────────────── = 1.571/3 = 0.524           │
│                    2 + 1                                        │
└─────────────────────────────────────────────────────────────────┘
```

##### 📌 BƯỚC 3: Tổng hợp kết quả

```
╔═══════╦════════════╦═════╦══════════════╦═══════════════════════════════════╗
║  Mẫu  ║ Thành phố  ║  y  ║  Ordered TS  ║            Giải thích             ║
╠═══════╬════════════╬═════╬══════════════╬═══════════════════════════════════╣
║   1   ║   Hanoi    ║  1  ║    0.500     ║  Không có mẫu trước → mặc định    ║
║   2   ║   HCMC     ║  0  ║    0.500     ║  Không có mẫu trước → mặc định    ║
║   3   ║   Hanoi    ║  1  ║    0.750     ║  1 Hanoi trước, mean=1.0          ║
║   4   ║   Danang   ║  0  ║    0.500     ║  Không có mẫu trước → mặc định    ║
║   5   ║   HCMC     ║  1  ║    0.250     ║  1 HCMC trước, mean=0.0           ║
║   6   ║   Hanoi    ║  0  ║    0.867     ║  2 Hanoi trước, mean=1.0          ║
║   7   ║   Danang   ║  1  ║    0.250     ║  1 Danang trước, mean=0.0         ║
║   8   ║   HCMC     ║  1  ║    0.524     ║  2 HCMC trước, mean=0.5           ║
╚═══════╩════════════╩═════╩══════════════╩═══════════════════════════════════╝
```

##### 💡 SO SÁNH: Target Statistics vs Ordered Target Statistics

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         SO SÁNH HAI PHƯƠNG PHÁP                              │
├────────────────────────────────┬────────────────────────────────────────────┤
│      Target Statistics         │       Ordered Target Statistics            │
├────────────────────────────────┼────────────────────────────────────────────┤
│  Hanoi: 0.6565 (tất cả mẫu)   │  Hanoi: 0.500 → 0.750 → 0.867 (thay đổi)  │
│  HCMC:  0.6565 (tất cả mẫu)   │  HCMC:  0.500 → 0.250 → 0.524 (thay đổi)  │
│  Danang: 0.5417 (tất cả mẫu)  │  Danang: 0.500 → 0.250 (thay đổi)         │
├────────────────────────────────┴────────────────────────────────────────────┤
│  ❌ Target Leakage: CÓ          │  ✅ Target Leakage: KHÔNG                 │
│  ❌ Mỗi category = 1 giá trị   │  ✅ Mỗi mẫu = giá trị riêng              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Mẫu 7 (Danang, y=1):**
- Mẫu Danang trước: {Mẫu 4: y=0}
- count_before = 1, mean_before = 0.0
- global_mean_before = (1+0+1+0+1+0)/6 = 0.5
```
Ordered_TS(7) = (1 × 0.0 + 1 × 0.5) / (1 + 1)
              = (0.0 + 0.5) / 2
              = 0.25
```

**Mẫu 8 (HCMC, y=1):**
- Mẫu HCMC trước: {Mẫu 2: y=0, Mẫu 5: y=1}
- count_before = 2, mean_before = 1/2 = 0.5
- global_mean_before = (1+0+1+0+1+0+1)/7 = 4/7 ≈ 0.571
```
Ordered_TS(8) = (2 × 0.5 + 1 × 0.571) / (2 + 1)
              = (1.0 + 0.571) / 3
              = 0.524
```

#### Bước 2: Tổng hợp kết quả

| Mẫu | Thành phố | y | Ordered TS |
|-----|-----------|---|------------|
| 1   | Hanoi     | 1 | 0.500      |
| 2   | HCMC      | 0 | 0.500      |
| 3   | Hanoi     | 1 | 0.750      |
| 4   | Danang    | 0 | 0.500      |
| 5   | HCMC      | 1 | 0.250      |
| 6   | Hanoi     | 0 | 0.867      |
| 7   | Danang    | 1 | 0.250      |
| 8   | HCMC      | 1 | 0.524      |

**Ưu điểm của Ordered Target Statistics:**
- Tránh target leakage (rò rỉ thông tin từ target)
- Mỗi mẫu có giá trị khác nhau dựa trên vị trí
- Giảm overfitting đáng kể

---

### 📝 BÀI TẬP 3: Tính Gradient và Cập nhật Residuals

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  🎯 MỤC TIÊU: Hiểu cơ chế Gradient Boosting - cách các cây học từ sai số   │
│               của cây trước để cải thiện dự đoán                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 📋 ĐỀ BÀI

Cho bài toán binary classification với LogLoss. Sau iteration đầu tiên, model dự đoán:

```
┌───────┬──────────┬─────────────────┐
│  Mẫu  │  y thực  │  p (dự đoán)    │
├───────┼──────────┼─────────────────┤
│   1   │    1     │      0.6        │
│   2   │    0     │      0.3        │
│   3   │    1     │      0.4        │
│   4   │    0     │      0.7        │
│   5   │    1     │      0.8        │
└───────┴──────────┴─────────────────┘
```

**Yêu cầu:** Tính gradient (residuals) cho mỗi mẫu để cây tiếp theo học.

---

#### ✏️ LỜI GIẢI CHI TIẾT

##### 📌 BƯỚC 1: Hiểu công thức Gradient cho LogLoss

```
╔════════════════════════════════════════════════════════════════════════════╗
║                        HÀM MẤT MÁT LOGLOSS                                  ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                             ║
║   LogLoss = -[ y × log(p) + (1-y) × log(1-p) ]                              ║
║                                                                             ║
╠════════════════════════════════════════════════════════════════════════════╣
║                     GRADIENT (RESIDUAL)                                     ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                             ║
║   residual = y - p                                                          ║
║                                                                             ║
║   • residual > 0  →  Model dự đoán THẤP hơn thực tế  →  Cần TĂNG           ║
║   • residual < 0  →  Model dự đoán CAO hơn thực tế   →  Cần GIẢM           ║
║   • residual ≈ 0  →  Model dự đoán ĐÚNG              →  Giữ nguyên         ║
╚════════════════════════════════════════════════════════════════════════════╝
```

##### 📌 BƯỚC 2: Tính Residual cho từng mẫu

**🔹 MẪU 1 (y=1, p=0.6):**
```
┌─────────────────────────────────────────────────────────────────┐
│   residual₁ = y - p = 1 - 0.6 = +0.4                            │
│                                                                  │
│   📊 Phân tích:                                                  │
│   • y thực = 1 (Có mua hàng)                                     │
│   • Model đoán p = 0.6 (60% mua)                                 │
│   • Cần TĂNG thêm 0.4 để đạt 100%                               │
└─────────────────────────────────────────────────────────────────┘
```

**🔹 MẪU 2 (y=0, p=0.3):**
```
┌─────────────────────────────────────────────────────────────────┐
│   residual₂ = y - p = 0 - 0.3 = -0.3                            │
│                                                                  │
│   📊 Phân tích:                                                  │
│   • y thực = 0 (Không mua)                                       │
│   • Model đoán p = 0.3 (30% mua)                                 │
│   • Cần GIẢM 0.3 để đạt 0%                                      │
└─────────────────────────────────────────────────────────────────┘
```

**🔹 MẪU 3 (y=1, p=0.4):**
```
┌─────────────────────────────────────────────────────────────────┐
│   residual₃ = y - p = 1 - 0.4 = +0.6                            │
│                                                                  │
│   📊 Phân tích:                                                  │
│   • y thực = 1 (Có mua)                                          │
│   • Model đoán p = 0.4 (chỉ 40%)  ⚠️ SAI NHIỀU!                  │
│   • Cần TĂNG MẠNH 0.6                                           │
└─────────────────────────────────────────────────────────────────┘
```

**🔹 MẪU 4 (y=0, p=0.7):**
```
┌─────────────────────────────────────────────────────────────────┐
│   residual₄ = y - p = 0 - 0.7 = -0.7                            │
│                                                                  │
│   📊 Phân tích:                                                  │
│   • y thực = 0 (Không mua)                                       │
│   • Model đoán p = 0.7 (70% mua)  ⚠️ SAI NHIỀU!                  │
│   • Cần GIẢM MẠNH 0.7                                           │
└─────────────────────────────────────────────────────────────────┘
```

**🔹 MẪU 5 (y=1, p=0.8):**
```
┌─────────────────────────────────────────────────────────────────┐
│   residual₅ = y - p = 1 - 0.8 = +0.2                            │
│                                                                  │
│   📊 Phân tích:                                                  │
│   • y thực = 1 (Có mua)                                          │
│   • Model đoán p = 0.8 (80%) ✅ Gần đúng!                        │
│   • Chỉ cần tăng nhẹ 0.2                                         │
└─────────────────────────────────────────────────────────────────┘
```

##### 📌 BƯỚC 3: Tổng hợp Residuals

```
╔═══════╦═════╦═══════╦════════════╦═══════════════════════════════════════════╗
║  Mẫu  ║  y  ║   p   ║  Residual  ║              Ý nghĩa                      ║
╠═══════╬═════╬═══════╬════════════╬═══════════════════════════════════════════╣
║   1   ║  1  ║  0.6  ║   +0.4     ║  ⬆️ Tăng vừa phải                         ║
║   2   ║  0  ║  0.3  ║   -0.3     ║  ⬇️ Giảm vừa phải                         ║
║   3   ║  1  ║  0.4  ║   +0.6     ║  ⬆️⬆️ Tăng mạnh (sai nhiều)               ║
║   4   ║  0  ║  0.7  ║   -0.7     ║  ⬇️⬇️ Giảm mạnh (sai nhiều)               ║
║   5   ║  1  ║  0.8  ║   +0.2     ║  ⬆️ Tăng nhẹ (gần đúng)                   ║
╚═══════╩═════╩═══════╩════════════╩═══════════════════════════════════════════╝
```

##### 📌 BƯỚC 4: Cây tiếp theo học từ Residuals

```
╔════════════════════════════════════════════════════════════════════════════╗
║                  QUÁ TRÌNH HỌC CỦA CÂY THỨ 2                                ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                             ║
║   Cây 2 được huấn luyện với:                                                ║
║   • Input:  X (các features gốc)                                            ║
║   • Target: Residuals = [+0.4, -0.3, +0.6, -0.7, +0.2]                      ║
║                                                                             ║
║   → Cây 2 học cách dự đoán mức độ điều chỉnh cần thiết                      ║
╚════════════════════════════════════════════════════════════════════════════╝
```

**Giả sử Cây 2 dự đoán:**
```
┌─────────────────────────────────────────────────────┐
│   tree_2_predictions = [0.3, -0.2, 0.4, -0.5, 0.1]  │
└─────────────────────────────────────────────────────┘
```

**Cập nhật prediction với learning_rate = 0.1:**

```
╔════════════════════════════════════════════════════════════════════════════╗
║   new_p = old_p + learning_rate × tree_2_prediction                         ║
╚════════════════════════════════════════════════════════════════════════════╝

┌───────┬────────┬─────────────┬───────────────────────────────┬─────────┐
│  Mẫu  │  p cũ  │  Tree2 pred │          Tính toán            │  p mới  │
├───────┼────────┼─────────────┼───────────────────────────────┼─────────┤
│   1   │  0.60  │    +0.3     │  0.60 + 0.1 × 0.3  = 0.60+0.03│  0.63   │
│   2   │  0.30  │    -0.2     │  0.30 + 0.1 ×(-0.2)= 0.30-0.02│  0.28   │
│   3   │  0.40  │    +0.4     │  0.40 + 0.1 × 0.4  = 0.40+0.04│  0.44   │
│   4   │  0.70  │    -0.5     │  0.70 + 0.1 ×(-0.5)= 0.70-0.05│  0.65   │
│   5   │  0.80  │    +0.1     │  0.80 + 0.1 × 0.1  = 0.80+0.01│  0.81   │
└───────┴────────┴─────────────┴───────────────────────────────┴─────────┘
```

##### 💡 MINH HỌA QUÁ TRÌNH GRADIENT BOOSTING

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   🌲 Tree 1          🌲 Tree 2          🌲 Tree 3         ...    🎯 Final    │
│   ─────────         ─────────          ─────────                ─────────   │
│      │                  │                  │                        │       │
│      ▼                  ▼                  ▼                        ▼       │
│   p₁ = 0.6   +    Δ₁ = 0.03    +    Δ₂ = 0.02    + ...  →   p = 0.92      │
│   (base)          (adjust)          (adjust)                  (final)      │
│                                                                             │
│   📝 Mỗi cây học để SỬA SAI của các cây trước!                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 📝 BÀI TẬP 4: Xây dựng Symmetric Tree (Đặc trưng CatBoost)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  🎯 MỤC TIÊU: Hiểu cấu trúc Symmetric Tree - đặc điểm riêng của CatBoost    │
│               giúp tăng tốc độ inference và giảm overfitting                │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 📋 ĐỀ BÀI

Cho dữ liệu dự báo thời tiết:

```
┌───────┬───────────────┬───────────┬────────────┬─────────┐
│  Mẫu  │ Nhiệt độ (°C) │ Độ ẩm (%) │ Gió (km/h) │   Mưa   │
├───────┼───────────────┼───────────┼────────────┼─────────┤
│   1   │      32       │    85     │     10     │   Có    │
│   2   │      28       │    60     │     15     │  Không  │
│   3   │      35       │    90     │      5     │   Có    │
│   4   │      25       │    55     │     20     │  Không  │
│   5   │      30       │    80     │      8     │   Có    │
│   6   │      27       │    50     │     25     │  Không  │
│   7   │      33       │    88     │     12     │   Có    │
│   8   │      26       │    45     │     18     │  Không  │
└───────┴───────────────┴───────────┴────────────┴─────────┘
```

**Yêu cầu:** Xây dựng Symmetric Tree depth=2 theo phương pháp CatBoost

---

#### ✏️ LỜI GIẢI CHI TIẾT

##### 📌 BƯỚC 1: Hiểu Symmetric Tree là gì?

```
╔════════════════════════════════════════════════════════════════════════════╗
║                        SYMMETRIC TREE LÀ GÌ?                                ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                             ║
║  Cây thông thường (XGBoost, LightGBM):    Cây đối xứng (CatBoost):         ║
║                                                                             ║
║         Root                                    Root                        ║
║        /    \                                  /    \                       ║
║     A≤5    A>5                              A≤5     A≤5     ← CÙNG điều kiện║
║     /  \     / \                           /   \   /   \                    ║
║   B≤3  C≤7  D≤2 E≤4                      B≤3  B≤3 B≤3 B≤3  ← CÙNG điều kiện║
║                                                                             ║
║  ❌ Mỗi node có thể split                ✅ Mỗi level dùng CÙNG split       ║
║     theo feature khác nhau                   cho TẤT CẢ nodes              ║
║                                                                             ║
╚════════════════════════════════════════════════════════════════════════════╝
```

##### 📌 BƯỚC 2: Tìm Best Split cho Level 1 (Root)

**🔹 Thử split: Nhiệt độ ≤ 29°C**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Nhiệt độ ≤ 29°C ?                                        │
│                     /              \                                        │
│                   YES               NO                                      │
│             ┌───────────┐     ┌───────────┐                                 │
│             │ Mẫu 2,4,6,8│     │ Mẫu 1,3,5,7│                                │
│             │  28,25,27,26│     │  32,35,30,33│                               │
│             └───────────┘     └───────────┘                                 │
│                   ↓                 ↓                                       │
│            Mưa: [K,K,K,K]      Mưa: [C,C,C,C]                               │
│            Gini = 0.0          Gini = 0.0                                   │
│            (thuần nhất!)       (thuần nhất!)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Weighted Gini = (4/8)×0.0 + (4/8)×0.0 = 0.0  ✅ HOÀN HẢO!                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**→ Chọn Split Level 1: Nhiệt độ ≤ 29°C**

##### 📌 BƯỚC 3: Tìm Best Split cho Level 2 (CÙNG cho tất cả nodes)

**🔹 Thử split: Độ ẩm ≤ 75%**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ⚠️ QUAN TRỌNG: Trong Symmetric Tree, Level 2 dùng CÙNG điều kiện          │
│     cho CẢ nhánh trái và nhánh phải!                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

##### 📌 BƯỚC 4: Xây dựng cây hoàn chỉnh

```
                         ┌─────────────────────┐
                         │   Nhiệt độ ≤ 29?    │  ← Level 1
                         │      (Root)         │
                         └─────────┬───────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                   YES                           NO
                    │                             │
         ┌──────────┴──────────┐      ┌──────────┴──────────┐
         │   Mẫu: 2, 4, 6, 8   │      │   Mẫu: 1, 3, 5, 7   │
         │   (≤29°C)           │      │   (>29°C)           │
         └──────────┬──────────┘      └──────────┬──────────┘
                    │                             │
         ┌──────────┴──────────┐      ┌──────────┴──────────┐
         │   Độ ẩm ≤ 75?       │      │   Độ ẩm ≤ 75?       │  ← Level 2
         └─────────┬───────────┘      └─────────┬───────────┘
                   │                             │       (CÙNG điều kiện)
         ┌─────────┴─────────┐         ┌─────────┴─────────┐
        YES                 NO        YES                 NO
         │                   │         │                   │
   ┌─────┴─────┐       ┌─────┴─────┐   ┌─────┴─────┐ ┌─────┴─────┐
   │  Mẫu:     │       │  Mẫu:     │   │  Mẫu:     │ │  Mẫu:     │
   │  4, 6, 8  │       │    2      │   │   (rỗng)  │ │ 1,3,5,7   │
   │           │       │           │   │           │ │           │
   │  Độ ẩm:   │       │  Độ ẩm:   │   │           │ │  Độ ẩm:   │
   │ 55,50,45  │       │    60     │   │           │ │ 85,90,80,88│
   └─────┬─────┘       └─────┬─────┘   └─────┬─────┘ └─────┬─────┘
         │                   │               │             │
         ▼                   ▼               ▼             ▼
    ┌─────────┐        ┌─────────┐     ┌─────────┐   ┌─────────┐
    │ KHÔNG   │        │ KHÔNG   │     │  N/A    │   │   CÓ    │
    │ (100%)  │        │ (100%)  │     │         │   │ (100%)  │
    └─────────┘        └─────────┘     └─────────┘   └─────────┘
```

##### 📌 BƯỚC 5: Tóm tắt các Leaf nodes

```
╔═══════╦══════════════════════════════════╦═══════════════╦══════════════════╗
║ Leaf  ║           Điều kiện              ║     Mẫu       ║   Dự đoán        ║
╠═══════╬══════════════════════════════════╬═══════════════╬══════════════════╣
║  L1   ║ Nhiệt độ ≤ 29  AND  Độ ẩm ≤ 75   ║   4, 6, 8     ║ KHÔNG mưa (100%) ║
║  L2   ║ Nhiệt độ ≤ 29  AND  Độ ẩm > 75   ║      2        ║ KHÔNG mưa (100%) ║
║  L3   ║ Nhiệt độ > 29  AND  Độ ẩm ≤ 75   ║    (rỗng)     ║      N/A         ║
║  L4   ║ Nhiệt độ > 29  AND  Độ ẩm > 75   ║  1, 3, 5, 7   ║ CÓ mưa (100%)    ║
╚═══════╩══════════════════════════════════╩═══════════════╩══════════════════╝
```

##### 💡 ƯU ĐIỂM CỦA SYMMETRIC TREE

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ƯU ĐIỂM SYMMETRIC TREE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ✅ INFERENCE NHANH                                                         │
│     • Mỗi level chỉ cần kiểm tra 1 điều kiện                               │
│     • Có thể dùng bitwise operations để duyệt cây                          │
│     • Tốc độ nhanh hơn 10-40% so với cây thông thường                      │
│                                                                             │
│  ✅ ÍT PARAMETERS HƠN                                                       │
│     • Cây depth=d chỉ cần d điều kiện split                                │
│     • Thay vì 2^d - 1 điều kiện như cây thông thường                       │
│                                                                             │
│  ✅ GIẢM OVERFITTING                                                        │
│     • Ít parameters → ít nguy cơ overfitting                               │
│     • Cây đơn giản hơn nhưng vẫn hiệu quả                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 📝 BÀI TẬP 5: Tính LogLoss và Đánh giá Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  🎯 MỤC TIÊU: Học cách đánh giá model classification bằng LogLoss          │
│               - metric quan trọng trong CatBoost                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 📋 ĐỀ BÀI

Model CatBoost dự đoán xác suất mưa cho 6 ngày:

```
┌────────┬──────────┬─────────────────┐
│  Ngày  │  y thực  │  p (dự đoán)    │
├────────┼──────────┼─────────────────┤
│   1    │    1     │      0.90       │
│   2    │    0     │      0.20       │
│   3    │    1     │      0.70       │
│   4    │    0     │      0.40       │
│   5    │    1     │      0.85       │
│   6    │    0     │      0.10       │
└────────┴──────────┴─────────────────┘
```

**Yêu cầu:** Tính LogLoss chi tiết

---

#### ✏️ LỜI GIẢI CHI TIẾT

##### 📌 BƯỚC 1: Hiểu công thức LogLoss

```
╔════════════════════════════════════════════════════════════════════════════╗
║                         CÔNG THỨC LOGLOSS                                   ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                             ║
║                    1   n                                                    ║
║   LogLoss = - ─── × Σ [ yᵢ×log(pᵢ) + (1-yᵢ)×log(1-pᵢ) ]                   ║
║                    n  i=1                                                   ║
║                                                                             ║
╠════════════════════════════════════════════════════════════════════════════╣
║   Đơn giản hóa cho từng mẫu:                                                ║
║                                                                             ║
║   • Nếu y = 1 (thực sự có mưa):   loss = -log(p)                            ║
║     → p càng gần 1 → loss càng nhỏ ✅                                       ║
║                                                                             ║
║   • Nếu y = 0 (không có mưa):     loss = -log(1-p)                          ║
║     → p càng gần 0 → loss càng nhỏ ✅                                       ║
║                                                                             ║
╚════════════════════════════════════════════════════════════════════════════╝
```

##### 📌 BƯỚC 2: Tính Loss cho từng ngày

**🔹 NGÀY 1 (y=1, p=0.90)** - Có mưa, đoán 90% mưa

```
┌─────────────────────────────────────────────────────────────────┐
│   loss₁ = -log(p) = -log(0.90)                                  │
│         = -(-0.1054)                                            │
│         = 0.1054                                                │
│                                                                  │
│   ✅ Loss thấp → Model đoán TỐT (90% cho y=1)                   │
└─────────────────────────────────────────────────────────────────┘
```

**🔹 NGÀY 2 (y=0, p=0.20)** - Không mưa, đoán 20% mưa

```
┌─────────────────────────────────────────────────────────────────┐
│   loss₂ = -log(1-p) = -log(1-0.20) = -log(0.80)                 │
│         = -(-0.2231)                                            │
│         = 0.2231                                                │
│                                                                  │
│   ✅ Loss tương đối thấp → Model đoán khá tốt                   │
└─────────────────────────────────────────────────────────────────┘
```

**🔹 NGÀY 3 (y=1, p=0.70)** - Có mưa, đoán 70% mưa

```
┌─────────────────────────────────────────────────────────────────┐
│   loss₃ = -log(p) = -log(0.70)                                  │
│         = -(-0.3567)                                            │
│         = 0.3567                                                │
│                                                                  │
│   ⚠️ Loss trung bình → Đoán đúng nhưng chưa tự tin lắm          │
└─────────────────────────────────────────────────────────────────┘
```

**🔹 NGÀY 4 (y=0, p=0.40)** - Không mưa, đoán 40% mưa

```
┌─────────────────────────────────────────────────────────────────┐
│   loss₄ = -log(1-p) = -log(1-0.40) = -log(0.60)                 │
│         = -(-0.5108)                                            │
│         = 0.5108                                                │
│                                                                  │
│   ❌ Loss CAO NHẤT → Model đoán kém (40% cho y=0 là quá cao!)   │
└─────────────────────────────────────────────────────────────────┘
```

**🔹 NGÀY 5 (y=1, p=0.85)** - Có mưa, đoán 85% mưa

```
┌─────────────────────────────────────────────────────────────────┐
│   loss₅ = -log(p) = -log(0.85)                                  │
│         = -(-0.1625)                                            │
│         = 0.1625                                                │
│                                                                  │
│   ✅ Loss thấp → Model đoán TỐT                                 │
└─────────────────────────────────────────────────────────────────┘
```

**🔹 NGÀY 6 (y=0, p=0.10)** - Không mưa, đoán 10% mưa

```
┌─────────────────────────────────────────────────────────────────┐
│   loss₆ = -log(1-p) = -log(1-0.10) = -log(0.90)                 │
│         = -(-0.1054)                                            │
│         = 0.1054                                                │
│                                                                  │
│   ✅ Loss thấp → Model đoán RẤT TỐT (10% cho y=0)               │
└─────────────────────────────────────────────────────────────────┘
```

##### 📌 BƯỚC 3: Tính LogLoss trung bình

```
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                             ║
║   LogLoss = (loss₁ + loss₂ + loss₃ + loss₄ + loss₅ + loss₆) / 6            ║
║                                                                             ║
║           = (0.1054 + 0.2231 + 0.3567 + 0.5108 + 0.1625 + 0.1054) / 6      ║
║                                                                             ║
║           = 1.4639 / 6                                                      ║
║                                                                             ║
║           = 0.2440                                                          ║
║                                                                             ║
╚════════════════════════════════════════════════════════════════════════════╝
```

##### 📌 BƯỚC 4: Tổng hợp và đánh giá

```
╔════════╦═══════╦═══════╦══════════╦══════════════════════════════════════════╗
║  Ngày  ║   y   ║   p   ║   Loss   ║                Đánh giá                  ║
╠════════╬═══════╬═══════╬══════════╬══════════════════════════════════════════╣
║   1    ║   1   ║  0.90 ║  0.1054  ║ ✅ Xuất sắc - Tự tin cao, đúng          ║
║   2    ║   0   ║  0.20 ║  0.2231  ║ ✅ Tốt - Đoán đúng hướng                 ║
║   3    ║   1   ║  0.70 ║  0.3567  ║ ⚠️ Trung bình - Chưa đủ tự tin           ║
║   4    ║   0   ║  0.40 ║  0.5108  ║ ❌ Kém - Đoán sai hướng (gần 50-50)      ║
║   5    ║   1   ║  0.85 ║  0.1625  ║ ✅ Rất tốt - Tự tin và đúng              ║
║   6    ║   0   ║  0.10 ║  0.1054  ║ ✅ Xuất sắc - Rất tự tin, đúng           ║
╠════════╩═══════╩═══════╩══════════╩══════════════════════════════════════════╣
║                        TỔNG LogLoss = 0.2440                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

##### 💡 ĐÁNH GIÁ LOGLOSS

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THANG ĐÁNH GIÁ LOGLOSS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   LogLoss < 0.2    →  🌟 Xuất sắc                                           │
│   LogLoss 0.2-0.4  →  ✅ Tốt          ← Model của ta: 0.2440                │
│   LogLoss 0.4-0.6  →  ⚠️ Trung bình                                         │
│   LogLoss > 0.6    →  ❌ Kém                                                │
│                                                                             │
│   📊 Accuracy = 5/6 = 83.3% (đoán đúng 5 ngày)                              │
│                                                                             │
│   💡 KẾT LUẬN: Model hoạt động TỐT với LogLoss = 0.2440                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 📝 BÀI TẬP 6: Ensemble nhiều cây với Learning Rate

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  🎯 MỤC TIÊU: Hiểu cách CatBoost kết hợp nhiều cây với learning rate       │
│               và chuyển đổi từ log-odds sang probability                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 📋 ĐỀ BÀI

CatBoost model có **3 cây** với **learning_rate = 0.3**. Cho mẫu test với:
- Initial prediction (bias): **0.0**
- Tree 1 output: **+2.0**
- Tree 2 output: **+1.5**
- Tree 3 output: **-0.8**

**Yêu cầu:** Tính final prediction và chuyển sang probability

---

#### ✏️ LỜI GIẢI CHI TIẾT

##### 📌 BƯỚC 1: Hiểu công thức Ensemble

```
╔════════════════════════════════════════════════════════════════════════════╗
║                       CÔNG THỨC ENSEMBLE                                    ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                             ║
║   F(x) = F₀ + η × f₁(x) + η × f₂(x) + ... + η × fₙ(x)                      ║
║                                                                             ║
║   Trong đó:                                                                 ║
║   • F₀    = initial prediction (bias) = 0.0                                 ║
║   • η     = learning_rate = 0.3                                             ║
║   • fₜ(x) = output của cây thứ t                                            ║
║                                                                             ║
║   📝 Learning rate nhỏ → điều chỉnh từ từ → ổn định hơn                     ║
║                                                                             ║
╚════════════════════════════════════════════════════════════════════════════╝
```

##### 📌 BƯỚC 2: Tính từng bước

**🔹 Khởi tạo (Ban đầu):**
```
┌─────────────────────────────────────────────────────────────────┐
│   F₀ = 0.0  (bias mặc định)                                     │
│                                                                  │
│   Chuyển sang probability:                                       │
│   p = 1 / (1 + e⁻⁰) = 1 / (1 + 1) = 0.5  (50-50)                │
└─────────────────────────────────────────────────────────────────┘
```

**🔹 Sau Tree 1 (output = +2.0):**
```
┌─────────────────────────────────────────────────────────────────┐
│   F₁ = F₀ + η × f₁                                              │
│      = 0.0 + 0.3 × 2.0                                          │
│      = 0.0 + 0.6                                                │
│      = 0.6                                                      │
│                                                                  │
│   Chuyển sang probability:                                       │
│   p = 1 / (1 + e⁻⁰·⁶) = 1 / (1 + 0.549) = 1/1.549 = 0.646      │
│                                                                  │
│   📈 Tăng từ 50% → 64.6%                                         │
└─────────────────────────────────────────────────────────────────┘
```

**🔹 Sau Tree 2 (output = +1.5):**
```
┌─────────────────────────────────────────────────────────────────┐
│   F₂ = F₁ + η × f₂                                              │
│      = 0.6 + 0.3 × 1.5                                          │
│      = 0.6 + 0.45                                               │
│      = 1.05                                                     │
│                                                                  │
│   Chuyển sang probability:                                       │
│   p = 1 / (1 + e⁻¹·⁰⁵) = 1 / (1 + 0.350) = 1/1.350 = 0.741     │
│                                                                  │
│   📈 Tăng từ 64.6% → 74.1%                                       │
└─────────────────────────────────────────────────────────────────┘
```

**🔹 Sau Tree 3 (output = -0.8):**
```
┌─────────────────────────────────────────────────────────────────┐
│   F₃ = F₂ + η × f₃                                              │
│      = 1.05 + 0.3 × (-0.8)                                      │
│      = 1.05 - 0.24                                              │
│      = 0.81                                                     │
│                                                                  │
│   Chuyển sang probability (Sigmoid):                             │
│   p = 1 / (1 + e⁻⁰·⁸¹) = 1 / (1 + 0.445) = 1/1.445 = 0.692     │
│                                                                  │
│   📉 Giảm từ 74.1% → 69.2% (Tree 3 điều chỉnh ngược!)           │
└─────────────────────────────────────────────────────────────────┘
```

##### 📌 BƯỚC 3: Tổng hợp quá trình

```
╔════════╦═══════╦═════════════╦═════════════════════╦═══════════╦═════════════╗
║  Bước  ║  Cây  ║ Tree Output ║ Sau × learning_rate ║  F tích   ║ Probability ║
║        ║       ║   (raw)     ║      (η=0.3)        ║   lũy     ║  (Sigmoid)  ║
╠════════╬═══════╬═════════════╬═════════════════════╬═══════════╬═════════════╣
║   0    ║   -   ║      -      ║         -           ║   0.00    ║    0.500    ║
║   1    ║  T1   ║    +2.0     ║       +0.60         ║   0.60    ║    0.646    ║
║   2    ║  T2   ║    +1.5     ║       +0.45         ║   1.05    ║    0.741    ║
║   3    ║  T3   ║    -0.8     ║       -0.24         ║   0.81    ║    0.692    ║
╚════════╩═══════╩═════════════╩═════════════════════╩═══════════╩═════════════╝
```

##### 💡 MINH HỌA TRỰC QUAN

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     TIẾN TRÌNH DỰ ĐOÁN                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Probability                                                               │
│                                                                             │
│   100% ─┤                                                                   │
│         │                                                                   │
│    80% ─┤                              ●───────●                            │
│         │                           74.1%    ↘                              │
│    60% ─┤             ●───────────↗            ●  69.2%  ← KẾT QUẢ          │
│         │          64.6%                                                    │
│    50% ─┤    ●                                                              │
│         │  50.0%                                                            │
│    40% ─┤   ↑                                                               │
│         │ Start                                                             │
│    20% ─┤                                                                   │
│         │                                                                   │
│     0% ─┼────────┬─────────┬──────────┬─────────→ Iterations               │
│              T1        T2         T3                                        │
│             (+0.6)   (+0.45)   (-0.24)                                      │
│                                                                             │
│   📝 Tree 3 có output âm → làm GIẢM probability (điều chỉnh overshoot)     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

##### 💡 NHẬN XÉT QUAN TRỌNG

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  ✅ Learning rate = 0.3 → mỗi cây chỉ đóng góp 30% output gốc              │
│     → Giúp training ổn định, không "nhảy" quá mạnh                         │
│                                                                             │
│  ✅ Tree 3 có output ÂM (-0.8) → làm GIẢM probability                       │
│     → Đây là cơ chế "tự điều chỉnh" của Gradient Boosting                  │
│     → Nếu các cây trước overshoot, cây sau sẽ kéo ngược lại                │
│                                                                             │
│  ✅ Final: p = 69.2% → Dự đoán Class 1 (threshold = 0.5)                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 📝 BÀI TẬP 7: Tính Feature Importance theo CatBoost

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  🎯 MỤC TIÊU: Học cách tính và giải thích Feature Importance                │
│               - công cụ quan trọng để hiểu model                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 📋 ĐỀ BÀI

Cho model CatBoost với 3 features và thống kê splits:

```
┌────────────┬──────────────┬────────────┐
│  Feature   │ Số lần split │ Tổng Gain  │
├────────────┼──────────────┼────────────┤
│  Nhiệt độ  │      15      │    45.6    │
│  Độ ẩm     │      10      │    32.1    │
│  Gió       │       5      │    12.3    │
└────────────┴──────────────┴────────────┘
```

**Yêu cầu:** Tính Feature Importance theo phương pháp Gain

---

#### ✏️ LỜI GIẢI CHI TIẾT

##### 📌 BƯỚC 1: Hiểu Feature Importance

```
╔════════════════════════════════════════════════════════════════════════════╗
║                    FEATURE IMPORTANCE LÀ GÌ?                                ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                             ║
║  Feature Importance cho biết mức độ ĐÓNG GÓP của mỗi feature               ║
║  vào khả năng dự đoán của model.                                           ║
║                                                                             ║
║  Phương pháp GAIN:                                                          ║
║  • Đo lường mức độ GIẢM impurity khi split theo feature đó                  ║
║  • Gain cao → Feature quan trọng (giúp phân chia dữ liệu tốt)              ║
║                                                                             ║
║                      Gain của Feature i                                     ║
║  Importance_i = ─────────────────────────── × 100%                          ║
║                    Tổng Gain tất cả Features                                ║
║                                                                             ║
╚════════════════════════════════════════════════════════════════════════════╝
```

##### 📌 BƯỚC 2: Tính Total Gain

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│   Total_Gain = Gain_NhietDo + Gain_DoAm + Gain_Gio               │
│              = 45.6 + 32.1 + 12.3                                │
│              = 90.0                                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

##### 📌 BƯỚC 3: Tính Importance từng Feature

**🔹 NHIỆT ĐỘ:**
```
┌─────────────────────────────────────────────────────────────────┐
│                     Gain_NhietDo                                 │
│   Importance = ──────────────────── × 100%                       │
│                    Total_Gain                                    │
│                                                                  │
│                   45.6                                           │
│              = ─────── × 100%                                    │
│                  90.0                                            │
│                                                                  │
│              = 0.5067 × 100%                                     │
│                                                                  │
│              = 50.67%                                            │
└─────────────────────────────────────────────────────────────────┘
```

**🔹 ĐỘ ẨM:**
```
┌─────────────────────────────────────────────────────────────────┐
│                      Gain_DoAm                                   │
│   Importance = ──────────────────── × 100%                       │
│                    Total_Gain                                    │
│                                                                  │
│                   32.1                                           │
│              = ─────── × 100%                                    │
│                  90.0                                            │
│                                                                  │
│              = 0.3567 × 100%                                     │
│                                                                  │
│              = 35.67%                                            │
└─────────────────────────────────────────────────────────────────┘
```

**🔹 GIÓ:**
```
┌─────────────────────────────────────────────────────────────────┐
│                       Gain_Gio                                   │
│   Importance = ──────────────────── × 100%                       │
│                    Total_Gain                                    │
│                                                                  │
│                   12.3                                           │
│              = ─────── × 100%                                    │
│                  90.0                                            │
│                                                                  │
│              = 0.1367 × 100%                                     │
│                                                                  │
│              = 13.67%                                            │
└─────────────────────────────────────────────────────────────────┘
```

##### 📌 BƯỚC 4: Tổng hợp và xếp hạng

```
╔══════════╦════════════╦════════════╦═════════════╦══════════════════════════╗
║ Xếp hạng ║  Feature   ║   Gain     ║ Importance  ║        Đánh giá          ║
╠══════════╬════════════╬════════════╬═════════════╬══════════════════════════╣
║    🥇    ║  Nhiệt độ  ║   45.6     ║   50.67%    ║ Quan trọng NHẤT          ║
║    🥈    ║  Độ ẩm     ║   32.1     ║   35.67%    ║ Quan trọng               ║
║    🥉    ║  Gió       ║   12.3     ║   13.67%    ║ Ít quan trọng            ║
╠══════════╩════════════╩════════════╩═════════════╩══════════════════════════╣
║                        TỔNG       │   90.0     │   100.00%                  ║
╚════════════════════════════════════════════════════════════════════════════╝
```

##### 💡 MINH HỌA TRỰC QUAN

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BIỂU ĐỒ FEATURE IMPORTANCE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Nhiệt độ  ████████████████████████████████████████████████████  50.67%    │
│                                                                             │
│  Độ ẩm     ████████████████████████████████████                  35.67%    │
│                                                                             │
│  Gió       ██████████████                                        13.67%    │
│                                                                             │
│            0%       20%       40%       60%       80%       100%            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

##### 💡 KẾT LUẬN VÀ ỨNG DỤNG

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  📊 PHÂN TÍCH:                                                              │
│                                                                             │
│  • NHIỆT ĐỘ chiếm hơn 50% importance                                        │
│    → Yếu tố quan trọng nhất quyết định có mưa hay không                    │
│    → Cần đo lường chính xác và đầy đủ                                      │
│                                                                             │
│  • ĐỘ ẨM đứng thứ 2 với ~36%                                                │
│    → Có mối liên hệ chặt chẽ với mưa                                       │
│    → Kết hợp với nhiệt độ cho dự đoán tốt                                  │
│                                                                             │
│  • GIÓ chỉ chiếm ~14%                                                       │
│    → Ít ảnh hưởng đến việc có mưa hay không                                │
│    → Có thể cân nhắc loại bỏ nếu muốn đơn giản hóa model                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 📝 BÀI TẬP 8: So sánh One-Hot Encoding vs Target Statistics

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  🎯 MỤC TIÊU: So sánh 2 phương pháp mã hóa categorical features             │
│               và hiểu tại sao CatBoost chọn Target Statistics              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 📋 ĐỀ BÀI

Cho categorical feature **"Mùa"** với 4 giá trị: Xuân, Hạ, Thu, Đông

```
┌───────┬────────┬───────────┐
│  Mẫu  │  Mùa   │  Mưa (y)  │
├───────┼────────┼───────────┤
│   1   │  Xuân  │     1     │
│   2   │  Hạ    │     1     │
│   3   │  Thu   │     0     │
│   4   │  Đông  │     0     │
│   5   │  Xuân  │     1     │
│   6   │  Hạ    │     1     │
│   7   │  Thu   │     1     │
│   8   │  Đông  │     0     │
└───────┴────────┴───────────┘
```

**Yêu cầu:** So sánh One-Hot Encoding với Target Statistics (prior=1)

---

#### ✏️ LỜI GIẢI CHI TIẾT

##### 📌 PHƯƠNG PHÁP 1: ONE-HOT ENCODING (Truyền thống)

```
╔════════════════════════════════════════════════════════════════════════════╗
║                      ONE-HOT ENCODING                                       ║
╠════════════════════════════════════════════════════════════════════════════╣
║  Tạo 1 cột binary (0/1) cho MỖI category                                   ║
╚════════════════════════════════════════════════════════════════════════════╝

┌───────┬────────┬────────┬────────┬────────┬─────────┐
│  Mẫu  │  Xuân  │   Hạ   │  Thu   │  Đông  │  y      │
├───────┼────────┼────────┼────────┼────────┼─────────┤
│   1   │   1    │   0    │   0    │   0    │   1     │
│   2   │   0    │   1    │   0    │   0    │   1     │
│   3   │   0    │   0    │   1    │   0    │   0     │
│   4   │   0    │   0    │   0    │   1    │   0     │
│   5   │   1    │   0    │   0    │   0    │   1     │
│   6   │   0    │   1    │   0    │   0    │   1     │
│   7   │   0    │   0    │   1    │   0    │   1     │
│   8   │   0    │   0    │   0    │   1    │   0     │
└───────┴────────┴────────┴────────┴────────┴─────────┘

⚠️ VẤN ĐỀ:
• 1 feature "Mùa" → 4 features mới
• Ma trận sparse (nhiều số 0)
• Không chứa thông tin về target
• Tăng kích thước dữ liệu
```

##### 📌 PHƯƠNG PHÁP 2: TARGET STATISTICS (CatBoost)

**🔹 Bước 1: Tính Global Mean**
```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│   global_mean = Tổng y / Số mẫu                                  │
│               = (1+1+0+0+1+1+1+0) / 8                            │
│               = 5/8                                              │
│               = 0.625                                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**🔹 Bước 2: Thống kê theo từng Mùa**

```
┌──────────┬─────────────┬─────────┬──────────┬─────────────────┐
│   Mùa    │   Các mẫu   │  count  │  sum(y)  │  mean_target    │
├──────────┼─────────────┼─────────┼──────────┼─────────────────┤
│   Xuân   │    1, 5     │    2    │  1+1=2   │  2/2 = 1.0      │
│   Hạ     │    2, 6     │    2    │  1+1=2   │  2/2 = 1.0      │
│   Thu    │    3, 7     │    2    │  0+1=1   │  1/2 = 0.5      │
│   Đông   │    4, 8     │    2    │  0+0=0   │  0/2 = 0.0      │
└──────────┴─────────────┴─────────┴──────────┴─────────────────┘
```

**🔹 Bước 3: Tính Target Statistics**

```
╔════════════════════════════════════════════════════════════════════════════╗
║   TS = (count × mean_target + prior × global_mean) / (count + prior)        ║
╚════════════════════════════════════════════════════════════════════════════╝
```

**XUÂN:**
```
┌─────────────────────────────────────────────────────────────────┐
│         2 × 1.0 + 1 × 0.625       2.0 + 0.625                   │
│   TS = ──────────────────────── = ────────────── = 0.875        │
│              2 + 1                     3                        │
└─────────────────────────────────────────────────────────────────┘
```

**HẠ:**
```
┌─────────────────────────────────────────────────────────────────┐
│         2 × 1.0 + 1 × 0.625       2.0 + 0.625                   │
│   TS = ──────────────────────── = ────────────── = 0.875        │
│              2 + 1                     3                        │
└─────────────────────────────────────────────────────────────────┘
```

**THU:**
```
┌─────────────────────────────────────────────────────────────────┐
│         2 × 0.5 + 1 × 0.625       1.0 + 0.625                   │
│   TS = ──────────────────────── = ────────────── = 0.542        │
│              2 + 1                     3                        │
└─────────────────────────────────────────────────────────────────┘
```

**ĐÔNG:**
```
┌─────────────────────────────────────────────────────────────────┐
│         2 × 0.0 + 1 × 0.625       0.0 + 0.625                   │
│   TS = ──────────────────────── = ────────────── = 0.208        │
│              2 + 1                     3                        │
└─────────────────────────────────────────────────────────────────┘
```

**🔹 Bước 4: Kết quả với Target Statistics**

```
┌───────┬──────────────────────┬─────────┐
│  Mẫu  │  Mùa_TS (số thực)    │    y    │
├───────┼──────────────────────┼─────────┤
│   1   │       0.875          │    1    │  ← Xuân: mưa nhiều
│   2   │       0.875          │    1    │  ← Hạ: mưa nhiều
│   3   │       0.542          │    0    │  ← Thu: mưa vừa
│   4   │       0.208          │    0    │  ← Đông: ít mưa
│   5   │       0.875          │    1    │  ← Xuân
│   6   │       0.875          │    1    │  ← Hạ
│   7   │       0.542          │    1    │  ← Thu
│   8   │       0.208          │    0    │  ← Đông
└───────┴──────────────────────┴─────────┘

✅ Chỉ CẦN 1 CỘT thay vì 4 cột!
✅ Giá trị phản ánh xác suất mưa của từng mùa!
```

##### 📌 SO SÁNH HAI PHƯƠNG PHÁP

```
╔═════════════════════════╦══════════════════════╦═══════════════════════════╗
║       TIÊU CHÍ          ║    ONE-HOT ENCODING  ║    TARGET STATISTICS      ║
╠═════════════════════════╬══════════════════════╬═══════════════════════════╣
║  Số features tạo ra     ║         4            ║           1               ║
╠═════════════════════════╬══════════════════════╬═══════════════════════════╣
║  Sparsity (thưa)        ║       CAO            ║         THẤP              ║
╠═════════════════════════╬══════════════════════╬═══════════════════════════╣
║  Thông tin target       ║     KHÔNG CÓ         ║          CÓ               ║
╠═════════════════════════╬══════════════════════╬═══════════════════════════╣
║  Giải thích được        ║    Rõ ràng (0/1)     ║    Số thực (xác suất)     ║
╠═════════════════════════╬══════════════════════╬═══════════════════════════╣
║  Tốc độ training        ║     CHẬM HƠN         ║        NHANH HƠN          ║
╠═════════════════════════╬══════════════════════╬═══════════════════════════╣
║  High cardinality       ║    ❌ Không ổn        ║       ✅ Tốt               ║
║  (nhiều categories)     ║    (quá nhiều cột)   ║       (vẫn 1 cột)         ║
╚═════════════════════════╩══════════════════════╩═══════════════════════════╝
```

##### 💡 MINH HỌA TRỰC QUAN

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│     ONE-HOT ENCODING                   TARGET STATISTICS                    │
│                                                                             │
│   ┌─────────────────────┐             ┌───────────────────┐                 │
│   │ Mùa │Xu│Hạ│Thu│Đông│             │ Mùa │  Mùa_TS      │                 │
│   ├─────┼──┼──┼───┼────┤             ├─────┼──────────────┤                 │
│   │Xuân │ 1│ 0│  0│   0│             │Xuân │    0.875     │ ← Mưa nhiều    │
│   │Hạ   │ 0│ 1│  0│   0│    ──→      │Hạ   │    0.875     │ ← Mưa nhiều    │
│   │Thu  │ 0│ 0│  1│   0│             │Thu  │    0.542     │ ← Mưa vừa      │
│   │Đông │ 0│ 0│  0│   1│             │Đông │    0.208     │ ← Ít mưa       │
│   └─────┴──┴──┴───┴────┘             └─────┴──────────────┘                 │
│                                                                             │
│      4 cột sparse                        1 cột có ý nghĩa                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

##### 💡 KẾT LUẬN

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  🏆 TARGET STATISTICS (CatBoost) có nhiều ưu điểm hơn:                      │
│                                                                             │
│  ✅ Giảm số chiều dữ liệu (dimensionality reduction)                        │
│  ✅ Encode thông tin hữu ích về target vào feature                          │
│  ✅ Xử lý được categorical có nhiều giá trị (high cardinality)              │
│  ✅ Tăng tốc độ training và inference                                       │
│  ✅ Không cần tiền xử lý thủ công                                           │
│                                                                             │
│  → Đây là lý do CatBoost MẠNH với dữ liệu categorical!                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 14. Tài Liệu Tham Khảo

### 14.1. Official Documentation
- [CatBoost Official Documentation](https://catboost.ai/docs/)
- [CatBoost GitHub Repository](https://github.com/catboost/catboost)
- [CatBoost Tutorials](https://catboost.ai/docs/concepts/tutorials.html)

### 14.2. Research Papers
- Prokhorenkova, L., et al. (2018). "CatBoost: unbiased boosting with categorical features"
- Dorogush, A. V., et al. (2018). "CatBoost: gradient boosting with categorical features support"

### 14.3. Bài viết tham khảo
- [Interdata - CatBoost là gì?](https://interdata.vn/blog/catboost-la-gi/)
- [FUNiX - CatBoost: Một thư viện máy học để xử lý dữ liệu](https://funix.edu.vn/chia-se-kien-thuc/catboost-mot-thu-vien-may-hoc-de-xu-ly-du-lieu/)

### 14.4. API Reference

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
