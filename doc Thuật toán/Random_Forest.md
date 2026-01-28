# Random Forest - Thuật Toán Rừng Ngẫu Nhiên

## Mục Lục
1. [Giới thiệu tổng quan](#1-giới-thiệu-tổng-quan)
2. [Random Forest là gì?](#2-random-forest-là-gì)
3. [Nguyên lý hoạt động](#3-nguyên-lý-hoạt-động)
4. [Xây dựng thuật toán Random Forest](#4-xây-dựng-thuật-toán-random-forest)
5. [Tại sao Random Forest hiệu quả?](#5-tại-sao-random-forest-hiệu-quả)
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

**Random Forest** (Rừng Ngẫu Nhiên) là một trong những thuật toán học máy phổ biến và hiệu quả nhất hiện nay. Thuật toán này thuộc nhóm **Ensemble Learning** - phương pháp kết hợp nhiều mô hình để đưa ra dự đoán tốt hơn.

### 1.1. Ý tưởng cốt lõi

> "Random là ngẫu nhiên, Forest là rừng - ở thuật toán Random Forest, ta xây dựng nhiều cây quyết định (Decision Tree), mỗi cây có yếu tố random khác nhau, sau đó kết quả dự đoán được tổng hợp từ các cây."

### 1.2. Loại bài toán

Random Forest là thuật toán **Supervised Learning**, có thể giải quyết cả:
- **Classification** (Phân loại): Dự đoán nhãn/lớp
- **Regression** (Hồi quy): Dự đoán giá trị số liên tục

---

## 2. Random Forest Là Gì?

### 2.1. Định nghĩa

**Random Forest** là một thuật toán học máy sử dụng nhiều cây quyết định (Decision Tree) để đưa ra dự đoán tốt hơn. Mỗi cây nhìn vào các phần ngẫu nhiên khác nhau của dữ liệu, và kết quả cuối cùng được tổng hợp bằng:
- **Voting** (bình chọn) cho bài toán phân loại
- **Averaging** (lấy trung bình) cho bài toán hồi quy

### 2.2. Ensemble Learning

Random Forest là một kỹ thuật **Ensemble Learning** - phương pháp kết hợp nhiều "weak learners" (các mô hình yếu) để tạo thành một "strong learner" (mô hình mạnh).

```
┌─────────────────────────────────────────────────────────────────┐
│                    RANDOM FOREST CONCEPT                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│    Input Data                                                   │
│        │                                                        │
│        ▼                                                        │
│   ┌────┴────┐                                                   │
│   │         │                                                   │
│   ▼         ▼         ▼         ▼         ▼                     │
│ Tree 1   Tree 2   Tree 3   Tree 4   Tree 5  ...                 │
│   │         │         │         │         │                     │
│   ▼         ▼         ▼         ▼         ▼                     │
│ Pred 1   Pred 2   Pred 3   Pred 4   Pred 5                      │
│   │         │         │         │         │                     │
│   └────────────────────┬────────────────────┘                   │
│                        │                                        │
│                        ▼                                        │
│              ┌─────────────────┐                                │
│              │  Aggregation   │                                 │
│              │ (Vote/Average) │                                 │
│              └────────┬───────┘                                 │
│                       │                                         │
│                       ▼                                         │
│               Final Prediction                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3. The Wisdom of Crowds

Ý tưởng của Random Forest tương tự với khái niệm **"The Wisdom of Crowds"** (Trí tuệ đám đông) được đề xuất bởi James Surowiecki vào năm 2004:

> "Thông thường, tổng hợp thông tin từ một nhóm sẽ tốt hơn từ một cá nhân."

**Ví dụ thực tế:** Khi mua sản phẩm trên Tiki/Shopee:
- Nếu chỉ đọc 1 review → có thể là ý kiến chủ quan
- Đọc tất cả reviews → có cái nhìn tổng quan và chính xác hơn

```
┌─────────────────────────────────────────────────────────────────┐
│          SO SÁNH RANDOM FOREST VÀ WISDOM OF CROWDS              │
├──────────────────────────────┬──────────────────────────────────┤
│       Wisdom of Crowds       │         Random Forest            │
├──────────────────────────────┼──────────────────────────────────┤
│ Nhiều người đánh giá         │ Nhiều cây quyết định             │
│ Mỗi người có góc nhìn khác   │ Mỗi cây dùng data/features khác  │
│ Tổng hợp ý kiến của tất cả   │ Tổng hợp dự đoán của tất cả      │
│ Kết quả chính xác hơn        │ Dự đoán chính xác hơn            │
└──────────────────────────────┴──────────────────────────────────┘
```

---

## 3. Nguyên Lý Hoạt Động

### 3.1. Quy trình tổng quan

```
┌─────────────────────────────────────────────────────────────────┐
│                 RANDOM FOREST WORKFLOW                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  TRAINING PHASE:                                                │
│  ===============                                                │
│  1. Tạo nhiều bootstrap samples từ training data                │
│  2. Với mỗi sample, chọn ngẫu nhiên k features                  │
│  3. Xây dựng Decision Tree cho mỗi sample                       │
│  4. Lặp lại để tạo N cây (forest)                               │
│                                                                 │
│  PREDICTION PHASE:                                              │
│  =================                                              │
│  1. Đưa dữ liệu mới vào TẤT CẢ các cây                          │
│  2. Mỗi cây đưa ra một dự đoán                                  │
│  3. Tổng hợp kết quả:                                           │
│     - Classification: Majority voting (đa số thắng)             │
│     - Regression: Average (lấy trung bình)                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2. Các bước hoạt động chi tiết

#### Bước 1: Tạo nhiều Decision Trees
- Thuật toán tạo ra nhiều cây quyết định
- Mỗi cây sử dụng một phần ngẫu nhiên của dữ liệu
- Mỗi cây có thể khác nhau về cấu trúc

#### Bước 2: Chọn ngẫu nhiên Features
- Khi xây dựng mỗi cây, không xét tất cả features
- Chọn ngẫu nhiên một số features để quyết định cách split
- Giúp các cây đa dạng, không giống nhau

#### Bước 3: Mỗi cây đưa ra dự đoán
- Mỗi cây đưa ra kết quả dựa trên những gì đã học
- Từ phần dữ liệu của riêng nó

#### Bước 4: Tổng hợp kết quả
- **Classification**: Chọn class có nhiều cây vote nhất (majority voting)
- **Regression**: Lấy trung bình các dự đoán

### 3.3. Minh họa quá trình dự đoán

```
INPUT: New Data Point
         │
         ▼
    ┌────┴────┬────┬────┬────┬────┐
    │         │    │    │    │    │
    ▼         ▼    ▼    ▼    ▼    ▼
 Tree 1    Tree 2  T3   T4   T5   T6
    │         │    │    │    │    │
    ▼         ▼    ▼    ▼    ▼    ▼
   "1"       "1"  "0"  "1"  "1"  "1"
    │         │    │    │    │    │
    └─────────┴────┴────┴────┴────┘
                   │
                   ▼
           ┌─────────────┐
           │  VOTING:    │
           │  "1" = 5    │
           │  "0" = 1    │
           └──────┬──────┘
                  │
                  ▼
         Final: "1" (đa số thắng)
```

---

## 4. Xây Dựng Thuật Toán Random Forest

### 4.1. Input
- Bộ dữ liệu gồm **n samples** (dữ liệu)
- Mỗi sample có **d features** (thuộc tính)

### 4.2. Quy trình xây dựng mỗi cây

#### Bước 1: Bootstrapping (Random Sampling with Replacement)

```python
# Lấy ngẫu nhiên n dữ liệu từ bộ dữ liệu gốc
# Cho phép trùng lặp (sampling with replacement)

Original: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                    │
                    ▼ (Bootstrap sampling)
Sample 1: [2, 3, 3, 5, 7, 7, 8, 9, 10, 10]  # Có thể trùng
Sample 2: [1, 1, 2, 4, 4, 6, 7, 8, 9, 10]
Sample 3: [1, 3, 3, 5, 5, 6, 8, 8, 9, 9]
...
```

**Đặc điểm của Bootstrapping:**
- Khi sample được 1 dữ liệu, **không bỏ ra** mà giữ lại
- Tiếp tục sample cho đến khi đủ n dữ liệu
- Kết quả: tập dữ liệu mới có thể có dữ liệu **trùng lặp**

#### Bước 2: Random Feature Selection

```python
# Chọn ngẫu nhiên k features (k < d)

All Features: [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10]
                              │
                              ▼ (Random selection, k=3)
Tree 1 Features: [F2, F5, F8]
Tree 2 Features: [F1, F4, F7]
Tree 3 Features: [F3, F6, F9]
...
```

**Giá trị k thường dùng:**
- **Classification**: $k = \sqrt{d}$ (căn bậc 2 của số features)
- **Regression**: $k = d/3$ (1/3 số features)

#### Bước 3: Xây dựng Decision Tree

```
Dùng thuật toán Decision Tree để xây dựng cây với:
- Dữ liệu: n samples từ bước 1
- Features: k features từ bước 2
```

### 4.3. Pseudo-code

```python
def build_random_forest(data, n_trees, max_features):
    forest = []
    n_samples = len(data)
    
    for i in range(n_trees):
        # Bước 1: Bootstrap sampling
        bootstrap_indices = random.choices(range(n_samples), k=n_samples)
        bootstrap_sample = data[bootstrap_indices]
        
        # Bước 2: Random feature selection
        selected_features = random.sample(all_features, k=max_features)
        bootstrap_sample = bootstrap_sample[selected_features]
        
        # Bước 3: Build Decision Tree
        tree = DecisionTree()
        tree.fit(bootstrap_sample)
        
        forest.append(tree)
    
    return forest

def predict_random_forest(forest, new_data):
    predictions = []
    for tree in forest:
        pred = tree.predict(new_data)
        predictions.append(pred)
    
    # Majority voting (classification)
    return mode(predictions)
    
    # Hoặc average (regression)
    # return mean(predictions)
```

### 4.4. Tóm tắt quá trình

```
┌────────────────────────────────────────────────────────────────┐
│              QUY TRÌNH XÂY DỰNG RANDOM FOREST                  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Original Dataset (n samples, d features)                      │
│          │                                                     │
│          ▼                                                     │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ Repeat for each tree (i = 1, 2, ..., N):                  │ │
│  │                                                           │ │
│  │   1. Bootstrap: Lấy ngẫu nhiên n samples (có trùng lặp)   │ │
│  │   2. Random Features: Chọn ngẫu nhiên k features          │ │
│  │   3. Build Tree: Xây Decision Tree với data từ 1, 2       │ │
│  │   4. Add tree to forest                                   │ │
│  │                                                           │ │
│  └───────────────────────────────────────────────────────────┘ │
│          │                                                     │
│          ▼                                                     │
│  Random Forest = [Tree_1, Tree_2, ..., Tree_N]                 │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 5. Tại Sao Random Forest Hiệu Quả?

### 5.1. Vấn đề với Decision Tree đơn lẻ

Trong thuật toán Decision Tree:
- Nếu để độ sâu tùy ý → cây phân loại đúng hết training data
- Dẫn đến **overfitting** (high variance)
- Dự đoán tệ trên validation/test data

### 5.2. Cách Random Forest giải quyết

Random Forest giải quyết vấn đề overfitting bằng **2 yếu tố ngẫu nhiên**:

```
┌─────────────────────────────────────────────────────────────────┐
│           CƠ CHẾ CHỐNG OVERFITTING CỦA RANDOM FOREST            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 1. RANDOM DATA (Bootstrapping)                                  │
│    - Mỗi cây chỉ dùng MỘT PHẦN dữ liệu training                 │
│    - Không cây nào thấy được TẤT CẢ dữ liệu                     │
│                                                                 │
│ 2. RANDOM FEATURES                                              │
│    - Mỗi cây chỉ dùng MỘT SỐ features                           │
│    - Không cây nào dùng TẤT CẢ features                         │
│                                                                 │
│ KẾT QUẢ:                                                        │
│    - Mỗi cây riêng lẻ có thể dự đoán KHÔNG TỐT (high bias)      │
│    - Nhưng tổng hợp các cây → BỔ SUNG THÔNG TIN cho nhau        │
│    - → LOW BIAS + LOW VARIANCE                                  │
│    - → Dự đoán TỐT hơn!                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3. Bias-Variance Trade-off

| Mô hình | Bias | Variance | Kết quả |
|---------|------|----------|---------|
| Decision Tree (deep) | Low | High | Overfitting |
| Decision Tree (shallow) | High | Low | Underfitting |
| **Random Forest** | **Low** | **Low** | **Tốt nhất!** |

### 5.4. Tại sao hoạt động?

1. **Diversity (Đa dạng)**:
   - Các cây khác nhau do random data + random features
   - Mỗi cây học được những pattern khác nhau

2. **Error Reduction (Giảm lỗi)**:
   - Lỗi của các cây thường uncorrelated (không tương quan)
   - Khi tổng hợp, lỗi có xu hướng triệt tiêu nhau

3. **Robustness (Bền vững)**:
   - Một vài cây sai → không ảnh hưởng nhiều đến kết quả cuối
   - Majority voting / averaging giúp "lọc" noise

---

## 6. Các Tham Số Quan Trọng

### 6.1. Tham số chính của Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,      # Số lượng cây
    max_depth=None,        # Độ sâu tối đa của cây
    max_features='sqrt',   # Số features cho mỗi split
    min_samples_split=2,   # Số samples tối thiểu để split
    min_samples_leaf=1,    # Số samples tối thiểu ở leaf
    bootstrap=True,        # Có dùng bootstrap hay không
    random_state=42,       # Seed cho reproducibility
    n_jobs=-1              # Số CPU cores sử dụng
)
```

### 6.2. Chi tiết các tham số

#### **n_estimators** (int, default=100)
- Số lượng cây quyết định trong forest
- Nhiều cây hơn → kết quả ổn định hơn → nhưng chậm hơn

```python
# Khuyến nghị
n_estimators = 100   # Mặc định, đủ cho hầu hết cases
n_estimators = 200   # Cho bộ dữ liệu lớn
n_estimators = 500   # Khi cần độ chính xác cao
n_estimators = 1000  # Cho production (nếu có đủ tài nguyên)
```

#### **max_depth** (int, default=None)
- Độ sâu tối đa của mỗi cây
- `None` = cây mở rộng cho đến khi tất cả leaves đều pure

```python
max_depth = None  # Để cây phát triển tự do
max_depth = 10    # Giới hạn để tránh overfitting
max_depth = 20    # Cho bộ dữ liệu phức tạp
```

#### **max_features** (int, float, string)
- Số features xét khi tìm best split

```python
max_features = 'sqrt'  # sqrt(n_features) - tốt cho classification
max_features = 'log2'  # log2(n_features)
max_features = 0.5     # 50% features
max_features = 5       # Chính xác 5 features
max_features = None    # Dùng tất cả features
```

#### **min_samples_split** (int, default=2)
- Số samples tối thiểu để split một node

```python
min_samples_split = 2   # Mặc định
min_samples_split = 5   # Để giảm overfitting
min_samples_split = 10  # Cho bộ dữ liệu lớn
```

#### **min_samples_leaf** (int, default=1)
- Số samples tối thiểu ở leaf node

```python
min_samples_leaf = 1   # Mặc định
min_samples_leaf = 5   # Để giảm overfitting
min_samples_leaf = 10  # Cho bộ dữ liệu lớn
```

#### **bootstrap** (bool, default=True)
- Có sử dụng bootstrap sampling hay không

```python
bootstrap = True   # Dùng bootstrap (khuyến nghị)
bootstrap = False  # Mỗi cây dùng toàn bộ dataset
```

### 6.3. Bảng tổng hợp tham số

| Tham số | Mặc định | Ảnh hưởng |
|---------|----------|-----------|
| `n_estimators` | 100 | Nhiều hơn → ổn định hơn, chậm hơn |
| `max_depth` | None | Nhỏ hơn → giảm overfitting |
| `max_features` | 'sqrt' | Nhỏ hơn → đa dạng hơn |
| `min_samples_split` | 2 | Lớn hơn → giảm overfitting |
| `min_samples_leaf` | 1 | Lớn hơn → giảm overfitting |
| `bootstrap` | True | True → tăng diversity |

---

## 7. So Sánh Với Các Thuật Toán Khác

### 7.1. Random Forest vs Decision Tree

| Tiêu chí | Decision Tree | Random Forest |
|----------|---------------|---------------|
| Số lượng cây | 1 | Nhiều (100+) |
| Overfitting | Dễ bị | Ít bị hơn |
| Variance | High | Low |
| Interpretability | Dễ giải thích | Khó hơn |
| Tốc độ training | Nhanh | Chậm hơn |
| Tốc độ inference | Nhanh | Chậm hơn |
| Accuracy | Thấp hơn | Cao hơn |

### 7.2. Random Forest vs Gradient Boosting

| Tiêu chí | Random Forest | Gradient Boosting |
|----------|---------------|-------------------|
| Cách xây dựng | Song song (parallel) | Tuần tự (sequential) |
| Overfitting | Ít | Có thể bị |
| Hyperparameter tuning | Ít cần thiết | Cần nhiều |
| Accuracy | Tốt | Có thể tốt hơn |
| Training time | Có thể song song hóa | Phải tuần tự |
| Robustness | Cao | Trung bình |

### 7.3. Random Forest vs Bagging

| Tiêu chí | Bagging | Random Forest |
|----------|---------|---------------|
| Random data | ✓ | ✓ |
| Random features | ✗ | ✓ |
| Diversity | Thấp hơn | Cao hơn |
| Performance | Tốt | Tốt hơn |

---

## 8. Ưu Điểm Và Hạn Chế

### 8.1. Ưu điểm

```
┌─────────────────────────────────────────────────────────────────┐
│                    ƯU ĐIỂM CỦA RANDOM FOREST                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ ★ Độ chính xác cao                                              │
│   - Đạt accuracy cao trên nhiều loại bài toán                   │
│   - Thường nằm trong top performers                             │
│                                                                 │
│ ★ Xử lý missing data tốt                                        │
│   - Có thể hoạt động với dữ liệu thiếu                          │
│   - Không bắt buộc phải fill missing values                     │
│                                                                 │
│ ★ Không cần normalize/standardize                               │
│   - Hoạt động tốt với raw data                                  │
│   - Không ảnh hưởng bởi scale của features                      │
│                                                                 │
│ ★ Feature Importance                                            │
│   - Cho biết features nào quan trọng nhất                       │
│   - Hữu ích cho feature selection                               │
│                                                                 │
│ ★ Chống Overfitting tốt                                         │
│   - Nhờ ensemble và randomization                               │
│   - Giảm variance đáng kể                                       │
│                                                                 │
│ ★ Xử lý dữ liệu lớn và phức tạp                                 │
│   - Có thể parallel training                                    │
│   - Hoạt động tốt với nhiều features                            │
│                                                                 │
│ ★ Đa năng                                                       │
│   - Classification và Regression                                │
│   - Dữ liệu số và categorical                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2. Hạn chế

```
┌─────────────────────────────────────────────────────────────────┐
│                    HẠN CHẾ CỦA RANDOM FOREST                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ ✗ Tốn tài nguyên tính toán                                      │
│   - Training chậm với nhiều cây                                 │
│   - Tốn memory để lưu trữ tất cả cây                            │
│                                                                 │
│ ✗ Khó giải thích                                                │
│   - Không thể visualize như single decision tree                │
│   - "Black box" model                                           │
│                                                                 │
│ ✗ Prediction chậm                                               │
│   - Phải đi qua tất cả các cây                                  │
│   - Không phù hợp cho real-time applications                    │
│                                                                 │
│ ✗ Không hoạt động tốt với dữ liệu rất thưa (sparse)             │
│   - Text data, one-hot encoded data                             │
│                                                                 │
│ ✗ Có thể bias với imbalanced data                               │
│   - Cần xử lý class imbalance riêng                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Ứng Dụng Thực Tế

### 9.1. Phân loại (Classification)

#### Dự đoán khả năng sống sót (Titanic)
```python
# Features: Pclass, Sex, Age, SibSp, Parch, Fare
# Target: Survived (0/1)
```

#### Phát hiện spam email
```python
# Features: word frequencies, sender info, etc.
# Target: spam / not spam
```

#### Chẩn đoán bệnh
```python
# Features: symptoms, test results, medical history
# Target: disease type / healthy
```

### 9.2. Hồi quy (Regression)

#### Dự đoán giá nhà
```python
# Features: location, size, bedrooms, age, etc.
# Target: house price (continuous value)
```

#### Dự đoán doanh thu
```python
# Features: historical sales, season, promotions, etc.
# Target: revenue (continuous value)
```

### 9.3. Dự báo thời tiết

```python
# Features: temperature, humidity, pressure, wind, season, location
# Target: 
#   - Classification: weather type (sunny, rainy, cloudy)
#   - Regression: temperature, rainfall amount

# Ưu điểm cho weather forecasting:
# - Xử lý được nhiều features đa dạng
# - Robust với missing data
# - Có thể extract feature importance
```

### 9.4. Các ứng dụng khác

- **Banking**: Credit scoring, fraud detection
- **Healthcare**: Disease prediction, drug response
- **E-commerce**: Customer churn, product recommendation
- **Manufacturing**: Quality control, predictive maintenance
- **Agriculture**: Crop yield prediction, soil classification

---

## 10. Hướng Dẫn Cài Đặt Và Sử Dụng

### 10.1. Cài đặt

```bash
# Scikit-learn (recommended)
pip install scikit-learn

# Hoặc với conda
conda install scikit-learn
```

### 10.2. Import

```python
# Classification
from sklearn.ensemble import RandomForestClassifier

# Regression
from sklearn.ensemble import RandomForestRegressor

# Utilities
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_squared_error, r2_score
```

### 10.3. Basic Usage

```python
# 1. Khởi tạo model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 2. Train model
model.fit(X_train, y_train)

# 3. Predict
y_pred = model.predict(X_test)

# 4. Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
```

---

## 11. Ví Dụ Thực Hành

### 11.1. Bài toán phân loại - Titanic Survival

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# 1. Load dữ liệu
titanic_data = pd.read_csv('titanic.csv')

# 2. Tiền xử lý
titanic_data = titanic_data.dropna(subset=['Survived'])

# 3. Chọn features và target
X = titanic_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = titanic_data['Survived']

# 4. Encode categorical
X['Sex'] = X['Sex'].map({'female': 0, 'male': 1})
X['Age'] = X['Age'].fillna(X['Age'].median())

# 5. Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Khởi tạo và train
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# 7. Dự đoán
y_pred = rf_classifier.predict(X_test)

# 8. Đánh giá
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 9. Thử với 1 sample
sample = X_test.iloc[0:1]
prediction = rf_classifier.predict(sample)
print(f"\nSample: {sample.iloc[0].to_dict()}")
print(f"Predicted: {'Survived' if prediction[0] == 1 else 'Did Not Survive'}")
```

### 11.2. Bài toán hồi quy - California Housing

```python
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 1. Load dữ liệu
california_housing = fetch_california_housing()
X = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
y = california_housing.target

# 2. Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Khởi tạo và train
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# 4. Dự đoán
y_pred = rf_regressor.predict(X_test)

# 5. Đánh giá
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# 6. Thử với 1 sample
single_data = X_test.iloc[0:1]
predicted_value = rf_regressor.predict(single_data)
print(f"\nPredicted Value: {predicted_value[0]:.2f}")
print(f"Actual Value: {y_test.iloc[0]:.2f}")
```

### 11.3. Feature Importance

```python
import matplotlib.pyplot as plt
import pandas as pd

# Lấy feature importance
feature_importance = rf_classifier.feature_importances_
feature_names = X.columns

# Tạo DataFrame
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("Feature Importance:")
print(importance_df)

# Visualization
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Random Forest Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

### 11.4. Cross-Validation

```python
from sklearn.model_selection import cross_val_score

# Khởi tạo model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 5-Fold Cross Validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"CV Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f}")
print(f"Std CV Score: {cv_scores.std():.4f}")
```

### 11.5. Hyperparameter Tuning với GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

# Định nghĩa parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Khởi tạo model
rf = RandomForestClassifier(random_state=42)

# Grid Search
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='accuracy',
    verbose=1
)

grid_search.fit(X_train, y_train)

# Best parameters
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_:.4f}")

# Best model
best_model = grid_search.best_estimator_
```

### 11.6. Out-of-Bag (OOB) Score

```python
# OOB Score - built-in validation không cần split data

model = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,  # Enable OOB score
    random_state=42
)

model.fit(X, y)

print(f"OOB Score: {model.oob_score_:.4f}")
```

### 11.7. Lưu và tải Model

```python
import joblib

# Lưu model
joblib.dump(rf_classifier, 'random_forest_model.joblib')

# Tải model
loaded_model = joblib.load('random_forest_model.joblib')

# Dự đoán với model đã tải
predictions = loaded_model.predict(X_test)
```

---

## 12. Best Practices

### 12.1. Chọn số lượng cây (n_estimators)

```python
# Vẽ đồ thị accuracy vs n_estimators
import matplotlib.pyplot as plt

n_trees = [10, 50, 100, 200, 300, 500]
scores = []

for n in n_trees:
    model = RandomForestClassifier(n_estimators=n, random_state=42)
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))

plt.plot(n_trees, scores, marker='o')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Trees')
plt.show()
```

### 12.2. Xử lý Imbalanced Data

```python
from sklearn.ensemble import RandomForestClassifier

# Cách 1: Class weight
model = RandomForestClassifier(
    class_weight='balanced',  # Tự động balance
    random_state=42
)

# Cách 2: Custom weight
model = RandomForestClassifier(
    class_weight={0: 1, 1: 5},  # Class 1 quan trọng hơn 5 lần
    random_state=42
)
```

### 12.3. Parallel Processing

```python
# Sử dụng tất cả CPU cores
model = RandomForestClassifier(
    n_estimators=100,
    n_jobs=-1,  # -1 = tất cả cores
    random_state=42
)
```

### 12.4. Tips tối ưu

```
┌─────────────────────────────────────────────────────────────────┐
│                   BEST PRACTICES SUMMARY                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 1. BẮT ĐẦU VỚI DEFAULTS                                         │
│    - Mặc định của sklearn thường đã tốt                         │
│    - Chỉ tune khi cần cải thiện thêm                            │
│                                                                 │
│ 2. TĂNG SỐ CÂY ĐẦU TIÊN                                         │
│    - n_estimators = 100 → 200 → 500                             │
│    - Thường cải thiện accuracy, ít khi làm hại                  │
│                                                                 │
│ 3. GIẢM max_depth NẾU OVERFITTING                               │
│    - max_depth = None → 20 → 10                                 │
│    - Hoặc tăng min_samples_split                                │
│                                                                 │
│ 4. SỬ DỤNG CROSS-VALIDATION                                     │
│    - Luôn dùng CV để đánh giá model                             │
│    - Tránh overfitting trên validation set                      │
│                                                                 │
│ 5. CHECK FEATURE IMPORTANCE                                     │
│    - Xem features nào quan trọng                                │
│    - Có thể remove features không cần thiết                     │
│                                                                 │
│ 6. SỬ DỤNG n_jobs=-1                                            │
│    - Tận dụng multi-core processing                             │
│    - Giảm đáng kể training time                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 13. Tài Liệu Tham Khảo

### 13.1. Official Documentation
- [Scikit-learn Random Forest Documentation](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)
- [Scikit-learn API Reference](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

### 13.2. Research Papers
- Breiman, L. (2001). "Random Forests". Machine Learning, 45(1), 5-32.
- Ho, T. K. (1995). "Random Decision Forests". Proceedings of the 3rd International Conference on Document Analysis and Recognition.

### 13.3. Bài viết tham khảo
- [Machine Learning cơ bản - Random Forest](https://machinelearningcoban.com/tabml_book/ch_model/random_forest.html)
- [GeeksforGeeks - Random Forest Algorithm](https://www.geeksforgeeks.org/random-forest-algorithm-in-machine-learning/)

### 13.4. Thuật ngữ chính

| Thuật ngữ | Tiếng Việt | Giải thích |
|-----------|------------|------------|
| Random Forest | Rừng ngẫu nhiên | Ensemble của nhiều Decision Trees |
| Bootstrapping | Lấy mẫu có hoàn lại | Sampling with replacement |
| Bagging | - | Bootstrap Aggregating |
| Ensemble | Tập hợp | Kết hợp nhiều models |
| Voting | Bình chọn | Majority voting trong classification |
| OOB Score | Điểm Out-of-Bag | Validation không cần split data |
| Feature Importance | Độ quan trọng đặc trưng | Đánh giá tầm quan trọng của mỗi feature |

---

## Kết Luận

**Random Forest** là một thuật toán mạnh mẽ và linh hoạt, phù hợp cho nhiều loại bài toán machine learning. Với những ưu điểm như:

- ✅ Độ chính xác cao
- ✅ Chống overfitting tốt
- ✅ Xử lý missing data
- ✅ Không cần normalize
- ✅ Cung cấp feature importance

Random Forest là lựa chọn an toàn để bắt đầu với bất kỳ bài toán classification hoặc regression nào. Đặc biệt trong bài toán **dự báo thời tiết**, Random Forest có thể được sử dụng để:

- Phân loại điều kiện thời tiết (sunny, rainy, cloudy)
- Dự đoán nhiệt độ, lượng mưa
- Xác định features quan trọng nhất ảnh hưởng đến thời tiết

---

*Tài liệu được tổng hợp và biên soạn cho dự án Weather Forecast App*

*Cập nhật: Tháng 1/2026*
