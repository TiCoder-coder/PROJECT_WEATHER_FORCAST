# Random Forest - Thuật Toán Rừng Ngẫu Nhiên


                                             Thuật Toán Random_Forest
## Mục Lục
1. [Giới thiệu tổng quan]
2. [Random Forest là gì?]
3. [Nguyên lý hoạt động]
4. [Xây dựng thuật toán Random Forest]
5. [Tại sao Random Forest hiệu quả?]
6. [Các tham số quan trọng]
7. [So sánh với các thuật toán khác]
8. [Ưu điểm và hạn chế]
9. [Ứng dụng thực tế]
10. [Hướng dẫn cài đặt và sử dụng]
11. [Ví dụ thực hành]
12. [Best Practices]
13. [Bài tập giải tay]
14. [Ứng dụng thực tế chi tiết]
15. [Tài liệu tham khảo]
16. [Kết luận]

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
│  Tốn tài nguyên tính toán                                      │
│   - Training chậm với nhiều cây                                 │
│   - Tốn memory để lưu trữ tất cả cây                            │
│                                                                 │
│  Khó giải thích                                                │
│   - Không thể visualize như single decision tree                │
│   - "Black box" model                                           │
│                                                                 │
│  Prediction chậm                                               │
│   - Phải đi qua tất cả các cây                                  │
│   - Không phù hợp cho real-time applications                    │
│                                                                 │
│  Không hoạt động tốt với dữ liệu rất thưa (sparse)             │
│   - Text data, one-hot encoded data                             │
│                                                                 │
│  Có thể bias với imbalanced data                               │
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

## 13. Bài Tập Giải Tay (Hand-Calculated Exercises)

### 13.1. Bài tập 1: Xây dựng Random Forest từ đầu và dự đoán

** Đề bài:** 

Cho bộ dữ liệu nhỏ về dự đoán thời tiết có mưa hay không:

| ID | Nhiệt độ (°C) | Độ ẩm (%) | Gió (km/h) | Áp suất (hPa) | Mưa? |
|----|---------------|-----------|------------|---------------|------|
| 1  | 30            | 85        | 10         | 1008          | Yes  |
| 2  | 28            | 90        | 5          | 1005          | Yes  |
| 3  | 35            | 40        | 15         | 1015          | No   |
| 4  | 32            | 75        | 8          | 1010          | Yes  |
| 5  | 25            | 50        | 20         | 1018          | No   |
| 6  | 33            | 45        | 12         | 1012          | No   |

**Yêu cầu:** 
1. Xây dựng Random Forest với 3 cây (n_estimators=3)
2. Mỗi cây chọn 2 features (max_features=2)
3. Dự đoán cho sample mới: Nhiệt độ=29°C, Độ ẩm=80%, Gió=7km/h, Áp suất=1007hPa

---

** LỜI GIẢI CHI TIẾT:**

#### **BƯỚC 1: Bootstrap Sampling (Lấy mẫu có hoàn lại)**

Với n=6 samples, ta lấy ngẫu nhiên 6 lần **có hoàn lại** cho mỗi cây:

```
 Dữ liệu gốc: [1, 2, 3, 4, 5, 6]

 Tree 1 - Bootstrap sampling:
   Lần 1: random → chọn ID 2
   Lần 2: random → chọn ID 4
   Lần 3: random → chọn ID 2  (trùng lặp, được phép!)
   Lần 4: random → chọn ID 6
   Lần 5: random → chọn ID 1
   Lần 6: random → chọn ID 4  (trùng lặp, được phép!)
   
   → Bootstrap Sample 1: [2, 4, 2, 6, 1, 4] = {1, 2, 4, 6}
   → OOB (Out-of-Bag): {3, 5} (không được chọn)

 Tree 2 - Bootstrap sampling:
   → Bootstrap Sample 2: [1, 3, 5, 1, 3, 6] = {1, 3, 5, 6}
   → OOB: {2, 4}

 Tree 3 - Bootstrap sampling:
   → Bootstrap Sample 3: [2, 3, 4, 5, 2, 1] = {1, 2, 3, 4, 5}
   → OOB: {6}
```

#### **BƯỚC 2: Random Feature Selection (Chọn ngẫu nhiên features)**

Với 4 features và max_features=2, mỗi cây chọn ngẫu nhiên 2 features:

```
 Tất cả features: [Nhiệt độ, Độ ẩm, Gió, Áp suất]

 Tree 1 chọn ngẫu nhiên 2: [Độ ẩm, Áp suất]
 Tree 2 chọn ngẫu nhiên 2: [Nhiệt độ, Gió]  
 Tree 3 chọn ngẫu nhiên 2: [Độ ẩm, Gió]
```

#### **BƯỚC 3: Xây dựng Decision Tree cho từng Bootstrap Sample**

** TREE 1: Sử dụng [Độ ẩm, Áp suất] với dữ liệu {1, 2, 4, 6}**

```
Dữ liệu Tree 1:
| ID | Độ ẩm | Áp suất | Mưa? |
|----|-------|---------|------|
| 1  | 85    | 1008    | Yes  |
| 2  | 90    | 1005    | Yes  |
| 4  | 75    | 1010    | Yes  |
| 6  | 45    | 1012    | No   |

 Tính Gini Impurity ban đầu:
   - Yes: 3/4 = 0.75
   - No:  1/4 = 0.25
   - Gini = 1 - (0.75² + 0.25²) = 1 - (0.5625 + 0.0625) = 0.375

 Thử split theo Độ ẩm ≤ 60:
   - Left (≤60):  {6} → No=1/1  → Gini = 1 - 1² = 0
   - Right (>60): {1,2,4} → Yes=3/3 → Gini = 1 - 1² = 0
   - Weighted Gini = (1/4)*0 + (3/4)*0 = 0 
→ Cây quyết định Tree 1:
                    Độ ẩm ≤ 60?
                    /        \
                 Yes         No
                  ↓           ↓
              [No]        [Yes]
```

** TREE 2: Sử dụng [Nhiệt độ, Gió] với dữ liệu {1, 3, 5, 6}**

```
Dữ liệu Tree 2:
| ID | Nhiệt độ | Gió | Mưa? |
|----|----------|-----|------|
| 1  | 30       | 10  | Yes  |
| 3  | 35       | 15  | No   |
| 5  | 25       | 20  | No   |
| 6  | 33       | 12  | No   |

 Gini ban đầu:
   - Yes: 1/4, No: 3/4
   - Gini = 1 - (0.25² + 0.75²) = 0.375

 Thử split theo Nhiệt độ ≤ 31:
   - Left (≤31):  {1, 5} → Yes=1, No=1 → Gini = 0.5
   - Right (>31): {3, 6} → No=2/2 → Gini = 0
   - Weighted = (2/4)*0.5 + (2/4)*0 = 0.25

 Thử split theo Gió ≤ 11:
   - Left (≤11):  {1} → Yes=1/1 → Gini = 0
   - Right (>11): {3, 5, 6} → No=3/3 → Gini = 0
   - Weighted = (1/4)*0 + (3/4)*0 = 0  Best split!

→ Cây quyết định Tree 2:
                    Gió ≤ 11?
                    /       \
                 Yes        No
                  ↓          ↓
              [Yes]       [No]
```

** TREE 3: Sử dụng [Độ ẩm, Gió] với dữ liệu {1, 2, 3, 4, 5}**

```
Dữ liệu Tree 3:
| ID | Độ ẩm | Gió | Mưa? |
|----|-------|-----|------|
| 1  | 85    | 10  | Yes  |
| 2  | 90    | 5   | Yes  |
| 3  | 40    | 15  | No   |
| 4  | 75    | 8   | Yes  |
| 5  | 50    | 20  | No   |

 Gini ban đầu:
   - Yes: 3/5 = 0.6, No: 2/5 = 0.4
   - Gini = 1 - (0.36 + 0.16) = 0.48

 Thử split theo Độ ẩm ≤ 60:
   - Left (≤60):  {3, 5} → No=2/2 → Gini = 0
   - Right (>60): {1, 2, 4} → Yes=3/3 → Gini = 0
   - Weighted = 0  Perfect split!

→ Cây quyết định Tree 3:
                    Độ ẩm ≤ 60?
                    /        \
                 Yes         No
                  ↓           ↓
              [No]        [Yes]
```

#### **BƯỚC 4: Dự đoán cho sample mới**

**Sample mới:** Nhiệt độ=29°C, Độ ẩm=80%, Gió=7km/h, Áp suất=1007hPa

```
 Tree 1 (Độ ẩm, Áp suất):
   Độ ẩm = 80% > 60 → đi nhánh phải → Dự đoán: YES ✓

 Tree 2 (Nhiệt độ, Gió):
   Gió = 7 km/h ≤ 11 → đi nhánh trái → Dự đoán: YES ✓

 Tree 3 (Độ ẩm, Gió):
   Độ ẩm = 80% > 60 → đi nhánh phải → Dự đoán: YES ✓

┌─────────────────────────────────────────────┐
│           MAJORITY VOTING                   │
├─────────────────────────────────────────────┤
│  Tree 1: YES                                │
│  Tree 2: YES                                │
│  Tree 3: YES                                │
├─────────────────────────────────────────────┤
│  Tổng: YES = 3 votes, NO = 0 votes          │
│                                             │
│   KẾT QUẢ: YES (Có mưa)                   │
└─────────────────────────────────────────────┘
```

---

### 13.2. Bài tập 2: Tính Gini Impurity và chọn Best Split

** Đề bài:**

Cho node có 10 samples với phân bố: 6 "Yes" và 4 "No".
Có 2 cách split:
- **Split A:** Left (4 Yes, 1 No) | Right (2 Yes, 3 No)
- **Split B:** Left (5 Yes, 0 No) | Right (1 Yes, 4 No)

Tính Gini Impurity và chọn split tốt nhất.

---

** LỜI GIẢI CHI TIẾT:**

#### **BƯỚC 1: Tính Gini Impurity của node gốc**

```
Công thức: Gini = 1 - Σ(p_i)²

Node gốc: 6 Yes, 4 No (tổng 10 samples)
   p(Yes) = 6/10 = 0.6
   p(No)  = 4/10 = 0.4

Gini(root) = 1 - (0.6² + 0.4²)
           = 1 - (0.36 + 0.16)
           = 1 - 0.52
           = 0.48
```

#### **BƯỚC 2: Tính Gini cho Split A**

```
Split A: Left (4 Yes, 1 No) | Right (2 Yes, 3 No)

 Left node (5 samples):
   p(Yes) = 4/5 = 0.8
   p(No)  = 1/5 = 0.2
   Gini(Left) = 1 - (0.8² + 0.2²)
              = 1 - (0.64 + 0.04)
              = 1 - 0.68
              = 0.32

 Right node (5 samples):
   p(Yes) = 2/5 = 0.4
   p(No)  = 3/5 = 0.6
   Gini(Right) = 1 - (0.4² + 0.6²)
               = 1 - (0.16 + 0.36)
               = 1 - 0.52
               = 0.48

 Weighted Gini của Split A:
   Weighted_Gini(A) = (n_left/n_total) × Gini(Left) + (n_right/n_total) × Gini(Right)
                    = (5/10) × 0.32 + (5/10) × 0.48
                    = 0.5 × 0.32 + 0.5 × 0.48
                    = 0.16 + 0.24
                    = 0.40
```

#### **BƯỚC 3: Tính Gini cho Split B**

```
Split B: Left (5 Yes, 0 No) | Right (1 Yes, 4 No)

 Left node (5 samples):
   p(Yes) = 5/5 = 1.0
   p(No)  = 0/5 = 0.0
   Gini(Left) = 1 - (1.0² + 0.0²)
              = 1 - 1
              = 0 ←  node!

 Right node (5 samples):
   p(Yes) = 1/5 = 0.2
   p(No)  = 4/5 = 0.8
   Gini(Right) = 1 - (0.2² + 0.8²)
               = 1 - (0.04 + 0.64)
               = 1 - 0.68
               = 0.32

 Weighted Gini của Split B:
   Weighted_Gini(B) = (5/10) × 0 + (5/10) × 0.32
                    = 0 + 0.16
                    = 0.16
```

#### **BƯỚC 4: So sánh và chọn Best Split**

```
┌─────────────────────────────────────────────────────────┐
│                   SO SÁNH 2 SPLIT                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Gini ban đầu (root):        0.48                       │
│                                                         │
│  Split A - Weighted Gini:    0.40                       │
│  → Gini Gain = 0.48 - 0.40 = 0.08                       │
│                                                         │
│  Split B - Weighted Gini:    0.16                       │
│  → Gini Gain = 0.48 - 0.16 = 0.32                       │
│                                                         │
├─────────────────────────────────────────────────────────┤
│   CHỌN SPLIT B (Gini thấp hơn = tốt hơn)              │
│     Giảm Gini được: 0.32 (so với 0.08 của Split A)      │
└─────────────────────────────────────────────────────────┘
```

---

### 13.3. Bài tập 3: Tính Feature Importance chi tiết

** Đề bài:**

Random Forest có 3 cây với thông tin Gini decrease tại mỗi node:

**Tree 1:**
```
Node 1 (root): Split by F1, Gini decrease = 0.15, samples = 100
Node 2: Split by F2, Gini decrease = 0.08, samples = 60
Node 3: Split by F1, Gini decrease = 0.05, samples = 40
```

**Tree 2:**
```
Node 1 (root): Split by F2, Gini decrease = 0.12, samples = 100
Node 2: Split by F3, Gini decrease = 0.10, samples = 55
Node 3: Split by F1, Gini decrease = 0.06, samples = 45
```

**Tree 3:**
```
Node 1 (root): Split by F1, Gini decrease = 0.18, samples = 100
Node 2: Split by F3, Gini decrease = 0.07, samples = 50
Node 3: Split by F2, Gini decrease = 0.04, samples = 50
```

Tính Feature Importance cho F1, F2, F3.

---

** LỜI GIẢI CHI TIẾT:**

#### **BƯỚC 1: Tính Weighted Gini Decrease cho mỗi feature trong từng cây**

```
Công thức: Weighted_Decrease = (samples/total_samples) × Gini_decrease

 TREE 1 (total = 100 samples):
   F1: Node1 + Node3 = (100/100)×0.15 + (40/100)×0.05
                     = 0.15 + 0.02 = 0.17
   F2: Node2 = (60/100)×0.08 = 0.048
   F3: không có = 0

 TREE 2 (total = 100 samples):
   F1: Node3 = (45/100)×0.06 = 0.027
   F2: Node1 = (100/100)×0.12 = 0.12
   F3: Node2 = (55/100)×0.10 = 0.055

 TREE 3 (total = 100 samples):
   F1: Node1 = (100/100)×0.18 = 0.18
   F2: Node3 = (50/100)×0.04 = 0.02
   F3: Node2 = (50/100)×0.07 = 0.035
```

#### **BƯỚC 2: Tính tổng Importance cho mỗi feature qua tất cả cây**

```
F1_total = Tree1(F1) + Tree2(F1) + Tree3(F1)
         = 0.17 + 0.027 + 0.18
         = 0.377

F2_total = Tree1(F2) + Tree2(F2) + Tree3(F2)
         = 0.048 + 0.12 + 0.02
         = 0.188

F3_total = Tree1(F3) + Tree2(F3) + Tree3(F3)
         = 0 + 0.055 + 0.035
         = 0.090
```

#### **BƯỚC 3: Chuẩn hóa để tổng = 1 (100%)**

```
Tổng tất cả = 0.377 + 0.188 + 0.090 = 0.655

F1_importance = 0.377 / 0.655 = 0.5756 ≈ 57.56%
F2_importance = 0.188 / 0.655 = 0.2870 ≈ 28.70%
F3_importance = 0.090 / 0.655 = 0.1374 ≈ 13.74%

Kiểm tra: 57.56% + 28.70% + 13.74% = 100% ✓
```

#### **BƯỚC 4: Kết quả**

```
┌─────────────────────────────────────────────────────────┐
│              FEATURE IMPORTANCE RANKING                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   F1: 57.56%  ████████████████████████████░░░░░░░░░░  │
│   F2: 28.70%  ██████████████░░░░░░░░░░░░░░░░░░░░░░░░  │
│   F3: 13.74%  ███████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│                                                         │
├─────────────────────────────────────────────────────────┤
│   KẾT LUẬN:                                           │
│  • F1 quan trọng nhất (chiếm hơn nửa tổng importance)   │
│  • F2 quan trọng thứ 2                                  │
│  • F3 ít quan trọng nhất                                │
└─────────────────────────────────────────────────────────┘
```

---



### 13.4. Bài tập 5: Random Forest Regression - Dự đoán giá nhà

** Đề bài:**

Dự đoán giá nhà (tỷ VND) bằng Random Forest với 5 cây.
Cho sample mới, mỗi cây dự đoán:

| Tree | Prediction (tỷ VND) |
|------|---------------------|
| 1 | 2.5 |
| 2 | 2.8 |
| 3 | 2.3 |
| 4 | 2.9 |
| 5 | 2.5 |

Tính giá dự đoán cuối cùng và độ tin cậy.

---

** LỜI GIẢI CHI TIẾT:**

#### **BƯỚC 1: Tổng hợp predictions từ tất cả cây**

```
Predictions = [2.5, 2.8, 2.3, 2.9, 2.5]
Số cây n = 5
```

#### **BƯỚC 2: Tính trung bình (Final Prediction)**

```
Công thức: ŷ = (1/n) × Σ(predictions)

Final Prediction = (2.5 + 2.8 + 2.3 + 2.9 + 2.5) / 5
                 = 13.0 / 5
                 = 2.60 tỷ VND
```

#### **BƯỚC 3: Tính độ lệch chuẩn (để đánh giá độ tin cậy)**

```
Công thức: σ = √[(1/n) × Σ(y_i - ŷ)²]

Bước 3.1: Tính độ lệch từ trung bình
   Tree 1: 2.5 - 2.6 = -0.1  → (-0.1)² = 0.01
   Tree 2: 2.8 - 2.6 = +0.2  → (+0.2)² = 0.04
   Tree 3: 2.3 - 2.6 = -0.3  → (-0.3)² = 0.09
   Tree 4: 2.9 - 2.6 = +0.3  → (+0.3)² = 0.09
   Tree 5: 2.5 - 2.6 = -0.1  → (-0.1)² = 0.01

Bước 3.2: Tính variance
   Variance = (0.01 + 0.04 + 0.09 + 0.09 + 0.01) / 5
            = 0.24 / 5
            = 0.048

Bước 3.3: Tính standard deviation
   σ = √0.048 = 0.219 tỷ VND
```

#### **BƯỚC 4: Tính khoảng tin cậy 95%**

```
Confidence Interval (95%) ≈ ŷ ± 1.96 × σ

Lower bound = 2.60 - 1.96 × 0.219 = 2.60 - 0.43 = 2.17 tỷ
Upper bound = 2.60 + 1.96 × 0.219 = 2.60 + 0.43 = 3.03 tỷ
```

#### **BƯỚC 5: Kết quả**

```
┌─────────────────────────────────────────────────────────┐
│           KẾT QUẢ DỰ ĐOÁN GIÁ NHÀ                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   Giá dự đoán:    2.60 tỷ VND                         │
│   Độ lệch chuẩn:  0.22 tỷ VND                         │
│   Khoảng tin cậy: [2.17 - 3.03] tỷ VND (95%)          │
│                                                         │
│   Min prediction: 2.3 tỷ (Tree 3)                     │
│   Max prediction: 2.9 tỷ (Tree 4)                     │
│   Range:          0.6 tỷ                              │
│                                                         │
├─────────────────────────────────────────────────────────┤
│   ĐÁNH GIÁ ĐỘ TIN CẬY:                                │
│  • Variance thấp → các cây đồng thuận cao              │
│  • Khoảng tin cậy hẹp → dự đoán đáng tin cậy           │
└─────────────────────────────────────────────────────────┘
```

---

### 13.6. Bài tập 6: Tính số features tối ưu (max_features)

** Đề bài:**

Cho bộ dữ liệu với d = 20 features.
Tính số features nên chọn cho mỗi split trong các trường hợp:
1. Classification
2. Regression
3. max_features = 'log2'
4. max_features = 0.3 (30%)

---

** LỜI GIẢI CHI TIẾT:**

#### **Trường hợp 1: Classification (mặc định 'sqrt')**

```
Công thức: k = √d

k = √20 
  = 4.47

Làm tròn: k = 4 hoặc 5 features

 Scikit-learn làm tròn xuống: k = 4 features
```

#### **Trường hợp 2: Regression (mặc định 1.0 hoặc d/3)**

```
Theo sklearn mặc định: max_features = 1.0 (dùng tất cả)
Theo kinh nghiệm: k = d/3

k = 20/3 = 6.67

Làm tròn: k = 6 hoặc 7 features
```

#### **Trường hợp 3: max_features = 'log2'**

```
Công thức: k = log₂(d)

k = log₂(20)
  = ln(20) / ln(2)
  = 2.996 / 0.693
  = 4.32

Làm tròn xuống: k = 4 features
```

#### **Trường hợp 4: max_features = 0.3 (30%)**

```
Công thức: k = d × ratio

k = 20 × 0.3
  = 6 features
```

#### **Tổng hợp kết quả:**

```
┌─────────────────────────────────────────────────────────┐
│              BẢNG TỔNG HỢP max_features                 │
├──────────────────────┬──────────────┬───────────────────┤
│ Trường hợp           │ Công thức    │ Kết quả (d=20)    │
├──────────────────────┼──────────────┼───────────────────┤
│ Classification       │ √d           │ 4 features        │
│ Regression           │ d (or d/3)   │ 20 (or 7) feat.   │
│ 'log2'               │ log₂(d)      │ 4 features        │
│ 0.3 (30%)            │ d × 0.3      │ 6 features        │
│ 0.5 (50%)            │ d × 0.5      │ 10 features       │
│ 5 (fixed)            │ 5            │ 5 features        │
└──────────────────────┴──────────────┴───────────────────┘

 GHI NHỚ:
• max_features nhỏ → mỗi cây khác biệt nhiều → giảm variance
• max_features lớn → mỗi cây giống nhau → có thể overfitting
• Classification thường dùng 'sqrt'
• Regression thường dùng tất cả hoặc 1/3
```

---

## 14. Ứng Dụng Thực Tế Chi Tiết

### 14.1. Ứng dụng trong Dự Báo Thời Tiết (Weather Forecasting)

#### 14.1.1. Mô tả bài toán

```
┌─────────────────────────────────────────────────────────────────┐
│           RANDOM FOREST CHO DỰ BÁO THỜI TIẾT                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT FEATURES:                                                │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ • Nhiệt độ hiện tại (°C)                                   ││
│  │ • Độ ẩm (%)                                                 ││
│  │ • Áp suất khí quyển (hPa)                                   ││
│  │ • Tốc độ gió (km/h)                                         ││
│  │ • Hướng gió (độ)                                            ││
│  │ • Lượng mây (%)                                             ││
│  │ • Lượng mưa 24h trước (mm)                                  ││
│  │ • Nhiệt độ 24h trước (°C)                                   ││
│  │ • Mùa trong năm (Spring/Summer/Fall/Winter)                 ││
│  │ • Giờ trong ngày (0-23)                                     ││
│  │ • Vị trí địa lý (lat, lon)                                  ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  OUTPUT:                                                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Classification: Loại thời tiết (Sunny/Cloudy/Rainy/Stormy) ││
│  │ Regression: Nhiệt độ dự đoán, Lượng mưa dự đoán             ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 14.1.2. Code ví dụ thực tế

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error

# 1. Chuẩn bị dữ liệu thời tiết
weather_data = pd.DataFrame({
    'temperature': [28, 32, 25, 20, 35, 18, 30, 22, 27, 33],
    'humidity': [65, 45, 80, 90, 30, 95, 55, 85, 70, 40],
    'pressure': [1013, 1015, 1008, 1005, 1020, 1002, 1012, 1006, 1010, 1018],
    'wind_speed': [10, 5, 20, 25, 8, 30, 12, 18, 15, 6],
    'cloud_cover': [20, 10, 70, 90, 5, 100, 30, 80, 50, 15],
    'prev_rainfall': [0, 0, 5, 15, 0, 20, 0, 10, 2, 0],
    'weather_type': ['Sunny', 'Sunny', 'Cloudy', 'Rainy', 'Sunny', 
                     'Rainy', 'Sunny', 'Rainy', 'Cloudy', 'Sunny']
})

# 2. Encode categorical target
weather_mapping = {'Sunny': 0, 'Cloudy': 1, 'Rainy': 2}
weather_data['weather_encoded'] = weather_data['weather_type'].map(weather_mapping)

# 3. Split features và target
X = weather_data[['temperature', 'humidity', 'pressure', 'wind_speed', 
                   'cloud_cover', 'prev_rainfall']]
y = weather_data['weather_encoded']

# 4. Train Random Forest
rf_weather = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
rf_weather.fit(X, y)

# 5. Feature Importance cho thời tiết
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_weather.feature_importances_
}).sort_values('Importance', ascending=False)

print(" Feature Importance cho Dự báo Thời tiết:")
print(importance_df.to_string(index=False))

# 6. Dự đoán thời tiết mới
new_weather = pd.DataFrame({
    'temperature': [26],
    'humidity': [75],
    'pressure': [1007],
    'wind_speed': [15],
    'cloud_cover': [60],
    'prev_rainfall': [3]
})

prediction = rf_weather.predict(new_weather)
inverse_mapping = {0: 'Sunny', 1: 'Cloudy', 2: 'Rainy'}
print(f"\n Dự đoán thời tiết: {inverse_mapping[prediction[0]]}")

# 7. Xác suất dự đoán
proba = rf_weather.predict_proba(new_weather)
print(f" Xác suất: Sunny={proba[0][0]:.1%}, Cloudy={proba[0][1]:.1%}, Rainy={proba[0][2]:.1%}")
```

**Output:**
```
 Feature Importance cho Dự báo Thời tiết:
       Feature  Importance
      humidity       0.35
   cloud_cover       0.25
  prev_rainfall      0.20
   temperature       0.12
    wind_speed       0.05
      pressure       0.03

 Dự đoán thời tiết: Cloudy
 Xác suất: Sunny=20.0%, Cloudy=55.0%, Rainy=25.0%
```

---

### 14.2. Ứng dụng trong Y tế (Healthcare)

#### 14.2.1. Chẩn đoán bệnh tiểu đường

```python
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Dữ liệu bệnh tiểu đường
# Features: BMI, Blood Pressure, Glucose Level, Age, etc.

# Ví dụ thực tế
patient_data = pd.DataFrame({
    'pregnancies': [6, 1, 8, 1, 0, 5, 3, 10, 2, 8],
    'glucose': [148, 85, 183, 89, 137, 116, 78, 115, 197, 125],
    'blood_pressure': [72, 66, 64, 66, 40, 74, 50, 0, 70, 96],
    'skin_thickness': [35, 29, 0, 23, 35, 0, 32, 0, 45, 0],
    'insulin': [0, 0, 0, 94, 168, 0, 88, 0, 543, 0],
    'bmi': [33.6, 26.6, 23.3, 28.1, 43.1, 25.6, 31.0, 35.3, 30.5, 0],
    'diabetes_pedigree': [0.627, 0.351, 0.672, 0.167, 2.288, 0.201, 0.248, 0.134, 0.158, 0.232],
    'age': [50, 31, 32, 21, 33, 30, 26, 29, 53, 54],
    'outcome': [1, 0, 1, 0, 1, 0, 1, 0, 1, 1]  # 1=Tiểu đường, 0=Không
})

X = patient_data.drop('outcome', axis=1)
y = patient_data['outcome']

# Train model
rf_diabetes = RandomForestClassifier(n_estimators=100, random_state=42)
rf_diabetes.fit(X, y)

# Feature importance
print(" Yếu tố quan trọng trong chẩn đoán tiểu đường:")
for feature, importance in sorted(
    zip(X.columns, rf_diabetes.feature_importances_), 
    key=lambda x: x[1], 
    reverse=True
):
    print(f"   • {feature}: {importance:.1%}")
```

**Ứng dụng thực tế:**
- **Phát hiện sớm bệnh** dựa trên các chỉ số xét nghiệm
- **Xác định yếu tố nguy cơ** quan trọng nhất
- **Hỗ trợ bác sĩ** đưa ra quyết định

---

### 14.3. Ứng dụng trong Tài chính (Finance)

#### 14.3.1. Phát hiện gian lận thẻ tín dụng

```python
# Features cho fraud detection
credit_features = [
    'transaction_amount',      # Số tiền giao dịch
    'transaction_hour',        # Giờ giao dịch
    'merchant_category',       # Loại cửa hàng
    'distance_from_home',      # Khoảng cách từ nhà
    'distance_from_last_transaction',
    'ratio_to_median_purchase',
    'repeat_retailer',         # Mua lại cùng nơi
    'used_chip',               # Sử dụng chip
    'used_pin',                # Nhập PIN
    'online_order'             # Mua online
]

# Tại sao Random Forest phù hợp:
# 1. Xử lý tốt imbalanced data (fraud << normal)
# 2. Feature importance giúp hiểu pattern gian lận
# 3. Robust với noise trong dữ liệu
# 4. Không cần normalize features

rf_fraud = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    class_weight='balanced',  # Xử lý imbalanced
    random_state=42,
    n_jobs=-1
)
```

#### 14.3.2. Đánh giá tín dụng (Credit Scoring)

```
┌─────────────────────────────────────────────────────────────────┐
│              RANDOM FOREST CHO CREDIT SCORING                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Features đầu vào:                                              │
│  • Thu nhập hàng tháng                                          │
│  • Tổng nợ hiện tại                                             │
│  • Lịch sử thanh toán                                           │
│  • Số năm làm việc                                              │
│  • Độ tuổi                                                      │
│  • Tình trạng hôn nhân                                          │
│  • Số tài khoản ngân hàng                                       │
│  • Lịch sử vay trước đó                                         │
│                                                                 │
│  Output: Xác suất vỡ nợ (Default Probability)                   │
│                                                                 │
│  Ưu điểm:                                                       │
│  ✓ Giải thích được quyết định (Feature Importance)              │
│  ✓ Robust với outliers (thu nhập bất thường)                    │
│  ✓ Không cần xử lý missing values phức tạp                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### 14.4. Ứng dụng trong E-commerce

#### 14.4.1. Dự đoán khách hàng rời bỏ (Customer Churn)

```python
# E-commerce churn prediction
ecommerce_features = pd.DataFrame({
    'days_since_last_order': [5, 30, 90, 2, 180, 15],
    'total_orders': [50, 10, 3, 100, 1, 25],
    'avg_order_value': [500000, 200000, 1000000, 300000, 150000, 450000],
    'total_spent': [25000000, 2000000, 3000000, 30000000, 150000, 11250000],
    'customer_tenure_days': [365, 180, 90, 730, 60, 400],
    'support_tickets': [2, 5, 8, 1, 3, 0],
    'returned_items': [1, 2, 1, 0, 0, 0],
    'churned': [0, 0, 1, 0, 1, 0]  # 1=Đã rời bỏ
})

# Ứng dụng:
# - Xác định khách hàng có nguy cơ rời bỏ
# - Đưa ra chiến lược giữ chân khách hàng
# - Tối ưu chi phí marketing
```

#### 14.4.2. Gợi ý sản phẩm (Product Recommendation)

```python
# Collaborative Filtering với Random Forest
# Dự đoán rating sản phẩm

product_features = [
    'user_avg_rating',         # Rating trung bình của user
    'product_avg_rating',      # Rating trung bình của sản phẩm
    'user_total_purchases',    # Tổng đơn hàng của user
    'product_category_match',  # Phù hợp với category yêu thích
    'price_range_match',       # Phù hợp với mức giá hay mua
    'time_since_last_purchase' # Thời gian từ lần mua cuối
]

# Random Forest Regression để dự đoán rating
rf_recommender = RandomForestRegressor(n_estimators=100)
```

---

### 14.5. Ứng dụng trong Sản xuất (Manufacturing)

#### 14.5.1. Bảo trì dự đoán (Predictive Maintenance)

```
┌─────────────────────────────────────────────────────────────────┐
│            PREDICTIVE MAINTENANCE VỚI RANDOM FOREST             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Bài toán: Dự đoán khi nào máy móc sẽ hỏng                      │
│                                                                 │
│  Features từ cảm biến IoT:                                      │
│  • Nhiệt độ động cơ                                             │
│  • Độ rung                                                      │
│  • Áp suất dầu                                                  │
│  • Cường độ dòng điện                                           │
│  • Số giờ hoạt động                                             │
│  • Số lần bảo trì trước đó                                      │
│  • Tuổi thiết bị                                                │
│                                                                 │
│  Target:                                                        │
│  • Classification: Sẽ hỏng trong 7 ngày tới? (Yes/No)           │
│  • Regression: Còn bao nhiêu giờ trước khi hỏng?                │
│                                                                 │
│  Lợi ích:                                                       │
│  ✓ Giảm downtime không mong muốn                                │
│  ✓ Tối ưu lịch bảo trì                                          │
│  ✓ Tiết kiệm chi phí sửa chữa                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 14.5.2. Kiểm soát chất lượng (Quality Control)

```python
# Dự đoán sản phẩm lỗi
quality_features = [
    'material_batch_id',
    'temperature_during_production',
    'humidity_during_production',
    'machine_id',
    'operator_id',
    'time_of_day',
    'production_speed',
    'raw_material_quality_score'
]

# Target: defect_type (0=OK, 1=Minor, 2=Major, 3=Critical)

rf_quality = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    random_state=42
)

# Feature importance giúp:
# - Xác định nguyên nhân gây lỗi
# - Cải tiến quy trình sản xuất
# - Đào tạo nhân viên tập trung vào yếu tố quan trọng
```

---

### 14.6. Ứng dụng trong Nông nghiệp (Agriculture)

```python
# Dự đoán năng suất cây trồng
agriculture_data = {
    'features': [
        'soil_ph',              # Độ pH đất
        'nitrogen_level',       # Hàm lượng Nitơ
        'phosphorus_level',     # Hàm lượng Phốt pho
        'potassium_level',      # Hàm lượng Kali
        'temperature_avg',      # Nhiệt độ trung bình
        'humidity_avg',         # Độ ẩm trung bình
        'rainfall_total',       # Tổng lượng mưa
        'sunlight_hours',       # Số giờ nắng
        'crop_type',           # Loại cây trồng
        'planting_date'        # Ngày gieo trồng
    ],
    'target': 'yield_per_hectare'  # Năng suất (tấn/ha)
}

# Ứng dụng:
# 1. Dự đoán năng suất mùa vụ
# 2. Tối ưu hóa phân bón (dựa trên feature importance)
# 3. Lựa chọn thời điểm gieo trồng tốt nhất
# 4. Cảnh báo sớm về rủi ro thất bại mùa vụ
```

---

### 14.7. Bảng tổng hợp ứng dụng

| Lĩnh vực | Bài toán | Loại | Key Features |
|----------|----------|------|--------------|
| **Thời tiết** | Dự báo thời tiết | Classification/Regression | Nhiệt độ, độ ẩm, áp suất |
| **Y tế** | Chẩn đoán bệnh | Classification | Triệu chứng, xét nghiệm |
| **Tài chính** | Phát hiện gian lận | Classification | Giao dịch, hành vi |
| **E-commerce** | Customer Churn | Classification | Hành vi mua hàng |
| **Sản xuất** | Predictive Maintenance | Regression | Dữ liệu cảm biến |
| **Nông nghiệp** | Dự đoán năng suất | Regression | Thời tiết, đất |
| **HR** | Employee Attrition | Classification | Hiệu suất, đánh giá |
| **Marketing** | Customer Segmentation | Classification | Demographics, behavior |
| **Bất động sản** | Định giá nhà | Regression | Vị trí, diện tích |
| **Giao thông** | Dự đoán tắc đường | Classification | Thời gian, vị trí |

---

## 15. Tài Liệu Tham Khảo

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

-  Độ chính xác cao
-  Chống overfitting tốt
-  Xử lý missing data
-  Không cần normalize
-  Cung cấp feature importance

Random Forest là lựa chọn an toàn để bắt đầu với bất kỳ bài toán classification hoặc regression nào. Đặc biệt trong bài toán **dự báo thời tiết**, Random Forest có thể được sử dụng để:

- Phân loại điều kiện thời tiết (sunny, rainy, cloudy)
- Dự đoán nhiệt độ, lượng mưa
- Xác định features quan trọng nhất ảnh hưởng đến thời tiết

---

