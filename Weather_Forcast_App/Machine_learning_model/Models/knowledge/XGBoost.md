# Thuật Toán XGBoost Trong Dự Báo Thời Tiết

## 🎯 Tổng Quan Thuật Toán

XGBoost (Extreme Gradient Boosting) là thuật toán machine learning mạnh mẽ sử dụng phương pháp **Gradient Boosting** để xây dựng mô hình dự báo. Thuật toán này đặc biệt hiệu quả cho các bài toán regression như dự báo thời tiết.

## 🔄 Cách Thuật Toán Hoạt Động

### 1. **Khởi Tạo Model Cơ Sở**
```
F₀(x) = argmin_γ ∑ᵢ L(yᵢ, γ)
```
- Bắt đầu với một hàm dự đoán đơn giản (thường là giá trị trung bình)
- Đây là "base learner" đầu tiên

### 2. **Tính Gradient (Đạo Hàm)**
```
gᵢ = ∂L(yᵢ, F(xᵢ))/∂F(xᵢ)
hᵢ = ∂²L(yᵢ, F(xᵢ))/∂F²(xᵢ)
```
- Tính first-order gradient (gᵢ) và second-order gradient (hᵢ)
- Gradient cho biết hướng cần điều chỉnh để giảm lỗi

### 3. **Xây Dựng Decision Tree**
```
Gain = ½[∑(g_L²/(h_L+λ)) + ∑(g_R²/(h_R+λ)) - ∑(g²/(h+λ))] - γ
```
- Chia dữ liệu thành các node dựa trên gain function
- Sử dụng second-order derivatives để tối ưu
- Regularization terms λ và γ tránh overfitting

### 4. **Cập Nhật Model**
```
Fₘ(x) = Fₘ₋₁(x) + η * fₘ(x)
```
- Thêm tree mới vào model với learning rate η
- η thường = 0.1 để tránh overfitting

### 5. **Lặp Lại Cho Đến Convergence**
- Lặp lại bước 2-4 cho đến khi đạt số trees tối đa
- Hoặc dừng sớm nếu validation error không cải thiện

## 🚀 Quy Trình Thực Thi Trong Code

### **Bước 1: Chuẩn Bị Dữ Liệu**
```python
def prepare_data(self, data_path, target_column, test_size=0.2):
    # 1. Đọc dữ liệu thời tiết từ CSV
    df = pd.read_csv(data_path)

    # 2. Xử lý missing values
    df = df.dropna()

    # 3. One-hot encoding cho categorical features
    X = pd.get_dummies(X, drop_first=True)

    # 4. Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
```

### **Bước 2: Khởi Tạo Model**
```python
def __init__(self, config=None):
    # Tham số mặc định cho thời tiết
    self.params = {
        'objective': 'reg:squarederror',  # MSE loss
        'eval_metric': 'rmse',           # Root Mean Square Error
        'max_depth': 6,                  # Độ sâu cây
        'learning_rate': 0.1,           # Tốc độ học
        'n_estimators': 100,            # Số lượng cây
        'subsample': 0.8,              # Bootstrap sampling
        'colsample_bytree': 0.8,       # Feature sampling
    }
```

### **Bước 3: Training Process**
```python
def train(self, X_train, y_train):
    # 1. Scale features (Standardization)
    X_train_scaled = self.scaler.fit_transform(X_train)

    # 2. Tạo DMatrix (XGBoost's optimized data structure)
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train)

    # 3. Train với early stopping
    self.model = xgb.train(
        self.params,
        dtrain,
        num_boost_round=100,
        early_stopping_rounds=10
    )
```

### **Bước 4: Prediction Process**
```python
def predict(self, X):
    # 1. Scale input features
    X_scaled = self.scaler.transform(X)

    # 2. Tạo DMatrix cho prediction
    dtest = xgb.DMatrix(X_scaled)

    # 3. Dự đoán
    predictions = self.model.predict(dtest)

    return predictions
```

## 🎯 Hướng Giải Quyết Bài Toán Dự Báo Thời Tiết

### **Bài Toán**
- **Input**: Dữ liệu lịch sử thời tiết (nhiệt độ, độ ẩm, áp suất, gió...)
- **Output**: Dự báo giá trị thời tiết trong tương lai
- **Mục tiêu**: Tối thiểu hóa sai số dự báo

### **Chiến Lược Giải Quyết**

#### **1. Feature Engineering**
```
Temperature(t) = f(Temperature(t-1), Humidity(t-1), Pressure(t-1), Wind(t-1), ...)
```
- Sử dụng dữ liệu quá khứ để dự báo tương lai
- Tạo lag features (t-1, t-2, t-3...)
- Seasonal decomposition

#### **2. Model Selection**
- XGBoost phù hợp vì:
  - Xử lý được non-linear relationships
  - Robust với outliers
  - Feature importance built-in
  - Handle missing values

#### **3. Loss Function**
```
L(y, ŷ) = (y - ŷ)²  # MSE cho regression
```
- Penalize large errors heavily
- Differentiable for gradient descent

#### **4. Optimization**
```
θ* = argmin_θ ∑ᵢ L(yᵢ, F(xᵢ; θ))
```
- Sử dụng gradient descent để tối ưu
- Regularization để tránh overfitting

## 📊 Ví Dụ Minh Họa Quy Trình

### **Dữ Liệu Thời Tiết**
```
Date        | Temp | Humidity | Pressure | Wind | Temp_next_day
2024-01-01  | 25.5 | 65       | 1013    | 5.2  | 26.8
2024-01-02  | 26.8 | 70       | 1010    | 4.8  | 24.2
2024-01-03  | 24.2 | 75       | 1008    | 6.1  | 27.1
```

### **Quy Trình Training**
```
1. F₀(x) = 25.5 (mean temperature)
2. Tính residuals: rᵢ = yᵢ - F₀(xᵢ)
3. Xây tree đầu tiên fit residuals
4. F₁(x) = F₀(x) + η * Tree₁(x)
5. Lặp lại với residuals mới
6. F_final(x) = F₀(x) + η * (Tree₁ + Tree₂ + ... + Treeₙ)
```

### **Prediction**
```
Input: [Temp=25.5, Humidity=65, Pressure=1013, Wind=5.2]
Output: Temp_next_day = 26.8°C
```

## 🔧 Tham Số Quan Trọng Trong Dự Báo Thời Tiết

| Tham số | Ý nghĩa | Giá trị đề xuất |
|---------|---------|-----------------|
| `max_depth` | Độ sâu cây | 4-8 (tránh overfitting) |
| `learning_rate` | Tốc độ học | 0.05-0.1 |
| `n_estimators` | Số cây | 100-500 |
| `subsample` | Tỷ lệ mẫu | 0.8 (80% dữ liệu) |
| `colsample_bytree` | Tỷ lệ features | 0.8 |

## 📈 Đánh Giá Hiệu Suất

### **Metrics Chính**
- **RMSE**: Sai số trung bình (đơn vị °C cho nhiệt độ)
- **MAE**: Sai số tuyệt đối trung bình
- **R²**: Hệ số xác định (0-1)

### **Trong Dự Báo Thời Tiết**
```
RMSE < 2°C: Tốt cho nhiệt độ
MAE < 1.5°C: Rất tốt
R² > 0.85: Model tốt
```

## 🎨 Visualization & Interpretability

### **Feature Importance**
```python
importance = model.get_feature_importance()
# Temperature(t-1): 35%
# Humidity(t-1): 25%
# Pressure(t-1): 20%
# Wind(t-1): 15%
# Other: 5%
```

### **Partial Dependence Plots**
- Hiểu ảnh hưởng của từng feature
- Visualize non-linear relationships

## 🚀 Tối Ưu Hóa Cho Thời Tiết

### **1. Time Series Features**
- Lag features: Temp(t-1), Temp(t-2), Temp(t-3)
- Rolling statistics: Mean 7 days, Std 7 days
- Seasonal features: Month, Day of week

### **2. Domain Knowledge**
- Weather patterns: Monsoon, El Niño
- Geographical factors: Latitude, Longitude
- Historical trends: Climate change

### **3. Hyperparameter Tuning**
```python
# Grid Search cho thời tiết
param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300]
}
```

## 🔄 So Sánh Với Các Thuật Toán Khác

| Thuật Toán | Ưu điểm | Nhược điểm | Phù hợp thời tiết |
|------------|---------|------------|-------------------|
| **XGBoost** | Chính xác cao, Robust | Chậm training | ✅ Rất tốt |
| **Random Forest** | Nhanh, Ít overfit | Không tối ưu | ⚠️ Trung bình |
| **Linear Regression** | Đơn giản, Nhanh | Non-linear | ❌ Kém |
| **LSTM** | Sequential data | Cần nhiều data | ✅ Tốt |

## 🎯 Kết Luận

XGBoost giải quyết bài toán dự báo thời tiết bằng cách:

1. **Ensemble Learning**: Kết hợp nhiều weak learners
2. **Gradient Boosting**: Tối ưu từng bước với gradient
3. **Regularization**: Tránh overfitting
4. **Scalability**: Xử lý big data hiệu quả

**Kết quả**: Model có thể dự báo nhiệt độ với độ chính xác cao, giúp cải thiện chất lượng dự báo thời tiết cho người dùng.

---

*Thuật toán XGBoost đã được chứng minh hiệu quả trong nhiều ứng dụng thực tế, đặc biệt là dự báo thời tiết với độ chính xác vượt trội.*