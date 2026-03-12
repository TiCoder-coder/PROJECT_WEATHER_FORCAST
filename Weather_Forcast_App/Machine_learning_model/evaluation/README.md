# 📊 evaluation/ — Model Evaluation & Metrics Module

## 📁 Overview

Module `evaluation/` chứa **toàn bộ logic đánh giá (evaluate) Machine Learning models** cho hệ thống dự báo thời tiết. Nó cung cấp:
- **Metrics functions**: Các hàm tính chỉ số đánh giá (MAE, MSE, RMSE, R², MAPE)
- **Reporting utilities**: Xuất báo cáo so sánh models, biểu đồ, CSV/JSON reports
- **Visualizations**: Vẽ biểu đồ actual vs predicted, residuals, feature importance

Module này **cực kỳ quan trọng** để:
- Đánh giá chất lượng model sau khi train
- So sánh nhiều models để chọn model tốt nhất
- Tạo báo cáo nghiên cứu khoa học (research papers, thesis)
- Xuất biểu đồ để insert vào báo cáo cuối kỳ


## 📂 Directory Structure

```
evaluation/
├── __init__.py                    # Package initializer
├── metrics.py                     # 📐 Metric calculations (MAE, RMSE, R2, MAPE)
└── report.py                      # 📄 Report generation (CSV, JSON, charts)
```


## 🎯 Purpose

### ❓ Tại sao cần folder này?

| Vấn đề | Giải pháp |
|--------|-----------|
| **Làm sao biết model tốt hay không?** | Tính MAE, RMSE, R² để định lượng sai số |
| **So sánh XGBoost vs LightGBM?** | Dùng `report.py` để tạo comparison table |
| **Cần biểu đồ cho báo cáo nghiên cứu** | Xuất actual vs predicted charts |
| **Phải export metrics ra file** | Lưu CSV/JSON với `ModelEvaluationResult` |

### ✅ Lợi ích

- **Tách biệt logic evaluation**: Không trộn lẫn tính metric với training logic
- **Reusable**: Metrics có thể dùng cho bất kỳ model nào (XGBoost, LightGBM, CatBoost)
- **Chuẩn hóa**: Mọi model đều đánh giá theo cùng một bộ metrics
- **Automatic reporting**: Export reports tự động sau mỗi lần train


## 📄 Files Explained

### 1. `metrics.py` — Metric Calculations

**Mục đích**: Định nghĩa các hàm tính chỉ số đánh giá cho regression models.

#### 🔢 Regression Metrics

```python
from Weather_Forcast_App.Machine_learning_model.evaluation.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    calculate_all_metrics
)
```

**Các metrics có sẵn**:

| Metric | Công thức | Ý nghĩa | Đơn vị | Càng nhỏ càng tốt? |
|--------|-----------|---------|--------|---------------------|
| **MAE** | `(1/n) * Σ|y_true - y_pred|` | Sai số tuyệt đối trung bình | Giống target (mm, °C) | ✅ Có |
| **MSE** | `(1/n) * Σ(y_true - y_pred)²` | Sai số bình phương trung bình | Bình phương target | ✅ Có |
| **RMSE** | `√MSE` | Căn bậc hai của MSE | Giống target | ✅ Có |
| **R²** | `1 - (SS_res / SS_tot)` | Hệ số xác định (explained variance) | Không có đơn vị (0-1) | ❌ Càng lớn càng tốt |
| **MAPE** | `(100/n) * Σ|y_true - y_pred| / |y_true|` | Phần trăm sai số | % | ✅ Có |

#### 📊 Data Classes

```python
from dataclasses import dataclass

@dataclass
class MetricResult:
    """Kết quả của một metric đơn lẻ"""
    name: str              # Tên metric (VD: "MAE")
    value: float           # Giá trị (VD: 2.345)
    description: str       # Mô tả (VD: "Mean Absolute Error")
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "value": round(self.value, 6),
            "description": self.description
        }

@dataclass  
class EvaluationReport:
    """Báo cáo tổng hợp tất cả metrics"""
    metrics: Dict[str, float]   # {"mae": 2.34, "rmse": 3.45, ...}
    target_column: str          # "Precipitation_mm"
    n_samples: int              # Số lượng samples đánh giá
```

#### 🔧 Usage Examples

```python
import numpy as np
from Weather_Forcast_App.Machine_learning_model.evaluation.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    calculate_all_metrics
)

# Giả sử có predictions và ground truth
y_true = np.array([10.5, 12.3, 8.7, 15.2, 9.8])
y_pred = np.array([10.1, 12.8, 8.5, 14.9, 10.2])

# Tính từng metric riêng lẻ
mae = mean_absolute_error(y_true, y_pred)
rmse = root_mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"MAE: {mae:.3f} mm")    # MAE: 0.400 mm
print(f"RMSE: {rmse:.3f} mm")  # RMSE: 0.458 mm
print(f"R²: {r2:.3f}")         # R²: 0.956

# Hoặc tính tất cả metrics cùng lúc
all_metrics = calculate_all_metrics(y_true, y_pred)
print(all_metrics)
# {
#     "mae": 0.4,
#     "mse": 0.21,
#     "rmse": 0.458,
#     "r2": 0.956,
#     "mape": 3.85
# }
```

**Error handling**:

```python
# Lỗi khi độ dài không khớp
y_true = np.array([1, 2, 3])
y_pred = np.array([1, 2])  # Thiếu 1 phần tử
mae = mean_absolute_error(y_true, y_pred)
# ValueError: Độ dài không khớp: y_true=3, y_pred=2
```


### 2. `report.py` — Report Generation

**Mục đích**: Tạo báo cáo đánh giá dạng CSV, JSON, Markdown, HTML. Vẽ biểu đồ so sánh models.

#### 📦 Data Classes

```python
from dataclasses import dataclass

@dataclass
class ModelEvaluationResult:
    """Kết quả đánh giá của một model."""
    model_name: str                          # "XGBoost_v1"
    metrics: Dict[str, float]                # {"mae": 2.34, "rmse": 3.21}
    y_true: Optional[np.ndarray] = None      # Ground truth values
    y_pred: Optional[np.ndarray] = None      # Predictions
    feature_importance: Optional[Dict[str, float]] = None
    training_time: float = 0.0               # Training duration (seconds)
    n_samples: int = 0                       # Number of samples
    n_features: int = 0                      # Number of features
    hyperparameters: Dict[str, Any] = {}     # Model config
    notes: str = ""                          # Additional notes

@dataclass
class ComparisonReport:
    """Báo cáo so sánh nhiều models."""
    title: str                               # "Model Comparison - Jan 2024"
    description: str                         # "Comparing XGBoost vs LightGBM"
    dataset_name: str                        # "weather_data_2023.csv"
    target_column: str                       # "Precipitation_mm"
    created_at: datetime                     # Timestamp
    models: List[ModelEvaluationResult]      # Danh sách models
    best_model: Optional[str] = None         # Tên model tốt nhất
```

#### 🔧 Usage: Export Comparison Report

```python
from Weather_Forcast_App.Machine_learning_model.evaluation.report import (
    ModelEvaluationResult,
    ComparisonReport,
    save_comparison_report,
    plot_comparison_chart
)

# Tạo kết quả đánh giá cho 2 models
xgboost_result = ModelEvaluationResult(
    model_name="XGBoost",
    metrics={"mae": 2.34, "rmse": 3.21, "r2": 0.87},
    training_time=45.2,
    n_samples=10000,
    n_features=25,
    hyperparameters={"learning_rate": 0.1, "max_depth": 5}
)

lightgbm_result = ModelEvaluationResult(
    model_name="LightGBM",
    metrics={"mae": 2.15, "rmse": 3.05, "r2": 0.89},
    training_time=32.1,
    n_samples=10000,
    n_features=25,
    hyperparameters={"learning_rate": 0.1, "num_leaves": 31}
)

# Tạo comparison report
report = ComparisonReport(
    title="Model Comparison - Weather Forecast",
    description="XGBoost vs LightGBM on 2023 data",
    dataset_name="weather_2023.csv",
    target_column="Precipitation_mm",
    models=[xgboost_result, lightgbm_result],
    best_model="LightGBM"  # LightGBM có MAE thấp hơn
)

# Export to JSON
save_comparison_report(
    report, 
    output_path="output/model_comparison.json",
    format="json"
)

# Export to CSV
save_comparison_report(
    report, 
    output_path="output/model_comparison.csv",
    format="csv"
)

# Vẽ biểu đồ so sánh
plot_comparison_chart(
    report,
    metric="mae",
    output_path="output/mae_comparison.png"
)
```

**File JSON output**:

```json
{
  "title": "Model Comparison - Weather Forecast",
  "description": "XGBoost vs LightGBM on 2023 data",
  "dataset_name": "weather_2023.csv",
  "target_column": "Precipitation_mm",
  "created_at": "2024-01-15T10:30:00",
  "best_model": "LightGBM",
  "models": [
    {
      "model_name": "XGBoost",
      "metrics": {
        "mae": 2.34,
        "rmse": 3.21,
        "r2": 0.87
      },
      "training_time": 45.2,
      "n_samples": 10000,
      "n_features": 25,
      "hyperparameters": {
        "learning_rate": 0.1,
        "max_depth": 5
      }
    },
    {
      "model_name": "LightGBM",
      "metrics": {
        "mae": 2.15,
        "rmse": 3.05,
        "r2": 0.89
      },
      "training_time": 32.1,
      "n_samples": 10000,
      "n_features": 25,
      "hyperparameters": {
        "learning_rate": 0.1,
        "num_leaves": 31
      }
    }
  ]
}
```

#### 📊 Visualization Functions

**Có sẵn trong `report.py` (require matplotlib, seaborn)**:

| Function | Mục đích | Output |
|----------|----------|--------|
| `plot_actual_vs_predicted()` | Vẽ scatter plot: thực tế vs dự đoán | PNG/SVG chart |
| `plot_residuals()` | Vẽ residual plot (sai số phần dư) | PNG/SVG chart |
| `plot_feature_importance()` | Vẽ bar chart feature importance | PNG/SVG chart |
| `plot_comparison_chart()` | So sánh nhiều models theo metric | PNG/SVG chart |

```python
from Weather_Forcast_App.Machine_learning_model.evaluation.report import (
    plot_actual_vs_predicted,
    plot_residuals,
    plot_feature_importance
)

# Vẽ actual vs predicted
plot_actual_vs_predicted(
    y_true=y_test,
    y_pred=y_pred,
    title="Weather Prediction - Test Set",
    output_path="output/actual_vs_predicted.png"
)

# Vẽ residuals
plot_residuals(
    y_true=y_test,
    y_pred=y_pred,
    output_path="output/residuals.png"
)

# Vẽ feature importance (nếu model hỗ trợ)
plot_feature_importance(
    feature_importance=model.feature_importances_,
    feature_names=feature_columns,
    top_n=20,
    output_path="output/feature_importance.png"
)
```

**Chart outputs**:
- **Actual vs Predicted**: Scatter plot với diagonal line (y=x). Points gần line = predictions tốt.
- **Residuals**: Histogram hoặc scatter plot của sai số. Distribution tập trung quanh 0 = tốt.
- **Feature Importance**: Bar chart showing top features. Giúp hiểu model dựa vào features nào.


## 🔧 How to Use

### 1️⃣ Evaluate Single Model

```python
from Weather_Forcast_App.Machine_learning_model.evaluation.metrics import calculate_all_metrics
from Weather_Forcast_App.Machine_learning_model.evaluation.report import ModelEvaluationResult

# Train model
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Calculate metrics
metrics = calculate_all_metrics(y_test, y_pred)

# Create evaluation result
result = ModelEvaluationResult(
    model_name="XGBoost_v1",
    metrics=metrics,
    y_true=y_test,
    y_pred=y_pred,
    training_time=training_duration,
    n_samples=len(X_test),
    n_features=X_test.shape[1],
    hyperparameters=model.get_params()
)

# Save to JSON
import json
with open("output/xgboost_evaluation.json", "w") as f:
    json.dump(result.to_dict(), f, indent=2)
```


### 2️⃣ Compare Multiple Models

```python
from Weather_Forcast_App.Machine_learning_model.evaluation.report import (
    ModelEvaluationResult,
    ComparisonReport,
    save_comparison_report,
    plot_comparison_chart
)

# Evaluate multiple models
models_to_compare = ["XGBoost", "LightGBM", "CatBoost"]
results = []

for model_name in models_to_compare:
    # Train model
    model = get_model(model_name)
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    metrics = calculate_all_metrics(y_test, y_pred)
    
    # Store result
    results.append(ModelEvaluationResult(
        model_name=model_name,
        metrics=metrics,
        training_time=training_time,
        n_samples=len(X_test),
        n_features=X_test.shape[1]
    ))

# Create comparison report
report = ComparisonReport(
    title="Model Selection for Weather Forecast",
    description="Comparing 3 gradient boosting models",
    dataset_name="weather_data_2023.csv",
    target_column="Precipitation_mm",
    models=results,
    best_model=min(results, key=lambda x: x.metrics["mae"]).model_name
)

# Export
save_comparison_report(report, "output/comparison.json", format="json")
save_comparison_report(report, "output/comparison.csv", format="csv")

# Visualize
plot_comparison_chart(report, metric="mae", output_path="output/mae_chart.png")
plot_comparison_chart(report, metric="rmse", output_path="output/rmse_chart.png")
```


### 3️⃣ Generate Research Report

```python
from Weather_Forcast_App.Machine_learning_model.evaluation.report import generate_markdown_report

# Generate full Markdown report with charts
generate_markdown_report(
    report=comparison_report,
    output_path="output/research_report.md",
    include_charts=True,
    charts_dir="output/charts/"
)
```

**Markdown output** (ready for thesis/papers):

```markdown
# Model Comparison - Weather Forecast

**Dataset**: weather_data_2023.csv  
**Target**: Precipitation_mm  
**Date**: 2024-01-15

## Results Summary

| Model | MAE | RMSE | R² | Training Time (s) |
|-------|-----|------|----|--------------------|
| XGBoost | 2.34 | 3.21 | 0.87 | 45.2 |
| LightGBM | **2.15** | **3.05** | **0.89** | 32.1 |
| CatBoost | 2.42 | 3.35 | 0.85 | 52.7 |

**Best Model**: LightGBM (lowest MAE)

## Charts

![Actual vs Predicted](charts/actual_vs_predicted.png)
![Feature Importance](charts/feature_importance.png)
```


## 📊 Metrics Interpretation Guide

### 📐 MAE (Mean Absolute Error)

**Công thức**: `MAE = (1/n) * Σ|y_true - y_pred|`

**Ý nghĩa**:
- Sai số tuyệt đối trung bình
- Đơn vị giống với target (mm mưa, °C nhiệt độ)
- **Easy to interpret**: "Model sai trung bình 2.5mm mưa"

**Giải thích kết quả**:
- MAE = 0: Perfect predictions (không bao giờ đạt được trong thực tế)
- MAE = 2.5mm: Dự đoán sai trung bình 2.5mm so với thực tế
- MAE càng nhỏ càng tốt

**Khi nào dùng**:
- ✅ Khi cần số liệu dễ hiểu cho non-technical audience
- ✅ Khi outliers không quan trọng
- ❌ Khi cần phạt nặng outliers (dùng RMSE thay thế)


### 📐 RMSE (Root Mean Squared Error)

**Công thức**: `RMSE = √[(1/n) * Σ(y_true - y_pred)²]`

**Ý nghĩa**:
- Căn bậc hai của sai số bình phương trung bình
- Đơn vị giống target (như MAE)
- **Phạt nặng outliers** vì bình phương sai số trước khi tính trung bình

**Giải thích kết quả**:
- RMSE luôn ≥ MAE (do bình phương)
- RMSE = 3.2mm: Độ lệch chuẩn của sai số ~3.2mm
- RMSE càng nhỏ càng tốt

**Khi nào dùng**:
- ✅ Khi outliers rất quan trọng (VD: forecast lượng mưa lớn)
- ✅ Khi cần metric phản ánh variance cao
- ❌ Khi outliers là noise cần bỏ qua (dùng MAE)

**So sánh MAE vs RMSE**:

| Scenario | MAE | RMSE | Reason |
|----------|-----|------|--------|
| Predictions đều đặn | 2.5 | 2.8 | RMSE gần MAE → ít outliers |
| Có nhiều outliers | 2.5 | 5.3 | RMSE >> MAE → nhiều outliers |


### 📐 R² (R-squared / Coefficient of Determination)

**Công thức**: `R² = 1 - (SS_res / SS_tot)`

Trong đó:
- `SS_res` = Σ(y_true - y_pred)² (residual sum of squares)
- `SS_tot` = Σ(y_true - mean(y_true))² (total sum of squares)

**Ý nghĩa**:
- **Explained variance**: R² = 0.89 nghĩa là model giải thích được 89% variance trong data
- Không có đơn vị (dimensionless)
- Range: (-∞, 1], nhưng thường trong [0, 1]

**Giải thích kết quả**:
- R² = 1.0: Perfect fit
- R² = 0.89: Model giải thích 89% variance, còn 11% là noise hoặc missing features
- R² = 0.0: Model không tốt hơn baseline (dự đoán = mean)
- R² < 0: Model tệ hơn baseline (rất hiếm, xảy ra khi model quá sai)

**Khi nào dùng**:
- ✅ Khi cần biết "model giải thích bao nhiêu % variance"
- ✅ Trong research papers (R² là standard metric)
- ❌ Khi có outliers nhiều (R² sensitive to outliers)


### 📐 MAPE (Mean Absolute Percentage Error)

**Công thức**: `MAPE = (100/n) * Σ|y_true - y_pred| / |y_true|`

**Ý nghĩa**:
- Sai số phần trăm trung bình
- Đơn vị: % (percent)
- **Scale-independent**: Có thể so sánh MAPE giữa datasets khác nhau

**Giải thích kết quả**:
- MAPE = 5%: Dự đoán sai trung bình 5% so với giá trị thực
- MAPE < 10%: Excellent forecast
- MAPE 10-20%: Good forecast
- MAPE > 20%: Poor forecast

**Khi nào dùng**:
- ✅ Khi cần metric dễ hiểu cho business (% error)
- ✅ Khi so sánh models trên datasets khác scale
- ❌ Khi y_true có giá trị = 0 (division by zero)
- ❌ Khi y_true rất nhỏ (MAPE sẽ rất lớn)


## 🐛 Common Issues

### ❌ Issue 1: ModuleNotFoundError: No module named 'matplotlib'

**Triệu chứng**:
```
ImportError: matplotlib not installed. Charts will not be available.
```

**Nguyên nhân**: `matplotlib` và `seaborn` không được cài đặt (optional dependencies).

**Giải pháp**:
```bash
pip install matplotlib seaborn
```

**Note**: Metrics functions vẫn hoạt động bình thường, chỉ visualization functions bị disabled.


### ❌ Issue 2: ValueError: Độ dài không khớp

**Triệu chứng**:
```
ValueError: Độ dài không khớp: y_true=1000, y_pred=950
```

**Nguyên nhân**: `y_true` và `y_pred` có số lượng samples khác nhau.

**Giải pháp**:
```python
# Kiểm tra trước khi tính metrics
assert len(y_true) == len(y_pred), "Length mismatch!"

# Hoặc filter NaN values
mask = ~(np.isnan(y_true) | np.isnan(y_pred))
y_true_clean = y_true[mask]
y_pred_clean = y_pred[mask]

metrics = calculate_all_metrics(y_true_clean, y_pred_clean)
```


### ❌ Issue 3: R² negative hoặc rất thấp

**Triệu chứng**:
```python
r2 = r2_score(y_test, y_pred)
print(r2)  # -0.35 hoặc 0.02
```

**Nguyên nhân**:
- R² < 0: Model tệ hơn baseline (predict = mean)
- R² gần 0: Model không học được gì từ data

**Giải pháp**:
1. Kiểm tra data leakage (target column trong features?)
2. Thử feature engineering khác
3. Hyperparameter tuning
4. Thử model architecture khác (XGBoost → LightGBM)


### ❌ Issue 4: MAPE = inf hoặc rất lớn

**Triệu chứng**:
```python
mape = mean_absolute_percentage_error(y_true, y_pred)
print(mape)  # inf hoặc 500000%
```

**Nguyên nhân**: `y_true` có giá trị = 0 hoặc rất nhỏ, gây division by zero.

**Giải pháp**:
```python
# Filter out zero values
mask = y_true != 0
y_true_filtered = y_true[mask]
y_pred_filtered = y_pred[mask]

mape = mean_absolute_percentage_error(y_true_filtered, y_pred_filtered)
```

**Alternative**: Dùng sMAPE (symmetric MAPE) thay vì MAPE:

```python
def symmetric_mape(y_true, y_pred):
    """sMAPE - ít sensitive với zero values"""
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return 100 * np.mean(numerator / denominator)
```


## 🚀 Future Enhancements

- [ ] **Add classification metrics**: Precision, Recall, F1-Score cho classification tasks
- [ ] **Time series metrics**: SMAPE, MASE cho weather forecasting
- [ ] **Cross-validation reporting**: K-fold CV results với std deviation
- [ ] **HTML report export**: Interactive HTML reports với Plotly charts
- [ ] **Automated hyperparameter comparison**: Visualize hyperparameter impact
- [ ] **Confidence intervals**: Bootstrap confidence intervals cho metrics
- [ ] **Model explainability**: Integrate SHAP values, LIME explanations
- [ ] **A/B testing utils**: Statistical tests để so sánh models


## 📞 Related Files

**Imports từ evaluation/**:
```python
# Trong training script
from Weather_Forcast_App.Machine_learning_model.evaluation.metrics import (
    calculate_all_metrics,
    mean_absolute_error,
    r2_score
)

from Weather_Forcast_App.Machine_learning_model.evaluation.report import (
    ModelEvaluationResult,
    ComparisonReport,
    save_comparison_report
)
```

**Related modules**:
- `trainning/` — Uses evaluation metrics after training
- `Models/` — Model wrappers return evaluation reports
- `interface/predictor.py` — May use metrics for prediction validation
- `Machine_learning_artifacts/` — Stores Metrics.json generated by this module


## 👨‍💻 Maintainer

**Võ Anh Nhật** - voanhnhat1612@gmail.com

*Last Updated: March 8, 2026*
