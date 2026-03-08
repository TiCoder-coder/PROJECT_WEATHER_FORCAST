# 🔌 interface/ — Prediction Interface (Inference API)

## 📁 Overview

Module `interface/` cung cấp **API dự đoán (inference interface)** cho Weather Forecast ML Pipeline. Nó chịu trách nhiệm:
- **Load trained artifacts**: Model, Transform Pipeline, Feature List từ `Machine_learning_artifacts/`
- **Process new data**: Nhận DataFrame mới → build features → transform → predict
- **Return predictions**: Trả về kết quả dự đoán dưới dạng chuẩn hóa

Module này là **cầu nối giữa trained model và production system**. Django views sẽ gọi `WeatherPredictor` để thực hiện predictions real-time.


## 📂 Directory Structure

```
interface/
├── __init__.py                    # Package initializer (exports WeatherPredictor)
└── predictor.py                   # 🔮 WeatherPredictor class - main inference interface
```


## 🎯 Purpose

### ❓ Tại sao cần folder này?

| Vấn đề | Giải pháp |
|--------|-----------|
| **Sau khi train xong, làm sao predict với data mới?** | Load artifacts từ `latest/` folder và gọi `predictor.predict(df_new)` |
| **Phải đảm bảo predict dùng đúng features như lúc train** | Load `Feature_list.json` để biết exact features cần có |
| **Transform data mới theo pipeline đã train** | Load `Transform_pipeline.pkl` và apply transformations |
| **Tránh code duplication giữa training và inference** | Reuse `WeatherFeatureBuilder` và `WeatherTransformPipeline` |

### ✅ Lợi ích

- **Separation of concerns**: Training code tách biệt khỏi inference code
- **Artifact-driven**: Tự động load toàn bộ artifacts cần thiết từ folder
- **Consistency**: Đảm bảo predict sử dụng exact pipeline như training
- **Production-ready**: Interface sẵn sàng cho Django views gọi


## 📄 Files Explained

### 1. `__init__.py` — Package Initializer

**Mục đích**: Export `WeatherPredictor` class để import dễ dàng.

```python
# Weather_Forcast_App/Machine_learning_model/interface/__init__.py

from .predictor import WeatherPredictor

__all__ = ["WeatherPredictor"]
```

**Usage**:
```python
# Có thể import trực tiếp từ interface package
from Weather_Forcast_App.Machine_learning_model.interface import WeatherPredictor

# Thay vì phải import từ predictor.py
# from Weather_Forcast_App.Machine_learning_model.interface.predictor import WeatherPredictor
```


### 2. `predictor.py` — WeatherPredictor Class

**Mục đích**: Class chính để load artifacts và thực hiện predictions.

#### 🏗️ Architecture

```
WeatherPredictor
├── __init__()           # Constructor (nhận model, pipeline, features)
├── from_artifacts()     # Factory method để load từ folder (RECOMMENDED)
├── predict()            # Main prediction method
├── predict_single()     # Predict 1 row duy nhất
├── get_info()           # Lấy thông tin model/pipeline
└── validate_input()     # Kiểm tra data đầu vào có đúng format không
```

#### 📦 Required Artifacts

`WeatherPredictor` cần **4 files** trong artifacts folder:

| File | Required? | Mô tả |
|------|-----------|-------|
| `Model.pkl` | ✅ **Bắt buộc** | Trained model (XGBoost, LightGBM, CatBoost) |
| `Transform_pipeline.pkl` | ⚠️ Tùy chọn | Pipeline để transform data (Imputer, Scaler, Encoder) |
| `Feature_list.json` | ⚠️ Tùy chọn | Danh sách exact features để sử dụng |
| `Train_info.json` | ⚠️ Tùy chọn | Metadata (target column, group_by, train date) |

#### 🔧 Class Attributes

```python
class WeatherPredictor:
    def __init__(
        self,
        model: Any,                          # Trained model object
        pipeline: Any,                       # Transform pipeline (optional)
        feature_columns: List[str],          # Exact feature names
        target_column: str = "",             # Target column name
        feature_builder: Any = None,         # Feature builder (optional)
        train_info: Optional[Dict[str, Any]] = None,  # Training metadata
    ):
        self.model = model
        self.pipeline = pipeline
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.feature_builder = feature_builder
        self.train_info = train_info or {}
```

#### 🔧 Usage Examples

**Example 1: Load from artifacts folder**

```python
from Weather_Forcast_App.Machine_learning_model.interface import WeatherPredictor

# Load predictor từ folder mặc định (Machine_learning_artifacts/latest)
predictor = WeatherPredictor.from_artifacts()

# Hoặc load từ folder cụ thể
predictor = WeatherPredictor.from_artifacts(
    artifacts_dir="/path/to/Machine_learning_artifacts/old_model"
)
```

**Example 2: Predict with new data**

```python
import pandas as pd

# Prepare new data (DataFrame với các cột cần thiết)
new_data = pd.DataFrame({
    "Date": ["2024-01-15"],
    "Temperature_C": [25.3],
    "Humidity_%": [78.5],
    "Pressure_hPa": [1013.2],
    "Wind_Speed_km_h": [12.5]
    # ... các cột khác
})

# Predict
result = predictor.predict(new_data)

# Access predictions
print(result["predictions"])
# [15.3]  # Dự đoán lượng mưa 15.3mm

# Access full result
print(result)
{
    "predictions": [15.3],
    "n_samples": 1,
    "feature_columns": ["Temperature_C", "Humidity_%", ...],
    "target_column": "Precipitation_mm",
    "model_name": "XGBoost"
}
```

**Example 3: Predict single row**

```python
# Predict 1 row duy nhất (return scalar thay vì array)
single_row = {
    "Date": "2024-01-15",
    "Temperature_C": 25.3,
    "Humidity_%": 78.5,
    "Pressure_hPa": 1013.2,
    "Wind_Speed_km_h": 12.5
}

prediction = predictor.predict_single(single_row)
print(prediction)  # 15.3
```

**Example 4: Get model info**

```python
# Lấy thông tin về model và pipeline
info = predictor.get_info()
print(info)
{
    "model_name": "XGBoost",
    "n_features": 25,
    "feature_columns": ["Temperature_C", "Humidity_%", ...],
    "target_column": "Precipitation_mm",
    "pipeline_steps": ["Imputer", "Scaler", "FeatureSelector"],
    "train_date": "2024-01-10",
    "train_samples": 10000
}
```


## 🔧 How to Use

### 1️⃣ Basic Workflow

```python
from Weather_Forcast_App.Machine_learning_model.interface import WeatherPredictor
import pandas as pd

# Step 1: Load predictor
predictor = WeatherPredictor.from_artifacts()

# Step 2: Prepare new data
new_data = pd.read_csv("new_weather_data.csv")

# Step 3: Predict
result = predictor.predict(new_data)

# Step 4: Use predictions
predictions = result["predictions"]
print(f"Predicted precipitation: {predictions[0]:.2f} mm")
```


### 2️⃣ Integration with Django Views

```python
# Weather_Forcast_App/views/PredictView.py

from django.http import JsonResponse
from Weather_Forcast_App.Machine_learning_model.interface import WeatherPredictor
import pandas as pd

# Load predictor once at module level (singleton pattern)
predictor = WeatherPredictor.from_artifacts()

def predict_weather(request):
    """API endpoint để predict thời tiết"""
    
    # Get data from request
    data = {
        "Date": request.GET.get("date"),
        "Temperature_C": float(request.GET.get("temperature")),
        "Humidity_%": float(request.GET.get("humidity")),
        "Pressure_hPa": float(request.GET.get("pressure")),
        "Wind_Speed_km_h": float(request.GET.get("wind_speed"))
    }
    
    # Predict
    try:
        prediction = predictor.predict_single(data)
        return JsonResponse({
            "success": True,
            "precipitation_mm": round(prediction, 2),
            "message": f"Dự đoán lượng mưa: {prediction:.2f} mm"
        })
    except Exception as e:
        return JsonResponse({
            "success": False,
            "error": str(e)
        }, status=400)
```


### 3️⃣ Batch Prediction

```python
from Weather_Forcast_App.Machine_learning_model.interface import WeatherPredictor
import pandas as pd

# Load predictor
predictor = WeatherPredictor.from_artifacts()

# Load large dataset
df_forecast = pd.read_csv("weather_forecast_next_7_days.csv")

# Predict for all rows
result = predictor.predict(df_forecast)

# Add predictions to DataFrame
df_forecast["Predicted_Precipitation_mm"] = result["predictions"]

# Save results
df_forecast.to_csv("forecast_results.csv", index=False)

print(f"Predicted {len(df_forecast)} records successfully!")
```


### 4️⃣ Error Handling

```python
from Weather_Forcast_App.Machine_learning_model.interface import WeatherPredictor
import pandas as pd

try:
    # Load predictor
    predictor = WeatherPredictor.from_artifacts()
    
    # Prepare data
    new_data = pd.DataFrame({
        "Date": ["2024-01-15"],
        "Temperature_C": [25.3],
        # Missing Humidity_% column!
    })
    
    # Predict (will fail due to missing features)
    result = predictor.predict(new_data)
    
except FileNotFoundError as e:
    print(f"❌ Artifacts not found: {e}")
    print("→ Run training first to generate artifacts")
    
except ValueError as e:
    print(f"❌ Invalid input data: {e}")
    print("→ Check that all required features are present")
    
except Exception as e:
    print(f"❌ Unexpected error: {e}")
```


## 📊 Prediction Flow Diagram

```
┌─────────────────────┐
│   New Data (DF)     │
│  - Temperature_C    │
│  - Humidity_%       │
│  - Pressure_hPa     │
│  - ...              │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ WeatherPredictor    │
│ .predict(df)        │
└──────────┬──────────┘
           │
           ├─ 1. Validate Input
           │   (check required columns)
           │
           ├─ 2. Build Features (if feature_builder exists)
           │   (create lag features, rolling stats)
           │
           ├─ 3. Select Features
           │   (keep only feature_columns from Feature_list.json)
           │
           ├─ 4. Transform (if pipeline exists)
           │   (Impute → Scale → Encode)
           │
           ├─ 5. Predict
           │   (model.predict(X_transformed))
           │
           ▼
┌─────────────────────┐
│   Predictions       │
│  [15.3, 8.7, 0.0]   │
└─────────────────────┘
```


## 🛠️ Internal Methods

### `from_artifacts()` — Factory Method

**Signature**:
```python
@classmethod
def from_artifacts(
    cls,
    artifacts_dir: Optional[Union[str, Path]] = None,
) -> "WeatherPredictor":
```

**Mục đích**: Load predictor từ folder artifacts (RECOMMENDED way to create predictor).

**Process**:
1. Set artifacts_dir (default: `Machine_learning_artifacts/latest`)
2. Load `Model.pkl` (required)
3. Load `Transform_pipeline.pkl` (optional)
4. Load `Feature_list.json` (optional)
5. Load `Train_info.json` (optional)
6. Create `WeatherFeatureBuilder` if needed
7. Return `WeatherPredictor` instance

**Raises**:
- `FileNotFoundError`: Nếu artifacts_dir hoặc Model.pkl không tồn tại


### `predict()` — Main Prediction Method

**Signature**:
```python
def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
```

**Mục đích**: Predict cho nhiều rows trong DataFrame.

**Parameters**:
- `df`: DataFrame chứa data mới (phải có các cột cần thiết)

**Returns**:
```python
{
    "predictions": np.ndarray,        # Array predictions
    "n_samples": int,                 # Số lượng predictions
    "feature_columns": List[str],     # Features đã sử dụng
    "target_column": str,             # Target column name
    "model_name": str                 # Model name
}
```

**Process**:
1. Validate input (check required columns)
2. Build features (nếu có `feature_builder`)
3. Select features (dùng `feature_columns`)
4. Transform (nếu có `pipeline`)
5. Predict (model.predict())
6. Return result dict


### `predict_single()` — Single Row Prediction

**Signature**:
```python
def predict_single(self, data: Dict[str, Any]) -> float:
```

**Mục đích**: Predict cho 1 row duy nhất (tiện cho API calls).

**Parameters**:
- `data`: Dictionary với keys = column names

**Returns**: Scalar prediction (float)

**Example**:
```python
prediction = predictor.predict_single({
    "Temperature_C": 25.3,
    "Humidity_%": 78.5,
    "Pressure_hPa": 1013.2
})
# Returns: 15.3
```


### `get_info()` — Get Model Info

**Signature**:
```python
def get_info(self) -> Dict[str, Any]:
```

**Mục đích**: Lấy metadata về model, pipeline, features.

**Returns**:
```python
{
    "model_name": "XGBoost",
    "n_features": 25,
    "feature_columns": [...],
    "target_column": "Precipitation_mm",
    "pipeline_steps": ["Imputer", "Scaler"],
    "train_date": "2024-01-10",
    "train_samples": 10000
}
```


## 🐛 Common Issues

### ❌ Issue 1: FileNotFoundError: Artifacts dir not found

**Triệu chứng**:
```
FileNotFoundError: Artifacts dir not found: /path/to/Machine_learning_artifacts/latest
```

**Nguyên nhân**: Chưa train model hoặc artifacts folder bị xóa.

**Giải pháp**:
```bash
# Run training để generate artifacts
python manage.py train_model
```


### ❌ Issue 2: FileNotFoundError: Model.pkl not found

**Triệu chứng**:
```
FileNotFoundError: Model.pkl not found in /path/to/artifacts/latest
```

**Nguyên nhân**: Training script không save Model.pkl.

**Giải pháp**:
```python
# Trong training script, đảm bảo save model
import joblib

joblib.dump(model, artifacts_dir / "Model.pkl")
```


### ❌ Issue 3: ValueError: Missing required features

**Triệu chứng**:
```
ValueError: Missing required features: ['Humidity_%', 'Wind_Speed_km_h']
```

**Nguyên nhân**: Input DataFrame thiếu các cột cần thiết.

**Giải pháp**:
```python
# Kiểm tra trước khi predict
required_features = predictor.feature_columns
missing = set(required_features) - set(df.columns)

if missing:
    print(f"Missing features: {missing}")
    # Thêm các cột thiếu với giá trị mặc định hoặc impute
    for col in missing:
        df[col] = 0  # hoặc giá trị khác
```


### ❌ Issue 4: Model predictions are NaN

**Triệu chứng**:
```python
result = predictor.predict(df)
print(result["predictions"])  # [nan, nan, nan]
```

**Nguyên nhân**:
- Input data có NaN values mà pipeline không handle
- Feature transformation tạo ra NaN values

**Giải pháp**:
```python
# Kiểm tra NaN trong input
print(df.isnull().sum())

# Impute NaN values trước khi predict
df = df.fillna(df.mean())

# Hoặc filter out rows có NaN
df = df.dropna()

result = predictor.predict(df)
```


### ❌ Issue 5: Predictions khác xa so với training

**Triệu chứng**:
```python
# Training: MAE = 2.5mm
# Inference: predictions = [10000.0, 50000.0, ...]
```

**Nguyên nhân**:
- Feature units khác nhau (training: meters, inference: kilometers)
- Không apply transform pipeline
- Data leakage trong training

**Giải pháp**:
1. Kiểm tra units của features (temperature: °C vs °F?)
2. Đảm bảo Transform_pipeline.pkl được load và áp dụng
3. Validate predictions trên test set trước khi deploy:

```python
# Validate predictions
import numpy as np

predictions = predictor.predict(df_test)["predictions"]

# Check range
assert np.min(predictions) >= 0, "Negative predictions!"
assert np.max(predictions) < 500, "Predictions too large!"

# Check mean
mean_pred = np.mean(predictions)
assert 0 < mean_pred < 50, f"Mean prediction {mean_pred} looks wrong"
```


## 🚀 Future Enhancements

- [ ] **Prediction caching**: Cache predictions cho same input để tối ưu performance
- [ ] **Batch processing**: Chunking lớn datasets cho memory efficiency
- [ ] **Prediction confidence**: Return confidence intervals hoặc prediction variance
- [ ] **Feature importance at inference**: Explain predictions với SHAP values
- [ ] **Input validation schema**: JSON Schema để validate input data format
- [ ] **Multi-model ensemble**: Load nhiều models và ensemble predictions
- [ ] **A/B testing support**: Switch giữa model versions (latest vs old_model)
- [ ] **Prediction logging**: Log tất cả predictions vào database cho monitoring
- [ ] **Auto-reload artifacts**: Detect khi có model mới train và auto-reload


## 📞 Related Files

**Imports trong `predictor.py`**:
```python
from Weather_Forcast_App.Machine_learning_model.features.Build_transfer import (
    WeatherFeatureBuilder,
)

from Weather_Forcast_App.Machine_learning_model.features.Transformers import (
    WeatherTransformPipeline,
)
```

**Files được load**:
- `Machine_learning_artifacts/latest/Model.pkl`
- `Machine_learning_artifacts/latest/Transform_pipeline.pkl`
- `Machine_learning_artifacts/latest/Feature_list.json`
- `Machine_learning_artifacts/latest/Train_info.json`

**Được gọi bởi**:
- Django views: `Weather_Forcast_App/views/PredictView.py`
- Management commands: `Weather_Forcast_App/Machine_learning_model/WeatherForcast/WeatherForcast.py`
- Test scripts: `Weather_Forcast_App/Machine_learning_model/TEST/`


## 👨‍💻 Maintainer

**Võ Anh Nhật** - voanhnhat1612@gmail.com

*Last Updated: March 8, 2026*
