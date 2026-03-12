# 🤖 ml_models/ — Model Definitions & Wrappers

## 📁 Overview

Module `ml_models/` được thiết kế để chứa **model definitions và wrappers** (các class bọc models) cho ML Pipeline. Hiện tại folder này **chỉ có `__init__.py`**, nghĩa là:
- Model definitions có thể được thêm vào sau
- Hiện tại project sử dụng scikit-learn models trực tiếp (XGBoost, LightGBM, CatBoost) mà không cần custom wrappers

Trong tương lai, folder này có thể chứa:
- Custom model classes kế thừa từ `BaseForecastModel`
- Ensemble models (VotingRegressor, StackingRegressor)
- Neural network models (PyTorch, TensorFlow)
- Auto-tuning wrappers (AutoML, Optuna integration)


## 📂 Directory Structure

```
ml_models/
└── __init__.py                    # Package initializer (hiện tại: empty)
```


## 🎯 Purpose

### ❓ Tại sao cần folder này?

| Vấn đề | Giải pháp |
|--------|-----------|
| **Cần một nơi tập trung model definitions** | `ml_models/` là folder chuyên cho model code |
| **Muốn tạo custom model classes** | Tạo `BaseForecastModel` trong folder này |
| **Cần ensemble nhiều models** | Tạo `EnsembleModel` wrapper |
| **Hỗ trợ nhiều model frameworks** | Organized structure: `sklearn_models.py`, `pytorch_models.py` |

### ✅ Lợi ích (khi implement)

- **Separation of concerns**: Model code tách biệt khỏi training/feature engineering
- **Reusability**: Model wrappers có thể dùng cho nhiều experiments
- **Standardization**: Tất cả models implement cùng interface
- **Testing**: Dễ test models riêng lẻ


## 📄 Files Explained

### 1. `__init__.py` — Package Initializer

**Hiện tại**: File rỗng, chỉ để Python nhận diện folder là package.

**Tương lai**: Export model classes để import dễ dàng.

```python
# Tương lai có thể như này:

# from .base_model import BaseForecastModel
# from .sklearn_models import XGBoostModel, LightGBMModel, CatBoostModel
# from .ensemble_models import WeatherEnsembleModel
# from .neural_models import LSTMForecastModel

# __all__ = [
#     "BaseForecastModel",
#     "XGBoostModel",
#     "LightGBMModel",
#     "CatBoostModel",
#     "WeatherEnsembleModel",
#     "LSTMForecastModel",
# ]
```


## 🔧 Current Usage

**Hiện tại**: Project sử dụng models trực tiếp từ scikit-learn/XGBoost/LightGBM:

```python
# Trong training script

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Tạo model trực tiếp
model = XGBRegressor(
    learning_rate=0.1,
    max_depth=5,
    n_estimators=100
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
```

**Ưu điểm**: Đơn giản, không cần wrapper code.

**Nhược điểm**: Thiếu standardization, khó quản lý khi có nhiều models.


## 🚀 Future Implementation Ideas

### 1️⃣ Base Model Class

**File**: `ml_models/base_model.py`

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

class BaseForecastModel(ABC):
    """Base class cho tất cả forecast models"""
    
    def __init__(self, **kwargs):
        self.model = None
        self.is_fitted = False
        self.hyperparameters = kwargs
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train model"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict"""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance"""
        pass
    
    def validate_input(self, X: pd.DataFrame):
        """Validate input data shape"""
        if not self.is_fitted:
            raise ValueError("Model chưa được train!")
        # Additional validation logic...
```

**Lợi ích**:
- Standardization: Tất cả models implement cùng interface
- Type safety: Ensures consistent API
- Easier testing: Mock models cho unit tests


### 2️⃣ Sklearn Model Wrappers

**File**: `ml_models/sklearn_models.py`

```python
from .base_model import BaseForecastModel
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd

class XGBoostModel(BaseForecastModel):
    """XGBoost wrapper với custom logic"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = XGBRegressor(**kwargs)
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.validate_input(X)
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        if not self.is_fitted:
            return {}
        importance = self.model.feature_importances_
        feature_names = self.model.feature_names_in_
        return dict(zip(feature_names, importance))


class LightGBMModel(BaseForecastModel):
    """LightGBM wrapper"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = LGBMRegressor(**kwargs)
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.validate_input(X)
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        if not self.is_fitted:
            return {}
        importance = self.model.feature_importances_
        feature_names = self.model.feature_names_in_
        return dict(zip(feature_names, importance))
```

**Usage**:
```python
from Weather_Forcast_App.Machine_learning_model.ml_models import XGBoostModel

# Create model
model = XGBoostModel(
    learning_rate=0.1,
    max_depth=5,
    n_estimators=100
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Feature importance
importance = model.get_feature_importance()
print(importance)
# {"Temperature_C": 0.35, "Humidity_%": 0.28, ...}
```


### 3️⃣ Ensemble Model

**File**: `ml_models/ensemble_models.py`

```python
from .base_model import BaseForecastModel
from typing import List
import numpy as np
import pandas as pd

class WeatherEnsembleModel(BaseForecastModel):
    """Ensemble nhiều models với weighted averaging"""
    
    def __init__(self, models: List[BaseForecastModel], weights: List[float] = None):
        super().__init__()
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        
        if len(self.models) != len(self.weights):
            raise ValueError("Số lượng models và weights không khớp!")
        
        if abs(sum(self.weights) - 1.0) > 1e-6:
            raise ValueError("Tổng weights phải = 1.0!")
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Train tất cả models
        for model in self.models:
            model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.validate_input(X)
        
        # Get predictions từ tất cả models
        predictions = [model.predict(X) for model in self.models]
        
        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        return ensemble_pred
    
    def get_feature_importance(self) -> Dict[str, float]:
        # Average feature importance từ tất cả models
        all_importance = [model.get_feature_importance() for model in self.models]
        
        # Combine
        combined = {}
        for importance_dict in all_importance:
            for feature, value in importance_dict.items():
                combined[feature] = combined.get(feature, 0) + value
        
        # Normalize
        total = sum(combined.values())
        return {k: v / total for k, v in combined.items()}
```

**Usage**:
```python
from Weather_Forcast_App.Machine_learning_model.ml_models import (
    XGBoostModel,
    LightGBMModel,
    CatBoostModel,
    WeatherEnsembleModel
)

# Create individual models
xgb = XGBoostModel(learning_rate=0.1, max_depth=5)
lgb = LightGBMModel(learning_rate=0.1, num_leaves=31)
cat = CatBoostModel(learning_rate=0.1, depth=5, verbose=False)

# Create ensemble
ensemble = WeatherEnsembleModel(
    models=[xgb, lgb, cat],
    weights=[0.4, 0.3, 0.3]  # XGBoost có weight cao nhất
)

# Train (trains all 3 models)
ensemble.fit(X_train, y_train)

# Predict (weighted average of 3 predictions)
y_pred = ensemble.predict(X_test)
```


### 4️⃣ Neural Network Model (Future)

**File**: `ml_models/neural_models.py`

```python
from .base_model import BaseForecastModel
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class LSTMForecastModel(BaseForecastModel):
    """LSTM model cho time series forecasting"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.model = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32)
        
        # Training loop...
        # optimizer = torch.optim.Adam(self.model.parameters())
        # loss_fn = nn.MSELoss()
        # ... training code ...
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.validate_input(X)
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        
        with torch.no_grad():
            lstm_out, _ = self.model(X_tensor)
            predictions = self.fc(lstm_out[:, -1, :])
        
        return predictions.numpy().flatten()
```


## 🐛 Common Issues (Future)

### ❌ Issue 1: ImportError: cannot import name 'XGBoostModel'

**Nguyên nhân**: Chưa implement model wrapper hoặc quên export trong `__init__.py`.

**Giải pháp**:
```python
# Implement model wrapper trong sklearn_models.py
# Sau đó export trong __init__.py

from .sklearn_models import XGBoostModel
__all__ = ["XGBoostModel"]
```


### ❌ Issue 2: Model không implement fit/predict methods

**Nguyên nhân**: Custom model không kế thừa từ `BaseForecastModel` hoặc không implement abstract methods.

**Giải pháp**:
```python
from .base_model import BaseForecastModel

class MyCustomModel(BaseForecastModel):
    def fit(self, X, y):
        # Required implementation
        pass
    
    def predict(self, X):
        # Required implementation
        pass
    
    def get_feature_importance(self):
        # Required implementation
        pass
```


## 🚀 Future Enhancements

- [ ] **Implement BaseForecastModel**: Abstract base class cho standardization
- [ ] **Create sklearn wrappers**: XGBoostModel, LightGBMModel, CatBoostModel
- [ ] **Ensemble models**: WeatherEnsembleModel với weighted averaging
- [ ] **Auto-tuning wrapper**: AutoML-style model với Optuna hyperparameter tuning
- [ ] **Neural network models**: LSTM, GRU cho time series forecasting
- [ ] **Model registry**: Central registry để track model versions
- [ ] **Model comparison**: Automated comparison với cross-validation
- [ ] **Transfer learning**: Pre-trained models cho weather forecasting


## 📞 Related Files

**Sẽ được import bởi** (khi implement):
- `trainning/` — Training scripts sẽ sử dụng model wrappers
- `interface/predictor.py` — Predictor sẽ load models từ ml_models/
- `evaluation/` — Evaluation scripts sẽ test models

**Tương tự với**:
- `Models/` (Django models) — Đây là database models, khác với ML models
- `WeatherForcast/` — Prediction runner (sử dụng models cho inference)


## 👨‍💻 Maintainer

**Võ Anh Nhật** - voanhnhat1612@gmail.com

*Last Updated: March 8, 2026*
