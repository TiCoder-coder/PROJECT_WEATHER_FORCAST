# ----------------------------- MODELS PACKAGE INIT -----------------------------------------------------------
"""
Weather_Forcast_App.Models Package

Package này chứa tất cả các Model definitions cho ứng dụng.

Modules:
    - Base_model: Lớp cơ sở (BaseModel) với các trường và phương thức chung
    - Login: Model cho quản lý user authentication
    - Random_Forest_Model: Thuật toán Random Forest cho dự báo thời tiết
    - CatBoost_Model: Thuật toán CatBoost (Gradient Boosting) cho dự báo thời tiết

Usage:
    # Database Models
    from Weather_Forcast_App.Models import BaseModel, LoginModel
    from Weather_Forcast_App.Models import UUIDBaseModel, TimestampMixin, SoftDeleteMixin
    from Weather_Forcast_App.Models import ActiveManager, DeletedManager
    
    # ML Models
    from Weather_Forcast_App.Models import WeatherRandomForest, WeatherCatBoost
    from Weather_Forcast_App.Models import create_rf_classifier, create_rf_regressor
    from Weather_Forcast_App.Models import create_cb_classifier, create_cb_regressor
"""

# Import Base Models và Mixins
from .Base_model import (
    BaseModel,
    UUIDBaseModel,
    TimestampMixin,
    SoftDeleteMixin,
    ActiveManager,
    DeletedManager
)

# Import Application Models
from .Login import LoginModel

# Import ML Models - Random Forest
from .Random_Forest_Model import (
    WeatherRandomForest,
    create_classifier as create_rf_classifier,
    create_regressor as create_rf_regressor,
    TrainingResult as RFTrainingResult,
    PredictionResult as RFPredictionResult,
)

# Import ML Models - CatBoost
try:
    from .CatBoost_Model import (
        WeatherCatBoost,
        create_classifier as create_cb_classifier,
        create_regressor as create_cb_regressor,
        TrainingResult as CBTrainingResult,
        PredictionResult as CBPredictionResult,
    )
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    WeatherCatBoost = None
    create_cb_classifier = None
    create_cb_regressor = None

# Define public API
__all__ = [
    # ===== Database Models =====
    # Base Models
    'BaseModel',
    'UUIDBaseModel',
    
    # Mixins
    'TimestampMixin',
    'SoftDeleteMixin',
    
    # Managers
    'ActiveManager',
    'DeletedManager',
    
    # Application Models
    'LoginModel',
    
    # ===== Machine Learning Models =====
    # Random Forest
    'WeatherRandomForest',
    'create_rf_classifier',
    'create_rf_regressor',
    'RFTrainingResult',
    'RFPredictionResult',
    
    # CatBoost
    'WeatherCatBoost',
    'create_cb_classifier',
    'create_cb_regressor',
    'CBTrainingResult',
    'CBPredictionResult',
    'CATBOOST_AVAILABLE',
]
