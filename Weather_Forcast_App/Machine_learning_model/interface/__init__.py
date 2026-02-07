# ----------------------------- INTERFACE MODULE -----------------------------------------------------------
"""
Interface Module - Cổng giao tiếp giữa ML models và Django app

Module này cung cấp các components để Django views/API có thể:
    - Load và sử dụng model đã train
    - Dự báo thời tiết với input mới
    - Lấy thông tin về model (metrics, feature importance, etc.)

Exports:
    - WeatherPredictor: Class chính để dự báo
    - get_predictor: Lấy singleton instance của predictor
    - predict_weather: Hàm tiện ích để dự báo nhanh
    - PredictionResult: Data class chứa kết quả dự báo
    - ModelInfo: Data class chứa thông tin model

Cách sử dụng trong Django views:
    from Weather_Forcast_App.Machine_learning_model.interface import (
        get_predictor,
        predict_weather,
        WeatherPredictor
    )
    
    # Cách 1: Sử dụng singleton (recommended)
    predictor = get_predictor()
    result = predictor.predict(input_data)
    
    # Cách 2: Sử dụng hàm tiện ích
    result = predict_weather(input_data)
    
    # Cách 3: Tạo instance mới
    predictor = WeatherPredictor()
    result = predictor.predict(input_data)
"""

from .predictor import (
    # Main class
    WeatherPredictor,
    
    # Singleton getter
    get_predictor,
    
    # Quick prediction function
    predict_weather,
    
    # Data classes
    PredictionResult,
    ModelInfo,
    
    # Exceptions
    PredictorError,
    ModelNotLoadedError,
    FeatureListNotLoadedError,
    InvalidInputError,
    MissingFeaturesError,
)

__all__ = [
    # Main class
    'WeatherPredictor',
    
    # Functions
    'get_predictor',
    'predict_weather',
    
    # Data classes
    'PredictionResult',
    'ModelInfo',
    
    # Exceptions
    'PredictorError',
    'ModelNotLoadedError',
    'FeatureListNotLoadedError',
    'InvalidInputError',
    'MissingFeaturesError',
]
