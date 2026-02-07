# ----------------------------- FEATURES MODULE -----------------------------------------------------------
"""
Features module - Feature engineering và transformation cho Weather Forecast ML Pipeline.

Modules:
    - Build_transfer: Build features từ raw data (lag, rolling, time, location features)
    - Transformers: Transform data (scaler, encoder, missing handler, pipeline)

Usage:
    # Feature Building
    from Weather_Forcast_App.Machine_learning_model.features import WeatherFeatureBuilder
    builder = WeatherFeatureBuilder()
    X_features = builder.build_all_features(df)
    
    # Transformation Pipeline
    from Weather_Forcast_App.Machine_learning_model.features import WeatherTransformPipeline
    pipeline = WeatherTransformPipeline()
    X_train_transformed = pipeline.fit_transform(X_train)
    pipeline.save('pipeline.pkl')
    
    # Prediction (sử dụng cùng pipeline)
    X_pred_transformed = pipeline.transform(X_pred)
"""

# Import từ Build_transfer
from .Build_transfer import (
    WeatherFeatureBuilder,
    DEFAULT_LAG_PERIODS,
    DEFAULT_ROLLING_WINDOWS,
    MAIN_WEATHER_COLUMNS,
    VIETNAM_SEASON_MAP,
    VIETNAM_REGIONS
)

# Import từ Transformers
from .Transformers import (
    # Base
    BaseWeatherTransformer,
    
    # Transformers
    WeatherScaler,
    CategoricalEncoder,
    MissingValueHandler,
    OutlierHandler,
    
    # Pipeline
    WeatherTransformPipeline,
    
    # Factory functions
    create_default_pipeline,
    create_minimal_pipeline,
    get_scaler,
    get_encoder,
    
    # Constants
    SCALER_TYPES,
    IMPUTER_STRATEGIES,
    ENCODER_TYPES
)

__all__ = [
    # Build_transfer
    'WeatherFeatureBuilder',
    'DEFAULT_LAG_PERIODS',
    'DEFAULT_ROLLING_WINDOWS',
    'MAIN_WEATHER_COLUMNS',
    'VIETNAM_SEASON_MAP',
    'VIETNAM_REGIONS',
    
    # Transformers - Base
    'BaseWeatherTransformer',
    
    # Transformers - Classes
    'WeatherScaler',
    'CategoricalEncoder', 
    'MissingValueHandler',
    'OutlierHandler',
    'WeatherTransformPipeline',
    
    # Transformers - Functions
    'create_default_pipeline',
    'create_minimal_pipeline',
    'get_scaler',
    'get_encoder',
    
    # Constants
    'SCALER_TYPES',
    'IMPUTER_STRATEGIES',
    'ENCODER_TYPES'
]
