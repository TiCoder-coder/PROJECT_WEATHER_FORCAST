# ----------------------------- TRANSFORMERS - MODULE BIẾN ĐỔI DỮ LIỆU -----------------------------------------------------------
"""
Transformers.py - Các transformer dạng module dùng lại cho machine learning pipeline

Mục đích:
    - StandardScaler/MinMaxScaler/RobustScaler (có thể fit & transform riêng)
    - Encoding cho categorical features (Label/OneHot/Target encoding)
    - Xử lý missing values nâng cao (KNN imputer, iterative imputer)
    - Pipeline transform thống nhất cho train & predict
    - Đảm bảo: "train và predict dùng đúng cùng 1 kiểu transform"

Chức năng chính:
    - WeatherScaler: Wrapper cho các loại scaler
    - CategoricalEncoder: Encoding categorical features
    - MissingValueHandler: Xử lý missing values
    - WeatherTransformPipeline: Pipeline tổng hợp

Cách sử dụng:
    from Weather_Forcast_App.Machine_learning_model.features.Transformers import WeatherTransformPipeline
    
    # Training
    pipeline = WeatherTransformPipeline()
    X_train_transformed = pipeline.fit_transform(X_train)
    pipeline.save('pipeline.pkl')
    
    # Prediction
    pipeline = WeatherTransformPipeline.load('pipeline.pkl')
    X_pred_transformed = pipeline.transform(X_pred)
"""

import pandas as pd
import numpy as np
import joblib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin

# Setup logging
logger = logging.getLogger(__name__)


# ============================= CONSTANTS =============================

SCALER_TYPES = {
    'standard': StandardScaler,
    'minmax': MinMaxScaler,
    'robust': RobustScaler
}

IMPUTER_STRATEGIES = ['mean', 'median', 'most_frequent', 'constant', 'knn']

ENCODER_TYPES = ['label', 'onehot', 'ordinal', 'target']


# ============================= BASE TRANSFORMER =============================

class BaseWeatherTransformer(ABC, BaseEstimator, TransformerMixin):
    """
    Base class cho tất cả Weather Transformers.
    
    Kế thừa từ sklearn BaseEstimator và TransformerMixin để:
    - Tương thích với sklearn Pipeline
    - Có sẵn fit_transform() method
    - Có thể clone và serialize
    """
    
    def __init__(self, name: str = "BaseTransformer"):
        self.name = name
        self.is_fitted = False
        self._fitted_columns: List[str] = []
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit transformer với dữ liệu training."""
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform dữ liệu."""
        pass
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit và transform trong một bước."""
        self.fit(X, y)
        return self.transform(X)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Lấy parameters của transformer."""
        return {'name': self.name}
    
    def set_params(self, **params) -> 'BaseWeatherTransformer':
        """Set parameters cho transformer."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


# ============================= WEATHER SCALER =============================

class WeatherScaler(BaseWeatherTransformer):
    """
    Scaler cho numerical features trong dữ liệu thời tiết.
    
    Wrapper cho StandardScaler/MinMaxScaler/RobustScaler với:
    - Tự động detect numerical columns
    - Lưu lại column names để đảm bảo consistency
    - Support fit/transform riêng biệt
    
    Attributes:
        scaler_type: Loại scaler ('standard', 'minmax', 'robust')
        scaler: Sklearn scaler instance
        columns: Danh sách columns được scale
    """
    
    def __init__(
        self,
        scaler_type: str = 'standard',
        columns: Optional[List[str]] = None,
        **scaler_kwargs
    ):
        """
        Khởi tạo WeatherScaler.
        
        Args:
            scaler_type: Loại scaler ('standard', 'minmax', 'robust')
            columns: Danh sách columns cần scale (None = auto detect)
            **scaler_kwargs: Kwargs truyền vào scaler
        """
        super().__init__(name="WeatherScaler")
        
        if scaler_type not in SCALER_TYPES:
            raise ValueError(f"Unsupported scaler type: {scaler_type}. "
                           f"Choose from: {list(SCALER_TYPES.keys())}")
        
        self.scaler_type = scaler_type
        self.scaler = SCALER_TYPES[scaler_type](**scaler_kwargs)
        self.columns = columns
        self._scale_columns: List[str] = []
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'WeatherScaler':
        """
        Fit scaler với dữ liệu training.
        
        Args:
            X: DataFrame training data
            y: Target (không sử dụng, để tương thích với sklearn)
        
        Returns:
            self
        """
        # Xác định columns cần scale
        if self.columns is not None:
            self._scale_columns = [c for c in self.columns if c in X.columns]
        else:
            # Auto detect numerical columns
            self._scale_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(self._scale_columns) == 0:
            logger.warning("Không có numerical columns để scale!")
            self.is_fitted = True
            return self
        
        # Fit scaler
        self.scaler.fit(X[self._scale_columns])
        
        self.is_fitted = True
        self._fitted_columns = X.columns.tolist()
        
        logger.info(f"✅ WeatherScaler ({self.scaler_type}) đã fit {len(self._scale_columns)} columns")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dữ liệu với scaler đã fit.
        
        Args:
            X: DataFrame cần transform
        
        Returns:
            DataFrame đã được scale
        """
        if not self.is_fitted:
            raise ValueError("WeatherScaler chưa được fit! Gọi fit() trước.")
        
        X_result = X.copy()
        
        if len(self._scale_columns) == 0:
            return X_result
        
        # Chỉ scale các columns có trong fitted columns
        cols_to_scale = [c for c in self._scale_columns if c in X_result.columns]
        
        if len(cols_to_scale) > 0:
            X_result[cols_to_scale] = self.scaler.transform(X_result[cols_to_scale])
        
        return X_result
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform để lấy lại giá trị gốc.
        
        Args:
            X: DataFrame đã scale
        
        Returns:
            DataFrame với giá trị gốc
        """
        if not self.is_fitted:
            raise ValueError("WeatherScaler chưa được fit!")
        
        X_result = X.copy()
        
        if len(self._scale_columns) == 0:
            return X_result
        
        cols_to_inverse = [c for c in self._scale_columns if c in X_result.columns]
        
        if len(cols_to_inverse) > 0:
            X_result[cols_to_inverse] = self.scaler.inverse_transform(X_result[cols_to_inverse])
        
        return X_result
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Lấy parameters."""
        params = super().get_params(deep)
        params.update({
            'scaler_type': self.scaler_type,
            'columns': self.columns
        })
        return params


# ============================= CATEGORICAL ENCODER =============================

class CategoricalEncoder(BaseWeatherTransformer):
    """
    Encoder cho categorical features.
    
    Hỗ trợ:
    - Label Encoding: Chuyển categories thành số (0, 1, 2, ...)
    - One-Hot Encoding: Tạo dummy columns
    - Ordinal Encoding: Encoding có thứ tự
    - Target Encoding: Encode dựa trên mean của target
    
    Attributes:
        encoding_type: Loại encoding
        columns: Danh sách columns cần encode
        encoders: Dict chứa encoder cho mỗi column
    """
    
    def __init__(
        self,
        encoding_type: str = 'label',
        columns: Optional[List[str]] = None,
        handle_unknown: str = 'ignore'
    ):
        """
        Khởi tạo CategoricalEncoder.
        
        Args:
            encoding_type: Loại encoding ('label', 'onehot', 'ordinal')
            columns: Danh sách columns cần encode (None = auto detect)
            handle_unknown: Cách xử lý categories mới ('ignore', 'error')
        """
        super().__init__(name="CategoricalEncoder")
        
        if encoding_type not in ENCODER_TYPES:
            raise ValueError(f"Unsupported encoding type: {encoding_type}. "
                           f"Choose from: {ENCODER_TYPES}")
        
        self.encoding_type = encoding_type
        self.columns = columns
        self.handle_unknown = handle_unknown
        self.encoders: Dict[str, Any] = {}
        self._encode_columns: List[str] = []
        self._onehot_columns: List[str] = []  # Columns sau one-hot encoding
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'CategoricalEncoder':
        """
        Fit encoder với dữ liệu training.
        
        Args:
            X: DataFrame training data
            y: Target (sử dụng cho target encoding)
        
        Returns:
            self
        """
        # Xác định columns cần encode
        if self.columns is not None:
            self._encode_columns = [c for c in self.columns if c in X.columns]
        else:
            # Auto detect categorical columns
            self._encode_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(self._encode_columns) == 0:
            logger.info("Không có categorical columns để encode")
            self.is_fitted = True
            return self
        
        # Fit encoders
        for col in self._encode_columns:
            if self.encoding_type == 'label':
                encoder = LabelEncoder()
                encoder.fit(X[col].astype(str).fillna('__MISSING__'))
                self.encoders[col] = encoder
                
            elif self.encoding_type == 'ordinal':
                encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                encoder.fit(X[[col]].astype(str).fillna('__MISSING__'))
                self.encoders[col] = encoder
                
            elif self.encoding_type == 'onehot':
                # Lưu unique values để tạo dummy columns nhất quán
                unique_values = X[col].astype(str).fillna('__MISSING__').unique()
                self.encoders[col] = list(unique_values)
                
            elif self.encoding_type == 'target' and y is not None:
                # Target encoding: encode bằng mean của target
                target_means = X.groupby(col)[y.name].mean() if y.name else {}
                global_mean = y.mean()
                self.encoders[col] = {'means': target_means.to_dict(), 'global_mean': global_mean}
        
        self.is_fitted = True
        self._fitted_columns = X.columns.tolist()
        
        logger.info(f"✅ CategoricalEncoder ({self.encoding_type}) đã fit {len(self._encode_columns)} columns")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dữ liệu với encoder đã fit.
        
        Args:
            X: DataFrame cần transform
        
        Returns:
            DataFrame đã được encode
        """
        if not self.is_fitted:
            raise ValueError("CategoricalEncoder chưa được fit! Gọi fit() trước.")
        
        X_result = X.copy()
        
        if len(self._encode_columns) == 0:
            return X_result
        
        for col in self._encode_columns:
            if col not in X_result.columns:
                continue
            
            if self.encoding_type == 'label':
                encoder = self.encoders.get(col)
                if encoder:
                    # Xử lý unknown categories
                    values = X_result[col].astype(str).fillna('__MISSING__')
                    known_classes = set(encoder.classes_)
                    values = values.apply(lambda x: x if x in known_classes else '__UNKNOWN__')
                    
                    # Thêm __UNKNOWN__ vào classes nếu chưa có
                    if '__UNKNOWN__' not in known_classes:
                        encoder.classes_ = np.append(encoder.classes_, '__UNKNOWN__')
                    
                    X_result[col] = encoder.transform(values)
            
            elif self.encoding_type == 'ordinal':
                encoder = self.encoders.get(col)
                if encoder:
                    X_result[col] = encoder.transform(X_result[[col]].astype(str).fillna('__MISSING__')).flatten()
            
            elif self.encoding_type == 'onehot':
                unique_values = self.encoders.get(col, [])
                for val in unique_values:
                    col_name = f"{col}_{val}"
                    X_result[col_name] = (X_result[col].astype(str).fillna('__MISSING__') == val).astype(int)
                # Drop original column
                X_result = X_result.drop(columns=[col])
            
            elif self.encoding_type == 'target':
                encoding_info = self.encoders.get(col, {})
                means = encoding_info.get('means', {})
                global_mean = encoding_info.get('global_mean', 0)
                X_result[col] = X_result[col].map(means).fillna(global_mean)
        
        return X_result
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Lấy parameters."""
        params = super().get_params(deep)
        params.update({
            'encoding_type': self.encoding_type,
            'columns': self.columns,
            'handle_unknown': self.handle_unknown
        })
        return params


# ============================= MISSING VALUE HANDLER =============================

class MissingValueHandler(BaseWeatherTransformer):
    """
    Handler xử lý missing values nâng cao.
    
    Hỗ trợ:
    - SimpleImputer: mean, median, most_frequent, constant
    - KNNImputer: Sử dụng K-nearest neighbors
    - Forward/Backward fill: Cho time series
    - Custom strategy cho từng column
    
    Attributes:
        strategy: Chiến lược impute mặc định
        column_strategies: Dict chiến lược riêng cho từng column
        imputers: Dict chứa imputer cho mỗi nhóm columns
    """
    
    def __init__(
        self,
        strategy: str = 'median',
        fill_value: Optional[Any] = None,
        column_strategies: Optional[Dict[str, str]] = None,
        n_neighbors: int = 5
    ):
        """
        Khởi tạo MissingValueHandler.
        
        Args:
            strategy: Chiến lược mặc định ('mean', 'median', 'most_frequent', 'constant', 'knn', 'ffill', 'bfill')
            fill_value: Giá trị fill nếu strategy='constant'
            column_strategies: Dict chiến lược riêng cho từng column
            n_neighbors: Số neighbors cho KNN imputer
        """
        super().__init__(name="MissingValueHandler")
        
        self.strategy = strategy
        self.fill_value = fill_value
        self.column_strategies = column_strategies or {}
        self.n_neighbors = n_neighbors
        self.imputers: Dict[str, Any] = {}
        self._numeric_columns: List[str] = []
        self._categorical_columns: List[str] = []
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'MissingValueHandler':
        """
        Fit imputers với dữ liệu training.
        
        Args:
            X: DataFrame training data
            y: Target (không sử dụng)
        
        Returns:
            self
        """
        # Phân loại columns
        self._numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        self._categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Lưu fill values cho từng column (đơn giản và robust hơn sklearn imputer)
        self._column_fill_values = {}
        
        # Cho numerical columns
        for col in self._numeric_columns:
            if self.strategy == 'median':
                self._column_fill_values[col] = X[col].median()
            elif self.strategy == 'mean':
                self._column_fill_values[col] = X[col].mean()
            elif self.strategy == 'most_frequent':
                mode = X[col].mode()
                self._column_fill_values[col] = mode[0] if len(mode) > 0 else 0
            elif self.strategy == 'constant':
                self._column_fill_values[col] = self.fill_value if self.fill_value is not None else 0
            else:
                self._column_fill_values[col] = X[col].median()
        
        # Cho categorical columns
        self._categorical_fill_values = {}
        for col in self._categorical_columns:
            mode = X[col].mode()
            self._categorical_fill_values[col] = mode[0] if len(mode) > 0 else '__MISSING__'
        
        # Vẫn fit KNN imputer nếu cần (cho trường hợp đặc biệt)
        if self.strategy == 'knn' and len(self._numeric_columns) > 0:
            try:
                imputer = KNNImputer(n_neighbors=self.n_neighbors)
                imputer.fit(X[self._numeric_columns])
                self.imputers['numeric'] = imputer
            except Exception:
                pass  # Fallback to simple fill
        
        self.is_fitted = True
        self._fitted_columns = X.columns.tolist()
        
        logger.info(f"✅ MissingValueHandler ({self.strategy}) đã fit "
                   f"{len(self._numeric_columns)} numeric, {len(self._categorical_columns)} categorical columns")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dữ liệu - fill missing values.
        
        Args:
            X: DataFrame cần transform
        
        Returns:
            DataFrame đã được fill missing values
        """
        if not self.is_fitted:
            raise ValueError("MissingValueHandler chưa được fit! Gọi fit() trước.")
        
        X_result = X.copy()
        
        # Handle numerical columns - sử dụng giá trị đã lưu
        for col in self._numeric_columns:
            if col in X_result.columns and col in self._column_fill_values:
                fill_val = self._column_fill_values[col]
                # Handle NaN fill value
                if pd.isna(fill_val):
                    fill_val = 0
                X_result[col] = X_result[col].fillna(fill_val)
        
        # Handle categorical columns
        for col in self._categorical_columns:
            if col in X_result.columns and hasattr(self, '_categorical_fill_values'):
                fill_val = self._categorical_fill_values.get(col, '__MISSING__')
                X_result[col] = X_result[col].fillna(fill_val)
        
        # Handle column-specific strategies
        for col, strat in self.column_strategies.items():
            if col not in X_result.columns:
                continue
            
            if strat == 'ffill':
                X_result[col] = X_result[col].ffill()
            elif strat == 'bfill':
                X_result[col] = X_result[col].bfill()
            elif strat == 'zero':
                X_result[col] = X_result[col].fillna(0)
        
        return X_result
    
    def get_missing_report(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo báo cáo missing values.
        
        Args:
            X: DataFrame cần kiểm tra
        
        Returns:
            DataFrame báo cáo missing values
        """
        missing_count = X.isnull().sum()
        missing_percent = (X.isnull().sum() / len(X)) * 100
        
        report = pd.DataFrame({
            'column': X.columns,
            'missing_count': missing_count.values,
            'missing_percent': missing_percent.values,
            'dtype': X.dtypes.values
        })
        
        return report[report['missing_count'] > 0].sort_values('missing_count', ascending=False)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Lấy parameters."""
        params = super().get_params(deep)
        params.update({
            'strategy': self.strategy,
            'fill_value': self.fill_value,
            'column_strategies': self.column_strategies,
            'n_neighbors': self.n_neighbors
        })
        return params


# ============================= OUTLIER HANDLER =============================

class OutlierHandler(BaseWeatherTransformer):
    """
    Handler xử lý outliers.
    
    Hỗ trợ:
    - IQR method: Clip/remove values ngoài Q1-1.5*IQR và Q3+1.5*IQR
    - Z-score method: Clip/remove values có |z| > threshold
    - Percentile method: Clip giá trị ngoài percentile range
    
    Attributes:
        method: Phương pháp detect outliers ('iqr', 'zscore', 'percentile')
        action: Hành động với outliers ('clip', 'remove', 'nan')
        columns: Danh sách columns cần xử lý
    """
    
    def __init__(
        self,
        method: str = 'iqr',
        action: str = 'clip',
        columns: Optional[List[str]] = None,
        iqr_multiplier: float = 1.5,
        zscore_threshold: float = 3.0,
        percentile_range: Tuple[float, float] = (1, 99)
    ):
        """
        Khởi tạo OutlierHandler.
        
        Args:
            method: Phương pháp detect ('iqr', 'zscore', 'percentile')
            action: Hành động ('clip', 'remove', 'nan')
            columns: Danh sách columns (None = auto detect)
            iqr_multiplier: Multiplier cho IQR method
            zscore_threshold: Threshold cho z-score method
            percentile_range: Range cho percentile method (lower, upper)
        """
        super().__init__(name="OutlierHandler")
        
        self.method = method
        self.action = action
        self.columns = columns
        self.iqr_multiplier = iqr_multiplier
        self.zscore_threshold = zscore_threshold
        self.percentile_range = percentile_range
        
        self._bounds: Dict[str, Tuple[float, float]] = {}
        self._handle_columns: List[str] = []
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'OutlierHandler':
        """
        Fit - tính toán bounds cho mỗi column.
        
        Args:
            X: DataFrame training data
            y: Target (không sử dụng)
        
        Returns:
            self
        """
        # Xác định columns
        if self.columns is not None:
            self._handle_columns = [c for c in self.columns if c in X.columns]
        else:
            self._handle_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Tính bounds cho mỗi column
        for col in self._handle_columns:
            if self.method == 'iqr':
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Edge case: IQR=0 (e.g. zero-inflated data ≥75% same value)
                # → clipping to [Q1, Q3] would destroy ALL variation.
                # Fallback: use min/max (no clipping) for this column.
                if IQR == 0:
                    lower = X[col].min()
                    upper = X[col].max()
                else:
                    lower = Q1 - self.iqr_multiplier * IQR
                    upper = Q3 + self.iqr_multiplier * IQR
                
            elif self.method == 'zscore':
                mean = X[col].mean()
                std = X[col].std()
                lower = mean - self.zscore_threshold * std
                upper = mean + self.zscore_threshold * std
                
            elif self.method == 'percentile':
                lower = X[col].quantile(self.percentile_range[0] / 100)
                upper = X[col].quantile(self.percentile_range[1] / 100)
            
            else:
                lower, upper = X[col].min(), X[col].max()
            
            self._bounds[col] = (lower, upper)
        
        self.is_fitted = True
        self._fitted_columns = X.columns.tolist()
        
        logger.info(f"✅ OutlierHandler ({self.method}) đã fit {len(self._handle_columns)} columns")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform - xử lý outliers.
        
        Args:
            X: DataFrame cần transform
        
        Returns:
            DataFrame đã xử lý outliers
        """
        if not self.is_fitted:
            raise ValueError("OutlierHandler chưa được fit! Gọi fit() trước.")
        
        X_result = X.copy()
        
        for col in self._handle_columns:
            if col not in X_result.columns:
                continue
            
            lower, upper = self._bounds.get(col, (X_result[col].min(), X_result[col].max()))
            
            if self.action == 'clip':
                X_result[col] = X_result[col].clip(lower=lower, upper=upper)
            
            elif self.action == 'nan':
                mask = (X_result[col] < lower) | (X_result[col] > upper)
                X_result.loc[mask, col] = np.nan
            
            # 'remove' sẽ được handle ở level pipeline
        
        return X_result
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Lấy parameters."""
        params = super().get_params(deep)
        params.update({
            'method': self.method,
            'action': self.action,
            'columns': self.columns,
            'iqr_multiplier': self.iqr_multiplier,
            'zscore_threshold': self.zscore_threshold,
            'percentile_range': self.percentile_range
        })
        return params


# ============================= WEATHER TRANSFORM PIPELINE =============================

class WeatherTransformPipeline:
    """
    Pipeline transform thống nhất cho train & predict.
    
    Đây là class chính để đảm bảo:
    - Train và predict dùng đúng cùng 1 kiểu transform
    - Có thể save/load pipeline
    - Tự động xử lý theo đúng thứ tự
    
    Pipeline steps:
        1. Missing value handling
        2. Outlier handling (optional)
        3. Categorical encoding
        4. Numerical scaling
    
    Attributes:
        steps: List các transformer steps
        is_fitted: Trạng thái đã fit chưa
    """
    
    def __init__(
        self,
        missing_strategy: str = 'median',
        scaler_type: str = 'standard',
        encoding_type: str = 'label',
        handle_outliers: bool = False,
        outlier_method: str = 'iqr',
        custom_steps: Optional[List[BaseWeatherTransformer]] = None
    ):
        """
        Khởi tạo WeatherTransformPipeline.
        
        Args:
            missing_strategy: Chiến lược xử lý missing values
            scaler_type: Loại scaler
            encoding_type: Loại encoding cho categorical
            handle_outliers: Có xử lý outliers không
            outlier_method: Phương pháp xử lý outliers
            custom_steps: List các custom transformer steps
        """
        self.missing_strategy = missing_strategy
        self.scaler_type = scaler_type
        self.encoding_type = encoding_type
        self.handle_outliers = handle_outliers
        self.outlier_method = outlier_method
        
        # Initialize steps
        if custom_steps is not None:
            self.steps = custom_steps
        else:
            self.steps = self._create_default_steps()
        
        self.is_fitted = False
        self._feature_names: List[str] = []
        self._fitted_at: Optional[str] = None
    
    def _create_default_steps(self) -> List[BaseWeatherTransformer]:
        """Tạo default pipeline steps."""
        steps = []
        
        # Step 1: Missing value handler
        steps.append(MissingValueHandler(strategy=self.missing_strategy))
        
        # Step 2: Outlier handler (optional)
        if self.handle_outliers:
            steps.append(OutlierHandler(method=self.outlier_method, action='clip'))
        
        # Step 3: Categorical encoder
        steps.append(CategoricalEncoder(encoding_type=self.encoding_type))
        
        # Step 4: Scaler
        steps.append(WeatherScaler(scaler_type=self.scaler_type))
        
        return steps
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'WeatherTransformPipeline':
        """
        Fit tất cả steps trong pipeline.
        
        Args:
            X: DataFrame training data
            y: Target (truyền cho target encoding nếu cần)
        
        Returns:
            self
        """
        logger.info("🚀 Bắt đầu fit WeatherTransformPipeline...")
        
        X_current = X.copy()
        
        for i, step in enumerate(self.steps):
            logger.info(f"   Step {i+1}/{len(self.steps)}: {step.name}")
            X_current = step.fit_transform(X_current, y)
        
        self._feature_names = X_current.columns.tolist()
        self.is_fitted = True
        self._fitted_at = datetime.now().isoformat()
        
        logger.info(f"✅ WeatherTransformPipeline đã fit thành công!")
        logger.info(f"   - Input shape: {X.shape}")
        logger.info(f"   - Output shape: {X_current.shape}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dữ liệu với pipeline đã fit.
        
        Args:
            X: DataFrame cần transform
        
        Returns:
            DataFrame đã được transform
        """
        if not self.is_fitted:
            raise ValueError("WeatherTransformPipeline chưa được fit! Gọi fit() trước.")
        
        X_current = X.copy()
        
        for step in self.steps:
            X_current = step.transform(X_current)
        
        return X_current
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit và transform trong một bước.
        
        Args:
            X: DataFrame training data
            y: Target
        
        Returns:
            DataFrame đã được transform
        """
        self.fit(X, y)
        return self.transform(X)
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform (chỉ cho scaler).
        
        Args:
            X: DataFrame đã transform
        
        Returns:
            DataFrame với giá trị gốc (chỉ numerical columns)
        """
        X_current = X.copy()
        
        # Inverse transform theo thứ tự ngược
        for step in reversed(self.steps):
            if hasattr(step, 'inverse_transform'):
                X_current = step.inverse_transform(X_current)
        
        return X_current
    
    def get_feature_names(self) -> List[str]:
        """Lấy danh sách feature names sau transform."""
        return self._feature_names
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Lưu pipeline ra file.
        
        Args:
            path: Đường dẫn file (.pkl)
        """
        if not self.is_fitted:
            raise ValueError("Pipeline chưa được fit! Không thể save.")
        
        save_data = {
            'steps': self.steps,
            'is_fitted': self.is_fitted,
            'feature_names': self._feature_names,
            'fitted_at': self._fitted_at,
            'config': {
                'missing_strategy': self.missing_strategy,
                'scaler_type': self.scaler_type,
                'encoding_type': self.encoding_type,
                'handle_outliers': self.handle_outliers,
                'outlier_method': self.outlier_method
            }
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(save_data, path)
        logger.info(f"✅ Pipeline đã được lưu tại: {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'WeatherTransformPipeline':
        """
        Load pipeline từ file.
        
        Args:
            path: Đường dẫn file (.pkl)
        
        Returns:
            WeatherTransformPipeline instance
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file pipeline: {path}")
        
        save_data = joblib.load(path)
        
        # Tạo instance mới
        config = save_data.get('config', {})
        pipeline = cls(
            missing_strategy=config.get('missing_strategy', 'median'),
            scaler_type=config.get('scaler_type', 'standard'),
            encoding_type=config.get('encoding_type', 'label'),
            handle_outliers=config.get('handle_outliers', False),
            outlier_method=config.get('outlier_method', 'iqr')
        )
        
        # Restore state
        pipeline.steps = save_data['steps']
        pipeline.is_fitted = save_data['is_fitted']
        pipeline._feature_names = save_data['feature_names']
        pipeline._fitted_at = save_data.get('fitted_at')
        
        logger.info(f"✅ Pipeline đã được load từ: {path}")
        return pipeline
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Lấy thông tin về pipeline."""
        return {
            'is_fitted': self.is_fitted,
            'fitted_at': self._fitted_at,
            'n_steps': len(self.steps),
            'steps': [step.name for step in self.steps],
            'n_features': len(self._feature_names),
            'config': {
                'missing_strategy': self.missing_strategy,
                'scaler_type': self.scaler_type,
                'encoding_type': self.encoding_type,
                'handle_outliers': self.handle_outliers
            }
        }


# ============================= UTILITY FUNCTIONS =============================

def create_default_pipeline() -> WeatherTransformPipeline:
    """
    Tạo pipeline mặc định cho weather data.
    
    Returns:
        WeatherTransformPipeline instance
    """
    return WeatherTransformPipeline(
        missing_strategy='median',
        scaler_type='standard',
        encoding_type='label',
        handle_outliers=True,
        outlier_method='iqr'
    )


def create_minimal_pipeline() -> WeatherTransformPipeline:
    """
    Tạo pipeline tối giản (chỉ xử lý missing và scale).
    
    Returns:
        WeatherTransformPipeline instance
    """
    return WeatherTransformPipeline(
        missing_strategy='median',
        scaler_type='standard',
        encoding_type='label',
        handle_outliers=False
    )


def get_scaler(scaler_type: str = 'standard', **kwargs) -> WeatherScaler:
    """
    Factory function để tạo scaler.
    
    Args:
        scaler_type: Loại scaler
        **kwargs: Additional kwargs
    
    Returns:
        WeatherScaler instance
    """
    return WeatherScaler(scaler_type=scaler_type, **kwargs)


def get_encoder(encoding_type: str = 'label', **kwargs) -> CategoricalEncoder:
    """
    Factory function để tạo encoder.
    
    Args:
        encoding_type: Loại encoding
        **kwargs: Additional kwargs
    
    Returns:
        CategoricalEncoder instance
    """
    return CategoricalEncoder(encoding_type=encoding_type, **kwargs)


# ============================= MODULE TEST =============================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Testing Transformers Module")
    print("=" * 60)
    
    # Tạo sample data
    np.random.seed(42)
    n_samples = 100
    
    sample_data = pd.DataFrame({
        'nhiet_do': np.random.uniform(20, 35, n_samples),
        'do_am': np.random.uniform(60, 95, n_samples),
        'ap_suat': np.random.uniform(1005, 1020, n_samples),
        'toc_do_gio': np.random.uniform(0, 15, n_samples),
        'luong_mua': np.random.exponential(2, n_samples),
        'region': np.random.choice(['north', 'central', 'south'], n_samples),
        'station': np.random.choice(['A', 'B', 'C'], n_samples)
    })
    
    # Thêm missing values
    sample_data.loc[np.random.choice(n_samples, 10, replace=False), 'nhiet_do'] = np.nan
    sample_data.loc[np.random.choice(n_samples, 5, replace=False), 'do_am'] = np.nan
    
    # Thêm outliers
    sample_data.loc[0, 'nhiet_do'] = 100  # Outlier
    sample_data.loc[1, 'luong_mua'] = 500  # Outlier
    
    print(f"📊 Sample data shape: {sample_data.shape}")
    print(f"📊 Missing values:\n{sample_data.isnull().sum()}")
    
    # Test pipeline
    print("\n🔄 Testing WeatherTransformPipeline...")
    pipeline = WeatherTransformPipeline(
        missing_strategy='median',
        scaler_type='standard',
        encoding_type='label',
        handle_outliers=True
    )
    
    X_transformed = pipeline.fit_transform(sample_data)
    
    print(f"\n📊 Transformed shape: {X_transformed.shape}")
    print(f"📊 Missing after transform: {X_transformed.isnull().sum().sum()}")
    print(f"\n📋 Pipeline info:")
    for key, value in pipeline.get_pipeline_info().items():
        print(f"   {key}: {value}")
    
    # Test save/load
    print("\n💾 Testing save/load...")
    pipeline.save('test_pipeline.pkl')
    loaded_pipeline = WeatherTransformPipeline.load('test_pipeline.pkl')
    print(f"✅ Pipeline loaded successfully")
    
    # Clean up
    import os
    os.remove('test_pipeline.pkl')
    
    print("\n" + "=" * 60)
    print("🏁 Test hoàn thành")
    print("=" * 60)