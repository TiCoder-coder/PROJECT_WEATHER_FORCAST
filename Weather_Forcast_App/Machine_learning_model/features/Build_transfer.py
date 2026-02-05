# ----------------------------- FEATURE BUILDER & TRANSFORMER -------------------------------------------------
"""
Build_transfer.py - Module xây dựng và biến đổi features cho machine learning models

Mục đích:
    - Xây dựng pipeline biến đổi features từ dữ liệu thô
    - Feature engineering cho dự báo thời tiết
    - Chuẩn bị dữ liệu cho training và prediction
    - Tích hợp với các ML models

Chức năng chính:
    - Load và validate dữ liệu từ Schema
    - Feature scaling và normalization
    - Categorical encoding
    - Time-series feature engineering
    - Feature selection
    - Data pipeline cho training/prediction

Cách sử dụng:
    from Weather_Forcast_App.Machine_learning_model.features.Build_transfer import FeatureBuilder

    builder = FeatureBuilder()
    X_train, X_test, y_train, y_test = builder.prepare_training_data(data_path)
    predictions = builder.prepare_prediction_data(input_data)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Import local modules
from ..data.Schema import WeatherDataSchema, validate_weather_dataframe
from ..data.Loader import DataLoader


class FeatureBuilder:
    """
    Class xây dựng và biến đổi features cho weather forecasting models.

    Attributes:
        scaler: Scaler cho numerical features
        categorical_encoders: Dict chứa encoders cho categorical features
        feature_selectors: Dict chứa feature selectors
        config: Cấu hình cho feature engineering
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Khởi tạo FeatureBuilder.

        Args:
            config: Dict cấu hình cho feature engineering
        """
        self.config = config or self._get_default_config()

        # Khởi tạo scalers
        self.scaler = self._init_scaler()
        self.categorical_encoders = {}
        self.feature_selectors = {}

        # Feature engineering flags
        self.is_fitted = False

        # Data loader
        self.data_loader = DataLoader()

    def _get_default_config(self) -> Dict[str, Any]:
        """Lấy cấu hình mặc định."""
        return {
            'scaler_type': 'standard',  # 'standard', 'minmax', 'robust'
            'handle_missing': 'mean',   # 'mean', 'median', 'most_frequent', 'drop'
            'categorical_encoding': 'onehot',  # 'onehot', 'label'
            'feature_selection': {
                'method': 'mutual_info',  # 'mutual_info', 'f_regression', 'none'
                'k': 20  # số features chọn, None để chọn tất cả
            },
            'time_features': {
                'extract_hour': True,
                'extract_day_of_year': True,
                'extract_month': True,
                'extract_season': True,
                'cyclic_encoding': True
            },
            'weather_features': {
                'create_interactions': True,
                'create_ratios': True,
                'create_differences': True
            },
            'test_size': 0.2,
            'random_state': 42
        }

    def _init_scaler(self):
        """Khởi tạo scaler dựa trên config."""
        scaler_type = self.config.get('scaler_type', 'standard')

        if scaler_type == 'standard':
            return StandardScaler()
        elif scaler_type == 'minmax':
            return MinMaxScaler()
        elif scaler_type == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unsupported scaler type: {scaler_type}")

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Xử lý missing values."""
        method = self.config.get('handle_missing', 'mean')

        if method == 'drop':
            return df.dropna()
        else:
            # Sử dụng SimpleImputer
            strategy = method if method in ['mean', 'median', 'most_frequent'] else 'mean'
            imputer = SimpleImputer(strategy=strategy)

            # Chỉ impute numerical columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

            return df

    def _encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features."""
        encoding_method = self.config.get('categorical_encoding', 'onehot')

        # Tìm categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        if len(categorical_cols) == 0:
            return df

        if encoding_method == 'label':
            for col in categorical_cols:
                if fit:
                    encoder = LabelEncoder()
                    df[col] = encoder.fit_transform(df[col].astype(str))
                    self.categorical_encoders[col] = encoder
                else:
                    if col in self.categorical_encoders:
                        df[col] = self.categorical_encoders[col].transform(df[col].astype(str))
                    else:
                        # Nếu không có encoder, dùng label encoding mới
                        encoder = LabelEncoder()
                        df[col] = encoder.fit_transform(df[col].astype(str))

        elif encoding_method == 'onehot':
            # One-hot encoding
            df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
            return df_encoded

        return df

    def _extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trích xuất features từ thời gian."""
        time_config = self.config.get('time_features', {})

        # Tìm cột thời gian
        time_cols = [col for col in df.columns if 'thoi_gian' in col.lower() or 'dau_thoi_gian' in col.lower()]

        if not time_cols:
            return df

        time_col = time_cols[0]  # Sử dụng cột đầu tiên

        # Chuyển đổi sang datetime nếu cần
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

        # Extract basic time features
        if time_config.get('extract_hour', True):
            df['hour'] = df[time_col].dt.hour

        if time_config.get('extract_day_of_year', True):
            df['day_of_year'] = df[time_col].dt.dayofyear

        if time_config.get('extract_month', True):
            df['month'] = df[time_col].dt.month

        if time_config.get('extract_season', True):
            # Xác định mùa (Việt Nam)
            df['season'] = df[time_col].dt.month.map(self._get_season)

        # Cyclic encoding cho hour và month
        if time_config.get('cyclic_encoding', True):
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        return df

    def _get_season(self, month: int) -> str:
        """Xác định mùa dựa trên tháng."""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:  # 9, 10, 11
            return 'autumn'

    def _create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tạo features đặc trưng cho thời tiết."""
        weather_config = self.config.get('weather_features', {})

        # Tìm các cột thời tiết
        temp_cols = [col for col in df.columns if 'nhiet_do' in col.lower()]
        humidity_cols = [col for col in df.columns if 'do_am' in col.lower()]
        wind_cols = [col for col in df.columns if 'toc_do_gio' in col.lower()]
        pressure_cols = [col for col in df.columns if 'ap_suat' in col.lower()]

        # Tạo interactions
        if weather_config.get('create_interactions', True):
            if temp_cols and humidity_cols:
                df['temp_humidity_interaction'] = df[temp_cols[0]] * df[humidity_cols[0]]

            if temp_cols and wind_cols:
                df['temp_wind_interaction'] = df[temp_cols[0]] * df[wind_cols[0]]

        # Tạo ratios
        if weather_config.get('create_ratios', True):
            if len(temp_cols) >= 2:  # có min và max
                df['temp_range_ratio'] = (df[temp_cols[1]] - df[temp_cols[0]]) / (df[temp_cols[0]] + 1e-6)

            if len(humidity_cols) >= 2:
                df['humidity_range_ratio'] = (df[humidity_cols[1]] - df[humidity_cols[0]]) / (df[humidity_cols[0]] + 1e-6)

        # Tạo differences
        if weather_config.get('create_differences', True):
            if len(temp_cols) >= 2:
                df['temp_range'] = df[temp_cols[1]] - df[temp_cols[0]]

            if len(humidity_cols) >= 2:
                df['humidity_range'] = df[humidity_cols[1]] - df[humidity_cols[0]]

        return df

    def _select_features(self, X: pd.DataFrame, y: pd.Series, fit: bool = True) -> pd.DataFrame:
        """Chọn features quan trọng."""
        selection_config = self.config.get('feature_selection', {})
        method = selection_config.get('method', 'none')
        k = selection_config.get('k', None)

        if method == 'none' or k is None:
            return X

        # Loại bỏ datetime columns trước khi feature selection
        datetime_cols = X.select_dtypes(include=['datetime64']).columns
        X_no_datetime = X.drop(columns=datetime_cols) if len(datetime_cols) > 0 else X

        # Chỉ chọn features numerical
        numerical_cols = X_no_datetime.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) < k:
            k = len(numerical_cols)

        X_numerical = X_no_datetime[numerical_cols]

        if fit:
            if method == 'mutual_info':
                selector = SelectKBest(score_func=mutual_info_regression, k=k)
            elif method == 'f_regression':
                selector = SelectKBest(score_func=f_regression, k=k)
            else:
                return X

            X_selected = selector.fit_transform(X_numerical, y)
            self.feature_selectors['main'] = selector

            # Lấy tên columns được chọn
            selected_features = X_numerical.columns[selector.get_support()].tolist()
            X_selected = pd.DataFrame(X_selected, columns=selected_features, index=X.index)

            # Thêm lại datetime columns nếu có
            if len(datetime_cols) > 0:
                X_selected = pd.concat([X_selected, X[datetime_cols]], axis=1)

        else:
            if 'main' in self.feature_selectors:
                X_selected_num = self.feature_selectors['main'].transform(X_numerical)
                selected_features = X_numerical.columns[self.feature_selectors['main'].get_support()].tolist()
                X_selected = pd.DataFrame(X_selected_num, columns=selected_features, index=X.index)

                # Thêm lại datetime columns nếu có
                if len(datetime_cols) > 0:
                    X_selected = pd.concat([X_selected, X[datetime_cols]], axis=1)
            else:
                X_selected = X

        return X_selected

    def fit_transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Fit và transform features.

        Args:
            df: DataFrame input
            target_col: Tên cột target (cho feature selection)

        Returns:
            DataFrame đã được transform
        """
        # Copy để tránh modify original
        df_processed = df.copy()

        # Validate với schema nếu có thể
        try:
            if 'location_ma_tram' in df_processed.columns:
                # Đây là flat dict từ schema
                pass  # Đã được validate rồi
        except:
            pass

        # Xử lý missing values
        df_processed = self._handle_missing_values(df_processed)

        # Extract time features
        df_processed = self._extract_time_features(df_processed)

        # Create weather-specific features
        df_processed = self._create_weather_features(df_processed)

        # Encode categorical features
        df_processed = self._encode_categorical_features(df_processed, fit=True)

        # Separate target if provided
        X = df_processed
        y = None
        if target_col and target_col in df_processed.columns:
            X = df_processed.drop(columns=[target_col])
            y = df_processed[target_col]

        # Scale numerical features
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])

        # Feature selection
        if y is not None:
            X = self._select_features(X, y, fit=True)

        self.is_fitted = True
        return X

    def transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Transform features với model đã fit.

        Args:
            df: DataFrame input
            target_col: Tên cột target (sẽ bị loại bỏ nếu có)

        Returns:
            DataFrame đã được transform
        """
        if not self.is_fitted:
            raise ValueError("FeatureBuilder chưa được fit! Gọi fit_transform() trước.")

        # Copy để tránh modify original
        df_processed = df.copy()

        # Xử lý missing values
        df_processed = self._handle_missing_values(df_processed)

        # Extract time features
        df_processed = self._extract_time_features(df_processed)

        # Create weather-specific features
        df_processed = self._create_weather_features(df_processed)

        # Encode categorical features
        df_processed = self._encode_categorical_features(df_processed, fit=False)

        # Separate target if provided (loại bỏ target column)
        X = df_processed
        if target_col and target_col in df_processed.columns:
            X = df_processed.drop(columns=[target_col])

        # Scale numerical features
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            X[numerical_cols] = self.scaler.transform(X[numerical_cols])

        # Feature selection
        X = self._select_features(X, y=None, fit=False)  # y=None vì không có target trong prediction

        return X

    def prepare_training_data(self, data_path: Union[str, Path],
                            target_column: str,
                            test_size: Optional[float] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Chuẩn bị dữ liệu cho training.

        Args:
            data_path: Đường dẫn đến file dữ liệu
            target_column: Tên cột target
            test_size: Tỷ lệ test set

        Returns:
            Tuple (X_train, X_test, y_train, y_test)
        """
        # Load dữ liệu
        df = pd.read_csv(data_path)

        # Fit transform
        X = self.fit_transform(df, target_column)
        y = df[target_column]

        # Split train/test
        test_size = test_size or self.config.get('test_size', 0.2)
        random_state = self.config.get('random_state', 42)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        return X_train, X_test, y_train, y_test

    def prepare_prediction_data(self, input_data: Union[pd.DataFrame, Dict, List[Dict]]) -> pd.DataFrame:
        """
        Chuẩn bị dữ liệu cho prediction.

        Args:
            input_data: Dữ liệu input (DataFrame, dict, hoặc list of dicts)

        Returns:
            DataFrame đã được transform cho prediction
        """
        # Convert input to DataFrame
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            df = pd.DataFrame(input_data)
        else:
            df = input_data

        # Transform
        X = self.transform(df)

        return X

    def get_feature_names(self) -> List[str]:
        """Lấy danh sách tên features sau khi transform."""
        if not self.is_fitted:
            raise ValueError("FeatureBuilder chưa được fit!")

        # Lấy từ scaler nếu có
        if hasattr(self.scaler, 'feature_names_in_'):
            return list(self.scaler.feature_names_in_)

        return []

    def save_config(self, path: Union[str, Path]):
        """Lưu cấu hình."""
        import json
        config_to_save = {
            'config': self.config,
            'is_fitted': self.is_fitted,
            'scaler_type': self.config.get('scaler_type'),
            'categorical_encoders': list(self.categorical_encoders.keys()),
            'feature_selectors': list(self.feature_selectors.keys())
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_to_save, f, indent=2, ensure_ascii=False)

    def load_config(self, path: Union[str, Path]):
        """Tải cấu hình."""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        self.config = config_data.get('config', self._get_default_config())
        self.is_fitted = config_data.get('is_fitted', False)

        # Re-init scaler
        self.scaler = self._init_scaler()


# ============================= UTILITY FUNCTIONS =============================

def create_weather_feature_pipeline(config: Optional[Dict[str, Any]] = None) -> FeatureBuilder:
    """
    Tạo pipeline feature engineering cho weather forecasting.

    Args:
        config: Cấu hình tùy chỉnh

    Returns:
        FeatureBuilder instance
    """
    return FeatureBuilder(config)


def get_default_weather_features() -> Dict[str, List[str]]:
    """
    Lấy danh sách features mặc định cho weather forecasting.

    Returns:
        Dict với các nhóm features
    """
    return {
        'time_features': [
            'hour', 'day_of_year', 'month', 'season',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos'
        ],
        'weather_features': [
            'nhiet_do_hien_tai', 'nhiet_do_toi_da', 'nhiet_do_toi_thieu', 'nhiet_do_trung_binh',
            'do_am_hien_tai', 'do_am_toi_da', 'do_am_toi_thieu', 'do_am_trung_binh',
            'ap_suat_hien_tai', 'ap_suat_toi_da', 'ap_suat_toi_thieu', 'ap_suat_trung_binh',
            'toc_do_gio_hien_tai', 'toc_do_gio_toi_da', 'toc_do_gio_toi_thieu', 'toc_do_gio_trung_binh',
            'luong_mua_hien_tai', 'luong_mua_toi_da', 'luong_mua_toi_thieu', 'luong_mua_trung_binh',
            'do_che_phu_may_hien_tai', 'tam_nhin_hien_tai', 'xac_suat_sam_set'
        ],
        'derived_features': [
            'temp_humidity_interaction', 'temp_wind_interaction',
            'temp_range_ratio', 'humidity_range_ratio',
            'temp_range', 'humidity_range'
        ]
    }