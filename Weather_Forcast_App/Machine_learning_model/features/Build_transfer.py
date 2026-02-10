# ----------------------------- FEATURE BUILDER - XÂY DỰNG FEATURES TỪ RAW DATA -----------------------------------------------------------
"""
Build_transfer.py - Module xây dựng features từ dữ liệu thô cho machine learning models

Mục đích:
    - Xây dựng LAG features: rain(t-1), rain(t-7), temp(t-1), ...
    - Xây dựng ROLLING features: mean_7days, std_3days, ...
    - Xây dựng TIME features: day/month, sin/cos theo chu kỳ
    - Xây dựng LOCATION features (nếu có): one-hot encoding cho vùng miền
    - Feature engineering đặc thù cho dự báo thời tiết

Chức năng chính:
    - create_lag_features(): Tạo lag features cho time series
    - create_rolling_features(): Tạo rolling statistics
    - create_time_features(): Trích xuất time-based features
    - create_location_features(): Features theo vị trí địa lý
    - create_weather_interaction_features(): Tương tác giữa các biến thời tiết
    - build_all_features(): Pipeline tổng hợp tất cả features

Cách sử dụng:
    from Weather_Forcast_App.Machine_learning_model.features.Build_transfer import WeatherFeatureBuilder
    
    builder = WeatherFeatureBuilder()
    df_features = builder.build_all_features(df_raw, target_col='luong_mua')
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import warnings
import logging
import json

warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)


# ============================= CONSTANTS =============================

# Default lag periods cho các biến thời tiết
DEFAULT_LAG_PERIODS = [1, 2, 3, 6, 12, 24, 48, 168]  # 1h, 2h, 3h, 6h, 12h, 24h, 48h, 7days

# Default rolling windows
DEFAULT_ROLLING_WINDOWS = [3, 6, 12, 24, 48, 168]  # 3h, 6h, 12h, 24h, 48h, 7days

# Các cột thời tiết chính cần tạo features
MAIN_WEATHER_COLUMNS = [
    'nhiet_do_hien_tai', 'nhiet_do_trung_binh',
    'do_am_hien_tai', 'do_am_trung_binh',
    'ap_suat_hien_tai', 'ap_suat_trung_binh',
    'toc_do_gio_hien_tai', 'toc_do_gio_trung_binh',
    'luong_mua_hien_tai', 'tong_luong_mua',
    'do_che_phu_may_hien_tai'
]

# Mapping mùa cho Việt Nam
VIETNAM_SEASON_MAP = {
    1: 'winter', 2: 'winter', 3: 'spring',
    4: 'spring', 5: 'summer', 6: 'summer',
    7: 'summer', 8: 'summer', 9: 'autumn',
    10: 'autumn', 11: 'autumn', 12: 'winter'
}

# Vùng miền Việt Nam theo tọa độ
VIETNAM_REGIONS = {
    'north': {'lat_min': 20.0, 'lat_max': 23.5},      # Bắc Bộ
    'central': {'lat_min': 15.0, 'lat_max': 20.0},    # Trung Bộ
    'south': {'lat_min': 8.0, 'lat_max': 15.0}        # Nam Bộ
}


# ============================= WEATHER FEATURE BUILDER =============================

class WeatherFeatureBuilder:
    """
    Class xây dựng features từ raw data cho weather forecasting.
    
    Features được tạo:
        1. LAG features: Giá trị quá khứ của các biến
        2. ROLLING features: Thống kê trượt (mean, std, min, max)
        3. TIME features: Hour, day, month, season, cyclic encoding
        4. LOCATION features: Vùng miền, tỉnh thành
        5. INTERACTION features: Tương tác giữa các biến thời tiết
        6. DIFFERENCE features: Sự thay đổi giữa các thời điểm
    
    Attributes:
        config: Cấu hình cho feature building
        feature_names: Danh sách tên features đã tạo
        is_fitted: Trạng thái đã fit chưa
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Khởi tạo WeatherFeatureBuilder.
        
        Args:
            config: Dict cấu hình cho feature building
        """
        self.config = config or self._get_default_config()
        self.feature_names: List[str] = []
        self.is_fitted = False
        self._fitted_columns: List[str] = []
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Lấy cấu hình mặc định."""
        return {
            # Lag features config
            'lag_features': {
                'enabled': True,
                'periods': [1, 3, 6, 12, 24, 168],  # hours
                'columns': None  # None = tự động detect
            },
            # Rolling features config
            'rolling_features': {
                'enabled': True,
                'windows': [3, 6, 12, 24, 168],  # hours
                'functions': ['mean', 'std', 'min', 'max'],
                'columns': None  # None = tự động detect
            },
            # Time features config
            'time_features': {
                'enabled': True,
                'extract_hour': True,
                'extract_day': True,
                'extract_day_of_week': True,
                'extract_day_of_year': True,
                'extract_month': True,
                'extract_quarter': True,
                'extract_season': True,
                'cyclic_encoding': True,
                'is_weekend': True,
                'is_holiday': False  # Cần thêm calendar
            },
            # Location features config
            'location_features': {
                'enabled': True,
                'encode_region': True,
                'encode_province': True,
                'use_coordinates': True
            },
            # Weather interaction features config
            'interaction_features': {
                'enabled': True,
                'temp_humidity': True,
                'temp_wind': True,
                'pressure_humidity': True,
                'create_ratios': True,
                'create_differences': True
            },
            # Difference features config
            'difference_features': {
                'enabled': True,
                'periods': [1, 6, 24]  # hours
            },
            # Target config
            'target_column': 'luong_mua_hien_tai',
            'time_column': 'dau_thoi_gian',
            'sort_by_time': True
        }
    
    # ============================= LAG FEATURES =============================
    
    def create_lag_features(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        lag_periods: Optional[List[int]] = None,
        group_by: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Tạo LAG features - giá trị quá khứ của các biến.
        
        Lag features rất quan trọng cho time series forecasting:
        - rain(t-1): Lượng mưa 1 giờ trước
        - rain(t-24): Lượng mưa 24 giờ trước (cùng giờ hôm qua)
        - rain(t-168): Lượng mưa 7 ngày trước (cùng giờ tuần trước)
        
        Args:
            df: DataFrame input
            columns: Danh sách cột cần tạo lag (None = auto detect)
            lag_periods: Danh sách các khoảng lag [1, 3, 6, 12, 24, 168]
            group_by: Cột để group (ví dụ: 'location_ma_tram')
        
        Returns:
            DataFrame với lag features đã thêm
        """
        df_result = df.copy()
        
        # Lấy config
        lag_config = self.config.get('lag_features', {})
        if not lag_config.get('enabled', True):
            return df_result
        
        # Xác định columns
        if columns is None:
            columns = lag_config.get('columns') or self._get_numeric_weather_columns(df)
        
        # Xác định lag periods
        if lag_periods is None:
            lag_periods = lag_config.get('periods', DEFAULT_LAG_PERIODS)
        
        # Tạo lag features
        for col in columns:
            if col not in df_result.columns:
                continue
                
            for lag in lag_periods:
                lag_col_name = f'{col}_lag_{lag}h'
                
                if group_by and group_by in df_result.columns:
                    # Lag theo group (mỗi trạm quan trắc)
                    df_result[lag_col_name] = df_result.groupby(group_by)[col].shift(lag)
                else:
                    # Lag toàn bộ
                    df_result[lag_col_name] = df_result[col].shift(lag)
                
                self.feature_names.append(lag_col_name)
        
        logger.info(f"✅ Đã tạo {len(lag_periods) * len(columns)} lag features")
        return df_result
    
    # ============================= ROLLING FEATURES =============================
    
    def create_rolling_features(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        windows: Optional[List[int]] = None,
        functions: Optional[List[str]] = None,
        group_by: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Tạo ROLLING features - thống kê trượt.
        
        Rolling features giúp capture xu hướng ngắn/trung hạn:
        - mean_7days: Trung bình lượng mưa 7 ngày
        - std_24h: Độ biến động trong 24h
        - max_3h: Giá trị cực đại trong 3h gần nhất
        
        Args:
            df: DataFrame input
            columns: Danh sách cột cần tạo rolling
            windows: Danh sách window sizes [3, 6, 12, 24, 168]
            functions: Danh sách hàm thống kê ['mean', 'std', 'min', 'max']
            group_by: Cột để group
        
        Returns:
            DataFrame với rolling features đã thêm
        """
        df_result = df.copy()
        
        # Lấy config
        rolling_config = self.config.get('rolling_features', {})
        if not rolling_config.get('enabled', True):
            return df_result
        
        # Xác định columns
        if columns is None:
            columns = rolling_config.get('columns') or self._get_numeric_weather_columns(df)
        
        # Xác định windows
        if windows is None:
            windows = rolling_config.get('windows', DEFAULT_ROLLING_WINDOWS)
        
        # Xác định functions
        if functions is None:
            functions = rolling_config.get('functions', ['mean', 'std', 'min', 'max'])
        
        # Tạo rolling features
        for col in columns:
            if col not in df_result.columns:
                continue
                
            for window in windows:
                for func in functions:
                    feature_name = f'{col}_rolling_{func}_{window}h'
                    
                    if group_by and group_by in df_result.columns:
                        rolling_obj = df_result.groupby(group_by)[col].rolling(window=window, min_periods=1)
                    else:
                        rolling_obj = df_result[col].rolling(window=window, min_periods=1)
                    
                    # Áp dụng hàm thống kê
                    if func == 'mean':
                        df_result[feature_name] = rolling_obj.mean().reset_index(level=0, drop=True) if group_by else rolling_obj.mean()
                    elif func == 'std':
                        df_result[feature_name] = rolling_obj.std().reset_index(level=0, drop=True) if group_by else rolling_obj.std()
                    elif func == 'min':
                        df_result[feature_name] = rolling_obj.min().reset_index(level=0, drop=True) if group_by else rolling_obj.min()
                    elif func == 'max':
                        df_result[feature_name] = rolling_obj.max().reset_index(level=0, drop=True) if group_by else rolling_obj.max()
                    elif func == 'sum':
                        df_result[feature_name] = rolling_obj.sum().reset_index(level=0, drop=True) if group_by else rolling_obj.sum()
                    elif func == 'median':
                        df_result[feature_name] = rolling_obj.median().reset_index(level=0, drop=True) if group_by else rolling_obj.median()
                    
                    self.feature_names.append(feature_name)
        
        logger.info(f"✅ Đã tạo {len(windows) * len(columns) * len(functions)} rolling features")
        return df_result
    
    # ============================= TIME FEATURES =============================
    
    def create_time_features(
        self,
        df: pd.DataFrame,
        time_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Tạo TIME features - trích xuất từ timestamp.
        
        Time features giúp capture tính chu kỳ:
        - hour_sin, hour_cos: Cyclic encoding cho giờ (24h cycle)
        - month_sin, month_cos: Cyclic encoding cho tháng (12 month cycle)
        - day_of_week: Ngày trong tuần (0-6)
        - season: Mùa trong năm
        
        Args:
            df: DataFrame input
            time_column: Tên cột thời gian
        
        Returns:
            DataFrame với time features đã thêm
        """
        df_result = df.copy()
        
        # Lấy config
        time_config = self.config.get('time_features', {})
        if not time_config.get('enabled', True):
            return df_result
        
        # Xác định time column
        if time_column is None:
            time_column = self.config.get('time_column', 'dau_thoi_gian')
        
        # Tìm cột thời gian
        if time_column not in df_result.columns:
            time_cols = [col for col in df_result.columns 
                        if 'thoi_gian' in col.lower() or 'time' in col.lower() or 'date' in col.lower()]
            if time_cols:
                time_column = time_cols[0]
            else:
                logger.warning("Không tìm thấy cột thời gian")
                return df_result
        
        # Chuyển sang datetime
        if not pd.api.types.is_datetime64_any_dtype(df_result[time_column]):
            df_result[time_column] = pd.to_datetime(df_result[time_column], errors='coerce')
        
        dt = df_result[time_column]
        
        # Extract basic features
        if time_config.get('extract_hour', True):
            df_result['hour'] = dt.dt.hour
            self.feature_names.append('hour')
        
        if time_config.get('extract_day', True):
            df_result['day'] = dt.dt.day
            self.feature_names.append('day')
        
        if time_config.get('extract_day_of_week', True):
            df_result['day_of_week'] = dt.dt.dayofweek
            self.feature_names.append('day_of_week')
        
        if time_config.get('extract_day_of_year', True):
            df_result['day_of_year'] = dt.dt.dayofyear
            self.feature_names.append('day_of_year')
        
        if time_config.get('extract_month', True):
            df_result['month'] = dt.dt.month
            self.feature_names.append('month')
        
        if time_config.get('extract_quarter', True):
            df_result['quarter'] = dt.dt.quarter
            self.feature_names.append('quarter')
        
        if time_config.get('extract_season', True):
            df_result['season'] = dt.dt.month.map(VIETNAM_SEASON_MAP)
            self.feature_names.append('season')
        
        if time_config.get('is_weekend', True):
            df_result['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
            self.feature_names.append('is_weekend')
        
        # Cyclic encoding - Rất quan trọng cho time series!
        if time_config.get('cyclic_encoding', True):
            # Hour encoding (24h cycle)
            df_result['hour_sin'] = np.sin(2 * np.pi * df_result['hour'] / 24)
            df_result['hour_cos'] = np.cos(2 * np.pi * df_result['hour'] / 24)
            
            # Day of week encoding (7 day cycle)
            df_result['dow_sin'] = np.sin(2 * np.pi * df_result['day_of_week'] / 7)
            df_result['dow_cos'] = np.cos(2 * np.pi * df_result['day_of_week'] / 7)
            
            # Month encoding (12 month cycle)
            df_result['month_sin'] = np.sin(2 * np.pi * df_result['month'] / 12)
            df_result['month_cos'] = np.cos(2 * np.pi * df_result['month'] / 12)
            
            # Day of year encoding (365 day cycle)
            df_result['doy_sin'] = np.sin(2 * np.pi * df_result['day_of_year'] / 365)
            df_result['doy_cos'] = np.cos(2 * np.pi * df_result['day_of_year'] / 365)
            
            self.feature_names.extend([
                'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
                'month_sin', 'month_cos', 'doy_sin', 'doy_cos'
            ])
        
        logger.info(f"✅ Đã tạo time features từ cột '{time_column}'")
        return df_result
    
    # ============================= LOCATION FEATURES =============================
    
    def create_location_features(
        self,
        df: pd.DataFrame,
        lat_column: str = 'location_vi_do',
        lon_column: str = 'location_kinh_do',
        province_column: str = 'location_tinh_thanh_pho'
    ) -> pd.DataFrame:
        """
        Tạo LOCATION features - features theo vị trí địa lý.
        
        Location features giúp model hiểu sự khác biệt theo vùng:
        - region: Bắc/Trung/Nam
        - is_coastal: Có ven biển không
        - latitude_scaled: Vĩ độ chuẩn hóa
        
        Args:
            df: DataFrame input
            lat_column: Tên cột vĩ độ
            lon_column: Tên cột kinh độ
            province_column: Tên cột tỉnh/thành phố
        
        Returns:
            DataFrame với location features đã thêm
        """
        df_result = df.copy()
        
        # Lấy config
        loc_config = self.config.get('location_features', {})
        if not loc_config.get('enabled', True):
            return df_result
        
        # Encode region từ tọa độ
        if loc_config.get('encode_region', True) and lat_column in df_result.columns:
            df_result['region'] = df_result[lat_column].apply(self._get_region_from_lat)
            
            # One-hot encode region
            region_dummies = pd.get_dummies(df_result['region'], prefix='region')
            df_result = pd.concat([df_result, region_dummies], axis=1)
            
            self.feature_names.append('region')
            self.feature_names.extend(region_dummies.columns.tolist())
        
        # Use coordinates
        if loc_config.get('use_coordinates', True):
            if lat_column in df_result.columns:
                # Chuẩn hóa vĩ độ (8-24 -> 0-1)
                df_result['lat_scaled'] = (df_result[lat_column] - 8) / 16
                self.feature_names.append('lat_scaled')
            
            if lon_column in df_result.columns:
                # Chuẩn hóa kinh độ (102-110 -> 0-1)
                df_result['lon_scaled'] = (df_result[lon_column] - 102) / 8
                self.feature_names.append('lon_scaled')
            
            # Tạo interaction lat * lon
            if lat_column in df_result.columns and lon_column in df_result.columns:
                df_result['lat_lon_interaction'] = df_result['lat_scaled'] * df_result['lon_scaled']
                self.feature_names.append('lat_lon_interaction')
        
        # Encode province
        if loc_config.get('encode_province', True) and province_column in df_result.columns:
            province_dummies = pd.get_dummies(df_result[province_column], prefix='province')
            df_result = pd.concat([df_result, province_dummies], axis=1)
            self.feature_names.extend(province_dummies.columns.tolist())
        
        logger.info("✅ Đã tạo location features")
        return df_result
    
    def _get_region_from_lat(self, lat: float) -> str:
        """Xác định vùng miền từ vĩ độ."""
        if pd.isna(lat):
            return 'unknown'
        
        for region, bounds in VIETNAM_REGIONS.items():
            if bounds['lat_min'] <= lat <= bounds['lat_max']:
                return region
        return 'unknown'
    
    # ============================= INTERACTION FEATURES =============================
    
    def create_weather_interaction_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Tạo INTERACTION features - tương tác giữa các biến thời tiết.
        
        Interaction features capture mối quan hệ phi tuyến:
        - temp_humidity: Nhiệt độ * Độ ẩm (cảm giác nóng)
        - temp_wind: Nhiệt độ * Gió (wind chill)
        - pressure_change: Biến đổi áp suất (dự báo mưa)
        
        Args:
            df: DataFrame input
        
        Returns:
            DataFrame với interaction features đã thêm
        """
        df_result = df.copy()
        
        # Lấy config
        inter_config = self.config.get('interaction_features', {})
        if not inter_config.get('enabled', True):
            return df_result
        
        # Tìm các cột thời tiết
        temp_col = self._find_column(df_result, ['nhiet_do_hien_tai', 'nhiet_do_trung_binh', 'nhiet_do'])
        humidity_col = self._find_column(df_result, ['do_am_hien_tai', 'do_am_trung_binh', 'do_am'])
        wind_col = self._find_column(df_result, ['toc_do_gio_hien_tai', 'toc_do_gio_trung_binh', 'toc_do_gio'])
        pressure_col = self._find_column(df_result, ['ap_suat_hien_tai', 'ap_suat_trung_binh', 'ap_suat'])
        cloud_col = self._find_column(df_result, ['do_che_phu_may_hien_tai', 'do_che_phu_may'])
        
        # Temperature * Humidity interaction
        if inter_config.get('temp_humidity', True) and temp_col and humidity_col:
            df_result['temp_humidity_index'] = df_result[temp_col] * df_result[humidity_col] / 100
            df_result['heat_index'] = self._calculate_heat_index(df_result[temp_col], df_result[humidity_col])
            self.feature_names.extend(['temp_humidity_index', 'heat_index'])
        
        # Temperature * Wind interaction (Wind Chill)
        if inter_config.get('temp_wind', True) and temp_col and wind_col:
            df_result['temp_wind_index'] = df_result[temp_col] - (df_result[wind_col] * 0.5)
            df_result['wind_chill'] = self._calculate_wind_chill(df_result[temp_col], df_result[wind_col])
            self.feature_names.extend(['temp_wind_index', 'wind_chill'])
        
        # Pressure * Humidity interaction
        if inter_config.get('pressure_humidity', True) and pressure_col and humidity_col:
            df_result['pressure_humidity_index'] = df_result[pressure_col] * df_result[humidity_col] / 100
            self.feature_names.append('pressure_humidity_index')
        
        # Ratios
        if inter_config.get('create_ratios', True):
            # Temp range ratio
            temp_max = self._find_column(df_result, ['nhiet_do_toi_da'])
            temp_min = self._find_column(df_result, ['nhiet_do_toi_thieu'])
            if temp_max and temp_min:
                df_result['temp_range'] = df_result[temp_max] - df_result[temp_min]
                df_result['temp_range_ratio'] = df_result['temp_range'] / (df_result[temp_min] + 273.15)  # Kelvin
                self.feature_names.extend(['temp_range', 'temp_range_ratio'])
            
            # Humidity range ratio
            hum_max = self._find_column(df_result, ['do_am_toi_da'])
            hum_min = self._find_column(df_result, ['do_am_toi_thieu'])
            if hum_max and hum_min:
                df_result['humidity_range'] = df_result[hum_max] - df_result[hum_min]
                self.feature_names.append('humidity_range')
        
        # Cloud-rain relationship
        if cloud_col:
            df_result['cloud_rain_potential'] = df_result[cloud_col] / 100  # 0-1 scale
            self.feature_names.append('cloud_rain_potential')
        
        logger.info("✅ Đã tạo weather interaction features")
        return df_result
    
    def _calculate_heat_index(self, temp: pd.Series, humidity: pd.Series) -> pd.Series:
        """Tính Heat Index (Chỉ số nhiệt)."""
        # Simplified heat index formula
        return temp + 0.5 * (humidity / 100) * (temp - 14)
    
    def _calculate_wind_chill(self, temp: pd.Series, wind_speed: pd.Series) -> pd.Series:
        """Tính Wind Chill (Chỉ số gió lạnh)."""
        # Simplified wind chill formula
        return 13.12 + 0.6215 * temp - 11.37 * (wind_speed ** 0.16) + 0.3965 * temp * (wind_speed ** 0.16)
    
    # ============================= DIFFERENCE FEATURES =============================
    
    def create_difference_features(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        periods: Optional[List[int]] = None,
        group_by: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Tạo DIFFERENCE features - sự thay đổi giữa các thời điểm.
        
        Difference features capture xu hướng và biến động:
        - temp_diff_1h: Thay đổi nhiệt độ trong 1h
        - pressure_diff_6h: Thay đổi áp suất trong 6h (quan trọng cho dự báo mưa!)
        
        Args:
            df: DataFrame input
            columns: Danh sách cột cần tạo difference
            periods: Danh sách periods [1, 6, 24]
            group_by: Cột để group
        
        Returns:
            DataFrame với difference features đã thêm
        """
        df_result = df.copy()
        
        # Lấy config
        diff_config = self.config.get('difference_features', {})
        if not diff_config.get('enabled', True):
            return df_result
        
        # Xác định columns
        if columns is None:
            columns = self._get_numeric_weather_columns(df)
        
        # Xác định periods
        if periods is None:
            periods = diff_config.get('periods', [1, 6, 24])
        
        # Tạo difference features
        for col in columns:
            if col not in df_result.columns:
                continue
            
            for period in periods:
                diff_name = f'{col}_diff_{period}h'
                pct_name = f'{col}_pct_change_{period}h'
                
                if group_by and group_by in df_result.columns:
                    df_result[diff_name] = df_result.groupby(group_by)[col].diff(period)
                    df_result[pct_name] = df_result.groupby(group_by)[col].pct_change(period)
                else:
                    df_result[diff_name] = df_result[col].diff(period)
                    df_result[pct_name] = df_result[col].pct_change(period)
                
                self.feature_names.extend([diff_name, pct_name])
        
        logger.info(f"✅ Đã tạo {len(periods) * len(columns) * 2} difference features")
        return df_result
    
    # ============================= BUILD ALL FEATURES =============================
    
    def build_all_features(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        group_by: Optional[str] = None,
        drop_na: bool = False
    ) -> pd.DataFrame:
        """
        Pipeline tổng hợp - xây dựng tất cả features.
        
        Thứ tự thực hiện:
            1. Sort by time (nếu cần)
            2. Create time features
            3. Create location features
            4. Create lag features
            5. Create rolling features
            6. Create difference features
            7. Create interaction features
        
        Args:
            df: DataFrame input (raw data)
            target_column: Tên cột target
            group_by: Cột để group (ví dụ: 'location_ma_tram')
            drop_na: Drop rows có missing values
        
        Returns:
            DataFrame với tất cả features đã thêm
        """
        logger.info("🚀 Bắt đầu build features...")
        
        df_result = df.copy()
        self.feature_names = []  # Reset feature names
        
        # Xác định target column
        if target_column is None:
            target_column = self.config.get('target_column', 'luong_mua_hien_tai')
        
        # Sort by time
        time_column = self.config.get('time_column', 'dau_thoi_gian')
        if self.config.get('sort_by_time', True) and time_column in df_result.columns:
            if not pd.api.types.is_datetime64_any_dtype(df_result[time_column]):
                df_result[time_column] = pd.to_datetime(df_result[time_column], errors='coerce')
            
            if group_by and group_by in df_result.columns:
                df_result = df_result.sort_values([group_by, time_column])
            else:
                df_result = df_result.sort_values(time_column)
            
            df_result = df_result.reset_index(drop=True)
        
        # 1. Time features
        df_result = self.create_time_features(df_result, time_column)
        
        # 2. Location features
        df_result = self.create_location_features(df_result)
        
        # 3. Lag features
        df_result = self.create_lag_features(df_result, group_by=group_by)
        
        # 4. Rolling features
        df_result = self.create_rolling_features(df_result, group_by=group_by)
        
        # 5. Difference features
        df_result = self.create_difference_features(df_result, group_by=group_by)
        
        # 6. Interaction features
        df_result = self.create_weather_interaction_features(df_result)
        
        # Handle NaN
        if drop_na:
            df_result = df_result.dropna()
        else:
            # Fill NaN với median cho numeric columns
            numeric_cols = df_result.select_dtypes(include=[np.number]).columns
            df_result[numeric_cols] = df_result[numeric_cols].fillna(df_result[numeric_cols].median())
        
        self.is_fitted = True
        self._fitted_columns = df_result.columns.tolist()
        
        logger.info(f"✅ Hoàn thành! Tổng cộng {len(self.feature_names)} features mới được tạo")
        logger.info(f"📊 Shape: {df_result.shape}")
        
        return df_result
    
    # ============================= UTILITY METHODS =============================
    
    def _get_numeric_weather_columns(self, df: pd.DataFrame) -> List[str]:
        """Lấy danh sách cột numeric liên quan đến thời tiết."""
        weather_keywords = ['nhiet_do', 'do_am', 'ap_suat', 'toc_do_gio', 
                          'luong_mua', 'do_che_phu_may', 'tam_nhin']
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        weather_cols = [col for col in numeric_cols 
                       if any(kw in col.lower() for kw in weather_keywords)]
        
        return weather_cols if weather_cols else numeric_cols[:10]  # Fallback to first 10 numeric
    
    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Tìm cột đầu tiên tồn tại trong DataFrame."""
        for col in candidates:
            if col in df.columns:
                return col
        return None
    
    def get_feature_names(self) -> List[str]:
        """Lấy danh sách tên features đã tạo."""
        return self.feature_names
    
    def save_feature_list(self, path: Union[str, Path]) -> None:
        """Lưu danh sách features ra file JSON."""
        feature_info = {
            'feature_names': self.feature_names,
            'total_features': len(self.feature_names),
            'config': self.config,
            'created_at': datetime.now().isoformat()
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(feature_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Đã lưu feature list tại: {path}")
    
    def load_feature_list(self, path: Union[str, Path]) -> List[str]:
        """Load danh sách features từ file JSON."""
        with open(path, 'r', encoding='utf-8') as f:
            feature_info = json.load(f)
        
        self.feature_names = feature_info.get('feature_names', [])
        self.config = feature_info.get('config', self._get_default_config())
        
        return self.feature_names


# ============================= UTILITY FUNCTIONS =============================

def create_feature_builder(config: Optional[Dict[str, Any]] = None) -> WeatherFeatureBuilder:
    """
    Factory function để tạo WeatherFeatureBuilder.
    
    Args:
        config: Cấu hình tùy chỉnh
    
    Returns:
        WeatherFeatureBuilder instance
    """
    return WeatherFeatureBuilder(config)


def build_features_for_training(
    df: pd.DataFrame,
    target_column: str = 'luong_mua_hien_tai',
    group_by: Optional[str] = 'location_ma_tram'
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Hàm tiện ích để build features cho training.
    
    Args:
        df: DataFrame raw data
        target_column: Tên cột target
        group_by: Cột để group
    
    Returns:
        Tuple (DataFrame với features, danh sách feature names)
    """
    builder = WeatherFeatureBuilder()
    df_features = builder.build_all_features(df, target_column, group_by)
    
    return df_features, builder.get_feature_names()


def build_features_for_prediction(
    df: pd.DataFrame,
    feature_list_path: Union[str, Path],
    group_by: Optional[str] = None
) -> pd.DataFrame:
    """
    Hàm tiện ích để build features cho prediction.
    
    Sử dụng feature list đã lưu từ training để đảm bảo consistency.
    
    Args:
        df: DataFrame input data
        feature_list_path: Đường dẫn đến file feature list
        group_by: Cột để group
    
    Returns:
        DataFrame với features (chỉ giữ lại features trong list)
    """
    builder = WeatherFeatureBuilder()
    feature_names = builder.load_feature_list(feature_list_path)
    
    df_features = builder.build_all_features(df, group_by=group_by)
    
    # Chỉ giữ lại features có trong list
    available_features = [f for f in feature_names if f in df_features.columns]
    
    return df_features[available_features]


# ============================= MODULE TEST =============================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Testing WeatherFeatureBuilder")
    print("=" * 60)
    
    # Tạo sample data
    np.random.seed(42)
    n_samples = 100
    
    sample_data = pd.DataFrame({
        'dau_thoi_gian': pd.date_range('2026-01-01', periods=n_samples, freq='H'),
        'nhiet_do_hien_tai': np.random.uniform(20, 35, n_samples),
        'nhiet_do_toi_da': np.random.uniform(30, 40, n_samples),
        'nhiet_do_toi_thieu': np.random.uniform(15, 25, n_samples),
        'do_am_hien_tai': np.random.uniform(60, 95, n_samples),
        'do_am_toi_da': np.random.uniform(80, 100, n_samples),
        'do_am_toi_thieu': np.random.uniform(40, 70, n_samples),
        'ap_suat_hien_tai': np.random.uniform(1005, 1020, n_samples),
        'toc_do_gio_hien_tai': np.random.uniform(0, 15, n_samples),
        'luong_mua_hien_tai': np.random.exponential(2, n_samples),
        'do_che_phu_may_hien_tai': np.random.uniform(0, 100, n_samples),
        'location_vi_do': np.random.uniform(10, 22, n_samples),
        'location_kinh_do': np.random.uniform(104, 109, n_samples),
        'location_ma_tram': ['STATION_A'] * 50 + ['STATION_B'] * 50
    })
    
    print(f"📊 Sample data shape: {sample_data.shape}")
    
    # Test feature building
    builder = WeatherFeatureBuilder()
    df_features = builder.build_all_features(
        sample_data, 
        target_column='luong_mua_hien_tai',
        group_by='location_ma_tram'
    )
    
    print(f"\n📊 Features data shape: {df_features.shape}")
    print(f"📋 Total new features: {len(builder.get_feature_names())}")
    print(f"\n🔹 Sample features (first 20):")
    for i, feat in enumerate(builder.get_feature_names()[:20]):
        print(f"   {i+1}. {feat}")
    
    print("\n" + "=" * 60)
    print("🏁 Test hoàn thành")
    print("=" * 60)