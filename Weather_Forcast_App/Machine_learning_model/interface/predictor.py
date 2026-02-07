# ----------------------------- PREDICTOR - INTERFACE DỰ BÁO THỜI TIẾT -----------------------------------------------------------
"""
predictor.py - Module "cổng" để app Django gọi dự báo thời tiết

Mục đích:
    - Load Model.pkl đã train từ Machine_learning_artifacts/latest/
    - Load Feature_list.json để đảm bảo input có đúng feature theo thứ tự
    - Nhận input mới từ view/API
    - Build features/transform giống lúc train
    - Predict và trả kết quả cho view/API

Luồng hoạt động:
    1. Load Model.pkl (model đã train)
    2. Load Feature_list.json (danh sách features)
    3. Nhận input từ request (dict, DataFrame, hoặc list of dicts)
    4. Transform input giống lúc train
    5. Đảm bảo columns khớp với Feature_list
    6. Predict
    7. Trả về kết quả dự báo

Cách sử dụng:
    from Weather_Forcast_App.Machine_learning_model.interface.predictor import WeatherPredictor
    
    # Khởi tạo predictor (tự động load model và feature list)
    predictor = WeatherPredictor()
    
    # Dự báo với input dict
    result = predictor.predict({
        'nhiet_do_hien_tai': 28.5,
        'do_am_hien_tai': 75,
        'ap_suat_hien_tai': 1013,
        'toc_do_gio_hien_tai': 3.5,
        ...
    })
    
    # Hoặc batch prediction với DataFrame
    df_input = pd.DataFrame([...])
    results = predictor.predict(df_input)
"""

import os
import json
import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field

# Setup logging
logger = logging.getLogger(__name__)


# ============================= CONSTANTS =============================

# Đường dẫn mặc định đến artifacts
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Weather_Forcast_App
ARTIFACTS_DIR = BASE_DIR / "Machine_learning_artifacts" / "latest"
DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "Model.pkl"
DEFAULT_FEATURE_LIST_PATH = ARTIFACTS_DIR / "Feature_list.json"
DEFAULT_TRAIN_INFO_PATH = ARTIFACTS_DIR / "Train_info.json"
DEFAULT_METRICS_PATH = ARTIFACTS_DIR / "Metrics.json"


# ============================= DATA CLASSES =============================

@dataclass
class PredictionResult:
    """
    Kết quả dự báo từ model.
    
    Attributes:
        predictions: Danh sách các giá trị dự báo
        target_column: Tên cột target được dự báo
        model_name: Tên model được sử dụng
        timestamp: Thời điểm dự báo
        input_count: Số lượng input samples
        metadata: Thông tin bổ sung
    """
    predictions: List[float]
    target_column: str
    model_name: str
    timestamp: str
    input_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thành dict để trả về cho API."""
        return {
            "predictions": self.predictions,
            "target_column": self.target_column,
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "input_count": self.input_count,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """Chuyển đổi thành JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class ModelInfo:
    """
    Thông tin về model đã load.
    
    Attributes:
        name: Tên model
        version: Phiên bản
        trained_at: Thời điểm train
        target_column: Cột target
        feature_count: Số lượng features
        metrics: Các chỉ số đánh giá
    """
    name: str
    version: str
    trained_at: str
    target_column: str
    feature_count: int
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "trained_at": self.trained_at,
            "target_column": self.target_column,
            "feature_count": self.feature_count,
            "metrics": self.metrics
        }


# ============================= EXCEPTIONS =============================

class PredictorError(Exception):
    """Base exception cho Predictor."""
    pass


class ModelNotLoadedError(PredictorError):
    """Exception khi model chưa được load."""
    pass


class FeatureListNotLoadedError(PredictorError):
    """Exception khi feature list chưa được load."""
    pass


class InvalidInputError(PredictorError):
    """Exception khi input không hợp lệ."""
    pass


class MissingFeaturesError(PredictorError):
    """Exception khi thiếu features."""
    pass


# ============================= WEATHER PREDICTOR =============================

class WeatherPredictor:
    """
    Class chính để dự báo thời tiết.
    
    Đây là "cổng" (interface) để app Django gọi dự báo.
    
    Workflow:
        1. Load Model.pkl
        2. Load Feature_list.json
        3. Nhận input mới
        4. Build features/transform giống lúc train
        5. Predict
        6. Trả kết quả cho view/API
    
    Attributes:
        model: Model đã load từ pkl file
        scaler: Scaler đã fit từ training (nếu có)
        feature_list: Danh sách features theo thứ tự
        train_info: Thông tin về lần train
        metrics: Các chỉ số đánh giá model
        is_loaded: Trạng thái đã load model chưa
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        feature_list_path: Optional[Union[str, Path]] = None,
        auto_load: bool = True
    ):
        """
        Khởi tạo WeatherPredictor.
        
        Args:
            model_path: Đường dẫn đến file Model.pkl (mặc định: artifacts/latest/Model.pkl)
            feature_list_path: Đường dẫn đến file Feature_list.json
            auto_load: Tự động load model khi khởi tạo (default: True)
        """
        # Paths
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.feature_list_path = Path(feature_list_path) if feature_list_path else DEFAULT_FEATURE_LIST_PATH
        self.train_info_path = DEFAULT_TRAIN_INFO_PATH
        self.metrics_path = DEFAULT_METRICS_PATH
        
        # Model components
        self.model = None
        self.scaler = None
        self.feature_list: List[str] = []
        self.train_info: Dict[str, Any] = {}
        self.metrics: Dict[str, float] = {}
        self.target_column: str = ""
        self.model_name: str = ""
        
        # State
        self.is_loaded = False
        
        # Auto load nếu được yêu cầu
        if auto_load:
            try:
                self.load()
            except Exception as e:
                logger.warning(f"Không thể auto-load model: {e}")
    
    # ============================= LOAD METHODS =============================
    
    def load(self) -> None:
        """
        Load tất cả components cần thiết (model, feature list, train info, metrics).
        
        Raises:
            FileNotFoundError: Nếu không tìm thấy file model hoặc feature list
            PredictorError: Nếu có lỗi khi load
        """
        self.load_model()
        self.load_feature_list()
        self.load_train_info()
        self.load_metrics()
        self.is_loaded = True
        logger.info(f"✅ Predictor đã load thành công. Model: {self.model_name}")
    
    def load_model(self) -> None:
        """
        Load model từ file pkl.
        
        File Model.pkl chứa:
            - model: Model đã train (XGBoost, LightGBM, CatBoost, etc.)
            - scaler: StandardScaler/MinMaxScaler đã fit
            - params: Hyperparameters của model
        
        Raises:
            FileNotFoundError: Nếu không tìm thấy file model
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file model tại: {self.model_path}\n"
                f"Hãy chắc chắn đã train model và lưu vào thư mục artifacts."
            )
        
        try:
            model_data = joblib.load(self.model_path)
            
            # Xử lý các format khác nhau của model file
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.scaler = model_data.get('scaler')
                self.model_name = model_data.get('model_name', 'Unknown')
            else:
                # Nếu file chỉ chứa model object
                self.model = model_data
                self.model_name = type(model_data).__name__
            
            logger.info(f"✅ Đã load model: {self.model_name} từ {self.model_path}")
            
        except Exception as e:
            raise PredictorError(f"Lỗi khi load model: {e}")
    
    def load_feature_list(self) -> None:
        """
        Load danh sách features từ Feature_list.json.
        
        Feature list đảm bảo:
            - Input có đúng features theo thứ tự
            - Tránh lỗi "thiếu cột", "sai thứ tự cột"
            - Là "hợp đồng" giữa features/ và models/
        
        Raises:
            FileNotFoundError: Nếu không tìm thấy file feature list
        """
        if not self.feature_list_path.exists():
            logger.warning(
                f"Không tìm thấy Feature_list.json tại: {self.feature_list_path}\n"
                f"Sẽ sử dụng tất cả features từ input."
            )
            return
        
        try:
            with open(self.feature_list_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Xử lý các format khác nhau
            if isinstance(data, list):
                self.feature_list = data
            elif isinstance(data, dict):
                self.feature_list = data.get('features', data.get('feature_names', []))
                self.target_column = data.get('target_column', '')
            
            logger.info(f"✅ Đã load {len(self.feature_list)} features từ Feature_list.json")
            
        except json.JSONDecodeError as e:
            logger.warning(f"Lỗi parse Feature_list.json: {e}")
        except Exception as e:
            logger.warning(f"Lỗi khi load Feature_list: {e}")
    
    def load_train_info(self) -> None:
        """
        Load thông tin train từ Train_info.json.
        
        Train info chứa:
            - Dataset path
            - Train timestamp
            - Split ratio
            - Model type & hyperparameters
        """
        if not self.train_info_path.exists():
            logger.debug(f"Không tìm thấy Train_info.json tại: {self.train_info_path}")
            return
        
        try:
            with open(self.train_info_path, 'r', encoding='utf-8') as f:
                self.train_info = json.load(f)
            
            # Extract target column nếu có
            if not self.target_column:
                self.target_column = self.train_info.get('target_column', '')
            
            logger.info("✅ Đã load Train_info.json")
            
        except Exception as e:
            logger.debug(f"Không thể load Train_info: {e}")
    
    def load_metrics(self) -> None:
        """
        Load metrics từ Metrics.json.
        
        Metrics chứa các chỉ số đánh giá:
            - MAE, RMSE, MAPE, R2, etc.
        """
        if not self.metrics_path.exists():
            logger.debug(f"Không tìm thấy Metrics.json tại: {self.metrics_path}")
            return
        
        try:
            with open(self.metrics_path, 'r', encoding='utf-8') as f:
                self.metrics = json.load(f)
            
            logger.info("✅ Đã load Metrics.json")
            
        except Exception as e:
            logger.debug(f"Không thể load Metrics: {e}")
    
    # ============================= PREDICT METHODS =============================
    
    def predict(
        self,
        input_data: Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]],
        return_dict: bool = True
    ) -> Union[PredictionResult, np.ndarray]:
        """
        Dự báo với input mới.
        
        Args:
            input_data: Dữ liệu input, có thể là:
                - Dict: Một sample đơn lẻ
                - List[Dict]: Nhiều samples
                - DataFrame: Nhiều samples dạng bảng
            return_dict: Trả về PredictionResult (True) hoặc numpy array (False)
        
        Returns:
            PredictionResult hoặc numpy array chứa các giá trị dự báo
        
        Raises:
            ModelNotLoadedError: Nếu model chưa được load
            InvalidInputError: Nếu input không hợp lệ
            MissingFeaturesError: Nếu thiếu features cần thiết
        
        Example:
            >>> predictor = WeatherPredictor()
            >>> result = predictor.predict({
            ...     'nhiet_do_hien_tai': 28.5,
            ...     'do_am_hien_tai': 75,
            ...     'ap_suat_hien_tai': 1013
            ... })
            >>> print(result.predictions)
            [15.5]  # Dự báo lượng mưa (ví dụ)
        """
        # Kiểm tra model đã load chưa
        if self.model is None:
            raise ModelNotLoadedError(
                "Model chưa được load! Gọi load() hoặc load_model() trước."
            )
        
        # Chuyển đổi input thành DataFrame
        df_input = self._convert_input_to_dataframe(input_data)
        
        # Transform features
        df_transformed = self._transform_features(df_input)
        
        # Đảm bảo columns khớp với feature list
        df_aligned = self._align_features(df_transformed)
        
        # Scale nếu có scaler
        X = self._scale_features(df_aligned)
        
        # Predict
        predictions = self._predict_internal(X)
        
        # Trả về kết quả
        if return_dict:
            return PredictionResult(
                predictions=predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
                target_column=self.target_column or "unknown",
                model_name=self.model_name,
                timestamp=datetime.now().isoformat(),
                input_count=len(df_input),
                metadata={
                    "model_metrics": self.metrics,
                    "feature_count": len(self.feature_list) if self.feature_list else df_aligned.shape[1]
                }
            )
        else:
            return predictions
    
    def predict_proba(
        self,
        input_data: Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]]
    ) -> Optional[np.ndarray]:
        """
        Dự báo xác suất (chỉ cho classification models).
        
        Args:
            input_data: Dữ liệu input
        
        Returns:
            numpy array chứa xác suất cho mỗi class, hoặc None nếu model không hỗ trợ
        """
        if self.model is None:
            raise ModelNotLoadedError("Model chưa được load!")
        
        if not hasattr(self.model, 'predict_proba'):
            logger.warning("Model không hỗ trợ predict_proba (có thể là regression model)")
            return None
        
        df_input = self._convert_input_to_dataframe(input_data)
        df_transformed = self._transform_features(df_input)
        df_aligned = self._align_features(df_transformed)
        X = self._scale_features(df_aligned)
        
        return self.model.predict_proba(X)
    
    # ============================= INTERNAL METHODS =============================
    
    def _convert_input_to_dataframe(
        self,
        input_data: Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]]
    ) -> pd.DataFrame:
        """
        Chuyển đổi input thành DataFrame.
        
        Args:
            input_data: Dict, List[Dict], hoặc DataFrame
        
        Returns:
            pandas DataFrame
        
        Raises:
            InvalidInputError: Nếu input không hợp lệ
        """
        if isinstance(input_data, pd.DataFrame):
            return input_data.copy()
        
        if isinstance(input_data, dict):
            return pd.DataFrame([input_data])
        
        if isinstance(input_data, list):
            if not input_data:
                raise InvalidInputError("Input list rỗng!")
            if isinstance(input_data[0], dict):
                return pd.DataFrame(input_data)
        
        raise InvalidInputError(
            f"Input type không được hỗ trợ: {type(input_data)}. "
            f"Hãy sử dụng Dict, List[Dict], hoặc DataFrame."
        )
    
    def _transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features giống lúc train.
        
        Bao gồm:
            - Extract time features (hour, month, season, cyclic encoding)
            - Create weather features (interactions, ratios)
            - Handle missing values
        
        Args:
            df: DataFrame input
        
        Returns:
            DataFrame đã transform
        """
        df_processed = df.copy()
        
        # 1. Extract time features
        df_processed = self._extract_time_features(df_processed)
        
        # 2. Create weather-specific features
        df_processed = self._create_weather_features(df_processed)
        
        # 3. Handle missing values
        df_processed = self._handle_missing_values(df_processed)
        
        # 4. Drop non-numeric columns (datetime, object) trước khi predict
        datetime_cols = df_processed.select_dtypes(include=['datetime64']).columns
        object_cols = df_processed.select_dtypes(include=['object']).columns
        cols_to_drop = list(datetime_cols) + list(object_cols)
        
        if cols_to_drop:
            df_processed = df_processed.drop(columns=cols_to_drop, errors='ignore')
        
        return df_processed
    
    def _extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trích xuất features từ thời gian."""
        # Tìm cột thời gian
        time_cols = [col for col in df.columns 
                     if 'thoi_gian' in col.lower() or 'time' in col.lower() or 'date' in col.lower()]
        
        if not time_cols:
            return df
        
        time_col = time_cols[0]
        
        # Chuyển đổi sang datetime
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        
        # Skip nếu không parse được datetime
        if df[time_col].isna().all():
            return df
        
        # Extract features
        df['hour'] = df[time_col].dt.hour.fillna(12).astype(int)
        df['day_of_year'] = df[time_col].dt.dayofyear.fillna(1).astype(int)
        df['month'] = df[time_col].dt.month.fillna(1).astype(int)
        
        # Season mapping (Vietnam)
        season_map = {
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'autumn', 10: 'autumn', 11: 'autumn'
        }
        df['season'] = df['month'].map(season_map)
        
        # Cyclic encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def _create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tạo weather-specific features."""
        # Tìm các cột thời tiết
        temp_cols = [col for col in df.columns if 'nhiet_do' in col.lower()]
        humidity_cols = [col for col in df.columns if 'do_am' in col.lower()]
        wind_cols = [col for col in df.columns if 'toc_do_gio' in col.lower()]
        
        # Interactions
        if temp_cols and humidity_cols:
            df['temp_humidity_interaction'] = df[temp_cols[0]] * df[humidity_cols[0]]
        
        if temp_cols and wind_cols:
            df['temp_wind_interaction'] = df[temp_cols[0]] * df[wind_cols[0]]
        
        # Ranges
        if len(temp_cols) >= 2:
            df['temp_range'] = df[temp_cols[1]] - df[temp_cols[0]]
            df['temp_range_ratio'] = (df[temp_cols[1]] - df[temp_cols[0]]) / (df[temp_cols[0]] + 1e-6)
        
        if len(humidity_cols) >= 2:
            df['humidity_range'] = df[humidity_cols[1]] - df[humidity_cols[0]]
            df['humidity_range_ratio'] = (df[humidity_cols[1]] - df[humidity_cols[0]]) / (df[humidity_cols[0]] + 1e-6)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Xử lý missing values."""
        # Fill numerical với mean
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mean())
        
        return df
    
    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Đảm bảo columns khớp với feature list.
        
        Args:
            df: DataFrame đã transform
        
        Returns:
            DataFrame với columns đúng thứ tự như feature list
        
        Raises:
            MissingFeaturesError: Nếu thiếu features quan trọng
        """
        if not self.feature_list:
            # Không có feature list, trả về tất cả numeric columns
            return df.select_dtypes(include=[np.number])
        
        # Tìm features bị thiếu
        missing_features = set(self.feature_list) - set(df.columns)
        
        if missing_features:
            logger.warning(f"⚠️ Thiếu {len(missing_features)} features: {list(missing_features)[:5]}...")
            # Thêm các columns thiếu với giá trị 0
            for feat in missing_features:
                df[feat] = 0
        
        # Sắp xếp columns theo feature list
        try:
            df_aligned = df[self.feature_list]
        except KeyError as e:
            raise MissingFeaturesError(f"Không thể align features: {e}")
        
        return df_aligned
    
    def _scale_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Scale features nếu có scaler.
        
        Args:
            df: DataFrame đã align
        
        Returns:
            numpy array đã scale (hoặc chưa scale nếu không có scaler)
        """
        X = df.values
        
        if self.scaler is not None:
            try:
                X = self.scaler.transform(X)
            except Exception as e:
                logger.warning(f"Không thể scale features: {e}. Sử dụng raw values.")
        
        return X
    
    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        """
        Gọi model.predict() internal.
        
        Xử lý các loại model khác nhau:
            - XGBoost: cần DMatrix
            - Sklearn models: gọi predict() trực tiếp
            - Others
        """
        try:
            # Kiểm tra nếu là XGBoost native model
            if hasattr(self.model, 'get_score'):  # XGBoost native
                import xgboost as xgb
                dmatrix = xgb.DMatrix(X)
                return self.model.predict(dmatrix)
            else:
                # Sklearn-like interface
                return self.model.predict(X)
                
        except Exception as e:
            raise PredictorError(f"Lỗi khi predict: {e}")
    
    # ============================= UTILITY METHODS =============================
    
    def get_model_info(self) -> ModelInfo:
        """
        Lấy thông tin về model đã load.
        
        Returns:
            ModelInfo object
        """
        return ModelInfo(
            name=self.model_name,
            version=self.train_info.get('version', '1.0.0'),
            trained_at=self.train_info.get('trained_at', 'Unknown'),
            target_column=self.target_column,
            feature_count=len(self.feature_list) if self.feature_list else 0,
            metrics=self.metrics
        )
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Lấy feature importance từ model.
        
        Returns:
            Dict mapping feature name -> importance score
        """
        if self.model is None:
            return None
        
        importance = None
        
        # XGBoost native
        if hasattr(self.model, 'get_score'):
            importance = self.model.get_score(importance_type='gain')
        
        # Sklearn-like (feature_importances_)
        elif hasattr(self.model, 'feature_importances_'):
            if self.feature_list:
                importance = dict(zip(self.feature_list, self.model.feature_importances_))
            else:
                importance = {f'feature_{i}': v for i, v in enumerate(self.model.feature_importances_)}
        
        # Coefficients (Linear models)
        elif hasattr(self.model, 'coef_'):
            coef = self.model.coef_
            if len(coef.shape) == 1:
                if self.feature_list:
                    importance = dict(zip(self.feature_list, np.abs(coef)))
                else:
                    importance = {f'feature_{i}': v for i, v in enumerate(np.abs(coef))}
        
        return importance
    
    def health_check(self) -> Dict[str, Any]:
        """
        Kiểm tra trạng thái của predictor.
        
        Returns:
            Dict chứa thông tin health check
        """
        return {
            "status": "healthy" if self.is_loaded else "not_ready",
            "model_loaded": self.model is not None,
            "model_name": self.model_name,
            "feature_list_loaded": len(self.feature_list) > 0,
            "feature_count": len(self.feature_list),
            "scaler_loaded": self.scaler is not None,
            "target_column": self.target_column,
            "artifacts_path": str(ARTIFACTS_DIR),
            "model_path_exists": self.model_path.exists(),
            "feature_list_path_exists": self.feature_list_path.exists()
        }
    
    def reload(self) -> None:
        """Reload tất cả components."""
        self.is_loaded = False
        self.load()


# ============================= SINGLETON INSTANCE =============================

# Global instance để reuse trong app
_predictor_instance: Optional[WeatherPredictor] = None


def get_predictor(force_reload: bool = False) -> WeatherPredictor:
    """
    Lấy singleton instance của WeatherPredictor.
    
    Sử dụng singleton để:
        - Tránh load model nhiều lần (tốn memory & thời gian)
        - Dễ dàng sử dụng trong views/API
    
    Args:
        force_reload: Force reload model từ file
    
    Returns:
        WeatherPredictor instance
    
    Example:
        from Weather_Forcast_App.Machine_learning_model.interface.predictor import get_predictor
        
        predictor = get_predictor()
        result = predictor.predict(input_data)
    """
    global _predictor_instance
    
    if _predictor_instance is None or force_reload:
        _predictor_instance = WeatherPredictor(auto_load=True)
    
    return _predictor_instance


# ============================= QUICK PREDICTION FUNCTION =============================

def predict_weather(
    input_data: Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]],
    model_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Hàm tiện ích để dự báo nhanh.
    
    Sử dụng cho trường hợp đơn giản, không cần quản lý predictor instance.
    
    Args:
        input_data: Dữ liệu input
        model_path: Đường dẫn model (optional, dùng default nếu không có)
    
    Returns:
        Dict chứa kết quả dự báo
    
    Example:
        from Weather_Forcast_App.Machine_learning_model.interface.predictor import predict_weather
        
        result = predict_weather({
            'nhiet_do_hien_tai': 28.5,
            'do_am_hien_tai': 75,
            'ap_suat_hien_tai': 1013
        })
        
        print(result['predictions'])
    """
    if model_path:
        predictor = WeatherPredictor(model_path=model_path, auto_load=True)
    else:
        predictor = get_predictor()
    
    result = predictor.predict(input_data, return_dict=True)
    return result.to_dict()


# ============================= MODULE TEST =============================

if __name__ == "__main__":
    # Test basic functionality
    print("=" * 60)
    print("🧪 Testing WeatherPredictor")
    print("=" * 60)
    
    # Khởi tạo predictor
    try:
        predictor = WeatherPredictor(auto_load=False)
        print(f"✅ Predictor khởi tạo thành công")
        
        # Health check
        health = predictor.health_check()
        print(f"\n📊 Health Check:")
        for key, value in health.items():
            print(f"   {key}: {value}")
        
        # Thử load model (sẽ fail nếu chưa có model file)
        print(f"\n📂 Đang thử load model từ: {predictor.model_path}")
        if predictor.model_path.exists():
            predictor.load()
            print(f"✅ Model loaded: {predictor.model_name}")
            
            # Test prediction với dummy data
            test_input = {
                'nhiet_do_hien_tai': 28.5,
                'do_am_hien_tai': 75,
                'ap_suat_hien_tai': 1013,
                'toc_do_gio_hien_tai': 3.5,
            }
            
            print(f"\n🔮 Testing prediction với dummy data...")
            try:
                result = predictor.predict(test_input)
                print(f"✅ Prediction result: {result.predictions}")
            except Exception as e:
                print(f"⚠️ Prediction failed (có thể do thiếu features): {e}")
        else:
            print(f"⚠️ Model file không tồn tại. Hãy train model trước.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("🏁 Test hoàn thành")
    print("=" * 60)
