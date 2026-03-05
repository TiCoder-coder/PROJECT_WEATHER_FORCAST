# ----------------------------- CATBOOST MODEL - MÔ HÌNH GRADIENT BOOSTING -----------------------------------------------------------
"""
CatBoost_Model.py - Module triển khai thuật toán CatBoost cho dự báo thời tiết

Mục đích:
    - Cung cấp wrapper class cho CatBoost algorithm
    - Hỗ trợ Classification, Regression và Ranking
    - Xử lý tự động categorical features
    - Tích hợp với pipeline dữ liệu thời tiết của project

Đặc điểm nổi bật của CatBoost:
    - Xử lý categorical features tự động (không cần encoding thủ công)
    - Ordered Boosting chống overfitting hiệu quả
    - Symmetric Decision Trees cho inference nhanh
    - Hỗ trợ GPU training
    - Ít cần hyperparameter tuning

Cách sử dụng:
    from Weather_Forcast_App.Models import WeatherCatBoost
    
    # Classification - Dự đoán loại thời tiết
    model = WeatherCatBoost(task_type='classification')
    model.train(X_train, y_train, cat_features=['season', 'location'])
    predictions = model.predict(X_test)
    
    # Regression - Dự đoán nhiệt độ
    model = WeatherCatBoost(task_type='regression')
    model.train(X_train, y_train)
    predictions = model.predict(X_test)
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

import numpy as np
import pandas as pd

# CatBoost imports
try:
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

# Setup logging
from Weather_Forcast_App.Machine_learning_model.Models import (
    TaskType,
    ModelStatus,
    TrainingResult,
    PredictionResult,
)

logger = logging.getLogger(__name__)


class LossFunction(Enum):
    """Các loss functions hỗ trợ."""
    # Classification
    LOGLOSS = 'Logloss'             # Binary classification
    CROSS_ENTROPY = 'CrossEntropy'   # Binary với probabilities
    MULTI_CLASS = 'MultiClass'       # Multi-class classification
    
    # Regression
    RMSE = 'RMSE'                    # Root Mean Squared Error
    MAE = 'MAE'                      # Mean Absolute Error
    MAPE = 'MAPE'                    # Mean Absolute Percentage Error
    QUANTILE = 'Quantile'            # Quantile regression
    
    # Ranking
    YETI_RANK = 'YetiRank'


# Default hyperparameters
DEFAULT_PARAMS = {
    'iterations': 1000,          # Số vòng lặp boosting
    'depth': 6,                  # Độ sâu tối đa của cây
    'learning_rate': 0.03,       # Tốc độ học
    'l2_leaf_reg': 3.0,          # L2 regularization
    'random_seed': 42,           # Seed
    'verbose': 100,              # In log mỗi 100 iterations
    'early_stopping_rounds': 50, # Early stopping
    'thread_count': -1,          # Sử dụng tất cả CPU
}

# Model save directory
MODEL_DIR = Path(__file__).parent.parent / 'ml_models'


@dataclass 
class ModelConfig:
    """
    Cấu hình model.
    """
    task_type: TaskType
    loss_function: str
    params: Dict[str, Any] = field(default_factory=lambda: DEFAULT_PARAMS.copy())
    cat_features: List[Union[int, str]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


# ============================= MAIN CLASS =============================

class WeatherCatBoost:
    """
    CatBoost Model cho dự báo thời tiết.
    
    CatBoost (Categorical Boosting) là thuật toán Gradient Boosting được phát triển
    bởi Yandex, đặc biệt mạnh trong việc xử lý categorical features.
    
    Tính năng:
        - Xử lý categorical features tự động
        - Ordered Boosting chống overfitting
        - Hỗ trợ GPU training
        - Early stopping
        - Feature importance analysis
        - Save/Load model
    
    Ưu điểm so với XGBoost/LightGBM:
        - Không cần encoding categorical features
        - Ít cần hyperparameter tuning
        - Chống overfitting tốt hơn
    
    Example:
        >>> # Classification với categorical features
        >>> model = WeatherCatBoost(task_type='classification')
        >>> model.train(
        ...     X_train, y_train,
        ...     cat_features=['season', 'location', 'weather_type']
        ... )
        >>> predictions = model.predict(X_test)
        >>> 
        >>> # Regression
        >>> model = WeatherCatBoost(task_type='regression', loss_function='MAE')
        >>> model.train(X_train, y_train)
        >>> temp_pred = model.predict(X_test)
    """
    
    def __init__(
        self,
        task_type: str = 'classification',
        loss_function: Optional[str] = None,
        use_gpu: bool = False,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Khởi tạo CatBoost Model.
        
        Args:
            task_type: 'classification', 'regression', hoặc 'ranking'
            loss_function: Hàm mất mát. None = auto select
            use_gpu: Sử dụng GPU training
            params: Dict hyperparameters (ưu tiên hơn kwargs). Hỗ trợ cả
                    CatBoost-native (iterations, depth) và sklearn-style
                    (n_estimators, max_depth).
            **kwargs: Các hyperparameters tùy chỉnh
                - iterations: Số vòng lặp (default=1000)
                - depth: Độ sâu cây (default=6)
                - learning_rate: Tốc độ học (default=0.03)
                - l2_leaf_reg: L2 regularization (default=3.0)
        """
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not installed. Run: pip install catboost")
        
        # ---- Handle params dict (from Ensemble or direct callers) ----
        if params and isinstance(params, dict):
            params = dict(params)  # copy to avoid mutating caller's dict
            # Extract named args from params dict
            task_type = params.pop('task_type', task_type)
            loss_function = params.pop('loss_function', loss_function)
            # Map sklearn-style param names → CatBoost-native names
            _PARAM_MAP = {
                'n_estimators': 'iterations',
                'max_depth': 'depth',
                'random_state': 'random_seed',
                'reg_alpha': 'l2_leaf_reg',  # approximate mapping
            }
            for old_key, new_key in _PARAM_MAP.items():
                if old_key in params and new_key not in params:
                    params[new_key] = params.pop(old_key)
                elif old_key in params:
                    params.pop(old_key)  # remove duplicate
            # Remove keys not understood by CatBoost
            for bad_key in ('reg_lambda', 'subsample', 'colsample_bytree',
                            'n_jobs', 'objective'):
                params.pop(bad_key, None)
            kwargs.update(params)
        
        # Validate task type
        try:
            self.task_type = TaskType(task_type.lower())
        except ValueError:
            raise ValueError(f"Invalid task_type: {task_type}")
        
        # Auto select loss function
        if loss_function is None:
            loss_function = self._default_loss_function()
        self.loss_function = loss_function
        
        # Merge params
        self.params = DEFAULT_PARAMS.copy()
        self.params['loss_function'] = loss_function
        self.params.update(kwargs)
        
        # GPU config
        if use_gpu:
            self.params['task_type'] = 'GPU'
            self.params['devices'] = '0'
        
        # Initialize model
        self.model = None
        self._init_model()
        
        # Preprocessors
        self.label_encoder = LabelEncoder()
        self._is_target_encoded = False
        
        # Feature info
        self.feature_names: List[str] = []
        self.cat_features: List[Union[int, str]] = []
        self.target_classes: Optional[np.ndarray] = None
        
        # Config & Status
        self.config = ModelConfig(
            task_type=self.task_type,
            loss_function=loss_function,
            params=self.params
        )
        self.status = ModelStatus.UNTRAINED
        
        # Training history
        self.training_history: List[TrainingResult] = []
        
        logger.info(f"Initialized WeatherCatBoost ({self.task_type.value}, loss={loss_function})")
    
    def _default_loss_function(self) -> str:
        """Chọn loss function mặc định theo task type."""
        defaults = {
            TaskType.CLASSIFICATION: 'Logloss',
            TaskType.REGRESSION: 'RMSE',
            TaskType.RANKING: 'YetiRank'
        }
        return defaults[self.task_type]
    
    def _init_model(self):
        """Khởi tạo CatBoost model dựa trên task type."""
        if self.task_type == TaskType.CLASSIFICATION:
            self.model = CatBoostClassifier(**self.params)
        elif self.task_type == TaskType.REGRESSION:
            self.model = CatBoostRegressor(**self.params)
        else:
            self.model = CatBoostRegressor(**self.params)
    
    # ============================= TRAINING METHODS =============================
    
    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        cat_features: Optional[List[Union[int, str]]] = None,
        validation_split: float = 0.2,
        plot: bool = False,
        verbose: bool = True,
        # Accept X_val/y_val/val_size from train.py (same interface as XGBoost/LightGBM)
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        val_size: float = 0.2,
        sample_weight=None,  # accepted but unused (CatBoost handles via Pool)
    ) -> TrainingResult:
        """
        Huấn luyện model.
        
        Args:
            X: Features (DataFrame hoặc ndarray)
            y: Target values
            cat_features: List các cột categorical (index hoặc tên)
            validation_split: Tỷ lệ validation set
            plot: Vẽ learning curves
            verbose: In thông tin
            
        Returns:
            TrainingResult với metrics và thông tin
            
        Example:
            >>> model = WeatherCatBoost(task_type='classification')
            >>> result = model.train(
            ...     X_train, y_train,
            ...     cat_features=['season', 'location', 'weather_type'],
            ...     validation_split=0.2
            ... )
            >>> print(f"Accuracy: {result.metrics['accuracy']:.4f}")
        """
        start_time = datetime.now()
        
        try:
            # Prepare data
            X_prepared, feature_names = self._prepare_features(X)
            y_prepared = self._prepare_target(y)
            
            self.feature_names = feature_names
            
            # Handle categorical features
            if cat_features is not None:
                self.cat_features = self._resolve_cat_features(cat_features, feature_names)
            else:
                self.cat_features = self._detect_cat_features(X_prepared)

            # Split data: prefer externally-provided val set (time-series safe)
            if X_val is not None and y_val is not None:
                X_train = X_prepared
                y_train = y_prepared
                X_val_prep, _ = self._prepare_features(X_val)
                y_val_prep = self._prepare_target(y_val)
            else:
                eff_split = float(val_size) if 0.0 < float(val_size) < 1.0 else float(validation_split)
                if 0.0 < eff_split < 1.0:
                    X_train, X_val_prep, y_train, y_val_prep = train_test_split(
                        X_prepared, y_prepared,
                        test_size=eff_split,
                        random_state=self.params['random_seed']
                    )
                else:
                    X_train, y_train = X_prepared, y_prepared
                    X_val_prep, y_val_prep = X_prepared, y_prepared
            
            # Create CatBoost Pools
            train_pool = Pool(
                X_train,
                y_train,
                cat_features=self.cat_features if self.cat_features else None,
                feature_names=feature_names
            )

            val_pool = Pool(
                X_val_prep,
                y_val_prep,
                cat_features=self.cat_features if self.cat_features else None,
                feature_names=feature_names
            )
            
            if verbose:
                print(f"🐱 Training CatBoost ({self.task_type.value})...")
                print(f"   📊 Training samples: {len(X_train)}")
                print(f"   📊 Validation samples: {len(X_val)}")
                print(f"   📊 Features: {len(feature_names)}")
                print(f"   📊 Categorical features: {len(self.cat_features)}")
                print(f"   🌳 Iterations: {self.params['iterations']}")
                print(f"   📉 Loss function: {self.loss_function}")
            
            # Train
            if validation_split and validation_split > 0:
                self.model.fit(
                    train_pool,
                    eval_set=val_pool,
                    plot=plot,
                    verbose=self.params.get('verbose', 100) if verbose else 0
                )
            else:
                self.model.fit(
                    train_pool,
                    plot=plot,
                    verbose=self.params.get('verbose', 100) if verbose else 0
                )
            
            # Evaluate
            y_pred = self.model.predict(X_val_prep)
            
            # Handle classification predictions
            if self.task_type == TaskType.CLASSIFICATION:
                if len(y_pred.shape) > 1:
                    y_pred = y_pred.argmax(axis=1)
            
            metrics = self._calculate_metrics(y_val, y_pred)
            
            # Feature importance
            feature_importances = dict(zip(
                self.feature_names,
                self.model.get_feature_importance().tolist()
            ))
            
            # Best iteration
            best_iteration = self.model.get_best_iteration()
            
            # Training time
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Update status
            self.status = ModelStatus.TRAINED
            self.config.cat_features = self.cat_features
            self.config.updated_at = datetime.now()
            
            # Create result
            result = TrainingResult(
                success=True,
                metrics=metrics,
                training_time=training_time,
                best_iteration=best_iteration,
                n_samples=len(X_prepared),
                n_features=len(feature_names),
                feature_names=self.feature_names,
                feature_importances=feature_importances,
                message="Training completed successfully"
            )
            
            self.training_history.append(result)
            
            if verbose:
                print(f"\n✅ Training completed in {training_time:.2f}s")
                print(f"   Best iteration: {best_iteration}")
                self._print_metrics(metrics)
            
            return result
            
        except Exception as e:
            self.status = ModelStatus.FAILED
            logger.error(f"Training failed: {e}")
            return TrainingResult(
                success=False,
                message=f"Training failed: {str(e)}"
            )
    
    def cross_validate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        cat_features: Optional[List[Union[int, str]]] = None,
        cv: int = 5
    ) -> Dict[str, Any]:
        """
        Cross-validation.
        
        Args:
            X: Features
            y: Target
            cat_features: Categorical features
            cv: Số folds
            
        Returns:
            Dict với mean và std scores
        """
        X_prepared, feature_names = self._prepare_features(X)
        y_prepared = self._prepare_target(y)
        
        if cat_features:
            self.cat_features = self._resolve_cat_features(cat_features, feature_names)
        
        pool = Pool(
            X_prepared, y_prepared,
            cat_features=self.cat_features if self.cat_features else None,
            feature_names=feature_names
        )
        
        from catboost import cv as catboost_cv
        
        cv_results = catboost_cv(
            pool=pool,
            params=self.params,
            fold_count=cv,
            verbose=False
        )
        
        return {
            'cv_folds': cv,
            'results': cv_results.to_dict(),
            'best_iteration': len(cv_results) - 1
        }
    
    def tune_hyperparameters(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        cat_features: Optional[List[Union[int, str]]] = None,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        cv: int = 3,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Tìm hyperparameters tối ưu bằng Grid Search.
        
        Args:
            X: Features
            y: Target
            cat_features: Categorical features
            param_grid: Dict tham số và giá trị thử
            cv: Số folds
            verbose: In thông tin
            
        Returns:
            Dict với best params
        """
        if param_grid is None:
            param_grid = {
                'depth': [4, 6, 8],
                'learning_rate': [0.01, 0.03, 0.1],
                'iterations': [500, 1000],
                'l2_leaf_reg': [1, 3, 5]
            }
        
        X_prepared, feature_names = self._prepare_features(X)
        y_prepared = self._prepare_target(y)
        
        if cat_features:
            self.cat_features = self._resolve_cat_features(cat_features, feature_names)
        
        if verbose:
            print("🔍 Tuning CatBoost hyperparameters...")
        
        pool = Pool(
            X_prepared, y_prepared,
            cat_features=self.cat_features if self.cat_features else None
        )
        
        grid_result = self.model.grid_search(
            param_grid,
            pool,
            cv=cv,
            verbose=verbose
        )
        
        # Update model with best params
        best_params = grid_result['params']
        self.params.update(best_params)
        self._init_model()
        
        if verbose:
            print(f"\n✅ Best params: {best_params}")
        
        return {
            'best_params': best_params,
            'cv_results': grid_result.get('cv_results', {})
        }
    
    # ============================= PREDICTION METHODS =============================
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        return_proba: bool = False
    ) -> PredictionResult:
        """
        Dự đoán với dữ liệu mới.
        
        Args:
            X: Features
            return_proba: Trả về probabilities (classification)
            
        Returns:
            PredictionResult
        """
        if self.status != ModelStatus.TRAINED:
            raise ValueError("Model chưa được train")
        
        start_time = datetime.now()
        
        X_prepared, _ = self._prepare_features(X)
        
        # Create Pool
        pool = Pool(
            X_prepared,
            cat_features=self.cat_features if self.cat_features else None
        )
        
        # Predict
        predictions = self.model.predict(pool)
        
        # Handle classification
        if self.task_type == TaskType.CLASSIFICATION:
            if len(predictions.shape) > 1:
                predictions = predictions.argmax(axis=1)
            
            # Decode labels
            if self._is_target_encoded:
                predictions = self.label_encoder.inverse_transform(predictions.astype(int))
        
        # Probabilities
        probabilities = None
        if return_proba and self.task_type == TaskType.CLASSIFICATION:
            probabilities = self.model.predict_proba(pool)
        
        timestamp = (datetime.now() - start_time).total_seconds()
        
        return PredictionResult(
            predictions=predictions,
            probabilities=probabilities,
            timestamp=timestamp
        )
    
    def predict_single(self, sample: Dict[str, Any]) -> Any:
        """
        Dự đoán cho một sample.
        
        Args:
            sample: Dict feature names và values
            
        Returns:
            Dự đoán
        """
        df = pd.DataFrame([sample])
        result = self.predict(df)
        return result.predictions[0]
    
    # ============================= EVALUATION METHODS =============================
    
    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Đánh giá model trên test set.
        """
        result = self.predict(X)
        y_prepared = self._prepare_target(y)
        
        # Align predictions
        predictions = result.predictions
        if self._is_target_encoded and self.task_type == TaskType.CLASSIFICATION:
            predictions = self.label_encoder.transform(predictions)
        
        metrics = self._calculate_metrics(y_prepared, predictions)
        
        if self.task_type == TaskType.CLASSIFICATION:
            metrics['confusion_matrix'] = confusion_matrix(y_prepared, predictions).tolist()
        
        return metrics
    
    def get_feature_importance(
        self,
        importance_type: str = 'FeatureImportance',
        top_n: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Lấy feature importance.
        
        Args:
            importance_type: 'FeatureImportance', 'ShapValues', 'PredictionValuesChange'
            top_n: Số features
            
        Returns:
            Dict feature name và importance
        """
        if self.status != ModelStatus.TRAINED:
            raise ValueError("Model chưa được train")
        
        importances = dict(zip(
            self.feature_names,
            self.model.get_feature_importance(type=importance_type)
        ))
        
        # Sort
        importances = dict(sorted(
            importances.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        if top_n:
            importances = dict(list(importances.items())[:top_n])
        
        return importances
    
    def get_shap_values(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Tính SHAP values để giải thích model.
        
        Args:
            X: Features
            
        Returns:
            SHAP values array
        """
        if self.status != ModelStatus.TRAINED:
            raise ValueError("Model chưa được train")
        
        X_prepared, _ = self._prepare_features(X)
        pool = Pool(X_prepared, cat_features=self.cat_features if self.cat_features else None)
        
        return self.model.get_feature_importance(data=pool, type='ShapValues')
    
    # ============================= SAVE/LOAD METHODS =============================
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Lưu model ra file.
        
        Args:
            filepath: Đường dẫn. None = auto generate
            
        Returns:
            Đường dẫn file
        """
        if self.status != ModelStatus.TRAINED:
            raise ValueError("Model chưa được train")
        
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"catboost_{self.task_type.value}_{timestamp}.cbm"
            filepath = str(MODEL_DIR / filename)
        
        # Save CatBoost model
        self.model.save_model(filepath)
        
        # Save metadata
        metadata = {
            'task_type': self.task_type.value,
            'loss_function': self.loss_function,
            'params': self.params,
            'feature_names': self.feature_names,
            'cat_features': self.cat_features,
            'target_classes': self.target_classes.tolist() if self.target_classes is not None else None,
            '_is_target_encoded': self._is_target_encoded,
            'label_encoder_classes': self.label_encoder.classes_.tolist() if self._is_target_encoded else None,
            'created_at': self.config.created_at.isoformat(),
            'updated_at': self.config.updated_at.isoformat(),
        }
        
        metadata_path = filepath.replace('.cbm', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to: {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> 'WeatherCatBoost':
        """
        Load model từ file.
        
        Args:
            filepath: Đường dẫn file .cbm
            
        Returns:
            WeatherCatBoost instance
        """
        # Load metadata
        metadata_path = filepath.replace('.cbm', '_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create instance
        instance = cls(
            task_type=metadata['task_type'],
            loss_function=metadata['loss_function']
        )
        
        # Load CatBoost model
        instance.model.load_model(filepath)
        
        # Restore attributes
        instance.feature_names = metadata['feature_names']
        instance.cat_features = metadata['cat_features']
        instance._is_target_encoded = metadata['_is_target_encoded']
        
        if metadata.get('label_encoder_classes'):
            instance.label_encoder.classes_ = np.array(metadata['label_encoder_classes'])
        
        if metadata.get('target_classes'):
            instance.target_classes = np.array(metadata['target_classes'])
        
        instance.status = ModelStatus.TRAINED
        
        logger.info(f"Model loaded from: {filepath}")
        return instance
    
    # ============================= PRIVATE METHODS =============================
    
    def _prepare_features(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Chuẩn bị features."""
        if isinstance(X, np.ndarray):
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=feature_names)
        else:
            feature_names = X.columns.tolist()
            X = X.copy()
        
        # Convert StringDtype and other extended dtypes to Python native types
        # This fixes compatibility with CatBoost and pandas 3.0
        for col in X.columns:
            dtype = X[col].dtype
            dtype_name = str(dtype).lower()
            
            # Handle string dtypes (StringDtype, ArrowDtype[string], etc.)
            if 'string' in dtype_name or dtype == 'object':
                # Check if it's a categorical/string column
                if X[col].dtype.name == 'string' or hasattr(dtype, 'pyarrow_dtype'):
                    X[col] = X[col].astype(str)
            
            # Handle nullable integer types (Int64, Int32, etc.)
            elif hasattr(dtype, 'numpy_dtype'):
                X[col] = X[col].astype(dtype.numpy_dtype)
        
        return X, feature_names
    
    def _prepare_target(
        self,
        y: Union[pd.Series, np.ndarray]
    ) -> np.ndarray:
        """Chuẩn bị target."""
        if isinstance(y, pd.Series):
            # Handle StringDtype in pandas 3.0+
            if hasattr(y.dtype, 'name') and 'string' in str(y.dtype).lower():
                y = y.astype(object)
            y = y.values
        
        # Convert StringDtype array to object if needed
        if hasattr(y.dtype, 'name') and 'string' in str(y.dtype).lower():
            y = y.astype(object)
        
        # Encode categorical targets
        if self.task_type == TaskType.CLASSIFICATION:
            if y.dtype == object or not np.issubdtype(y.dtype, np.number):
                y = self.label_encoder.fit_transform(y)
                self._is_target_encoded = True
                self.target_classes = self.label_encoder.classes_
        
        return y
    
    def _resolve_cat_features(
        self,
        cat_features: List[Union[int, str]],
        feature_names: List[str]
    ) -> List[int]:
        """Chuyển đổi cat_features thành indices."""
        resolved = []
        for cf in cat_features:
            if isinstance(cf, str):
                if cf in feature_names:
                    resolved.append(feature_names.index(cf))
            elif isinstance(cf, int):
                resolved.append(cf)
        return resolved
    
    def _detect_cat_features(
        self,
        X: pd.DataFrame
    ) -> List[int]:
        """Tự động phát hiện categorical features."""
        cat_indices = []
        for i, col in enumerate(X.columns):
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                cat_indices.append(i)
        return cat_indices
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Tính metrics."""
        if self.task_type == TaskType.CLASSIFICATION:
            return {
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
                'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
            }
        else:
            return {
                'mse': float(mean_squared_error(y_true, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                'mae': float(mean_absolute_error(y_true, y_pred)),
                'r2_score': float(r2_score(y_true, y_pred)),
            }
    
    def _print_metrics(self, metrics: Dict[str, float]):
        """In metrics."""
        print("\n📊 Evaluation Metrics:")
        for name, value in metrics.items():
            print(f"   {name}: {value:.4f}")
    
    # ============================= PROPERTIES =============================
    
    @property
    def is_trained(self) -> bool:
        return self.status == ModelStatus.TRAINED
    
    @property
    def n_iterations(self) -> int:
        return self.params.get('iterations', 1000)
    
    @property
    def info(self) -> Dict[str, Any]:
        return {
            'task_type': self.task_type.value,
            'status': self.status.value,
            'loss_function': self.loss_function,
            'iterations': self.n_iterations,
            'depth': self.params.get('depth', 6),
            'n_features': len(self.feature_names),
            'n_cat_features': len(self.cat_features),
            'cat_features': self.cat_features,
        }
    
    def __repr__(self) -> str:
        return f"WeatherCatBoost(task_type='{self.task_type.value}', loss='{self.loss_function}', status='{self.status.value}')"


# ============================= FACTORY FUNCTIONS =============================

def create_classifier(**kwargs) -> WeatherCatBoost:
    """Factory tạo CatBoost Classifier."""
    return WeatherCatBoost(task_type='classification', **kwargs)


def create_regressor(loss_function: str = 'RMSE', **kwargs) -> WeatherCatBoost:
    """Factory tạo CatBoost Regressor."""
    return WeatherCatBoost(task_type='regression', loss_function=loss_function, **kwargs)