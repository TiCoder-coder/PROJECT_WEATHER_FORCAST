# ----------------------------- RANDOM FOREST MODEL - MÔ HÌNH RỪNG NGẪU NHIÊN -----------------------------------------------------------
"""
Random_Forest_Model.py - Module triển khai thuật toán Random Forest cho dự báo thời tiết

Mục đích:
    - Cung cấp wrapper class cho Random Forest algorithm
    - Hỗ trợ cả Classification và Regression
    - Tích hợp với pipeline dữ liệu thời tiết của project
    - Cung cấp các phương thức tiện ích: train, predict, evaluate, save/load

Nguyên lý hoạt động:
    1. Tạo nhiều bootstrap samples từ training data
    2. Với mỗi sample, chọn ngẫu nhiên k features
    3. Xây dựng Decision Tree cho mỗi sample
    4. Tổng hợp kết quả: Voting (classification) hoặc Average (regression)

Cách sử dụng:
    from Weather_Forcast_App.Models import WeatherRandomForest
    
    # Classification
    model = WeatherRandomForest(task_type='classification')
    model.train(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Regression
    model = WeatherRandomForest(task_type='regression')
    model.train(X_train, y_train)
    predictions = model.predict(X_test)
"""

import os
import json
import joblib
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Setup logging
logger = logging.getLogger(__name__)


# ============================= ENUMS & CONSTANTS =============================

class TaskType(Enum):
    """Loại bài toán ML."""
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'


class ModelStatus(Enum):
    """Trạng thái của model."""
    UNTRAINED = 'untrained'
    TRAINED = 'trained'
    FAILED = 'failed'


# Default hyperparameters
DEFAULT_PARAMS = {
    'n_estimators': 100,      # Số cây
    'max_depth': None,        # Độ sâu tối đa
    'max_features': 'sqrt',   # Số features cho mỗi split
    'min_samples_split': 2,   # Số samples tối thiểu để split
    'min_samples_leaf': 1,    # Số samples tối thiểu ở leaf
    'bootstrap': True,        # Sử dụng bootstrap
    'random_state': 42,       # Seed cho reproducibility
    'n_jobs': -1,             # Sử dụng tất cả CPU cores
}

# Model save directory
MODEL_DIR = Path(__file__).parent.parent / 'ml_models'


# ============================= DATA CLASSES =============================

@dataclass
class TrainingResult:
    """
    Kết quả huấn luyện model.
    
    Attributes:
        success: Huấn luyện thành công hay không
        metrics: Dict các metrics đánh giá
        training_time: Thời gian huấn luyện (seconds)
        n_samples: Số samples training
        n_features: Số features
        feature_names: Tên các features
        feature_importances: Độ quan trọng của features
        message: Thông báo
    """
    success: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    training_time: float = 0.0
    n_samples: int = 0
    n_features: int = 0
    feature_names: List[str] = field(default_factory=list)
    feature_importances: Dict[str, float] = field(default_factory=dict)
    message: str = ''
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PredictionResult:
    """
    Kết quả dự đoán.
    
    Attributes:
        predictions: Array các dự đoán
        probabilities: Xác suất (cho classification)
        prediction_time: Thời gian dự đoán (seconds)
    """
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None
    prediction_time: float = 0.0
    
    def to_list(self) -> List[Any]:
        return self.predictions.tolist()


@dataclass
class ModelConfig:
    """
    Cấu hình model.
    
    Attributes:
        task_type: Loại bài toán
        params: Hyperparameters
        created_at: Thời điểm tạo
        updated_at: Thời điểm cập nhật
    """
    task_type: TaskType
    params: Dict[str, Any] = field(default_factory=lambda: DEFAULT_PARAMS.copy())
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


# ============================= MAIN CLASS =============================

class WeatherRandomForest:
    """
    Random Forest Model cho dự báo thời tiết.
    
    Random Forest là thuật toán Ensemble Learning, kết hợp nhiều Decision Trees
    để đưa ra dự đoán chính xác và ổn định hơn.
    
    Tính năng:
        - Hỗ trợ Classification và Regression
        - Xử lý missing values tự động
        - Cross-validation
        - Hyperparameter tuning
        - Feature importance analysis
        - Save/Load model
    
    Attributes:
        task_type: Loại bài toán (classification/regression)
        model: Sklearn model instance
        scaler: StandardScaler cho normalization
        label_encoder: LabelEncoder cho categorical targets
        config: ModelConfig
        status: Trạng thái model
    
    Example:
        >>> # Classification: Dự đoán loại thời tiết
        >>> model = WeatherRandomForest(task_type='classification')
        >>> model.train(X_train, y_train)
        >>> predictions = model.predict(X_test)
        >>> 
        >>> # Regression: Dự đoán nhiệt độ
        >>> model = WeatherRandomForest(task_type='regression')
        >>> model.train(X_train, y_train)
        >>> temp_predictions = model.predict(X_test)
        >>> 
        >>> # Lấy feature importance
        >>> importances = model.get_feature_importance()
        >>> print(importances)
    """
    
    def __init__(
        self,
        task_type: str = 'classification',
        **kwargs
    ):
        """
        Khởi tạo Random Forest Model.
        
        Args:
            task_type: 'classification' hoặc 'regression'
            **kwargs: Các hyperparameters tùy chỉnh
                - n_estimators: Số cây (default=100)
                - max_depth: Độ sâu tối đa (default=None)
                - max_features: Số features mỗi split (default='sqrt')
                - min_samples_split: Min samples để split (default=2)
                - min_samples_leaf: Min samples ở leaf (default=1)
                - random_state: Seed (default=42)
        """
        # Validate task type
        try:
            self.task_type = TaskType(task_type.lower())
        except ValueError:
            raise ValueError(f"Invalid task_type: {task_type}. Must be 'classification' or 'regression'")
        
        # Merge params
        self.params = DEFAULT_PARAMS.copy()
        self.params.update(kwargs)
        
        # Initialize model
        self._init_model()
        
        # Preprocessors
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self._is_target_encoded = False
        
        # Config & Status
        self.config = ModelConfig(
            task_type=self.task_type,
            params=self.params
        )
        self.status = ModelStatus.UNTRAINED
        
        # Feature info
        self.feature_names: List[str] = []
        self.target_classes: Optional[np.ndarray] = None
        
        # Training history
        self.training_history: List[TrainingResult] = []
        
        logger.info(f"Initialized WeatherRandomForest ({self.task_type.value})")
    
    def _init_model(self):
        """Khởi tạo sklearn model dựa trên task type."""
        if self.task_type == TaskType.CLASSIFICATION:
            self.model = RandomForestClassifier(**self.params)
        else:
            self.model = RandomForestRegressor(**self.params)
    
    # ============================= TRAINING METHODS =============================
    
    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        validation_split: float = 0.2,
        scale_features: bool = True,
        verbose: bool = True
    ) -> TrainingResult:
        """
        Huấn luyện model.
        
        Args:
            X: Features (DataFrame hoặc ndarray)
            y: Target values
            validation_split: Tỷ lệ validation set (0-1)
            scale_features: Có normalize features không
            verbose: In thông tin huấn luyện
            
        Returns:
            TrainingResult với metrics và thông tin
            
        Example:
            >>> model = WeatherRandomForest(task_type='regression')
            >>> result = model.train(X_train, y_train, validation_split=0.2)
            >>> print(f"R2 Score: {result.metrics['r2_score']:.4f}")
        """
        start_time = datetime.now()
        
        try:
            # Convert to numpy
            X_array, feature_names = self._prepare_features(X)
            y_array = self._prepare_target(y)
            
            self.feature_names = feature_names
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_array, y_array, 
                test_size=validation_split, 
                random_state=self.params['random_state']
            )
            
            # Scale features
            if scale_features:
                X_train = self.scaler.fit_transform(X_train)
                X_val = self.scaler.transform(X_val)
            
            # Train model
            if verbose:
                print(f"🌲 Training Random Forest ({self.task_type.value})...")
                print(f"   📊 Training samples: {len(X_train)}")
                print(f"   📊 Validation samples: {len(X_val)}")
                print(f"   📊 Features: {X_train.shape[1]}")
                print(f"   🌳 Trees: {self.params['n_estimators']}")
            
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_val)
            metrics = self._calculate_metrics(y_val, y_pred)
            
            # Feature importance
            feature_importances = dict(zip(
                self.feature_names,
                self.model.feature_importances_.tolist()
            ))
            
            # Calculate training time
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Update status
            self.status = ModelStatus.TRAINED
            self.config.updated_at = datetime.now()
            
            # Create result
            result = TrainingResult(
                success=True,
                metrics=metrics,
                training_time=training_time,
                n_samples=len(X_array),
                n_features=X_array.shape[1],
                feature_names=self.feature_names,
                feature_importances=feature_importances,
                message="Training completed successfully"
            )
            
            self.training_history.append(result)
            
            if verbose:
                print(f"\n✅ Training completed in {training_time:.2f}s")
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
        cv: int = 5,
        scale_features: bool = True
    ) -> Dict[str, Any]:
        """
        Cross-validation để đánh giá model.
        
        Args:
            X: Features
            y: Target
            cv: Số folds
            scale_features: Có normalize không
            
        Returns:
            Dict với mean và std của các metrics
            
        Example:
            >>> cv_results = model.cross_validate(X, y, cv=5)
            >>> print(f"Mean Score: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")
        """
        X_array, _ = self._prepare_features(X)
        y_array = self._prepare_target(y)
        
        if scale_features:
            X_array = self.scaler.fit_transform(X_array)
        
        # Determine scoring metric
        if self.task_type == TaskType.CLASSIFICATION:
            scoring = 'accuracy'
        else:
            scoring = 'r2'
        
        scores = cross_val_score(self.model, X_array, y_array, cv=cv, scoring=scoring)
        
        return {
            'scores': scores.tolist(),
            'mean_score': float(scores.mean()),
            'std_score': float(scores.std()),
            'cv_folds': cv,
            'scoring': scoring
        }
    
    def tune_hyperparameters(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        param_grid: Optional[Dict[str, List[Any]]] = None,
        cv: int = 5,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Tìm hyperparameters tối ưu bằng Grid Search.
        
        Args:
            X: Features
            y: Target
            param_grid: Dict các tham số và giá trị thử nghiệm
            cv: Số folds cho cross-validation
            verbose: In thông tin
            
        Returns:
            Dict với best params và best score
            
        Example:
            >>> param_grid = {
            ...     'n_estimators': [100, 200, 300],
            ...     'max_depth': [10, 20, None],
            ...     'min_samples_split': [2, 5, 10]
            ... }
            >>> best = model.tune_hyperparameters(X, y, param_grid)
            >>> print(f"Best params: {best['best_params']}")
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        X_array, _ = self._prepare_features(X)
        y_array = self._prepare_target(y)
        
        if verbose:
            print("🔍 Tuning hyperparameters...")
            print(f"   Grid: {param_grid}")
        
        # Determine scoring
        scoring = 'accuracy' if self.task_type == TaskType.CLASSIFICATION else 'r2'
        
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1 if verbose else 0
        )
        
        grid_search.fit(X_array, y_array)
        
        # Update model with best params
        self.params.update(grid_search.best_params_)
        self._init_model()
        
        if verbose:
            print(f"\n✅ Best params: {grid_search.best_params_}")
            print(f"   Best score: {grid_search.best_score_:.4f}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': float(grid_search.best_score_),
            'cv_results': {
                'mean_test_scores': grid_search.cv_results_['mean_test_score'].tolist(),
                'std_test_scores': grid_search.cv_results_['std_test_score'].tolist(),
            }
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
            X: Features để dự đoán
            return_proba: Trả về probabilities (chỉ cho classification)
            
        Returns:
            PredictionResult với predictions và probabilities
            
        Raises:
            ValueError: Nếu model chưa được train
            
        Example:
            >>> result = model.predict(X_test)
            >>> predictions = result.predictions
            >>> 
            >>> # Với probabilities
            >>> result = model.predict(X_test, return_proba=True)
            >>> probas = result.probabilities
        """
        if self.status != ModelStatus.TRAINED:
            raise ValueError("Model chưa được train. Gọi train() trước khi predict.")
        
        start_time = datetime.now()
        
        X_array, _ = self._prepare_features(X)
        
        # Scale if scaler was fitted
        if hasattr(self.scaler, 'mean_'):
            X_array = self.scaler.transform(X_array)
        
        # Predict
        predictions = self.model.predict(X_array)
        
        # Decode labels if encoded
        if self._is_target_encoded and self.task_type == TaskType.CLASSIFICATION:
            predictions = self.label_encoder.inverse_transform(predictions.astype(int))
        
        # Get probabilities for classification
        probabilities = None
        if return_proba and self.task_type == TaskType.CLASSIFICATION:
            probabilities = self.model.predict_proba(X_array)
        
        prediction_time = (datetime.now() - start_time).total_seconds()
        
        return PredictionResult(
            predictions=predictions,
            probabilities=probabilities,
            prediction_time=prediction_time
        )
    
    def predict_single(self, sample: Dict[str, Any]) -> Any:
        """
        Dự đoán cho một sample duy nhất.
        
        Args:
            sample: Dict với feature names và values
            
        Returns:
            Giá trị dự đoán
            
        Example:
            >>> sample = {
            ...     'temperature': 25.5,
            ...     'humidity': 80,
            ...     'wind_speed': 10.2
            ... }
            >>> prediction = model.predict_single(sample)
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
        
        Args:
            X: Test features
            y: True labels/values
            
        Returns:
            Dict với các metrics
        """
        result = self.predict(X)
        y_array = self._prepare_target(y)
        
        metrics = self._calculate_metrics(y_array, result.predictions)
        
        # Thêm confusion matrix cho classification
        if self.task_type == TaskType.CLASSIFICATION:
            metrics['confusion_matrix'] = confusion_matrix(y_array, result.predictions).tolist()
            metrics['classification_report'] = classification_report(
                y_array, result.predictions, output_dict=True
            )
        
        return metrics
    
    def get_feature_importance(self, top_n: Optional[int] = None) -> Dict[str, float]:
        """
        Lấy độ quan trọng của các features.
        
        Args:
            top_n: Số features quan trọng nhất (None = tất cả)
            
        Returns:
            Dict với feature name và importance score
            
        Example:
            >>> importances = model.get_feature_importance(top_n=10)
            >>> for name, score in importances.items():
            ...     print(f"{name}: {score:.4f}")
        """
        if self.status != ModelStatus.TRAINED:
            raise ValueError("Model chưa được train")
        
        importances = dict(zip(
            self.feature_names,
            self.model.feature_importances_
        ))
        
        # Sort by importance
        importances = dict(sorted(
            importances.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        if top_n:
            importances = dict(list(importances.items())[:top_n])
        
        return importances
    
    # ============================= SAVE/LOAD METHODS =============================
    
    def save(self, filepath: Optional[str] = None, include_metadata: bool = True) -> str:
        """
        Lưu model ra file.
        
        Args:
            filepath: Đường dẫn file (None = auto generate)
            include_metadata: Lưu metadata cùng model
            
        Returns:
            Đường dẫn file đã lưu
            
        Example:
            >>> path = model.save()
            >>> print(f"Model saved to: {path}")
        """
        if self.status != ModelStatus.TRAINED:
            raise ValueError("Model chưa được train")
        
        # Tạo thư mục nếu chưa có
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        # Generate filepath
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"random_forest_{self.task_type.value}_{timestamp}.joblib"
            filepath = str(MODEL_DIR / filename)
        
        # Prepare data to save
        save_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder if self._is_target_encoded else None,
            'feature_names': self.feature_names,
            'target_classes': self.target_classes,
            'params': self.params,
            'task_type': self.task_type.value,
            '_is_target_encoded': self._is_target_encoded,
        }
        
        if include_metadata:
            save_data['metadata'] = {
                'created_at': self.config.created_at.isoformat(),
                'updated_at': self.config.updated_at.isoformat(),
                'training_history': [r.to_dict() for r in self.training_history]
            }
        
        joblib.dump(save_data, filepath)
        logger.info(f"Model saved to: {filepath}")
        
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> 'WeatherRandomForest':
        """
        Load model từ file.
        
        Args:
            filepath: Đường dẫn file model
            
        Returns:
            WeatherRandomForest instance
            
        Example:
            >>> model = WeatherRandomForest.load('model.joblib')
            >>> predictions = model.predict(X_new)
        """
        save_data = joblib.load(filepath)
        
        # Create instance
        instance = cls(task_type=save_data['task_type'])
        
        # Restore attributes
        instance.model = save_data['model']
        instance.scaler = save_data['scaler']
        instance.feature_names = save_data['feature_names']
        instance.target_classes = save_data['target_classes']
        instance.params = save_data['params']
        instance._is_target_encoded = save_data['_is_target_encoded']
        
        if save_data['label_encoder']:
            instance.label_encoder = save_data['label_encoder']
        
        instance.status = ModelStatus.TRAINED
        
        logger.info(f"Model loaded from: {filepath}")
        
        return instance
    
    # ============================= PRIVATE METHODS =============================
    
    def _prepare_features(
        self, 
        X: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[np.ndarray, List[str]]:
        """Chuẩn bị features cho training/prediction."""
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X_array = X.values
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X_array = X
        
        # Handle missing values
        if np.isnan(X_array).any():
            X_array = np.nan_to_num(X_array, nan=np.nanmean(X_array))
        
        return X_array.astype(np.float32), feature_names
    
    def _prepare_target(
        self, 
        y: Union[pd.Series, np.ndarray]
    ) -> np.ndarray:
        """Chuẩn bị target cho training."""
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y
        
        # Encode categorical targets for classification
        if self.task_type == TaskType.CLASSIFICATION:
            if y_array.dtype == object or not np.issubdtype(y_array.dtype, np.number):
                y_array = self.label_encoder.fit_transform(y_array)
                self._is_target_encoded = True
                self.target_classes = self.label_encoder.classes_
        
        return y_array
    
    def _calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Tính toán metrics dựa trên task type."""
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
        """In metrics ra console."""
        print("\n📊 Evaluation Metrics:")
        for name, value in metrics.items():
            print(f"   {name}: {value:.4f}")
    
    # ============================= PROPERTIES =============================
    
    @property
    def is_trained(self) -> bool:
        """Kiểm tra model đã được train chưa."""
        return self.status == ModelStatus.TRAINED
    
    @property
    def n_trees(self) -> int:
        """Số cây trong forest."""
        return self.params.get('n_estimators', 100)
    
    @property
    def info(self) -> Dict[str, Any]:
        """Thông tin tổng quan về model."""
        return {
            'task_type': self.task_type.value,
            'status': self.status.value,
            'n_trees': self.n_trees,
            'max_depth': self.params.get('max_depth'),
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'params': self.params,
        }
    
    def __repr__(self) -> str:
        return f"WeatherRandomForest(task_type='{self.task_type.value}', status='{self.status.value}', n_trees={self.n_trees})"


# ============================= FACTORY FUNCTIONS =============================

def create_classifier(**kwargs) -> WeatherRandomForest:
    """
    Factory function tạo Random Forest Classifier.
    
    Returns:
        WeatherRandomForest configured for classification
    """
    return WeatherRandomForest(task_type='classification', **kwargs)


def create_regressor(**kwargs) -> WeatherRandomForest:
    """
    Factory function tạo Random Forest Regressor.
    
    Returns:
        WeatherRandomForest configured for regression
    """
    return WeatherRandomForest(task_type='regression', **kwargs)