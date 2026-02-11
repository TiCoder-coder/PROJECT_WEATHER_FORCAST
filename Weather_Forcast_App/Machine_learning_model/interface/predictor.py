# ----------------------------- PREDICTOR - INFERENCE INTERFACE -----------------------------------------------------------
"""
predictor.py - Interface dự đoán (inference) cho Weather Forecast ML Pipeline

Mục đích:
    - Load artifacts đã train (Model, Pipeline, Feature list)
    - Nhận dữ liệu mới → build features → transform → predict
    - Đảm bảo predict dùng đúng pipeline/features như lúc train

Cách sử dụng:
    from Weather_Forcast_App.Machine_learning_model.interface.predictor import WeatherPredictor

    # Load model đã train
    predictor = WeatherPredictor.from_artifacts("path/to/artifacts/latest")

    # Predict
    result = predictor.predict(df_new)
    print(result["predictions"])
"""

from __future__ import annotations

import json
import logging
import joblib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Paths mặc định
THIS_FILE = Path(__file__).resolve()
APP_DIR = THIS_FILE.parent.parent.parent  # Weather_Forcast_App
DEFAULT_ARTIFACTS_DIR = APP_DIR / "Machine_learning_artifacts" / "latest"


class WeatherPredictor:
    """
    Interface dự đoán thời tiết từ artifacts đã train.

    Load:
        - Model.pkl (model wrapper)
        - Transform_pipeline.pkl (pipeline transform)
        - Feature_list.json (danh sách features)
        - Train_info.json (thông tin train)

    Public API:
        - from_artifacts(artifacts_dir)  → tạo instance
        - predict(df)                    → trả dict predictions
        - get_info()                     → thông tin model/pipeline
    """

    def __init__(
        self,
        model: Any,
        pipeline: Any,
        feature_columns: List[str],
        target_column: str = "",
        feature_builder: Any = None,
        train_info: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.pipeline = pipeline
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.feature_builder = feature_builder
        self.train_info = train_info or {}

    @classmethod
    def from_artifacts(
        cls,
        artifacts_dir: Optional[Union[str, Path]] = None,
    ) -> "WeatherPredictor":
        """
        Load predictor từ thư mục artifacts đã train.

        Args:
            artifacts_dir: Đường dẫn thư mục artifacts.
                           None → dùng Machine_learning_artifacts/latest

        Returns:
            WeatherPredictor instance sẵn sàng predict.
        """
        if artifacts_dir is None:
            artifacts_dir = DEFAULT_ARTIFACTS_DIR
        artifacts_dir = Path(artifacts_dir)

        if not artifacts_dir.exists():
            raise FileNotFoundError(f"Artifacts dir not found: {artifacts_dir}")

        # --- Load model ---
        model_path = artifacts_dir / "Model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model.pkl not found in {artifacts_dir}")
        model = joblib.load(model_path)

        # --- Load pipeline ---
        pipeline_path = artifacts_dir / "Transform_pipeline.pkl"
        pipeline = None
        if pipeline_path.exists():
            from Weather_Forcast_App.Machine_learning_model.features.Transformers import (
                WeatherTransformPipeline,
            )
            pipeline = WeatherTransformPipeline.load(pipeline_path)

        # --- Load feature list ---
        features_path = artifacts_dir / "Feature_list.json"
        feature_columns: List[str] = []
        target_column = ""
        feature_builder = None

        if features_path.exists():
            with open(features_path, "r", encoding="utf-8") as f:
                feat_data = json.load(f)
            feature_columns = feat_data.get("all_feature_columns", [])
            target_column = feat_data.get("target_column", "")

            # Nếu có thông tin features config → tạo builder
            group_by = feat_data.get("group_by")
            created_features = feat_data.get("created_features", [])
            if created_features:
                try:
                    from Weather_Forcast_App.Machine_learning_model.features.Build_transfer import (
                        WeatherFeatureBuilder,
                    )
                    feature_builder = WeatherFeatureBuilder()
                except ImportError:
                    pass

        # --- Load train info ---
        info_path = artifacts_dir / "Train_info.json"
        train_info: Dict[str, Any] = {}
        if info_path.exists():
            with open(info_path, "r", encoding="utf-8") as f:
                train_info = json.load(f)

        logger.info("Loaded predictor from %s", artifacts_dir)
        return cls(
            model=model,
            pipeline=pipeline,
            feature_columns=feature_columns,
            target_column=target_column,
            feature_builder=feature_builder,
            train_info=train_info,
        )

    def predict(
        self,
        df: pd.DataFrame,
        *,
        build_features: bool = True,
        group_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Dự đoán từ DataFrame mới.

        Args:
            df: DataFrame dữ liệu đầu vào (raw hoặc đã build features).
            build_features: True → tự build features trước khi predict.
            group_by: Cột group by (nếu dùng feature builder).

        Returns:
            Dict chứa:
                - predictions: np.ndarray kết quả dự đoán
                - prediction_time: thời gian predict (s)
                - n_samples: số mẫu
        """
        start = datetime.now()

        X = df.copy()

        # 1) Build features nếu cần
        if build_features and self.feature_builder is not None and self.target_column:
            X = self.feature_builder.build_all_features(
                X, target_column=self.target_column, group_by=group_by
            )
            if self.target_column in X.columns:
                X = X.drop(columns=[self.target_column])

        # 2) Align columns theo feature_columns đã lưu
        if self.feature_columns:
            for col in self.feature_columns:
                if col not in X.columns:
                    X[col] = np.nan
            extra = [c for c in X.columns if c not in self.feature_columns]
            if extra:
                X = X.drop(columns=extra)
            X = X[self.feature_columns]

        # 3) Transform pipeline
        if self.pipeline is not None:
            X = self.pipeline.transform(X)

        # 4) Predict
        if hasattr(self.model, "predict"):
            preds = self.model.predict(X)
            # Handle PredictionResult dataclass
            if hasattr(preds, "predictions"):
                preds = preds.predictions
        else:
            raise RuntimeError(f"Model {type(self.model).__name__} has no predict()")

        elapsed = (datetime.now() - start).total_seconds()

        return {
            "predictions": np.array(preds),
            "prediction_time": elapsed,
            "n_samples": len(df),
        }

    def get_info(self) -> Dict[str, Any]:
        """Trả về thông tin model/pipeline/features."""
        info: Dict[str, Any] = {
            "model_type": type(self.model).__name__,
            "target_column": self.target_column,
            "n_features": len(self.feature_columns),
            "has_pipeline": self.pipeline is not None,
            "has_feature_builder": self.feature_builder is not None,
        }
        if self.pipeline is not None and hasattr(self.pipeline, "get_pipeline_info"):
            info["pipeline_info"] = self.pipeline.get_pipeline_info()
        if self.train_info:
            info["trained_at"] = self.train_info.get("trained_at", "")
            info["model_config"] = self.train_info.get("model", {})
        return info

    def __repr__(self) -> str:
        return (
            f"WeatherPredictor(model={type(self.model).__name__}, "
            f"features={len(self.feature_columns)}, "
            f"target='{self.target_column}')"
        )
