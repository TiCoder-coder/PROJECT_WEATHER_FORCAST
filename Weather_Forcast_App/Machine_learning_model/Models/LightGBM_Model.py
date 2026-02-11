# ----------------------------- LIGHTGBM MODEL - GRADIENT BOOSTING (GBDT) -----------------------------------------------------------
"""
Weather_Forcast_App.Machine_learning_model.Models.LightGBM_Model

Wrapper triển khai LightGBM (LGBMRegressor / LGBMClassifier) cho bài toán dự báo thời tiết.

Tính năng chính:
    - Hỗ trợ Regression & Classification
    - Xử lý missing values native của LightGBM (không cần impute bắt buộc)
    - Hỗ trợ categorical features bằng pandas "category" + lưu mapping categories để predict ổn định
    - Tự trích xuất datetime features (year/month/day/dow/hour/minute) và lưu schema
    - Train / Evaluate / Predict, Cross-Validate (sklearn), Tuning (RandomizedSearchCV)
    - Save/Load model (joblib) + Export artifacts về Machine_learning_artifacts/latest

Ghi chú cho dữ liệu weather (time-series):
    - Mặc định train() split validation với shuffle=False để tránh leak theo thời gian.
      Nếu bài toán tabular bình thường, đặt shuffle=True.

Ví dụ nhanh:
    from Weather_Forcast_App.Machine_learning_model.Models.LightGBM_Model import WeatherLightGBM

    # Regression
    mdl = WeatherLightGBM(task_type="regression")
    res = mdl.train(X, y, target_name="rain_mm", shuffle=False)
    preds = mdl.predict(X_test).predictions

    # Classification
    clf = WeatherLightGBM(task_type="classification")
    res = clf.train(X, y, target_name="weather_type", shuffle=True, stratify=True)
    preds = clf.predict(X_test).predictions
"""

from __future__ import annotations

import json
import joblib
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# LightGBM imports
try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except Exception:
    lgb = None
    LGBMClassifier = None
    LGBMRegressor = None
    LIGHTGBM_AVAILABLE = False


from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder

from Weather_Forcast_App.Machine_learning_model.Models import (
    TaskType,
    ModelStatus,
    TrainingResult,
    PredictionResult,
)

logger = logging.getLogger(__name__)


# Bộ tham số mặc định: thiên về "safe defaults" cho dữ liệu tabular/time-series weather.
# - n_estimators lớn + early stopping => model tự chọn best_iteration ổn định hơn.
DEFAULT_LGB_PARAMS: Dict[str, Any] = {
    # Core
    "boosting_type": "gbdt",     # gradient boosting desion tree
    "n_estimators": 3000,        # Số cây tối đa
    "learning_rate": 0.03,       # Tốc độ học ( nếu giảm chỉ số này thì phải tăng số cây quyết định lên)

    # Leaf-wise growth (quan trọng nhất)
    "num_leaves": 63,            # Số lá tối đa cho mỗi cây (càng lơn smoo hình mạnh hơn nhưng dễ overfit hơn)
    "max_depth": -1,             # Không giới hạn độ sâu (tối ưu theo leaf-wise)

    # Regularization / chống overfit
    "min_child_samples": 20,     # Số mẫu tối thiểu trong 1 leaf
    "subsample": 0.8,            # bagging_fraction
    "subsample_freq": 1,         # bagging_freq
    "colsample_bytree": 0.8,     # feature_fraction
    "reg_alpha": 0.0,            # Tăng chỉ số này lên nếu mô hình bị overfit
    "reg_lambda": 0.0,

    # Speed & reproducibility
    "n_jobs": -1,                # Dùng tất cả core của cpu
    "random_state": 42,          # 42 seed ( đạt seed cho các lần ngãy nhiên - để kq lần sau chay giông kết quả lân trước)
    "verbosity": -1,             # Giảm log
}

# Paths (giống các model khác)
# MODEL_DIR: nơi lưu các model đã train dạng .joblib
MODEL_DIR = Path(__file__).parent.parent / "ml_models"
# APP_DIR: trỏ về thư mục Weather_Forcast_App
APP_DIR = Path(__file__).resolve().parent.parent.parent  # .../Weather_Forcast_App
# LATEST_ARTIFACTS_DIR: nơi service/app lấy artifacts "latest" để inference
LATEST_ARTIFACTS_DIR = APP_DIR / "Machine_learning_artifacts" / "latest"


# ============================= MAIN CLASS =============================

class WeatherLightGBM:
    """
    Wrapper LightGBM cho dự báo thời tiết.

    Public API:
        - train()
        - predict() / predict_proba()
        - evaluate()
        - cross_validate()
        - tune_hyperparameters()
        - save() / load()
        - export_latest_artifacts() / load_latest_artifacts()
    """

    def __init__(
        self,
        task_type: str = "regression",
        params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        use_gpu: bool = False,
    ):
        # Kiểm tra dependency
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM chưa được cài. Hãy chạy: pip install lightgbm")

        # Xác định task: regression / classification
        self.task_type = TaskType(task_type.lower())

        # Seed cho reproducibility
        self.random_state = int(random_state)

        # Tuỳ chọn GPU (nếu lightgbm build hỗ trợ GPU)
        self.use_gpu = bool(use_gpu)

        # Trộn params người dùng + params mặc định
        base = DEFAULT_LGB_PARAMS.copy()
        if params:
            base.update(params)

        # đồng bộ random_state
        base["random_state"] = self.random_state

        # GPU (tuỳ chọn) — tham số phổ biến là device_type="gpu"
        if self.use_gpu:
            base.setdefault("device_type", "gpu")

        self.params: Dict[str, Any] = base

        # trạng thái model ban đầu
        self.status = ModelStatus.UNTRAINED

        # Estimator sklearn wrapper của LightGBM (LGBMRegressor hoặc LGBMClassifier)
        self.model: Union[LGBMRegressor, LGBMClassifier, None] = None

        # Schema / metadata
        # feature_names: cột feature sau preprocess (datetime -> derived, categorical -> category)
        self.feature_names: List[str] = []
        # target_name: tên target (để export artifacts/info)
        self.target_name: Optional[str] = None

        # Classification encoder
        # label_encoder: mã hoá nhãn string -> int để LightGBM train
        self.label_encoder: Optional[LabelEncoder] = None
        # cờ đã encode target hay chưa
        self._is_target_encoded: bool = False
        # danh sách class gốc (theo label_encoder.classes_)
        self.target_classes: Optional[np.ndarray] = None

        # Categorical
        # cat_features: list tên cột categorical dùng cho LightGBM
        self.cat_features: List[str] = []
        # _cat_categories: mapping col -> list categories để predict ổn định (set_categories)
        self._cat_categories: Dict[str, List[Any]] = {}

        # Datetime
        # datetime_cols: list tên cột datetime gốc
        self.datetime_cols: List[str] = []
        # _datetime_feature_map: mapping dt_col -> list derived columns (year/month/day/...)
        self._datetime_feature_map: Dict[str, List[str]] = {}

        # Training history / evals
        # training_history: lưu các TrainingResult qua nhiều lần train
        self.training_history: List[TrainingResult] = []
        # evals_result_: lưu lịch sử metric trong train (nếu LightGBM trả về)
        self.evals_result_: Dict[str, Any] = {}

        # Khởi tạo model ban đầu (objective sẽ chuẩn hoá lại khi train biết số class)
        self._init_model(n_classes=None)

    # ============================= PUBLIC =============================

    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray], # Biến đầu vào 
        y: Union[pd.Series, np.ndarray, List[Any]], # Target dự đoán/ Regession số thực/ Classification: nhãn
        *,
        target_name: Optional[str] = None,# Tên cột target để lưu vào metadata 
        cat_features: Optional[List[Union[str, int]]] = None, # Chỉ định các cột categotical
        datetime_cols: Optional[List[Union[str, int]]] = None, # Chỉ định các cột datetime để tách thành feature số
        val_size: float = 0.2, # Tỉ lệ split khi split từ data train
        shuffle: bool = False, # Có xáo trộn dữ liệu trước khi split không
        stratify: bool = True, # Chỉ số có í nghĩa khi classification + shuffle = True
        early_stopping_rounds: int = 100, # Dừng khi có validation
        eval_metric: Optional[Union[str, List[str]]] = None, # Metric để lightgbm theo dõi trên validation
        verbose_eval: int = 100, # Cữ mỗi 100 vòng boosting thì in log 1 lần
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray, List[Any]]] = None,
    ) -> TrainingResult:
        """
        Train model LightGBM.

        Luồng xử lý:
            1) preprocess X (datetime -> derived, categorical -> category, replace inf->NaN)
            2) preprocess y (regression -> float, classification -> label encode)
            3) tách validation (hoặc dùng X_val/y_val nếu truyền)
            4) init lại model (classification cần biết num_class)
            5) fit model với early stopping + log evaluation
            6) evaluate và trả TrainingResult
        """
        start_time = time.time()
        self.target_name = target_name

        try:
            # 1) Features (fit=True để "đóng băng" schema + mapping)
            # - self.feature_names sẽ được set tại đây
            # - self.cat_features / self.datetime_cols sẽ được lưu lại
            X_df, feature_names = self._prepare_features(
                X,
                fit=True,
                cat_features=cat_features,
                datetime_cols=datetime_cols,
            )

            # 2) Target
            # - regression: ép float
            # - classification: LabelEncoder.fit_transform + lưu classes
            y_arr = self._prepare_target(y, fit=True)

            # 3) Split train/val
            # Nếu user tự truyền X_val/y_val => dùng trực tiếp
            if X_val is not None and y_val is not None:
                # train set: toàn bộ X_df, y_arr
                X_train_df = X_df
                y_train = y_arr

                # preprocess validation theo schema đã fit
                X_val_df, _ = self._prepare_features(
                    X_val,
                    fit=False,
                    cat_features=self.cat_features,
                    datetime_cols=self.datetime_cols,
                )
                # chuẩn hoá y_val theo encoder đã có (fit=False)
                y_val_arr = self._prepare_target(y_val, fit=False)
            else:
                # Nếu val_size không hợp lệ => bỏ qua validation
                if not (0.0 < float(val_size) < 1.0):
                    X_train_df = X_df
                    y_train = y_arr
                    X_val_df = None
                    y_val_arr = None
                else:
                    # Tham số split
                    split_kwargs: Dict[str, Any] = {
                        "test_size": float(val_size),
                        "random_state": self.random_state,
                        "shuffle": bool(shuffle),
                    }

                    # Chỉ stratify khi classification + shuffle=True
                    if self.task_type == TaskType.CLASSIFICATION and shuffle and stratify:
                        split_kwargs["stratify"] = y_arr

                    # train_test_split trả về: X_train, X_val, y_train, y_val
                    X_train_df, X_val_df, y_train, y_val_arr = train_test_split(
                        X_df, y_arr, **split_kwargs
                    )

            # 4) Re-init model (classification cần biết num_class)
            # - regression: objective regression
            # - classification: binary hoặc multiclass + num_class
            n_classes = int(len(self.target_classes)) if self.target_classes is not None else None
            self._init_model(n_classes=n_classes)

            # 5) callbacks: early stopping + log
            callbacks = []
            # Nếu có validation => mới early stopping được
            if X_val_df is not None and y_val_arr is not None and early_stopping_rounds and early_stopping_rounds > 0:
                callbacks.append(
                    lgb.early_stopping(
                        stopping_rounds=int(early_stopping_rounds),
                        verbose=False
                    )
                )
            # log_evaluation: in log mỗi verbose_eval vòng
            if verbose_eval and verbose_eval > 0:
                callbacks.append(lgb.log_evaluation(period=int(verbose_eval)))

            # 6) Fit
            fit_kwargs: Dict[str, Any] = {}

            # Nếu user không truyền eval_metric, set default theo task
            if eval_metric is None:
                if self.task_type == TaskType.REGRESSION:
                    eval_metric = "rmse"
                else:
                    # binary => logloss, multiclass => multi_logloss
                    eval_metric = "logloss" if (n_classes is None or n_classes <= 2) else "multi_logloss"
            fit_kwargs["eval_metric"] = eval_metric

            # categorical_feature: LightGBM sklearn API chấp nhận list tên cột nếu input là DataFrame
            if self.cat_features:
                fit_kwargs["categorical_feature"] = self.cat_features

            # eval_set: truyền validation set để LightGBM theo dõi metric và early stopping
            if X_val_df is not None and y_val_arr is not None:
                fit_kwargs["eval_set"] = [(X_val_df, y_val_arr)]

            # callbacks: early stopping + log
            if callbacks:
                fit_kwargs["callbacks"] = callbacks

            # Thử fit với callbacks (LightGBM version mới thường ok)
            try:
                self.model.fit(X_train_df, y_train, **fit_kwargs)
            except TypeError:
                # Fallback cho một số version LightGBM cũ không nhận callbacks trong fit()
                fit_kwargs.pop("callbacks", None)

                # Với version cũ: dùng early_stopping_rounds + verbose=False
                if (X_val_df is not None and y_val_arr is not None) and early_stopping_rounds and early_stopping_rounds > 0:
                    fit_kwargs["early_stopping_rounds"] = int(early_stopping_rounds)
                    fit_kwargs["verbose"] = False

                self.model.fit(X_train_df, y_train, **fit_kwargs)
            self.status = ModelStatus.TRAINED

            # 7) Collect metrics / evals
            # best_iteration_ (nếu có early stopping)
            best_iter = getattr(self.model, "best_iteration_", None)
            # evals_result_ (nếu wrapper cung cấp)
            self.evals_result_ = getattr(self.model, "evals_result_", {}) or {}

            # Evaluate: nếu có validation => evaluate trên val, ngược lại evaluate trên train
            metrics = self.evaluate(
                X_val_df if (X_val_df is not None and y_val_arr is not None) else X_train_df,
                y_val_arr if (X_val_df is not None and y_val_arr is not None) else y_train,
                return_details=True,
            )

            # Feature importances
            feature_importances = self.get_feature_importance()

            # Tính thời gian train
            train_time = time.time() - start_time

            # Tạo TrainingResult trả về
            result = TrainingResult(
                success=True,
                metrics=metrics,
                training_time=float(train_time),
                n_samples=int(X_train_df.shape[0]),
                n_features=int(X_train_df.shape[1]),
                feature_names=feature_names,
                feature_importances=feature_importances,
                best_iteration=int(best_iter) if best_iter is not None else None,
                message="Train LightGBM thành công",
            )

            # Lưu lịch sử train
            self.training_history.append(result)

            # Set status
            self.status = ModelStatus.TRAINED
            return result

        except Exception as e:  # pragma: no cover
            # Nếu có lỗi, set status FAILED và trả TrainingResult thất bại
            self.status = ModelStatus.FAILED
            msg = f"Train LightGBM thất bại: {e}"
            logger.exception(msg)
            return TrainingResult(success=False, message=msg)

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        *,
        return_proba: bool = False,
    ) -> PredictionResult:
        """
        Dự đoán đầu ra từ feature X.

        - Regression: trả về mảng float
        - Classification: trả về label (đã decode về string nếu dùng LabelEncoder)
        - Nếu return_proba=True (classification): trả thêm probabilities
        """
        # Bảo đảm model sẵn sàng
        if self.status != ModelStatus.TRAINED or self.model is None:
            raise ValueError("Model chưa được train hoặc load.")

        start = time.time()

        # preprocess X theo schema đã fit (fit=False)
        X_df, _ = self._prepare_features(
            X,
            fit=False,
            cat_features=self.cat_features,
            datetime_cols=self.datetime_cols
        )

        probs = None
        if self.task_type == TaskType.CLASSIFICATION:
            # predict_proba trả xác suất theo từng class
            if return_proba:
                probs = self.model.predict_proba(X_df)

            # predict trả class index (int) theo LabelEncoder
            pred_encoded = self.model.predict(X_df)

            # decode về label gốc (ví dụ: "rainy", "sunny")
            pred = self._decode_target(np.array(pred_encoded))
        else:
            # regression: predict ra float
            pred = self.model.predict(X_df)

        return PredictionResult(
            predictions=np.array(pred),
            probabilities=probs,
            prediction_time=float(time.time() - start),
        )

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Shortcut cho classification:
            - trả về xác suất dự đoán theo từng class
        """
        if self.task_type != TaskType.CLASSIFICATION:
            raise ValueError("predict_proba chỉ dùng cho classification.")
        out = self.predict(X, return_proba=True).probabilities
        if out is None:
            raise RuntimeError("Không lấy được probabilities.")
        return out

    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y_true: Union[pd.Series, np.ndarray, List[Any]],
        *,
        return_details: bool = False,
    ) -> Dict[str, Any]:
        """
        Đánh giá model bằng metric phù hợp:

        - Regression:
            mse, rmse, mae, r2, mape_percent
        - Classification:
            accuracy, precision_macro, recall_macro, f1_macro
            return_details=True: thêm confusion_matrix + classification_report
        """
        if self.status != ModelStatus.TRAINED or self.model is None:
            raise ValueError("Model chưa được train hoặc load.")

        # preprocess X theo schema đã fit
        X_df, _ = self._prepare_features(
            X,
            fit=False,
            cat_features=self.cat_features,
            datetime_cols=self.datetime_cols
        )

        # preprocess y_true:
        # - regression: float
        # - classification: encode về int nếu y_true là string
        y_arr = self._prepare_target(y_true, fit=False)

        if self.task_type == TaskType.REGRESSION:
            # Dự đoán
            y_pred = self.model.predict(X_df)

            # Tính metric
            mse = mean_squared_error(y_arr, y_pred)
            rmse = float(np.sqrt(mse))
            mae = mean_absolute_error(y_arr, y_pred)
            r2 = r2_score(y_arr, y_pred)

            # MAPE an toàn (tránh chia 0)
            eps = 1e-9
            denom = np.maximum(np.abs(y_arr), eps)
            mape = float(np.mean(np.abs((y_arr - y_pred) / denom)) * 100.0)

            return {
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae),
                "r2": float(r2),
                "mape_percent": float(mape),
            }

        # Classification
        # predict trả class index (int)
        y_pred_encoded = self.model.predict(X_df)
        y_pred = np.array(y_pred_encoded, dtype=int)

        # Metric classification (macro để cân bằng giữa các class)
        acc = accuracy_score(y_arr, y_pred)
        prec = precision_score(y_arr, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_arr, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_arr, y_pred, average="macro", zero_division=0)

        out: Dict[str, Any] = {
            "accuracy": float(acc),
            "precision_macro": float(prec),
            "recall_macro": float(rec),
            "f1_macro": float(f1),
        }

        # Thêm thông tin chi tiết khi debug
        if return_details:
            out["confusion_matrix"] = confusion_matrix(y_arr, y_pred).tolist()
            out["classification_report"] = classification_report(y_arr, y_pred, zero_division=0)

        return out

    def cross_validate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, List[Any]],
        *,
        cv: int = 5,
        scoring: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Cross-validation nhanh (sklearn cross_val_score).

        Lưu ý:
            - Với time-series, CV ngẫu nhiên có thể gây data leakage.
              Khi muốn chuẩn time-series => dùng TimeSeriesSplit bên ngoài.
        """
        # nếu chưa có schema/encoder, chuẩn bị như "fit" để tránh schema rỗng
        # - trường hợp chạy CV trước khi train
        X_df, _ = self._prepare_features(
            X,
            fit=(not bool(self.feature_names))
        )

        # classification: nếu chưa encode target => encode luôn
        y_arr = self._prepare_target(
            y,
            fit=(self.task_type == TaskType.CLASSIFICATION and not self._is_target_encoded)
        )

        # đảm bảo estimator đúng objective multiclass/binary trước khi CV
        n_classes = int(len(self.target_classes)) if self.target_classes is not None else None
        estimator = self._build_estimator_for_search(n_classes=n_classes)

        # scoring mặc định
        if scoring is None:
            scoring = "r2" if self.task_type == TaskType.REGRESSION else "accuracy"

        # cross_val_score trả về array điểm theo từng fold
        scores = cross_val_score(estimator, X_df, y_arr, cv=int(cv), scoring=scoring)

        return {
            "cv": int(cv),
            "scoring": scoring,
            "scores": scores.tolist(),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
        }

    def tune_hyperparameters(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, List[Any]],
        *,
        n_iter: int = 30,
        cv: int = 3,
        scoring: Optional[str] = None,
        random_state: Optional[int] = None,
        param_distributions: Optional[Dict[str, Any]] = None,
        verbose: int = 1,
        n_jobs: int = -1,
    ) -> Dict[str, Any]:
        """
        Hyperparameter tuning bằng RandomizedSearchCV.

        Luồng:
            1) preprocess X/y (đảm bảo schema + encode)
            2) tạo base_estimator đúng objective (binary/multiclass)
            3) chạy RandomizedSearchCV
            4) update self.params theo best_params và init lại model
        """
        # nếu chưa có schema/encoder, chuẩn bị như "fit"
        X_df, _ = self._prepare_features(
            X,
            fit=(not bool(self.feature_names))
        )
        y_arr = self._prepare_target(
            y,
            fit=(self.task_type == TaskType.CLASSIFICATION and not self._is_target_encoded)
        )

        # random_state cho search
        if random_state is None:
            random_state = self.random_state

        # scoring mặc định:
        # - regression: neg_root_mean_squared_error (sklearn convention: maximize)
        # - classification: accuracy
        if scoring is None:
            scoring = "neg_root_mean_squared_error" if self.task_type == TaskType.REGRESSION else "accuracy"

        # Nếu user không truyền distribution => dùng default hợp lý cho LightGBM
        if param_distributions is None:
            param_distributions = {
                "n_estimators": [800, 1200, 2000, 3000, 5000],
                "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.1],
                "num_leaves": [31, 63, 127, 255],
                "max_depth": [-1, 6, 8, 10, 12],
                "min_child_samples": [10, 20, 30, 50, 80],
                "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                "reg_alpha": [0.0, 0.01, 0.05, 0.1, 0.5],
                "reg_lambda": [0.0, 0.01, 0.05, 0.1, 0.5, 1.0],
            }

        # xác định số class để set objective đúng
        n_classes = int(len(self.target_classes)) if self.target_classes is not None else None
        base_estimator = self._build_estimator_for_search(n_classes=n_classes)

        # Khởi tạo search
        search = RandomizedSearchCV(
            estimator=base_estimator,
            param_distributions=param_distributions,
            n_iter=int(n_iter),
            cv=int(cv),
            scoring=scoring,
            random_state=int(random_state),
            verbose=int(verbose),
            n_jobs=int(n_jobs),
        )

        # Fit search trên toàn bộ data
        search.fit(X_df, y_arr)

        # Lấy best params/score
        best_params = dict(search.best_params_)
        best_score = float(search.best_score_)

        # update params instance + init lại model theo best_params
        self.params.update(best_params)
        self._init_model(n_classes=n_classes)

        return {
            "best_params": best_params,
            "best_score": best_score,
            "cv": int(cv),
            "scoring": scoring,
            "n_iter": int(n_iter),
        }

    def get_feature_importance(self, top_k: Optional[int] = None) -> Dict[str, float]:
        """
        Lấy feature importance từ LightGBM wrapper.
        - Nếu chưa TRAINED => trả {}
        - top_k: nếu truyền => chỉ lấy top_k feature lớn nhất
        """
        if self.model is None or self.status != ModelStatus.TRAINED:
            return {}

        importances = getattr(self.model, "feature_importances_", None)
        if importances is None:
            return {}

        # Nếu vì lý do nào đó feature_names rỗng => fallback feature_0..n
        feats = self.feature_names or [f"feature_{i}" for i in range(len(importances))]

        # Ghép (feature_name, importance)
        pairs = list(zip(feats, np.array(importances, dtype=float)))

        # Sort giảm dần theo importance
        pairs.sort(key=lambda x: x[1], reverse=True)

        # Cắt top_k nếu cần
        if top_k is not None and int(top_k) > 0:
            pairs = pairs[: int(top_k)]

        # Trả về dict
        return {k: float(v) for k, v in pairs}

    def save(self, filepath: Optional[str] = None, include_metadata: bool = True) -> str:
        """
        Lưu model + metadata preprocessing ra joblib.

        Save gồm:
            - model (LGBMClassifier/Regressor)
            - schema feature_names
            - target_name
            - params
            - categorical mapping
            - datetime mapping
            - label encoder (classification)
            - evals_result_ + training history (nếu include_metadata)
        """
        if self.status != ModelStatus.TRAINED or self.model is None:
            raise ValueError("Model chưa được train")

        # Tạo thư mục lưu model nếu chưa có
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # Nếu không truyền đường dẫn => auto generate theo timestamp
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lightgbm_{self.task_type.value}_{timestamp}.joblib"
            filepath = str(MODEL_DIR / filename)

        # Gói dữ liệu cần dump
        save_data: Dict[str, Any] = {
            "model": self.model,
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "task_type": self.task_type.value,
            "params": self.params,
            "cat_features": self.cat_features,
            "_cat_categories": self._cat_categories,
            "datetime_cols": self.datetime_cols,
            "_datetime_feature_map": self._datetime_feature_map,
            "_is_target_encoded": self._is_target_encoded,
            "label_encoder": self.label_encoder if self._is_target_encoded else None,
            "target_classes": self.target_classes.tolist() if self.target_classes is not None else None,
            "evals_result_": self.evals_result_,
        }

        # Metadata bổ sung (không bắt buộc)
        if include_metadata:
            save_data["metadata"] = {
                "saved_at": datetime.now().isoformat(),
                "training_history": [r.to_dict() for r in self.training_history],
            }

        # Dump ra file
        joblib.dump(save_data, filepath)
        logger.info("LightGBM model saved to: %s", filepath)
        return filepath

    @classmethod
    def load(cls, filepath: str) -> "WeatherLightGBM":
        """
        Load model từ file joblib đã save().

        Flow:
            1) joblib.load
            2) tạo instance mới với task_type/params
            3) nạp lại model + metadata preprocess
            4) set status TRAINED
        """
        save_data = joblib.load(filepath)

        # Khởi tạo instance với params
        instance = cls(
            task_type=save_data["task_type"],
            params=save_data.get("params"),
            random_state=save_data.get("params", {}).get("random_state", 42),
        )

        # Restore estimator + metadata
        instance.model = save_data["model"]
        instance.feature_names = save_data.get("feature_names", [])
        instance.target_name = save_data.get("target_name")

        instance.cat_features = save_data.get("cat_features", [])
        instance._cat_categories = save_data.get("_cat_categories", {})

        instance.datetime_cols = save_data.get("datetime_cols", [])
        instance._datetime_feature_map = save_data.get("_datetime_feature_map", {})

        instance._is_target_encoded = bool(save_data.get("_is_target_encoded", False))
        instance.label_encoder = save_data.get("label_encoder", None)

        tc = save_data.get("target_classes")
        instance.target_classes = np.array(tc) if tc is not None else None

        instance.evals_result_ = save_data.get("evals_result_", {})

        # Mark trained
        instance.status = ModelStatus.TRAINED
        logger.info("LightGBM model loaded from: %s", filepath)
        return instance

    def export_latest_artifacts(
        self,
        *,
        artifacts_dir: Optional[Union[str, Path]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        train_info_extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Export artifacts sang Machine_learning_artifacts/latest để app/service dùng inference.

        Tạo ra:
            - Model.pkl (joblib dump wrapper chứa model + preprocess metadata)
            - Feature_list.json (list features)
            - Metrics.json (metric cuối)
            - Train_info.json (info train: params, task_type, best_iteration,...)
        """
        if self.status != ModelStatus.TRAINED or self.model is None:
            raise ValueError("Model chưa được train/load, không thể export artifacts.")

        # Nếu không truyền => dùng folder mặc định "latest"
        if artifacts_dir is None:
            artifacts_dir = LATEST_ARTIFACTS_DIR
        artifacts_dir = Path(artifacts_dir)

        # Tạo folder nếu chưa có
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Đường dẫn các file artifact
        model_path = artifacts_dir / "Model.pkl"
        features_path = artifacts_dir / "Feature_list.json"
        metrics_path = artifacts_dir / "Metrics.json"
        info_path = artifacts_dir / "Train_info.json"

        # 1) Dump model wrapper để inference có đủ mapping schema/categories/encoder
        joblib.dump(
            {
                "model": self.model,
                "task_type": self.task_type.value,
                "params": self.params,
                "feature_names": self.feature_names,
                "target_name": self.target_name,
                "cat_features": self.cat_features,
                "_cat_categories": self._cat_categories,
                "datetime_cols": self.datetime_cols,
                "_datetime_feature_map": self._datetime_feature_map,
                "_is_target_encoded": self._is_target_encoded,
                "label_encoder": self.label_encoder if self._is_target_encoded else None,
                "target_classes": self.target_classes.tolist() if self.target_classes is not None else None,
            },
            model_path,
        )

        # 2) Lưu danh sách feature
        with open(features_path, "w", encoding="utf-8") as f:
            json.dump({"features": self.feature_names}, f, ensure_ascii=False, indent=2)

        # 3) Lưu metrics: nếu không truyền, lấy metrics của lần train cuối
        if metrics is None:
            metrics = self.training_history[-1].metrics if self.training_history else {}

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        # 4) Lưu thông tin train
        info: Dict[str, Any] = {
            "algorithm": "LightGBM",
            "task_type": self.task_type.value,
            "target_name": self.target_name,
            "exported_at": datetime.now().isoformat(),
            "params": self.params,
            "cat_features": self.cat_features,
            "datetime_cols": self.datetime_cols,
            "best_iteration": (self.training_history[-1].best_iteration if self.training_history else None),
        }
        # nếu caller muốn bổ sung (dataset name, version, notes,...)
        if train_info_extra:
            info.update(train_info_extra)

        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        # Trả về path để debug/log
        return {
            "Model.pkl": str(model_path),
            "Feature_list.json": str(features_path),
            "Metrics.json": str(metrics_path),
            "Train_info.json": str(info_path),
        }

    @staticmethod
    def load_latest_artifacts(artifacts_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Load artifacts đã export_latest_artifacts() để inference nhanh.
        Trả về dict chứa:
            - model
            - feature_names
            - cat mapping
            - datetime mapping
            - label encoder (nếu classification)
        """
        if artifacts_dir is None:
            artifacts_dir = LATEST_ARTIFACTS_DIR
        artifacts_dir = Path(artifacts_dir)

        model_path = artifacts_dir / "Model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Không tìm thấy Model.pkl tại: {model_path}")

        return joblib.load(model_path)

    # ============================= PRIVATE =============================

    def _init_model(self, n_classes: Optional[int]) -> None:
        """
        Khởi tạo estimator LightGBM theo task.

        - Regression:
            objective=regression
            => LGBMRegressor
        - Classification:
            nếu n_classes > 2 => objective=multiclass + num_class
            else => objective=binary
            => LGBMClassifier
        """
        if self.task_type == TaskType.REGRESSION:
            p = self.params.copy()
            # đặt default objective nếu chưa có
            p.setdefault("objective", "regression")
            self.model = LGBMRegressor(**p)
            return

        # Classification
        p = self.params.copy()
        if n_classes is not None and int(n_classes) > 2:
            p.setdefault("objective", "multiclass")
            p["num_class"] = int(n_classes)
        else:
            p.setdefault("objective", "binary")
            # tránh để lại num_class từ config cũ
            p.pop("num_class", None)

        self.model = LGBMClassifier(**p)

    def _build_estimator_for_search(self, n_classes: Optional[int]):
        """
        Tạo estimator cho CV / RandomizedSearchCV.
        (Không dùng self.model trực tiếp để tránh side effects).
        """
        if self.task_type == TaskType.REGRESSION:
            p = self.params.copy()
            p.setdefault("objective", "regression")
            return LGBMRegressor(**p)

        p = self.params.copy()
        if n_classes is not None and int(n_classes) > 2:
            p["objective"] = "multiclass"
            p["num_class"] = int(n_classes)
        else:
            p["objective"] = "binary"
            p.pop("num_class", None)
        return LGBMClassifier(**p)

    def _prepare_target(self, y: Union[pd.Series, np.ndarray, List[Any]], fit: bool) -> np.ndarray:
        """
        Chuẩn hoá target y:
        - Regression: float
        - Classification:
            + fit=True: fit LabelEncoder và transform sang int
            + fit=False: nếu y là số => coi như đã encoded
                        nếu y là string => transform bằng encoder đã fit
        """
        # Cho phép user truyền list -> np.array
        if isinstance(y, list):
            y = np.array(y)

        # pandas Series -> numpy
        if isinstance(y, pd.Series):
            y = y.values

        y_arr = np.array(y)

        if self.task_type == TaskType.REGRESSION:
            # regression: ép float
            return y_arr.astype(float)

        # Classification
        # Nếu đang fit, hoặc chưa có encoder => fit encoder lần đầu
        if fit or (self.label_encoder is None and not self._is_target_encoded):
            self.label_encoder = LabelEncoder()
            # fit_transform: string labels -> ints
            y_enc = self.label_encoder.fit_transform(y_arr.astype(str))
            self._is_target_encoded = True
            self.target_classes = self.label_encoder.classes_
            return y_enc.astype(int)

        # fit=False:
        # nếu y_true đã là số => trả nguyên dạng int (dùng trong evaluate/cv)
        if np.issubdtype(y_arr.dtype, np.number):
            return y_arr.astype(int)

        # nếu y_true là string nhưng encoder chưa có => báo lỗi
        if self.label_encoder is None:
            raise ValueError(
                "Chưa có label_encoder nhưng y_true lại là string. "
                "Hãy train() trước hoặc gọi _prepare_target(fit=True)."
            )

        # transform string -> int bằng encoder đã fit
        return self.label_encoder.transform(y_arr.astype(str)).astype(int)

    def _decode_target(self, y_pred_encoded: np.ndarray) -> np.ndarray:
        """
        Decode output classification từ int -> label gốc.
        Regression thì trả nguyên.
        """
        if self.task_type != TaskType.CLASSIFICATION:
            return y_pred_encoded

        # Nếu đã encode target (có label_encoder) => inverse_transform về string label
        if self._is_target_encoded and self.label_encoder is not None:
            y_pred_encoded = np.array(y_pred_encoded, dtype=int)
            return self.label_encoder.inverse_transform(y_pred_encoded)

        # fallback: trả nguyên encoded
        return y_pred_encoded

    def _prepare_features(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        *,
        fit: bool,
        cat_features: Optional[List[Union[str, int]]] = None,
        datetime_cols: Optional[List[Union[str, int]]] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Chuẩn hoá features X:

        Steps:
            1) Convert ndarray -> DataFrame
            2) Replace inf -> NaN
            3) Datetime features: extract year/month/day/dow/hour/minute rồi drop cột gốc
            4) Categorical: cast sang pandas category + lưu danh sách category
            5) Align schema:
                - fit=True: lưu self.feature_names
                - fit=False: align theo self.feature_names (add missing, drop extra, reorder)
                  (nếu self.feature_names rỗng => tự set schema lần đầu để tránh rỗng)
        """
        # 1) To DataFrame
        if isinstance(X, np.ndarray):
            # Nếu user đưa ndarray thì tự sinh tên cột feature_0..feature_n
            df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        else:
            # DataFrame -> copy để không đụng dữ liệu gốc
            df = X.copy()

        # 2) clean inf: LightGBM hiểu NaN tốt hơn inf
        df = df.replace([np.inf, -np.inf], np.nan)

        # 3) datetime -> derived
        df = self._handle_datetime_features(df, fit=fit, datetime_cols=datetime_cols)

        # 4) categorical
        df = self._handle_categorical_features(df, fit=fit, cat_features=cat_features)

        # 5) schema align
        if fit or not self.feature_names:
            # fit=True => "đóng băng" schema cho các lần predict sau
            # hoặc chưa có schema (feature_names rỗng) => set schema lần đầu
            self.feature_names = df.columns.tolist()
            return df, self.feature_names

        # fit=False + có schema => align theo schema
        df = self._align_schema(df)
        return df, self.feature_names

    def _handle_datetime_features(
        self,
        df: pd.DataFrame,
        *,
        fit: bool,
        datetime_cols: Optional[List[Union[str, int]]] = None,
    ) -> pd.DataFrame:
        """
        Xử lý cột datetime:
            - xác định danh sách cột datetime (theo user truyền hoặc auto detect dtype)
            - với mỗi cột datetime:
                + convert to datetime nếu cần
                + tạo cột derived: year, month, day, dow, hour, minute
                + drop cột datetime gốc
            - fit=True: lưu self.datetime_cols và self._datetime_feature_map
        """
        out = df

        # determine dt cols
        dt_names: List[str] = []

        if datetime_cols:
            # user truyền datetime_cols dạng name hoặc index
            for c in datetime_cols:
                if isinstance(c, int):
                    # nếu là index, đổi sang tên cột
                    if 0 <= c < len(out.columns):
                        dt_names.append(out.columns[c])
                else:
                    # nếu là tên, kiểm tra tồn tại
                    c = str(c)
                    if c in out.columns:
                        dt_names.append(c)
        else:
            # auto detect theo dtype datetime64
            for col in out.columns:
                if pd.api.types.is_datetime64_any_dtype(out[col]):
                    dt_names.append(col)

        # fit=True: lưu danh sách datetime cols để predict dùng lại đúng cột
        if fit:
            self.datetime_cols = dt_names
            self._datetime_feature_map = {}

        # use_cols: nếu fit=False thì dùng self.datetime_cols (đã lưu)
        use_cols = self.datetime_cols if not fit else dt_names

        # loop từng cột datetime để tách feature
        for col in use_cols:
            # nếu cột không còn trong df (predict thiếu cột) => skip
            if col not in out.columns:
                continue

            # nếu dtype chưa phải datetime => convert
            if not pd.api.types.is_datetime64_any_dtype(out[col]):
                out[col] = pd.to_datetime(out[col], errors="coerce")

            prefix = str(col)
            derived_cols: List[str] = []

            # year
            out[f"{prefix}_year"] = out[col].dt.year
            derived_cols.append(f"{prefix}_year")

            # month
            out[f"{prefix}_month"] = out[col].dt.month
            derived_cols.append(f"{prefix}_month")

            # day
            out[f"{prefix}_day"] = out[col].dt.day
            derived_cols.append(f"{prefix}_day")

            # day of week (0=Mon..6=Sun)
            out[f"{prefix}_dow"] = out[col].dt.dayofweek
            derived_cols.append(f"{prefix}_dow")

            # hour
            out[f"{prefix}_hour"] = out[col].dt.hour
            derived_cols.append(f"{prefix}_hour")

            # minute
            out[f"{prefix}_minute"] = out[col].dt.minute
            derived_cols.append(f"{prefix}_minute")

            # fit=True: lưu mapping cột gốc -> danh sách cột derived
            if fit:
                self._datetime_feature_map[col] = derived_cols

            # drop cột datetime gốc (vì LightGBM không xử trực tiếp dạng datetime tốt bằng numeric)
            out = out.drop(columns=[col])

        return out

    def _handle_categorical_features(
        self,
        df: pd.DataFrame,
        *,
        fit: bool,
        cat_features: Optional[List[Union[str, int]]] = None,
    ) -> pd.DataFrame:
        """
        Xử lý cột categorical:
            - xác định danh sách cột categorical:
                + nếu user truyền => dùng theo name hoặc index
                + nếu không => auto detect (bool/category/object)
            - fit=True:
                + cast cột sang category
                + lưu categories list vào self._cat_categories
                + lưu self.cat_features
            - fit=False:
                + cast sang category
                + set_categories theo mapping train để đảm bảo ổn định khi predict
        """
        out = df

        cat_cols: List[str] = []

        if cat_features:
            # user truyền danh sách cột categorical dạng index hoặc name
            for c in cat_features:
                if isinstance(c, int):
                    if 0 <= c < len(out.columns):
                        cat_cols.append(out.columns[c])
                else:
                    c = str(c)
                    if c in out.columns:
                        cat_cols.append(c)
        else:
            # auto detect:
            # - bool
            # - category
            # - object (string)
            for col in out.columns:
                if (
                    pd.api.types.is_bool_dtype(out[col])
                    or pd.api.types.is_categorical_dtype(out[col])
                    or out[col].dtype == "object"
                ):
                    cat_cols.append(col)

        # fit=True: lưu danh sách categorical columns
        if fit:
            self.cat_features = sorted(set(cat_cols))
            self._cat_categories = {}

        # dùng cột đã lưu (fit=False) hoặc cột mới phát hiện (fit=True)
        use_cols = self.cat_features if not fit else sorted(set(cat_cols))

        # loop từng cột categorical để cast
        for col in use_cols:
            # nếu cột không tồn tại => skip
            if col not in out.columns:
                continue

            # convert object -> pandas "string" để xử lý hỗn hợp kiểu + giữ NaN tốt hơn
            if out[col].dtype == "object":
                out[col] = out[col].astype("string")

            if fit:
                # fit: cast category và lưu categories
                out[col] = out[col].astype("category")
                self._cat_categories[col] = out[col].cat.categories.tolist()
            else:
                # predict/eval: cast category và align categories theo train
                cats = self._cat_categories.get(col)

                # ép về string rồi sang category để tránh kiểu lẫn lộn
                out[col] = out[col].astype("string").astype("category")

                # set_categories để:
                # - category order/space giống train
                # - unseen category => NaN (LightGBM xử lý được)
                if cats is not None:
                    out[col] = out[col].cat.set_categories(cats)

        return out

    def _align_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Align schema DataFrame theo self.feature_names (schema lúc fit):
            - thiếu cột => add NaN
            - dư cột => drop
            - reorder theo đúng thứ tự self.feature_names

        Mục đích:
            - đảm bảo predict/evaluate luôn đúng shape và đúng order feature như lúc train
        """
        out = df.copy()
        expected = list(self.feature_names)

        # add missing columns
        for col in expected:
            if col not in out.columns:
                out[col] = np.nan

        # drop extra columns
        extra_cols = [c for c in out.columns if c not in expected]
        if extra_cols:
            out = out.drop(columns=extra_cols)

        # reorder columns đúng schema
        return out[expected]