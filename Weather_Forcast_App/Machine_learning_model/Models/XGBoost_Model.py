# Weather_Forcast_App.Machine_learning_model.Models.XGBoost_Model
# =============================================================================
# FILE NÀY LÀ GÌ?
# - Đây là “wrapper” (lớp bọc) cho XGBoost, giúp bạn dùng XGBRegressor / XGBClassifier
#   theo một pipeline thống nhất cho dự án dự báo thời tiết.
# - Hỗ trợ:
#   + Regression + Classification
#   + Tự động xử lý datetime -> tách year/month/day/dow/hour/minute
#   + Tự động xử lý categorical -> one-hot (pd.get_dummies)
#   + Align schema khi predict (đảm bảo train/predict cùng số cột)
#   + Train/Evaluate/Predict/CV/Tuning/Save/Load
# - MỤC TIÊU: Bạn gọi train/predict dễ, không phải tự viết lại tiền xử lý mỗi lần.
# =============================================================================

from __future__ import annotations

import joblib
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# =============================================================================
# XGBoost imports (try/except):
# - Vì môi trường có thể chưa cài xgboost -> code không crash khi import module.
# - Nếu chưa cài, XGBOOST_AVAILABLE = False và __init__ sẽ raise ImportError.
# =============================================================================
try:
    import xgboost as xgb
    from xgboost import XGBClassifier, XGBRegressor

    XGBOOST_AVAILABLE = True
except Exception:  # pragma: no cover
    XGBOOST_AVAILABLE = False
    xgb = None
    XGBClassifier = None
    XGBRegressor = None

# =============================================================================
# scikit-learn utilities:
# - train_test_split: chia train/val
# - cross_val_score: đánh giá cross-validation
# - RandomizedSearchCV: tuning hyperparameters
# - metrics: đo chất lượng dự đoán
# - LabelEncoder: mã hoá nhãn phân loại (string -> int)
# =============================================================================
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


# =============================================================================
# DEFAULT_XGB_PARAMS:
# Đây là cấu hình mặc định (baseline) cho XGBoost.
# Bạn có thể override bằng params=, config= hoặc kwargs.
#
# Ý nghĩa các hyperparameters chính:
# - n_estimators: số lượng cây boosting (nhiều hơn có thể tốt hơn nhưng lâu hơn, dễ overfit nếu không có early stopping)
# - learning_rate: bước học (nhỏ -> cần nhiều cây hơn nhưng ổn định hơn)
# - max_depth: độ sâu cây (cao -> mô hình phức tạp hơn, dễ overfit)
# - min_child_weight: ngưỡng tối thiểu tổng “trọng số” trong node con (lớn -> giảm overfit)
# - subsample: tỉ lệ lấy mẫu hàng (row sampling) mỗi cây (giảm overfit)
# - colsample_bytree: tỉ lệ lấy mẫu cột (feature sampling) mỗi cây (giảm overfit)
# - reg_alpha: L1 regularization (sparse/giảm overfit)
# - reg_lambda: L2 regularization (giảm overfit)
# - tree_method="hist": thuật toán dựng cây dạng histogram (nhanh trên CPU, cũng là default tốt)
# - n_jobs=-1: dùng tất cả CPU cores
# - random_state: seed đảm bảo tái lập kết quả
# - verbosity: mức log của xgboost (0 là ít log)
# =============================================================================
DEFAULT_XGB_PARAMS: Dict[str, Any] = {
    # Core
    "n_estimators": 1200,
    "learning_rate": 0.03,
    "max_depth": 8,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,

    # Speed / stability
    "tree_method": "hist",   # good default on CPU
    "n_jobs": -1,
    "random_state": 42,
    "verbosity": 0,
}

# =============================================================================
# MODEL_DIR: thư mục lưu model sau khi train (joblib file).
# Path(__file__).parent.parent / "ml_models"
# - __file__ trỏ tới file hiện tại
# - parent.parent đi lên 2 cấp thư mục
# =============================================================================
MODEL_DIR = Path(__file__).parent.parent / "ml_models"


# ============================= MAIN CLASS =============================

class WeatherXGBoost:
    """
    Wrapper XGBoost (XGBRegressor / XGBClassifier)

    - Regression + Classification
    - Auto datetime features: year/month/day/dow/hour/minute
    - Auto categorical -> one-hot (get_dummies) và align schema khi predict
    - Train / Evaluate / Predict / CV / Tuning
    - Save/Load (joblib)

    Ghi chú: Lớp này quản lý “schema” feature_names để đảm bảo lúc predict,
    số cột và thứ tự cột giống lúc train (rất quan trọng khi dùng one-hot).
    """

    def __init__(
        self,
        task_type: str = "regression",
        params: Optional[Dict[str, Any]] = None,
        # BACKWARD COMPAT: test cũ hay truyền config=
        config: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        use_gpu: bool = False,
        **kwargs: Any,  # để không “chết” nếu ai truyền dư tham số
    ):
        # Nếu xgboost chưa cài thì raise ngay (đỡ lỗi âm thầm)
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost chưa được cài. Hãy chạy: pip install xgboost")

        # task_type: ép về Enum để đảm bảo chỉ có classification/regression
        self.task_type = TaskType(task_type.lower())
        self.random_state = int(random_state)
        self.use_gpu = bool(use_gpu)

        # base params lấy từ DEFAULT
        base = DEFAULT_XGB_PARAMS.copy()

        # Merge theo thứ tự: default -> params -> config -> kwargs
        # Ý nghĩa:
        # - params override default
        # - config override params (hỗ trợ code/test cũ)
        # - kwargs override mạnh nhất (ai truyền trực tiếp sẽ ưu tiên nhất)
        if params:
            base.update(params)
        if config:
            base.update(config)
        if kwargs:
            base.update(kwargs)

        # đảm bảo random_state đúng với constructor arg
        base["random_state"] = self.random_state

        # Nếu bật GPU:
        # - XGBoost phiên bản mới thường dùng device="cuda"
        # - tree_method="hist" + device cuda -> dùng GPU (tuỳ bản xgboost)
        if self.use_gpu:
            base.setdefault("device", "cuda")
            base.setdefault("tree_method", "hist")

        self.params: Dict[str, Any] = base
        self.status = ModelStatus.UNTRAINED

        # model sẽ là XGBRegressor hoặc XGBClassifier tuỳ task_type
        from typing import Any, Optional
        self.model: Optional[Any] = None

        # metadata để lưu / debug
        self.feature_names: List[str] = []
        self.target_name: Optional[str] = None

        # classification encoder:
        # - LabelEncoder để map class string -> int
        self.label_encoder: Optional[LabelEncoder] = None
        self._is_target_encoded: bool = False
        self.target_classes: Optional[np.ndarray] = None

        # feature engineering schema:
        # - datetime_cols/cat_features được “đóng băng” sau lần fit đầu tiên
        self.datetime_cols: List[str] = []
        self.cat_features: List[str] = []

        # lưu lịch sử các lần train
        self.training_history: List[TrainingResult] = []

        # Khởi tạo model lần đầu (chưa biết n_classes)
        self._init_model(n_classes=None)

    # ============================= PUBLIC =============================

    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, List[Any]],
        *,
        # target_name chỉ để lưu metadata (tên nhãn trong dataset)
        target_name: Optional[str] = None,
        # cat_features: list các cột categorical (tên cột hoặc index)
        cat_features: Optional[List[Union[str, int]]] = None,
        # datetime_cols: list cột datetime (tên cột hoặc index)
        datetime_cols: Optional[List[Union[str, int]]] = None,
        # val_size: tỉ lệ validation khi tự split (0.2 = 20%)
        val_size: float = 0.2,
        # shuffle: có xáo dữ liệu trước khi split không (time-series thường False)
        shuffle: bool = False,
        # stratify: giữ tỉ lệ lớp khi split (chỉ áp dụng cho classification + shuffle=True)
        stratify: bool = True,
        # early_stopping_rounds: nếu metric không cải thiện trong N vòng -> dừng
        early_stopping_rounds: int = 50,
        # eval_metric: metric dùng để theo dõi (rmse/logloss/mlogloss...)
        eval_metric: Optional[Union[str, List[str]]] = None,
        # verbose: xgboost in log trong quá trình fit
        verbose: bool = False,
        # X_val/y_val: cho phép bạn tự đưa validation vào (chuẩn cho time-series)
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray, List[Any]]] = None,
    ) -> TrainingResult:
        start_time = time.time()
        self.target_name = target_name

        try:
            # =============================================================
            # (1) PREPARE FEATURES
            # - fit=True nghĩa là “đóng băng schema”:
            #   + xác định datetime_cols/cat_features
            #   + one-hot + tách datetime
            #   + lưu self.feature_names để sau này align khi predict
            # =============================================================
            X_df, feature_names = self._prepare_features(
                X,
                fit=True,
                cat_features=cat_features,
                datetime_cols=datetime_cols,
            )

            # =============================================================
            # (2) PREPARE TARGET
            # - Regression: ép float
            # - Classification:
            #   + fit=True -> LabelEncoder.fit_transform
            #   + lưu classes_ để decode khi predict
            # =============================================================
            y_arr = self._prepare_target(y, fit=True)

            # =============================================================
            # (3) SPLIT TRAIN/VAL
            # - Nếu user truyền X_val/y_val => dùng luôn (ưu tiên nhất)
            # - Nếu không truyền:
            #   + val_size không hợp lệ (<=0 hoặc >=1) -> không split
            #   + else -> train_test_split theo shuffle
            #   + nếu classification + shuffle + stratify -> stratify=y
            # =============================================================
            if X_val is not None and y_val is not None:
                # User đã có validation set riêng
                X_train_df = X_df
                y_train = y_arr

                # fit=False: không “học schema” nữa, chỉ biến đổi + align theo schema đã chốt
                X_val_df, _ = self._prepare_features(
                    X_val,
                    fit=False,
                    cat_features=self.cat_features,
                    datetime_cols=self.datetime_cols,
                )
                # fit=False: dùng label_encoder đã fit để transform (classification)
                y_val_arr = self._prepare_target(y_val, fit=False)
            else:
                # Không có X_val/y_val -> tự split dựa trên val_size
                if not (0.0 < float(val_size) < 1.0):
                    # Không split: dùng toàn bộ làm train, val=None
                    X_train_df = X_df
                    y_train = y_arr
                    X_val_df = None
                    y_val_arr = None
                else:
                    split_kwargs: Dict[str, Any] = {
                        "test_size": float(val_size),
                        "random_state": self.random_state,
                        "shuffle": bool(shuffle),
                    }
                    # Stratify chỉ dùng cho classification để giữ tỷ lệ class ổn định
                    if self.task_type == TaskType.CLASSIFICATION and shuffle and stratify:
                        split_kwargs["stratify"] = y_arr

                    X_train_df, X_val_df, y_train, y_val_arr = train_test_split(
                        X_df, y_arr, **split_kwargs
                    )

            # =============================================================
            # (4) RE-INIT MODEL THEO SỐ CLASS
            # - Với classification multi-class: cần num_class
            # - Với binary: không cần num_class
            # =============================================================
            n_classes = int(len(self.target_classes)) if self.target_classes is not None else None
            self._init_model(n_classes=n_classes)

            # =============================================================
            # (5) SET eval_metric + early_stopping (tương thích version)
            # - Nếu eval_metric None: auto chọn theo task:
            #   + regression -> rmse
            #   + classification -> logloss hoặc mlogloss
            # - Một số phiên bản xgboost đặt eval_metric/early_stopping_rounds
            #   qua set_params thay vì truyền trực tiếp vào fit.
            # =============================================================
            #    (xgboost mới không nhận eval_metric/early_stopping_rounds trong fit) :contentReference[oaicite:3]{index=3}
            if eval_metric is None:
                if self.task_type == TaskType.REGRESSION:
                    eval_metric = "rmse"
                else:
                    eval_metric = "logloss" if (n_classes is None or n_classes <= 2) else "mlogloss"

            # set_params(eval_metric=...) có thể fail tuỳ version -> try/except để “mềm”
            try:
                self.model.set_params(eval_metric=eval_metric)
            except Exception:
                pass

            # Only set early_stopping_rounds if validation is present
            if X_val_df is not None and y_val_arr is not None and early_stopping_rounds and early_stopping_rounds > 0:
                try:
                    self.model.set_params(early_stopping_rounds=int(early_stopping_rounds))
                except Exception:
                    pass

            # =============================================================
            # (6) FIT
            # - fit_kwargs:
            #   + verbose: in log quá trình train
            #   + eval_set: chỉ đưa vào nếu có validation
            # =============================================================
            fit_kwargs: Dict[str, Any] = {"verbose": bool(verbose)}
            if X_val_df is not None and y_val_arr is not None:
                fit_kwargs["eval_set"] = [(X_val_df, y_val_arr)]
                self.model.fit(X_train_df, y_train, **fit_kwargs)
            else:
                # No validation: fit with only X_train_df, y_train and minimal kwargs (no early stopping)
                minimal_fit_kwargs = {k: v for k, v in fit_kwargs.items() if k in ["verbose"]}
                self.model.fit(X_train_df, y_train, **minimal_fit_kwargs)

            # =============================================================
            # (7) UPDATE STATUS
            # - đánh dấu model đã train để các hàm evaluate/predict không bị chặn
            # =============================================================
            self.status = ModelStatus.TRAINED

            # =============================================================
            # (8) COLLECT METRICS + FEATURE IMPORTANCE
            # - best_iteration: nếu có early stopping, model thường có best_iteration
            # - metrics: evaluate trên validation nếu có, ngược lại evaluate trên train
            # - feature_importances: lấy từ model.feature_importances_
            # =============================================================
            best_iter = getattr(self.model, "best_iteration", None)
            metrics = self.evaluate(
                X_val_df if (X_val_df is not None and y_val_arr is not None) else X_train_df,
                y_val_arr if (X_val_df is not None and y_val_arr is not None) else y_train,
                return_details=True,
            )
            feature_importances = self.get_feature_importance()

            train_time = time.time() - start_time
            result = TrainingResult(
                success=True,
                metrics=metrics,
                training_time=float(train_time),
                n_samples=int(X_train_df.shape[0]),
                n_features=int(X_train_df.shape[1]),
                feature_names=feature_names,
                feature_importances=feature_importances,
                best_iteration=int(best_iter) if best_iter is not None else None,
                message="Train XGBoost thành công",
            )

            # lưu history để sau xem lại (so sánh các lần train)
            self.training_history.append(result)
            return result

        except Exception as e:
            # Nếu lỗi: set status FAILED + log stacktrace để debug
            self.status = ModelStatus.FAILED
            msg = f"Train XGBoost thất bại: {e}"
            logger.exception(msg)
            return TrainingResult(success=False, message=msg)

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        *,
        # return_proba: nếu classification và muốn lấy xác suất từng lớp
        return_proba: bool = False,
    ) -> PredictionResult:
        # Guard: phải train hoặc load trước khi predict
        if self.status != ModelStatus.TRAINED or self.model is None:
            raise ValueError("Model chưa được train hoặc load.")

        start = time.time()

        # fit=False: biến đổi features theo schema đã chốt lúc train + align cột
        X_df, _ = self._prepare_features(X, fit=False, cat_features=self.cat_features, datetime_cols=self.datetime_cols)

        probs = None
        if self.task_type == TaskType.CLASSIFICATION:
            # Nếu user muốn probabilities (ma trận n_samples x n_classes)
            if return_proba:
                probs = self.model.predict_proba(X_df)
            # model.predict trả label dạng encoded (int) khi classifier
            pred_encoded = self.model.predict(X_df)
            # decode về nhãn gốc (string) nếu trước đó đã LabelEncoder
            pred = self._decode_target(np.array(pred_encoded, dtype=int))
        else:
            # regression -> dự đoán trực tiếp số thực
            pred = self.model.predict(X_df)

        return PredictionResult(
            predictions=np.array(pred),
            probabilities=probs,
            timestamp=float(time.time() - start),
        )

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        # Hàm tiện ích: chỉ cho classification
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
        # return_details: nếu True -> thêm confusion_matrix + classification_report
        return_details: bool = False,
    ) -> Dict[str, Any]:
        # Guard: chỉ evaluate khi model đã train/load
        if self.status != ModelStatus.TRAINED or self.model is None:
            raise ValueError("Model chưa được train hoặc load.")

        # Chuẩn hoá feature theo schema đã chốt
        X_df, _ = self._prepare_features(X, fit=False, cat_features=self.cat_features, datetime_cols=self.datetime_cols)
        # Chuẩn hoá y_true:
        # - regression -> float
        # - classification -> encode giống lúc train
        y_arr = self._prepare_target(y_true, fit=False)

        if self.task_type == TaskType.REGRESSION:
            # Regression metrics:
            # - mse: mean squared error
            # - rmse: sqrt(mse) (thường trực quan hơn)
            # - mae: mean absolute error
            # - r2: hệ số xác định
            # - mape_percent: % sai số tuyệt đối trung bình (cẩn thận khi y gần 0)
            y_pred = self.model.predict(X_df)
            mse = mean_squared_error(y_arr, y_pred)
            rmse = float(np.sqrt(mse))
            mae = mean_absolute_error(y_arr, y_pred)
            r2 = r2_score(y_arr, y_pred)

            # Tránh chia 0 trong MAPE
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

        # Classification:
        # - model.predict -> class encoded (int)
        y_pred_encoded = self.model.predict(X_df)
        y_pred = np.array(y_pred_encoded, dtype=int)

        # Macro metrics: trung bình đều các lớp (hữu ích khi mất cân bằng)
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

        # return_details=True -> thêm báo cáo chi tiết để debug model
        if return_details:
            out["confusion_matrix"] = confusion_matrix(y_arr, y_pred).tolist()
            out["classification_report"] = classification_report(y_arr, y_pred, zero_division=0)

        return out

    def cross_validate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, List[Any]],
        *,
        # cv: số fold (5 = phổ biến)
        cv: int = 5,
        # scoring: metric của sklearn (r2/accuracy/neg_rmse...)
        scoring: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Chuẩn hoá features:
        # - fit=(not bool(self.feature_names)) nghĩa là:
        #   nếu chưa có feature_names (chưa train lần nào) thì cho phép “fit schema”
        X_df, _ = self._prepare_features(X, fit=(not bool(self.feature_names)))
        # Chuẩn hoá target:
        # - classification: nếu chưa encode thì encode
        y_arr = self._prepare_target(y, fit=(self.task_type == TaskType.CLASSIFICATION and not self._is_target_encoded))

        # xác định số class để build estimator phù hợp
        n_classes = int(len(self.target_classes)) if self.target_classes is not None else None
        estimator = self._build_estimator_for_search(n_classes=n_classes)

        # scoring default theo task
        if scoring is None:
            scoring = "r2" if self.task_type == TaskType.REGRESSION else "accuracy"

        # cross_val_score trả list score theo từng fold
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
        # n_iter: số lần thử ngẫu nhiên trong RandomizedSearch
        n_iter: int = 30,
        # cv: số fold cho tuning
        cv: int = 3,
        # scoring: metric dùng để chọn best params
        scoring: Optional[str] = None,
        # random_state: seed riêng cho search (nếu None dùng self.random_state)
        random_state: Optional[int] = None,
        # param_distributions: không truyền -> dùng grid mặc định bên dưới
        param_distributions: Optional[Dict[str, Any]] = None,
        verbose: int = 1,
        n_jobs: int = -1,
    ) -> Dict[str, Any]:
        # Chuẩn hoá feature và target tương tự cross_validate
        X_df, _ = self._prepare_features(X, fit=(not bool(self.feature_names)))
        y_arr = self._prepare_target(y, fit=(self.task_type == TaskType.CLASSIFICATION and not self._is_target_encoded))

        if random_state is None:
            random_state = self.random_state

        # scoring default:
        # - regression: neg_root_mean_squared_error (sklearn dùng “maximize”, nên RMSE phải neg)
        # - classification: accuracy
        if scoring is None:
            scoring = "neg_root_mean_squared_error" if self.task_type == TaskType.REGRESSION else "accuracy"

        # Nếu user không đưa param_distributions -> dùng bộ search “an toàn”
        if param_distributions is None:
            param_distributions = {
                "n_estimators": [300, 600, 1200, 2000],
                "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.1],
                "max_depth": [4, 6, 8, 10, 12],
                "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                "reg_alpha": [0.0, 0.01, 0.05, 0.1, 0.5],
                "reg_lambda": [0.5, 1.0, 2.0, 5.0],
                "min_child_weight": [1, 2, 5, 10],
            }

        n_classes = int(len(self.target_classes)) if self.target_classes is not None else None
        base_estimator = self._build_estimator_for_search(n_classes=n_classes)

        # RandomizedSearchCV:
        # - thử n_iter cấu hình ngẫu nhiên
        # - dùng cv folds
        # - chọn best theo scoring
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

        search.fit(X_df, y_arr)

        best_params = dict(search.best_params_)
        best_score = float(search.best_score_)

        # Sau khi tuning, update self.params và init lại model để dùng params mới
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
        # Guard: chỉ lấy importance khi model trained
        if self.model is None or self.status != ModelStatus.TRAINED:
            return {}

        importances = getattr(self.model, "feature_importances_", None)
        if importances is None:
            return {}

        # feats: ưu tiên feature_names đã lưu; nếu không có thì tự tạo feature_i
        feats = self.feature_names or [f"feature_{i}" for i in range(len(importances))]
        pairs = list(zip(feats, np.array(importances, dtype=float)))
        pairs.sort(key=lambda x: x[1], reverse=True)

        # top_k: nếu truyền -> cắt lấy top K feature quan trọng nhất
        if top_k is not None and int(top_k) > 0:
            pairs = pairs[: int(top_k)]

        return {k: float(v) for k, v in pairs}

    def save(self, filepath: Optional[str] = None) -> str:
        # Chỉ cho save khi model đã train
        if self.status != ModelStatus.TRAINED or self.model is None:
            raise ValueError("Model chưa được train")

        # đảm bảo thư mục tồn tại
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # Nếu không chỉ định filepath -> tạo tên file theo timestamp
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"xgboost_{self.task_type.value}_{timestamp}.joblib"
            filepath = str(MODEL_DIR / filename)

        # joblib.dump: lưu cả model + metadata cần thiết để load lại và predict đúng schema
        joblib.dump(
            {
                "model": self.model,
                "task_type": self.task_type.value,
                "params": self.params,
                "feature_names": self.feature_names,
                "target_name": self.target_name,
                "datetime_cols": self.datetime_cols,
                "cat_features": self.cat_features,
                "_is_target_encoded": self._is_target_encoded,
                "label_encoder": self.label_encoder if self._is_target_encoded else None,
                "target_classes": self.target_classes.tolist() if self.target_classes is not None else None,
            },
            filepath,
        )
        return filepath

    @classmethod
    def load(cls, filepath: str) -> "WeatherXGBoost":
        # load file joblib
        data = joblib.load(filepath)
        # tạo instance mới với task_type + params cũ
        inst = cls(
            task_type=data["task_type"],
            params=data.get("params"),
            random_state=data.get("params", {}).get("random_state", 42),
        )
        # restore model + metadata
        inst.model = data["model"]
        inst.feature_names = data.get("feature_names", [])
        inst.target_name = data.get("target_name")
        inst.datetime_cols = data.get("datetime_cols", [])
        inst.cat_features = data.get("cat_features", [])

        inst._is_target_encoded = bool(data.get("_is_target_encoded", False))
        inst.label_encoder = data.get("label_encoder", None)
        tc = data.get("target_classes")
        inst.target_classes = np.array(tc) if tc is not None else None

        inst.status = ModelStatus.TRAINED
        return inst

    # BACKWARD COMPAT alias (cho code/test cũ)
    def save_model(self, filepath: str) -> str:
        return self.save(filepath)

    def load_model(self, filepath: str) -> None:
        # load rồi copy toàn bộ __dict__ để object hiện tại trở thành object đã load
        loaded = self.load(filepath)
        self.__dict__.update(loaded.__dict__)

    @property
    def is_trained(self) -> bool:
        # property tiện: model đã train chưa?
        return self.status == ModelStatus.TRAINED

    # ============================= PRIVATE =============================

    def _init_model(self, n_classes: Optional[int]) -> None:
        # Hàm này chọn đúng “loại model” (regressor/classifier) + objective/metric
        if self.task_type == TaskType.REGRESSION:
            # Regression:
            # - objective: reg:squarederror (MSE)
            # - eval_metric: rmse (thước đo phổ biến)
            p = self.params.copy()
            p.setdefault("objective", "reg:squarederror")
            p.setdefault("eval_metric", "rmse")
            self.model = XGBRegressor(**p)
            return

        # Classification:
        p = self.params.copy()
        if n_classes is not None and int(n_classes) > 2:
            # Multi-class:
            # - objective multi:softprob -> trả xác suất cho từng lớp
            # - num_class bắt buộc để biết số lớp
            # - eval_metric mlogloss
            p.setdefault("objective", "multi:softprob")
            p["num_class"] = int(n_classes)
            p.setdefault("eval_metric", "mlogloss")
        else:
            # Binary:
            # - objective binary:logistic
            # - không cần num_class
            # - eval_metric logloss
            p.setdefault("objective", "binary:logistic")
            p.pop("num_class", None)
            p.setdefault("eval_metric", "logloss")

        self.model = XGBClassifier(**p)

    def _build_estimator_for_search(self, n_classes: Optional[int]):
        # Giống _init_model nhưng trả estimator “tạm” để dùng trong CV/search
        if self.task_type == TaskType.REGRESSION:
            p = self.params.copy()
            p.setdefault("objective", "reg:squarederror")
            p.setdefault("eval_metric", "rmse")
            return XGBRegressor(**p)

        p = self.params.copy()
        if n_classes is not None and int(n_classes) > 2:
            p["objective"] = "multi:softprob"
            p["num_class"] = int(n_classes)
            p.setdefault("eval_metric", "mlogloss")
        else:
            p["objective"] = "binary:logistic"
            p.pop("num_class", None)
            p.setdefault("eval_metric", "logloss")
        return XGBClassifier(**p)

    def _prepare_target(self, y: Union[pd.Series, np.ndarray, List[Any]], fit: bool) -> np.ndarray:
        # Chuẩn hoá y về numpy array để xử lý thống nhất
        if isinstance(y, list):
            y = np.array(y)
        if isinstance(y, pd.Series):
            y = y.values

        y_arr = np.array(y)

        # Regression: ép float (đảm bảo metric tính đúng)
        if self.task_type == TaskType.REGRESSION:
            return y_arr.astype(float)

        # Classification:
        # - Nếu fit=True (lúc train) hoặc chưa có label_encoder -> tạo encoder và fit_transform
        # - Lưu classes_ để decode dự đoán về label gốc
        if fit or (self.label_encoder is None and not self._is_target_encoded):
            self.label_encoder = LabelEncoder()
            y_enc = self.label_encoder.fit_transform(y_arr.astype(str))
            self._is_target_encoded = True
            self.target_classes = self.label_encoder.classes_
            return y_enc.astype(int)

        # Nếu y_arr đã là số -> coi như đã encoded sẵn
        if np.issubdtype(y_arr.dtype, np.number):
            return y_arr.astype(int)

        # Nếu y_true là string mà chưa có encoder -> không thể transform
        if self.label_encoder is None:
            raise ValueError("Chưa có label_encoder nhưng y_true lại là string. Hãy train() trước.")
        return self.label_encoder.transform(y_arr.astype(str)).astype(int)

    def _decode_target(self, y_pred_encoded: np.ndarray) -> np.ndarray:
        # Chỉ decode cho classification
        if self.task_type != TaskType.CLASSIFICATION:
            return y_pred_encoded
        # Nếu trước đó đã encode bằng LabelEncoder -> inverse_transform về nhãn gốc
        if self._is_target_encoded and self.label_encoder is not None:
            y_pred_encoded = np.array(y_pred_encoded, dtype=int)
            return self.label_encoder.inverse_transform(y_pred_encoded)
        return y_pred_encoded

    def _prepare_features(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        *,
        fit: bool,
        cat_features: Optional[List[Union[str, int]]] = None,
        datetime_cols: Optional[List[Union[str, int]]] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        # =============================================================
        # Pipeline feature:
        # (1) convert X -> DataFrame
        # (2) replace inf -> nan
        # (3) datetime feature engineering
        # (4) categorical one-hot
        # (5) align schema (nếu fit=False)
        # =============================================================

        # 1) To DataFrame
        if isinstance(X, np.ndarray):
            # Nếu là ndarray thì không có tên cột -> tự tạo feature_0..n
            df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        else:
            # DataFrame -> copy để tránh mutate dữ liệu gốc bên ngoài
            df = X.copy()

        # 2) clean inf
        # Inf/-Inf thường gây lỗi cho model/metrics -> đổi thành NaN để xử lý tiếp
        df = df.replace([np.inf, -np.inf], np.nan)

        # 3) datetime -> derived
        # Tách datetime thành các cột số (year/month/day/dow/hour/minute)
        df = self._handle_datetime_features(df, fit=fit, datetime_cols=datetime_cols)

        # 4) categorical -> one-hot
        # Categorical -> pd.get_dummies để model hiểu dưới dạng số
        df = self._handle_categorical_features(df, fit=fit, cat_features=cat_features)

        # 5) schema align
        # Nếu fit=True hoặc chưa có feature_names -> “đóng băng” schema mới
        if fit or not self.feature_names:
            self.feature_names = df.columns.tolist()
            return df, self.feature_names

        # Nếu fit=False -> align theo schema cũ:
        # - thêm cột thiếu = 0
        # - bỏ cột thừa
        # - reorder đúng thứ tự
        df = self._align_schema(df)
        return df, self.feature_names

    def _handle_datetime_features(
        self,
        df: pd.DataFrame,
        *,
        fit: bool,
        datetime_cols: Optional[List[Union[str, int]]] = None,
    ) -> pd.DataFrame:
        out = df

        # dt_names: danh sách cột datetime sẽ xử lý
        dt_names: List[str] = []
        if datetime_cols:
            # Nếu user chỉ định danh sách datetime_cols (theo index hoặc tên)
            for c in datetime_cols:
                if isinstance(c, int):
                    # index -> map sang tên cột
                    if 0 <= c < len(out.columns):
                        dt_names.append(out.columns[c])
                else:
                    # tên cột
                    c = str(c)
                    if c in out.columns:
                        dt_names.append(c)
        else:
            # Nếu user không chỉ định -> tự detect cột dtype datetime64
            for col in out.columns:
                if pd.api.types.is_datetime64_any_dtype(out[col]):
                    dt_names.append(col)

        # Nếu fit=True -> “chốt” datetime_cols vào self.datetime_cols để dùng về sau
        if fit:
            self.datetime_cols = dt_names

        # Nếu fit=False -> dùng datetime_cols đã chốt ở lần fit
        use_cols = self.datetime_cols if not fit else dt_names

        # Với mỗi cột datetime:
        # - ép về datetime nếu chưa đúng dtype
        # - tách thành các thành phần thời gian
        # - drop cột gốc
        for col in use_cols:
            if col not in out.columns:
                continue
            if not pd.api.types.is_datetime64_any_dtype(out[col]):
                out[col] = pd.to_datetime(out[col], errors="coerce")

            prefix = str(col)
            out[f"{prefix}_year"] = out[col].dt.year
            out[f"{prefix}_month"] = out[col].dt.month
            out[f"{prefix}_day"] = out[col].dt.day
            out[f"{prefix}_dow"] = out[col].dt.dayofweek
            out[f"{prefix}_hour"] = out[col].dt.hour
            out[f"{prefix}_minute"] = out[col].dt.minute
            out = out.drop(columns=[col])

        return out

    def _handle_categorical_features(
        self,
        df: pd.DataFrame,
        *,
        fit: bool,
        cat_features: Optional[List[Union[str, int]]] = None,
    ) -> pd.DataFrame:
        out = df

        # cat_cols: danh sách cột categorical sẽ one-hot
        cat_cols: List[str] = []
        if cat_features:
            # User chỉ định cột categorical (index hoặc tên)
            for c in cat_features:
                if isinstance(c, int):
                    if 0 <= c < len(out.columns):
                        cat_cols.append(out.columns[c])
                else:
                    c = str(c)
                    if c in out.columns:
                        cat_cols.append(c)
        else:
            # Auto detect categorical:
            # - bool
            # - category dtype
            # - object (string)
            for col in out.columns:
                if (
                    pd.api.types.is_bool_dtype(out[col])
                    or pd.api.types.is_categorical_dtype(out[col])
                    or out[col].dtype == "object"
                ):
                    cat_cols.append(col)

        # fit=True: chốt danh sách categorical vào self.cat_features
        if fit:
            self.cat_features = sorted(set(cat_cols))

        # fit=False: dùng danh sách cat_features đã chốt lúc train
        use_cols = self.cat_features if not fit else sorted(set(cat_cols))

        # one-hot cho đúng với mọi version xgboost:
        # - dummy_na=True: tạo thêm cột cho NaN (tránh mất thông tin missing)
        if use_cols:
            out = pd.get_dummies(out, columns=[c for c in use_cols if c in out.columns], dummy_na=True)

        return out

    def _align_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        # Mục tiêu:
        # - Đảm bảo df có đúng các cột như feature_names lúc train
        # - Nếu thiếu cột -> thêm cột = 0
        # - Nếu thừa cột -> drop
        # - Reorder đúng thứ tự (rất quan trọng)
        out = df.copy()
        expected = list(self.feature_names)

        # Thêm cột thiếu
        for col in expected:
            if col not in out.columns:
                out[col] = 0  # dummy col thiếu -> 0

        # Drop cột thừa (có thể xuất hiện nếu data predict có category mới)
        extra_cols = [c for c in out.columns if c not in expected]
        if extra_cols:
            out = out.drop(columns=extra_cols)

        # reorder theo expected
        return out[expected]


# Alias để test/import kiểu cũ không bị vỡ
WeatherXGBoostModel = WeatherXGBoost