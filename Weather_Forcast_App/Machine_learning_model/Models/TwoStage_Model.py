"""
TwoStage_Model.py — Two-stage model for zero-inflated rainfall prediction.

Architecture
────────────
  Stage 1 — Binary LightGBM Classifier:
    Input : X features (tất cả samples)
    Output: P(rain > threshold) — xác suất có mưa
    Objective: binary (cross-entropy), class_weight='balanced' (80% zero!!)

  Stage 2 — LightGBM Regressor (Tweedie loss):
    Input : X features (tất cả samples — Tweedie tự xử lý zeros)
    Output: E[rain | X] — lượng mưa kỳ vọng
    Objective: tweedie (thiết kế cho zero-inflated continuous data)

  Combined:
    final_pred = clf_proba * reg_pred
    → noise bị triệt tiêu ở những nơi model classifier không chắc
    → khi clf_proba < predict_threshold → output = 0 (hard-zero)

Interface tương thích với WeatherXGBoost / WeatherLightGBM:
    model.train(X, y, X_val=None, y_val=None, val_size=0.2, scale_features=False)
    model.predict(X)  →  PredictionResult(predictions=np.ndarray)
    model.save(path)  /  WeatherTwoStageModel.load(path)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from Weather_Forcast_App.Machine_learning_model.Models.Base_model import (
    PredictionResult,
    TrainingResult,
)


class WeatherTwoStageModel:
    """
    Two-stage model for zero-inflated rainfall prediction.

    Tham số
    ───────
    rain_threshold      : y > threshold → "rain" label cho Stage 1 classifier
    predict_threshold   : clf_proba < threshold → hard-zero (không mưa)
    n_estimators        : số cây cho cả 2 giai đoạn
    learning_rate       : learning rate chung
    max_depth           : max tree depth
    num_leaves          : LightGBM num_leaves (giúp fine-control complexity)
    min_child_samples   : tối thiểu samples trong leaf
    subsample           : row bagging
    colsample_bytree    : column bagging
    reg_alpha           : L1 regularization
    reg_lambda          : L2 regularization
    tweedie_variance_power : Tweedie power (1.0=Poisson, 1.5=compound-Poisson, 2.0=Gamma)
                             Với lượng mưa, 1.5 thường tối ưu nhất
    random_state        : seed
    clf_params          : dict override tham số riêng cho Stage 1 classifier
    reg_params          : dict override tham số riêng cho Stage 2 regressor
    """

    def __init__(
        self,
        # ── Ngưỡng phân loại
        rain_threshold: float = 0.1,
        predict_threshold: float = 0.3,
        # ── Hyperparams chung (dùng cho cả clf lẫn reg)
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        max_depth: int = 8,
        num_leaves: int = 63,
        min_child_samples: int = 10,
        subsample: float = 0.8,
        colsample_bytree: float = 0.7,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        tweedie_variance_power: float = 1.5,
        random_state: int = 42,
        # ── Override từng stage
        clf_params: Optional[Dict[str, Any]] = None,
        reg_params: Optional[Dict[str, Any]] = None,
        # ── Misc
        verbose: bool = True,
        task_type: str = "regression",  # accepted for compatibility, always regression
        **kwargs,  # absorb unknown kwargs from Ensemble / train.py
    ):
        self.rain_threshold = float(rain_threshold)
        self.predict_threshold = float(predict_threshold)
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.num_leaves = int(num_leaves)
        self.min_child_samples = int(min_child_samples)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.reg_alpha = float(reg_alpha)
        self.reg_lambda = float(reg_lambda)
        self.tweedie_variance_power = float(tweedie_variance_power)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)
        self._clf_params_override: Dict[str, Any] = clf_params or {}
        self._reg_params_override: Dict[str, Any] = reg_params or {}

        # Được set khi train()
        self.clf = None           # LGBMClassifier — Stage 1
        self.reg = None           # LGBMRegressor  — Stage 2
        self.is_trained: bool = False
        self.feature_names_: Optional[List[str]] = None
        self.train_metrics_: Optional[Dict[str, Any]] = None

    # ── Internal helpers ─────────────────────────────────────────────── #

    def _base_params(self) -> Dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbose": -1,
        }

    def _get_clf_params(self) -> Dict[str, Any]:
        p = self._base_params()
        p.update({
            "objective": "binary",
            "class_weight": "balanced",  # compensate 80% no-rain imbalance
        })
        p.update(self._clf_params_override)
        return p

    def _get_reg_params(self) -> Dict[str, Any]:
        p = self._base_params()
        p.update({
            "objective": "tweedie",
            "tweedie_variance_power": self.tweedie_variance_power,
        })
        p.update(self._reg_params_override)
        return p

    def _to_array(self, X, fit: bool = False) -> np.ndarray:
        """Convert X to float32 numpy array; drop datetime; align columns."""
        if isinstance(X, pd.DataFrame):
            # Drop datetime columns
            dt_cols = X.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns.tolist()
            if dt_cols:
                X = X.drop(columns=dt_cols)

            if fit:
                # First call during training: store feature schema
                self.feature_names_ = X.columns.tolist()
            elif self.feature_names_ is not None:
                # Align at inference time
                missing = [c for c in self.feature_names_ if c not in X.columns]
                if missing:
                    X = X.copy()
                    for c in missing:
                        X[c] = 0.0
                X = X[self.feature_names_]

            return X.values.astype(np.float32)

        return np.asarray(X, dtype=np.float32)

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae  = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2   = float(r2_score(y_true, y_pred))
        true_bin = (y_true > self.rain_threshold).astype(int)
        pred_bin = (y_pred > self.rain_threshold).astype(int)
        rain_acc = float((true_bin == pred_bin).mean())
        return {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Rain_Detection_Accuracy": rain_acc,
        }

    def _fit_callbacks(self, early_stop: int = 50):
        """Build LightGBM callbacks with fallback for older versions."""
        import lightgbm as lgb
        try:
            return [
                lgb.early_stopping(stopping_rounds=early_stop, verbose=False),
                lgb.log_evaluation(period=-1),
            ]
        except AttributeError:
            # LightGBM < 3.3 doesn't have these callables
            return None

    # ── Public API ───────────────────────────────────────────────────── #

    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        val_size: float = 0.2,
        scale_features: bool = False,   # WeatherTransformPipeline đã scale rồi
        verbose: bool = True,
        sample_weight: Optional[np.ndarray] = None,
        **kwargs,
    ) -> TrainingResult:
        """
        Train Two-Stage model.

        Parameters
        ──────────
        X, y          : tập training (đã được WeatherTransformPipeline scale)
        X_val, y_val  : tập validation (ưu tiên dùng nếu có, không split lại)
        val_size      : tỉ lệ split nếu X_val không có (0 → dùng X_train làm eval)
        scale_features: False vì pipeline upstream đã scale
        """
        from lightgbm import LGBMClassifier, LGBMRegressor
        from sklearn.model_selection import train_test_split

        t0 = time.time()
        verbose = verbose and self.verbose

        # ── 1. Chuẩn bị arrays ─────────────────────────────────────── #
        X_arr = self._to_array(X, fit=True)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1)

        # ── 2. Validation set ─────────────────────────────────────── #
        if X_val is not None and y_val is not None:
            X_val_arr = self._to_array(X_val, fit=False)
            y_val_arr = np.asarray(y_val, dtype=np.float32).reshape(-1)
        elif 0.0 < float(val_size) < 1.0:
            X_arr, X_val_arr, y_arr, y_val_arr = train_test_split(
                X_arr, y_arr,
                test_size=float(val_size),
                random_state=self.random_state,
            )
        else:
            # Fallback: eval on training data (happens when val_size=0 and no X_val)
            X_val_arr, y_val_arr = X_arr, y_arr

        n_train = len(X_arr)
        n_rain  = int((y_arr > self.rain_threshold).sum())
        zero_ratio = 1.0 - n_rain / max(n_train, 1)

        if verbose:
            print(f"[RAIN] Two-Stage Model - Training")
            print(f"   Train: {n_train} samples  ({n_rain} rain={1-zero_ratio:.1%}, {n_train-n_rain} dry={zero_ratio:.1%})")
            print(f"   Valid: {len(X_val_arr)} samples")
            print(f"   Features: {X_arr.shape[1]}")
            print(f"   Tweedie power: {self.tweedie_variance_power}")

        # ── 3. Stage 1: Binary Classifier ─────────────────────────── #
        y_bin_train = (y_arr     > self.rain_threshold).astype(np.int32)
        y_bin_val   = (y_val_arr > self.rain_threshold).astype(np.int32)

        if verbose:
            print(f"\n-- Stage 1: Classifier (rain vs no-rain, balanced weights)")

        callbacks_clf = self._fit_callbacks(early_stop=50)
        self.clf = LGBMClassifier(**self._get_clf_params())

        fit_clf_kwargs: Dict[str, Any] = {
            "eval_set": [(X_val_arr, y_bin_val)],
        }
        if callbacks_clf is not None:
            fit_clf_kwargs["callbacks"] = callbacks_clf
        else:
            fit_clf_kwargs["early_stopping_rounds"] = 50
            fit_clf_kwargs["verbose"] = False

        self.clf.fit(X_arr, y_bin_train, **fit_clf_kwargs)

        clf_proba_val = self.clf.predict_proba(X_val_arr)[:, 1]
        clf_pred_val  = (clf_proba_val >= self.predict_threshold).astype(int)
        clf_acc = float((clf_pred_val == y_bin_val).mean())

        if verbose:
            print(f"   Val Accuracy : {clf_acc:.4f}")
            print(f"   Val Precision: {(y_bin_val[clf_pred_val==1]).mean():.4f}" if clf_pred_val.sum() > 0 else "   Val Precision: N/A")
            print(f"   Val Recall   : {clf_pred_val[y_bin_val==1].mean():.4f}" if y_bin_val.sum() > 0 else "   Val Recall: N/A")

        # ── 4. Stage 2: Tweedie Regressor ─────────────────────────── #
        if verbose:
            print(f"\n-- Stage 2: Tweedie Regressor (power={self.tweedie_variance_power})")

        callbacks_reg = self._fit_callbacks(early_stop=50)
        self.reg = LGBMRegressor(**self._get_reg_params())

        fit_reg_kwargs: Dict[str, Any] = {
            "eval_set": [(X_val_arr, y_val_arr)],
        }
        if callbacks_reg is not None:
            fit_reg_kwargs["callbacks"] = callbacks_reg
        else:
            fit_reg_kwargs["early_stopping_rounds"] = 50
            fit_reg_kwargs["verbose"] = False

        if sample_weight is not None:
            fit_reg_kwargs["sample_weight"] = sample_weight

        self.reg.fit(X_arr, y_arr, **fit_reg_kwargs)

        # ── 5. Combined val metrics ─────────────────────────────────── #
        self.is_trained = True
        combined_val = self._combine(X_val_arr, clf_proba_override=clf_proba_val)
        metrics = self._compute_metrics(y_val_arr, combined_val)
        metrics["clf_val_accuracy"] = clf_acc
        metrics["n_train"] = n_train
        metrics["n_rain_train"] = n_rain
        metrics["zero_ratio"] = zero_ratio

        self.train_metrics_ = metrics

        if verbose:
            print(f"\n-- Combined Validation Results:")
            print(f"   R2:                    {metrics['R2']:.4f}")
            print(f"   MAE:                   {metrics['MAE']:.4f}")
            print(f"   RMSE:                  {metrics['RMSE']:.4f}")
            print(f"   Rain Detection Acc:    {metrics['Rain_Detection_Accuracy']:.4f}")
            print(f"   Total training time:   {time.time()-t0:.1f}s")

        return TrainingResult(success=True, metrics=metrics)

    def _combine(
        self,
        X_arr: np.ndarray,
        clf_proba_override: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Combined prediction (hard-switch):
          if clf_proba >= predict_threshold → return reg_pred
          else                             → return 0 (no rain)

        Hard-switch is preferred over soft-blend (clf_proba * reg_pred) because:
        - Soft-blend always attenuates rain predictions (prob < 1), causing
          systematic underestimation and misleadingly low train R².
        - Hard-switch is consistent with how TwoStage models evaluate
          internally (thresholded classifier → regressor).
        """
        clf_proba = clf_proba_override
        if clf_proba is None:
            clf_proba = self.clf.predict_proba(X_arr)[:, 1]

        reg_pred = np.clip(self.reg.predict(X_arr), 0.0, None)

        return np.where(clf_proba >= self.predict_threshold, reg_pred, 0.0)

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> PredictionResult:
        """
        Predict rainfall (mm).

        Returns PredictionResult với predictions = np.ndarray shape (n,).
        """
        if not self.is_trained or self.clf is None or self.reg is None:
            raise ValueError("Model chưa train. Gọi train() hoặc load() trước.")

        t0 = time.time()
        X_arr = self._to_array(X, fit=False)
        preds = self._combine(X_arr)

        return PredictionResult(
            predictions=preds,
            timestamp=float(time.time() - t0),
        )

    # ── Persistence ──────────────────────────────────────────────────── #

    def save(self, path: Union[str, Path]) -> None:
        """Lưu toàn bộ model (clf + reg + meta) bằng joblib."""
        import joblib
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "WeatherTwoStageModel":
        """Load model từ file (joblib dump)."""
        import joblib
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is {type(obj).__name__}, expected {cls.__name__}")
        return obj

    def __repr__(self) -> str:
        status = "trained" if self.is_trained else "not trained"
        return (
            f"WeatherTwoStageModel("
            f"n_estimators={self.n_estimators}, "
            f"tweedie_power={self.tweedie_variance_power}, "
            f"rain_threshold={self.rain_threshold}, "
            f"status={status})"
        )
