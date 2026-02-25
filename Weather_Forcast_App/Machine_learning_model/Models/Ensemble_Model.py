from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


ArrayLike = Union[np.ndarray, List[float]]


class WeatherEnsembleModel:
    """
    Ensemble model cho bài toán regression (dự báo thời tiết):
    - Train nhiều base models (wrapper hoặc sklearn estimator).
    - Lưu metrics (mse/rmse/mae/r2) cho từng model + ensemble.
    - Ensemble prediction: mean hoặc weighted_mean (ổn định & dễ kiểm soát).

    base_models config ví dụ:
    [
      {"type": "lightgbm", "params": {...}},
      {"type": "xgboost",  "params": {...}, "X_val": X_val, "y_val": y_val},
    ]

    model_registry ví dụ:
    {
      "lightgbm": "your_pkg.models.WeatherLightGBM",
      "xgboost":  "your_pkg.models.WeatherXGBoost",
    }
    """

    def __init__(
        self,
        base_models: List[Dict[str, Any]],
        model_registry: Dict[str, str],
        ensemble_mode: str = "mean",  # "mean" | "weighted_mean"
        weights: Optional[List[float]] = None,
        seed: int = 42,
        task_type: str = "regression",
        drop_datetime_cols: bool = True,
    ):
        self.base_models_cfg = base_models
        self.model_registry = {k.lower().strip(): v for k, v in (model_registry or {}).items()}

        self.ensemble_mode = (ensemble_mode or "mean").lower().strip()
        self.weights = weights
        self.seed = seed
        self.task_type = task_type
        self.drop_datetime_cols = drop_datetime_cols

        self.models: List[Any] = []
        self.model_metrics: Dict[str, Dict[str, float]] = {}
        self.is_trained: bool = False

    # -------------------------
    # Utils
    # -------------------------
    def _drop_datetime(self, X):
        # Chỉ drop nếu X là pandas DataFrame (có select_dtypes)
        if not self.drop_datetime_cols:
            return X
        if hasattr(X, "select_dtypes") and hasattr(X, "drop"):
            dt_cols = X.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns
            if len(dt_cols) > 0:
                return X.drop(columns=dt_cols)
        return X

    def _compute_metrics(self, y_true: ArrayLike, y_pred: ArrayLike) -> Dict[str, float]:
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)

        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return {"mse": float(mse), "rmse": rmse, "mae": float(mae), "r2_score": float(r2)}

    def _extract_prediction(self, pred: Any) -> np.ndarray:
        # Hỗ trợ output kiểu wrapper: pred.predictions / pred.prediction / pred.pred
        for attr in ("predictions", "prediction", "pred"):
            if hasattr(pred, attr):
                pred = getattr(pred, attr)
                break
        arr = np.asarray(pred)
        return arr.reshape(-1)

    def _sanitize_params(self, params: Dict[str, Any], model_type: Optional[str] = None) -> Dict[str, Any]:
        params = dict(params or {})

        # Remove các key hay gây lỗi clone/fit linh tinh
        for k in ("early_stopping_rounds", "eval_metric", "eval_set", "evals", "callbacks"):
            params.pop(k, None)

        model_type = (model_type or "").lower().strip() if model_type else None
        # Nếu là catboost: chỉ giữ random_seed hoặc random_state, ưu tiên random_seed
        if model_type == "catboost":
            if "random_seed" in params and "random_state" in params:
                params.pop("random_state", None)
            # Nếu thiếu cả hai thì thêm random_seed
            if "random_seed" not in params and "random_state" not in params:
                params["random_seed"] = self.seed
        else:
            # Inject random_state nếu thiếu (phổ biến với LGBM/XGB/RF/...)
            if "random_state" not in params or params.get("random_state") is None:
                params["random_state"] = self.seed

        # Nhiều wrapper cần task_type
        if "task_type" not in params and self.task_type:
            params["task_type"] = self.task_type

        return params

    def _create_model(self, model_type: str, model_config: Dict[str, Any]) -> Any:
        model_type = (model_type or "").lower().strip()
        if model_type not in self.model_registry:
            raise ValueError(
                f"Unsupported model_type='{model_type}'. Available: {sorted(self.model_registry.keys())}"
            )

        module_path, class_name = self.model_registry[model_type].rsplit(".", 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)

        cfg = self._sanitize_params(model_config, model_type)

        # Try wrapper signature: Class(params=...)
        try:
            return model_class(params=cfg)
        except Exception:
            # Try sklearn-style: Class(**cfg)
            return model_class(**cfg)

    def _fit_model(self, model: Any, X, y) -> None:
        if hasattr(model, "train"):
            model.train(X, y)
            return
        if hasattr(model, "fit"):
            model.fit(X, y)
            return
        raise RuntimeError(f"Model '{type(model).__name__}' has no train() or fit().")

    def _predict_model(self, model: Any, X) -> np.ndarray:
        # Nếu là wrapper có .model thì ưu tiên .model.predict
        model_obj = getattr(model, "model", model)
        if not hasattr(model_obj, "predict"):
            raise RuntimeError(f"Model '{type(model).__name__}' has no predict().")
        pred = model_obj.predict(X)
        return self._extract_prediction(pred)

    def _ensemble_predict_from_list(self, preds_list: List[np.ndarray]) -> np.ndarray:
        P = np.column_stack([p.reshape(-1) for p in preds_list])  # shape (n, m)

        if self.ensemble_mode == "weighted_mean":
            if not self.weights:
                raise ValueError("ensemble_mode='weighted_mean' nhưng chưa truyền weights.")
            w = np.asarray(self.weights, dtype=float).reshape(-1)
            if w.shape[0] != P.shape[1]:
                raise ValueError(f"weights length={w.shape[0]} != number of models={P.shape[1]}")
            if np.isclose(w.sum(), 0.0):
                raise ValueError("Sum(weights) == 0 (không hợp lệ).")
            w = w / w.sum()
            return (P * w).sum(axis=1)

        # default mean
        return P.mean(axis=1)

    # -------------------------
    # Public API
    # -------------------------
    def train(self, X, y) -> None:
        X = self._drop_datetime(X)
        y_arr = np.asarray(y).reshape(-1)

        self.models = []
        self.model_metrics = {}
        failed = 0

        for i, m_cfg in enumerate(self.base_models_cfg):
            mtype = m_cfg.get("type", "")
            mparams = dict(m_cfg.get("params", {}))
            name = f"{mtype}_{i}"

            try:
                model = self._create_model(mtype, mparams)
                self._fit_model(model, X, y_arr)

                # Evaluate (ưu tiên val nếu có)
                X_eval = m_cfg.get("X_val", None)
                y_eval = m_cfg.get("y_val", None)
                if X_eval is None or y_eval is None:
                    X_eval, y_eval = X, y_arr
                else:
                    X_eval = self._drop_datetime(X_eval)
                    y_eval = np.asarray(y_eval).reshape(-1)

                y_pred = self._predict_model(model, X_eval)
                self.model_metrics[name] = self._compute_metrics(y_eval, y_pred)

                self.models.append(model)

            except Exception as e:
                failed += 1
                print(f"[Ensemble] ❌ Model {name} failed: {e}")

        if not self.models:
            raise RuntimeError("No base models could be trained!")

        # Ensemble metrics
        try:
            preds_list = [self._predict_model(m, X) for m in self.models]
            y_pred_ens = self._ensemble_predict_from_list(preds_list)
            self.model_metrics["ensemble"] = self._compute_metrics(y_arr, y_pred_ens)
        except Exception as e:
            print(f"[Ensemble] ❌ Failed to compute ensemble metrics: {e}")

        self.is_trained = True

    def predict(self, X) -> np.ndarray:
        if not self.is_trained or not self.models:
            raise RuntimeError("Ensemble model is not trained yet!")

        X = self._drop_datetime(X)

        preds_list = [self._predict_model(m, X) for m in self.models]
        return self._ensemble_predict_from_list(preds_list)

    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        return dict(self.model_metrics)

    def get_base_models(self) -> List[Any]:
        return list(self.models)
