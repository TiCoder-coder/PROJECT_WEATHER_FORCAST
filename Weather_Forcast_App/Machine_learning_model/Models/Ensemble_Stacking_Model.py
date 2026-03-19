# Weather_Forcast_App.Machine_learning_model.Models.Ensemble_Stacking_Model
# =============================================================================
# SUPER LEARNER STACKING ENSEMBLE
# =============================================================================
# Architecture:
#
#  ┌──────────────────────────────────────────────────────────────────────┐
#  │  STAGE 7 — Classification Super Learner                              │
#  │                                                                      │
#  │  XGBCls │ RFCls │ CatCls │ LGBMCls                                  │
#  │       └──────────┬──────────┘                                        │
#  │          Z_cls (OOF, TimeSeriesSplit, 4 cols)                        │
#  │                  └──→ LGBMClassifier (meta, non-linear) ──→ p_rain   │
#  │                       [fallback: LogisticRegression class_weight=bal]│
#  ├──────────────────────────────────────────────────────────────────────┤
#  │  STAGE 8 — Regression Super Learner (rainy-only)                    │
#  │                                                                      │
#  │  XGBReg │ RFReg │ CatReg │ LGBMReg                                  │
#  │       └──────────┬──────────┘                                        │
#  │          Z_reg (OOF, TimeSeriesSplit, 4 cols, log1p target)          │
#  │          sample_weight = log1p(rain_mm) + 1 (upweight heavy rain)    │
#  │                  └──→ LGBMRegressor (meta, non-linear) ──→ log1p(mm)│
#  │                       [fallback: RidgeCV auto-alpha]                 │
#  ├──────────────────────────────────────────────────────────────────────┤
#  │  INFERENCE                                                           │
#  │  if p_rain > threshold  →  expm1(meta_reg(Z_reg(X)))                 │
#  │  else                   →  0.0 mm                                    │
#  └──────────────────────────────────────────────────────────────────────┘
#
# Training workflow (Giai đoạn 6–9):
#   Stage 6: Verify từng base model riêng, kiểm tra không có lỗi
#   Stage 7: OOF 5-fold classification
#            → Folds 1–4 OOF: train 4 thuật toán cơ sở, meta-features → train meta-classifier
#            → Fold 5 (dữ liệu gần nhất): đánh giá unbiased stacked classifier
#   Stage 8: OOF 5-fold regression (rainy-only) — cùng cơ chế Stage 7
#            → Folds 1–4 OOF: train 4 thuật toán cơ sở, meta-features → train meta-regressor
#            → Fold 5 (dữ liệu gần nhất): đánh giá unbiased stacked regressor
#   Stage 9: Refit tất cả base models trên full train
#
# Lý do OOF (không train rồi predict lại trên chính train):
#   - Nếu train XGBoost trên X_train rồi predict ngay X_train → model nhớ hoàn hảo
#     → meta-model học ngưỡng "quá tốt" không tổng quát hóa được
#   - OOF đảm bảo mỗi dự đoán Z[i] đến từ model chưa thấy X[i]
#   - TimeSeriesSplit thay vì KFold vì dữ liệu có tính thời gian
# =============================================================================

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# ── Optional dependencies ── (không crash khi thiếu, báo lỗi khi build model)
try:
    from xgboost import XGBClassifier, XGBRegressor
    _XGB_OK = True
except ImportError:  # pragma: no cover
    _XGB_OK = False
    XGBClassifier = XGBRegressor = None  # type: ignore

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _LGBM_OK = True
except ImportError:  # pragma: no cover
    _LGBM_OK = False
    LGBMClassifier = LGBMRegressor = None  # type: ignore

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    _CAT_OK = True
except ImportError:  # pragma: no cover
    _CAT_OK = False
    CatBoostClassifier = CatBoostRegressor = None  # type: ignore

# Dùng RAIN_THRESHOLD nhất quán với toàn hệ thống
from Weather_Forcast_App.Machine_learning_model.evaluation.metrics import RAIN_THRESHOLD

logger = logging.getLogger(__name__)

# Thư mục lưu model
_MODEL_DIR = Path(__file__).parent.parent / "ml_models"


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class StackingResult:
    """Kết quả trả về sau khi train StackingEnsemble."""
    success: bool = True
    message: str = ""
    training_time: float = 0.0
    n_cls_oof_samples: int = 0
    n_reg_oof_samples: int = 0
    meta_cls_coef: Optional[Any] = None
    meta_reg_coef: Optional[Any] = None
    stage_metrics: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Default hyperparameters
# =============================================================================

# ── Classifiers ──────────────────────────────────────────────────────────────
# Tham số tối ưu cho bài toán rain/no-rain với dữ liệu mất cân bằng nặng.
# scale_pos_weight / class_weight / auto_class_weights được set theo từng model.

_CLS_XGB = dict(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
    verbosity=0,
    use_label_encoder=False,   # suppress XGBoost warning
)

_CLS_RF = dict(
    n_estimators=300,
    max_depth=None,          # không giới hạn depth → học pattern phức tạp hơn
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced_subsample",  # tự handle imbalance
    n_jobs=-1,
    random_state=42,
)

_CLS_CAT = dict(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    auto_class_weights="Balanced",      # tự handle imbalance
    eval_metric="Logloss",
    random_seed=42,
    verbose=0,
    allow_writing_files=False,
)

_CLS_LGBM = dict(
    n_estimators=800,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=42,
    verbosity=-1,
)

# ── Regressors ────────────────────────────────────────────────────────────────
# Train trên rainy-only data + log1p(rain_mm) target.

_REG_XGB = dict(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
    verbosity=0,
)

_REG_RF = dict(
    n_estimators=300,
    max_depth=None,          # không giới hạn depth cho dữ liệu rainy-only
    min_samples_leaf=3,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42,
)

_REG_CAT = dict(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    random_seed=42,
    verbose=0,
    allow_writing_files=False,
)

_REG_LGBM = dict(
    n_estimators=800,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=42,
    verbosity=-1,
)


# =============================================================================
# StackingEnsemble
# =============================================================================

class StackingEnsemble:
    """
    Super Learner Stacking Ensemble 2 tầng cho dự báo lượng mưa.

    Tầng 1 (Classification):
        - 4 base classifiers: XGBoost, RandomForest, CatBoost, LightGBM
        - OOF predictions qua TimeSeriesSplit → Z_cls (n, 4)
        - Meta-learner: Logistic Regression → p_rain

    Tầng 2 (Regression, rainy-only):
        - 4 base regressors: XGBoost, RandomForest, CatBoost, LightGBM
        - OOF predictions qua TimeSeriesSplit (rainy samples only) → Z_reg (n_rainy, 4)
        - Target: log1p(rain_mm) để xử lý heavy-tail distribution
        - Meta-learner: Ridge Regression → dự báo log1p(mm)

    Inference:
        if p_rain > predict_threshold:  rain_mm = expm1(meta_reg(X))
        else:                           rain_mm = 0.0

    Imbalance handling:
        - XGBoost/LightGBM: scale_pos_weight = min(n_neg/n_pos, 20)
        - RandomForest: class_weight='balanced_subsample' (already in default)
        - CatBoost: auto_class_weights='Balanced' (already in default)

    Example:
        >>> stacking = StackingEnsemble(n_splits=5, predict_threshold=0.4)
        >>> result = stacking.fit(X_train, y_train)
        >>> print(stacking.summary())
        >>> y_pred = stacking.predict(X_test)
        >>> metrics = stacking.evaluate(X_test, y_test, "test")
    """

    def __init__(
        self,
        n_splits: int = 5,
        predict_threshold: float = 0.4,
        rain_threshold: float = RAIN_THRESHOLD,
        seed: int = 42,
        cls_params: Optional[Dict[str, Dict]] = None,
        reg_params: Optional[Dict[str, Dict]] = None,
        meta_cls_params: Optional[Dict] = None,
        meta_reg_params: Optional[Dict] = None,
        verbose: bool = True,
    ):
        """
        Args:
            n_splits:          Số fold trong TimeSeriesSplit (mặc định 5)
            predict_threshold: Ngưỡng xác suất để phân loại có/không mưa (mặc định 0.4)
                               Giảm threshold → tăng Recall (quan trọng với rain prediction)
            rain_threshold:    Ngưỡng (mm) để coi là có mưa (mặc định = RAIN_THRESHOLD = 0.1)
            seed:              Random seed
            cls_params:        Override params cho từng classifier
                               Ví dụ: {"xgb": {"n_estimators": 1000}, "lgbm": {...}}
                               Keys: "xgb", "rf", "cat", "lgbm"
            reg_params:        Override params cho từng regressor (cùng keys)
            meta_cls_params:   Override params cho LogisticRegression meta
            meta_reg_params:   Override params cho Ridge meta
            verbose:           In progress log ra stdout
        """
        self.n_splits = n_splits
        self.predict_threshold = predict_threshold
        self.rain_threshold = rain_threshold
        self.seed = seed
        self.verbose = verbose

        # Override params per model
        self._cls_overrides: Dict[str, Dict] = cls_params or {}
        self._reg_overrides: Dict[str, Dict] = reg_params or {}

        # Meta-models — dùng LightGBM phi tuyến nếu có, fallback về LR/RidgeCV
        self.meta_cls = self._build_meta_cls_model(seed, meta_cls_params)
        self.meta_reg = self._build_meta_reg_model(seed, meta_reg_params)

        # Final refit base models (populated in Stage 9)
        self.final_cls_models: List[Any] = []
        self.final_reg_models: List[Any] = []
        self.cls_model_names: List[str] = []
        self.reg_model_names: List[str] = []

        # State
        self.is_trained: bool = False
        self.oof_cls_shape: Tuple[int, int] = (0, 0)
        self.oof_reg_shape: Tuple[int, int] = (0, 0)
        self.stage_metrics: Dict[str, Any] = {}

    # =========================================================================
    # Model factories — mỗi lần gọi trả về danh sách models MỚI (chưa train)
    # =========================================================================

    def _merge(self, base: Dict, key: str, overrides: Dict[str, Dict]) -> Dict:
        """Merge base params với user overrides."""
        merged = dict(base)
        if key in overrides:
            merged.update(overrides[key])
        return merged

    def _cls_model_names(self) -> List[str]:
        """
        Trả về danh sách tên classifier THEO ĐÚNG THỨ TỰ _build_classifiers()
        mà KHÔNG tạo model instances (tránh tốn tài nguyên khi chỉ cần đếm/lấy tên).
        """
        names = []
        if _XGB_OK:
            names.append("xgb_cls")
        names.append("rf_cls")
        if _CAT_OK:
            names.append("cat_cls")
        if _LGBM_OK:
            names.append("lgbm_cls")
        return names

    def _reg_model_names(self) -> List[str]:
        """
        Trả về danh sách tên regressor THEO ĐÚNG THỨ TỰ _build_regressors()
        mà KHÔNG tạo model instances.
        """
        names = []
        if _XGB_OK:
            names.append("xgb_reg")
        names.append("rf_reg")
        if _CAT_OK:
            names.append("cat_reg")
        if _LGBM_OK:
            names.append("lgbm_reg")
        return names

    def _build_classifiers(
        self, imbalance_ratio: float = 1.0
    ) -> List[Tuple[str, Any]]:
        """
        Tạo list 4 fresh classifiers với imbalance handling tự động.

        imbalance_ratio = n_neg / n_pos
            → XGBoost/LightGBM: scale_pos_weight = min(ratio, 20)
            → RandomForest: class_weight='balanced_subsample' (trong default)
            → CatBoost: auto_class_weights='Balanced' (trong default)
        """
        models: List[Tuple[str, Any]] = []
        pos_weight = min(imbalance_ratio, 20.0)

        # ── XGBoost ──
        if _XGB_OK:
            p = self._merge(_CLS_XGB, "xgb", self._cls_overrides)
            p["random_state"] = self.seed
            p["scale_pos_weight"] = pos_weight
            # Loại bỏ kwarg không hợp lệ trong phiên bản mới
            p.pop("use_label_encoder", None)
            models.append(("xgb_cls", XGBClassifier(**p)))

        # ── RandomForest ──
        p = self._merge(_CLS_RF, "rf", self._cls_overrides)
        p["random_state"] = self.seed
        models.append(("rf_cls", RandomForestClassifier(**p)))

        # ── CatBoost ──
        if _CAT_OK:
            p = self._merge(_CLS_CAT, "cat", self._cls_overrides)
            p.pop("random_state", None)          # CatBoost dùng random_seed
            p["random_seed"] = self.seed
            models.append(("cat_cls", CatBoostClassifier(**p)))

        # ── LightGBM ──
        if _LGBM_OK:
            p = self._merge(_CLS_LGBM, "lgbm", self._cls_overrides)
            p["random_state"] = self.seed
            p["scale_pos_weight"] = pos_weight
            models.append(("lgbm_cls", LGBMClassifier(**p)))

        if len(models) < 2:
            raise RuntimeError(
                "Cần ít nhất 2 classifiers. Cài thêm: pip install xgboost lightgbm catboost"
            )
        return models

    def _build_regressors(self) -> List[Tuple[str, Any]]:
        """Tạo list 4 fresh regressors cho log1p(rain_mm) target."""
        models: List[Tuple[str, Any]] = []

        if _XGB_OK:
            p = self._merge(_REG_XGB, "xgb", self._reg_overrides)
            p["random_state"] = self.seed
            models.append(("xgb_reg", XGBRegressor(**p)))

        p = self._merge(_REG_RF, "rf", self._reg_overrides)
        p["random_state"] = self.seed
        models.append(("rf_reg", RandomForestRegressor(**p)))

        if _CAT_OK:
            p = self._merge(_REG_CAT, "cat", self._reg_overrides)
            p.pop("random_state", None)
            p["random_seed"] = self.seed
            models.append(("cat_reg", CatBoostRegressor(**p)))

        if _LGBM_OK:
            p = self._merge(_REG_LGBM, "lgbm", self._reg_overrides)
            p["random_state"] = self.seed
            models.append(("lgbm_reg", LGBMRegressor(**p)))

        if len(models) < 2:
            raise RuntimeError("Cần ít nhất 2 regressors.")
        return models

    # =========================================================================
    # Meta-model factories
    # =========================================================================

    @staticmethod
    def _build_meta_cls_model(seed: int, meta_cls_params: Optional[Dict]) -> Any:
        """
        Tạo meta-classifier.
        Ưu tiên LGBMClassifier (phi tuyến, bắt tương tác giữa base models) nếu có,
        fallback LogisticRegression(class_weight='balanced') khi LightGBM không cài.

        LGBMClassifier ở meta layer cho phép:
          - Bắt được non-linear interaction giữa 4 base model probs
          - is_unbalance=True tự xử lý mất cân bằng trong tập nhỏ meta-features
          - Tốc độ nhanh vì input chỉ có n_models (4 features)
        """
        if _LGBM_OK:
            params: Dict[str, Any] = dict(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=15,
                min_child_samples=5,
                subsample=0.8,
                colsample_bytree=0.8,
                is_unbalance=True,       # tự xử lý mất cân bằng ở meta layer
                random_state=seed,
                verbosity=-1,
                n_jobs=-1,
            )
            if meta_cls_params:
                params.update(meta_cls_params)
            return LGBMClassifier(**params)
        else:
            params = dict(
                C=1.0, max_iter=1000, random_state=seed,
                solver="lbfgs", class_weight="balanced",
            )
            if meta_cls_params:
                params.update(meta_cls_params)
            return LogisticRegression(**params)

    @staticmethod
    def _build_meta_reg_model(seed: int, meta_reg_params: Optional[Dict]) -> Any:
        """
        Tạo meta-regressor.
        Ưu tiên LGBMRegressor (phi tuyến, tốt với heavy-tail rain distribution) nếu có,
        fallback RidgeCV (tự chọn alpha tối ưu) khi LightGBM không cài.

        LGBMRegressor ở meta layer cho phép:
          - Bắt được non-linear blending giữa 4 base regressor outputs
          - Kết hợp tốt hơn với sample_weight cho heavy rain samples
          - Input chỉ có n_models (4 features) → không overfitting
        """
        if _LGBM_OK:
            params: Dict[str, Any] = dict(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=15,
                min_child_samples=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                verbosity=-1,
                n_jobs=-1,
            )
            if meta_reg_params:
                params.update(meta_reg_params)
            return LGBMRegressor(**params)
        else:
            # RidgeCV tự chọn alpha tối ưu thay vì dùng alpha=1.0 cố định
            return RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])

    # =========================================================================
    # Stage 6: Verify từng base model
    # =========================================================================

    def _stage6_verify(
        self,
        X_train: np.ndarray,
        y_cls: np.ndarray,
        X_rainy: np.ndarray,
        y_reg: np.ndarray,
        imbalance_ratio: float,
    ) -> Dict[str, Any]:
        """
        Giai đoạn 6: Kiểm tra nhanh từng base model trên subsample.

        Mục tiêu:
        - Đảm bảo không có model nào bị lỗi trước khi bước vào OOF.
        - Ghi lại train-set metrics để có baseline so sánh.
        - Dùng tối đa 2000 mẫu (evenly spaced) để tiết kiệm thời gian.
          Stage 9 đã có full refit, Stage 6 chỉ cần verify không crash.
        """
        # Lấy subsample đều theo thời gian để verify nhanh
        _N_V = min(2000, X_train.shape[0])
        if X_train.shape[0] > _N_V:
            v_idx = np.round(np.linspace(0, X_train.shape[0] - 1, _N_V)).astype(int)
            X_v, y_v = X_train[v_idx], y_cls[v_idx]
        else:
            X_v, y_v = X_train, y_cls

        _N_VR = min(1000, X_rainy.shape[0])
        if X_rainy.shape[0] > _N_VR:
            vr_idx = np.round(np.linspace(0, X_rainy.shape[0] - 1, _N_VR)).astype(int)
            X_vr, y_vr = X_rainy[vr_idx], y_reg[vr_idx]
        else:
            X_vr, y_vr = X_rainy, y_reg

        metrics: Dict[str, Any] = {}
        self._log(
            f"  [Stage 6] Verifying classification base models "
            f"(subsample={len(X_v)}/{X_train.shape[0]})..."
        )

        for name, model in self._build_classifiers(imbalance_ratio):
            t0 = time.time()
            try:
                model.fit(X_v, y_v)
                proba = model.predict_proba(X_v)[:, 1]
                auc = (
                    roc_auc_score(y_v, proba)
                    if len(np.unique(y_v)) > 1
                    else float("nan")
                )
                elapsed = round(time.time() - t0, 2)
                metrics[name] = {"subsample_roc_auc": round(auc, 4), "time_s": elapsed, "ok": True}
                self._log(f"    ✓ {name}: ROC-AUC={auc:.4f} ({elapsed}s)")
            except Exception as exc:
                metrics[name] = {"ok": False, "error": str(exc)}
                self._log(f"    ✗ {name} FAILED: {exc}")

        self._log(
            f"  [Stage 6] Verifying regression base models "
            f"(subsample={len(X_vr)}/{X_rainy.shape[0]})..."
        )
        for name, model in self._build_regressors():
            t0 = time.time()
            try:
                model.fit(X_vr, y_vr)
                pred = model.predict(X_vr)
                mae = round(mean_absolute_error(y_vr, pred), 4)
                elapsed = round(time.time() - t0, 2)
                metrics[name] = {"subsample_mae_log1p": mae, "time_s": elapsed, "ok": True}
                self._log(f"    ✓ {name}: MAE_log1p={mae:.4f} ({elapsed}s)")
            except Exception as exc:
                metrics[name] = {"ok": False, "error": str(exc)}
                self._log(f"    ✗ {name} FAILED: {exc}")

        return metrics

    # =========================================================================
    # Stage 7: OOF Classification
    # =========================================================================

    def _stage7_oof_cls(
        self,
        X_train: np.ndarray,
        y_cls: np.ndarray,
        imbalance_ratio: float,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Giai đoạn 7: Tạo OOF classification predictions.

        Cơ chế:
        1. Chia X_train thành n_splits fold theo TimeSeriesSplit (chronological)
        2. Với mỗi fold:
           - Train 4 classifiers trên phần train của fold
           - Predict_proba class 1 trên phần validation của fold
           - Ghi kết quả vào Z_cls[val_idx]
        3. Trả về Z_cls với shape (n_train, n_models)

        Quan trọng:
        - Mỗi fold tạo model MỚI → tránh data leakage tuyệt đối
        - Các mẫu đầu (không có val fold) được điền bằng mean của cột

        Returns:
            Z_cls: np.ndarray shape (n_train, n_models), probabilities class 1
            fold_cls_scores: Dict[str, Dict[str, Any]] — per-fold per-model metrics
                             {"0": {"xgb_cls": {"roc_auc": 0.82, "f1": 0.65}, ...}, ...}
        """
        n = X_train.shape[0]
        # Lấy tên models mà KHÔNG tạo instances (dùng helper nhẹ)
        _model_names = self._cls_model_names()
        n_models = len(_model_names)

        Z_cls = np.full((n, n_models), np.nan, dtype=np.float64)
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        fold_cls_scores: Dict[str, Any] = {}   # {fold_str: {model_name: {metric: value}}}

        for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_tr, y_tr = X_train[tr_idx], y_cls[tr_idx]
            X_val = X_train[val_idx]
            y_val = y_cls[val_idx]

            # Bỏ qua fold nếu chỉ có 1 class trong train (không thể classify)
            if len(np.unique(y_tr)) < 2:
                self._log(
                    f"    [Stage 7] Fold {fold_idx + 1}: train fold chỉ có 1 class → skip"
                )
                continue

            self._log(
                f"    [Stage 7] Fold {fold_idx + 1}/{self.n_splits}: "
                f"train={len(tr_idx)}, val={len(val_idx)}"
            )

            fold_key = str(fold_idx)
            fold_cls_scores[fold_key] = {"train_idx": tr_idx.tolist(), "val_idx": val_idx.tolist()}

            # Tạo fresh models cho fold này
            for m_idx, (name, model) in enumerate(
                self._build_classifiers(imbalance_ratio)
            ):
                try:
                    model.fit(X_tr, y_tr)
                    proba = model.predict_proba(X_val)[:, 1]
                    Z_cls[val_idx, m_idx] = proba
                    # Thu thập per-fold per-model metrics cho schema analysis
                    if len(np.unique(y_val)) > 1:
                        fold_auc = float(roc_auc_score(y_val, proba))
                        fold_f1  = float(f1_score(y_val, (proba > 0.5).astype(int),
                                                   zero_division=0))
                        fold_cls_scores[fold_key][name] = {
                            "roc_auc": round(fold_auc, 4),
                            "f1":      round(fold_f1, 4),
                        }
                    else:
                        fold_cls_scores[fold_key][name] = {"roc_auc": float("nan"), "f1": float("nan")}
                except Exception as exc:
                    self._log(f"      ✗ {name} fold {fold_idx + 1} FAILED: {exc}")
                    fold_cls_scores.setdefault(fold_key, {})[name] = {"error": str(exc)}

        # Điền NaN bằng 0.5 (xác suất trung tính - maximum uncertainty)
        # Dùng 0.5 thay vì column mean vì không muốn bias meta-model theo distribution train
        nan_mask = np.isnan(Z_cls).any(axis=1)
        if nan_mask.sum() > 0:
            for col_i in range(n_models):
                nan_rows = np.isnan(Z_cls[:, col_i])
                Z_cls[nan_rows, col_i] = 0.5   # neutral probability (maximum uncertainty)
            self._log(f"    [Stage 7] Điền {nan_mask.sum()} NaN rows bằng 0.5 (xác suất trung tính)")

        self._log(
            f"    [Stage 7] Z_cls shape: {Z_cls.shape}, "
            f"model_names: {_model_names}"
        )
        return Z_cls, fold_cls_scores

    # =========================================================================
    # Stage 8: OOF Regression (rainy-only)
    # =========================================================================

    def _stage8_oof_reg(
        self,
        X_rainy: np.ndarray,
        y_reg: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Giai đoạn 8: Tạo OOF regression predictions trên tập rainy-only.

        Cơ chế:
        1. Lọc tập train_rainy = X_train[rain > rain_threshold] (đã được thực hiện ở fit())
        2. Chia X_rainy theo TimeSeriesSplit (giữ thứ tự thời gian)
        3. Với mỗi fold:
           - Train 4 regressors trên phần train_rainy của fold
           - Dự đoán log1p(mm) trên phần val_rainy
           - Ghi Z_reg[val_idx]
        4. Trả về Z_reg với shape (n_rainy, n_models)

        Target: y_reg = log1p(rain_mm) — đã transform trước khi truyền vào

        Returns:
            Z_reg: np.ndarray shape (n_rainy, n_models), dự báo log1p(mm)
            fold_reg_scores: Dict[str, Dict[str, Any]] — per-fold per-model metrics
                             {"0": {"xgb_reg": {"mae_mm": 2.1, "r2": 0.42}, ...}, ...}
        """
        n = X_rainy.shape[0]
        # Lấy tên models mà KHÔNG tạo instances (dùng helper nhẹ)
        _model_names = self._reg_model_names()
        n_models = len(_model_names)

        Z_reg = np.full((n, n_models), np.nan, dtype=np.float64)
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        fold_reg_scores: Dict[str, Any] = {}   # {fold_str: {model_name: {metric: value}}}

        for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X_rainy)):
            X_tr, y_tr = X_rainy[tr_idx], y_reg[tr_idx]
            X_val = X_rainy[val_idx]
            y_val_log = y_reg[val_idx]

            self._log(
                f"    [Stage 8] Fold {fold_idx + 1}/{self.n_splits}: "
                f"train={len(tr_idx)}, val={len(val_idx)}"
            )

            fold_key = str(fold_idx)
            fold_reg_scores[fold_key] = {"train_idx": tr_idx.tolist(), "val_idx": val_idx.tolist()}

            if len(tr_idx) < max(5, n_models):
                self._log(
                    f"    [Stage 8] Fold {fold_idx + 1}: quá ít mẫu mưa trong train fold "
                    f"({len(tr_idx)} < {max(5, n_models)}) → skip"
                )
                continue

            # sample_weight: upweight heavy rain (đưa về mm scale rồi tính weight)
            sw_tr = np.log1p(np.expm1(np.clip(y_tr, 0, 20))) + 1.0

            for m_idx, (name, model) in enumerate(self._build_regressors()):
                try:
                    try:
                        model.fit(X_tr, y_tr, sample_weight=sw_tr)
                    except TypeError:
                        model.fit(X_tr, y_tr)
                    pred = model.predict(X_val)
                    Z_reg[val_idx, m_idx] = pred
                    # Thu thập per-fold per-model metrics cho schema analysis
                    pred_mm  = np.expm1(np.clip(pred, 0, 20))       # clip log1p trước expm1
                    true_mm  = np.expm1(np.clip(y_val_log, 0, 20))
                    fold_mae = float(mean_absolute_error(true_mm, pred_mm))
                    ss_res = float(np.sum((true_mm - pred_mm) ** 2))
                    ss_tot = float(np.sum((true_mm - true_mm.mean()) ** 2))
                    fold_r2  = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-10 else float("nan")
                    fold_reg_scores[fold_key][name] = {
                        "mae_mm": round(fold_mae, 4),
                        "r2":     round(fold_r2, 4),
                    }
                except Exception as exc:
                    self._log(f"      ✗ {name} fold {fold_idx + 1} FAILED: {exc}")
                    fold_reg_scores.setdefault(fold_key, {})[name] = {"error": str(exc)}

        # Điền NaN bằng median của cột
        nan_mask = np.isnan(Z_reg).any(axis=1)
        if nan_mask.sum() > 0:
            col_medians = np.nanmedian(Z_reg, axis=0)
            col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
            for col_i in range(n_models):
                nan_rows = np.isnan(Z_reg[:, col_i])
                Z_reg[nan_rows, col_i] = col_medians[col_i]
            self._log(f"    [Stage 8] Điền {nan_mask.sum()} NaN rows bằng column median")

        self._log(
            f"    [Stage 8] Z_reg shape: {Z_reg.shape}, "
            f"model_names: {_model_names}"
        )
        return Z_reg, fold_reg_scores

    # =========================================================================
    # Stage 9: Refit trên full train
    # =========================================================================

    def _stage9_refit_cls(
        self, X_train: np.ndarray, y_cls: np.ndarray, imbalance_ratio: float
    ) -> None:
        """
        Giai đoạn 9a: Refit 4 classifiers trên toàn bộ X_train.

        Lý do: OOF chỉ dùng để train meta-model.
        Để inference tốt nhất, base models cần thấy toàn bộ dữ liệu.
        """
        self._log("  [Stage 9] Refit classifiers on full train...")
        self.final_cls_models = []
        self.cls_model_names = []

        for name, model in self._build_classifiers(imbalance_ratio):
            t0 = time.time()
            model.fit(X_train, y_cls)
            self.final_cls_models.append(model)
            self.cls_model_names.append(name)
            self._log(f"    ✓ {name} refit ({time.time() - t0:.1f}s)")

    def _stage9_refit_reg(
        self, X_rainy: np.ndarray, y_reg: np.ndarray
    ) -> None:
        """
        Giai đoạn 9b: Refit 4 regressors trên toàn bộ train_rainy.

        Lý do: tương tự 9a — tận dụng tối đa dữ liệu lịch sử mưa.
        """
        self._log("  [Stage 9] Refit regressors on full train_rainy...")
        self.final_reg_models = []
        self.reg_model_names = []

        # sample_weight: upweight heavy rain để model ưu tiên dự đoán đúng mưa lớn
        _sw_reg = np.log1p(np.expm1(np.clip(y_reg, 0, 20))) + 1.0

        for name, model in self._build_regressors():
            t0 = time.time()
            try:
                model.fit(X_rainy, y_reg, sample_weight=_sw_reg)
            except TypeError:
                model.fit(X_rainy, y_reg)
            self.final_reg_models.append(model)
            self.reg_model_names.append(name)
            self._log(f"    ✓ {name} refit ({time.time() - t0:.1f}s)")

    # =========================================================================
    # Main fit (Stages 6 → 9)
    # =========================================================================

    def fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> StackingResult:
        """
        Toàn bộ pipeline training: Giai đoạn 6 → 7 → 8 → 9.

        Args:
            X_train:  Feature matrix đã preprocessed (numeric only)
            y_train:  Target gốc đơn vị mm (chưa log transform)
            X_val:    Optional validation set để evaluate sau khi train
            y_val:    Optional validation target (mm)

        Returns:
            StackingResult với đầy đủ metrics từng stage
        """
        t_total = time.time()

        # ── Chuẩn bị numpy arrays ──
        X_np = self._to_numpy(X_train)
        y_np = np.asarray(y_train, dtype=np.float64).reshape(-1)
        n_samples = len(y_np)

        self._log(
            f"\n{'=' * 70}\n"
            f"StackingEnsemble.fit  |  n_samples={n_samples}, n_features={X_np.shape[1]}\n"
            f"{'=' * 70}"
        )

        # ── Giai đoạn 1 (nội bộ): Tạo targets ──
        # y_cls: binary — có mưa hay không
        y_cls = (y_np > self.rain_threshold).astype(np.int32)

        # rainy-only subset (giữ thứ tự thời gian)
        rainy_mask = y_np > self.rain_threshold
        X_rainy = X_np[rainy_mask]
        y_reg = np.log1p(y_np[rainy_mask])   # log1p transform

        n_pos = int(rainy_mask.sum())
        n_neg = n_samples - n_pos
        zero_ratio = n_neg / n_samples
        imbalance_ratio = float(n_neg) / n_pos if n_pos > 0 else 1.0

        self._log(
            f"  Rainy samples : {n_pos} ({n_pos / n_samples:.1%})\n"
            f"  Dry samples   : {n_neg} ({zero_ratio:.1%})\n"
            f"  Imbalance ratio (n_neg/n_pos): {imbalance_ratio:.2f}\n"
            f"  rain_threshold: {self.rain_threshold} mm"
        )

        # Kiểm tra đủ data để OOF
        if n_pos < self.n_splits * 3:
            raise ValueError(
                f"Quá ít mẫu mưa ({n_pos}) cho TimeSeriesSplit "
                f"(n_splits={self.n_splits}). Cần ít nhất {self.n_splits * 3}."
            )

        # ─────────────────────────────────────────────────────────────────────
        # GIAI ĐOẠN 6: Verify base models
        # ─────────────────────────────────────────────────────────────────────
        self._log(f"\n{'─' * 60}")
        self._log("GIAI ĐOẠN 6 — Verify từng base model riêng")
        self._log(f"{'─' * 60}")
        t6 = time.time()
        stage6_metrics = self._stage6_verify(
            X_np, y_cls, X_rainy, y_reg, imbalance_ratio
        )
        self._log(f"  ✅ Stage 6 done ({time.time() - t6:.1f}s)")

        # ─────────────────────────────────────────────────────────────────────
        # GIAI ĐOẠN 7: OOF Classification → train meta-cls
        # ─────────────────────────────────────────────────────────────────────
        self._log(f"\n{'─' * 60}")
        self._log(
            f"GIAI ĐOẠN 7 — OOF Classification Super Learner  "
            f"(TimeSeriesSplit n_splits={self.n_splits})"
        )
        self._log(f"{'─' * 60}")
        t7 = time.time()
        Z_cls, fold_cls_scores = self._stage7_oof_cls(X_np, y_cls, imbalance_ratio)

        # ── Chia 5 fold: folds 1–4 dùng train meta, fold 5 dùng đánh giá ──
        # Fold 5 = dữ liệu gần nhất (chronologically latest) — chưa thấy khi train meta
        _tscv_eval = TimeSeriesSplit(n_splits=self.n_splits)
        _cls_folds = list(_tscv_eval.split(X_np))
        fold5_val_cls = _cls_folds[-1][1]           # indices của fold 5 (val cuối)
        meta_train_mask_cls = np.ones(len(y_cls), dtype=bool)
        meta_train_mask_cls[fold5_val_cls] = False  # loại fold 5 khỏi meta-train
        n_meta_train_cls = int(meta_train_mask_cls.sum())
        n_fold5_cls = len(fold5_val_cls)
        self._log(
            f"  [Stage 7] Chia {self.n_splits} fold — "
            f"meta-train (folds 1–{self.n_splits - 1}): {n_meta_train_cls} mẫu | "
            f"fold-{self.n_splits} holdout (đánh giá): {n_fold5_cls} mẫu"
        )

        # Train meta-classifier chỉ trên folds 1–4 OOF
        self.meta_cls.fit(Z_cls[meta_train_mask_cls], y_cls[meta_train_mask_cls])

        # Đánh giá unbiased trên fold-5 holdout (meta-model chưa thấy fold này)
        eval_cls_proba = self.meta_cls.predict_proba(Z_cls[fold5_val_cls])[:, 1]
        eval_y_cls = y_cls[fold5_val_cls]
        oof_roc_auc = (
            roc_auc_score(eval_y_cls, eval_cls_proba)
            if len(np.unique(eval_y_cls)) > 1
            else float("nan")
        )
        oof_pr_auc = average_precision_score(eval_y_cls, eval_cls_proba)
        oof_cls_pred = (eval_cls_proba > self.predict_threshold).astype(int)
        oof_f1 = f1_score(eval_y_cls, oof_cls_pred, zero_division=0)
        oof_recall = recall_score(eval_y_cls, oof_cls_pred, zero_division=0)

        self._log(
            f"  Meta-cls Fold-{self.n_splits} holdout (unbiased): "
            f"ROC-AUC={oof_roc_auc:.4f}, PR-AUC={oof_pr_auc:.4f}, "
            f"F1={oof_f1:.4f}, Recall={oof_recall:.4f}"
        )
        self._log(f"  ✅ Stage 7 done ({time.time() - t7:.1f}s)")
        self.oof_cls_shape = Z_cls.shape

        # ─────────────────────────────────────────────────────────────────────
        # GIAI ĐOẠN 8: OOF Regression (rainy-only) → train meta-reg
        # ─────────────────────────────────────────────────────────────────────
        self._log(f"\n{'─' * 60}")
        self._log(
            f"GIAI ĐOẠN 8 — OOF Regression Super Learner (rainy-only)  "
            f"(TimeSeriesSplit n_splits={self.n_splits})"
        )
        self._log(f"{'─' * 60}")
        t8 = time.time()
        Z_reg, fold_reg_scores = self._stage8_oof_reg(X_rainy, y_reg)

        # ── Chia 5 fold trên rainy-only: folds 1–4 train meta, fold 5 đánh giá ──
        _tscv_eval_reg = TimeSeriesSplit(n_splits=self.n_splits)
        _reg_folds = list(_tscv_eval_reg.split(X_rainy))
        fold5_val_reg = _reg_folds[-1][1]             # fold 5 val indices (rainy-only)
        meta_train_mask_reg = np.ones(len(y_reg), dtype=bool)
        meta_train_mask_reg[fold5_val_reg] = False    # loại fold 5 khỏi meta-train
        n_meta_train_reg = int(meta_train_mask_reg.sum())
        n_fold5_reg = len(fold5_val_reg)
        self._log(
            f"  [Stage 8] Chia {self.n_splits} fold — "
            f"meta-train (folds 1–{self.n_splits - 1}): {n_meta_train_reg} mẫu mưa | "
            f"fold-{self.n_splits} holdout (đánh giá): {n_fold5_reg} mẫu mưa"
        )

        # Train meta-regressor chỉ trên folds 1–4 OOF (rainy-only)
        # Dùng sample_weight để meta-reg cũng ưu tiên dự đoán đúng mưa lớn
        _meta_sw_reg = np.log1p(np.expm1(np.clip(y_reg[meta_train_mask_reg], 0, 20))) + 1.0
        try:
            self.meta_reg.fit(
                Z_reg[meta_train_mask_reg],
                y_reg[meta_train_mask_reg],
                sample_weight=_meta_sw_reg,
            )
        except TypeError:
            self.meta_reg.fit(Z_reg[meta_train_mask_reg], y_reg[meta_train_mask_reg])

        # Đánh giá unbiased trên fold-5 holdout (rainy-only, meta chưa thấy fold này)
        oof_reg_log = self.meta_reg.predict(Z_reg[fold5_val_reg])
        oof_reg_mm = np.expm1(oof_reg_log).clip(min=0)
        true_mm = np.expm1(y_reg[fold5_val_reg])
        oof_mae = round(mean_absolute_error(true_mm, oof_reg_mm), 4)
        oof_rmse = round(float(np.sqrt(mean_squared_error(true_mm, oof_reg_mm))), 4)
        oof_r2 = round(r2_score(true_mm, oof_reg_mm), 4)

        self._log(
            f"  Meta-reg Fold-{self.n_splits} holdout (rainy-only, unbiased): "
            f"MAE={oof_mae}mm, RMSE={oof_rmse}mm, R²={oof_r2}"
        )
        self._log(f"  ✅ Stage 8 done ({time.time() - t8:.1f}s)")
        self.oof_reg_shape = Z_reg.shape

        # ─────────────────────────────────────────────────────────────────────
        # GIAI ĐOẠN 9: Refit base models trên full train
        # ─────────────────────────────────────────────────────────────────────
        self._log(f"\n{'─' * 60}")
        self._log("GIAI ĐOẠN 9 — Refit base models trên full train")
        self._log(f"{'─' * 60}")
        t9 = time.time()
        self._stage9_refit_cls(X_np, y_cls, imbalance_ratio)
        self._stage9_refit_reg(X_rainy, y_reg)
        self._log(f"  ✅ Stage 9 done ({time.time() - t9:.1f}s)")

        # ── Kết thúc training ──
        self.is_trained = True
        total_time = time.time() - t_total

        # ── Lưu OOF data để FoldSchemaModelBank có thể phân tích schema ──
        self._oof_Z_cls       = Z_cls              # (n_train, n_cls_models) — OOF probabilities
        self._oof_y_cls       = y_cls              # (n_train,) — binary labels
        self._oof_rainy_mask  = rainy_mask         # (n_train,) — bool: rainy samples
        self._oof_Z_reg       = Z_reg              # (n_rainy, n_reg_models) — OOF log1p preds
        self._oof_y_reg_mm    = np.expm1(y_reg)   # (n_rainy,) — ground truth rain_mm
        self._oof_fold_indices_cls = _cls_folds   # list of (tr_idx, val_idx) per fold
        self._oof_fold_indices_reg = _reg_folds   # list of (tr_idx, val_idx) per fold (rainy)

        self.stage_metrics = {
            "stage6": stage6_metrics,
            "stage7": {
                # Fold-5 holdout (unbiased) — meta-model chưa thấy fold này khi train
                "fold5_holdout_roc_auc": oof_roc_auc,
                "fold5_holdout_pr_auc": oof_pr_auc,
                "fold5_holdout_f1": oof_f1,
                "fold5_holdout_recall": oof_recall,
                "fold5_n_samples": n_fold5_cls,
                "meta_train_n_samples": n_meta_train_cls,
                # Per-fold per-model AUC/F1 cho schema analysis
                "per_fold_model_scores": fold_cls_scores,
            },
            "stage8": {
                # Fold-5 holdout (unbiased, rainy-only)
                "fold5_holdout_mae_mm": oof_mae,
                "fold5_holdout_rmse_mm": oof_rmse,
                "fold5_holdout_r2": oof_r2,
                "fold5_n_samples": n_fold5_reg,
                "meta_train_n_samples": n_meta_train_reg,
                # Per-fold per-model MAE/R² cho schema analysis
                "per_fold_model_scores": fold_reg_scores,
            },
        }

        # Optional: evaluate trên validation set
        if X_val is not None and y_val is not None:
            self._log("\n  [Validation Evaluation]")
            val_metrics = self.evaluate(X_val, y_val, dataset_name="validation")
            self.stage_metrics["validation"] = val_metrics

        self._log(
            f"\n{'=' * 70}\n"
            f"✅ StackingEnsemble training hoàn tất  |  total={total_time:.1f}s\n"
            f"{'=' * 70}"
        )

        return StackingResult(
            success=True,
            training_time=total_time,
            n_cls_oof_samples=Z_cls.shape[0],
            n_reg_oof_samples=Z_reg.shape[0],
            meta_cls_coef=(
                self.meta_cls.coef_.copy()
                if hasattr(self.meta_cls, "coef_")
                else getattr(self.meta_cls, "feature_importances_", None)
            ),
            meta_reg_coef=(
                self.meta_reg.coef_.copy()
                if hasattr(self.meta_reg, "coef_")
                else getattr(self.meta_reg, "feature_importances_", None)
            ),
            stage_metrics=self.stage_metrics,
        )

    # =========================================================================
    # Inference
    # =========================================================================

    def _get_cls_stack(self, X_np: np.ndarray) -> np.ndarray:
        """
        Tạo Z_cls từ final classifiers → meta_cls → p_rain.
        Shape: (n_samples,)
        """
        n_models = len(self.final_cls_models)
        Z = np.zeros((X_np.shape[0], n_models), dtype=np.float64)
        for i, model in enumerate(self.final_cls_models):
            Z[:, i] = model.predict_proba(X_np)[:, 1]
        return self.meta_cls.predict_proba(Z)[:, 1]

    def _get_reg_stack(self, X_np: np.ndarray) -> np.ndarray:
        """
        Tạo Z_reg từ final regressors → meta_reg → expm1(log1p(mm)).
        Shape: (n_samples,), unit: mm
        """
        n_models = len(self.final_reg_models)
        Z = np.zeros((X_np.shape[0], n_models), dtype=np.float64)
        for i, model in enumerate(self.final_reg_models):
            Z[:, i] = model.predict(X_np)
        log_pred = self.meta_reg.predict(Z)
        return np.expm1(log_pred).clip(min=0)

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """
        Dự đoán lượng mưa (mm).

        Pipeline:
        1. Tất cả mẫu: classifier stacking → p_rain
        2. Gate: if p_rain > predict_threshold → run regressor stacking
                 else → 0.0 mm
        3. Kết hợp output

        Returns:
            np.ndarray shape (n_samples,), dự báo lượng mưa mm (>= 0)
        """
        self._check_trained()
        X_np = self._to_numpy(X)

        p_rain = self._get_cls_stack(X_np)
        has_rain = p_rain > self.predict_threshold

        result = np.zeros(X_np.shape[0], dtype=np.float64)
        if has_rain.sum() > 0:
            result[has_rain] = self._get_reg_stack(X_np[has_rain])

        return result

    def predict_proba_rain(
        self,
        X: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """Trả về xác suất có mưa từ meta-classifier. Shape: (n_samples,)."""
        self._check_trained()
        return self._get_cls_stack(self._to_numpy(X))

    def predict_full(
        self,
        X: Union[np.ndarray, pd.DataFrame],
    ) -> Dict[str, np.ndarray]:
        """
        Dự đoán đầy đủ, trả về dict gồm:
            - "predictions":    lượng mưa cuối cùng (mm), qua classification gate
            - "p_rain":         xác suất có mưa (0–1)
            - "has_rain":       nhãn binary 0/1 theo predict_threshold
            - "rain_mm_ungated": lượng mưa từ regression không qua gate
                                 (hữu ích để debug/phân tích)
        """
        self._check_trained()
        X_np = self._to_numpy(X)

        p_rain = self._get_cls_stack(X_np)
        rain_mm_ungated = self._get_reg_stack(X_np)
        has_rain = p_rain > self.predict_threshold
        predictions = np.where(has_rain, rain_mm_ungated, 0.0)

        return {
            "predictions": predictions,
            "p_rain": p_rain,
            "has_rain": has_rain.astype(np.int32),
            "rain_mm_ungated": rain_mm_ungated,
        }

    # =========================================================================
    # Evaluation
    # =========================================================================

    def evaluate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y_true: Union[np.ndarray, pd.Series],
        dataset_name: str = "",
    ) -> Dict[str, Any]:
        """
        Đánh giá đầy đủ trên 1 dataset.

        Trả về dict với 3 nhóm metrics:
        - "classification": ROC-AUC, PR-AUC, F1, Recall, Precision, Accuracy
        - "regression_rainy": MAE, RMSE, R² trên tập rainy-only (y_true > threshold)
        - "end_to_end": MAE, RMSE trên toàn bộ dự đoán cuối

        Lý do tách 2 nhóm regression:
        - end_to_end bị kéo bởi nhiều mẫu dry → không phản ánh chất lượng
          của model khi thực sự có mưa
        - regression_rainy mới là chỉ số quan trọng nhất để tune regressor
        """
        self._check_trained()
        X_np = self._to_numpy(X)
        y_np = np.asarray(y_true, dtype=np.float64).reshape(-1)

        full = self.predict_full(X_np)
        p_rain = full["p_rain"]
        y_pred = full["predictions"]
        y_pred_cls = (p_rain > self.predict_threshold).astype(int)
        y_cls_true = (y_np > self.rain_threshold).astype(int)

        # ── Classification metrics ──
        cls_m: Dict[str, Any] = {}
        if len(np.unique(y_cls_true)) > 1:
            cls_m["roc_auc"] = round(roc_auc_score(y_cls_true, p_rain), 4)
            cls_m["pr_auc"] = round(average_precision_score(y_cls_true, p_rain), 4)
        cls_m["accuracy"] = round(accuracy_score(y_cls_true, y_pred_cls), 4)
        cls_m["f1"] = round(f1_score(y_cls_true, y_pred_cls, zero_division=0), 4)
        cls_m["recall"] = round(recall_score(y_cls_true, y_pred_cls, zero_division=0), 4)
        cls_m["precision"] = round(precision_score(y_cls_true, y_pred_cls, zero_division=0), 4)

        # ── Regression metrics (rainy-only) ──
        rainy_mask = y_np > self.rain_threshold
        reg_m: Dict[str, Any] = {}
        if rainy_mask.sum() > 0:
            yr_true = y_np[rainy_mask]
            yr_pred = y_pred[rainy_mask]
            reg_m["mae_mm"] = round(mean_absolute_error(yr_true, yr_pred), 4)
            reg_m["rmse_mm"] = round(
                float(np.sqrt(mean_squared_error(yr_true, yr_pred))), 4
            )
            reg_m["r2"] = round(r2_score(yr_true, yr_pred), 4)
            reg_m["n_rainy"] = int(rainy_mask.sum())

        # ── End-to-end metrics (all samples) ──
        e2e_m = {
            "mae_mm": round(mean_absolute_error(y_np, y_pred), 4),
            "rmse_mm": round(float(np.sqrt(mean_squared_error(y_np, y_pred))), 4),
        }

        result = {
            "classification": cls_m,
            "regression_rainy": reg_m,
            "end_to_end": e2e_m,
        }

        prefix = f"[{dataset_name}] " if dataset_name else ""
        self._log(
            f"  {prefix}CLS  — ROC-AUC={cls_m.get('roc_auc', 'n/a')}, "
            f"PR-AUC={cls_m.get('pr_auc', 'n/a')}, "
            f"F1={cls_m['f1']}, Recall={cls_m['recall']}, "
            f"Precision={cls_m['precision']}"
        )
        if reg_m:
            self._log(
                f"  {prefix}REG  — MAE={reg_m['mae_mm']}mm, "
                f"RMSE={reg_m['rmse_mm']}mm, R²={reg_m.get('r2', 'n/a')}"
            )
        self._log(
            f"  {prefix}E2E  — MAE={e2e_m['mae_mm']}mm, "
            f"RMSE={e2e_m['rmse_mm']}mm"
        )
        return result

    # =========================================================================
    # Save / Load
    # =========================================================================

    def save(self, filepath: Optional[str] = None) -> str:
        """Lưu toàn bộ StackingEnsemble ra file joblib."""
        if not self.is_trained:
            raise RuntimeError("Model chưa train, không thể lưu.")

        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        if filepath is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            filepath = str(_MODEL_DIR / f"stacking_ensemble_{ts}.joblib")

        payload = {
            "version": "1.0",
            "n_splits": self.n_splits,
            "predict_threshold": self.predict_threshold,
            "rain_threshold": self.rain_threshold,
            "seed": self.seed,
            "meta_cls": self.meta_cls,
            "meta_reg": self.meta_reg,
            "final_cls_models": self.final_cls_models,
            "final_reg_models": self.final_reg_models,
            "cls_model_names": self.cls_model_names,
            "reg_model_names": self.reg_model_names,
            "oof_cls_shape": self.oof_cls_shape,
            "oof_reg_shape": self.oof_reg_shape,
            "stage_metrics": self.stage_metrics,
        }
        joblib.dump(payload, filepath)
        self._log(f"  ✅ Saved → {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: str) -> "StackingEnsemble":
        """Load StackingEnsemble từ file joblib."""
        payload = joblib.load(filepath)

        instance = cls(
            n_splits=payload.get("n_splits", 5),
            predict_threshold=payload.get("predict_threshold", 0.4),
            rain_threshold=payload.get("rain_threshold", RAIN_THRESHOLD),
            seed=payload.get("seed", 42),
        )
        instance.meta_cls = payload["meta_cls"]
        instance.meta_reg = payload["meta_reg"]
        instance.final_cls_models = payload.get("final_cls_models", [])
        instance.final_reg_models = payload.get("final_reg_models", [])
        instance.cls_model_names = payload.get("cls_model_names", [])
        instance.reg_model_names = payload.get("reg_model_names", [])
        instance.oof_cls_shape = payload.get("oof_cls_shape", (0, 0))
        instance.oof_reg_shape = payload.get("oof_reg_shape", (0, 0))
        instance.stage_metrics = payload.get("stage_metrics", {})
        instance.is_trained = True

        logger.info("StackingEnsemble loaded from %s", filepath)
        return instance

    # =========================================================================
    # Utilities
    # =========================================================================

    def _to_numpy(self, X: Any) -> np.ndarray:
        """Chuyển DataFrame/array về numpy float64 (chỉ numeric cols)."""
        if hasattr(X, "select_dtypes"):
            # pandas DataFrame: lấy numeric cols
            X = X.select_dtypes(include=[np.number]).values
        arr = np.asarray(X)
        if arr.dtype.kind not in ("f", "i", "u"):
            arr = arr.astype(np.float64)
        elif arr.dtype != np.float64:
            arr = arr.astype(np.float64)
        return arr

    def _check_trained(self) -> None:
        if not self.is_trained:
            raise RuntimeError(
                "StackingEnsemble chưa được train. Gọi .fit(X_train, y_train) trước."
            )

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)
        logger.info(msg)

    def summary(self) -> str:
        """In tóm tắt trạng thái model."""
        lines = [
            "=" * 70,
            "  StackingEnsemble Summary",
            "=" * 70,
            f"  is_trained         : {self.is_trained}",
            f"  n_splits           : {self.n_splits}",
            f"  predict_threshold  : {self.predict_threshold}",
            f"  rain_threshold     : {self.rain_threshold} mm",
            f"  seed               : {self.seed}",
            f"  OOF cls shape      : {self.oof_cls_shape}",
            f"  OOF reg shape      : {self.oof_reg_shape}",
            f"  Classifiers        : {self.cls_model_names}",
            f"  Regressors         : {self.reg_model_names}",
            f"  Meta-classifier    : {type(self.meta_cls).__name__}",
            f"  Meta-regressor     : {type(self.meta_reg).__name__}",
        ]
        if "stage7" in self.stage_metrics:
            m7 = self.stage_metrics["stage7"]
            lines += [
                f"  OOF ROC-AUC        : {m7.get('fold5_holdout_roc_auc', 'n/a')}",
                f"  OOF PR-AUC         : {m7.get('fold5_holdout_pr_auc', 'n/a')}",
                f"  OOF F1             : {m7.get('fold5_holdout_f1', 'n/a')}",
                f"  OOF Recall         : {m7.get('fold5_holdout_recall', 'n/a')}",
            ]
        if "stage8" in self.stage_metrics:
            m8 = self.stage_metrics["stage8"]
            lines += [
                f"  OOF MAE (mm)       : {m8.get('fold5_holdout_mae_mm', 'n/a')}",
                f"  OOF RMSE (mm)      : {m8.get('fold5_holdout_rmse_mm', 'n/a')}",
                f"  OOF R²             : {m8.get('fold5_holdout_r2', 'n/a')}",
            ]
        if "validation" in self.stage_metrics:
            mv = self.stage_metrics["validation"]
            cls_m = mv.get("classification", {})
            reg_m = mv.get("regression_rainy", {})
            lines += [
                f"  Val ROC-AUC        : {cls_m.get('roc_auc', 'n/a')}",
                f"  Val PR-AUC         : {cls_m.get('pr_auc', 'n/a')}",
                f"  Val F1             : {cls_m.get('f1', 'n/a')}",
                f"  Val MAE (mm)       : {reg_m.get('mae_mm', 'n/a')}",
                f"  Val RMSE (mm)      : {reg_m.get('rmse_mm', 'n/a')}",
            ]
        lines.append("=" * 70)
        return "\n".join(lines)

    def __repr__(self) -> str:
        status = "trained" if self.is_trained else "untrained"
        return (
            f"StackingEnsemble({status}, "
            f"n_splits={self.n_splits}, "
            f"threshold={self.predict_threshold}, "
            f"cls_models={len(self.final_cls_models)}, "
            f"reg_models={len(self.final_reg_models)})"
        )
