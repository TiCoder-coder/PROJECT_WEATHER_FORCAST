"""
Schema_Selector.py
==================
FoldSchemaModelBank — Phân tích hiệu suất từng base model trên từng schema
dữ liệu sau quá trình 5-fold OOF của StackingEnsemble; tuyển chọn và lưu trữ
model tốt nhất cho mỗi schema; phục vụ dự đoán có định tuyến schema.

Luồng hoạt động
===============
Sau khi StackingEnsemble.fit() tạo xong OOF predictions (Stage 7 & 8), model
lưu lại:
    stacking._oof_Z_cls         : (n_train, n_cls_models)  — OOF probabilities
    stacking._oof_y_cls         : (n_train,)               — binary labels
    stacking._oof_rainy_mask    : (n_train,) bool           — rainy samples
    stacking._oof_Z_reg         : (n_rainy, n_reg_models)  — OOF log1p preds
    stacking._oof_y_reg_mm      : (n_rainy,)               — ground truth mm
    stacking._oof_fold_indices_cls / _reg : list of (tr,val) per fold

FoldSchemaModelBank:
    1. Sử dụng OOF data để tính per-model score trên từng schema slice
    2. Chọn best model type cho mỗi schema key
    3. Lưu schema_routing: {schema_key → {cls_model_name, reg_model_name}}
    4. Không cần retrain — dùng Stage-9 final models từ StackingEnsemble
    5. Tại predict time: detect schema từ X features → route đến model phù hợp

Schemas được định nghĩa
=======================
1. rain_intensity  (5 levels)   — dùng y_train làm ground truth (analytic only;
                                   predict-time routing dùng p_rain hint)
   - no_rain      : y = 0
   - light        : 0 < y ≤ 2.5 mm
   - moderate     : 2.5 < y ≤ 7.5 mm
   - heavy        : 7.5 < y ≤ 25 mm
   - very_heavy   : y > 25 mm

2. season          (2 levels)   — routing thực tiễn bằng month features
   - rainy_season  : tháng 5–10 (Việt Nam — mùa mưa)
   - dry_season    : tháng 11–4

3. time_fold       (n_splits)   — temporal window trong training data
   - fold_0..fold_4              routing: dùng season làm proxy

Đầu ra artifacts
================
schema_model_bank/
    routing_config.json      # schema → best model names
    performance_report.json  # chi tiết per-schema per-model scores
    season_centroids.json    # centroids trong scaled feature space (month routing)
    model_bank.pkl           # toàn bộ FoldSchemaModelBank (joblib)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import joblib
    _JOBLIB_OK = True
except ImportError:
    _JOBLIB_OK = False

try:
    from sklearn.metrics import (
        f1_score,
        mean_absolute_error,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    _SKL_OK = True
except ImportError:
    _SKL_OK = False

logger = logging.getLogger(__name__)

# =============================================================================
# Hằng số schema
# =============================================================================

#: Rain intensity bins: (lo, hi] — lo không bao gồm, hi bao gồm (ngoại trừ no_rain)
RAIN_INTENSITY_BINS: Dict[str, Tuple[float, float]] = {
    "no_rain":    (float("-inf"), 0.0),       # y == 0 → lo= -inf, hi=0 (exact 0)
    "light":      (0.0,  2.5),
    "moderate":   (2.5,  7.5),
    "heavy":      (7.5,  25.0),
    "very_heavy": (25.0, float("inf")),
}

#: Tháng mùa mưa Việt Nam
VN_RAINY_MONTHS = {5, 6, 7, 8, 9, 10}

#: Tên schemas & metric chính
_CLS_PRIMARY_METRIC = "roc_auc"    # metric so sánh classifier → cao hơn = tốt hơn
_REG_PRIMARY_METRIC = "mae_mm"     # metric so sánh regressor → thấp hơn = tốt hơn


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class SchemaModelEntry:
    """Kết quả cho một schema key."""
    schema_type: str              # "rain_intensity" | "season" | "time_fold"
    schema_key: str               # "heavy" | "rainy_season" | "fold_3"
    best_cls_model: Optional[str] = None     # tên model tốt nhất (cls)
    best_reg_model: Optional[str] = None     # tên model tốt nhất (reg)
    cls_scores: Dict[str, float] = field(default_factory=dict)  # {model_name: roc_auc}
    reg_scores: Dict[str, float] = field(default_factory=dict)  # {model_name: mae_mm}
    n_samples_cls: int = 0
    n_samples_reg: int = 0
    note: str = ""


@dataclass
class SchemaModelBankState:
    """Trạng thái serializable của FoldSchemaModelBank."""
    trained_at: str = ""
    n_train_samples: int = 0
    n_rainy_samples: int = 0
    cls_model_names: List[str] = field(default_factory=list)
    reg_model_names: List[str] = field(default_factory=list)
    schema_entries: Dict[str, Any] = field(default_factory=dict)
    routing_config: Dict[str, Any] = field(default_factory=dict)
    season_centroids: Dict[str, Any] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    note: str = ""


# =============================================================================
# Tiện ích nội bộ
# =============================================================================

def _now_tag() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _save_json(path: Path, data: Any) -> None:
    def _default(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return v if np.isfinite(v) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=_default),
                    encoding="utf-8")


def _rain_intensity_mask(y_mm: np.ndarray, schema_key: str) -> np.ndarray:
    """Trả về boolean mask cho rain intensity schema key."""
    lo, hi = RAIN_INTENSITY_BINS[schema_key]
    if schema_key == "no_rain":
        return y_mm == 0.0
    return (y_mm > lo) & (y_mm <= hi)


def _detect_month_from_scaled_features(
    X: np.ndarray,
    feature_names: List[str],
    month_sin_mean: float,
    month_sin_std: float,
    month_cos_mean: float,
    month_cos_std: float,
) -> Optional[np.ndarray]:
    """
    Phục hồi approximate tháng (1–12) từ scaled month_sin / month_cos.

    Sau StandardScaler:  x_scaled = (x - mean) / std
    Để recover:          x = x_scaled * std + mean  (denormalize)
    Sau đó:  angle = arctan2(sin_val, cos_val)
             month ≈ round((angle / (2π) * 12 + 12) % 12 + 1)

    Returns None nếu không tìm thấy month features.
    """
    try:
        sin_idx = feature_names.index("month_sin")
        cos_idx = feature_names.index("month_cos")
    except ValueError:
        return None

    sin_scaled = X[:, sin_idx] if X.ndim == 2 else np.array([X[sin_idx]])
    cos_scaled = X[:, cos_idx] if X.ndim == 2 else np.array([X[cos_idx]])

    # Denormalize
    sin_raw = sin_scaled * month_sin_std + month_sin_mean
    cos_raw = cos_scaled * month_cos_std + month_cos_mean

    # Recover month from circular encoding
    angle = np.arctan2(sin_raw, cos_raw)                        # [-π, π]
    month_float = (angle / (2.0 * np.pi) * 12.0 + 12.0) % 12  # [0, 12)
    month = (month_float.round().astype(int) % 12) + 1          # [1, 12]
    return month


def _detect_season_from_centroids(
    X: np.ndarray,
    feature_names: List[str],
    rainy_centroid: np.ndarray,
    dry_centroid: np.ndarray,
) -> np.ndarray:
    """
    Xác định mùa dựa trên khoảng cách Euclidean đến centroid của mỗi mùa
    trong scaled feature space (chỉ dùng month_sin, month_cos).

    Không cần denormalize — centroid cũng ở scaled space.
    Returns: array of "rainy_season" | "dry_season" | "unknown"
    """
    try:
        sin_idx = feature_names.index("month_sin")
        cos_idx = feature_names.index("month_cos")
    except ValueError:
        return np.array(["unknown"] * (X.shape[0] if X.ndim == 2 else 1))

    X2 = X if X.ndim == 2 else X.reshape(1, -1)
    feat = X2[:, [sin_idx, cos_idx]]

    d_rainy = np.linalg.norm(feat - rainy_centroid, axis=1)
    d_dry   = np.linalg.norm(feat - dry_centroid,   axis=1)

    result = np.where(d_rainy <= d_dry, "rainy_season", "dry_season")
    return result


# =============================================================================
# FoldSchemaModelBank
# =============================================================================

class FoldSchemaModelBank:
    """
    Phân tích và tuyển chọn model tốt nhất cho từng schema dữ liệu.

    Khởi tạo qua factory method:
        bank = FoldSchemaModelBank.from_stacking(stacking, y_train_mm, feature_names)

    Sau đó:
        bank.analyze()
        bank.save(artifacts_dir / "schema_model_bank")

    Dự đoán:
        result = bank.predict_full(X_new, feature_names)
        # result["predictions_mm"]   — array (n,)
        # result["schema_detected"]  — array (n,)  ("rainy_season", "dry_season", ...)
        # result["model_used_cls"]   — array (n,)  tên classifier được dùng
        # result["model_used_reg"]   — array (n,)  tên regressor được dùng
    """

    def __init__(
        self,
        stacking: Any,
        y_train_mm: np.ndarray,
        feature_names: List[str],
        predict_threshold: float = 0.4,
        rain_threshold: float = 0.1,
        verbose: bool = True,
    ) -> None:
        """
        Args:
            stacking:           StackingEnsemble đã trained (sau Stage 9).
            y_train_mm:         Target mm cho toàn bộ training data (n_train,).
                                Thứ tự phải khớp với X_train được dùng khi train.
            feature_names:      Danh sách tên feature sau pipeline transform.
            predict_threshold:  Ngưỡng probability để quyết định có mưa.
            rain_threshold:     Ngưỡng mm để phân biệt rainy / dry.
            verbose:            Bật/tắt logging.
        """
        self.stacking         = stacking
        self.y_train_mm       = np.asarray(y_train_mm, dtype=np.float64).ravel()
        self.feature_names    = list(feature_names)
        self.predict_threshold = float(predict_threshold)
        self.rain_threshold   = float(rain_threshold)
        self.verbose          = verbose

        self._state: Optional[SchemaModelBankState] = None
        self._is_analyzed     = False

        # Computed sau analyze()
        self._routing_config: Dict[str, Any] = {}        # schema_key → {cls, reg}
        self._season_centroids: Dict[str, np.ndarray] = {}  # rainy_season / dry_season
        self._month_stats: Dict[str, float] = {}          # mean/std để denormalize

    # ─────────────────────────────────────────────────────────────────────────
    # Factory
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_stacking(
        cls,
        stacking: Any,
        y_train_mm: np.ndarray,
        feature_names: List[str],
        predict_threshold: float = 0.4,
        rain_threshold: float = 0.1,
        verbose: bool = True,
    ) -> "FoldSchemaModelBank":
        """
        Tạo FoldSchemaModelBank từ StackingEnsemble đã trained.

        Yêu cầu stacking phải có các attributes sau (được gán trong fit()):
            _oof_Z_cls, _oof_y_cls, _oof_rainy_mask,
            _oof_Z_reg, _oof_y_reg_mm,
            _oof_fold_indices_cls, _oof_fold_indices_reg,
            cls_model_names, reg_model_names,
            final_cls_models, final_reg_models
        """
        bank = cls(
            stacking=stacking,
            y_train_mm=y_train_mm,
            feature_names=feature_names,
            predict_threshold=predict_threshold,
            rain_threshold=rain_threshold,
            verbose=verbose,
        )
        return bank

    # ─────────────────────────────────────────────────────────────────────────
    # Phân tích chính
    # ─────────────────────────────────────────────────────────────────────────

    def analyze(self) -> "FoldSchemaModelBank":
        """
        Chạy toàn bộ schema analysis trên OOF data.

        Bước:
        1. Validate OOF data có sẵn
        2. Phân tích rain_intensity schema
        3. Phân tích season schema
        4. Phân tích time_fold schema
        5. Xây dựng routing config
        6. Tính season centroids để routing tại predict time
        """
        self._log("=" * 60)
        self._log("FoldSchemaModelBank.analyze() — Bắt đầu phân tích schema")
        self._log("=" * 60)

        stacking = self.stacking

        # ── Kiểm tra OOF data ──
        missing = [attr for attr in [
            "_oof_Z_cls", "_oof_y_cls", "_oof_rainy_mask",
            "_oof_Z_reg", "_oof_y_reg_mm",
        ] if not hasattr(stacking, attr)]
        if missing:
            raise AttributeError(
                f"StackingEnsemble thiếu OOF attributes: {missing}. "
                f"Hãy chạy lại stacking.fit() với phiên bản mới nhất."
            )

        Z_cls      = stacking._oof_Z_cls        # (n_train, n_cls)
        y_cls      = stacking._oof_y_cls        # (n_train,)
        rainy_mask = stacking._oof_rainy_mask   # (n_train,) bool
        Z_reg      = stacking._oof_Z_reg        # (n_rainy, n_reg)
        y_reg_mm   = stacking._oof_y_reg_mm     # (n_rainy,) mm
        y_mm       = self.y_train_mm            # (n_train,) mm

        cls_names  = list(stacking.cls_model_names)
        reg_names  = list(stacking.reg_model_names)

        n_train  = len(y_cls)
        n_rainy  = int(rainy_mask.sum())

        self._log(f"  n_train={n_train}, n_rainy={n_rainy}")
        self._log(f"  cls_models: {cls_names}")
        self._log(f"  reg_models: {reg_names}")

        schema_entries: Dict[str, SchemaModelEntry] = {}
        routing: Dict[str, Any] = {}

        # ── 1. Rain intensity schema ──────────────────────────────────────────
        self._log("\n  [Schema 1] rain_intensity")
        for sk, (lo, hi) in RAIN_INTENSITY_BINS.items():
            mask_cls = _rain_intensity_mask(y_mm, sk)
            n_sk     = int(mask_cls.sum())
            if n_sk < 10:
                self._log(f"    {sk}: n={n_sk} — quá ít samples, skip")
                continue

            entry = SchemaModelEntry(
                schema_type="rain_intensity",
                schema_key=sk,
                n_samples_cls=n_sk,
            )

            # Đánh giá từng classifier trên slice này
            y_cls_sk = y_cls[mask_cls]
            Z_cls_sk = Z_cls[mask_cls]
            cls_scores_sk: Dict[str, float] = {}

            for m_i, m_name in enumerate(cls_names):
                proba_sk = Z_cls_sk[:, m_i]
                if len(np.unique(y_cls_sk)) > 1:
                    try:
                        auc = float(roc_auc_score(y_cls_sk, proba_sk))
                    except Exception:
                        auc = float("nan")
                else:
                    # Chỉ 1 class → dùng F1 thay thế
                    pred_sk = (proba_sk > self.predict_threshold).astype(int)
                    auc = float(f1_score(y_cls_sk, pred_sk, zero_division=0))
                cls_scores_sk[m_name] = round(auc, 4)

            entry.cls_scores = cls_scores_sk
            if cls_scores_sk:
                valid_scores = {k: v for k, v in cls_scores_sk.items()
                                if np.isfinite(v)}
                if valid_scores:
                    entry.best_cls_model = max(valid_scores, key=valid_scores.get)

            # Đánh giá từng regressor trên rainy subset của slice này
            if sk != "no_rain":
                # rainy samples trong slice này
                rainy_in_sk = mask_cls & rainy_mask
                n_rainy_sk  = int(rainy_in_sk.sum())
                entry.n_samples_reg = n_rainy_sk

                if n_rainy_sk >= 5:
                    # Chuyển sang rainy-only indices
                    rainy_positions = np.where(rainy_mask)[0]
                    rainy_in_sk_subset = rainy_in_sk[rainy_mask]  # boolean subset trong rainy

                    y_reg_sk   = y_reg_mm[rainy_in_sk_subset]
                    Z_reg_sk   = Z_reg[rainy_in_sk_subset]
                    reg_scores_sk: Dict[str, float] = {}

                    for m_i, m_name in enumerate(reg_names):
                        pred_mm_sk = np.expm1(
                            np.clip(Z_reg_sk[:, m_i], 0, 20)
                        )
                        try:
                            mae = float(mean_absolute_error(y_reg_sk, pred_mm_sk))
                        except Exception:
                            mae = float("nan")
                        reg_scores_sk[m_name] = round(mae, 4)

                    entry.reg_scores = reg_scores_sk
                    if reg_scores_sk:
                        valid_maes = {k: v for k, v in reg_scores_sk.items()
                                      if np.isfinite(v)}
                        if valid_maes:
                            entry.best_reg_model = min(valid_maes, key=valid_maes.get)

            log_cls = {k: f"{v:.4f}" for k, v in entry.cls_scores.items()}
            log_reg = {k: f"{v:.4f}" for k, v in entry.reg_scores.items()}
            self._log(
                f"    {sk} (n={n_sk}): "
                f"best_cls={entry.best_cls_model} {log_cls} | "
                f"best_reg={entry.best_reg_model} {log_reg}"
            )

            schema_entries[f"rain_intensity/{sk}"] = entry

        # ── 2. Season schema ──────────────────────────────────────────────────
        self._log("\n  [Schema 2] season (Vietnam rainy=5-10, dry=11-4)")

        # Thử detect month từ X nếu có OOF X_train ... nhưng ta chỉ có y và Z
        # Dùng fold indices để approximate mùa theo fold time-window
        season_schema_available = False

        if hasattr(stacking, "_oof_fold_indices_cls") and stacking._oof_fold_indices_cls:
            fold_indices = stacking._oof_fold_indices_cls
            n_folds = len(fold_indices)

            # Approximate: fold đầu = dữ liệu cũ hơn, fold cuối = dữ liệu mới hơn
            # Chia đều n_folds cho 2 mùa (ví dụ: folds 0,1 = rainy; 2,3,4 = dry hoặc mix)
            # Cách đơn giản nhất: mùa mưa = nửa cuối training (recent data), mùa khô = nửa đầu
            # Đây là xấp xỉ — calendar info không có sẵn trong OOF context
            half = n_folds // 2
            rainy_fold_ids = list(range(half, n_folds))     # các fold mới hơn = approximate rainy
            dry_fold_ids   = list(range(0, half))           # các fold cũ hơn = approximate dry

            season_mask_rainy = np.zeros(n_train, dtype=bool)
            season_mask_dry   = np.zeros(n_train, dtype=bool)

            for fi, (tr_idx, val_idx) in enumerate(fold_indices):
                if fi in rainy_fold_ids:
                    season_mask_rainy[val_idx] = True
                else:
                    season_mask_dry[val_idx]   = True

            for sk, s_mask in [("rainy_season", season_mask_rainy),
                                ("dry_season",   season_mask_dry)]:
                n_sk = int(s_mask.sum())
                if n_sk < 10:
                    continue

                entry = SchemaModelEntry(
                    schema_type="season",
                    schema_key=sk,
                    n_samples_cls=n_sk,
                    note="approximate từ fold temporal order (mới hơn=rainy, cũ hơn=dry)",
                )

                y_cls_sk = y_cls[s_mask]
                Z_cls_sk = Z_cls[s_mask]
                cls_scores_sk = {}

                for m_i, m_name in enumerate(cls_names):
                    proba_sk = Z_cls_sk[:, m_i]
                    if len(np.unique(y_cls_sk)) > 1:
                        try:
                            auc = float(roc_auc_score(y_cls_sk, proba_sk))
                        except Exception:
                            auc = float("nan")
                    else:
                        pred_sk = (proba_sk > self.predict_threshold).astype(int)
                        auc = float(f1_score(y_cls_sk, pred_sk, zero_division=0))
                    cls_scores_sk[m_name] = round(auc, 4)

                entry.cls_scores = cls_scores_sk
                valid_scores = {k: v for k, v in cls_scores_sk.items() if np.isfinite(v)}
                if valid_scores:
                    entry.best_cls_model = max(valid_scores, key=valid_scores.get)

                # Regressor
                rainy_in_sk   = s_mask & rainy_mask
                n_rainy_sk    = int(rainy_in_sk.sum())
                entry.n_samples_reg = n_rainy_sk

                if n_rainy_sk >= 5:
                    rainy_in_sk_sub = rainy_in_sk[rainy_mask]
                    y_reg_sk = y_reg_mm[rainy_in_sk_sub]
                    Z_reg_sk = Z_reg[rainy_in_sk_sub]
                    reg_scores_sk = {}

                    for m_i, m_name in enumerate(reg_names):
                        pred_mm_sk = np.expm1(np.clip(Z_reg_sk[:, m_i], 0, 20))
                        try:
                            mae = float(mean_absolute_error(y_reg_sk, pred_mm_sk))
                        except Exception:
                            mae = float("nan")
                        reg_scores_sk[m_name] = round(mae, 4)

                    entry.reg_scores = reg_scores_sk
                    valid_maes = {k: v for k, v in reg_scores_sk.items() if np.isfinite(v)}
                    if valid_maes:
                        entry.best_reg_model = min(valid_maes, key=valid_maes.get)

                self._log(
                    f"    {sk} (n={n_sk}, rainy_sub={n_rainy_sk}): "
                    f"best_cls={entry.best_cls_model} | best_reg={entry.best_reg_model}"
                )
                schema_entries[f"season/{sk}"] = entry

            season_schema_available = True

        # ── 3. Time fold schema ────────────────────────────────────────────────
        self._log("\n  [Schema 3] time_fold (per-fold performance)")

        if hasattr(stacking, "stage_metrics") and "stage7" in stacking.stage_metrics:
            pf_scores = stacking.stage_metrics["stage7"].get("per_fold_model_scores", {})
            pf_reg    = stacking.stage_metrics["stage8"].get("per_fold_model_scores", {})

            for fold_key in sorted(pf_scores.keys()):
                fdata = pf_scores[fold_key]
                val_idx_fold = np.array(
                    stacking._oof_fold_indices_cls[int(fold_key)][1]
                    if hasattr(stacking, "_oof_fold_indices_cls")
                    else [], dtype=int
                )
                n_sk = len(val_idx_fold) if len(val_idx_fold) > 0 else 0

                entry = SchemaModelEntry(
                    schema_type="time_fold",
                    schema_key=f"fold_{fold_key}",
                    n_samples_cls=n_sk,
                )

                cls_scores_fold = {}
                for m_name in cls_names:
                    if m_name in fdata and "roc_auc" in fdata[m_name]:
                        cls_scores_fold[m_name] = fdata[m_name]["roc_auc"]
                entry.cls_scores = cls_scores_fold
                valid_cls = {k: v for k, v in cls_scores_fold.items() if np.isfinite(float(v)) if not isinstance(v, str)}
                if valid_cls:
                    entry.best_cls_model = max(valid_cls, key=lambda k: float(valid_cls[k]))

                reg_scores_fold = {}
                freg_data = pf_reg.get(fold_key, {})
                for m_name in reg_names:
                    if m_name in freg_data and "mae_mm" in freg_data[m_name]:
                        reg_scores_fold[m_name] = freg_data[m_name]["mae_mm"]
                entry.reg_scores = reg_scores_fold
                valid_reg = {k: v for k, v in reg_scores_fold.items() if np.isfinite(float(v)) if not isinstance(v, str)}
                if valid_reg:
                    entry.best_reg_model = min(valid_reg, key=lambda k: float(valid_reg[k]))

                self._log(
                    f"    fold_{fold_key} (n≈{n_sk}): "
                    f"best_cls={entry.best_cls_model} | best_reg={entry.best_reg_model}"
                )
                schema_entries[f"time_fold/fold_{fold_key}"] = entry

        # ── 4. Overall best (fallback cho routing) ─────────────────────────────
        self._log("\n  [Schema 4] overall (fallback routing)")
        all_cls_auc: Dict[str, List[float]] = {n: [] for n in cls_names}
        all_reg_mae: Dict[str, List[float]] = {n: [] for n in reg_names}

        for entry in schema_entries.values():
            for m, v in entry.cls_scores.items():
                if m in all_cls_auc and np.isfinite(v):
                    all_cls_auc[m].append(v)
            for m, v in entry.reg_scores.items():
                if m in all_reg_mae and np.isfinite(v):
                    all_reg_mae[m].append(v)

        avg_cls = {m: float(np.mean(vs)) for m, vs in all_cls_auc.items() if vs}
        avg_reg = {m: float(np.mean(vs)) for m, vs in all_reg_mae.items() if vs}

        overall_best_cls = max(avg_cls, key=avg_cls.get) if avg_cls else (cls_names[0] if cls_names else None)
        overall_best_reg = min(avg_reg, key=avg_reg.get) if avg_reg else (reg_names[0] if reg_names else None)

        self._log(
            f"    overall best_cls={overall_best_cls} (avg_auc={avg_cls.get(overall_best_cls, 'N/A'):.4f}) | "
            f"best_reg={overall_best_reg} (avg_mae={avg_reg.get(overall_best_reg, 'N/A'):.4f})"
        )

        overall_entry = SchemaModelEntry(
            schema_type="overall",
            schema_key="fallback",
            best_cls_model=overall_best_cls,
            best_reg_model=overall_best_reg,
            cls_scores={m: round(v, 4) for m, v in avg_cls.items()},
            reg_scores={m: round(v, 4) for m, v in avg_reg.items()},
            n_samples_cls=n_train,
            n_samples_reg=n_rainy,
            note="Average across all schema slices",
        )
        schema_entries["overall/fallback"] = overall_entry

        # ── 5. Build routing config ────────────────────────────────────────────
        routing = {
            "overall": {
                "cls": overall_best_cls,
                "reg": overall_best_reg,
            },
            "rain_intensity": {},
            "season": {},
            "time_fold": {},
        }
        for sk in RAIN_INTENSITY_BINS:
            key = f"rain_intensity/{sk}"
            if key in schema_entries:
                routing["rain_intensity"][sk] = {
                    "cls": schema_entries[key].best_cls_model or overall_best_cls,
                    "reg": schema_entries[key].best_reg_model or overall_best_reg,
                }

        for sk in ["rainy_season", "dry_season"]:
            key = f"season/{sk}"
            if key in schema_entries:
                routing["season"][sk] = {
                    "cls": schema_entries[key].best_cls_model or overall_best_cls,
                    "reg": schema_entries[key].best_reg_model or overall_best_reg,
                }

        for fi in range(stacking.n_splits if hasattr(stacking, "n_splits") else 5):
            key = f"time_fold/fold_{fi}"
            if key in schema_entries:
                routing["time_fold"][f"fold_{fi}"] = {
                    "cls": schema_entries[key].best_cls_model or overall_best_cls,
                    "reg": schema_entries[key].best_reg_model or overall_best_reg,
                }

        # ── 6. Tính season centroids để routing tại predict time ───────────────
        # Dùng fold temporal split để tính centroid trong scaled feature space
        # (nếu X_train không có sẵn, chỉ lưu routing dựa trên fold)
        season_centroids: Dict[str, Any] = {}
        if season_schema_available:
            # Centroid được tính tại predict time nếu X_train có sẵn
            # Ở đây lưu placeholder để indicate centroid-based routing available
            season_centroids["method"] = "temporal_fold_proxy"
            season_centroids["rainy_fold_ids"] = rainy_fold_ids
            season_centroids["dry_fold_ids"]   = dry_fold_ids
            season_centroids["note"] = (
                "Routing dùng temporal fold proxy. "
                "Gọi update_season_centroids(X_train, feature_names) để dùng "
                "month_sin/month_cos centroid routing chính xác hơn."
            )

        self._routing_config    = routing
        self._season_centroids  = season_centroids
        self._is_analyzed       = True

        # ── Lưu state ──────────────────────────────────────────────────────────
        self._state = SchemaModelBankState(
            trained_at        = _now_tag(),
            n_train_samples   = n_train,
            n_rainy_samples   = n_rainy,
            cls_model_names   = cls_names,
            reg_model_names   = reg_names,
            schema_entries    = {
                k: {
                    "schema_type":   e.schema_type,
                    "schema_key":    e.schema_key,
                    "best_cls_model": e.best_cls_model,
                    "best_reg_model": e.best_reg_model,
                    "cls_scores":    e.cls_scores,
                    "reg_scores":    e.reg_scores,
                    "n_samples_cls": e.n_samples_cls,
                    "n_samples_reg": e.n_samples_reg,
                    "note":          e.note,
                }
                for k, e in schema_entries.items()
            },
            routing_config    = routing,
            season_centroids  = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                  for k, v in season_centroids.items()},
            feature_names     = self.feature_names,
            note              = "FoldSchemaModelBank — per-fold schema analysis",
        )

        self._log("\n  ✅ Schema analysis hoàn tất!")
        self._log(f"  Routing config:\n{json.dumps(routing, ensure_ascii=False, indent=4)}")

        return self

    # ─────────────────────────────────────────────────────────────────────────
    # Cập nhật season centroids từ X_train (tùy chọn, chính xác hơn)
    # ─────────────────────────────────────────────────────────────────────────

    def update_season_centroids(
        self,
        X_train: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "FoldSchemaModelBank":
        """
        Tính lại season centroids từ X_train thực tế để routing
        dựa trên month_sin/month_cos features chính xác hơn.

        Gọi sau analyze() nếu X_train có sẵn.

        Args:
            X_train:       Feature matrix đã transform (n_train, n_features).
            feature_names: Tên features (nếu khác self.feature_names).
        """
        fn = feature_names or self.feature_names
        if "month_sin" not in fn or "month_cos" not in fn:
            self._log("  [update_season_centroids] month_sin/month_cos không có trong feature_names — skip")
            return self

        sin_idx = fn.index("month_sin")
        cos_idx = fn.index("month_cos")

        rainy_mask = self.stacking._oof_rainy_mask if hasattr(self.stacking, "_oof_rainy_mask") else None

        # Chia X_train theo fold temporal proxy (same approach as analyze())
        if hasattr(self.stacking, "_oof_fold_indices_cls"):
            fold_indices = self.stacking._oof_fold_indices_cls
            n_folds = len(fold_indices)
            half = n_folds // 2
            rainy_fold_ids = list(range(half, n_folds))
            dry_fold_ids   = list(range(0, half))

            n_train = X_train.shape[0]
            mask_rainy = np.zeros(n_train, dtype=bool)
            mask_dry   = np.zeros(n_train, dtype=bool)

            for fi, (tr_idx, val_idx) in enumerate(fold_indices):
                if fi in rainy_fold_ids:
                    mask_rainy[val_idx] = True
                else:
                    mask_dry[val_idx]   = True

            centroid_rainy = X_train[mask_rainy][:, [sin_idx, cos_idx]].mean(axis=0)
            centroid_dry   = X_train[mask_dry][:, [sin_idx, cos_idx]].mean(axis=0)

            self._season_centroids = {
                "method":        "month_feature_centroid",
                "rainy_season":  centroid_rainy.tolist(),
                "dry_season":    centroid_dry.tolist(),
                "sin_feature_idx": sin_idx,
                "cos_feature_idx": cos_idx,
                "note": "L1+L2 centroid của scaled month_sin/cos cho mỗi season",
            }

            # Lưu mean/std để denormalize nếu cần
            self._month_stats = {
                "month_sin_mean": float(X_train[:, sin_idx].mean()),
                "month_sin_std":  float(X_train[:, sin_idx].std() + 1e-9),
                "month_cos_mean": float(X_train[:, cos_idx].mean()),
                "month_cos_std":  float(X_train[:, cos_idx].std() + 1e-9),
            }

            if self._state:
                self._state.season_centroids = {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in self._season_centroids.items()
                }

            self._log(
                f"  [update_season_centroids] Đã cập nhật centroids từ X_train "
                f"(rainy={centroid_rainy}, dry={centroid_dry})"
            )

        return self

    # ─────────────────────────────────────────────────────────────────────────
    # Detect schema từ X tại predict time
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_season(self, X: np.ndarray, fn: List[str]) -> np.ndarray:
        """
        Xác định mùa cho từng sample trong X.

        Returns array of "rainy_season" | "dry_season" | "unknown"
        """
        sc = self._season_centroids
        if sc.get("method") == "month_feature_centroid":
            rainy_c = np.array(sc["rainy_season"])
            dry_c   = np.array(sc["dry_season"])
            return _detect_season_from_centroids(X, fn, rainy_c, dry_c)

        # Fallback: không đủ thông tin → unknown
        n = X.shape[0] if X.ndim == 2 else 1
        return np.array(["unknown"] * n)

    def _get_best_models_for_season(
        self, season_key: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Trả về (best_cls_name, best_reg_name) cho season."""
        fallback_cls = self._routing_config.get("overall", {}).get("cls")
        fallback_reg = self._routing_config.get("overall", {}).get("reg")
        season_route = self._routing_config.get("season", {}).get(season_key, {})
        return (season_route.get("cls") or fallback_cls,
                season_route.get("reg") or fallback_reg)

    def _get_best_models_for_rain_intensity(
        self, intensity_key: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Trả về (best_cls_name, best_reg_name) cho rain intensity."""
        fallback_cls = self._routing_config.get("overall", {}).get("cls")
        fallback_reg = self._routing_config.get("overall", {}).get("reg")
        ri_route = self._routing_config.get("rain_intensity", {}).get(intensity_key, {})
        return (ri_route.get("cls") or fallback_cls,
                ri_route.get("reg") or fallback_reg)

    def _get_model_by_name(self, name: str, kind: str) -> Optional[Any]:
        """Lấy model instance từ Stage-9 final models của stacking."""
        stacking = self.stacking
        if kind == "cls":
            names = list(stacking.cls_model_names)
            models = list(stacking.final_cls_models)
        else:
            names = list(stacking.reg_model_names)
            models = list(stacking.final_reg_models)
        if name in names:
            return models[names.index(name)]
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Predict
    # ─────────────────────────────────────────────────────────────────────────

    def predict(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        season_hint: Optional[str] = None,
    ) -> np.ndarray:
        """
        Dự đoán lượng mưa (mm) với schema routing theo mùa.

        Args:
            X:            Feature matrix đã transform (n, n_features).
            feature_names: Tên features (dùng self.feature_names nếu None).
            season_hint:  Ghi đè season detection ("rainy_season" | "dry_season").

        Returns:
            np.ndarray (n,) — dự báo rain_mm.
        """
        result = self.predict_full(X, feature_names=feature_names,
                                   season_hint=season_hint)
        return result["predictions_mm"]

    def predict_full(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        season_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Dự đoán đầy đủ + thông tin routing schema.

        Returns dict:
            predictions_mm     : (n,) rain_mm
            p_rain             : (n,) probability of rain
            has_rain           : (n,) bool
            schema_detected    : (n,) mùa được detect ("rainy_season" | "dry_season" | "unknown")
            model_used_cls     : (n,) tên classifier được dùng cho từng sample
            model_used_reg     : (n,) tên regressor được dùng cho từng sample
            routing_applied    : bool  — True nếu schema routing đã được áp dụng
        """
        if not self._is_analyzed:
            raise RuntimeError("Gọi analyze() trước khi predict.")

        fn     = feature_names or self.feature_names
        X_np   = np.asarray(X, dtype=np.float64)
        n      = X_np.shape[0] if X_np.ndim == 2 else 1
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)

        # ── 1. Detect season ──
        if season_hint:
            seasons = np.array([season_hint] * n)
        else:
            seasons = self._detect_season(X_np, fn)

        # ── 2. Unique seasons → batch predict per season ──
        predictions_mm  = np.zeros(n, dtype=np.float64)
        p_rain_arr      = np.zeros(n, dtype=np.float64)
        model_used_cls  = np.empty(n, dtype=object)
        model_used_reg  = np.empty(n, dtype=object)

        stacking = self.stacking
        fallback_cls_name = self._routing_config.get("overall", {}).get("cls")
        fallback_reg_name = self._routing_config.get("overall", {}).get("reg")

        unique_seasons = np.unique(seasons)
        for sea in unique_seasons:
            idx = np.where(seasons == sea)[0]
            Xi  = X_np[idx]

            cls_name, reg_name = self._get_best_models_for_season(sea)
            cls_name = cls_name or fallback_cls_name
            reg_name = reg_name or fallback_reg_name

            cls_model = self._get_model_by_name(cls_name, "cls")
            reg_model = self._get_model_by_name(reg_name, "reg")

            if cls_model is None:
                # Fallback to full stacking predict
                preds = stacking.predict(Xi)
                predictions_mm[idx]  = preds
                p_rain_arr[idx]       = (preds > self.rain_threshold).astype(float)
                model_used_cls[idx]   = "stacking_fallback"
                model_used_reg[idx]   = "stacking_fallback"
                continue

            # Classification: determine has_rain
            p_rain_i = cls_model.predict_proba(Xi)[:, 1]
            has_rain_i = p_rain_i > self.predict_threshold

            # Regression: predict mm for rainy samples
            pred_mm_i = np.zeros(len(idx), dtype=np.float64)
            rainy_in_batch = np.where(has_rain_i)[0]

            if len(rainy_in_batch) > 0 and reg_model is not None:
                Xi_rainy = Xi[rainy_in_batch]
                log_pred = reg_model.predict(Xi_rainy)
                pred_mm_i[rainy_in_batch] = np.expm1(np.clip(log_pred, 0, 20))
            elif len(rainy_in_batch) > 0:
                # Fallback: use stacking regressor
                Xi_rainy = Xi[rainy_in_batch]
                n_reg = len(stacking.final_reg_models)
                Z_r = np.zeros((len(Xi_rainy), n_reg))
                for mi, rm in enumerate(stacking.final_reg_models):
                    Z_r[:, mi] = rm.predict(Xi_rainy)
                log_pr = stacking.meta_reg.predict(Z_r)
                pred_mm_i[rainy_in_batch] = np.expm1(np.clip(log_pr, 0, 20))
                reg_name = "stacking_reg_fallback"

            predictions_mm[idx] = pred_mm_i
            p_rain_arr[idx]      = p_rain_i
            model_used_cls[idx]  = cls_name
            model_used_reg[idx]  = reg_name

        has_rain_arr = (p_rain_arr > self.predict_threshold)

        return {
            "predictions_mm":  predictions_mm,
            "p_rain":          p_rain_arr,
            "has_rain":        has_rain_arr,
            "schema_detected": seasons,
            "model_used_cls":  model_used_cls,
            "model_used_reg":  model_used_reg,
            "routing_applied": True,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Evaluate so sánh schema routing vs stacking baseline
    # ─────────────────────────────────────────────────────────────────────────

    def compare_with_stacking(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        feature_names: Optional[List[str]] = None,
        dataset_name: str = "eval",
    ) -> Dict[str, Any]:
        """
        So sánh schema routing prediction vs stacking ensemble gốc.

        Returns dict:
            stacking     : metrics của stacking ensemble gốc
            schema_bank  : metrics của schema routing
            improvement  : delta (schema_bank - stacking) cho từng metric
        """
        y_arr   = np.asarray(y_true, dtype=np.float64)
        y_cls_t = (y_arr > self.rain_threshold).astype(int)

        # Stacking baseline
        stack_preds = self.stacking.predict(X)
        stack_cls   = (stack_preds > self.rain_threshold).astype(int)
        stack_f1    = float(f1_score(y_cls_t, stack_cls, zero_division=0))
        stack_mae   = float(mean_absolute_error(y_arr, stack_preds))

        # Schema routing
        schema_preds = self.predict(X, feature_names=feature_names)
        schema_cls   = (schema_preds > self.rain_threshold).astype(int)
        schema_f1    = float(f1_score(y_cls_t, schema_cls, zero_division=0))
        schema_mae   = float(mean_absolute_error(y_arr, schema_preds))

        # Rainy-only MAE
        rainy_mask  = y_arr > self.rain_threshold
        if rainy_mask.sum() > 0:
            st_rainy_mae  = float(mean_absolute_error(y_arr[rainy_mask], stack_preds[rainy_mask]))
            sc_rainy_mae  = float(mean_absolute_error(y_arr[rainy_mask], schema_preds[rainy_mask]))
        else:
            st_rainy_mae = sc_rainy_mae = float("nan")

        result = {
            "dataset":   dataset_name,
            "n_samples": len(y_arr),
            "stacking": {
                "f1_rain":       round(stack_f1, 4),
                "mae_mm_all":    round(stack_mae, 4),
                "mae_mm_rainy":  round(st_rainy_mae, 4) if np.isfinite(st_rainy_mae) else None,
            },
            "schema_bank": {
                "f1_rain":       round(schema_f1, 4),
                "mae_mm_all":    round(schema_mae, 4),
                "mae_mm_rainy":  round(sc_rainy_mae, 4) if np.isfinite(sc_rainy_mae) else None,
            },
            "improvement": {
                "f1_rain":      round(schema_f1 - stack_f1, 4),
                "mae_mm_all":   round(stack_mae - schema_mae, 4),   # positive = schema better
                "mae_mm_rainy": (round(st_rainy_mae - sc_rainy_mae, 4)
                                 if np.isfinite(st_rainy_mae) and np.isfinite(sc_rainy_mae)
                                 else None),
            },
        }

        self._log(
            f"\n[compare_with_stacking] {dataset_name}\n"
            f"  Stacking:    F1={stack_f1:.4f}, MAE={stack_mae:.4f}mm\n"
            f"  SchemaBank:  F1={schema_f1:.4f}, MAE={schema_mae:.4f}mm\n"
            f"  Improvement: ΔF1={schema_f1 - stack_f1:+.4f}, "
            f"ΔMAE={stack_mae - schema_mae:+.4f}mm"
        )

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Save / Load
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, output_dir: Union[str, Path]) -> Path:
        """
        Lưu FoldSchemaModelBank artifacts vào output_dir.

        Artifacts:
            routing_config.json       — routing rules
            performance_report.json   — chi tiết per-schema per-model
            season_centroids.json     — centroids để routing
            model_bank.pkl            — toàn bộ bank object (joblib)
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        if not self._is_analyzed:
            raise RuntimeError("Gọi analyze() trước khi save().")

        # routing_config.json
        _save_json(out / "routing_config.json", self._routing_config)

        # season_centroids.json
        _save_json(out / "season_centroids.json", {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in self._season_centroids.items()
        })

        # performance_report.json
        if self._state:
            _save_json(out / "performance_report.json", {
                "trained_at":    self._state.trained_at,
                "n_train":       self._state.n_train_samples,
                "n_rainy":       self._state.n_rainy_samples,
                "cls_models":    self._state.cls_model_names,
                "reg_models":    self._state.reg_model_names,
                "schema_entries": self._state.schema_entries,
                "routing":        self._state.routing_config,
            })

        # model_bank.pkl  (joblib, không lưu stacking để tránh duplicate)
        if _JOBLIB_OK:
            bank_snapshot = {
                "routing_config":   self._routing_config,
                "season_centroids": {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in self._season_centroids.items()
                },
                "month_stats":      self._month_stats,
                "feature_names":    self.feature_names,
                "predict_threshold": self.predict_threshold,
                "rain_threshold":    self.rain_threshold,
                "state":            self._state,
                "is_analyzed":      self._is_analyzed,
            }
            joblib.dump(bank_snapshot, out / "model_bank.pkl", compress=3)

        self._log(f"  ✅ FoldSchemaModelBank saved → {out}")
        return out

    @classmethod
    def load(
        cls,
        output_dir: Union[str, Path],
        stacking: Any,
        y_train_mm: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> "FoldSchemaModelBank":
        """
        Load FoldSchemaModelBank từ artifacts.

        Args:
            output_dir:    Thư mục chứa artifacts đã save.
            stacking:      StackingEnsemble đã trained (cần để predict).
            y_train_mm:    Optional — chỉ cần nếu muốn gọi analyze() lại.
            feature_names: Optional override.
            verbose:       Logging.
        """
        out = Path(output_dir)
        if not _JOBLIB_OK:
            raise ImportError("Cần cài joblib: pip install joblib")

        bank_pkl = out / "model_bank.pkl"
        if not bank_pkl.exists():
            raise FileNotFoundError(f"Không tìm thấy {bank_pkl}")

        snapshot = joblib.load(bank_pkl)

        fn = feature_names or snapshot.get("feature_names", [])
        y_mm = y_train_mm if y_train_mm is not None else np.array([])

        bank = cls(
            stacking=stacking,
            y_train_mm=y_mm,
            feature_names=fn,
            predict_threshold=snapshot.get("predict_threshold", 0.4),
            rain_threshold=snapshot.get("rain_threshold", 0.1),
            verbose=verbose,
        )

        bank._routing_config   = snapshot.get("routing_config", {})
        bank._month_stats      = snapshot.get("month_stats", {})
        bank._is_analyzed      = snapshot.get("is_analyzed", True)
        bank._state            = snapshot.get("state")

        # Restore season centroids as numpy arrays where needed
        sc_raw = snapshot.get("season_centroids", {})
        season_centroids: Dict[str, Any] = {}
        for k, v in sc_raw.items():
            if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
                season_centroids[k] = np.array(v)
            else:
                season_centroids[k] = v
        bank._season_centroids = season_centroids

        if verbose:
            logger.info(f"FoldSchemaModelBank loaded from {out}")

        return bank

    # ─────────────────────────────────────────────────────────────────────────
    # Truy vấn routing info
    # ─────────────────────────────────────────────────────────────────────────

    def get_routing_config(self) -> Dict[str, Any]:
        """Trả về routing config dict."""
        return dict(self._routing_config)

    def get_performance_report(self) -> Dict[str, Any]:
        """Trả về performance report dict."""
        if self._state:
            return dict(self._state.schema_entries)
        return {}

    def summary(self) -> str:
        """Trả về string tóm tắt FoldSchemaModelBank."""
        if not self._is_analyzed:
            return "FoldSchemaModelBank [chưa analyze]"

        lines = ["=" * 60, "FoldSchemaModelBank Summary", "=" * 60]
        if self._state:
            lines.append(f"  Trained at  : {self._state.trained_at}")
            lines.append(f"  n_train     : {self._state.n_train_samples}")
            lines.append(f"  n_rainy     : {self._state.n_rainy_samples}")
            lines.append(f"  cls_models  : {', '.join(self._state.cls_model_names)}")
            lines.append(f"  reg_models  : {', '.join(self._state.reg_model_names)}")

        lines.append("\n  Routing Config:")
        overall = self._routing_config.get("overall", {})
        lines.append(f"    overall fallback: cls={overall.get('cls')} | reg={overall.get('reg')}")

        ri_routes = self._routing_config.get("rain_intensity", {})
        if ri_routes:
            lines.append("  Rain Intensity Routing:")
            for k, v in ri_routes.items():
                lines.append(f"    {k}: cls={v.get('cls')} | reg={v.get('reg')}")

        sea_routes = self._routing_config.get("season", {})
        if sea_routes:
            lines.append("  Season Routing:")
            for k, v in sea_routes.items():
                lines.append(f"    {k}: cls={v.get('cls')} | reg={v.get('reg')}")

        lines.append("=" * 60)
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal logging
    # ─────────────────────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if self.verbose:
            logger.info(msg)
            print(msg)
