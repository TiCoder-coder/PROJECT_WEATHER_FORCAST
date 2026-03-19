"""
predictor_by_stacking_ensemble.py
==================================
Interface dự đoán thống nhất cho toàn bộ pipeline StackingEnsemble + Schema-Aware Routing.

Luồng đầy đủ (Training → Predict)
===================================

TRAINING (train_stacking_ensemble.py):
  BƯỚC 1-4  : Load data → split (80/10/10 chronological) → build features → remove static/const
  BƯỚC 5    : WeatherTransformPipeline.fit_transform()  → lưu Transform_pipeline.pkl
  BƯỚC 6    : Feature_list.json  (all_feature_columns + metadata)
  BƯỚC 7    : StackingEnsemble.fit()  →  Stage 6: verify  →  Stage 7: OOF cls  →  Stage 8: OOF reg
                                         Stage 9: refit on full train
              Lưu OOF attributes: _oof_Z_cls, _oof_y_cls, _oof_Z_reg, _oof_y_reg_mm,
                                   _oof_rainy_mask, _oof_fold_indices_cls/_reg
  BƯỚC 8-10 : Evaluate (val + test) → Metrics.json → save Model.pkl / stacking_ensemble_<ts>.joblib
  BƯỚC 11   : FoldSchemaModelBank.from_stacking() → .analyze() → .update_season_centroids()
              → .save(schema_model_bank/)
              Phân tích per-model per-schema (rain_intensity / season / time_fold)
              Lưu: routing_config.json, performance_report.json, season_centroids.json, model_bank.pkl
  BƯỚC 12   : Train_info.json  (full metadata, artifacts paths, schema_bank_routing)

PREDICT (StackingPredictor):
  1. Load Model.pkl        → StackingEnsemble (Stage-9 final models + meta-models)
  2. Load Transform_pipeline.pkl → WeatherTransformPipeline
  3. Load Feature_list.json       → all_feature_columns, target_column
  4. Load Train_info.json         → feature_builder_config, forecast_horizon, ...
  5. Load schema_model_bank/      → FoldSchemaModelBank (nếu có)
  6. Nhận DataFrame → rename cột → build features (WeatherFeatureBuilder) → align
  7. Pipeline.transform()
  8. Coerce object/datetime columns → numeric
  9. Align với all_feature_columns (thêm missing=0, xóa extra, reorder)
  10. Predict via Schema Bank:
        a. Detect season từ month_sin/month_cos centroids
        b. Route → best classifier + best regressor cho season đó
        c. p_rain = cls_model.predict_proba()[:, 1]
        d. rain_mm = reg_model.predict() qua expm1 gate (if p_rain > threshold)
      Fallback → StackingEnsemble.predict_full() nếu schema bank không có

  Trả về:  dict với predictions, p_rain, has_rain, schema_info, prediction_time

Cấu trúc artifacts
===================
Machine_learning_artifacts/stacking_ensemble/latest/
    Model.pkl                    # StackingEnsemble (Stage-9 full train)
    stacking_ensemble_<ts>.joblib # bản sao dạng joblib (backup)
    Transform_pipeline.pkl
    Feature_list.json
    Metrics.json
    Train_info.json
    schema_model_bank/
        routing_config.json      # schema → {cls: model_name, reg: model_name}
        performance_report.json  # per-schema per-model AUC/MAE
        season_centroids.json    # centroids scaled space (month_sin/cos)
        model_bank.pkl           # FoldSchemaModelBank snapshot (joblib)

Sử dụng
========
    # Load predictor (auto-detect artifacts dir)
    predictor = StackingPredictor.from_artifacts()

    # Hoặc chỉ định path cụ thể
    predictor = StackingPredictor.from_artifacts("/path/to/stacking_ensemble/latest")

    # Predict từ raw DataFrame
    result = predictor.predict_full(df)
    rain_mm        = result["predictions"]         # (n,) rain mm
    p_rain         = result["p_rain"]              # (n,) probability
    schemas        = result["schema_info"]["schema_detected"]  # (n,) season
    model_cls_used = result["schema_info"]["model_used_cls"]   # (n,) model name

    # Predict đơn giản
    rain_mm = predictor.predict(df)

    # Thông tin model
    print(predictor.get_info())
    print(predictor.summary())

CLI
===
    python predictor_by_stacking_ensemble.py --input data.csv --output preds.csv
    python predictor_by_stacking_ensemble.py --input data.csv --artifacts path/to/latest
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Đường dẫn mặc định
# ─────────────────────────────────────────────────────────────────────────────
_THIS_FILE   = Path(__file__).resolve()
_APP_ROOT    = _THIS_FILE.parent.parent.parent   # Weather_Forcast_App/
_ML_ROOT     = _THIS_FILE.parent.parent          # Machine_learning_model/

# Artifacts của StackingEnsemble (khác với ensemble_average artifacts/latest)
DEFAULT_STACKING_ARTIFACTS = (
    _APP_ROOT / "Machine_learning_artifacts" / "stacking_ensemble" / "latest"
)

# Cột identifier/metadata không dùng làm feature
_NON_FEATURE_COLS = frozenset({
    "timestamp", "data_time", "data_quality",
    "location_station_id", "location_station_name",
    "location_province", "location_district",
})

# Rename map từ schema crawler → tên feature builder chuẩn
_RENAME_MAP: Dict[str, str] = {
    "station_id":    "location_station_id",
    "station_name":  "location_station_name",
    "province":      "location_province",
    "district":      "location_district",
    "latitude":      "location_latitude",
    "longitude":     "location_longitude",
}


# =============================================================================
# StackingPredictor
# =============================================================================

class StackingPredictor:
    """
    Interface dự đoán thống nhất cho StackingEnsemble + FoldSchemaModelBank.

    Hỗ trợ hai chế độ:
      - **Schema routing**: nếu FoldSchemaModelBank đã được build và lưu
        → route mỗi sample đến classifier/regressor phù hợp nhất theo mùa
      - **Full stacking**: fallback về StackingEnsemble.predict_full() gốc

    Public API:
        predict(df)       → np.ndarray rain mm
        predict_full(df)  → dict (predictions, p_rain, has_rain, schema_info, ...)
        get_info()        → dict thông tin model/artifacts
        summary()         → string tóm tắt
        from_artifacts()  → factory classmethod
    """

    # ──────────────────────────────────────────────────────────────────────
    # Constructor
    # ──────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        stacking: Any,
        pipeline: Any,
        feature_columns: List[str],
        target_column: str = "rain_total",
        feature_builder: Any = None,
        schema_bank: Any = None,
        train_info: Optional[Dict[str, Any]] = None,
        artifacts_dir: Optional[Path] = None,
    ) -> None:
        """
        Args:
            stacking:        StackingEnsemble đã trained (Stage-9 models + meta-models).
            pipeline:        WeatherTransformPipeline đã fit.
            feature_columns: Danh sách tất cả feature columns sau build + selection.
            target_column:   Tên cột target (mặc định "rain_total").
            feature_builder: WeatherFeatureBuilder khởi tạo với config từ train.
                             None → không build features (dùng khi df đã có sẵn features).
            schema_bank:     FoldSchemaModelBank (schema-aware routing). None → dùng stacking.
            train_info:      Dict từ Train_info.json.
            artifacts_dir:   Đường dẫn thư mục artifacts (để log).
        """
        self.stacking        = stacking
        self.pipeline        = pipeline
        self.feature_columns = list(feature_columns)
        self.target_column   = target_column
        self.feature_builder = feature_builder
        self.schema_bank     = schema_bank
        self.train_info      = train_info or {}
        self.artifacts_dir   = artifacts_dir

        # Thông số từ StackingEnsemble
        self.predict_threshold: float = getattr(stacking, "predict_threshold", 0.4)
        self.rain_threshold: float    = getattr(stacking, "rain_threshold", 0.1)
        self.forecast_horizon: int    = int(self.train_info.get("forecast_horizon", 0))

    # ──────────────────────────────────────────────────────────────────────
    # Factory
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def from_artifacts(
        cls,
        artifacts_dir: Optional[Union[str, Path]] = None,
        *,
        load_schema_bank: bool = True,
        verbose: bool = True,
    ) -> "StackingPredictor":
        """
        Load StackingPredictor từ thư mục artifacts.

        Args:
            artifacts_dir:    Đường dẫn đến thư mục chứa artifacts stacking.
                              None → dùng Machine_learning_artifacts/stacking_ensemble/latest/
            load_schema_bank: True → cố gắng load FoldSchemaModelBank.
                              False → chỉ dùng StackingEnsemble thuần.
            verbose:          In log khi load.

        Returns:
            StackingPredictor sẵn sàng predict.

        Raises:
            FileNotFoundError: Khi artifacts_dir không tồn tại hoặc thiếu Model.pkl.
        """
        if artifacts_dir is None:
            artifacts_dir = DEFAULT_STACKING_ARTIFACTS
        art_dir = Path(artifacts_dir).resolve()

        if not art_dir.exists():
            raise FileNotFoundError(
                f"Artifacts dir không tồn tại: {art_dir}\n"
                f"Hãy chạy train_stacking_ensemble.py trước."
            )

        _log = logger.info if verbose else lambda *a, **k: None
        _log(f"[StackingPredictor] Loading artifacts từ: {art_dir}")

        # ── 1. Load StackingEnsemble (Model.pkl) ──────────────────────────
        model_path = art_dir / "Model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model.pkl không tìm thấy trong {art_dir}.\n"
                f"Hãy chạy train_stacking_ensemble.py để tạo artifacts."
            )
        try:
            import joblib
            stacking = joblib.load(model_path)
            _log(f"  ✓ Model.pkl loaded  (type={type(stacking).__name__}, "
                 f"trained={getattr(stacking, 'is_trained', '?')})")
        except Exception as e:
            raise RuntimeError(f"Không load được Model.pkl: {e}") from e

        # ── 2. Load WeatherTransformPipeline ─────────────────────────────
        pipeline = None
        pipeline_path = art_dir / "Transform_pipeline.pkl"
        if pipeline_path.exists():
            try:
                from Weather_Forcast_App.Machine_learning_model.features.Transformers import (
                    WeatherTransformPipeline,
                )
                pipeline = WeatherTransformPipeline.load(pipeline_path)
                _log("  ✓ Transform_pipeline.pkl loaded")
            except Exception as e:
                logger.warning(f"  ⚠ Không load được pipeline: {e} — sẽ predict không transform")

        # ── 3. Load Feature_list.json ─────────────────────────────────────
        feature_columns: List[str] = []
        target_column   = "rain_total"
        feat_meta: Dict[str, Any] = {}
        feature_list_path = art_dir / "Feature_list.json"
        if feature_list_path.exists():
            with open(feature_list_path, encoding="utf-8") as f:
                feat_meta = json.load(f)
            feature_columns = feat_meta.get("all_feature_columns", [])
            target_column   = feat_meta.get("target_column", "rain_total")
            _log(f"  ✓ Feature_list.json loaded  ({len(feature_columns)} features, "
                 f"target='{target_column}')")
        else:
            logger.warning(f"  ⚠ Feature_list.json không tìm thấy trong {art_dir}")

        # ── 4. Load Train_info.json ───────────────────────────────────────
        train_info: Dict[str, Any] = {}
        info_path = art_dir / "Train_info.json"
        if info_path.exists():
            with open(info_path, encoding="utf-8") as f:
                train_info = json.load(f)
            _log(f"  ✓ Train_info.json loaded  "
                 f"(trained_at={train_info.get('trained_at', '?')})")

        # ── 5. Khởi tạo WeatherFeatureBuilder với đúng config từ training ─
        feature_builder = None
        fb_config = train_info.get("feature_builder_config")
        if fb_config is not None:
            try:
                from Weather_Forcast_App.Machine_learning_model.features.Build_transfer import (
                    WeatherFeatureBuilder,
                )
                feature_builder = WeatherFeatureBuilder(config=fb_config)
                _log("  ✓ WeatherFeatureBuilder khởi tạo với config từ training")
            except ImportError as e:
                logger.warning(f"  ⚠ Không import được WeatherFeatureBuilder: {e}")
        elif feat_meta.get("created_features"):
            # Fallback: nếu không có config nhưng có created_features → builder mặc định
            try:
                from Weather_Forcast_App.Machine_learning_model.features.Build_transfer import (
                    WeatherFeatureBuilder,
                )
                feature_builder = WeatherFeatureBuilder()
                _log("  ✓ WeatherFeatureBuilder khởi tạo với config mặc định (fallback)")
            except ImportError:
                pass

        # ── 6. Load FoldSchemaModelBank (tùy chọn) ───────────────────────
        schema_bank = None
        if load_schema_bank:
            # Ưu tiên đường dẫn từ Train_info.json
            bank_dir_from_info = train_info.get("artifacts", {}).get("schema_model_bank")
            bank_dir = (
                Path(bank_dir_from_info)
                if bank_dir_from_info
                else art_dir / "schema_model_bank"
            )
            bank_pkl = bank_dir / "model_bank.pkl"
            if bank_pkl.exists():
                try:
                    from Weather_Forcast_App.Machine_learning_model.Models.Schema_Selector import (
                        FoldSchemaModelBank,
                    )
                    schema_bank = FoldSchemaModelBank.load(
                        bank_dir,
                        stacking=stacking,
                        feature_names=feature_columns or None,
                        verbose=verbose,
                    )
                    _log(f"  ✓ FoldSchemaModelBank loaded từ {bank_dir}")
                    if verbose:
                        try:
                            _log(schema_bank.summary())
                        except Exception:
                            pass
                except Exception as e:
                    logger.warning(
                        f"  ⚠ Không load được FoldSchemaModelBank: {e} "
                        f"→ sẽ dùng StackingEnsemble trực tiếp"
                    )
            else:
                _log(
                    f"  ℹ Schema bank chưa được build "
                    f"({bank_pkl} không tồn tại) → dùng StackingEnsemble"
                )

        instance = cls(
            stacking=stacking,
            pipeline=pipeline,
            feature_columns=feature_columns,
            target_column=target_column,
            feature_builder=feature_builder,
            schema_bank=schema_bank,
            train_info=train_info,
            artifacts_dir=art_dir,
        )
        _log(
            f"[StackingPredictor] Sẵn sàng predict  "
            f"(routing={'schema_bank' if schema_bank else 'stacking_direct'})"
        )
        return instance

    # ──────────────────────────────────────────────────────────────────────
    # Tiền xử lý DataFrame → numpy
    # ──────────────────────────────────────────────────────────────────────

    def _prepare_features(
        self,
        df: pd.DataFrame,
        build_features: bool = True,
        group_by: Optional[str] = None,
    ) -> np.ndarray:
        """
        Toàn bộ pipeline chuẩn bị features từ raw DataFrame:

        1. Copy + rename cột crawler → tên chuẩn
        2. Build features (nếu build_features=True và có feature_builder)
        3. Drop target_column (nếu còn trong df)
        4. Drop _non_feature_cols
        5. Align với feature_columns (thêm missing=0, xóa extra, reorder)
        6. WeatherTransformPipeline.transform()
        7. Coerce object / datetime → numeric

        Returns:
            np.ndarray shape (n_samples, n_features), dtype float64.
        """
        X = df.copy()

        # 1. Rename cột từ crawler schema → tên feature builder chuẩn
        rename_map = {old: new for old, new in _RENAME_MAP.items()
                      if old in X.columns and new not in X.columns}
        if rename_map:
            X = X.rename(columns=rename_map)

        # 2. Build features với WeatherFeatureBuilder
        if build_features and self.feature_builder is not None:
            fb_target = self.target_column
            # Thêm group_by từ train_info nếu chưa truyền vào
            if group_by is None:
                group_by = self.train_info.get("group_by")
            try:
                X = self.feature_builder.build_all_features(
                    X,
                    target_column=fb_target,
                    group_by=group_by,
                )
            except Exception as e:
                logger.warning(
                    f"  ⚠ build_all_features thất bại ({e}) "
                    f"→ tiếp tục với features hiện có"
                )

        # 3. Drop target nếu có
        if self.target_column and self.target_column in X.columns:
            X = X.drop(columns=[self.target_column])

        # 4. Drop các cột identifier / metadata không phải feature
        drop_meta = [c for c in _NON_FEATURE_COLS if c in X.columns]
        if drop_meta:
            X = X.drop(columns=drop_meta)

        # 5. Align theo feature_columns
        if self.feature_columns:
            # Thêm cột thiếu với giá trị 0
            for col in self.feature_columns:
                if col not in X.columns:
                    X[col] = 0.0
            # Loại cột thừa
            extra = [c for c in X.columns if c not in self.feature_columns]
            if extra:
                X = X.drop(columns=extra)
            # Reorder đúng thứ tự
            X = X[self.feature_columns]

        # 6. Pipeline transform
        if self.pipeline is not None:
            try:
                X = self.pipeline.transform(X)
            except Exception as e:
                logger.warning(
                    f"  ⚠ pipeline.transform() lỗi ({e}) "
                    f"→ tiếp tục với dữ liệu chưa transform"
                )

        # 7. Coerce object / datetime → numeric
        if isinstance(X, pd.DataFrame):
            for col in list(X.columns):
                if X[col].dtype == "object":
                    try:
                        converted = pd.to_datetime(X[col], errors="coerce")
                        if converted.notna().any():
                            X[col] = converted
                    except Exception:
                        pass
                if pd.api.types.is_datetime64_any_dtype(X[col]):
                    X[col] = X[col].astype(np.int64) // 10 ** 9  # epoch seconds

            # Remaining object cols → category code
            obj_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
            for col in obj_cols:
                try:
                    X[col] = X[col].astype("category").cat.codes.astype(float)
                except Exception:
                    X[col] = 0.0

            X = X.astype(np.float64)

        return X if isinstance(X, np.ndarray) else X.values

    # ──────────────────────────────────────────────────────────────────────
    # Dự đoán chính
    # ──────────────────────────────────────────────────────────────────────

    def predict(
        self,
        df: pd.DataFrame,
        *,
        build_features: bool = True,
        group_by: Optional[str] = None,
        season_hint: Optional[str] = None,
    ) -> np.ndarray:
        """
        Dự đoán lượng mưa (mm) từ DataFrame.

        Args:
            df:             DataFrame đầu vào (raw data hoặc đã build features).
            build_features: True → tự build features qua WeatherFeatureBuilder.
                            False → dùng df như hiện có (khi đã preprocess thủ công).
            group_by:       Cột group by cho feature builder (None → lấy từ train_info).
            season_hint:    Ghi đè season detection ("rainy_season" | "dry_season").
                            Chỉ dùng khi schema bank khả dụng.

        Returns:
            np.ndarray shape (n_samples,), dự báo rain_mm >= 0.
        """
        result = self.predict_full(
            df,
            build_features=build_features,
            group_by=group_by,
            season_hint=season_hint,
        )
        return result["predictions"]

    def predict_full(
        self,
        df: pd.DataFrame,
        *,
        build_features: bool = True,
        group_by: Optional[str] = None,
        season_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Dự đoán đầy đủ, trả về dict với tất cả thông tin kèm theo.

        Args:
            df:             DataFrame đầu vào.
            build_features: True → build features trước.
            group_by:       Cột group by cho feature builder.
            season_hint:    Ghi đè season detection.

        Returns:
            Dict:
                predictions         (np.ndarray)  : rain_mm dự báo
                p_rain              (np.ndarray)  : xác suất mưa [0, 1]
                has_rain            (np.ndarray)  : nhãn binary (0/1)
                rain_mm_ungated     (np.ndarray)  : mm từ regression không qua gate
                prediction_time     (float)       : thời gian tính bằng giây
                n_samples           (int)
                forecast_horizon    (int)         : số giờ dự báo trước
                routing_mode        (str)         : "schema_bank" | "stacking_direct"
                schema_info         (dict | None) : chi tiết routing (chỉ khi schema bank)
                    schema_detected     (np.ndarray) : season per sample
                    model_used_cls      (np.ndarray) : tên classifier per sample
                    model_used_reg      (np.ndarray) : tên regressor per sample
                    routing_applied     (bool)
        """
        t0 = time.perf_counter()

        # ── Chuẩn bị features ──
        try:
            X_np = self._prepare_features(df, build_features=build_features, group_by=group_by)
        except Exception as e:
            logger.error(f"[StackingPredictor] _prepare_features thất bại: {e}")
            raise

        # ── Dự đoán ──
        routing_mode = "stacking_direct"
        schema_info: Optional[Dict[str, Any]] = None

        if self.schema_bank is not None:
            # ── Chế độ 1: Schema-aware routing qua FoldSchemaModelBank ──
            try:
                result_schema = self.schema_bank.predict_full(
                    X_np,
                    feature_names=self.feature_columns if self.feature_columns else None,
                    season_hint=season_hint,
                )
                predictions    = result_schema["predictions_mm"]
                p_rain         = result_schema["p_rain"]
                has_rain       = result_schema["has_rain"].astype(np.int32)

                # rain_mm_ungated: tính từ mean regressor OOF (không qua cls gate)
                # → dùng full stacking regressor stack để nhất quán
                try:
                    full_st = self.stacking.predict_full(X_np)
                    rain_mm_ungated = full_st["rain_mm_ungated"]
                except Exception:
                    rain_mm_ungated = predictions.copy()

                routing_mode = "schema_bank"
                schema_info = {
                    "schema_detected": result_schema.get("schema_detected"),
                    "model_used_cls":  result_schema.get("model_used_cls"),
                    "model_used_reg":  result_schema.get("model_used_reg"),
                    "routing_applied": result_schema.get("routing_applied", True),
                }
            except Exception as e:
                logger.warning(
                    f"  ⚠ FoldSchemaModelBank.predict_full() lỗi ({e}) "
                    f"→ fallback về StackingEnsemble"
                )
                # Fallback
                full_st         = self.stacking.predict_full(X_np)
                predictions     = full_st["predictions"]
                p_rain          = full_st["p_rain"]
                has_rain        = full_st["has_rain"]
                rain_mm_ungated = full_st["rain_mm_ungated"]
                routing_mode    = "stacking_direct"
        else:
            # ── Chế độ 2: StackingEnsemble trực tiếp ──
            full_st         = self.stacking.predict_full(X_np)
            predictions     = full_st["predictions"]
            p_rain          = full_st["p_rain"]
            has_rain        = full_st["has_rain"]
            rain_mm_ungated = full_st["rain_mm_ungated"]

        # Clip âm → 0
        predictions     = np.clip(predictions, 0.0, None)
        rain_mm_ungated = np.clip(rain_mm_ungated, 0.0, None)

        elapsed = time.perf_counter() - t0

        return {
            "predictions":      predictions,
            "p_rain":           p_rain,
            "has_rain":         has_rain,
            "rain_mm_ungated":  rain_mm_ungated,
            "prediction_time":  elapsed,
            "n_samples":        len(df),
            "forecast_horizon": self.forecast_horizon,
            "routing_mode":     routing_mode,
            "schema_info":      schema_info,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Thông tin & tóm tắt
    # ──────────────────────────────────────────────────────────────────────

    def get_info(self) -> Dict[str, Any]:
        """
        Trả về dict thông tin đầy đủ về predictor.

        Bao gồm:
          - model_type, trained_at, n_features
          - stacking config (n_splits, predict_threshold, rain_threshold)
          - pipeline info
          - schema_bank info (nếu có)
          - artifacts paths
        """
        info: Dict[str, Any] = {
            "model_type":        type(self.stacking).__name__,
            "target_column":     self.target_column,
            "n_features":        len(self.feature_columns),
            "forecast_horizon":  self.forecast_horizon,
            "predict_threshold": self.predict_threshold,
            "rain_threshold":    self.rain_threshold,
            "has_pipeline":      self.pipeline is not None,
            "has_feature_builder": self.feature_builder is not None,
            "has_schema_bank":   self.schema_bank is not None,
            "routing_mode":      "schema_bank" if self.schema_bank else "stacking_direct",
            "artifacts_dir":     str(self.artifacts_dir) if self.artifacts_dir else None,
        }

        # Thông tin từ Train_info.json
        if self.train_info:
            info["trained_at"]   = self.train_info.get("trained_at", "unknown")
            info["model_type"]   = self.train_info.get("model_type", info["model_type"])

            stacking_cfg = self.train_info.get("stacking_config", {})
            info["stacking_config"] = {
                "n_splits":          stacking_cfg.get("n_splits", getattr(self.stacking, "n_splits", None)),
                "predict_threshold": stacking_cfg.get("predict_threshold", self.predict_threshold),
                "rain_threshold":    stacking_cfg.get("rain_threshold", self.rain_threshold),
                "cls_model_names":   stacking_cfg.get("cls_model_names", getattr(self.stacking, "cls_model_names", [])),
                "reg_model_names":   stacking_cfg.get("reg_model_names", getattr(self.stacking, "reg_model_names", [])),
                "n_cls_oof_samples": stacking_cfg.get("n_cls_oof_samples"),
                "n_reg_oof_samples": stacking_cfg.get("n_reg_oof_samples"),
            }

            info["feature_info"] = self.train_info.get("feature_info", {})
            info["target_info"]  = self.train_info.get("target_info", {})
            info["artifacts"]    = self.train_info.get("artifacts", {})

        # Pipeline info
        if self.pipeline is not None and hasattr(self.pipeline, "get_pipeline_info"):
            info["pipeline_info"] = self.pipeline.get_pipeline_info()

        # Schema bank routing config
        if self.schema_bank is not None:
            try:
                info["schema_routing"] = self.schema_bank.get_routing_config()
            except Exception:
                pass

        return info

    def get_schema_report(self) -> Optional[Dict[str, Any]]:
        """
        Trả về performance report của FoldSchemaModelBank (per-schema per-model AUC/MAE).
        None nếu schema bank chưa load.
        """
        if self.schema_bank is None:
            return None
        try:
            return self.schema_bank.get_performance_report()
        except Exception:
            return None

    def compare_routing(
        self,
        df: pd.DataFrame,
        y_true: Union[np.ndarray, pd.Series],
        *,
        build_features: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        So sánh schema routing vs full stacking trên một tập dữ liệu đã có nhãn.

        Args:
            df:             DataFrame có features (hoặc raw nếu build_features=True).
            y_true:         Ground truth rain_mm.
            build_features: True → build features trước.

        Returns:
            Dict từ FoldSchemaModelBank.compare_with_stacking() — None nếu không có schema bank.
        """
        if self.schema_bank is None:
            logger.info("compare_routing: không có schema_bank → return None")
            return None
        try:
            X_np = self._prepare_features(df, build_features=build_features)
            y_arr = np.asarray(y_true, dtype=np.float64)
            return self.schema_bank.compare_with_stacking(
                X_np, y_arr,
                feature_names=self.feature_columns if self.feature_columns else None,
                dataset_name="user_data",
            )
        except Exception as e:
            logger.error(f"compare_routing lỗi: {e}")
            return None

    def summary(self) -> str:
        """
        Tóm tắt trạng thái predictor dưới dạng string human-readable.
        """
        info = self.get_info()
        lines = [
            "=" * 60,
            "StackingPredictor Summary",
            "=" * 60,
            f"  Trained at      : {info.get('trained_at', 'unknown')}",
            f"  Model type      : {info.get('model_type', '?')}",
            f"  Target          : {self.target_column}",
            f"  Features        : {info['n_features']}",
            f"  Forecast horizon: {self.forecast_horizon}h",
            f"  Predict threshold: {self.predict_threshold}",
            f"  Rain threshold  : {self.rain_threshold} mm",
            f"  Pipeline        : {'✓' if self.pipeline else '✗'}",
            f"  Feature builder : {'✓' if self.feature_builder else '✗'}",
            f"  Routing mode    : {info['routing_mode']}",
        ]

        # Stacking config
        s_cfg = info.get("stacking_config", {})
        if s_cfg:
            cls_names = ", ".join(s_cfg.get("cls_model_names") or [])
            reg_names = ", ".join(s_cfg.get("reg_model_names") or [])
            lines += [
                f"  n_splits        : {s_cfg.get('n_splits', '?')}",
                f"  Classifiers     : [{cls_names}]",
                f"  Regressors      : [{reg_names}]",
            ]

        # Schema bank
        if self.schema_bank is not None:
            try:
                lines.append("")
                lines.append("  FoldSchemaModelBank:")
                rc = self.schema_bank.get_routing_config()
                overall = rc.get("overall", {})
                lines.append(
                    f"    overall fallback → cls={overall.get('cls', '?')} "
                    f"| reg={overall.get('reg', '?')}"
                )
                ri = rc.get("rain_intensity", {})
                for key, route in ri.items():
                    lines.append(f"    rain_intensity/{key:12s} → cls={route.get('cls','?')} | reg={route.get('reg','?')}")
                sea = rc.get("season", {})
                for key, route in sea.items():
                    lines.append(f"    season/{key:20s} → cls={route.get('cls','?')} | reg={route.get('reg','?')}")
            except Exception:
                lines.append("  FoldSchemaModelBank: (không đọc được routing config)")

        # Artifacts
        arts = info.get("artifacts", {})
        if arts:
            lines.append("")
            lines.append("  Artifacts:")
            for k, v in arts.items():
                if v:
                    lines.append(f"    {k}: ...{str(v)[-50:]}")

        lines.append("=" * 60)
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────
    # Compatibility alias (View_Predict.py uses predictor.model)
    # ──────────────────────────────────────────────────────────────────────

    @property
    def model(self) -> Any:
        """Alias của self.stacking — tương thích interface WeatherPredictor cũ."""
        return self.stacking

    def __repr__(self) -> str:
        return (
            f"StackingPredictor("
            f"model=StackingEnsemble, "
            f"features={len(self.feature_columns)}, "
            f"target='{self.target_column}', "
            f"routing='{('schema_bank' if self.schema_bank else 'stacking_direct')}')"
        )


# =============================================================================
# Convenience factory (tương thích với View_Predict.py gọi WeatherPredictor)
# =============================================================================

class WeatherPredictorStacking(StackingPredictor):
    """
    Alias của StackingPredictor với tên quen thuộc WeatherPredictor để
    tương thích khi View_Predict.py cần import.

    Sử dụng:
        from Weather_Forcast_App.Machine_learning_model.interface.predictor_by_stacking_ensemble import (
            WeatherPredictorStacking
        )
        predictor = WeatherPredictorStacking.from_artifacts()
        result = predictor.predict_full(df)
    """

    def predict(
        self,
        df: pd.DataFrame,
        *,
        build_features: bool = True,
        group_by: Optional[str] = None,
        season_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Override để trả về dict (tương thích interface WeatherPredictor cũ).

        Returns:
            Dict:
                predictions     : np.ndarray rain_mm
                prediction_time : float (giây)
                n_samples       : int
                forecast_horizon: int
                p_rain          : np.ndarray
                has_rain        : np.ndarray
                routing_mode    : str
                schema_info     : dict | None
        """
        return self.predict_full(
            df,
            build_features=build_features,
            group_by=group_by,
            season_hint=season_hint,
        )


# =============================================================================
# CLI Entry Point
# =============================================================================

def _cli_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="StackingEnsemble Weather Predictor — Schema-Aware Inference"
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Path to input CSV file.",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Path to output CSV file (default: <input>_stacking_pred.csv).",
    )
    parser.add_argument(
        "--artifacts", "-a", type=str, default=None,
        help=(
            "Path to artifacts directory "
            "(default: Machine_learning_artifacts/stacking_ensemble/latest)."
        ),
    )
    parser.add_argument(
        "--no-schema-bank", action="store_true",
        help="Tắt schema-aware routing, chỉ dùng StackingEnsemble trực tiếp.",
    )
    parser.add_argument(
        "--season-hint", type=str, default=None,
        choices=["rainy_season", "dry_season"],
        help="Ghi đè season detection cho tất cả samples.",
    )
    parser.add_argument(
        "--nrows", type=int, default=None,
        help="Chỉ đọc N dòng đầu tiên (để test nhanh).",
    )
    parser.add_argument(
        "--no-build-features", action="store_true",
        help="Bỏ qua bước build features (dùng khi df đã có features).",
    )
    args = parser.parse_args()

    # ── Setup logging ──
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Load predictor ──
    print(f"[CLI] Loading predictor từ: {args.artifacts or DEFAULT_STACKING_ARTIFACTS}")
    predictor = StackingPredictor.from_artifacts(
        args.artifacts,
        load_schema_bank=not args.no_schema_bank,
        verbose=True,
    )
    print(predictor.summary())

    # ── Đọc data ──
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[CLI] ❌ Input file không tồn tại: {input_path}")
        return

    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path, nrows=args.nrows)
    else:
        df = pd.read_excel(input_path, nrows=args.nrows)

    print(f"[CLI] Đọc {len(df)} dòng, {len(df.columns)} cột từ {input_path.name}")

    # ── Predict ──
    print("[CLI] Đang dự đoán...")
    result = predictor.predict_full(
        df,
        build_features=not args.no_build_features,
        season_hint=args.season_hint,
    )

    predictions     = result["predictions"]
    p_rain          = result["p_rain"]
    has_rain        = result["has_rain"]
    pred_time       = result["prediction_time"]
    routing_mode    = result["routing_mode"]
    schema_info     = result.get("schema_info")

    print(
        f"[CLI] ✅ Dự đoán xong  "
        f"({len(predictions)} mẫu | {pred_time:.3f}s | routing={routing_mode})"
    )
    print(
        f"  rain_mm: mean={predictions.mean():.3f}, "
        f"max={predictions.max():.3f}, "
        f"rainy_count={has_rain.sum()}"
    )
    if schema_info:
        seas, cnts = np.unique(schema_info["schema_detected"], return_counts=True)
        print("  Season distribution:")
        for s, c in zip(seas, cnts):
            print(f"    {s}: {c} samples")

    # ── Gắn kết quả vào df ──
    df["y_pred"]  = predictions
    df["p_rain"]  = p_rain
    df["has_rain"] = has_rain
    if schema_info is not None:
        df["schema_season"]    = schema_info.get("schema_detected", "")
        df["model_used_cls"]   = schema_info.get("model_used_cls", "")
        df["model_used_reg"]   = schema_info.get("model_used_reg", "")

    # ── Thêm cột forecast_for nếu có forecast_horizon ──
    if predictor.forecast_horizon > 0:
        from datetime import timedelta
        now = datetime.now()
        df["forecast_for"] = (now + timedelta(hours=predictor.forecast_horizon)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    # ── Lưu output ──
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / (input_path.stem + "_stacking_pred.csv")

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[CLI] Kết quả đã lưu → {output_path}")


if __name__ == "__main__":
    _cli_main()
