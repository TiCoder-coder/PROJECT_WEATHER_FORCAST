"""
TRANSFORMERS (Feature Engineering + Preprocess Pipeline)
=======================================================

Mục tiêu của file này:
- Tạo một "module transformer" dùng lại được cho mọi thuật toán (RandomForest / CatBoost / XGBoost / LightGBM...).
- Đảm bảo: TRAIN và PREDICT luôn đi qua CÙNG 1 luồng transform => không lệch schema, không lệch số cột.
- Hỗ trợ:
    (1) Xử lý missing values nâng cao (numeric/categorical/datetime)
    (2) Scaling (StandardScaler / MinMaxScaler) cho numeric nếu cần
    (3) Encoding categorical (OneHotEncoder, handle_unknown='ignore')
    (4) Datetime feature extraction (year/month/day/dow/hour/minute...) nếu có
    (5) Pipeline thống nhất (fit -> transform -> save/load)
    (6) “Schema freezing”: lưu lại danh sách output feature_names sau fit để predict luôn match

Lưu ý:
- Với CatBoost/LightGBM: đôi khi bạn muốn giữ categorical dạng category/string (native handling).
  Module này vẫn hỗ trợ "onehot" (an toàn cho XGBoost/RandomForest).
  Nếu bạn muốn native categorical -> bạn có thể đặt categorical_strategy="passthrough"
  và tự xử lý cat_features ở model wrapper.

Ví dụ nhanh:

    from Weather_Forcast_App.Machine_learning_model.features.Transformers import WeatherFeatureTransformer

    tfm = WeatherFeatureTransformer(
        numeric_scaler="standard",
        categorical_encoder="onehot",
        add_datetime_features=True
    )

    X_train_t = tfm.fit_transform(X_train)
    X_test_t  = tfm.transform(X_test)

    tfm.save(".../transformer.joblib")
    tfm2 = WeatherFeatureTransformer.load(".../transformer.joblib")
    X_new_t = tfm2.transform(X_new)

"""

from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer


# =========================================================
# Helper: đảm bảo input luôn là DataFrame
# =========================================================
def _to_dataframe(X: Union[pd.DataFrame, np.ndarray, Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
    """
    Convert nhiều kiểu input sang pandas DataFrame.

    Support:
    - pd.DataFrame: giữ nguyên
    - np.ndarray: tự tạo cột feature_0..feature_n
    - dict: coi như 1 sample
    - list[dict]: coi như nhiều sample
    """
    if isinstance(X, pd.DataFrame):
        return X.copy()

    if isinstance(X, np.ndarray):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

    if isinstance(X, dict):
        return pd.DataFrame([X])

    if isinstance(X, list) and len(X) > 0 and isinstance(X[0], dict):
        return pd.DataFrame(X)

    # fallback: cố gắng DataFrame(...)
    return pd.DataFrame(X)


# =========================================================
# Helper: detect datetime columns
# =========================================================
def _is_datetime_like(s: pd.Series) -> bool:
    """
    Heuristic: kiểm tra cột có phải datetime-like không.
    - Nếu dtype đã là datetime => True
    - Nếu object/string => thử parse 1 sample nhỏ
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return True

    if s.dtype == "object" or pd.api.types.is_string_dtype(s):
        sample = s.dropna().astype(str).head(30)
        if len(sample) == 0:
            return False
        parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
        # nếu parse được nhiều => coi là datetime-like
        ratio = parsed.notna().mean()
        return ratio >= 0.7

    return False


# =========================================================
# Dataclass lưu metadata/schema của transformer
# =========================================================
@dataclass
class TransformerSchema:
    # input columns lúc fit
    input_columns: List[str]

    # numeric/categorical/datetime columns được xác định
    numeric_cols: List[str]
    categorical_cols: List[str]
    datetime_cols: List[str]

    # datetime features đã tạo ra (mapping dt_col -> derived cols)
    datetime_feature_map: Dict[str, List[str]]

    # output feature names sau preprocess (sau onehot + numeric scaling)
    output_feature_names: List[str]

    # config options
    numeric_scaler: str
    categorical_encoder: str
    add_datetime_features: bool
    datetime_features: List[str]
    handle_unknown: str


# =========================================================
# Custom transformer: tạo feature từ datetime
# =========================================================
class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Transformer sinh feature từ datetime columns.

    - Input: DataFrame
    - Output: DataFrame với các cột datetime được thay bằng các cột derived (numeric)
    - Lưu mapping columns để predict luôn giống train.

    datetime_features default:
    - year, month, day, dow, hour, minute
    """

    def __init__(
        self,
        datetime_cols: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
        fit_parse: bool = True,
    ):
        self.datetime_cols = datetime_cols or []
        self.features = features or ["year", "month", "day", "dow", "hour", "minute"]
        self.fit_parse = fit_parse

        # mapping dt_col -> derived cols (được set ở fit)
        self.feature_map_: Dict[str, List[str]] = {}

    def fit(self, X: pd.DataFrame, y=None):
        df = _to_dataframe(X)

        # Nếu user chưa truyền datetime_cols => auto detect
        if not self.datetime_cols:
            self.datetime_cols = [c for c in df.columns if _is_datetime_like(df[c])]

        # Build feature_map_ (để schema cố định)
        self.feature_map_ = {}
        for col in self.datetime_cols:
            derived = []
            for f in self.features:
                derived.append(f"{col}__{f}")
            self.feature_map_[col] = derived

        return self

    def transform(self, X: pd.DataFrame):
        df = _to_dataframe(X)

        # Đảm bảo tồn tại các cột datetime (nếu thiếu -> tạo NaT để không crash)
        for col in self.datetime_cols:
            if col not in df.columns:
                df[col] = pd.NaT

        out = df.copy()

        for col in self.datetime_cols:
            # parse sang datetime (coerce lỗi -> NaT)
            if not pd.api.types.is_datetime64_any_dtype(out[col]):
                out[col] = pd.to_datetime(out[col], errors="coerce")

            # sinh features
            if "year" in self.features:
                out[f"{col}__year"] = out[col].dt.year
            if "month" in self.features:
                out[f"{col}__month"] = out[col].dt.month
            if "day" in self.features:
                out[f"{col}__day"] = out[col].dt.day
            if "dow" in self.features:
                out[f"{col}__dow"] = out[col].dt.dayofweek
            if "hour" in self.features:
                out[f"{col}__hour"] = out[col].dt.hour
            if "minute" in self.features:
                out[f"{col}__minute"] = out[col].dt.minute
            if "weekofyear" in self.features:
                # pandas mới: isocalendar().week
                out[f"{col}__weekofyear"] = out[col].dt.isocalendar().week.astype("float")
            if "dayofyear" in self.features:
                out[f"{col}__dayofyear"] = out[col].dt.dayofyear

            # drop cột gốc datetime
            out = out.drop(columns=[col], errors="ignore")

        return out


# =========================================================
# Main: WeatherFeatureTransformer
# =========================================================
class WeatherFeatureTransformer:
    """
    Transformer thống nhất cho TRAIN & PREDICT.

    Các tuỳ chọn quan trọng:
    - numeric_scaler:
        "none" | "standard" | "minmax"
    - categorical_encoder:
        "onehot" | "passthrough"
        (onehot an toàn nhất cho XGBoost/RandomForest)
    - handle_unknown:
        "ignore" (khuyên dùng) => category mới khi predict không crash
    - add_datetime_features:
        True/False => có tạo feature datetime không
    - datetime_features:
        list các feature datetime cần tạo

    Workflow chuẩn:
        fit(X_train) -> transform(X_train) -> save()
        load() -> transform(X_new) -> đưa vào model.predict(...)
    """

    def __init__(
        self,
        *,
        numeric_scaler: str = "standard",
        categorical_encoder: str = "onehot",
        handle_unknown: str = "ignore",
        add_datetime_features: bool = True,
        datetime_features: Optional[List[str]] = None,
        # missing strategies
        numeric_impute_strategy: str = "median",
        categorical_impute_strategy: str = "most_frequent",
    ):
        self.numeric_scaler = (numeric_scaler or "standard").lower()
        self.categorical_encoder = (categorical_encoder or "onehot").lower()
        self.handle_unknown = (handle_unknown or "ignore").lower()

        self.add_datetime_features = bool(add_datetime_features)
        self.datetime_features = datetime_features or ["year", "month", "day", "dow", "hour", "minute"]

        self.numeric_impute_strategy = numeric_impute_strategy
        self.categorical_impute_strategy = categorical_impute_strategy

        # các field sẽ được set sau fit
        self._is_fitted: bool = False
        self.schema_: Optional[TransformerSchema] = None

        self.datetime_extractor_: Optional[DateTimeFeatureExtractor] = None
        self.preprocessor_: Optional[ColumnTransformer] = None
        self.pipeline_: Optional[Pipeline] = None

        # feature names output sau fit
        self.output_feature_names_: List[str] = []

    # -----------------------------------------------------
    # 1) Auto-detect column types
    # -----------------------------------------------------
    def _infer_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
        """
        Tách cột numeric / categorical / datetime.

        Rules:
        - datetime: nếu add_datetime_features=True và cột datetime-like => datetime
        - numeric: int/float
        - categorical: object/bool/category/string
        """
        cols = list(df.columns)

        datetime_cols: List[str] = []
        if self.add_datetime_features:
            for c in cols:
                if _is_datetime_like(df[c]):
                    datetime_cols.append(c)

        # numeric: số (nhưng loại bỏ datetime gốc)
        numeric_cols = [
            c for c in cols
            if c not in datetime_cols and pd.api.types.is_numeric_dtype(df[c])
        ]

        # categorical: còn lại (trừ datetime + numeric)
        categorical_cols = [
            c for c in cols
            if c not in datetime_cols and c not in numeric_cols
        ]

        return numeric_cols, categorical_cols, datetime_cols

    # -----------------------------------------------------
    # 2) Build sklearn preprocessors
    # -----------------------------------------------------
    def _build_numeric_pipeline(self) -> Pipeline:
        """
        Numeric pipeline:
        - Imputer (median/mean/most_frequent/constant)
        - Optional scaler (standard/minmax/none)
        """
        steps = []
        steps.append(("imputer", SimpleImputer(strategy=self.numeric_impute_strategy)))

        if self.numeric_scaler == "standard":
            steps.append(("scaler", StandardScaler()))
        elif self.numeric_scaler == "minmax":
            steps.append(("scaler", MinMaxScaler()))
        elif self.numeric_scaler == "none":
            pass
        else:
            # fallback: standard
            steps.append(("scaler", StandardScaler()))

        return Pipeline(steps)

    def _build_categorical_pipeline(self) -> Pipeline:
        """
        Categorical pipeline:
        - Imputer (most_frequent/constant)
        - Encoder:
            - onehot: OneHotEncoder(handle_unknown='ignore')
            - passthrough: không encode (giữ nguyên)
        """
        steps = []
        steps.append(("imputer", SimpleImputer(strategy=self.categorical_impute_strategy)))

        if self.categorical_encoder == "onehot":
            steps.append(
                ("onehot", OneHotEncoder(handle_unknown=self.handle_unknown, sparse_output=False))
            )
        elif self.categorical_encoder == "passthrough":
            # Không encode => trả ra object, nhưng nhiều model không nhận.
            # Chỉ dùng khi model của bạn tự xử lý categorical.
            # (lưu ý: Pipeline + ColumnTransformer có thể trả dtype object)
            steps.append(("passthrough", "passthrough"))
        else:
            steps.append(
                ("onehot", OneHotEncoder(handle_unknown=self.handle_unknown, sparse_output=False))
            )

        return Pipeline(steps)

    # -----------------------------------------------------
    # 3) Fit / Transform
    # -----------------------------------------------------
    def fit(self, X: Union[pd.DataFrame, np.ndarray, Dict[str, Any], List[Dict[str, Any]]]) -> "WeatherFeatureTransformer":
        """
        Fit transformer:
        - Detect column types
        - Fit datetime extractor (nếu enable)
        - Build ColumnTransformer
        - Fit full pipeline
        - Freeze schema (output_feature_names)
        """
        df = _to_dataframe(X)

        # 1) infer column types
        numeric_cols, categorical_cols, datetime_cols = self._infer_column_types(df)

        # 2) datetime extractor
        datetime_feature_map: Dict[str, List[str]] = {}
        if self.add_datetime_features and len(datetime_cols) > 0:
            self.datetime_extractor_ = DateTimeFeatureExtractor(
                datetime_cols=datetime_cols,
                features=self.datetime_features,
            )
            self.datetime_extractor_.fit(df)
            datetime_feature_map = dict(self.datetime_extractor_.feature_map_)
            df = self.datetime_extractor_.transform(df)
        else:
            self.datetime_extractor_ = None

        # Sau khi transform datetime: columns thay đổi => infer lại numeric/cat cho chính xác
        numeric_cols2, categorical_cols2, _ = self._infer_column_types(df)
        # datetime cols đã drop nên list dt lúc này rỗng, OK.

        # 3) build ColumnTransformer
        numeric_pipe = self._build_numeric_pipeline()
        cat_pipe = self._build_categorical_pipeline()

        transformers = []
        if len(numeric_cols2) > 0:
            transformers.append(("num", numeric_pipe, numeric_cols2))
        if len(categorical_cols2) > 0:
            transformers.append(("cat", cat_pipe, categorical_cols2))

        # Nếu dataset rỗng cột => vẫn cho chạy (nhưng thường không hợp lý)
        self.preprocessor_ = ColumnTransformer(
            transformers=transformers,
            remainder="drop",  # drop cột không thuộc num/cat
            verbose_feature_names_out=False,
        )

        # 4) build pipeline (hiện tại pipeline chỉ có preprocessor,
        # nhưng giữ Pipeline để sau này muốn add step khác rất dễ)
        self.pipeline_ = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor_),
            ]
        )

        # 5) fit pipeline
        self.pipeline_.fit(df)

        # 6) freeze output feature names
        self.output_feature_names_ = self._get_output_feature_names(
            numeric_cols=numeric_cols2,
            categorical_cols=categorical_cols2,
        )

        # 7) Save schema
        self.schema_ = TransformerSchema(
            input_columns=list(_to_dataframe(X).columns),
            numeric_cols=list(numeric_cols2),
            categorical_cols=list(categorical_cols2),
            datetime_cols=list(datetime_cols),
            datetime_feature_map=datetime_feature_map,
            output_feature_names=list(self.output_feature_names_),
            numeric_scaler=self.numeric_scaler,
            categorical_encoder=self.categorical_encoder,
            add_datetime_features=self.add_datetime_features,
            datetime_features=list(self.datetime_features),
            handle_unknown=self.handle_unknown,
        )

        self._is_fitted = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray, Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
        """
        Transform X theo schema đã fit.
        Output trả về DataFrame với columns = output_feature_names_.

        IMPORTANT:
        - Nếu predict thiếu 1 cột input: ta auto add NaN để pipeline xử lý (imputer).
        - Nếu predict thừa cột: pipeline sẽ drop nếu không thuộc schema num/cat.
        """
        if not self._is_fitted or self.pipeline_ is None or self.schema_ is None:
            raise ValueError("Transformer chưa fit. Hãy gọi fit() trước.")

        df = _to_dataframe(X)

        # 1) Align input columns: add missing columns = NaN
        # (để tránh crash khi user predict thiếu cột)
        for col in self.schema_.input_columns:
            if col not in df.columns:
                df[col] = np.nan

        # drop extra columns? không bắt buộc, ColumnTransformer đã drop remainder,
        # nhưng ta có thể reorder theo input_columns để ổn định
        df = df[self.schema_.input_columns]

        # 2) Apply datetime extractor theo mapping train
        if self.add_datetime_features and self.datetime_extractor_ is not None:
            df = self.datetime_extractor_.transform(df)

        # 3) Pipeline transform => ndarray
        arr = self.pipeline_.transform(df)

        # 4) Convert về DataFrame với đúng feature names
        out = pd.DataFrame(arr, columns=self.schema_.output_feature_names)

        return out

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray, Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
        """Fit + transform shortcut."""
        self.fit(X)
        return self.transform(X)

    # -----------------------------------------------------
    # 4) Output feature names
    # -----------------------------------------------------
    def _get_output_feature_names(self, numeric_cols: List[str], categorical_cols: List[str]) -> List[str]:
        """
        Lấy danh sách output feature_names sau preprocess.
        - numeric: giữ nguyên tên cột numeric
        - categorical onehot: tạo tên cột dạng col=value
        """
        if self.preprocessor_ is None:
            return []

        try:
            # sklearn >= 1.0
            names = self.preprocessor_.get_feature_names_out()
            return list(names)
        except Exception:
            # fallback thủ công (an toàn nhưng không đầy đủ bằng sklearn)
            out_names: List[str] = []
            out_names.extend(list(numeric_cols))

            if self.categorical_encoder == "onehot" and len(categorical_cols) > 0:
                # cố gắng lấy categories từ onehot
                try:
                    cat_pipe: Pipeline = self.preprocessor_.named_transformers_["cat"]
                    ohe: OneHotEncoder = cat_pipe.named_steps["onehot"]
                    cats = ohe.categories_
                    for col, cat_list in zip(categorical_cols, cats):
                        for v in cat_list:
                            out_names.append(f"{col}={v}")
                except Exception:
                    # nếu không lấy được thì append chung chung
                    out_names.extend([f"{c}__onehot" for c in categorical_cols])
            else:
                # passthrough => giữ nguyên
                out_names.extend(list(categorical_cols))

            return out_names

    # -----------------------------------------------------
    # 5) Save / Load
    # -----------------------------------------------------
    def save(self, filepath: Union[str, Path]) -> str:
        """
        Lưu transformer ra joblib.
        Lưu:
        - pipeline_
        - datetime_extractor_
        - schema_
        """
        if not self._is_fitted or self.schema_ is None:
            raise ValueError("Transformer chưa fit, không thể save.")

        filepath = str(filepath)
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "pipeline": self.pipeline_,
            "datetime_extractor": self.datetime_extractor_,
            "schema": asdict(self.schema_),
        }
        joblib.dump(payload, filepath)
        return filepath

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "WeatherFeatureTransformer":
        """
        Load transformer từ joblib.
        """
        payload = joblib.load(str(filepath))

        schema_dict = payload.get("schema", {})
        obj = cls(
            numeric_scaler=schema_dict.get("numeric_scaler", "standard"),
            categorical_encoder=schema_dict.get("categorical_encoder", "onehot"),
            handle_unknown=schema_dict.get("handle_unknown", "ignore"),
            add_datetime_features=schema_dict.get("add_datetime_features", True),
            datetime_features=schema_dict.get("datetime_features", ["year", "month", "day", "dow", "hour", "minute"]),
        )

        obj.pipeline_ = payload.get("pipeline")
        obj.preprocessor_ = obj.pipeline_.named_steps["preprocessor"] if obj.pipeline_ is not None else None
        obj.datetime_extractor_ = payload.get("datetime_extractor")

        obj.schema_ = TransformerSchema(**schema_dict)
        obj.output_feature_names_ = list(obj.schema_.output_feature_names)

        obj._is_fitted = True
        return obj

    # -----------------------------------------------------
    # 6) Utilities
    # -----------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def get_schema(self) -> Dict[str, Any]:
        """
        Trả schema (dict) để debug/log.
        """
        if self.schema_ is None:
            return {}
        return asdict(self.schema_)


# =========================================================
# OPTIONAL: helper tạo path default lưu transformer
# =========================================================
def default_transformer_path(base_dir: Union[str, Path], name: str = "transformer") -> str:
    """
    Tạo path gợi ý:
        <base_dir>/<name>_YYYYMMDD_HHMMSS.joblib
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(base_dir / f"{name}_{ts}.joblib")
