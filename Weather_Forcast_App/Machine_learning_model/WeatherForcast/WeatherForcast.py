#!/usr/bin/env python3
# Dòng shebang:
# - Giúp file Python này có thể chạy trực tiếp như một script trên Linux/macOS
# - `/usr/bin/env python3` sẽ tìm đúng trình thông dịch python3 trong môi trường hiện tại
#"""Prediction runner that reuses artifacts from the latest training run."""
# Dòng trên đang là một chuỗi mô tả bị comment lại.
# Nghĩa của nó:
# - Đây là script dùng để chạy dự đoán (prediction)
# - Nó tái sử dụng các artifact (model đã train, pipeline transform, metadata, danh sách feature...)
#   từ lần train gần nhất

from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

ARTIFACT_DIR = (
	ROOT
	/ "Weather_Forcast_App"
	/ "Machine_learning_artifacts"
	/ "latest"
)

# Ensure root is importable for local packages
# Đảm bảo ROOT có trong sys.path để Python có thể import package local trong project.
# Nếu không thêm bước này, khi chạy script trực tiếp có thể bị lỗi ModuleNotFoundError.
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

# ── Django setup (bắt buộc trước khi import bất kỳ model dùng Django ORM) ──
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "WeatherForcast.settings")
import django
django.setup()

from Weather_Forcast_App.Machine_learning_model.features.Build_transfer import (
	WeatherFeatureBuilder,
)

# Định dạng log in ra màn hình.
# Ví dụ: [INFO] Loading model artifacts...
LOG_FORMAT = "[%(levelname)s] %(message)s"

# Cấu hình logging toàn cục cho script.
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

# Tạo logger riêng cho file này.
logger = logging.getLogger(__name__)


def _drop_known_bad_rows(df: pd.DataFrame) -> pd.DataFrame:
	# Kiểm tra file debug_top50_errors.csv (sinh bởi scripts/run_diagnostics.py)
	# Nếu tồn tại → lọc bỏ các dòng dữ liệu sai khỏi df trước khi forecast.
	# Nếu không tồn tại → trả về df nguyên vẹn.

	diagnostics_path = ROOT / "debug_top50_errors.csv"
	if not diagnostics_path.exists():
		return df

	try:
		bad_df = pd.read_csv(diagnostics_path)
	except Exception as e:
		logger.warning("Could not read %s: %s", diagnostics_path, e)
		return df

	# Bỏ các cột diagnostic được thêm bởi run_diagnostics.py
	for col in ("y_true", "y_pred", "abs_err"):
		if col in bad_df.columns:
			bad_df = bad_df.drop(columns=[col])

	common_cols = [c for c in bad_df.columns if c in df.columns]
	if not common_cols:
		return df

	# So khớp dòng theo fingerprint
	bad_keys = set(
		bad_df[common_cols].astype(str).apply("|".join, axis=1)
	)
	mask = df[common_cols].astype(str).apply("|".join, axis=1).isin(bad_keys)
	n_dropped = int(mask.sum())

	if n_dropped > 0:
		logger.warning(
			"Removed %d known bad rows from input data (source: debug_top50_errors.csv)",
			n_dropped,
		)

	return df[~mask].reset_index(drop=True)


def load_train_info(path: Path) -> Dict[str, Any]:
	# Hàm này dùng để đọc file Train_info.json
	# Mục đích:
	# - lấy target_column đã dùng lúc train
	# - lấy thông tin group_by hoặc metadata khác
	#
	# Tham số:
	# - path: đường dẫn tới file JSON
	#
	# Trả về:
	# - dictionary chứa nội dung JSON

	# Nếu file không tồn tại thì dừng ngay với lỗi rõ ràng.
	if not path.exists():
		raise FileNotFoundError(f"Train info not found at {path}")

	# Đọc toàn bộ text của file với mã hóa utf-8 rồi chuyển thành dict Python.
	return json.loads(path.read_text(encoding="utf-8"))


def _normalize_feature_names(value: Any) -> list[str] | None:
	# Hàm phụ để chuẩn hóa danh sách tên cột feature về cùng một kiểu:
	# - đầu ra luôn là list[str]
	# - nếu không chuẩn hóa được thì trả về None
	#
	# Tại sao cần?
	# Vì tên feature có thể được lưu dưới nhiều dạng khác nhau:
	# - numpy.ndarray
	# - list
	# - tuple
	# - hoặc có thể không tồn tại

	# Nếu value là None thì không có gì để chuẩn hóa
	if value is None:
		return None

	# Nếu feature name đang ở dạng numpy array thì đổi sang list Python
	if isinstance(value, np.ndarray):
		return [str(v) for v in value.tolist()]

	# Nếu đang là list hoặc tuple thì convert từng phần tử về string
	if isinstance(value, (list, tuple)):
		return [str(v) for v in value]

	# Nếu không thuộc các kiểu đã hỗ trợ thì trả None
	return None


def _get_feature_names_from_estimator(estimator: Any) -> list[str] | None:
	# Cố gắng lấy tên feature từ một estimator/model cụ thể.
	#
	# Vì mỗi loại model lưu metadata khác nhau:
	# - XGBoost booster có thể lưu trong booster.feature_names
	# - scikit-learn có thể dùng feature_names_in_
	# - một số transformer/model custom dùng các tên khác
	#
	# Hàm này thử nhiều cách để tối đa hóa khả năng khôi phục đúng danh sách feature.

	# Trường hợp estimator có method `get_booster` (thường là XGBoost)
	if hasattr(estimator, "get_booster"):
		try:
			booster = estimator.get_booster()
			# Lấy danh sách feature_names từ booster nếu có
			names = getattr(booster, "feature_names", None)
			normalized = _normalize_feature_names(names)
			if normalized:
				return normalized
		except Exception:  # pragma: no cover - best effort
			# Nếu có lỗi thì bỏ qua, tiếp tục thử cách khác
			pass

	# Duyệt qua nhiều tên thuộc tính phổ biến mà estimator có thể dùng để lưu feature names
	for attr in (
		"feature_names",
		"feature_name_",
		"feature_names_",
		"feature_names_in_",
		"feature_names_out_",
		"feature_name_out",
		"columns_",
	):
		value = getattr(estimator, attr, None)
		normalized = _normalize_feature_names(value)
		if normalized:
			return normalized

	# Nếu không lấy được tên feature từ estimator thì trả None
	return None


def _infer_model_feature_names(model: Any) -> list[str] | None:
	# Hàm này suy luận danh sách feature mà model thực sự mong đợi.
	#
	# Ý tưởng:
	# 1. Thử lấy trực tiếp từ chính model
	# 2. Nếu model là ensemble/custom wrapper và có base models
	#    thì thử lấy từ từng base model bên trong

	# Thử lấy trực tiếp từ model
	names = _get_feature_names_from_estimator(model)
	if names:
		return names

	# Nếu model có method `get_base_models`, nghĩa là có thể đây là model tổ hợp
	# hoặc wrapper custom do bạn tự xây.
	if hasattr(model, "get_base_models"):
		for base in model.get_base_models():
			# Một số base model có thể bọc model thật trong thuộc tính `.model`
			base_estimator = getattr(base, "model", base)
			names = _get_feature_names_from_estimator(base_estimator)
			if names:
				return names

	# Nếu vẫn không suy ra được thì trả None
	return None


def _ensure_pipeline(pipeline: Any) -> Any:
	# Hàm này đảm bảo object load ra từ Transform_pipeline.pkl
	# có thể dùng được như một pipeline với method `.transform(...)`.
	#
	# Vì file pipeline pickle có thể được lưu dưới nhiều dạng:
	# - một object pipeline chuẩn có transform
	# - một dict chứa list steps
	# - một dict chứa pipeline trong key nào đó
	#
	# Mục tiêu:
	# - cuối cùng phải trả về object có method `transform`

	# Nếu object đã có sẵn transform thì dùng luôn
	if hasattr(pipeline, "transform"):
		return pipeline

	# Nếu pipeline là dictionary thì cần dò xem cấu trúc nằm ở đâu
	if isinstance(pipeline, dict):
		# Trường hợp dict có key "steps" và steps là list
		# => tự tạo wrapper để transform tuần tự qua từng step
		if "steps" in pipeline and isinstance(pipeline["steps"], list):
			class _TransformWrapper:
				# Wrapper nội bộ này mô phỏng một pipeline đơn giản

				def __init__(self, steps):
					# Lưu danh sách các bước transform
					self.steps = steps

				def transform(self, X):
					# Chạy dữ liệu qua từng step theo đúng thứ tự
					Xt = X
					for step in self.steps:
						# Nếu step có transform thì dùng transform
						if hasattr(step, "transform"):
							Xt = step.transform(Xt)
						# Nếu không có transform nhưng có fit_transform
						# thì dùng fit_transform như một phương án dự phòng
						elif hasattr(step, "fit_transform"):
							Xt = step.fit_transform(Xt)
						else:
							# Nếu step không có cách nào để transform thì dừng với lỗi
							raise RuntimeError(f"Step {step!r} has no transform method")
					return Xt

			return _TransformWrapper(pipeline["steps"])

		# Thử tìm các key phổ biến có thể chứa pipeline thật
		for key in ("pipeline", "transformer", "pipe", "transform_pipeline"):
			candidate = pipeline.get(key)
			if hasattr(candidate, "transform"):
				return candidate

		# Nếu chưa thấy key chuẩn, duyệt mọi value trong dict
		# value nào có transform thì lấy luôn
		for value in pipeline.values():
			if hasattr(value, "transform"):
				return value

	# Nếu tới đây mà vẫn không trích được transformer hợp lệ
	raise RuntimeError("Unable to extract transformer from saved pipeline")


def _coerce_object_columns(X_t: pd.DataFrame) -> pd.DataFrame:
	# Hàm này dùng để xử lý các cột kiểu object sau khi transform.
	#
	# Tại sao cần?
	# - Model ML đa phần chỉ predict được trên dữ liệu số
	# - Sau transform đôi khi vẫn còn cột object do:
	#   + dữ liệu chuỗi
	#   + datetime dạng text
	#   + giá trị lẫn lộn kiểu dữ liệu
	#
	# Chiến lược xử lý:
	# 1. Thử ép object thành số
	# 2. Nếu không được thì thử parse datetime rồi đổi sang timestamp giây
	# 3. Nếu vẫn không được thì encode category thành mã số

	# Lấy danh sách các cột có dtype là object
	obj_cols = X_t.select_dtypes(include=["object"]).columns.tolist()

	# Nếu không có cột object nào thì trả về luôn, không cần xử lý
	if not obj_cols:
		return X_t

	# Tạo bản sao để tránh sửa trực tiếp DataFrame gốc đầu vào
	X_work = X_t.copy()

	# Xử lý lần lượt từng cột object
	for col in obj_cols:
		# Bước 1: thử ép về numeric
		coerced = pd.to_numeric(X_work[col], errors="coerce")
		# Nếu sau khi ép có ít nhất một giá trị không NaN
		# thì xem như cột này có thể dùng numeric được
		if coerced.notna().any():
			X_work[col] = coerced
			continue

		# Bước 2: nếu không ép số được thì thử parse datetime
		try:
			dt = pd.to_datetime(X_work[col], errors="coerce")
			if dt.notna().any():
				# Đổi datetime sang int64 nanoseconds rồi chia 10**9 để thành epoch seconds
				X_work[col] = dt.astype("int64") // 10**9
				continue
		except Exception:
			# Nếu parse datetime lỗi thì bỏ qua
			pass

		# Bước 3: nếu không phải số cũng không phải ngày giờ
		# thì chuyển sang category codes
		# Mỗi giá trị chuỗi khác nhau sẽ được map thành một số nguyên
		X_work[col] = X_work[col].astype("category").cat.codes

	return X_work


def _align_features(df: pd.DataFrame, expected: Iterable[str]) -> pd.DataFrame:
	# Hàm này căn chỉnh DataFrame feature để khớp chính xác với danh sách feature mong đợi.
	#
	# Mục tiêu:
	# - thêm các cột còn thiếu với giá trị mặc định 0
	# - loại bỏ các cột dư không nằm trong expected
	# - sắp xếp lại đúng thứ tự cột mà model/pipeline mong đợi
	#
	# Đây là bước rất quan trọng để tránh lỗi:
	# - số lượng cột không khớp
	# - tên cột không khớp
	# - thứ tự cột không khớp

	# Sao chép dữ liệu đầu vào để xử lý an toàn
	df_result = df.copy()

	# Với mỗi cột model/pipeline mong đợi:
	# nếu hiện tại chưa có thì thêm cột đó và gán 0
	for column in expected:
		if column not in df_result.columns:
			df_result[column] = 0

	# Tìm các cột đang có nhưng không nằm trong expected => xem là dư
	extra = [col for col in df_result.columns if col not in expected]

	# Nếu có cột dư thì loại bỏ
	if extra:
		df_result = df_result.drop(columns=extra)

	# Trả về DataFrame chỉ gồm các cột expected theo đúng thứ tự
	return df_result.loc[:, list(expected)]


class ForecastRunner:
	# Lớp chính điều phối toàn bộ quy trình forecast:
	# 1. Đọc metadata train
	# 2. Đọc dữ liệu đầu vào
	# 3. Load model + transform pipeline
	# 4. Build feature
	# 5. Transform dữ liệu
	# 6. Căn chỉnh feature cho đúng model
	# 7. Predict
	# 8. Ghi kết quả ra CSV

	def __init__(
		self,
		input_path: Path,
		output_path: Path,
	) -> None:
		# Hàm khởi tạo lớp ForecastRunner
		#
		# input_path:
		# - đường dẫn tới file CSV dữ liệu đầu vào cần dự đoán
		#
		# output_path:
		# - nơi lưu file dự đoán đầu ra

		self.input_path = input_path
		self.output_path = output_path

		# Đọc thông tin train từ artifact
		self.info = load_train_info(ARTIFACT_DIR / "Train_info.json")

		# Lấy tên cột target đã dùng khi train, nếu không có thì mặc định "rain_total"
		self.target_column = self.info.get("target_column", "rain_total")

		# Đọc feature config từ Feature_list.json để khởi tạo builder
		# đúng như lúc train (ví dụ: tắt lag/rolling nếu data là cross-sectional)
		try:
			_feat_meta = json.loads(
				(ARTIFACT_DIR / "Feature_list.json").read_text(encoding="utf-8")
			)
			_detected_type = _feat_meta.get("detected_data_type", "unknown")
			_feature_cfg: Dict[str, Any] | None = None
			if _detected_type == "cross_sectional":
				_feature_cfg = {
					"lag_features": False,
					"rolling_features": False,
					"difference_features": False,
				}
			self.feature_builder = WeatherFeatureBuilder(config=_feature_cfg)
		except Exception:
			# Nếu không đọc được Feature_list.json thì dùng default config
			self.feature_builder = WeatherFeatureBuilder()

		# Đọc cờ log1p: nếu train có apply log1p target thì khi predict
		# phải inverse (expm1) kết quả về scale gốc
		self.applied_log_target: bool = bool(
			self.info.get("target_transform", {}).get("log1p_applied", False)
		)

	def _load_data(self) -> pd.DataFrame:
		# Đọc toàn bộ file CSV đầu vào thành DataFrame
		# Sau đó lọc bỏ các dòng dữ liệu sai (nếu debug_top50_errors.csv tồn tại)

		# Kiểm tra file đầu vào có tồn tại không
		if not self.input_path.exists():
			raise FileNotFoundError(f"Input data not found: {self.input_path}")

		df = pd.read_csv(self.input_path)
		df = _drop_known_bad_rows(df)
		return df

	def _load_model_and_pipeline(self) -> tuple[Any, Any]:
		# Load model và transform pipeline từ thư mục artifact
		#
		# Model.pkl:
		# - mô hình đã train xong
		#
		# Transform_pipeline.pkl:
		# - pipeline tiền xử lý dữ liệu trước khi predict

		model_path = ARTIFACT_DIR / "Model.pkl"
		pipeline_path = ARTIFACT_DIR / "Transform_pipeline.pkl"

		# Nếu thiếu một trong hai file thì không thể forecast
		if not model_path.exists() or not pipeline_path.exists():
			raise FileNotFoundError("Model or pipeline pickle missing")

		# Load model (dùng joblib — Django đã được setup ở đầu file)
		model = joblib.load(model_path)

		# Load pipeline — dùng WeatherTransformPipeline.load() vì .save() lưu dict
		# (joblib.load trả về dict thô, không có .transform)
		from Weather_Forcast_App.Machine_learning_model.features.Transformers import (
			WeatherTransformPipeline,
		)
		pipeline = WeatherTransformPipeline.load(pipeline_path)
		return model, pipeline

	def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
		# Tạo feature engineering từ dữ liệu raw đầu vào
		#
		# Có thể dùng thêm thông tin group_by từ Train_info.json
		# để đảm bảo logic build feature đồng nhất với lúc train

		group_by = self.info.get("group_by")
		return self.feature_builder.build_all_features(
			df, target_column=self.target_column, group_by=group_by
		)

	def _prepare_prediction_data(self, df: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
		# Hàm gộp 2 bước:
		# 1. build feature từ raw data
		# 2. align các cột feature cho khớp danh sách expected

		df_feat = self._build_features(df)
		return _align_features(df_feat, expected)

	def _write_output(self, df: pd.DataFrame) -> None:
		# Ghi DataFrame kết quả ra file CSV
		#
		# Nếu thư mục cha chưa tồn tại thì sẽ tự tạo

		self.output_path.parent.mkdir(parents=True, exist_ok=True)
		# utf-8-sig: nhất quán với Cleardata.py để Excel mở không bị lỗi tiếng Việt
		df.to_csv(self.output_path, index=False, encoding="utf-8-sig")
		logger.info(f"Saved forecast to {self.output_path}")

	def run(self) -> None:
		# Hàm chạy chính của pipeline forecast.
		#
		# Trình tự tổng quát:
		# 1. Load model + pipeline
		# 2. Đọc danh sách feature expected từ Feature_list.json
		# 3. Suy luận danh sách feature model thật sự muốn nhận
		# 4. Đọc dữ liệu đầu vào
		# 5. Build + align feature
		# 6. Transform dữ liệu
		# 7. Chuyển dữ liệu thành DataFrame nếu cần
		# 8. Ép kiểu object -> numeric/datetime/category codes
		# 9. Thêm thiếu, bỏ dư, sắp thứ tự đúng theo model
		# 10. Predict
		# 11. Ghi ra CSV
		# 12. Nếu có y thật thì tính RMSE để tham khảo

		logger.info("Loading model artifacts...")

		# Load model và transform pipeline
		model, pipeline = self._load_model_and_pipeline()

		# Đọc danh sách toàn bộ feature columns từ file JSON
		expected_features = json.loads(
			(ARTIFACT_DIR / "Feature_list.json").read_text(encoding="utf-8")
		).get("all_feature_columns", [])

		# Cố suy luận feature names model thật sự mong đợi
		# Nếu không suy ra được thì fallback về expected_features
		model_feature_names = _infer_model_feature_names(model) or expected_features

		# Đọc dữ liệu input
		df_in = self._load_data()
		logger.info(f"Read {len(df_in)} rows from {self.input_path}")

		# Build feature và align với danh sách expected
		df_pred = self._prepare_prediction_data(df_in, expected_features)

		logger.info("Transforming features")
		# Chạy dữ liệu qua pipeline transform
		X_transformed = pipeline.transform(df_pred)

		# Nếu đầu ra của pipeline là ndarray hoặc object không có `.columns`
		# thì ta bọc lại thành DataFrame để dễ xử lý theo tên cột.
		if isinstance(X_transformed, np.ndarray) or not hasattr(X_transformed, "columns"):
			# Cố gắng tìm tên cột từ pipeline trước
			columns = getattr(pipeline, "feature_names", None) or getattr(pipeline, "feature_names_in_", None)
			columns = columns or model_feature_names or expected_features

			# Guard: số cột phải khớp với số chiều của ndarray
			_arr = np.array(X_transformed)
			if _arr.ndim == 2 and _arr.shape[1] != len(columns):
				logger.warning(
					"Column count mismatch after transform: array has %d cols but expected %d. "
					"Falling back to positional column names.",
					_arr.shape[1], len(columns),
				)
				columns = [f"f_{i}" for i in range(_arr.shape[1])]

			# Chuyển ndarray -> DataFrame
			X_transformed = pd.DataFrame(X_transformed, columns=columns)

		# Xử lý các cột object còn sót lại để model predict được
		X_transformed = _coerce_object_columns(X_transformed)

		# Tìm các cột model cần nhưng hiện chưa có trong X_transformed
		missing = [c for c in model_feature_names if c not in X_transformed.columns]
		if missing:
			logger.warning("Adding missing columns expected by model: %s", missing)
			# Thêm các cột thiếu với giá trị 0
			for col in missing:
				X_transformed[col] = 0

		# Tìm các cột đang có nhưng model không cần
		extra = [c for c in X_transformed.columns if c not in model_feature_names]
		if extra:
			# Loại bỏ cột dư
			X_transformed = X_transformed.drop(columns=extra, errors="ignore")

		# Sắp xếp lại cột đúng theo thứ tự model mong đợi
		X_transformed = X_transformed.loc[:, model_feature_names]

		logger.info("Predicting")
		# Thực hiện dự đoán
		preds = model.predict(X_transformed)

		# Một số model custom có thể trả object chứa thuộc tính `.predictions`
		# nên ở đây thử bóc ra nếu có
		try:
			preds = getattr(preds, "predictions", preds)
		except Exception:
			pass

		# Đảm bảo preds thành numpy array 1 chiều
		preds = np.array(preds).reshape(-1)

		# [BUG FIX] Inverse log1p nếu lúc train có apply log1p target
		# Model học trên log1p(y) nên output cũng ở log-scale → phải expm1 về scale gốc
		if self.applied_log_target:
			logger.info("Applying expm1 inverse transform (log1p was used during training)")
			preds = np.expm1(preds).clip(min=0)

		# Tạo output bằng cách copy input gốc rồi thêm cột dự đoán
		df_out = df_in.copy()
		df_out["y_pred"] = preds

		# Nếu trong input có sẵn cột target thật
		# thì tính RMSE để xem dự đoán trên batch này tốt tới đâu
		if self.target_column in df_out:
			y_true = df_out[self.target_column].astype(float).values
			rmse = float(np.sqrt(((y_true - preds) ** 2).mean()))
			logger.info("Prediction RMSE on provided target: %.4f", rmse)

		# Ghi kết quả ra file CSV
		self._write_output(df_out)


def parse_args() -> argparse.Namespace:
	# Hàm phân tích tham số dòng lệnh.
	#
	# Các tham số hỗ trợ:
	# -i / --input   : file CSV đầu vào (bắt buộc)
	# -o / --output  : file CSV đầu ra, nếu không truyền sẽ dùng forecast_results.csv ở ROOT
	# --nrows        : giới hạn số dòng đọc vào, mặc định 0 = đọc tất cả

	parser = argparse.ArgumentParser(
		description="Run weather forecast with pretrained model artifacts"
	)

	# Tham số input bắt buộc
	parser.add_argument(
		"-i",
		"--input",
		type=Path,
		required=True,
		help="Path to raw weather data file (CSV)",
	)

	# Tham số output không bắt buộc
	parser.add_argument(
		"-o",
		"--output",
		type=Path,
		default=ROOT / "data" / "data_forecast" / "forecast_results.csv",
		help="Where to save predictions",
	)

	# Trả về namespace chứa toàn bộ argument đã parse
	return parser.parse_args()


def main() -> None:
	# Hàm entry point chính của script khi chạy từ terminal.
	#
	# Nhiệm vụ:
	# 1. đọc argument
	# 2. tạo ForecastRunner
	# 3. chạy
	# 4. log thời gian hoàn thành

	args = parse_args()

	# Tạo runner
	runner = ForecastRunner(args.input, args.output)

	# Ghi nhận thời gian bắt đầu
	start = time.time()

	# Chạy dự đoán
	runner.run()

	# Tính thời gian đã chạy
	elapsed = time.time() - start
	logger.info("Forecast completed in %.1fs", elapsed)


if __name__ == "__main__":
	# Khối này chỉ chạy khi file được gọi trực tiếp, ví dụ:
	# python predict_runner.py -i input.csv
	#
	# Nếu file được import như module thì khối này sẽ không chạy
	try:
		main()
	except Exception:
		# Nếu có lỗi bất kỳ trong quá trình chạy:
		# - ghi full traceback ra log
		# - thoát chương trình với mã lỗi 1
		logger.exception("Forecast run failed")
		sys.exit(1)