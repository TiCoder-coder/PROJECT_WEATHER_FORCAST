#!/usr/bin/env python3
"""
=============================================================================
Script: run_diagnostics.py
Mục đích: Chạy đánh giá chẩn đoán (diagnostics) sau khi huấn luyện mô hình.
          Bao gồm: tính RMSE baseline, tìm 50 mẫu có sai số lớn nhất,
          và tính RMSE theo từng trạm thời tiết (per-station RMSE).

Cách dùng:
    python3 scripts/run_diagnostics.py

Kết quả sinh ra:
    - debug_top50_errors.csv      : 50 dòng dữ liệu mà mô hình dự đoán sai nhất
    - debug_worst_stations.csv    : RMSE theo từng trạm, sắp xếp từ tệ nhất đến tốt nhất
    - In tóm tắt (summary) ra màn hình stdout
=============================================================================
"""

# ---------------------------------------------------------------------------
# PHẦN 1: IMPORT CÁC THƯ VIỆN CẦN THIẾT
# ---------------------------------------------------------------------------

import json        # Đọc/ghi file JSON (dùng để đọc Train_info.json, Metrics.json, Feature_list.json)
import joblib      # Tải/lưu model và pipeline đã được serialize (lưu dưới dạng .pkl)
import pathlib     # Làm việc với đường dẫn file/thư mục theo cách hướng đối tượng (OOP)
import sys         # Tương tác với hệ thống: thoát chương trình (sys.exit), quản lý sys.path
import traceback   # In chi tiết lỗi khi xảy ra exception (stack trace đầy đủ)
from typing import Any  # Khai báo kiểu dữ liệu linh hoạt (bất kỳ loại nào cũng được)

import numpy as np    # Thư viện tính toán số học với mảng đa chiều, rất hiệu quả
import pandas as pd   # Thư viện xử lý dữ liệu dạng bảng (DataFrame), đọc CSV, ...
import time           # Đo thời gian thực thi (dùng time.time() để bấm giờ)
import os             # Tương tác với hệ điều hành: đọc biến môi trường (os.environ)


# ---------------------------------------------------------------------------
# PHẦN 2: KHAI BÁO CÁC HẰNG SỐ ĐƯỜNG DẪN TOÀN CỤC
# ---------------------------------------------------------------------------

# pathlib.Path(__file__) : lấy đường dẫn tuyệt đối của file script này
# .resolve()             : chuyển thành đường dẫn tuyệt đối (giải quyết symlink nếu có)
# .parents[1]            : lấy thư mục cha cấp 2 (scripts/ -> PROJECT_ROOT/)
# Ví dụ: nếu file này là /project/scripts/run_diagnostics.py
#         thì ROOT = /project
ROOT = pathlib.Path(__file__).resolve().parents[1]

# Xây dựng đường dẫn đến thư mục chứa các artifact (kết quả) của model đã train
# Trong thư mục này sẽ có: Model.pkl, Transform_pipeline.pkl,
#                           Train_info.json, Metrics.json, Feature_list.json
ART = ROOT / "Weather_Forcast_App" / "Machine_learning_artifacts" / "latest"

# ---------------------------------------------------------------------------
# Đảm bảo thư mục gốc của project nằm trong sys.path
# Lý do: khi joblib.load() giải nén file .pkl, Python cần import được các
#        module nội bộ của project (ví dụ: WeatherTransformPipeline).
#        Nếu ROOT không có trong sys.path thì sẽ bị lỗi ModuleNotFoundError.
# ---------------------------------------------------------------------------
import sys as _sys               # Import lại sys với alias _sys để tránh nhầm lẫn
if str(ROOT) not in _sys.path:   # Kiểm tra nếu ROOT chưa có trong danh sách đường dẫn tìm kiếm
    _sys.path.insert(0, str(ROOT))  # Thêm ROOT vào đầu danh sách (ưu tiên cao nhất)


# ---------------------------------------------------------------------------
# PHẦN 3: CÁC HÀM TIỆN ÍCH (UTILITY FUNCTIONS)
# ---------------------------------------------------------------------------

def load_info() -> Any:
    """
    Tải và trả về nội dung file Train_info.json dưới dạng dict Python.

    File Train_info.json chứa thông tin về quá trình huấn luyện model, bao gồm:
        - split_saved_paths: đường dẫn đến file train/test sau khi chia dữ liệu
        - target_column: tên cột mục tiêu cần dự đoán (ví dụ: "rain_total")
        - group_by: cột dùng để nhóm dữ liệu khi xây dựng features (ví dụ: "station_id")
        - ... và nhiều thông tin khác

    Returns:
        Any: dict chứa toàn bộ thông tin từ Train_info.json
    """
    # (ART / "Train_info.json")  : tạo đường dẫn đến file Train_info.json
    # .read_text(encoding="utf-8"): đọc nội dung file dưới dạng chuỗi văn bản UTF-8
    # json.loads(...)             : chuyển chuỗi JSON thành dict Python
    return json.loads((ART / "Train_info.json").read_text(encoding="utf-8"))


def _normalize_feature_names(value: Any) -> list[str] | None:
    """
    Chuẩn hóa danh sách tên feature về dạng list[str].

    Vấn đề: Tên feature có thể được lưu dưới nhiều dạng khác nhau tùy thư viện
    (numpy array, list Python, tuple Python...). Hàm này đưa tất cả về list[str]
    đồng nhất để dễ so sánh và xử lý.

    Args:
        value (Any): Giá trị đầu vào, có thể là None, np.ndarray, list, hoặc tuple

    Returns:
        list[str] | None:
            - list[str] nếu chuyển đổi thành công
            - None nếu value là None hoặc kiểu không hỗ trợ
    """
    # Trường hợp 1: value là None -> không có tên feature -> trả về None
    if value is None:
        return None

    # Trường hợp 2: value là numpy array (ví dụ: np.array(["temp", "humidity"]))
    # .tolist() chuyển numpy array sang list Python thông thường
    # str(v) đảm bảo mỗi phần tử là chuỗi (tránh numpy.str_ gây lỗi)
    if isinstance(value, np.ndarray):
        return [str(v) for v in value.tolist()]

    # Trường hợp 3: value là list hoặc tuple Python thông thường
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]

    # Trường hợp 4: kiểu không được hỗ trợ (số nguyên, dict, ...) -> trả về None
    return None


def _get_feature_names_from_estimator(estimator: Any) -> list[str] | None:
    """
    Truy xuất danh sách tên feature trực tiếp từ đối tượng estimator (model/transformer).

    Chiến lược: Thử nhiều cách khác nhau để lấy tên feature vì mỗi thư viện ML
    lưu tên feature vào thuộc tính khác nhau:
        - XGBoost  : booster.feature_names
        - LightGBM : feature_name_
        - Sklearn  : feature_names_in_
        - Custom   : feature_names, columns_, ...

    Args:
        estimator (Any): Đối tượng model hoặc transformer cần kiểm tra

    Returns:
        list[str] | None: Danh sách tên feature nếu tìm thấy, ngược lại None
    """
    # --- Cách 1: Dành riêng cho XGBoost ---
    # XGBoost lưu booster bên trong estimator, booster có thuộc tính feature_names
    if hasattr(estimator, "get_booster"):  # Kiểm tra xem estimator có phải XGBoost không
        try:
            booster = estimator.get_booster()                    # Lấy đối tượng Booster từ XGBModel
            names = getattr(booster, "feature_names", None)      # Đọc thuộc tính feature_names của booster
            normalized = _normalize_feature_names(names)         # Chuẩn hóa về list[str]
            if normalized:                                        # Nếu lấy được tên feature thành công
                return normalized                                 # Trả về ngay, không cần thử tiếp
        except Exception:
            pass  # Nếu có lỗi (ví dụ: booster chưa được fit) thì bỏ qua, thử cách khác

    # --- Cách 2: Duyệt qua danh sách các thuộc tính phổ biến ---
    # Mỗi thư viện ML dùng tên thuộc tính khác nhau để lưu tên feature:
    #   "feature_names"     : Custom models
    #   "feature_name_"     : LightGBM
    #   "feature_names_"    : Một số phiên bản cũ của sklearn
    #   "feature_names_in_" : sklearn >= 1.0 (sau khi fit với DataFrame)
    #   "feature_names_out_": sklearn transformers (sau khi transform)
    #   "feature_name_out"  : Một số custom transformers
    #   "columns_"          : Một số pandas-based transformers
    for attr in (
        "feature_names",
        "feature_name_",
        "feature_names_",
        "feature_names_in_",
        "feature_names_out_",
        "feature_name_out",
        "columns_",
    ):
        value = getattr(estimator, attr, None)  # Đọc thuộc tính, trả về None nếu không tồn tại
        normalized = _normalize_feature_names(value)  # Chuẩn hóa về list[str]
        if normalized:           # Nếu tìm được (không None và không rỗng)
            return normalized    # Trả về kết quả

    # Không tìm thấy tên feature qua bất kỳ cách nào
    return None


def _infer_model_feature_names(model: Any) -> list[str] | None:
    """
    Suy luận danh sách tên feature mà model sử dụng khi training.

    Khác với _get_feature_names_from_estimator, hàm này xử lý thêm trường hợp
    model là một Ensemble (tập hợp nhiều model con bên trong), ví dụ:
    Stacking, Voting, hay các custom Ensemble wrapper.

    Chiến lược:
        1. Thử lấy tên feature trực tiếp từ model chính
        2. Nếu không được, thử lấy từ các model con bên trong (base models)

    Args:
        model (Any): Đối tượng model ML (có thể là single model hoặc ensemble)

    Returns:
        list[str] | None: Danh sách tên feature hoặc None nếu không tìm được
    """
    # Bước 1: Thử lấy tên feature trực tiếp từ model
    names = _get_feature_names_from_estimator(model)
    if names:      # Nếu tìm thấy
        return names   # Trả về ngay

    # Bước 2: Nếu model là Ensemble, duyệt qua các model con
    # get_base_models() là method đặc trưng của custom Ensemble wrapper trong project này
    if hasattr(model, "get_base_models"):
        for base in model.get_base_models():  # Lặp qua từng model con trong ensemble
            # Mỗi base model có thể bọc model thực sự trong thuộc tính .model
            # getattr(base, "model", base) : lấy base.model nếu có, nếu không thì dùng chính base
            base_estimator = getattr(base, "model", base)
            names = _get_feature_names_from_estimator(base_estimator)
            if names:         # Nếu model con này có tên feature
                return names  # Trả về ngay (giả sử tất cả base models dùng cùng features)

    # Không tìm được tên feature từ model lẫn các model con
    return None


# ---------------------------------------------------------------------------
# PHẦN 4: HÀM CHÍNH main()
# ---------------------------------------------------------------------------

def main():
    """
    Hàm điều phối chính của script, thực hiện tuần tự:
        1. Đọc thông tin training từ Train_info.json
        2. Tải dữ liệu test từ file CSV
        3. Tính RMSE baseline (dùng giá trị trung bình làm dự đoán)
        4. Tải model và transform pipeline từ file .pkl
        5. Xử lý trường hợp pipeline được lưu dưới dạng dict
        6. Xây dựng features cho tập test
        7. Transform features và dự đoán bằng model
        8. Lưu 50 mẫu sai số lớn nhất ra CSV
        9. Tính và lưu RMSE theo từng trạm ra CSV
    """

    # -----------------------------------------------------------------------
    # BƯỚC 1: ĐỌC THÔNG TIN TRAINING
    # -----------------------------------------------------------------------
    try:
        info = load_info()  # Gọi hàm load_info() đã định nghĩa ở trên
    except Exception as e:
        # Nếu không đọc được file (file không tồn tại, JSON lỗi, ...) -> dừng chương trình
        print("[ERROR] Cannot read Train_info.json:", e)
        sys.exit(1)  # Thoát với exit code 1 (báo hiệu lỗi)

    # Lấy đường dẫn file test từ thông tin training
    # info["split_saved_paths"]["test"] là đường dẫn đến file CSV test
    # Dùng .get() thay vì [] để tránh KeyError nếu key không tồn tại
    test_path = info.get("split_saved_paths", {}).get("test")
    if not test_path:
        # Nếu không có đường dẫn file test thì không thể tiếp tục
        print("[ERROR] test path not found in Train_info.json")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # BƯỚC 2: TẢI DỮ LIỆU TEST
    # -----------------------------------------------------------------------
    print(f"[INFO] Loading test file: {test_path}")

    # Hỗ trợ chạy nhanh hơn bằng cách chỉ đọc một phần dữ liệu
    # Biến môi trường DIAG_NROWS cho phép giới hạn số dòng đọc vào
    # Ví dụ: DIAG_NROWS=1000 python3 scripts/run_diagnostics.py
    # os.environ.get("DIAG_NROWS", "0") : đọc biến môi trường, mặc định là "0" (đọc tất cả)
    nrows = int(os.environ.get("DIAG_NROWS", "0"))  # Chuyển sang số nguyên
    if nrows and nrows > 0:
        # Nếu DIAG_NROWS được đặt và > 0 -> chỉ đọc nrows dòng đầu tiên
        print(f"[INFO] Reading only first {nrows} rows from test for faster run (DIAG_NROWS={nrows})")
        df = pd.read_csv(test_path, nrows=nrows)  # nrows: giới hạn số dòng đọc
    else:
        # Đọc toàn bộ file CSV không giới hạn
        df = pd.read_csv(test_path)

    # Lấy tên cột mục tiêu (cần dự đoán)
    # Mặc định là "rain_total" (tổng lượng mưa), nếu Train_info.json không ghi thì dùng mặc định
    target_col = info.get("target_column", "rain_total")

    # Kiểm tra xem cột mục tiêu có tồn tại trong DataFrame hay không
    if target_col not in df.columns:
        print(f"[ERROR] target column '{target_col}' not present in test dataframe columns")
        print("Columns:", df.columns.tolist())  # In danh sách cột hiện có để debug
        sys.exit(1)

    # -----------------------------------------------------------------------
    # BƯỚC 3: TÍNH RMSE BASELINE
    # -----------------------------------------------------------------------
    # RMSE Baseline là thước đo "ngu nhất": nếu ta chỉ đoán mọi giá trị đều bằng
    # giá trị trung bình của tập test, RMSE sẽ là bao nhiêu?
    # Đây là ngưỡng tối thiểu - model tốt phải có RMSE nhỏ hơn baseline này.

    y = df[target_col].values  # Lấy giá trị thực tế của cột mục tiêu dưới dạng numpy array

    # Tính giá trị trung bình (mean) - đây là dự đoán "ngây thơ" nhất
    baseline = np.mean(y)

    # Công thức RMSE: sqrt( mean( (y_true - y_pred)^2 ) )
    # Ở đây y_pred = baseline (một hằng số), nên:
    # RMSE_baseline = sqrt( mean( (y - mean(y))^2 ) ) = độ lệch chuẩn (std) của y
    baseline_rmse = float(np.sqrt(((y - baseline) ** 2).mean()))

    # Đọc RMSE thực tế của model từ file Metrics.json
    # File này được sinh ra sau khi train, chứa các metric đánh giá trên tập test
    metrics = json.loads((ART / "Metrics.json").read_text(encoding="utf-8"))
    model_rmse = metrics.get("test", {}).get("RMSE")  # Lấy RMSE trên tập test

    # In tóm tắt để so sánh model với baseline
    print("[SUMMARY]")
    print(f" Baseline(mean) RMSE: {baseline_rmse:.4f}")  # In với 4 chữ số thập phân
    print(f" Model test RMSE:      {model_rmse}")

    # -----------------------------------------------------------------------
    # BƯỚC 4: TẢI MODEL VÀ TRANSFORM PIPELINE
    # -----------------------------------------------------------------------
    print("[INFO] Loading model and pipeline...")

    # Tải model đã train từ file Model.pkl
    # joblib.load() giải nén đối tượng Python đã được serialize (pickle)
    model = joblib.load(ART / "Model.pkl")

    # Tải transform pipeline (các bước tiền xử lý dữ liệu trước khi đưa vào model)
    pipeline = None
    try:
        # Thử tải trực tiếp bằng joblib (cách phổ biến nhất)
        pipeline = joblib.load(ART / "Transform_pipeline.pkl")
    except Exception:
        # Nếu joblib.load() thất bại (ví dụ: pipeline được lưu bằng class method tùy chỉnh)
        # -> thử dùng class loader dự phòng (fallback)
        from Weather_Forcast_App.Machine_learning_model.features.Transformers import WeatherTransformPipeline
        pipeline = WeatherTransformPipeline.load(ART / "Transform_pipeline.pkl")

    # -----------------------------------------------------------------------
    # BƯỚC 5: XỬ LÝ TRƯỜNG HỢP PIPELINE LÀ DICT
    # -----------------------------------------------------------------------
    # Đôi khi pipeline được lưu dưới dạng dict chứa metadata thay vì đối tượng trực tiếp.
    # Chúng ta cần trích xuất đối tượng có method .transform() từ dict đó.
    if isinstance(pipeline, dict):
        print("[DEBUG] Loaded pipeline is a dict; attempting to extract transform object from keys...")

        # --- Trường hợp 5a: Dict có key "steps" chứa danh sách các transformer ---
        # Đây là format custom của project: {"steps": [transformer1, transformer2, ...]}
        if "steps" in pipeline and isinstance(pipeline["steps"], list):
            steps = pipeline["steps"]  # Lấy danh sách các bước transformer

            # Tạo class wrapper nội bộ để giả lập pipeline chuẩn với method .transform()
            class _TransformWrapper:
                """
                Wrapper nội bộ: gói danh sách các transformer thành một pipeline
                có interface chuẩn với method transform().
                """
                def __init__(self, steps):
                    # Lưu danh sách các bước transformer
                    self.steps = steps

                def transform(self, X):
                    """
                    Áp dụng tuần tự từng transformer lên dữ liệu X.
                    Đầu ra của bước trước là đầu vào của bước sau (chain).
                    """
                    Xt = X       # Khởi tạo dữ liệu đầu vào (Xt = X_transformed)
                    for s in self.steps:   # Duyệt qua từng bước transformer theo thứ tự
                        if hasattr(s, "transform"):
                            # Cách chuẩn: gọi .transform() để biến đổi dữ liệu
                            Xt = s.transform(Xt)
                        elif hasattr(s, "fit_transform"):
                            # Fallback: dùng fit_transform nếu không có transform
                            # Lưu ý: fit_transform sẽ fit lại trên dữ liệu test - không lý tưởng
                            # nhưng dùng khi không có lựa chọn khác
                            Xt = s.fit_transform(Xt)
                        else:
                            # Nếu transformer không có cả transform lẫn fit_transform -> lỗi
                            raise RuntimeError(f"Step {s} has no transform method")
                    return Xt  # Trả về dữ liệu đã qua tất cả các bước biến đổi

            # Tạo instance của wrapper và gán vào biến pipeline
            pipeline = _TransformWrapper(steps)
            print("[DEBUG] Wrapped 'steps' list into a transformable pipeline wrapper")

        else:
            # --- Trường hợp 5b: Dict không có key "steps" ---
            # Thử tìm đối tượng có .transform() theo các key phổ biến

            # Duyệt qua 4 key thường gặp theo thứ tự ưu tiên
            for key in ("pipeline", "transformer", "pipe", "transform_pipeline"):
                if key in pipeline and hasattr(pipeline[key], "transform"):
                    # Tìm thấy đối tượng có .transform() tại key này
                    pipeline = pipeline[key]          # Trích xuất đối tượng
                    print(f"[DEBUG] Extracted pipeline from key: {key}")
                    break   # Dừng vòng lặp, đã tìm thấy rồi
            else:
                # Mệnh đề else của for: chạy khi vòng for kết thúc KHÔNG qua break
                # Tức là không tìm thấy pipeline ở bất kỳ key nào ở trên

                # Thử tìm kiếm toàn bộ các value trong dict
                found = False
                for k, v in pipeline.items():   # Duyệt qua tất cả cặp key-value
                    if hasattr(v, "transform"):  # Kiểm tra xem value có method .transform() không
                        pipeline = v             # Nếu có, dùng value đó làm pipeline
                        print(f"[DEBUG] Extracted pipeline from dict value key: {k}")
                        found = True
                        break   # Thoát vòng lặp ngay khi tìm thấy cái đầu tiên

                if not found:
                    # Đã thử tất cả cách nhưng vẫn không tìm được -> báo lỗi và dừng
                    print("[ERROR] Could not find transformable object inside Transform_pipeline.pkl dict.\n" \
                          "Please inspect the pickle contents or re-save the transform pipeline as a single object.")
                    sys.exit(1)

    # -----------------------------------------------------------------------
    # BƯỚC 6: XÂY DỰNG FEATURES CHO TẬP TEST
    # -----------------------------------------------------------------------
    # WeatherFeatureBuilder xây dựng tất cả các feature từ dữ liệu thô:
    # ví dụ: lag features (giá trị quá khứ), rolling statistics, time-based features, ...
    from Weather_Forcast_App.Machine_learning_model.features.Build_transfer import WeatherFeatureBuilder

    builder = WeatherFeatureBuilder()    # Khởi tạo đối tượng feature builder
    print("[INFO] Building features for test set (this may take a while)...")

    t0 = time.time()   # Ghi thời điểm bắt đầu (timestamp dạng float, đơn vị giây)

    # build_all_features: hàm chính để tạo toàn bộ features
    #   df           : DataFrame raw (dữ liệu thô)
    #   target_column: tên cột mục tiêu (để builder biết cột nào là label, không tạo lag cho nó)
    #   group_by     : cột nhóm (ví dụ "station_id") để tính lag/rolling theo từng trạm riêng
    df_feat = builder.build_all_features(df, target_column=target_col, group_by=info.get("group_by"))

    t1 = time.time()   # Ghi thời điểm kết thúc
    print(f"[INFO] Feature building done in {t1-t0:.1f}s")   # In thời gian thực thi (1 chữ số thập phân)

    # -----------------------------------------------------------------------
    # BƯỚC 7: ĐẢM BẢO ĐỦ CÁC CỘT FEATURE THEO DANH SÁCH KHI TRAINING
    # -----------------------------------------------------------------------
    # Đọc danh sách feature đã dùng khi training từ Feature_list.json
    feat = json.loads((ART / "Feature_list.json").read_text(encoding="utf-8"))
    # "all_feature_columns" là key chứa list tên tất cả feature
    expected = feat.get("all_feature_columns", [])

    # Đảm bảo tất cả feature cần thiết đều có trong df_feat
    # Nếu thiếu feature nào (do dữ liệu test không đủ) -> điền giá trị 0
    # Ví dụ: feature "lag_7_rain" có thể bị thiếu nếu dữ liệu test ngắn hơn 7 ngày
    for c in expected:
        if c not in df_feat.columns:
            df_feat[c] = 0  # Điền 0 cho cột bị thiếu (imputation)

    # Chọn đúng các cột feature theo thứ tự đã dùng lúc training
    # Thứ tự cột PHẢI khớp với lúc training, nếu không model có thể cho kết quả sai
    X = df_feat[expected]
    print(f"[INFO] Transforming features (n_rows={len(X)}, n_cols={len(X.columns)})")

    # -----------------------------------------------------------------------
    # BƯỚC 8: TRANSFORM FEATURES (TIỀN XỬ LÝ)
    # -----------------------------------------------------------------------
    t0 = time.time()
    # pipeline.transform(X): áp dụng các bước tiền xử lý như scaling, encoding, ...
    # X_t là kết quả sau khi transform (X_transformed)
    X_t = pipeline.transform(X)
    t1 = time.time()
    print(f"[INFO] Transform done in {t1-t0:.1f}s")

    # -----------------------------------------------------------------------
    # BƯỚC 9: DỰ ĐOÁN (PREDICTION)
    # -----------------------------------------------------------------------
    print("[INFO] Predicting...")

    # XGBoost và một số thư viện yêu cầu tất cả cột phải là kiểu số (numeric/bool/category)
    # Nếu có cột kiểu object (tức là string), cần chuyển đổi trước khi predict
    obj_cols = X_t.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        print(f"[DEBUG] Converting object columns before predict: {obj_cols}")
        for c in obj_cols:
            # Chiến lược chuyển đổi 3 bước theo thứ tự ưu tiên:

            # Cách 1: Thử chuyển sang số (numeric)
            # errors="coerce": nếu không chuyển được -> trả về NaN thay vì lỗi
            coerced = pd.to_numeric(X_t[c], errors="coerce")
            if coerced.notna().any():   # Nếu có ít nhất 1 giá trị chuyển thành công
                X_t[c] = coerced        # Dùng phiên bản đã chuyển
                continue                # Chuyển sang cột tiếp theo

            # Cách 2: Thử parse datetime rồi chuyển sang Unix timestamp (giây)
            # Hữu ích khi cột là chuỗi ngày tháng như "2024-01-15"
            try:
                dt = pd.to_datetime(X_t[c], errors="coerce")  # Parse sang datetime
                if dt.notna().any():    # Nếu parse được ít nhất 1 giá trị
                    # .astype('int64'): chuyển datetime sang nanoseconds từ epoch
                    # // 10**9: chia cho 1 tỷ để đổi từ nanoseconds sang seconds
                    X_t[c] = dt.astype('int64') // 10**9
                    continue
            except Exception:
                pass    # Nếu lỗi trong quá trình parse datetime -> bỏ qua, thử cách 3

            # Cách 3 (fallback): Mã hóa danh mục (categorical encoding)
            # Chuyển chuỗi thành số nguyên: "station_A"->0, "station_B"->1, ...
            X_t[c] = X_t[c].astype('category').cat.codes

    # -----------------------------------------------------------------------
    # BƯỚC 10: ĐẢM BẢO X_t CÓ ĐÚNG CÁC FEATURE MÀ MODEL MONG ĐỢI
    # -----------------------------------------------------------------------
    # Lấy danh sách feature mà model sử dụng khi training
    model_feature_names = _infer_model_feature_names(model)
    if model_feature_names is None:
        # Không lấy được từ model -> dùng danh sách expected từ Feature_list.json làm fallback
        model_feature_names = expected

    # Xử lý trường hợp X_t là numpy array (không có tên cột)
    # Điều kiện: X_t là list/tuple, hoặc là mảng 2D (ndim==2) nhưng không phải DataFrame (không có .columns)
    if isinstance(X_t, (list, tuple)) or (hasattr(X_t, "ndim") and getattr(X_t, "ndim") == 2 and not hasattr(X_t, "columns")):
        # Cố gắng lấy tên cột từ nhiều nguồn theo thứ tự ưu tiên
        cols = None

        if isinstance(pipeline, dict) and "feature_names" in pipeline:
            # Pipeline được lưu là dict và có key "feature_names"
            cols = pipeline["feature_names"]
        elif hasattr(pipeline, "feature_names_in_"):
            # Pipeline sklearn đã fit với DataFrame -> có thuộc tính feature_names_in_
            cols = list(pipeline.feature_names_in_)
        elif hasattr(pipeline, "steps"):
            # Pipeline có steps nhưng không đáng tin cậy -> bỏ qua
            cols = None

        if cols is None:
            # Không lấy được từ pipeline -> dùng model_feature_names làm fallback
            cols = model_feature_names

        # Chuyển numpy array thành DataFrame với tên cột đã xác định
        X_t = pd.DataFrame(X_t, columns=cols)

    # -----------------------------------------------------------------------
    # BƯỚC 11: ĐỒNG BỘ CỘT GIỮA X_t VÀ MODEL_FEATURE_NAMES
    # -----------------------------------------------------------------------
    # Đảm bảo X_t chỉ chứa đúng các cột model cần, theo đúng thứ tự
    if isinstance(X_t, pd.DataFrame):
        # Tìm các cột bị thiếu (model cần nhưng X_t không có)
        missing = [c for c in model_feature_names if c not in X_t.columns]
        # Tìm các cột thừa (X_t có nhưng model không dùng)
        extra = [c for c in X_t.columns if c not in model_feature_names]

        if missing:
            # Cảnh báo và điền 0 vào các cột bị thiếu
            print(f"[WARN] training data did not have the following fields: {missing}")
            for c in missing:
                X_t[c] = 0  # Điền 0 thay vì NaN để tránh lỗi trong model

        if extra:
            # Xóa các cột model không biết (model có thể lỗi nếu nhận cột lạ)
            print(f"[DEBUG] Dropping extra columns not expected by model: {extra}")
            X_t = X_t.drop(columns=extra)

        # Sắp xếp lại thứ tự cột theo ĐÚNG thứ tự model được training
        # Thứ tự sai sẽ cho kết quả predict sai hoàn toàn!
        X_t = X_t[model_feature_names]

    # -----------------------------------------------------------------------
    # BƯỚC 12: THỰC HIỆN DỰ ĐOÁN
    # -----------------------------------------------------------------------
    pred = model.predict(X_t)   # Dự đoán giá trị lượng mưa cho tất cả mẫu trong tập test

    # Một số model wrapper trả về object có thuộc tính .predictions thay vì mảng trực tiếp
    # ví dụ: AutoML frameworks hay custom wrapper
    try:
        pred = getattr(pred, "predictions", pred)  # Lấy .predictions nếu có, nếu không giữ nguyên pred
    except Exception:
        pass    # Nếu không truy cập được -> giữ nguyên pred

    # Đảm bảo pred là numpy array 1D
    # reshape(-1) : chuyển mảng bất kỳ hình dạng thành mảng 1D
    # Ví dụ: shape (100,1) -> (100,)
    pred = np.array(pred).reshape(-1)

    # -----------------------------------------------------------------------
    # BƯỚC 13: LƯU TOP 50 SAI SỐ LỚN NHẤT
    # -----------------------------------------------------------------------
    # Tạo DataFrame kết quả bao gồm toàn bộ dữ liệu gốc + thông tin dự đoán
    df_out = df.copy()               # Sao chép DataFrame gốc để không thay đổi dữ liệu gốc
    df_out["y_true"] = y             # Cột giá trị thực tế (ground truth)
    df_out["y_pred"] = pred          # Cột giá trị dự đoán của model
    # Tính sai số tuyệt đối: |y_true - y_pred|
    # .abs() lấy giá trị tuyệt đối (không quan tâm âm hay dương, chỉ xem độ lớn sai số)
    df_out["abs_err"] = (df_out["y_true"] - df_out["y_pred"]).abs()

    # Sắp xếp theo sai số tuyệt đối giảm dần và lấy 50 hàng đầu (sai nhất)
    # sort_values("abs_err", ascending=False): sắp xếp giảm dần (sai nhất lên đầu)
    # .head(50): lấy 50 hàng đầu tiên
    top50 = df_out.sort_values("abs_err", ascending=False).head(50)

    # Lưu ra file CSV ở thư mục gốc của project
    top50_path = ROOT / "debug_top50_errors.csv"
    top50.to_csv(top50_path, index=False)   # index=False: không lưu index của DataFrame vào file
    print(f"[OK] Saved top 50 errors to: {top50_path}")

    # -----------------------------------------------------------------------
    # BƯỚC 14: TÍNH VÀ LƯU RMSE THEO TỪNG TRẠM
    # -----------------------------------------------------------------------
    # Chỉ làm bước này nếu dữ liệu có cột "station_id" (mã trạm đo thời tiết)
    if "station_id" in df_out.columns:
        import numpy as _np  # Import lại numpy với alias _np để dùng trong lambda (tránh scope issue)

        # grp: nhóm dữ liệu theo station_id, tính n (số lượng mẫu) và RMSE
        # Lưu ý: cách agg này không ổn định (không truyền được y_pred vào lambda)
        # -> chỉ là bước trung gian, sẽ được thay thế bằng grp2 bên dưới
        grp = df_out.groupby("station_id").agg(n=(target_col, "size"), rmse=(lambda x: _np.sqrt(((x - df_out.loc[x.index, 'y_pred']) ** 2).mean())))

        # Định nghĩa hàm tính RMSE cho từng nhóm (từng trạm)
        # sub: DataFrame con của một trạm (station_id cụ thể)
        def station_rmse(sub):
            """Tính RMSE cho một trạm cụ thể."""
            # sub[target_col]: giá trị thực của trạm này
            # sub['y_pred']   : giá trị dự đoán cho trạm này
            # Công thức RMSE: sqrt( mean( (actual - predicted)^2 ) )
            return float(np.sqrt(((sub[target_col] - sub['y_pred']) ** 2).mean()))

        # Tính RMSE cho từng trạm bằng groupby + apply
        # .apply(station_rmse): áp dụng hàm station_rmse cho mỗi nhóm station_id
        # .rename("rmse")      : đặt tên cho cột kết quả là "rmse"
        # .reset_index()       : chuyển station_id từ index về cột bình thường
        grp2 = df_out.groupby("station_id").apply(station_rmse).rename("rmse").reset_index()

        # Sắp xếp theo RMSE giảm dần (trạm tệ nhất lên đầu)
        grp2 = grp2.sort_values("rmse", ascending=False)

        # Lưu ra file CSV
        stations_path = ROOT / "debug_worst_stations.csv"
        grp2.to_csv(stations_path, index=False)
        print(f"[OK] Saved per-station RMSE to: {stations_path}")
    else:
        # Nếu không có cột station_id -> bỏ qua bước này và in cảnh báo
        print("[WARN] 'station_id' not in dataframe; skipping per-station RMSE")

    print("[DONE]")   # In thông báo hoàn thành


# ---------------------------------------------------------------------------
# PHẦN 5: ĐIỂM VÀO CỦA CHƯƠNG TRÌNH (ENTRY POINT)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Khối này chỉ chạy khi script được gọi trực tiếp (python3 run_diagnostics.py)
    # Không chạy khi file được import như một module
    try:
        main()  # Gọi hàm chính
    except Exception:
        # Bắt bất kỳ exception nào chưa được xử lý
        traceback.print_exc()  # In đầy đủ stack trace để dễ debug
        sys.exit(2)            # Thoát với exit code 2 (báo hiệu lỗi nghiêm trọng không mong đợi)
