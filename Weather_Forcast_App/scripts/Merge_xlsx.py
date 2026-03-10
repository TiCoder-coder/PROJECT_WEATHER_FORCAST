import sys
import re
from pathlib import Path
import pandas as pd


# ============================================================
# FIX ENCODING CHO WINDOWS CONSOLE (chống lỗi in tiếng Việt)
# ============================================================
# - Trên Windows, terminal đôi khi dùng encoding mặc định không phải UTF-8
# - Khi print tiếng Việt dễ bị lỗi/ra ký tự lạ
# - Đoạn này bọc lại stdout/stderr để luôn dùng UTF-8
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


# ============================================================
# CẤU HÌNH THƯ MỤC & TÊN FILE
# ============================================================
# DYNAMIC PATH: tự tính từ vị trí project root, không hardcode Linux path
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
MERGE_DIR_ABS = str(Path(_PROJECT_ROOT) / "data" / "data_merge")
OUTPUT_DIR_ABS = str(Path(_PROJECT_ROOT) / "data" / "data_crawl")

# Tên file output sau khi merge cho nhóm "khác" (không phải vietnam_weather_)
MERGE_FILENAME = "merged_vrain_data.xlsx"

# Tên file output sau khi merge cho nhóm vietnam_weather_
MERGE_VIETNAM_FILENAME = "merged_vietnam_weather_data.xlsx"

# Log file dùng để lưu danh sách tên file đã merge (tránh merge lại lần sau)
LOG_FILENAME = "merged_files_log.txt"
LOG_VIETNAM_FILENAME = "merged_vietnam_files_log.txt"

# ============================================================
# CƠ CHẾ SAVE: dùng pandas to_excel (nhanh hơn openpyxl row-by-row)
# ============================================================
SAVE_EVERY_N_ROWS = 30000  # kept for backward compat, not used in new logic

MASTER_COLUMNS = [
    "station_id",
    "station_name",
    "province",
    "district",
    "latitude",
    "longitude",
    "timestamp",
    "data_source",
    "data_quality",
    "data_time",
    "temperature_current",
    "temperature_max",
    "temperature_min",
    "temperature_avg",
    "humidity_current",
    "humidity_max",
    "humidity_min",
    "humidity_avg",
    "pressure_current",
    "pressure_max",
    "pressure_min",
    "pressure_avg",
    "wind_speed_current",
    "wind_speed_max",
    "wind_speed_min",
    "wind_speed_avg",
    "wind_direction_current",
    "wind_direction_avg",
    "rain_current",
    "rain_max",
    "rain_min",
    "rain_avg",
    "rain_total",
    "cloud_cover_current",
    "cloud_cover_max",
    "cloud_cover_min",
    "cloud_cover_avg",
    "visibility_current",
    "visibility_max",
    "visibility_min",
    "visibility_avg",
    "thunder_probability",
    "status",
]

# Mapping chuẩn hóa từ tiếng Việt, camelCase, snake_case cũ sang snake_case chuẩn
COLUMN_SCHEMA_MAPPING = {
    # Tiếng Việt sang snake_case
    "Mã trạm": "station_id",
    "Tên trạm": "station_name",
    "Tỉnh/Thành phố": "province",
    "Huyện": "district",
    "Vĩ độ": "latitude",
    "Kinh độ": "longitude",
    "Dấu thời gian": "timestamp",
    "Nguồn dữ liệu": "data_source",
    "Chất lượng dữ liệu": "data_quality",
    "Thời gian cập nhật": "data_time",
    "Nhiệt độ hiện tại": "temperature_current",
    "Nhiệt độ tối đa": "temperature_max",
    "Nhiệt độ tối thiểu": "temperature_min",
    "Nhiệt độ trung bình": "temperature_avg",
    "Độ ẩm hiện tại": "humidity_current",
    "Độ ẩm tối đa": "humidity_max",
    "Độ ẩm tối thiểu": "humidity_min",
    "Độ ẩm trung bình": "humidity_avg",
    "Áp suất hiện tại": "pressure_current",
    "Áp suất tối đa": "pressure_max",
    "Áp suất tối thiểu": "pressure_min",
    "Áp suất trung bình": "pressure_avg",
    "Tốc độ gió hiện tại": "wind_speed_current",
    "Tốc độ gió tối đa": "wind_speed_max",
    "Tốc độ gió tối thiểu": "wind_speed_min",
    "Tốc độ gió trung bình": "wind_speed_avg",
    "Hướng gió hiện tại": "wind_direction_current",
    "Hướng gió trung bình": "wind_direction_avg",
    "Lượng mưa hiện tại": "rain_current",
    "Lượng mưa tối đa": "rain_max",
    "Lượng mưa tối thiểu": "rain_min",
    "Lượng mưa trung bình": "rain_avg",
    "Tổng lượng mưa": "rain_total",
    "Độ che phủ mây hiện tại": "cloud_cover_current",
    "Độ che phủ mây tối đa": "cloud_cover_max",
    "Độ che phủ mây tổi thiểu": "cloud_cover_min",
    "Độ che phủ mây trung bình": "cloud_cover_avg",
    "Tầm nhìn hiện tại": "visibility_current",
    "Tầm nhìn đa": "visibility_max",
    "Tầm nhìn tối thiểu": "visibility_min",
    "Tầm nhìn trung bình": "visibility_avg",
    "Xác xuất sấm sét": "thunder_probability",
    "Tình trạng": "status",
    # camelCase sang snake_case
    "stationId": "station_id",
    "stationName": "station_name",
    "dataSource": "data_source",
    "dataQuality": "data_quality",
    "dataTime": "data_time",
    "temperatureCurrent": "temperature_current",
    "temperatureMax": "temperature_max",
    "temperatureMin": "temperature_min",
    "temperatureAvg": "temperature_avg",
    "humidityCurrent": "humidity_current",
    "humidityMax": "humidity_max",
    "humidityMin": "humidity_min",
    "humidityAvg": "humidity_avg",
    "pressureCurrent": "pressure_current",
    "pressureMax": "pressure_max",
    "pressureMin": "pressure_min",
    "pressureAvg": "pressure_avg",
    "windSpeedCurrent": "wind_speed_current",
    "windSpeedMax": "wind_speed_max",
    "windSpeedMin": "wind_speed_min",
    "windSpeedAvg": "wind_speed_avg",
    "windDirectionCurrent": "wind_direction_current",
    "windDirectionAvg": "wind_direction_avg",
    "rainCurrent": "rain_current",
    "rainMax": "rain_max",
    "rainMin": "rain_min",
    "rainAvg": "rain_avg",
    "rainTotal": "rain_total",
    "cloudCoverCurrent": "cloud_cover_current",
    "cloudCoverMax": "cloud_cover_max",
    "cloudCoverMin": "cloud_cover_min",
    "cloudCoverAvg": "cloud_cover_avg",
    "visibilityCurrent": "visibility_current",
    "visibilityMax": "visibility_max",
    "visibilityMin": "visibility_min",
    "visibilityAvg": "visibility_avg",
    "thunderProbability": "thunder_probability",
    # snake_case cũ sang snake_case chuẩn (giữ nguyên)
    "station_id": "station_id",
    "station_name": "station_name",
    "data_source": "data_source",
    "data_quality": "data_quality",
    "data_time": "data_time",
    "temperature_current": "temperature_current",
    "temperature_max": "temperature_max",
    "temperature_min": "temperature_min",
    "temperature_avg": "temperature_avg",
    "humidity_current": "humidity_current",
    "humidity_max": "humidity_max",
    "humidity_min": "humidity_min",
    "humidity_avg": "humidity_avg",
    "pressure_current": "pressure_current",
    "pressure_max": "pressure_max",
    "pressure_min": "pressure_min",
    "pressure_avg": "pressure_avg",
    "wind_speed_current": "wind_speed_current",
    "wind_speed_max": "wind_speed_max",
    "wind_speed_min": "wind_speed_min",
    "wind_speed_avg": "wind_speed_avg",
    "wind_direction_current": "wind_direction_current",
    "wind_direction_avg": "wind_direction_avg",
    "rain_current": "rain_current",
    "rain_max": "rain_max",
    "rain_min": "rain_min",
    "rain_avg": "rain_avg",
    "rain_total": "rain_total",
    "cloud_cover_current": "cloud_cover_current",
    "cloud_cover_max": "cloud_cover_max",
    "cloud_cover_min": "cloud_cover_min",
    "cloud_cover_avg": "cloud_cover_avg",
    "visibility_current": "visibility_current",
    "visibility_max": "visibility_max",
    "visibility_min": "visibility_min",
    "visibility_avg": "visibility_avg",
    "thunder_probability": "thunder_probability",
}

# ============================================================
# COLUMN_ALIASES = BẢN ĐỒ CHUẨN HOÁ TÊN CỘT (alias -> chuẩn)
# ============================================================
# - Dữ liệu nguồn có thể bị gõ sai dấu hoặc khác cách viết
# - Ví dụ "tối thiểu" bị viết nhầm thành "tổi thiểu"
# - Ở đây bạn quyết định dùng "Độ che phủ mây tổi thiểu" làm dạng chuẩn
#   => mọi biến thể khác sẽ được map về dạng này
# - Điều này giúp tránh sinh ra 2 cột gần giống nhau chỉ vì khác chữ
COLUMN_ALIASES = {
    "Độ che phủ mây tối thiểu": "Độ che phủ mây tổi thiểu",
    "Tầm nhìn tối đa": "Tầm nhìn đa",
    "Xác suất sấm sét": "Xác xuất sấm sét",
    "Xác suất sét": "Xác xuất sấm sét",
    "Xác suất sấm sét": "Xác xuất sấm sét",
}


def norm_col(x) -> str:
    # ============================================================
    # CHUẨN HOÁ TÊN CỘT
    # ============================================================
    # - Ép về string + strip() để bỏ khoảng trắng đầu/cuối
    # - regex thay nhiều khoảng trắng liên tiếp thành 1 khoảng trắng
    # - Sau đó tra COLUMN_ALIASES:
    #   + nếu có alias -> trả về tên chuẩn
    #   + nếu không -> giữ nguyên
    s = re.sub(r"\s+", " ", str(x).strip())
    return COLUMN_ALIASES.get(s, s)


def load_processed_files(log_path: Path) -> set[str]:
    # ============================================================
    # ĐỌC LOG -> TẬP TÊN FILE ĐÃ XỬ LÝ
    # ============================================================
    # - Mục tiêu: chạy script nhiều lần nhưng không merge lại file cũ
    # - log file chỉ chứa mỗi dòng là tên file .xlsx đã merge
    # - Nếu log chưa tồn tại -> trả set rỗng
    if not log_path.exists():
        return set()
    processed = set()
    try:
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                if name:
                    processed.add(name)
    except Exception as e:
        # Nếu log lỗi cũng không cho crash toàn bộ merge
        print(f"Loi khi doc log file {log_path.name}: {e}")
    return processed


def save_processed_files(log_path: Path, processed_files: set[str]) -> None:
    # ============================================================
    # GHI LOG: LƯU DANH SÁCH FILE ĐÃ MERGE
    # ============================================================
    # - Ghi toàn bộ set processed_files ra file
    # - sorted(...) để log ổn định, dễ đọc/diff
    try:
        with log_path.open("w", encoding="utf-8") as f:
            for name in sorted(processed_files):
                f.write(name + "\n")
    except Exception as e:
        # Không crash nếu log ghi lỗi
        print(f"Loi khi ghi log file {log_path.name}: {e}")


def read_data_file(file_path: Path) -> pd.DataFrame:
    # ============================================================
    # ĐỌC 1 FILE EXCEL HOẶC CSV -> DATAFRAME
    # ============================================================
    try:
        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(file_path, encoding="utf-8-sig")
        elif suffix == ".xlsx":
            df = pd.read_excel(file_path, engine="openpyxl")
        else:
            print(f"  Bo qua file khong ho tro: {file_path.name}")
            return pd.DataFrame()
        if df is None or df.empty:
            return pd.DataFrame()
        return df
    except Exception as e:
        print(f"Loi khi doc file {file_path.name}: {e}")
        return pd.DataFrame()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # ============================================================
    # LÀM SẠCH DATAFRAME TRƯỚC KHI MERGE
    # ============================================================
    # Những việc làm chính:
    # 1) copy() để tránh sửa trực tiếp df gốc
    # 2) Chuẩn hoá tên cột bằng norm_col
    # 3) Loại bỏ cột trùng tên sau chuẩn hoá (duplicated)
    # 4) Xoá các cột "Unnamed: ..." (thường do excel xuất dư cột)
    df = df.copy()

    # Chuẩn hoá tên cột và mapping về snake_case chuẩn
    df.columns = [COLUMN_SCHEMA_MAPPING.get(norm_col(c), norm_col(c)) for c in df.columns]

    # Nếu sau chuẩn hoá bị trùng cột (vd: 2 cột khác nhau map về cùng alias)
    # -> giữ cột đầu tiên, bỏ cột trùng
    df = df.loc[:, ~df.columns.duplicated()]

    # Các cột kiểu "Unnamed: 0", "Unnamed: 1" thường do index/format excel
    drop_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    return df


def get_new_data_files(output_dir: Path, processed_files: set[str]) -> tuple[list[Path], list[Path]]:
    # ============================================================
    # QUÉT THƯ MỤC output -> LẤY CÁC FILE .xlsx VÀ .csv CHƯA XỬ LÝ
    # ============================================================
    if not output_dir.exists():
        print(f"Thu muc nguon khong ton tai: {output_dir}")
        return [], []

    all_data_files = sorted(
        [f for f in output_dir.iterdir()
         if f.is_file() and f.suffix.lower() in {".xlsx", ".csv"}
         and not f.name.startswith("~$")]
    )
    if not all_data_files:
        print(f"Khong tim thay file .xlsx/.csv nao trong thu muc: {output_dir}")
        return [], []

    vietnam_files, other_files = [], []
    for file_path in all_data_files:
        if file_path.name in processed_files:
            continue
        if file_path.name.startswith("vietnam_weather_"):
            vietnam_files.append(file_path)
        else:
            other_files.append(file_path)

    print(f"Tong so file trong output: {len(all_data_files)}")
    print(f"So file vietnam_weather_ moi: {len(vietnam_files)}")
    print(f"So file khac moi: {len(other_files)}")
    return vietnam_files, other_files


def merge_single_category_fast(
    file_list: list[Path],
    merge_path: Path,
    log_path: Path,
    processed_files: set[str],
    category_name: str,
) -> None:
    # ============================================================
    # MERGE 1 NHÓM FILE BẰNG PANDAS (NHANH)
    # ============================================================
    if not file_list:
        print(f"Khong co file {category_name} moi de merge.")
        return

    print(f"\n=== MERGE NHANH: {category_name.upper()} ===")
    print(f"So luong file moi: {len(file_list)}")
    print(f"File merge: {merge_path}")

    # 1) Đọc file merge hiện có (nếu tồn tại)
    existing_df = pd.DataFrame()
    if merge_path.exists():
        try:
            existing_df = pd.read_excel(merge_path, engine="openpyxl")
            existing_df.columns = [COLUMN_SCHEMA_MAPPING.get(norm_col(c), norm_col(c)) for c in existing_df.columns]
            print(f"  File merge hien co: {len(existing_df)} dong")
        except Exception as e:
            print(f"  Loi doc file merge cu: {e}, se tao moi.")
            existing_df = pd.DataFrame()

    # 2) Đọc từng file mới → clean → collect
    new_dfs = []
    ok_files = []
    for idx, file_path in enumerate(sorted(file_list), start=1):
        print(f"  [{idx}/{len(file_list)}] Doc: {file_path.name}", end=" ")
        df = read_data_file(file_path)
        if df.empty:
            print("-> rong, bo qua.")
            continue
        df = clean_dataframe(df)
        new_dfs.append(df)
        ok_files.append(file_path)
        print(f"-> {len(df)} dong")

    if not new_dfs:
        print(f"  Khong co du lieu moi de merge cho {category_name}.")
        return

    # 3) Concat tất cả
    all_new = pd.concat(new_dfs, ignore_index=True)
    print(f"  Tong dong moi: {len(all_new)}")

    if not existing_df.empty:
        merged = pd.concat([existing_df, all_new], ignore_index=True)
    else:
        merged = all_new

    # 4) Chuẩn hóa cột theo MASTER_COLUMNS
    for c in MASTER_COLUMNS:
        if c not in merged.columns:
            merged[c] = pd.NA
    # Giữ đúng thứ tự MASTER_COLUMNS + các cột thừa (nếu có)
    final_cols = [c for c in MASTER_COLUMNS if c in merged.columns]
    merged = merged[final_cols]

    # 5) Ghi ra file 1 lần (nhanh hơn row-by-row rất nhiều)
    merged.to_excel(merge_path, index=False, engine="openpyxl")
    print(f"  Da ghi {len(merged)} dong vao {merge_path.name}")

    # 6) Cập nhật log
    for fp in ok_files:
        processed_files.add(fp.name)
    save_processed_files(log_path, processed_files)

    print(f"\n=== XONG {category_name}: OK {len(ok_files)}/{len(file_list)} file ===")


def merge_excel_files_once(base_dir: Path) -> None:
    # ============================================================
    # HÀM "ĐIỀU PHỐI" MERGE CHÍNH (CHẠY 1 LẦN)
    # ============================================================
    # - base_dir: thư mục gốc (thường là root project)
    #
    # Tạo cấu trúc:
    # base_dir/output       : file excel nguồn
    # base_dir/Merge_data   : file merged + logs
    #
    # Merge theo 2 nhóm:
    # 1) vietnam_weather_  -> merged_vietnam_weather_data.xlsx + log riêng
    # 2) các file còn lại  -> merged_vrain_data.xlsx + log riêng

    output_dir = Path(OUTPUT_DIR_ABS)
    merge_dir = Path(MERGE_DIR_ABS)
    merge_dir.mkdir(parents=True, exist_ok=True)

    merge_vietnam_path = merge_dir / MERGE_VIETNAM_FILENAME
    log_vietnam_path = merge_dir / LOG_VIETNAM_FILENAME

    merge_other_path = merge_dir / MERGE_FILENAME
    log_other_path = merge_dir / LOG_FILENAME

    print("======== BAT DAU MERGE =========")
    print(f"Thu muc nguon (output):     {output_dir}")
    print(f"Thu muc merge (Merge_data): {merge_dir}")
    print("================================")

    # Đọc log đã xử lý của từng nhóm
    processed_vietnam = load_processed_files(log_vietnam_path)
    processed_other = load_processed_files(log_other_path)

    # Hợp 2 tập lại để lọc tất cả file đã xử lý (không phân biệt nhóm)
    processed_all = processed_vietnam.union(processed_other)

    # Lấy danh sách file mới trong output (chưa nằm trong processed_all)
    vietnam_files, other_files = get_new_data_files(output_dir, processed_all)

    # Merge nhóm vietnam_weather_
    merge_single_category_fast(
        vietnam_files,
        merge_vietnam_path,
        log_vietnam_path,
        processed_vietnam,
        "vietnam_weather_"
    )

    # Merge nhóm còn lại
    merge_single_category_fast(
        other_files,
        merge_other_path,
        log_other_path,
        processed_other,
        "khac"
    )

    print("\n======== KET THUC MERGE =========")


if __name__ == "__main__":
    # ============================================================
    # ENTRY POINT (khi chạy file python trực tiếp)
    # ============================================================
    # - SCRIPT_DIR: thư mục chứa file script hiện tại
    # - BASE_DIR: thư mục cha của SCRIPT_DIR (thường là root project)
    # Không cần BASE_DIR nữa, dùng đường dẫn tuyệt đối
    output_dir = Path(OUTPUT_DIR_ABS)
    if not output_dir.exists():
        print(f"ERROR: Khong tim thay thu muc output tai: {output_dir}")
        sys.exit(1)

    # Chạy merge 1 lần
    merge_excel_files_once(None)