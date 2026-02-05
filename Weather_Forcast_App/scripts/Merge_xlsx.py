import sys
import re
from pathlib import Path
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.worksheet.worksheet import Worksheet


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
# - OUTPUT_DIR_NAME: nơi chứa các file excel đầu vào (được crawler/export ra)
# - MERGE_DIR_NAME: nơi lưu các file excel đã merge + log theo dõi
MERGE_DIR_NAME = "Merge_data"
OUTPUT_DIR_NAME = "output"

# Tên file output sau khi merge cho nhóm "khác" (không phải vietnam_weather_)
MERGE_FILENAME = "merged_vrain_data.xlsx"

# Tên file output sau khi merge cho nhóm vietnam_weather_
MERGE_VIETNAM_FILENAME = "merged_vietnam_weather_data.xlsx"

# Log file dùng để lưu danh sách tên file đã merge (tránh merge lại lần sau)
LOG_FILENAME = "merged_files_log.txt"
LOG_VIETNAM_FILENAME = "merged_vietnam_files_log.txt"

# ============================================================
# CƠ CHẾ SAVE THEO BATCH (tránh mất dữ liệu, giảm rủi ro crash)
# ============================================================
# - Khi append quá nhiều dòng vào workbook, nếu crash giữa chừng sẽ mất công
# - SAVE_EVERY_N_ROWS giúp cứ mỗi N dòng append sẽ save 1 lần xuống disk
# - N càng nhỏ càng an toàn nhưng save nhiều -> chậm hơn
SAVE_EVERY_N_ROWS = 30000

# ============================================================
# MASTER_COLUMNS = "SCHEMA CHUẨN" CHO FILE MERGE
# ============================================================
# - Đây là danh sách cột chuẩn mà file merge sẽ có
# - Khi đọc file mới:
#   + Nếu thiếu cột nào so với MASTER_COLUMNS -> tự thêm cột đó (NA)
#   + Nếu phát hiện cột mới ngoài schema -> mở rộng header (tự thêm vào cuối)
# - Mục tiêu: các file nguồn có thể lệch nhau chút về cột, vẫn merge được
MASTER_COLUMNS = [
    "Mã trạm", "Tên trạm", "Tỉnh/Thành phố", "Huyện", "Vĩ độ", "Kinh độ",
    "Dấu thời gian", "Nguồn dữ liệu", "Chất lượng dữ liệu", "Thời gian cập nhật",
    "Nhiệt độ hiện tại", "Nhiệt độ tối đa", "Nhiệt độ tối thiểu", "Nhiệt độ trung bình",
    "Độ ẩm hiện tại", "Độ ẩm tối đa", "Độ ẩm tối thiểu", "Độ ẩm trung bình",
    "Áp suất hiện tại", "Áp suất tối đa", "Áp suất tối thiểu", "Áp suất trung bình",
    "Tốc độ gió hiện tại", "Tốc độ gió tối đa", "Tốc độ gió tối thiểu", "Tốc độ gió trung bình",
    "Hướng gió hiện tại", "Hướng gió trung bình",
    "Lượng mưa hiện tại", "Lượng mưa tối đa", "Lượng mưa tối thiểu", "Lượng mưa trung bình",
    "Tổng lượng mưa",
    "Độ che phủ mây hiện tại", "Độ che phủ mây tối đa", "Độ che phủ mây tổi thiểu", "Độ che phủ mây trung bình",
    "Tầm nhìn hiện tại", "Tầm nhìn đa", "Tầm nhìn tối thiểu", "Tầm nhìn trung bình",
    "Xác xuất sấm sét", "Tình trạng",
]

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


def read_excel_file(file_path: Path) -> pd.DataFrame:
    # ============================================================
    # ĐỌC 1 FILE EXCEL -> DATAFRAME
    # ============================================================
    # - pd.read_excel sẽ đọc sheet đầu tiên mặc định
    # - Nếu file lỗi/đọc không được -> trả DataFrame rỗng
    try:
        df = pd.read_excel(file_path)
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

    # Chuẩn hoá tên cột
    df.columns = [norm_col(c) for c in df.columns]

    # Nếu sau chuẩn hoá bị trùng cột (vd: 2 cột khác nhau map về cùng alias)
    # -> giữ cột đầu tiên, bỏ cột trùng
    df = df.loc[:, ~df.columns.duplicated()]

    # Các cột kiểu "Unnamed: 0", "Unnamed: 1" thường do index/format excel
    drop_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    return df


def get_new_excel_files(output_dir: Path, processed_files: set[str]) -> tuple[list[Path], list[Path]]:
    # ============================================================
    # QUÉT THƯ MỤC output -> LẤY CÁC FILE .xlsx CHƯA XỬ LÝ
    # ============================================================
    # - output_dir: nơi chứa file excel nguồn
    # - processed_files: tập tên file đã merge (từ log)
    #
    # Trả về 2 list:
    # 1) vietnam_files: file có prefix "vietnam_weather_"
    # 2) other_files: các file còn lại
    if not output_dir.exists():
        print(f"Thu muc nguon khong ton tai: {output_dir}")
        return [], []

    all_excel_files = sorted(output_dir.glob("*.xlsx"))
    if not all_excel_files:
        print(f"Khong tim thay file .xlsx nao trong thu muc: {output_dir}")
        return [], []

    vietnam_files, other_files = [], []
    for file_path in all_excel_files:
        # Nếu file đã nằm trong log -> bỏ qua
        if file_path.name in processed_files:
            continue

        # Phân loại theo prefix tên file
        if file_path.name.startswith("vietnam_weather_"):
            vietnam_files.append(file_path)
        else:
            other_files.append(file_path)

    # In thống kê để biết pipeline đang xử lý được bao nhiêu file mới
    print(f"Tong so file .xlsx trong output: {len(all_excel_files)}")
    print(f"So file vietnam_weather_ moi: {len(vietnam_files)}")
    print(f"So file khac moi: {len(other_files)}")
    return vietnam_files, other_files


def _load_or_create_wb_ws(merge_path: Path) -> tuple[Workbook, Worksheet, list[str]]:
    # ============================================================
    # MỞ FILE MERGE NẾU ĐÃ TỒN TẠI, HOẶC TẠO FILE MỚI
    # ============================================================
    # Trả về:
    # - wb: Workbook openpyxl
    # - ws: worksheet active
    # - header: list tên cột hiện có trong dòng 1 của file merge
    #
    # Ý tưởng:
    # - Nếu merge_path tồn tại:
    #   + load_workbook để tiếp tục append vào file cũ
    #   + đọc header row (row=1) để biết schema hiện tại
    # - Nếu không tồn tại:
    #   + tạo workbook mới và viết MASTER_COLUMNS làm header
    if merge_path.exists():
        wb = load_workbook(merge_path)
        ws = wb.active

        # Đọc header dòng 1 theo số cột hiện tại
        header = []
        max_col = ws.max_column
        for col in range(1, max_col + 1):
            v = ws.cell(row=1, column=col).value
            if v is None:
                continue
            header.append(norm_col(v))

        # Nếu file merge có nhưng dòng header bị trống/hỏng
        # -> reset lại header theo MASTER_COLUMNS
        if not header:
            header = list(MASTER_COLUMNS)
            for i, col_name in enumerate(header, start=1):
                ws.cell(row=1, column=i).value = col_name

        return wb, ws, header

    # Nếu file merge chưa tồn tại -> tạo mới
    wb = Workbook()
    ws = wb.active
    ws.title = "data"

    # Ghi header = MASTER_COLUMNS vào dòng 1
    header = list(MASTER_COLUMNS)
    for i, col_name in enumerate(header, start=1):
        ws.cell(row=1, column=i).value = col_name

    # Save ngay để tạo file vật lý trên disk
    wb.save(merge_path)
    return wb, ws, header


def _ensure_header_has_columns(ws: Worksheet, header: list[str], cols_to_ensure: list[str]) -> list[str]:
    # ============================================================
    # ĐẢM BẢO HEADER CÓ ĐỦ CÁC CỘT TRONG cols_to_ensure
    # ============================================================
    # - Nếu thiếu cột -> append vào cuối header + ghi vào row 1 của sheet
    # - Trả về header mới (có thể đã được mở rộng)
    changed = False
    for c in cols_to_ensure:
        c = norm_col(c)
        if c not in header:
            header.append(c)
            ws.cell(row=1, column=len(header)).value = c
            changed = True
    if changed:
        print(f"  + Da mo rong schema, tong so cot hien tai: {len(header)}")
    return header


def _to_excel_value(v):
    # ============================================================
    # CHUYỂN GIÁ TRỊ PANDAS -> GIÁ TRỊ HỢP LỆ CHO OPENPYXL
    # ============================================================
    # - pd.NA / NaN: openpyxl không thích -> đổi thành None
    # - pd.Timestamp: đổi sang datetime python bằng to_pydatetime()
    # - còn lại giữ nguyên
    try:
        if pd.isna(v):
            return None
    except Exception:
        # Một số kiểu object có thể khiến pd.isna lỗi -> bỏ qua
        pass
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    return v


def append_df_incremental(
    wb: Workbook,
    ws: Worksheet,
    header: list[str],
    df: pd.DataFrame,
    merge_path: Path,
    save_every_n_rows: int = SAVE_EVERY_N_ROWS,
) -> int:
    # ============================================================
    # APPEND DATAFRAME VÀO FILE EXCEL THEO KIỂU INCREMENTAL
    # ============================================================
    # Mục tiêu:
    # - Append từng dòng một vào worksheet để không phải giữ toàn bộ dữ liệu trong RAM
    # - Save định kỳ mỗi save_every_n_rows dòng để:
    #   + giảm rủi ro mất dữ liệu
    #   + tránh workbook quá lớn mà chưa save
    #
    # Bước chính:
    # 1) Bổ sung các cột còn thiếu trong df theo header (NA)
    # 2) Reorder df theo đúng thứ tự header
    # 3) itertuples để iterate nhanh hơn iterrows
    # 4) ws.append từng dòng
    # 5) save theo batch
    for c in header:
        # Nếu df không có cột nào đó trong schema -> tạo cột đó toàn NA
        if c not in df.columns:
            df[c] = pd.NA

    # Reorder đúng schema để dữ liệu nằm đúng cột
    df = df[header]

    appended = 0
    buffer_count = 0

    # itertuples(index=False, name=None) -> trả tuple thuần, nhanh
    for row in df.itertuples(index=False, name=None):
        # Convert giá trị sang dạng excel-friendly
        excel_row = [_to_excel_value(x) for x in row]
        ws.append(excel_row)
        appended += 1
        buffer_count += 1

        # Save định kỳ để giảm rủi ro crash/mất dữ liệu
        if save_every_n_rows > 0 and buffer_count >= save_every_n_rows:
            wb.save(merge_path)
            buffer_count = 0

    # Save cuối cùng sau khi append xong
    wb.save(merge_path)
    return appended


def merge_single_category_incremental(
    file_list: list[Path],
    merge_path: Path,
    log_path: Path,
    processed_files: set[str],
    category_name: str,
) -> None:
    # ============================================================
    # MERGE 1 NHÓM FILE (category) THEO KIỂU INCREMENTAL
    # ============================================================
    # Tham số:
    # - file_list: danh sách file cần merge (chưa processed)
    # - merge_path: file excel đích để append vào
    # - log_path: file log của category đó
    # - processed_files: set tên file đã xử lý (riêng cho category)
    # - category_name: tên hiển thị/log (vd: "vietnam_weather_" hoặc "khac")
    #
    # Luồng xử lý:
    # - Mở hoặc tạo file merge
    # - Đảm bảo header có MASTER_COLUMNS
    # - Với mỗi file:
    #   + đọc -> clean
    #   + nếu phát hiện cột mới -> mở rộng header
    #   + append dữ liệu incremental
    #   + ghi log ngay sau khi append thành công
    if not file_list:
        print(f"Khong co file {category_name} moi de merge.")
        return

    print(f"\n=== MERGE INCREMENTAL: {category_name.upper()} ===")
    print(f"So luong file: {len(file_list)}")
    print(f"File merge: {merge_path}")

    # Mở workbook merge hiện có hoặc tạo mới nếu chưa có
    wb, ws, header = _load_or_create_wb_ws(merge_path)

    # Đảm bảo header chứa đầy đủ MASTER_COLUMNS (schema chuẩn)
    header = _ensure_header_has_columns(ws, header, MASTER_COLUMNS)
    wb.save(merge_path)

    ok_count = 0
    for idx, file_path in enumerate(sorted(file_list), start=1):
        print(f"\n[{idx}/{len(file_list)}] Dang xu ly: {file_path.name}")

        # Đọc file excel nguồn
        df = read_excel_file(file_path)
        if df.empty:
            print("  - File rong/khong doc duoc, bo qua.")
            continue

        # Làm sạch: chuẩn hoá tên cột, bỏ cột unnamed, bỏ cột trùng
        df = clean_dataframe(df)

        # Nếu df có cột mới ngoài header hiện tại -> mở rộng schema
        new_cols = [c for c in df.columns if c not in header]
        if new_cols:
            print(f"  + Phat hien {len(new_cols)} cot moi: {new_cols}")
            header = _ensure_header_has_columns(ws, header, new_cols)
            wb.save(merge_path)

        try:
            # Append dữ liệu theo từng dòng + save theo batch
            appended = append_df_incremental(
                wb=wb,
                ws=ws,
                header=header,
                df=df,
                merge_path=merge_path,
                save_every_n_rows=SAVE_EVERY_N_ROWS,
            )
            print(f"  ✓ Da append {appended} dong tu {file_path.name}")

            # Sau khi append thành công:
            # - đánh dấu file đã xử lý
            # - ghi log ngay để nếu script dừng đột ngột vẫn không merge lại file này
            processed_files.add(file_path.name)
            save_processed_files(log_path, processed_files)
            ok_count += 1
        except Exception as e:
            # Nếu append lỗi:
            # - in lỗi
            # - KHÔNG ghi log (để lần sau chạy lại file này)
            print(f"  ✗ Loi khi append file {file_path.name}: {e}")
            print("  -> Khong danh dau processed (de lan sau chay lai).")

    # Save cuối cùng cho chắc chắn
    wb.save(merge_path)
    print(f"\n=== XONG {category_name}: OK {ok_count}/{len(file_list)} file ===")


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
    output_dir = base_dir / OUTPUT_DIR_NAME
    merge_dir = base_dir / MERGE_DIR_NAME
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
    vietnam_files, other_files = get_new_excel_files(output_dir, processed_all)

    # Merge nhóm vietnam_weather_
    merge_single_category_incremental(
        vietnam_files,
        merge_vietnam_path,
        log_vietnam_path,
        processed_vietnam,
        "vietnam_weather_"
    )

    # Merge nhóm còn lại
    merge_single_category_incremental(
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
    SCRIPT_DIR = Path(__file__).parent
    BASE_DIR = SCRIPT_DIR.parent

    print(f"Script dir: {SCRIPT_DIR}")
    print(f"Base dir:   {BASE_DIR}")

    # Kiểm tra xem thư mục output có tồn tại không, nếu không thì dừng
    output_dir = BASE_DIR / OUTPUT_DIR_NAME
    if not output_dir.exists():
        print(f"ERROR: Khong tim thay thu muc output tai: {output_dir}")
        sys.exit(1)

    # Chạy merge 1 lần
    merge_excel_files_once(BASE_DIR)               # (giữ nguyên theo yêu cầu: chỉ thêm chú thích, không đổi logic)
