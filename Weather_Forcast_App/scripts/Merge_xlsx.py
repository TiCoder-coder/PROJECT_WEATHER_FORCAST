import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

# Fix encoding cho Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


MERGE_DIR_NAME = "Merge_data"
OUTPUT_DIR_NAME = "output"
MERGE_FILENAME = "merged_weather_data.xlsx"
LOG_FILENAME = "merged_files_log.txt"


def load_processed_files(log_path: Path) -> set[str]:
    """
    Đọc danh sách các file .xlsx đã được merge trước đó
    từ file log (mỗi dòng = 1 tên file).
    """
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
        print(f"⚠️ Lỗi khi đọc log file {log_path.name}: {e}")
    return processed


def save_processed_files(log_path: Path, processed_files: set[str]) -> None:
    """
    Ghi lại danh sách các file đã merge vào log file.
    Mỗi dòng = 1 tên file .xlsx.
    """
    try:
        with log_path.open("w", encoding="utf-8") as f:
            for name in sorted(processed_files):
                f.write(name + "\n")
    except Exception as e:
        print(f"⚠️ Lỗi khi ghi log file {log_path.name}: {e}")


def get_new_excel_files(output_dir: Path, processed_files: set[str]) -> list[Path]:
    """
    Lấy danh sách các file .xlsx MỚI trong thư mục output
    (những file chưa có trong processed_files).
    """
    if not output_dir.exists():
        print(f"❌ Thư mục nguồn không tồn tại: {output_dir}")
        return []

    all_excel_files = sorted(output_dir.glob("*.xlsx"))
    if not all_excel_files:
        print(f"❌ Không tìm thấy file .xlsx nào trong thư mục: {output_dir}")
        return []

    new_files = [f for f in all_excel_files if f.name not in processed_files]

    print(f"📁 Tổng số file .xlsx trong output: {len(all_excel_files)}")
    print(f"🆕 Số file mới chưa merge: {len(new_files)}")

    return new_files


def merge_excel_files_once(base_dir: Path) -> None:
    """
    Hàm chính:
    - Đọc log các file đã merge
    - Tìm các file .xlsx mới trong thư mục output
    - Append dữ liệu mới vào file merge cũ (nếu có)
    - Cập nhật lại log
    """

    output_dir = base_dir / OUTPUT_DIR_NAME
    merge_dir = base_dir / MERGE_DIR_NAME
    merge_dir.mkdir(parents=True, exist_ok=True)

    merge_path = merge_dir / MERGE_FILENAME
    log_path = merge_dir / LOG_FILENAME

    print("======== BẮT ĐẦU MERGE =========")
    print(f"📂 Thư mục nguồn (output):    {output_dir}")
    print(f"📂 Thư mục merge (Merge_data): {merge_dir}")
    print(f"📝 File log:                    {log_path}")
    print(f"📊 File merge:                  {merge_path}")
    print("================================")

    processed_files = load_processed_files(log_path)
    if processed_files:
        print(f"✅ Đã từng merge {len(processed_files)} file trước đó.")
    else:
        print("ℹ️ Chưa có log hoặc log trống. Xem như chạy merge lần đầu.")

    new_files = get_new_excel_files(output_dir, processed_files)
    if not new_files:
        print("✅ Không có file mới để merge. Kết thúc.")
        return

    new_dfs = []
    for file_path in new_files:
        try:
            print(f"📥 Đang đọc file mới: {file_path.name}")
            df = pd.read_excel(file_path)
            new_dfs.append(df)
        except Exception as e:
            print(f"⚠️ Lỗi khi đọc file {file_path.name}: {e}")

    if not new_dfs:
        print("❌ Không đọc được dữ liệu hợp lệ từ các file mới.")
        return

    new_data = pd.concat(new_dfs, ignore_index=True)
    print(f"🆕 Tổng số dòng dữ liệu mới: {len(new_data)}")

    if merge_path.exists():
        try:
            print(f"📂 Đang đọc file merge cũ: {merge_path.name}")
            old_data = pd.read_excel(merge_path)
            before_rows = len(old_data)
            merged_df = pd.concat([old_data, new_data], ignore_index=True)

            print(f"🔗 Đã append {len(new_data)} dòng vào {before_rows} dòng cũ.")
            print(f"📊 Tổng số dòng sau khi merge: {len(merged_df)}")
        except Exception as e:
            print(f"⚠️ Lỗi khi đọc file merge cũ, chỉ dùng dữ liệu mới. Chi tiết: {e}")
            merged_df = new_data
    else:
        print("🆕 Chưa có file merge cũ. Tạo file merge mới từ dữ liệu mới.")
        merged_df = new_data
        print(f"📊 Tổng số dòng trong file merge mới: {len(merged_df)}")

    try:
        merged_df.to_excel(merge_path, index=False)
        print(f"🎉 Đã ghi file merge thành công tại:\n    {merge_path}")
    except Exception as e:
        print(f"💥 Lỗi khi ghi file Excel merge: {e}")
        return

    for f in new_files:
        processed_files.add(f.name)
    save_processed_files(log_path, processed_files)
    print(f"📝 Đã cập nhật log với {len(new_files)} file mới.")

    print("======== KẾT THÚC MERGE =========")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    merge_excel_files_once(BASE_DIR)