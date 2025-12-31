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
        print(f"Loi khi doc log file {log_path.name}: {e}")
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
        print(f"Loi khi ghi log file {log_path.name}: {e}")


def get_new_excel_files(output_dir: Path, processed_files: set[str]) -> list[Path]:
    """
    Lấy danh sách các file .xlsx MỚI trong thư mục output
    (những file chưa có trong processed_files).
    """
    if not output_dir.exists():
        print(f"Thu muc nguon khong ton tai: {output_dir}")
        return []

    all_excel_files = sorted(output_dir.glob("*.xlsx"))
    if not all_excel_files:
        print(f"Khong tim thay file .xlsx nao trong thu muc: {output_dir}")
        return []

    new_files = [f for f in all_excel_files if f.name not in processed_files]

    print(f"Tong so file .xlsx trong output: {len(all_excel_files)}")
    print(f"So file moi chua merge: {len(new_files)}")

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

    print("======== BAT DAU MERGE =========")
    print(f"Thu muc nguon (output):    {output_dir}")
    print(f"Thu muc merge (Merge_data): {merge_dir}")
    print(f"File log:                    {log_path}")
    print(f"File merge:                  {merge_path}")
    print("================================")

    processed_files = load_processed_files(log_path)
    if processed_files:
        print(f"Da tung merge {len(processed_files)} file truoc do.")
    else:
        print("Chua co log hoac log trong. Xem nhu chay merge lan dau.")

    new_files = get_new_excel_files(output_dir, processed_files)
    if not new_files:
        print("Khong co file moi de merge. Ket thuc.")
        return

    new_dfs = []
    for file_path in new_files:
        try:
            print(f"Dang doc file moi: {file_path.name}")
            df = pd.read_excel(file_path)
            new_dfs.append(df)
        except Exception as e:
            print(f"Loi khi doc file {file_path.name}: {e}")

    if not new_dfs:
        print("Khong doc duoc du lieu hop le tu cac file moi.")
        return

    new_data = pd.concat(new_dfs, ignore_index=True)
    print(f"Tong so dong du lieu moi: {len(new_data)}")

    if merge_path.exists():
        try:
            print(f"Dang doc file merge cu: {merge_path.name}")
            old_data = pd.read_excel(merge_path)
            before_rows = len(old_data)
            merged_df = pd.concat([old_data, new_data], ignore_index=True)

            print(f"Da append {len(new_data)} dong vao {before_rows} dong cu.")
            print(f"Tong so dong sau khi merge: {len(merged_df)}")
        except Exception as e:
            print(f"Loi khi doc file merge cu, chi dung du lieu moi. Chi tiet: {e}")
            merged_df = new_data
    else:
        print("Chua co file merge cu. Tao file merge moi tu du lieu moi.")
        merged_df = new_data
        print(f"Tong so dong trong file merge moi: {len(merged_df)}")

    try:
        merged_df.to_excel(merge_path, index=False)
        print(f"Da ghi file merge thanh cong tai: {merge_path}")
    except Exception as e:
        print(f"Loi khi ghi file Excel merge: {e}")
        return

    for f in new_files:
        processed_files.add(f.name)
    save_processed_files(log_path, processed_files)
    print(f"Da cap nhat log voi {len(new_files)} file moi.")

    print("======== KET THUC MERGE =========")


if __name__ == "__main__":
    # Xác định BASE_DIR từ vị trí script
    # Script nằm trong thư mục scripts, BASE_DIR là thư mục cha (Weather_Forcast_App)
    SCRIPT_DIR = Path(__file__).parent
    BASE_DIR = SCRIPT_DIR.parent  # Lên một cấp thư mục
    
    print(f"Script dir: {SCRIPT_DIR}")
    print(f"Base dir: {BASE_DIR}")
    
    # Kiểm tra thư mục output có tồn tại không
    output_dir = BASE_DIR / OUTPUT_DIR_NAME
    if not output_dir.exists():
        print(f"ERROR: Khong tim thay thu muc output tai: {output_dir}")
        print(f"Vui long tao thu muc: {output_dir}")
        sys.exit(1)
    
    merge_excel_files_once(BASE_DIR)