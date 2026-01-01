import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Fix encoding cho Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


MERGE_DIR_NAME = "Merge_data"
OUTPUT_DIR_NAME = "output"
MERGE_FILENAME = "merged_vrain_data.xlsx"
MERGE_VIETNAM_FILENAME = "merged_vietnam_weather_data.xlsx"
LOG_FILENAME = "merged_files_log.txt"
LOG_VIETNAM_FILENAME = "merged_vietnam_files_log.txt"


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


def read_excel_file(file_path: Path) -> pd.DataFrame:
    """Đọc file Excel với exception handling"""
    try:
        return pd.read_excel(file_path)
    except Exception as e:
        print(f"Loi khi doc file {file_path.name}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error


def get_new_excel_files(output_dir: Path, processed_files: set[str]) -> tuple[list[Path], list[Path]]:
    """
    Lấy danh sách các file .xlsx MỚI trong thư mục output
    và phân loại thành 2 nhóm: vietnam_weather_ và các file còn lại.
    """
    if not output_dir.exists():
        print(f"Thu muc nguon khong ton tai: {output_dir}")
        return [], []

    all_excel_files = sorted(output_dir.glob("*.xlsx"))
    if not all_excel_files:
        print(f"Khong tim thay file .xlsx nao trong thu muc: {output_dir}")
        return [], []

    # Phân loại file
    vietnam_files = []
    other_files = []
    
    for file_path in all_excel_files:
        if file_path.name not in processed_files:
            if file_path.name.startswith("vietnam_weather_"):
                vietnam_files.append(file_path)
            else:
                other_files.append(file_path)

    print(f"Tong so file .xlsx trong output: {len(all_excel_files)}")
    print(f"So file vietnam_weather_ moi: {len(vietnam_files)}")
    print(f"So file khac moi: {len(other_files)}")

    return vietnam_files, other_files


def merge_files_concurrently(file_list: list[Path], max_workers: int = 4) -> pd.DataFrame:
    """Đọc nhiều file Excel song song để tăng tốc độ"""
    dfs = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all reading tasks
        future_to_file = {
            executor.submit(read_excel_file, file_path): file_path 
            for file_path in file_list
        }
        
        # Process completed tasks
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                df = future.result()
                if not df.empty:
                    dfs.append(df)
                    print(f"✓ Da doc xong: {file_path.name}")
            except Exception as e:
                print(f"✗ Loi khi xu ly file {file_path.name}: {e}")
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def merge_single_category(file_list: list[Path], merge_path: Path, 
                         log_path: Path, processed_files: set[str],
                         category_name: str = "du lieu") -> None:
    """
    Merge một loại file cụ thể (vietnam_weather_ hoặc các file khác)
    """
    if not file_list:
        print(f"Khong co file {category_name} moi de merge.")
        return
    
    print(f"\n=== MERGE FILE {category_name.upper()} ===")
    print(f"So luong file: {len(file_list)}")
    
    # Đọc file song song để tăng tốc
    new_data = merge_files_concurrently(file_list)
    
    if new_data.empty:
        print(f"Khong doc duoc du lieu hop le tu cac file {category_name} moi.")
        return
    
    print(f"Tong so dong du lieu moi: {len(new_data)}")
    
    # Xử lý file merge cũ (nếu có)
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
        print(f"Chua co file merge cu. Tao file merge moi tu du lieu {category_name}.")
        merged_df = new_data
        print(f"Tong so dong trong file merge moi: {len(merged_df)}")
    
    # Lưu file merge
    try:
        merged_df.to_excel(merge_path, index=False)
        print(f"Da ghi file merge thanh cong tai: {merge_path}")
    except Exception as e:
        print(f"Loi khi ghi file Excel merge: {e}")
        return
    
    # Cập nhật log
    for f in file_list:
        processed_files.add(f.name)
    save_processed_files(log_path, processed_files)
    print(f"Da cap nhat log voi {len(file_list)} file {category_name} moi.")


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
    
    # Đường dẫn cho file vietnam_weather_
    merge_vietnam_path = merge_dir / MERGE_VIETNAM_FILENAME
    log_vietnam_path = merge_dir / LOG_VIETNAM_FILENAME
    
    # Đường dẫn cho các file khác
    merge_other_path = merge_dir / MERGE_FILENAME
    log_other_path = merge_dir / LOG_FILENAME
    
    print("======== BAT DAU MERGE =========")
    print(f"Thu muc nguon (output):    {output_dir}")
    print(f"Thu muc merge (Merge_data): {merge_dir}")
    print("================================")
    
    # Đọc log file đã xử lý
    processed_vietnam = load_processed_files(log_vietnam_path)
    processed_other = load_processed_files(log_other_path)
    
    if processed_vietnam:
        print(f"Da tung merge {len(processed_vietnam)} file vietnam_weather_ truoc do.")
    if processed_other:
        print(f"Da tung merge {len(processed_other)} file khac truoc do.")
    
    # Lấy danh sách file mới (đã phân loại)
    vietnam_files, other_files = get_new_excel_files(output_dir, 
                                                    processed_vietnam.union(processed_other))
    
    # Merge file vietnam_weather_ riêng
    merge_single_category(vietnam_files, merge_vietnam_path, 
                         log_vietnam_path, processed_vietnam,
                         "vietnam_weather_")
    
    # Merge các file khác
    merge_single_category(other_files, merge_other_path,
                         log_other_path, processed_other,
                         "khac")
    
    print("======== KET THUC MERGE =========")


if __name__ == "__main__":
    # Xác định BASE_DIR từ vị trí script
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
