from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd

@dataclass
class SplitConfig:
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10

    shuffle: bool = False

    sort_by_time_if_possible: bool = True # Sắp xếp các dữ liệu theo cột thời gian

    time_col_candidates: Tuple[str, ...] = ("timestamp",)

# Hàm dùng để tìm cột thời gian
def _find_time_col(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:

    # Tạo map đẻ tìm cột thời gian -- nếu không có cột thời gian thì tìm một cột nào đó đẻ sắp xếp
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]

    # Đoán tìm cột thời gian
    for c in df.columns:
        if df[c].dtype == "object": # Cột thưởng là kiểu object
            sample = df[c].dropna().astype(str).head(20) # Láy tối đa 20 dòng
            if sample.empty:
                continue
            parsed = pd.to_datetime(sample, errors="coerce", utc=False) ## Parse thử
            ok = parsed.notna().mean()
            if ok >= 0.7: # Nếu tỉ lệ thành công lớn hơn 70% là ok
                return c
    return None


# Nếu có cột thời gian thì sort theo cột thời gian
def _maybe_sort_time_series(df: pd.DataFrame, cfg: SplitConfig) -> pd.DataFrame:
    if not cfg.sort_by_time_if_possible:
        return df

    time_col = _find_time_col(df, cfg.time_col_candidates)
    if time_col is None:
        return df

    # parse + sort
    tmp = df.copy()
    tmp[time_col] = pd.to_datetime(tmp[time_col], errors="coerce")
    tmp = tmp.sort_values(by=time_col, kind="mergesort")  # stable sort
    tmp = tmp.reset_index(drop=True)
    return tmp

# Tính toán dữ liệu ra cho train/ test/ validate
def _compute_split_sizes(n: int, cfg: SplitConfig) -> Tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0

    # ưu tiên time-series: train (đầu) -> val (giữa) -> test (cuối, mới nhất)
    n_train = int(n * cfg.train_ratio)
    n_val = int(n * cfg.val_ratio)
    n_test = n - n_train - n_val

    # đảm bảo không âm
    if n_test < 0:
        n_test = 0

    # nếu dataset quá nhỏ, cố gắng chia tối thiểu hợp lý
    # n>=3 => mỗi tập ít nhất 1 dòng
    if n >= 3:
        if n_train == 0:
            n_train = 1
        if n_val == 0:
            n_val = 1
        if n_test == 0:
            n_test = 1
        # chỉnh lại cho đúng tổng
        total = n_train + n_val + n_test
        if total > n:
            # bớt từ train trước
            overflow = total - n
            n_train = max(1, n_train - overflow)
    else:
        # n < 3: dồn hết vào train cho an toàn
        n_train, n_val, n_test = n, 0, 0

    return n_train, n_val, n_test

# Tiến hành split
def split_dataframe(df: pd.DataFrame, cfg: SplitConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = _maybe_sort_time_series(df, cfg)

    n = len(df)
    n_train, n_val, n_test = _compute_split_sizes(n, cfg)

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:n_train + n_val + n_test].copy()

    return train_df, val_df, test_df

# Tạo thư mục để lưu (nếu chưa có)
def _ensure_dirs(base_out: Path) -> None:
    (base_out / "Train").mkdir(parents=True, exist_ok=True)
    (base_out / "Validate").mkdir(parents=True, exist_ok=True)
    (base_out / "Test").mkdir(parents=True, exist_ok=True)

# Lưu data sau khi đã chia vào các thư mục
def _save_split_files(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                      out_dir: Path, filename: str) -> None:
    _ensure_dirs(out_dir)

    train_path = out_dir / "Train" / filename
    val_path = out_dir / "Validate" / filename
    test_path = out_dir / "Test" / filename

    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

# Tiến hành tất cả các quy trình
def split_folder_to_output(
    input_folder: Path,
    output_folder: Path,
    cfg: SplitConfig,
) -> List[dict]:
    """
    - Đọc tất cả CSV trong input_folder (không đi sâu quá nhiều cấp)
    - Split 80/10/10
    - Ghi ra output_folder/{Train,Validate,Test}/filename.csv
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    _ensure_dirs(output_folder)

    csv_files = sorted(input_folder.glob("*.csv"))
    logs: List[dict] = []

    for f in csv_files:
        df = pd.read_csv(f)
        train_df, val_df, test_df = split_dataframe(df, cfg)
        _save_split_files(train_df, val_df, test_df, output_folder, f.name)

        logs.append({
            "file": str(f),
            "rows_total": int(len(df)),
            "rows_train": int(len(train_df)),
            "rows_validate": int(len(val_df)),
            "rows_test": int(len(test_df)),
            "out_dir": str(output_folder),
        })

    return logs

# Split cho merge và not merge
def run_split_all(
    cleaned_data_root: Path,
    dataset_after_split_root: Path,
    cfg: SplitConfig,
) -> None:
    """
    Cấu trúc input:
      cleaned_data/
        Clean_Data_For_File_Merge/*.csv
        Clean_Data_For_File_Not_Merge/*.csv

    Output:
      Dataset_after_split/
        Dataset_merge/{Train,Validate,Test}/*.csv
        Dataset_not_merge/{Train,Validate,Test}/*.csv
    """
    merge_in = cleaned_data_root / "Clean_Data_For_File_Merge"
    not_merge_in = cleaned_data_root / "Clean_Data_For_File_Not_Merge"

    merge_out = dataset_after_split_root / "Dataset_merge"
    not_merge_out = dataset_after_split_root / "Dataset_not_merge"

    all_logs: List[dict] = []

    if merge_in.exists():
        all_logs += split_folder_to_output(merge_in, merge_out, cfg)
    else:
        print(f"[WARN] Không tìm thấy folder: {merge_in}")

    if not_merge_in.exists():
        all_logs += split_folder_to_output(not_merge_in, not_merge_out, cfg)
    else:
        print(f"[WARN] Không tìm thấy folder: {not_merge_in}")

    # ghi log tổng để sếp kiểm tra nhanh
    dataset_after_split_root.mkdir(parents=True, exist_ok=True)
    log_path = dataset_after_split_root / "split_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(all_logs, f, ensure_ascii=False, indent=2)

    print(f"[OK] Split xong. Log: {log_path}")
    for item in all_logs:
        print(f" - {Path(item['file']).name}: total={item['rows_total']}, "
              f"train={item['rows_train']}, val={item['rows_validate']}, test={item['rows_test']}")

if __name__ == "__main__":
    # Dynamic path: tự tính từ vị trí project root
    _split_project_root = Path(__file__).resolve().parents[3]
    CLEANED_DATA_ROOT = Path(
        _split_project_root / "data" / "data_clean"
    )
    OUT_ROOT = Path(
        _split_project_root / "Weather_Forcast_App" / "Machine_learning_model" / "Dataset_after_split"
    )

    cfg = SplitConfig(train_ratio=0.80, val_ratio=0.10, test_ratio=0.10, shuffle=False)
    run_split_all(CLEANED_DATA_ROOT, OUT_ROOT, cfg)
