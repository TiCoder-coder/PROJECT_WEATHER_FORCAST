import pandas as pd

SRC = "data/data_clean/data_merge_clean/cleaned_merge_merged_vrain_data_20260214_230945.schema_en.clean_target.csv"

try:
    df = pd.read_csv(SRC)
    print(f"Số dòng: {len(df)}")
    print(f"Các cột: {list(df.columns)}")
    print(df.head())
except Exception as e:
    print(f"Lỗi khi đọc file: {e}")
