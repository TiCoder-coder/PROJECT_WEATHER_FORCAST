import pandas as pd
import numpy as np

SRC = "data/data_clean/data_merge_clean/cleaned_merge_merged_vrain_data_20260214_230945.schema_en.csv"
DST = "data/data_clean/data_merge_clean/cleaned_merge_merged_vrain_data_20260214_230945.schema_en.clean_target.csv"

def main():
    df = pd.read_csv(SRC)
    # Loại bỏ các dòng target NaN, rỗng, không phải số, hoặc vô cực
    df = df[pd.to_numeric(df['tong_luong_mua'], errors='coerce').notnull()]
    df = df[~df['tong_luong_mua'].isin([np.inf, -np.inf])]
    print(f"Số dòng hợp lệ còn lại: {len(df)}")
    df.to_csv(DST, index=False)
    print(f"Đã ghi file: {DST} (lọc target hợp lệ)")

if __name__ == "__main__":
    main()
