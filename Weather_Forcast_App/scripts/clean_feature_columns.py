import pandas as pd
import numpy as np

SRC = "data/data_clean/data_merge_clean/cleaned_merge_merged_vrain_data_20260214_230945.schema_en.clean_target.csv"
DST = "data/data_clean/data_merge_clean/cleaned_merge_merged_vrain_data_20260214_230945.schema_en.clean_target_features.csv"

def main():
    try:
        df = pd.read_csv(SRC)
        print(f"Số dòng đầu vào: {len(df)}")
        feature_cols = [col for col in df.columns if col != 'tong_luong_mua']
        print(f"Các cột feature: {feature_cols}")
        # Lọc NaN, inf, -inf
        mask = (~df[feature_cols].isnull().any(axis=1)) & \
               (~df[feature_cols].isin([np.inf, -np.inf]).any(axis=1))
        # Loại bỏ giá trị quá lớn (|x| > 1e10)
        mask = mask & (df[feature_cols].abs().le(1e10).all(axis=1))
        cleaned = df[mask].copy()
        print(f"Số dòng hợp lệ sau khi lọc feature: {len(cleaned)}")
        cleaned.to_csv(DST, index=False)
        print(f"Đã ghi file: {DST} (lọc feature hợp lệ)")
    except Exception as e:
        print(f"Lỗi khi lọc feature: {e}")

if __name__ == "__main__":
    main()
