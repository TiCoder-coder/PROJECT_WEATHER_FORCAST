def main():
    import pandas as pd
    import numpy as np
    SRC = "data/data_clean/data_merge_clean/cleaned_merge_merged_vrain_data_20260214_230945.schema_en.schema_fixed.csv"
    DST = "data/data_clean/data_merge_clean/cleaned_merge_merged_vrain_data_20260214_230945.schema_en.schema_fixed.clean.csv"
    df = pd.read_csv(SRC)
    feature_cols = [col for col in df.columns if col != 'tong_luong_mua']
    # Lọc NaN, inf, -inf
    mask = (~df[feature_cols].isnull().any(axis=1)) & \
           (~df[feature_cols].isin([np.inf, -np.inf]).any(axis=1))
    # Chỉ kiểm tra giá trị lớn với cột số
    num_cols = [col for col in feature_cols if np.issubdtype(df[col].dtype, np.number)]
    if num_cols:
        mask = mask & (df[num_cols].abs().le(1e10).all(axis=1))
    cleaned = df[mask].copy()
    print(f"Số dòng hợp lệ sau khi lọc feature: {len(cleaned)}")
    cleaned.to_csv(DST, index=False)
    print(f"Đã ghi file: {DST} (lọc feature hợp lệ)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        with open("data/data_clean/data_merge_clean/clean_feature_columns_final.log.txt", "w") as f:
            f.write(f"Lỗi: {e}\n")
        print(f"Lỗi khi chạy script: {e}")
