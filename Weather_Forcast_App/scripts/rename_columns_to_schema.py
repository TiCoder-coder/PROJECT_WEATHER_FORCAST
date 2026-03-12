import pandas as pd
import os

# Đường dẫn file gốc và file mới
SRC = "data/data_clean/data_merge_clean/cleaned_merge_merged_vrain_data_20260214_230945.csv"
DST = "data/data_clean/data_merge_clean/cleaned_merge_merged_vrain_data_20260214_230945.schema_en.csv"

# Mapping tên cột tiếng Việt sang schema LocationSchema/code
COLUMN_MAP = {
    "Mã trạm": "ma_tram",
    "Tên trạm": "ten_tram",
    "Tỉnh/Thành phố": "tinh_thanh_pho",
    "Huyện": "huyen",
    "Vĩ độ": "vi_do",
    "Kinh độ": "kinh_do",
    "Dấu thời gian": "dau_thoi_gian",
    "Nguồn dữ liệu": "nguon_du_lieu",
    "Chất lượng dữ liệu": "chat_luong_du_lieu",
    "Thời gian cập nhật": "thoi_gian_cap_nhat",
    "Tổng lượng mưa": "tong_luong_mua",
    # Các trường metrics giữ nguyên hoặc map thêm nếu cần
}

def main():
    df = pd.read_csv(SRC)
    df = df.rename(columns=COLUMN_MAP)
    df.to_csv(DST, index=False)
    print(f"Đã ghi file: {DST} với cột chuẩn schema.")

if __name__ == "__main__":
    main()
