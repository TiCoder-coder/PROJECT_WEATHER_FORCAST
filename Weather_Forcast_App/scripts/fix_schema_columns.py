import pandas as pd
import numpy as np

SRC = "data/data_clean/data_merge_clean/cleaned_merge_merged_vrain_data_20260214_230945.schema_en.clean_target.csv"
DST = "data/data_clean/data_merge_clean/cleaned_merge_merged_vrain_data_20260214_230945.schema_en.schema_fixed.csv"

COLUMN_MAP = {
    # Location
    "ma_tram": "ma_tram",
    "ten_tram": "ten_tram",
    "tinh_thanh_pho": "tinh_thanh_pho",
    "huyen": "huyen",
    "vi_do": "vi_do",
    "kinh_do": "kinh_do",
    # Time/Meta
    "dau_thoi_gian": "dau_thoi_gian",
    "nguon_du_lieu": "nguon_du_lieu",
    "chat_luong_du_lieu": "chat_luong_du_lieu",
    "thoi_gian_cap_nhat": "thoi_gian_cap_nhat",
    # Weather metrics (Vietnamese to schema)
    "Nhiệt độ hiện tại": "nhiet_do_hien_tai",
    "Nhiệt độ tối đa": "nhiet_do_toi_da",
    "Nhiệt độ tối thiểu": "nhiet_do_toi_thieu",
    "Nhiệt độ trung bình": "nhiet_do_trung_binh",
    "Độ ẩm hiện tại": "do_am_hien_tai",
    "Độ ẩm tối đa": "do_am_toi_da",
    "Độ ẩm tối thiểu": "do_am_toi_thieu",
    "Độ ẩm trung bình": "do_am_trung_binh",
    "Áp suất hiện tại": "ap_suat_hien_tai",
    "Áp suất tối đa": "ap_suat_toi_da",
    "Áp suất tối thiểu": "ap_suat_toi_thieu",
    "Áp suất trung bình": "ap_suat_trung_binh",
    "Tốc độ gió hiện tại": "toc_do_gio_hien_tai",
    "Tốc độ gió tối đa": "toc_do_gio_toi_da",
    "Tốc độ gió tối thiểu": "toc_do_gio_toi_thieu",
    "Tốc độ gió trung bình": "toc_do_gio_trung_binh",
    "Hướng gió hiện tại": "huong_gio_hien_tai",
    "Hướng gió trung bình": "huong_gio_trung_binh",
    "Lượng mưa hiện tại": "luong_mua_hien_tai",
    "Lượng mưa tối đa": "luong_mua_toi_da",
    "Lượng mưa tối thiểu": "luong_mua_toi_thieu",
    "Lượng mưa trung bình": "luong_mua_trung_binh",
    "tong_luong_mua": "tong_luong_mua",
    "Độ che phủ mây hiện tại": "do_che_phu_may_hien_tai",
    "Độ che phủ mây tối đa": "do_che_phu_may_toi_da",
    "Độ che phủ mây tổi thiểu": "do_che_phu_may_toi_thieu",
    "Độ che phủ mây trung bình": "do_che_phu_may_trung_binh",
    "Tầm nhìn hiện tại": "tam_nhin_hien_tai",
    "Tầm nhìn đa": "tam_nhin_toi_da",
    "Tầm nhìn tối thiểu": "tam_nhin_toi_thieu",
    "Tầm nhìn trung bình": "tam_nhin_trung_binh",
    "Xác xuất sấm sét": "xac_suat_sam_set",
    # Các cột khác giữ nguyên
}

def main():
    df = pd.read_csv(SRC)
    df = df.rename(columns=COLUMN_MAP)
    print(f"Cột sau khi đổi tên: {list(df.columns)}")
    df.to_csv(DST, index=False)
    print(f"Đã ghi file: {DST} (chuẩn schema)")

if __name__ == "__main__":
    main()
