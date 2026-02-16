import pandas as pd
import json
import os
from datetime import datetime

def safe_parse_datetime(val):
    # Thử parse datetime, nếu lỗi trả về None
    try:
        if pd.isnull(val):
            return None
        # Ưu tiên ISO, nếu không thì thử các định dạng phổ biến
        for fmt in ("%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%d/%m/%Y"):
            try:
                return datetime.strptime(str(val), fmt).isoformat()
            except Exception:
                continue
        return str(val)
    except Exception:
        return None

# Đường dẫn file excel gốc
EXCEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/data_merge/merged_vrain_data.xlsx'))
# Đường dẫn file json output
OUTPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../output/merged_vrain_data.weather_schema.json'))

# Mapping tên cột gốc sang schema chuẩn
COLUMN_MAP = {
    'Mã trạm': 'ma_tram',
    'Tên trạm': 'ten_tram',
    'Tỉnh/Thành huyện': 'tinh_thanh_pho',
    'Tỉnh/Thành phố': 'tinh_thanh_pho',
    'Huyện': 'huyen',
    'Vĩ độ': 'vi_do',
    'Kinh độ': 'kinh_do',
    'Dấu thời gian': 'dau_thoi_gian',
    'Thời gian': 'dau_thoi_gian',
    'Thời gian cập nhập': 'thoi_gian_cap_nhat',
    'Nguồn dữ': 'nguon_du_lieu',
    'Chất lượng': 'chat_luong_du_lieu',
}

LOCATION_FIELDS = ['ma_tram', 'ten_tram', 'tinh_thanh_pho', 'huyen', 'vi_do', 'kinh_do']
META_FIELDS = ['dau_thoi_gian', 'thoi_gian_cap_nhat', 'nguon_du_lieu', 'chat_luong_du_lieu']

# Các trường metrics phổ biến (có thể mở rộng)
METRICS_FIELDS = [
    'tong_luong_mua', 'nhiet_do_hien_tai', 'nhiet_do_toi_da', 'nhiet_do_toi_thieu', 'nhiet_do_trung_binh',
    'do_am_hien_tai', 'do_am_toi_da', 'do_am_toi_thieu', 'do_am_trung_binh',
    'ap_suat_hien_tai', 'ap_suat_toi_da', 'ap_suat_toi_thieu', 'ap_suat_trung_binh',
    'toc_do_gio_hien_tai', 'toc_do_gio_toi_da', 'toc_do_gio_toi_thieu', 'toc_do_gio_trung_binh',
    'huong_gio_hien_tai', 'huong_gio_trung_binh',
    'luong_mua_hien_tai', 'luong_mua_toi_da', 'luong_mua_toi_thieu', 'luong_mua_trung_binh',
    'do_che_phu_may_hien_tai', 'do_che_phu_may_toi_da', 'do_che_phu_may_toi_thieu', 'do_che_phu_may_trung_binh',
    'tam_nhin_hien_tai', 'tam_nhin_toi_da', 'tam_nhin_toi_thieu', 'tam_nhin_trung_binh',
    'xac_suat_sam_set'
]

def normalize_column(col):
    col = col.strip()
    if col in COLUMN_MAP:
        return COLUMN_MAP[col]
    # fallback: snake_case, không dấu
    return (
        col.lower()
        .replace(' ', '_')
        .replace('/', '_')
        .replace('-', '_')
        .replace('đ', 'd')
        .replace('Đ', 'D')
    )

def main():
    df = pd.read_excel(EXCEL_PATH)
    df = df.rename(columns={col: normalize_column(col) for col in df.columns})
    records = []
    for idx, row in df.iterrows():
        # Tạo location
        location = {k: row.get(k, None) for k in LOCATION_FIELDS}
        # Chuyển kiểu số
        for k in ['vi_do', 'kinh_do']:
            if location[k] is not None:
                try:
                    location[k] = float(location[k])
                except Exception:
                    location[k] = None
        # Meta
        dau_thoi_gian = safe_parse_datetime(row.get('dau_thoi_gian'))
        thoi_gian_cap_nhat = safe_parse_datetime(row.get('thoi_gian_cap_nhat'))
        nguon_du_lieu = row.get('nguon_du_lieu', 'vrain') or 'vrain'
        chat_luong_du_lieu = row.get('chat_luong_du_lieu', 'high') or 'high'
        # Metrics
        metrics = {k: row.get(k, None) for k in METRICS_FIELDS}
        # Chuyển kiểu số cho metrics
        for k in metrics:
            if metrics[k] is not None:
                try:
                    metrics[k] = float(metrics[k])
                except Exception:
                    metrics[k] = None
        # In log debug từng dòng
        print(f"Row {idx}: location={location}, dau_thoi_gian={dau_thoi_gian}, thoi_gian_cap_nhat={thoi_gian_cap_nhat}")
        # Nới lỏng điều kiện lọc: chỉ bỏ nếu thiếu hoàn toàn location (ma_tram, ten_tram, ...) hoặc thiếu cả hai trường thời gian
        if not location['ma_tram'] or not location['ten_tram'] or not location['tinh_thanh_pho']:
            print(f"Bỏ qua dòng {idx} do thiếu location chính!")
            continue
        # Nếu thiếu thời gian, để None
        record = {
            'location': location,
            'dau_thoi_gian': dau_thoi_gian,
            'thoi_gian_cap_nhat': thoi_gian_cap_nhat,
            'nguon_du_lieu': nguon_du_lieu,
            'chat_luong_du_lieu': chat_luong_du_lieu,
            'metrics': metrics
        }
        records.append(record)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f'Đã ghi file WeatherDataSchema JSON: {OUTPUT_PATH}')

if __name__ == '__main__':
    main()
