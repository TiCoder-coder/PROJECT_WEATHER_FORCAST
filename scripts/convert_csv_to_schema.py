import pandas as pd
from pathlib import Path
from datetime import datetime
import json

def convert_row(row):
    # Chuyển đổi 1 dòng từ CSV thô sang dict đúng schema
    # Mapping các trường location
    location = {
        'ma_tram': row.get('Tên trạm', ''),
        'ten_tram': row.get('Tên trạm', ''),
        'tinh_thanh_pho': row.get('Tỉnh/Thành phố', ''),
        'huyen': row.get('Huyện', ''),
        'vi_do': 0.0,  # Giá trị mặc định, cần cập nhật nếu có
        'kinh_do': 0.0  # Giá trị mặc định, cần cập nhật nếu có
    }
    # Thời gian
    try:
        dau_thoi_gian = datetime.strptime(row.get('Dấu thời gian', ''), '%d/%m/%Y %H:%M').isoformat()
    except Exception:
        dau_thoi_gian = datetime.now().isoformat()
    try:
        thoi_gian_cap_nhat = datetime.strptime(row.get('Thời gian cập nhập', ''), '%d/%m/%Y %H:%M:%S').isoformat()
    except Exception:
        thoi_gian_cap_nhat = datetime.now().isoformat()
    # Metadata
    nguon_du_lieu = 'vrain'
    chat_luong_du_lieu = 'high'
    # Metrics
    try:
        tong_luong_mua = float(row.get('Tổng lượng mưa', 0))
    except Exception:
        tong_luong_mua = 0.0
    metrics = {
        'tong_luong_mua': tong_luong_mua,
        'luong_mua_hien_tai': tong_luong_mua
    }
    return {
        'location': location,
        'dau_thoi_gian': dau_thoi_gian,
        'thoi_gian_cap_nhat': thoi_gian_cap_nhat,
        'nguon_du_lieu': nguon_du_lieu,
        'chat_luong_du_lieu': chat_luong_du_lieu,
        'metrics': metrics
    }

def convert_csv_to_schema(input_csv, output_jsonl):
    df = pd.read_csv(input_csv)
    records = [convert_row(row) for _, row in df.iterrows()]
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f"✅ Đã chuyển đổi {len(records)} bản ghi sang {output_jsonl}")

if __name__ == '__main__':
    input_csv = 'data/data_crawl/Bao_cao_20260211_173719.csv'
    output_jsonl = 'data/data_crawl/Bao_cao_20260211_173719.schema.jsonl'
    convert_csv_to_schema(input_csv, output_jsonl)
