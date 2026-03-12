import pandas as pd
import json
import os

# Đường dẫn file excel gốc
EXCEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/data_merge/merged_vrain_data.xlsx'))
# Đường dẫn file json output
OUTPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/data_crawl/merged_vrain_data.schema.json'))

# Mapping tên cột gốc sang schema chuẩn
COLUMN_MAP = {
    'Mã trạm': 'ma_tram',
    'Tên trạm': 'ten_tram',
    'Tỉnh/Thành huyện': 'tinh_thanh_pho',
    'Tỉnh/Thành phố': 'tinh_thanh_pho',
    'Huyện': 'huyen',
    'Vĩ độ': 'vi_do',
    'Kinh độ': 'kinh_do',
}

# Các trường bắt buộc theo LocationSchema
REQUIRED_FIELDS = ['ma_tram', 'ten_tram', 'tinh_thanh_pho', 'huyen', 'vi_do', 'kinh_do']

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
    df = pd.read_excel(EXCEL_PATH, dtype=str)
    # Đổi tên cột về chuẩn schema
    df = df.rename(columns={col: normalize_column(col) for col in df.columns})
    # Chỉ lấy các trường cần thiết
    df = df[[f for f in REQUIRED_FIELDS if f in df.columns]]
    # Loại bỏ dòng thiếu trường bắt buộc
    df = df.dropna(subset=REQUIRED_FIELDS)
    # Chuyển kiểu số
    for col in ['vi_do', 'kinh_do']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Loại bỏ dòng có vi_do, kinh_do không hợp lệ
    df = df.dropna(subset=['vi_do', 'kinh_do'])
    # Ghi ra file JSON array
    records = df.to_dict(orient='records')
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f'Đã ghi file JSON schema: {OUTPUT_PATH}')

if __name__ == '__main__':
    main()
