import json
import math
import os

INPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../output/merged_vrain_data.weather_schema.json'))
OUTPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../output/merged_vrain_data.weather_schema.cleaned.json'))
ERROR_LOG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../output/cleaned_error_records.json'))

REQUIRED_LOCATION = ['ma_tram', 'ten_tram', 'tinh_thanh_pho', 'huyen', 'vi_do', 'kinh_do']

def is_valid_location(loc):
    # location phải là dict và đủ key
    if not isinstance(loc, dict):
        return False
    if set(loc.keys()) != set(REQUIRED_LOCATION):
        return False
    for k in REQUIRED_LOCATION:
        v = loc.get(k, None)
        # Loại bỏ nếu value là None
        if v is None:
            return False
        # Loại bỏ nếu value là float('nan') hoặc 0
        if isinstance(v, float):
            if math.isnan(v) or v == 0.0:
                return False
        # Loại bỏ nếu value là int 0
        if isinstance(v, int):
            if v == 0:
                return False
        # Loại bỏ nếu value là chuỗi rỗng, 'nan', '0'
        if isinstance(v, str):
            if v.strip() == '' or v.strip().lower() == 'nan' or v.strip() == '0':
                return False
    return True

def main():
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    cleaned = []
    dropped = 0
    error_records = []
    for rec in data:
        # Loại bỏ record nếu không có trường location hoặc location không hợp lệ
        error_reason = None
        if 'location' not in rec:
            error_reason = 'missing_location_field'
        else:
            loc = rec['location']
            if not is_valid_location(loc):
                error_reason = 'invalid_location_schema'
        if not error_reason and not rec.get('dau_thoi_gian'):
            error_reason = 'missing_dau_thoi_gian'
        if error_reason:
            dropped += 1
            error_records.append({'error': error_reason, 'record': rec})
            continue
        cleaned.append(rec)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    with open(ERROR_LOG_PATH, 'w', encoding='utf-8') as f:
        json.dump(error_records, f, ensure_ascii=False, indent=2)
    print(f'Đã ghi file cleaned: {OUTPUT_PATH}, số record hợp lệ: {len(cleaned)}, số record bị loại: {dropped}')
    print(f'Đã ghi log lỗi: {ERROR_LOG_PATH}, số record lỗi: {len(error_records)}')

if __name__ == '__main__':
    main()
