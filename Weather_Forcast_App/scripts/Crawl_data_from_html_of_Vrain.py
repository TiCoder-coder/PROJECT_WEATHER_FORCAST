"""
CRAWL_DATA_FROM_HTML_OF_VRAIN.PY
================================

Script crawl lượng mưa từ VRAIN bằng Selenium + Regex cho 5 điểm ĐBSCL.

Phạm vi hiện tại:
    - Crawl cố định 5 URL theo yêu cầu:
      Vĩnh Long, Đồng Tháp, An Giang, Cần Thơ, Cà Mau

Luồng chính:
    1. Mở trang chủ VRAIN để lấy mốc ngày/giờ hiển thị dữ liệu
    2. Crawl từng URL tỉnh bằng Selenium (headless)
    3. Parse HTML để lấy:
       - station_id (đang dùng cùng giá trị với station_name)
       - station_name
       - province
       - district
       - rain_total
       - status
       - timestamp
       - data_time
    4. Ghi file tổng: data/data_crawl/Bao_cao_YYYYMMDD_HHMMSS.csv
    5. Ghi file theo tỉnh vào data/data_crawl/Bao_cao_vrain_DBSCL:
       - Bao_cao_<Tinh>_YYYYMMDD_HHMMSS.csv
       - Bao_cao_<Tinh>_YYYYMMDD_HHMMSS.xlsx
    6. Merge dữ liệu cũ + mới theo từng tỉnh, khử trùng lặp trước khi xuất

Lưu ý:
    - Script đang chạy đa luồng với ThreadPoolExecutor để giảm thời gian crawl.
    - Khi tỉnh không parse được station block, script in HTML snippet để debug.
"""

import re
import csv
import time
import threading
import unicodedata
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed
from openpyxl import Workbook
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

VN_TZ = ZoneInfo("Asia/Bangkok")
MAX_WORKERS = 6
MAX_RETRIES = 2


DBSCL_PROVINCES = {
    "an giang": "An_Giang",
    "ca mau": "Ca_Mau",
    "can tho": "Can_Tho",
    "dong thap": "Dong_Thap",
    "vinh long": "Vinh_Long",
}

# Mapping URL VRAIN → tên tỉnh tiếng Việt đầy đủ (theo thứ tự ảnh)
URL_PROVINCE_MAP = {
    "https://vrain.vn/63/overview?public_map=windy": "Vĩnh Long",
    "https://vrain.vn/54/overview?public_map=windy": "Đồng Tháp",
    "https://vrain.vn/46/overview?public_map=windy": "An Giang",
    "https://vrain.vn/53/overview?public_map=windy": "Cần Thơ",
    "https://vrain.vn/52/overview?public_map=windy": "Cà Mau",
}


def _normalize_ascii_text(value):
    """Chuẩn hóa text tiếng Việt về ASCII thường để so sánh."""
    if not value:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("đ", "d").replace("Đ", "D").lower().strip()
    text = re.sub(r"\b(tinh|tp\.?|thanh pho)\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _canonical_dbscl_province(raw_name):
    """Trả về tên tỉnh ĐBSCL chuẩn nếu thuộc vùng, ngược lại trả None."""
    key = _normalize_ascii_text(raw_name)
    return DBSCL_PROVINCES.get(key)


def _write_csv_rows(csv_path, rows, fieldnames):
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv_rows(csv_path, fieldnames):
    """Đọc dữ liệu CSV cũ và chuẩn hóa theo fieldnames hiện tại."""
    rows = []
    if not csv_path.exists():
        return rows
    try:
        with open(csv_path, "r", newline="", encoding="utf-8-sig") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                rows.append({col: row.get(col, "") for col in fieldnames})
    except Exception as e:
        print(f"  Cảnh báo: không đọc được file cũ {csv_path.name}: {e}")
    return rows


def _merge_rows(existing_rows, new_rows, fieldnames):
    """Merge dữ liệu cũ + mới, khử trùng lặp theo toàn bộ cột output."""
    merged = []
    seen = set()
    for row in existing_rows + new_rows:
        normalized = {col: str(row.get(col, "")).strip() for col in fieldnames}
        row_key = tuple(normalized[col] for col in fieldnames)
        if row_key in seen:
            continue
        seen.add(row_key)
        merged.append(normalized)
    return merged


def _write_xlsx_rows(xlsx_path, rows, fieldnames):
    wb = Workbook()
    ws = wb.active
    ws.title = "rain_data"
    ws.append(fieldnames)
    for row in rows:
        ws.append([row.get(col, "") for col in fieldnames])
    wb.save(xlsx_path)


def _create_driver():
    """Tạo Chrome driver với options tối ưu"""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-logging")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-notifications")
    options.add_argument("--blink-settings=imagesEnabled=false")
    options.add_argument("--window-size=1280,720")
    options.page_load_strategy = "eager"
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(60)
    driver.implicitly_wait(3)
    return driver


def _crawl_single_province(url, unified_datetime_info, current_crawl_datetime, results, seen_lock, seen_stations, retry=0, province_name_override=None):
    """Crawl 1 tỉnh, trả về list rows."""
    driver = None
    rows = []
    try:
        driver = _create_driver()
        print(f"\nĐang truy cập: {url}")
        driver.get(url)
        # Chờ Angular render xong: đợi phần tử chứa dữ liệu trạm xuất hiện
        # body luôn present ngay lập tức nên không dùng làm điều kiện chờ
        try:
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR,
                    "[class*='max-w-70'], [class*='station-row'], [class*='group']"))
            )
        except TimeoutException:
            print(f"  Cảnh báo: timeout chờ station selector tại {url}, thử parse HTML hiện tại")
        # Thêm thời gian để Angular render đầy đủ
        time.sleep(3)

        page_html = driver.page_source

        # === TRÍCH XUẤT DỮ LIỆU ===
        if province_name_override:
            province_name = province_name_override
        else:
            province_match = re.search(
                r"<div[^>]*app-title[^>]*>.*?<span[^>]*>([^<]+)</span>",
                page_html,
                re.DOTALL,
            )
            province_name = (
                province_match.group(1).strip() if province_match else "Không xác định"
            )
        print(f"  Tỉnh: {province_name}")

        datetime_info = unified_datetime_info

        station_blocks = re.findall(
            r'<div[^>]*class="[^"]*\bgroup\b[^"]*"[^>]*>(.*?)</div>\s*</div>\s*</div>',
            page_html,
            re.DOTALL,
        )
        if not station_blocks:
            station_blocks = re.findall(
                r'<div[^>]*class="[^"]*\bstation\b[^"]*"[^>]*>(.*?)</div>\s*</div>\s*</div>',
                page_html,
                re.DOTALL,
            )

        print(f"  Tìm thấy {len(station_blocks)} khối trạm")
        if len(station_blocks) == 0:
            # Debug: lưu snippet HTML để chẩn đoán khi 0 trạm
            snippet = page_html[:3000] if page_html else "(trống)"
            print(f"  [DEBUG] HTML snippet:\n{snippet}\n  --- end snippet ---")

        for block in station_blocks:
            station_match = re.search(
                r'<div[^>]*station-row-1[^>]*>.*?<span[^>]*class="[^"]*\bmax-w-70\b[^"]*"[^>]*>([^<]+)</span>',
                block,
                re.DOTALL,
            )
            station_name = (
                station_match.group(1).strip() if station_match else "N/A"
            )

            location_match = re.search(
                r'<div[^>]*station-row-2[^>]*>.*?<div[^>]*class="[^"]*\bsub-title\b[^"]*"[^>]*>([^<]+)</div>',
                block,
                re.DOTALL,
            )
            xa_phuong = location_match.group(1).strip() if location_match else "N/A"

            rainfall_match = re.search(
                r'<div[^>]*station-row-1[^>]*>.*?<span[^>]*class="[^"]*font-size-18px[^"]*"[^>]*>([\d.]+)\s*<span[^>]*>mm</span>',
                block,
                re.DOTALL,
            )
            rainfall = rainfall_match.group(1).strip() if rainfall_match else "0.0"

            status_match = re.search(
                r'<div[^>]*station-row-2[^>]*>.*?<div[^>]*class="[^"]*\blevel\b[^"]*"[^>]*>.*?<span[^>]*>([^<]+)</span>',
                block,
                re.DOTALL,
            )
            status = (
                status_match.group(1).strip() if status_match else "Không xác định"
            )

            unique_key = f"{province_name}_{station_name}".lower()
            with seen_lock:
                if unique_key in seen_stations:
                    continue
                seen_stations.add(unique_key)

            rows.append({
                "station_id": station_name,
                "station_name": station_name,
                "province": province_name,
                "district": xa_phuong,
                "rain_total": rainfall,
                "status": status,
                "timestamp": datetime_info,
                "data_time": current_crawl_datetime,
            })

        print(f"  Đã trích xuất {len(rows)} trạm từ {province_name}")

    except (TimeoutException, WebDriverException) as e:
        print(f"  Lỗi kết nối {url}: {e}")
        if retry < MAX_RETRIES:
            print(f"  🔄 Retry lần {retry + 1}...")
            if driver:
                try: driver.quit()
                except: pass
            time.sleep(1)
            return _crawl_single_province(url, unified_datetime_info, current_crawl_datetime, results, seen_lock, seen_stations, retry + 1, province_name_override)
    except Exception as e:
        print(f"  Lỗi khi xử lý {url}: {e}")
    finally:
        if driver:
            try: driver.quit()
            except: pass

    with seen_lock:
        results.extend(rows)
    return rows


def main():
    # === 1. LẤY NGÀY VÀ GIỞ CẬP NHẬT TỪ TRANG CHỦ ===
    print("Đang truy cập trang chủ để lấy ngày và giờ cập nhật...")
    driver = _create_driver()
    driver.get("https://vrain.vn/landing")
    time.sleep(3)

    all_text = driver.find_element(By.TAG_NAME, "body").text
    print("  Đang tìm kiếm ngày và giờ trong văn bản trang...")

    date_match = re.search(r"ngày\s*(\d{1,2}/\d{1,2})", all_text)
    hour_match = re.search(r"Tính từ\s*(\d{1,2})h", all_text)

    if date_match and hour_match:
        date_from_main = date_match.group(1)
        hour_from_main = hour_match.group(1)
        current_year = datetime.now(VN_TZ).strftime("%Y")
        unified_datetime_info = f"{date_from_main}/{current_year} {hour_from_main}:00"
        print(f"  Đã lấy ngày và giờ cập nhật từ trang chủ: {unified_datetime_info}")
    elif date_match:
        date_from_main = date_match.group(1)
        current_year = datetime.now(VN_TZ).strftime("%Y")
        unified_datetime_info = f"{date_from_main}/{current_year}"
        print(
            f"  Đã lấy ngày cập nhật từ trang chủ (không có giờ): {unified_datetime_info}"
        )
    else:
        unified_datetime_info = "N/A"
        print("  Cảnh báo: Không tìm thấy ngày cập nhật. Sử dụng ngày và giờ hiện tại.")
        unified_datetime_info = datetime.now(VN_TZ).strftime("%d/%m/%Y %H:%M")

    try:
        driver.quit()
    except Exception:
        pass

    current_crawl_datetime = datetime.now(VN_TZ).strftime("%d/%m/%Y %H:%M:%S")

    # URL crawl cố định – thứ tự theo ảnh: Vĩnh Long, Đồng Tháp, An Giang, Cần Thơ, Cà Mau
    province_urls = list(URL_PROVINCE_MAP.keys())

    # Dynamic path: tự tính từ vị trí project root
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    OUTPUT_DIR = _PROJECT_ROOT / "data" / "data_crawl"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(VN_TZ).strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUT_DIR / f"Bao_cao_{timestamp}.csv"

    # Multi-thread crawl
    results = []
    seen_stations = set()
    seen_lock = threading.Lock()

    print(f"\n🚀 Bắt đầu crawl {len(province_urls)} tỉnh với {MAX_WORKERS} workers...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                _crawl_single_province, url, unified_datetime_info,
                current_crawl_datetime, results, seen_lock, seen_stations,
                0, URL_PROVINCE_MAP.get(url)
            ): url for url in province_urls
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"❌ Thread exception: {e}")

    # Ghi CSV
    fieldnames = [
        "station_id", "station_name", "province", "district",
        "rain_total", "status", "timestamp", "data_time"
    ]
    _write_csv_rows(csv_path, results, fieldnames)

    # Xuất riêng dữ liệu ĐBSCL theo từng tỉnh (CSV + XLSX)
    dbscl_output_dir = OUTPUT_DIR / "Bao_cao_vrain_DBSCL"
    dbscl_output_dir.mkdir(parents=True, exist_ok=True)

    dbscl_rows_by_province = defaultdict(list)
    for row in results:
        province_std = _canonical_dbscl_province(row.get("province", ""))
        if not province_std:
            continue
        row_copy = row.copy()
        row_copy["province"] = province_std
        dbscl_rows_by_province[province_std].append(row_copy)

    for province_name, province_rows in sorted(dbscl_rows_by_province.items()):
        # Đọc dữ liệu cũ để merge vào file mới
        old_csv_files = sorted(dbscl_output_dir.glob(f"Bao_cao_{province_name}_*.csv"))
        old_xlsx_files = sorted(dbscl_output_dir.glob(f"Bao_cao_{province_name}_*.xlsx"))
        legacy_csv = dbscl_output_dir / f"{province_name}.csv"
        legacy_xlsx = dbscl_output_dir / f"{province_name}.xlsx"

        existing_rows = []
        for old_csv in old_csv_files:
            existing_rows.extend(_read_csv_rows(old_csv, fieldnames))
        existing_rows.extend(_read_csv_rows(legacy_csv, fieldnames))

        merged_rows = _merge_rows(existing_rows, province_rows, fieldnames)

        # Sau khi merge, chỉ giữ bộ file mới nhất theo timestamp hiện tại
        for old_csv in old_csv_files:
            old_csv.unlink(missing_ok=True)
        for old_xlsx in old_xlsx_files:
            old_xlsx.unlink(missing_ok=True)
        legacy_csv.unlink(missing_ok=True)
        legacy_xlsx.unlink(missing_ok=True)

        province_csv = dbscl_output_dir / f"Bao_cao_{province_name}_{timestamp}.csv"
        province_xlsx = dbscl_output_dir / f"Bao_cao_{province_name}_{timestamp}.xlsx"
        _write_csv_rows(province_csv, merged_rows, fieldnames)
        _write_xlsx_rows(province_xlsx, merged_rows, fieldnames)

    if dbscl_rows_by_province:
        print(f"Đã xuất dữ liệu ĐBSCL theo tỉnh tại: {dbscl_output_dir}")
    else:
        print("Cảnh báo: Không tìm thấy dữ liệu trạm thuộc ĐBSCL trong lần crawl này.")

    print("\n" + "=" * 50)
    print(f"Hoàn thành! Tổng số trạm: {len(results)}")
    print(f"Thời gian crawl: {current_crawl_datetime}")
    print(f"Dữ liệu đã được lưu vào: {csv_path}")


if __name__ == "__main__":
    main()