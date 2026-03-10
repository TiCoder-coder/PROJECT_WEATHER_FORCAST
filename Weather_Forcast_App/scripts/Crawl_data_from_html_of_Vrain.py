"""
CRAWL_DATA_FROM_HTML_OF_VRAIN.PY
================================

Script crawl dữ liệu thời tiết từ VRAIN.VN bằng HTML parsing (Selenium + Regex)

Mục đích:
    - Lấy dữ liệu lượng mưa, tên trạm từ các trang tỉnh VRAIN
    - Sử dụng Selenium để load dynamic content (JavaScript render)
    - Dùng Regex để parse HTML và trích xuất thông tin
    - Xuất dữ liệu trực tiếp sang CSV

Đặc điểm:
    - Crawl tuần tự (1 tỉnh / 1 tab) để tránh quá tải server
    - Parse HTML bằng Regex (nhanh hơn BeautifulSoup nhưng khó maintain hơn)
    - Xử lý các biến thể tên tỉnh, trạm (normalize Unicode)
    - Tự động lấy ngày/giờ cập nhật từ trang chủ
    - Xuất CSV với encoding UTF-8 BOM (compatible Excel)

Cách sử dụng:
    python Crawl_data_from_html_of_Vrain.py
    
    # Sẽ crawl tất cả 64 tỉnh và xuất sang CSV

Dữ liệu được lưu:
    - CSV: data/data_crawl/Bao_cao_YYYYMMDD_HHMMSS.csv
    - Columns:
      - Tỉnh/Thành phố
      - Tên trạm
      - Huyện
      - Tổng lượng mưa
      - Tình trạng
      - Dấu thời gian
      - Thời gian cập nhập

Luồng chạy:
    1. Truy cập trang chủ VRAIN để lấy ngày/giờ cập nhật
    2. Duyệt 34 URL tỉnh thành (hardcode)
    3. Với mỗi tỉnh:
       - Load trang bằng Selenium
       - Wait cho content load (max 15s)
       - Parse HTML bằng Regex để trích tên tỉnh, trạm, lượng mưa
       - Ghi dòng vào CSV
    4. Đóng browser khi xong

Biến cấu hình (hardcode trong code):
    - province_urls: danh sách 34 URL tỉnh (cần update khi VRAIN thay đổi)
    - OUTPUT_DIR: thư mục xuất CSV

Dependencies:
    - selenium: điều khiển Chrome headless
    - re (regex): parse HTML
    - csv: xuất CSV
    - datetime: thêm timestamp

Lưu ý:
    - Crawl tuần tự => chậm (2-3 phút)
    - Nếu muốn nhanh hơn => dùng Crawl_data_from_Vrain_by_Selenium.py (đa luồng)
    - Thư mục data/data_crawl/ sẽ được tạo tự động nếu chưa có

Author: Weather Forecast Team
Version: 1.0
Last Updated: 2026-02-06
"""

import re
import csv
import time
import hashlib
import threading
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

VN_TZ = ZoneInfo("Asia/Bangkok")
MAX_WORKERS = 6
MAX_RETRIES = 2


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


def _crawl_single_province(url, unified_datetime_info, current_crawl_datetime, results, seen_lock, seen_stations, retry=0):
    """Crawl 1 tỉnh, trả về list rows."""
    driver = None
    rows = []
    try:
        driver = _create_driver()
        print(f"\nĐang truy cập: {url}")
        driver.get(url)
        try:
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".landing-content, div[class*='station'], body"))
            )
        except TimeoutException:
            pass
        time.sleep(1)

        page_html = driver.page_source

        # === TRÍCH XUẤT DỮ LIỆU ===
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
                "station_id": hashlib.md5(unique_key.encode()).hexdigest()[:12],
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
            return _crawl_single_province(url, unified_datetime_info, current_crawl_datetime, results, seen_lock, seen_stations, retry + 1)
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

    # Danh sách các URL tỉnh thành - tự động sinh từ ID 1–63
    province_urls = [
        f"https://vrain.vn/{pid}/overview?public_map=windy"
        for pid in range(1, 64)
    ]

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
                current_crawl_datetime, results, seen_lock, seen_stations
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
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print("\n" + "=" * 50)
    print(f"Hoàn thành! Tổng số trạm: {len(results)}")
    print(f"Thời gian crawl: {current_crawl_datetime}")
    print(f"Dữ liệu đã được lưu vào: {csv_path}")


if __name__ == "__main__":
    main()
