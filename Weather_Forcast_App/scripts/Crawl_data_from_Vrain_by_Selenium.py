"""
CRAWL_DATA_FROM_VRAIN_BY_SELENIUM.PY
====================================

Crawl lượng mưa VRAIN bằng Selenium (parse element) cho 5 tỉnh ĐBSCL.

Đầu ra:
    1) File tổng: data/data_crawl/Bao_cao_YYYYMMDD_HHMMSS.csv
    2) File theo tỉnh tại data/data_crawl/Bao_cao_vrain_DBSCL:
       - Bao_cao_<Tinh>_YYYYMMDD_HHMMSS.csv
       - Bao_cao_<Tinh>_YYYYMMDD_HHMMSS.xlsx

Ghi chú:
    - Mỗi lần crawl sẽ merge dữ liệu cũ + mới theo từng tỉnh.
    - Sau khi merge chỉ giữ bộ file mới nhất (timestamp hiện tại).
"""

import csv
import re
import time
import unicodedata
import threading
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from openpyxl import Workbook
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException

VN_TZ = ZoneInfo("Asia/Bangkok")

DBSCL_PROVINCES = {
    "an giang": "An_Giang",
    "ca mau": "Ca_Mau",
    "can tho": "Can_Tho",
    "dong thap": "Dong_Thap",
    "vinh long": "Vinh_Long",
}

# Thứ tự theo yêu cầu
URL_PROVINCE_MAP = {
    "https://vrain.vn/63/overview?public_map=windy": "Vĩnh Long",
    "https://vrain.vn/54/overview?public_map=windy": "Đồng Tháp",
    "https://vrain.vn/46/overview?public_map=windy": "An Giang",
    "https://vrain.vn/53/overview?public_map=windy": "Cần Thơ",
    "https://vrain.vn/52/overview?public_map=windy": "Cà Mau",
}


class VrainCrawlerFinal:
    def __init__(self, headless=True, max_workers=5, max_retries=2):
        self.headless = headless
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.data_lock = threading.Lock()
        self.all_rainfall_data = []

    def _get_chrome_options(self):
        options = Options()
        if self.headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-logging")
        options.add_argument("--disable-infobars")
        options.add_argument("--disable-notifications")
        options.add_argument("--disable-popup-blocking")
        options.add_argument("--blink-settings=imagesEnabled=false")
        options.add_argument("--window-size=1280,720")
        options.page_load_strategy = "eager"
        return options

    def create_driver(self):
        driver = webdriver.Chrome(options=self._get_chrome_options())
        driver.set_page_load_timeout(60)
        driver.implicitly_wait(3)
        return driver

    @staticmethod
    def normalize_string(text):
        if not text:
            return ""
        text = unicodedata.normalize("NFC", str(text))
        return " ".join(text.strip().split())

    @staticmethod
    def normalize_ascii_text(value):
        if not value:
            return ""
        text = unicodedata.normalize("NFKD", str(value))
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        text = text.replace("đ", "d").replace("Đ", "D").lower().strip()
        text = re.sub(r"\b(tinh|tp\.?|thanh pho)\b", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def extract_rainfall(text):
        match = re.search(r"(\d+[\.,]?\d*)", text)
        return match.group(1).replace(",", ".") if match else "0.0"

    def canonical_dbscl_province(self, raw_name):
        key = self.normalize_ascii_text(raw_name)
        return DBSCL_PROVINCES.get(key)

    @staticmethod
    def _read_csv_rows(csv_path, fieldnames):
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

    @staticmethod
    def _merge_rows(existing_rows, new_rows, fieldnames):
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

    @staticmethod
    def _write_csv_rows(csv_path, rows, fieldnames):
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    @staticmethod
    def _write_xlsx_rows(xlsx_path, rows, fieldnames):
        wb = Workbook()
        ws = wb.active
        ws.title = "rain_data"
        ws.append(fieldnames)
        for row in rows:
            ws.append([row.get(col, "") for col in fieldnames])
        wb.save(xlsx_path)

    def get_unified_datetime(self):
        for attempt in range(self.max_retries + 1):
            driver = None
            try:
                driver = self.create_driver()
                print("Đang truy cập trang chủ để lấy ngày và giờ cập nhật...")
                driver.get("https://vrain.vn/landing")
                time.sleep(3)
                all_text = driver.find_element(By.TAG_NAME, "body").text

                date_match = re.search(r"ngày\s*(\d{1,2}/\d{1,2})", all_text)
                hour_match = re.search(r"Tính từ\s*(\d{1,2})h", all_text)

                if date_match and hour_match:
                    current_year = datetime.now(VN_TZ).strftime("%Y")
                    return f"{date_match.group(1)}/{current_year} {hour_match.group(1)}:00"
                if date_match:
                    current_year = datetime.now(VN_TZ).strftime("%Y")
                    return f"{date_match.group(1)}/{current_year}"
                return datetime.now(VN_TZ).strftime("%d/%m/%Y %H:%M")
            except Exception as e:
                if attempt < self.max_retries:
                    print(f"  ⚠️ Lỗi lấy datetime (lần {attempt+1}): {str(e)[:50]} - thử lại...")
                    time.sleep(2)
                else:
                    print(f"  ⚠️ Không lấy được datetime từ website, dùng giờ hệ thống")
                    return datetime.now(VN_TZ).strftime("%d/%m/%Y %H:%M")
            finally:
                if driver:
                    try:
                        driver.quit()
                    except Exception:
                        pass

    def crawl_province(self, url, province_override, unified_datetime_info, crawl_datetime, retry_count=0):
        driver = None
        rows = []
        try:
            driver = self.create_driver()
            print(f"\nĐang truy cập: {url}")
            driver.get(url)

            # Chờ Angular hydrate - tăng timeout lên 45 giây
            try:
                WebDriverWait(driver, 45).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "div[class*='group'], div[class*='station'], span[class*='max-w-70']")
                    )
                )
            except TimeoutException:
                print(f"  ⚠️ Timeout selector {url} - tiếp tục với dữ liệu có sẵn")

            # Tăng sleep để đảm bảo render hoàn toàn
            time.sleep(5)

            selectors = [
                "div[class*='group']",
                "div[class*='station']",
                "div[class*='station-row']",
                "tr.station-item",
                "table tbody tr",
            ]

            elements = []
            for selector in selectors:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if len(elements) > 0:
                    break

            print(f"  Tỉnh: {province_override}")
            print(f"  Tìm thấy {len(elements)} phần tử trạm")

            seen_station_names = set()
            for el in elements:
                try:
                    text_content = el.text.strip()
                    if not text_content or "Lượng mưa" in text_content:
                        continue

                    station_name = ""
                    district = ""
                    rain_total = "0.0"
                    status = "Không xác định"

                    # Ưu tiên lấy station_name từ span class max-w-70
                    station_name_nodes = el.find_elements(By.CSS_SELECTOR, "span[class*='max-w-70']")
                    if station_name_nodes:
                        station_name = self.normalize_string(station_name_nodes[0].text)

                    # Fallback từ text dòng đầu
                    if not station_name:
                        lines = [line.strip() for line in text_content.split("\n") if line.strip()]
                        if not lines:
                            continue
                        station_name = self.normalize_string(lines[0])

                    if not station_name or station_name in seen_station_names:
                        continue
                    seen_station_names.add(station_name)

                    # district
                    district_nodes = el.find_elements(By.CSS_SELECTOR, "div[class*='sub-title']")
                    if district_nodes:
                        district = self.normalize_string(district_nodes[0].text)

                    # rain_total
                    rainfall_nodes = el.find_elements(By.CSS_SELECTOR, "span[class*='font-size-18px']")
                    if rainfall_nodes:
                        rain_total = self.extract_rainfall(rainfall_nodes[0].text)
                    else:
                        rain_total = self.extract_rainfall(text_content)

                    # status
                    status_nodes = el.find_elements(By.CSS_SELECTOR, "div[class*='level'] span")
                    if status_nodes:
                        status = self.normalize_string(status_nodes[0].text)

                    rows.append(
                        {
                            "station_id": station_name,
                            "station_name": station_name,
                            "province": province_override,
                            "district": district,
                            "rain_total": rain_total,
                            "status": status,
                            "timestamp": unified_datetime_info,
                            "data_time": crawl_datetime,
                        }
                    )
                except Exception:
                    continue

            print(f"  Đã trích xuất {len(rows)} trạm từ {province_override}")
            return rows

        except (TimeoutException, WebDriverException) as e:
            if retry_count < self.max_retries:
                print(f"  ⚠️  Lỗi {url}: {str(e)[:80]} | retry {retry_count + 1}")
                time.sleep(1)
                return self.crawl_province(
                    url,
                    province_override,
                    unified_datetime_info,
                    crawl_datetime,
                    retry_count + 1,
                )
            print(f"  ❌ Bỏ qua {url} sau {self.max_retries} lần thử")
            return []
        finally:
            if driver:
                try:
                    driver.quit()
                except Exception:
                    pass

    def run(self):
        unified_datetime_info = self.get_unified_datetime()
        crawl_datetime = datetime.now(VN_TZ).strftime("%d/%m/%Y %H:%M:%S")
        timestamp = datetime.now(VN_TZ).strftime("%Y%m%d_%H%M%S")

        province_pairs = list(URL_PROVINCE_MAP.items())
        print(f"\n🚀 Bắt đầu crawl {len(province_pairs)} tỉnh với {self.max_workers} workers...")

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self.crawl_province,
                    url,
                    province_name,
                    unified_datetime_info,
                    crawl_datetime,
                ): (url, province_name)
                for url, province_name in province_pairs
            }
            for future in as_completed(futures):
                try:
                    rows = future.result()
                    with self.data_lock:
                        results.extend(rows)
                except Exception as e:
                    url, province_name = futures[future]
                    print(f"❌ Thread exception {province_name} ({url}): {e}")

        self.all_rainfall_data = results
        self.export(timestamp)

    def export(self, timestamp):
        if not self.all_rainfall_data:
            print("⚠️ Không có dữ liệu để xuất!")
            return

        fieldnames = [
            "station_id",
            "station_name",
            "province",
            "district",
            "rain_total",
            "status",
            "timestamp",
            "data_time",
        ]

        project_root = Path(__file__).resolve().parents[2]
        output_dir = project_root / "data" / "data_crawl"
        output_dir.mkdir(parents=True, exist_ok=True)

        # file tổng
        total_csv = output_dir / f"Bao_cao_{timestamp}.csv"
        self._write_csv_rows(total_csv, self.all_rainfall_data, fieldnames)

        # file theo tỉnh
        dbscl_output_dir = output_dir / "Bao_cao_vrain_DBSCL"
        dbscl_output_dir.mkdir(parents=True, exist_ok=True)

        rows_by_province = defaultdict(list)
        for row in self.all_rainfall_data:
            province_std = self.canonical_dbscl_province(row.get("province", ""))
            if not province_std:
                continue
            row_copy = row.copy()
            row_copy["province"] = province_std
            rows_by_province[province_std].append(row_copy)

        for province_name, province_rows in sorted(rows_by_province.items()):
            old_csv_files = sorted(dbscl_output_dir.glob(f"Bao_cao_{province_name}_*.csv"))
            old_xlsx_files = sorted(dbscl_output_dir.glob(f"Bao_cao_{province_name}_*.xlsx"))
            legacy_csv = dbscl_output_dir / f"{province_name}.csv"
            legacy_xlsx = dbscl_output_dir / f"{province_name}.xlsx"

            existing_rows = []
            for old_csv in old_csv_files:
                existing_rows.extend(self._read_csv_rows(old_csv, fieldnames))
            existing_rows.extend(self._read_csv_rows(legacy_csv, fieldnames))

            merged_rows = self._merge_rows(existing_rows, province_rows, fieldnames)

            for old_csv in old_csv_files:
                old_csv.unlink(missing_ok=True)
            for old_xlsx in old_xlsx_files:
                old_xlsx.unlink(missing_ok=True)
            legacy_csv.unlink(missing_ok=True)
            legacy_xlsx.unlink(missing_ok=True)

            province_csv = dbscl_output_dir / f"Bao_cao_{province_name}_{timestamp}.csv"
            province_xlsx = dbscl_output_dir / f"Bao_cao_{province_name}_{timestamp}.xlsx"
            self._write_csv_rows(province_csv, merged_rows, fieldnames)
            self._write_xlsx_rows(province_xlsx, merged_rows, fieldnames)

        print("\n" + "=" * 60)
        print(f"📍 Tổng số trạm thu thập: {len(self.all_rainfall_data)}")
        print(f"📄 File tổng: {total_csv}")
        print(f"📄 File theo tỉnh: {dbscl_output_dir}")
        print("=" * 60)


if __name__ == "__main__":
    crawler = VrainCrawlerFinal(headless=True, max_workers=5, max_retries=2)
    crawler.run()
