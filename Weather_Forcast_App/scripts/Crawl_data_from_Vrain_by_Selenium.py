from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
import pandas as pd
import time
import json
import os
import re
import unicodedata
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class VrainCrawlerFinal:
    def __init__(self, headless=True, max_workers=5, max_retries=3):
        self.base_url = "https://www.vrain.vn"
        self.all_rainfall_data = []
        self.headless = headless
        self.max_workers = max_workers
        self.max_retries = max_retries  # Số lần thử lại
        self.data_lock = threading.Lock()
        self.unique_stations = {}
        self.failed_provinces = []  # Lưu các tỉnh thất bại

    def create_driver(self):
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless=new")

        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_experimental_option(
            "prefs", {"profile.default_content_setting_values": {"images": 2}}
        )
        chrome_options.page_load_strategy = "eager"

        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(30)
        return driver

    def normalize_string(self, text):
        """Chuẩn hóa tiếng Việt và loại bỏ khoảng trắng thừa"""
        if not text:
            return ""
        text = unicodedata.normalize("NFC", text)
        return " ".join(text.strip().split())

    def extract_rainfall(self, text):
        """Trích xuất số từ chuỗi '12.5 mm'"""
        match = re.search(r"(\d+[\.,]?\d*)", text)
        return match.group(1).replace(",", ".") if match else "0"

    def get_province_name(self, driver):
        """Lấy tên tỉnh từ các selector khác nhau"""
        selectors = [
            "span[_ngcontent-ng-c641299110]",
            "span[_ngcontent-serverapp-c641299110]",
            "span[class*='ng-']",
            ".app-title span",
            "h1 span",
            ".province-name",
            "h1",
            "title",
        ]

        for selector in selectors:
            try:
                element = driver.find_element(By.CSS_SELECTOR, selector)
                text = element.text.strip()
                if text and len(text) > 1:
                    # Loại bỏ "VRAIN", "Lượng mưa", ký tự thừa
                    text = re.sub(r"(VRAIN|Lượng mưa.*|[-|])", "", text).strip()
                    if text and not any(
                        x in text.lower() for x in ["tại các trạm", "ngày"]
                    ):
                        return self.normalize_string(text)
            except:
                continue

        # Thử lấy từ title trang
        try:
            title = driver.title
            if title and "VRAIN" in title:
                text = title.replace("VRAIN", "").replace("-", "").strip()
                if text:
                    return self.normalize_string(text)
        except:
            pass

        return None

    def crawl_province(self, province_id, retry_count=0):
        """Crawl một tỉnh với cơ chế retry"""
        driver = None
        try:
            driver = self.create_driver()
            url = f"{self.base_url}/{province_id}/overview?public_map=windy"
            driver.get(url)

            wait = WebDriverWait(driver, 15)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "body")))
            time.sleep(4)  # Tăng thời gian chờ JS render

            # Lấy tên tỉnh
            province_name = self.get_province_name(driver)

            if not province_name:
                province_name = f"ID_{province_id}"

            found_count = 0
            crawl_time = datetime.now().strftime("%d/%m/%Y %H:%M")

            # Tìm trạm - thêm nhiều selector hơn
            selectors = [
                "div[class*='station-row']",
                "div[class*='station']",
                "tr.station-item",
                ".station-list-item",
                "table tbody tr",
                "table tr",
                ".data-row",
            ]

            elements = []
            for selector in selectors:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if len(elements) > 2:
                    break

            for el in elements:
                try:
                    text_content = el.text.strip()
                    if not text_content or "Lượng mưa" in text_content:
                        continue

                    lines = [
                        line.strip()
                        for line in text_content.split("\n")
                        if line.strip()
                    ]

                    if len(lines) >= 2:
                        station_name = self.normalize_string(lines[0])
                        rainfall_val = self.extract_rainfall(lines[-1])

                        unique_key = f"{province_name}_{station_name}".lower()

                        with self.data_lock:
                            if unique_key not in self.unique_stations:
                                record = {
                                    "province_id": province_id,
                                    "tinh": province_name,
                                    "tram": station_name,
                                    "luong_mua": float(rainfall_val),
                                    "thoi_gian": crawl_time,
                                }
                                self.unique_stations[unique_key] = record
                                found_count += 1
                except:
                    continue

            # Kiểm tra nếu không lấy được dữ liệu
            if found_count == 0 or province_name.startswith("ID_"):
                if retry_count < self.max_retries:
                    print(
                        f"⚠️  ID {province_id}: {province_name} - Không có dữ liệu, thử lại lần {retry_count + 1}..."
                    )
                    if driver:
                        driver.quit()
                    time.sleep(2)  # Đợi trước khi thử lại
                    return self.crawl_province(province_id, retry_count + 1)
                else:
                    print(
                        f"❌ ID {province_id}: Thất bại sau {self.max_retries} lần thử"
                    )
                    with self.data_lock:
                        self.failed_provinces.append(province_id)
                    return 0

            print(f"✅ ID {province_id}: {province_name} - Lấy được {found_count} trạm")
            return found_count

        except Exception as e:
            if retry_count < self.max_retries:
                print(
                    f"⚠️  ID {province_id} lỗi: {str(e)[:30]} - Thử lại lần {retry_count + 1}..."
                )
                if driver:
                    driver.quit()
                time.sleep(2)
                return self.crawl_province(province_id, retry_count + 1)
            else:
                print(
                    f"❌ Lỗi ID {province_id} sau {self.max_retries} lần thử: {str(e)[:50]}"
                )
                with self.data_lock:
                    self.failed_provinces.append(province_id)
                return 0
        finally:
            if driver:
                driver.quit()

    def run(self, start_id=1, end_id=63):
        print(f"🚀 Bắt đầu crawl từ ID {start_id} đến {end_id}...")
        print(f"🔄 Số lần thử lại tối đa: {self.max_retries}")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            executor.map(self.crawl_province, range(start_id, end_id + 1))

        # Retry các tỉnh thất bại một lần nữa (tuần tự)
        if self.failed_provinces:
            print(f"\n🔄 Đang retry {len(self.failed_provinces)} tỉnh thất bại...")
            retry_failed = []
            for province_id in self.failed_provinces:
                time.sleep(1)
                result = self.crawl_province(province_id, 0)
                if result == 0:
                    retry_failed.append(province_id)

            self.failed_provinces = retry_failed

        # Chuyển dict thành list và SẮP XẾP THEO ID
        self.all_rainfall_data = list(self.unique_stations.values())
        self.all_rainfall_data.sort(key=lambda x: (x["province_id"], x["tram"]))

    def export(self):
        if not self.all_rainfall_data:
            print("⚠️ Không có dữ liệu để xuất!")
            return

        df = pd.DataFrame(self.all_rainfall_data)

        # Đánh số thứ tự sau khi đã sắp xếp
        df.insert(0, "STT", range(1, len(df) + 1))

        # Đổi tên cột cho đẹp
        df.columns = [
            "STT",
            "ID Tỉnh",
            "Tỉnh/Thành phố",
            "Tên Trạm",
            "Lượng mưa (mm)",
            "Thời gian cập nhật",
        ]

        output_dir = "/PROJECT_WEATHER_FORECAST/Weather_Forcast_App/output"
        os.makedirs(output_dir, exist_ok=True)  # tạo thư mục nếu chưa có

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = os.path.join(output_dir, f"Bao_cao_mua_{timestamp}.xlsx")
        df.to_excel(excel_path, index=False)

        print(f"\n{'=' * 60}")
        print(f"📊 TỔNG KẾT:")
        print(f"📍 Tổng số trạm thu thập: {len(df)}")
        print(f"🗂️  Số tỉnh/thành: {df['ID Tỉnh'].nunique()}")

        if self.failed_provinces:
            print(
                f"❌ Các ID vẫn thất bại: {', '.join(map(str, self.failed_provinces))}"
            )
        else:
            print(f"✅ Tất cả tỉnh đều lấy dữ liệu thành công!")

        print(f"📄 File đã lưu tại: {excel_path}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    # max_retries=3: Mỗi tỉnh sẽ thử tối đa 3 lần nếu thất bại
    # max_workers=3: Giảm xuống để tránh quá tải server
    crawler = VrainCrawlerFinal(headless=False, max_workers=63, max_retries=3)
    crawler.run(start_id=1, end_id=63)
    crawler.export()
