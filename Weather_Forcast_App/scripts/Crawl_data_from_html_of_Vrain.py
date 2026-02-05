import re
import csv
import time
from pathlib import Path
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# === CẤU HÌNH SELENIUM ===
options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(options=options)

# === 1. LẤY NGÀY VÀ GIỜ CẬP NHẬT TỪ TRANG CHỦ ===
print("Đang truy cập trang chủ để lấy ngày và giờ cập nhật...")
driver.get("https://vrain.vn/landing")
time.sleep(5)

all_text = driver.find_element(By.TAG_NAME, "body").text
print("  Đang tìm kiếm ngày và giờ trong văn bản trang...")

date_match = re.search(r"ngày\s*(\d{1,2}/\d{1,2})", all_text)
hour_match = re.search(r"Tính từ\s*(\d{1,2})h", all_text)

if date_match and hour_match:
    date_from_main = date_match.group(1)
    hour_from_main = hour_match.group(1)
    current_year = datetime.now().strftime("%Y")
    unified_datetime_info = f"{date_from_main}/{current_year} {hour_from_main}:00"
    print(f"  Đã lấy ngày và giờ cập nhật từ trang chủ: {unified_datetime_info}")
elif date_match:
    date_from_main = date_match.group(1)
    current_year = datetime.now().strftime("%Y")
    unified_datetime_info = f"{date_from_main}/{current_year}"
    print(
        f"  Đã lấy ngày cập nhật từ trang chủ (không có giờ): {unified_datetime_info}"
    )
else:
    unified_datetime_info = "N/A"
    print("  Cảnh báo: Không tìm thấy ngày cập nhật. Sử dụng ngày và giờ hiện tại.")
    unified_datetime_info = datetime.now().strftime("%d/%m/%Y %H:%M")

current_crawl_datetime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

# Danh sách các URL tỉnh thành
province_urls = [
    "https://vrain.vn/20/overview?public_map=windy",
    "https://vrain.vn/2/overview?public_map=windy",
    "https://vrain.vn/4/overview?public_map=windy",
    "https://vrain.vn/5/overview?public_map=windy",
    "https://vrain.vn/6/overview?public_map=windy",
    "https://vrain.vn/7/overview?public_map=windy",
    "https://vrain.vn/8/overview?public_map=windy",
    "https://vrain.vn/11/overview?public_map=windy",
    "https://vrain.vn/18/overview?public_map=windy",
    "https://vrain.vn/12/overview?public_map=windy",
    "https://vrain.vn/14/overview?public_map=windy",
    "https://vrain.vn/13/overview?public_map=windy",
    "https://vrain.vn/17/overview?public_map=windy",
    "https://vrain.vn/22/overview?public_map=windy",
    "https://vrain.vn/24/overview?public_map=windy",
    "https://vrain.vn/27/overview?public_map=windy",
    "https://vrain.vn/26/overview?public_map=windy",
    "https://vrain.vn/28/overview?public_map=windy",
    "https://vrain.vn/30/overview?public_map=windy",
    "https://vrain.vn/31/overview?public_map=windy",
    "https://vrain.vn/32/overview?public_map=windy",
    "https://vrain.vn/34/overview?public_map=windy",
    "https://vrain.vn/37/overview?public_map=windy",
    "https://vrain.vn/41/overview?public_map=windy",
    "https://vrain.vn/42/overview?public_map=windy",
    "https://vrain.vn/44/overview?public_map=windy",
    "https://vrain.vn/61/overview?public_map=windy",
    "https://vrain.vn/45/overview?public_map=windy",
    "https://vrain.vn/56/overview?public_map=windy",
    "https://vrain.vn/63/overview?public_map=windy",
    "https://vrain.vn/54/overview?public_map=windy",
    "https://vrain.vn/46/overview?public_map=windy",
    "https://vrain.vn/53/overview?public_map=windy",
    "https://vrain.vn/52/overview?public_map=windy",
]

OUTPUT_DIR = Path("/media/voanhnhat/SDD_OUTSIDE5/PROJECT_WEATHER_FORECAST/Weather_Forcast_App/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = OUTPUT_DIR / f"Bao_cao_{timestamp}.csv"

with open(csv_path, "w", newline="", encoding="utf-8-sig") as csvfile:
    fieldnames = [
        "Tỉnh/Thành phố",
        "Tên trạm",
        "Huyện",
        "Tổng lượng mưa",
        "Tình trạng",
        "Dấu thời gian",
        "Thời gian cập nhập",
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for url in province_urls:
        try:
            print(f"\nĐang truy cập: {url}")
            driver.get(url)
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CLASS_NAME, "landing-content"))
            )
            time.sleep(2)

            page_html = driver.page_source

            # === TRÍCH XUẤT DỮ LIỆU ===
            # 1. Tên tỉnh
            province_match = re.search(
                r"<div[^>]*app-title[^>]*>.*?<span[^>]*>([^<]+)</span>",
                page_html,
                re.DOTALL,
            )
            province_name = (
                province_match.group(1).strip() if province_match else "Không xác định"
            )
            print(f"  Tỉnh: {province_name}")

            # 2. SỬ DỤNG NGÀY VÀ GIỜ ĐÃ LẤY TỪ TRANG CHỦ
            datetime_info = unified_datetime_info

            # 3. Tìm các khối thông tin trạm
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

                writer.writerow(
                    {
                        "Tỉnh/Thành phố": province_name,
                        "Tên trạm": station_name,
                        "Huyện": xa_phuong,
                        "Tổng lượng mưa": rainfall,
                        "Tình trạng": status,
                        "Dấu thời gian": datetime_info,
                        "Thời gian cập nhập": current_crawl_datetime,
                    }
                )

            print(f"  Đã trích xuất {len(station_blocks)} trạm.")

        except Exception as e:
            print(f"  Lỗi khi xử lý {url}: {e}")

try:
    driver.quit()
except Exception as e:
    print(f"[WARN] Lỗi khi đóng trình duyệt: {e}")

print("\n" + "=" * 50)
print(f"Hoàn thành! Thời gian crawl: {current_crawl_datetime}")
print(f"Dữ liệu đã được lưu vào: {csv_path}")