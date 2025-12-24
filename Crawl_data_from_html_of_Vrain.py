import re
import csv
import time
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# === CẤU HÌNH SELENIUM ===
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(options=options)

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
    "https://vrain.vn/52/overview?public_map=windy"
]

# Lấy ngày hiện tại (ngày crawl) - Định dạng: dd/mm/YYYY
current_crawl_date = datetime.now().strftime("%d/%m/%Y")

# Mở file CSV để ghi dữ liệu
with open('luong_mua_thong_ke_selenium.csv', 'w', newline='', encoding='utf-8-sig') as csvfile:
    # THÊM CỘT "Ngày crawl" vào fieldnames
    fieldnames = ['Tỉnh', 'Trạm', 'Xã/Phường', 'Lượng mưa (mm)', 'Tình trạng', 'Ngày cập nhật', 'Ngày crawl']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for url in province_urls:
        try:
            print(f"Đang truy cập: {url}")
            driver.get(url)
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CLASS_NAME, "landing-content"))
            )
            time.sleep(2)

            page_html = driver.page_source

            # === TRÍCH XUẤT DỮ LIỆU ===
            # 1. Tên tỉnh
            province_match = re.search(r'<div[^>]*app-title[^>]*>.*?<span[^>]*>([^<]+)</span>', page_html, re.DOTALL)
            province_name = province_match.group(1).strip() if province_match else "Không xác định"
            print(f"  Tỉnh: {province_name}")

            # 2. Ngày cập nhật (từ trang web)
            date_match = re.search(r'Lượng mưa tại các trạm đo ngày\s*(\d{1,2}/\d{1,2})', page_html)
            date_info = date_match.group(1) if date_match else "N/A"

            # 3. Tìm các khối thông tin trạm
            station_blocks = re.findall(r'<div[^>]*class="[^"]*\bgroup\b[^"]*"[^>]*>(.*?)</div>\s*</div>\s*</div>', page_html, re.DOTALL)
            if not station_blocks:
                station_blocks = re.findall(r'<div[^>]*class="[^"]*\bstation\b[^"]*"[^>]*>(.*?)</div>\s*</div>\s*</div>', page_html, re.DOTALL)

            for block in station_blocks:
                # Trích xuất thông tin từng trạm
                station_match = re.search(r'<div[^>]*station-row-1[^>]*>.*?<span[^>]*class="[^"]*\bmax-w-70\b[^"]*"[^>]*>([^<]+)</span>', block, re.DOTALL)
                station_name = station_match.group(1).strip() if station_match else "N/A"

                location_match = re.search(r'<div[^>]*station-row-2[^>]*>.*?<div[^>]*class="[^"]*\bsub-title\b[^"]*"[^>]*>([^<]+)</div>', block, re.DOTALL)
                xa_phuong = location_match.group(1).strip() if location_match else "N/A"

                rainfall_match = re.search(r'<div[^>]*station-row-1[^>]*>.*?<span[^>]*class="[^"]*font-size-18px[^"]*"[^>]*>([\d.]+)\s*<span[^>]*>mm</span>', block, re.DOTALL)
                rainfall = rainfall_match.group(1).strip() if rainfall_match else "0.0"

                status_match = re.search(r'<div[^>]*station-row-2[^>]*>.*?<div[^>]*class="[^"]*\blevel\b[^"]*"[^>]*>.*?<span[^>]*>([^<]+)</span>', block, re.DOTALL)
                status = status_match.group(1).strip() if status_match else "Không xác định"

                # Ghi dữ liệu vào CSV - THÊM "Ngày crawl"
                writer.writerow({
                    'Tỉnh': province_name,
                    'Trạm': station_name,
                    'Xã/Phường': xa_phuong,
                    'Lượng mưa (mm)': rainfall,
                    'Tình trạng': status,
                    'Ngày cập nhật': date_info,
                    'Ngày crawl': current_crawl_date  # Thêm ngày crawl
                })

            print(f"  Đã trích xuất {len(station_blocks)} trạm.")

        except Exception as e:
            print(f"  Lỗi khi xử lý {url}: {e}")

# Đóng trình duyệt
driver.quit()
print("\n" + "="*50)
print(f"Hoàn thành! Ngày crawl: {current_crawl_date}")
print("Dữ liệu đã được lưu vào 'luong_mua_thong_ke_selenium.csv'")