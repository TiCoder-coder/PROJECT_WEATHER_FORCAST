# 📁 scripts

## Tổng quan
Thư mục này chứa các script Python phục vụ cho việc xử lý dữ liệu, crawl, merge, validate, và các tác vụ backend liên quan đến dữ liệu thời tiết.

## Chức năng chung
- Xử lý dữ liệu thô sau crawl (clean, merge, validate).
- Tích hợp API crawl dữ liệu thời tiết từ nhiều nguồn.
- Gửi email xác thực, kiểm tra hợp lệ.

## Chức năng riêng lẻ
- `Cleardata.py`: Làm sạch dữ liệu sau crawl/merge.
- `Crawl_data_by_API.py`: Crawl dữ liệu thời tiết qua API.
- `Crawl_data_from_Vrain_by_API.py`: Crawl dữ liệu mưa từ Vrain qua API.
- `Crawl_data_from_Vrain_by_Selenium.py`: Crawl dữ liệu mưa từ Vrain bằng Selenium.
- `Crawl_data_from_html_of_Vrain.py`: Crawl dữ liệu mưa từ HTML Vrain.
- `Email_validator.py`: Kiểm tra hợp lệ email.
- `Login_services.py`: Xử lý đăng nhập, xác thực.
- `Merge_xlsx.py`: Gộp file xlsx/csv thành dataset chung.
- `email_templates.py`: Template email gửi OTP, xác thực.

## Cấu trúc thư mục
<ul>
  <li>🐍 Cleardata.py</li>
  <li>🐍 Crawl_data_by_API.py</li>
  <li>🐍 Crawl_data_from_Vrain_by_API.py</li>
  <li>🐍 Crawl_data_from_Vrain_by_Selenium.py</li>
  <li>🐍 Crawl_data_from_html_of_Vrain.py</li>
  <li>🐍 Email_validator.py</li>
  <li>🐍 Login_services.py</li>
  <li>🐍 Merge_xlsx.py</li>
  <li>🐍 email_templates.py</li>
</ul>

---

## 👤 Maintainer / Profile Info
- 🧑‍💻 Maintainer: Võ Anh Nhật, Dư Quốc Việt, Trương Hoài Tú, Võ Huỳnh Anh Tuần
- 🎓 University: UTH
- 📧 Email: voanhnhat1612@gmmail.com, vohuynhanhtuan0512@gmail.com, hoaitu163@gmail.com, duviet720@gmail.com
- 📞 Phone: 0335052899

---

## License
MIT License
