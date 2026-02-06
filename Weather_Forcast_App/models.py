"""
Django Models Configuration
===========================

File định nghĩa models cho app Weather_Forcast_App.

Ghi chú:
    - Project dùng MongoDB làm primary database (không dùng SQLite cho dữ liệu chính)
    - Django models ở đây chủ yếu cho Django admin, auth framework, session management
    - Models thực tế (LoginModel, v.v.) được định nghĩa riêng trong Weather_Forcast_App/Models/

Cấu trúc:
    - Models cho django.contrib app (auth, session, ...)
    - Custom models ở folder Models/ (ví dụ: LoginModel)
    - Database mapping ở db_connection.py (MongoDB)
    
Author: Weather Forecast Team
"""

from django.db import models

# Thêm custom models ở đây nếu cần (ví dụ):
# from Weather_Forcast_App.Models.Login import LoginModel
# (Tuy nhiên LoginModel dùng MongoDB, không cần migrate Django ORM)