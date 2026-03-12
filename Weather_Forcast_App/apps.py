"""
App Configuration - Weather_Forcast_App
========================================

File cấu hình Django app cho Weather_Forcast_App.

Mục đích:
    - Định nghĩa AppConfig cho app
    - Cấu hình default_auto_field (kiểu ID tự động)
    - Có thể thêm ready() hook để chạy signal, init code khi app khởi động

Thông tin:
    - name: tên app ("Weather_Forcast_App")
    - default_auto_field: kiểu field ID tự động cho models
      + "django.db.models.BigAutoField" = 64-bit integer (tối đa ~9 tỷ record)
      + "django.db.models.AutoField" = 32-bit integer (cũ hơn)

Author: Weather Forecast Team
"""

from django.apps import AppConfig


class WeatherForcastConfig(AppConfig):
    # Kiểu field auto-increment mặc định cho models trong app này
    default_auto_field = "django.db.models.BigAutoField"
    
    # Tên app (phải khớp với folder name)
    name = "Weather_Forcast_App"
    
    # Có thể thêm ready() hook nếu cần khởi tạo gì khi app start:
    # def ready(self):
    #     from . import signals  # import signals để register events
