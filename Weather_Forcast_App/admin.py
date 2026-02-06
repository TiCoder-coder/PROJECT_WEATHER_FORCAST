"""
Django Admin Configuration - Weather_Forcast_App
=================================================

File cấu hình Django Admin panel cho app Weather_Forcast_App.

Mục đích:
    - Đăng ký models với Django Admin
    - Tùy chỉnh giao diện admin (list_display, filters, search, ...)
    - Cung cấp interface quản lý dữ liệu (CRUD)

Ghi chú:
    - Django Admin tự động tạo based on models
    - Để quản lý models ngoài admin, thêm admin.site.register()
    - Hiện tại file này rỗng => models chưa được đăng ký
    
Cách sử dụng (ví dụ):
    from django.contrib import admin
    from .Models.Login import LoginModel
    
    @admin.register(LoginModel)
    class LoginModelAdmin(admin.ModelAdmin):
        list_display = ['username', 'email', 'role', 'is_active']
        list_filter = ['role', 'is_active']
        search_fields = ['username', 'email']
        readonly_fields = ['createdAt', 'updatedAt']

Author: Weather Forecast Team
"""

from django.contrib import admin

# import models nếu cần đăng ký admin
# ví dụ: from Weather_Forcast_App.Models.Login import LoginModel

# Đăng ký models với Django Admin ở đây
# admin.site.register(Model)
