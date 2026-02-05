from django.core.management.base import BaseCommand
from decouple import config
from Weather_Forcast_App.scripts.Login_services import ManagerService

# =========================
# LỚP GIẢ LẬP USER "SEED"
# =========================
class SeedUser:
    # role dùng để ManagerService nhận biết người đang thực hiện thao tác
    # (thường service sẽ kiểm tra quyền: chỉ admin/manager mới được tạo manager)
    role = "admin"


# =========================
# DJANGO MANAGEMENT COMMAND
# =========================
class Command(BaseCommand):
    # help: mô tả lệnh, hiển thị khi chạy `python manage.py help`
    help = "Seed admin account into MongoDB (first time)."

    def handle(self, *args, **options):
        # ------------------------------------------------------------
        # 1) ĐỌC BIẾN MÔI TRƯỜNG TỪ FILE .env
        # ------------------------------------------------------------
        # decouple.config() sẽ đọc từ .env (hoặc từ biến môi trường hệ thống)
        # - USER_NAME_ADMIN: username admin cần seed
        # - ADMIN_PASSWORD: password admin cần seed
        # - ADMIN_EMAIL: email admin (nếu không có thì lấy mặc định)
        #
        # default=None nghĩa là nếu không tìm thấy key trong .env
        # thì biến này sẽ thành None (để mình check thiếu dữ liệu bên dưới)
        username = config("USER_NAME_ADMIN", default=None)
        password = config("ADMIN_PASSWORD", default=None)
        email = config("ADMIN_EMAIL", default="admin@local.com")

        # ------------------------------------------------------------
        # 2) VALIDATE DỮ LIỆU BẮT BUỘC
        # ------------------------------------------------------------
        # Chỉ username + password là bắt buộc để tạo tài khoản admin.
        # Nếu thiếu 1 trong 2 thì dừng luôn để tránh tạo user lỗi/không đăng nhập được.
        if not username or not password:
            # self.stdout.write: in ra console theo format của Django command
            # self.style.ERROR: tô màu/format dạng lỗi (đỏ)
            self.stdout.write(self.style.ERROR("LACK USER_NAME_ADMIN OR ADMIN_PASSWORD in .env"))
            return

        try:
            # ------------------------------------------------------------
            # 3) TẠO "NGƯỜI THỰC HIỆN" (ACTOR) ĐỂ GỌI SERVICE
            # ------------------------------------------------------------
            # seed_user ở đây là 1 object rất đơn giản, chỉ mang role="admin"
            # Mục đích: nhiều service sẽ yêu cầu "ai đang thực hiện hành động"
            # để kiểm tra quyền (authorization).
            seed_user = SeedUser()

            # ------------------------------------------------------------
            # 4) GỌI SERVICE TẠO MANAGER/ADMIN TRONG DATABASE (MongoDB)
            # ------------------------------------------------------------
            # ManagerService.create_manager(...) chịu trách nhiệm:
            # - validate dữ liệu (username/email có trùng không, password hợp lệ không, v.v.)
            # - hash password (thường bcrypt/argon2 tuỳ project)
            # - insert document vào MongoDB
            # - gán role="admin" cho tài khoản được tạo
            #
            # Tham số truyền vào:
            #   seed_user          : actor (người gọi) để check quyền
            #   "Administrator"    : tên hiển thị / full name (tuỳ schema)
            #   username           : username admin lấy từ .env
            #   password           : password admin lấy từ .env
            #   email              : email admin (mặc định admin@local.com)
            #   role="admin"       : role của tài khoản được tạo
            ManagerService.create_manager(
                seed_user,
                "Administrator",
                username,
                password,
                email,
                role="admin"
            )

            # ------------------------------------------------------------
            # 5) THÔNG BÁO THÀNH CÔNG
            # ------------------------------------------------------------
            # self.style.SUCCESS: format màu xanh (thành công)
            self.stdout.write(self.style.SUCCESS(f"Admin '{username}' created successfully in MongoDB!"))

        except Exception as e:
            # ------------------------------------------------------------
            # 6) BẮT LỖI & THÔNG BÁO
            # ------------------------------------------------------------
            # Nếu service ném lỗi (ví dụ: user đã tồn tại, email trùng, lỗi kết nối DB,...)
            # thì in ra dạng WARNING (thường màu vàng).
            self.stdout.write(self.style.WARNING(str(e)))
