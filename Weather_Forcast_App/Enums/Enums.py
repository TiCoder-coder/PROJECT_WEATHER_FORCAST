from enum import Enum  # Import Enum: lớp cơ sở trong Python để tạo kiểu liệt kê (enumeration)

# ============================================================
# CustomEnum: lớp Enum "mở rộng" để sau này các Enum con có thể
# có thêm thuộc tính/phương thức dùng chung (ví dụ: description).
# ============================================================
class CustomEnum(Enum):
    @property
    def description(self):
        # ------------------------------------------------------------
        # @property biến method này thành "thuộc tính chỉ đọc"
        # => có thể gọi: Role.Admin.description (không cần ngoặc)
        #
        # Ý tưởng của bạn:
        # - Mỗi giá trị Enum sẽ có thêm mô tả dễ hiểu (tiếng Việt/tiếng Anh)
        # - Giúp hiển thị UI / log / API response thân thiện hơn
        #
        # LƯU Ý QUAN TRỌNG:
        # - Hiện tại bạn để `pass` => nghĩa là chưa có logic trả về mô tả,
        #   nên nếu ai gọi .description sẽ nhận None (hoặc không có giá trị).
        # - Thông thường, chỗ này sẽ "return ..." một chuỗi mô tả tùy theo từng member,
        #   hoặc dựa vào một dict mapping từ value -> mô tả.
        #
        # Ví dụ cách dùng trong tương lai (chỉ minh hoạ, KHÔNG sửa code theo yêu cầu):
        # - return ROLE_DESCRIPTIONS.get(self.value, self.name)
        # ------------------------------------------------------------
        pass


# ============================================================
# Role: Enum định nghĩa các vai trò (role) trong hệ thống.
#
# Tại sao kế thừa CustomEnum?
# - Để Role có sẵn thuộc tính .description (và các tiện ích khác sau này)
# - Có thể tái sử dụng logic chung cho nhiều Enum khác (Status, Permission, ...)
# ============================================================

# Dinh nghia class Role
class Role(CustomEnum):
    # ------------------------------------------------------------
    # Mỗi dòng dưới đây tạo ra một "member" của Enum Role.
    #
    # Cú pháp: TênMember = "Giá trị"
    #
    # - TênMember (Guest/Manager/Admin) là tên định danh trong code.
    # - Giá trị chuỗi ("Guest"/"Manager"/"Admin") là thứ thường được lưu trong DB,
    #   hoặc gửi/nhận trong API (JSON), vì string dễ serialize.
    #
    # Cách dùng:
    # - Role.Guest          -> member Enum
    # - Role.Guest.value    -> "Guest" (giá trị string)
    # - Role.Guest.name     -> "Guest" (tên member)
    #
    # Tại sao dùng string thay vì int?
    # - Dễ đọc, dễ debug, ít nhầm lẫn khi nhìn DB/log/API.
    # - Tránh việc "1/2/3" không có ý nghĩa nếu thiếu mapping.
    # ------------------------------------------------------------
    Guest = "Guest"      # Vai trò khách (quyền hạn thấp nhất, thường chỉ xem)
    Manager = "Manager"  # Vai trò quản lý (quyền cao hơn: quản lý dữ liệu/chức năng)
    Admin = "Admin"      # Vai trò admin (quyền cao nhất: cấu hình hệ thống/toàn quyền)
