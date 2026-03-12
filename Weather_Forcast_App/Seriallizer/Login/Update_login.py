from .Base_login import BaseSerializerLogin

# ============================================================
# LoginUpdate: SERIALIZER DÙNG CHO API CẬP NHẬT THÔNG TIN LOGIN
# ============================================================
# Mục tiêu của serializer này:
# 1) Khi UPDATE (PATCH/PUT), thường bạn muốn cho phép cập nhật "một phần" (partial update)
#    => các field KHÔNG bắt buộc phải có đủ như lúc create.
# 2) Không cho phép client gửi/chỉnh field "id" (hoặc không muốn validate id trong body)
#    => remove "id" khỏi serializer fields nếu tồn tại.
#
# Lưu ý:
# - Serializer này kế thừa BaseSerializerLogin (ModelSerializer cho LoginModel)
# - BaseSerializerLogin có thể đang fields='__all__' nên sẽ auto sinh đủ field.
# - Việc chỉnh required/popping field phải làm trong __init__ sau khi super() chạy
class LoginUpdate(BaseSerializerLogin):
    def __init__(self, *args, **kwargs):
        # ============================================================
        # GỌI INIT CỦA LỚP CHA
        # ============================================================
        # - Tạo self.fields (dict/OrderedDict chứa các serializer fields)
        # - Gắn validators, default, required... theo model/serializer config
        super().__init__(*args, **kwargs)

        # ============================================================
        # SET TẤT CẢ FIELD -> required = False
        # ============================================================
        # Ý nghĩa:
        # - Cho phép update dạng "partial":
        #   client chỉ cần gửi những field muốn đổi, không cần gửi đủ bộ field.
        # - Nếu để required=True (mặc định một số field), DRF sẽ báo lỗi:
        #   "This field is required." khi client không gửi field đó.
        #
        # Cách làm:
        # - self.fields.values() trả về danh sách Field object của serializer
        # - set field.required = False cho tất cả
        for field in self.fields.values():
            field.required = False                                              # Cac thuoc tinh la khong bat buoc
        
        # ============================================================
        # LOẠI BỎ FIELD "id" NẾU TỒN TẠI
        # ============================================================
        # Vì sao cần pop("id")?
        # - Trong update, bạn thường xác định đối tượng cần update qua URL:
        #   /logins/<id>/  (id nằm trong path param)
        # - Không muốn client gửi id trong body để tránh:
        #   + nhầm lẫn id
        #   + cố tình đổi id
        #   + validation rườm rà
        #
        # Check "id" in self.fields để tránh KeyError nếu serializer không có field này
        if "id" in self.fields:
            # pop("id") sẽ xoá field khỏi serializer
            # => field này sẽ không:
            #    - xuất hiện trong output response (serialize)
            #    - được đọc/validate từ input request (deserialize)
            self.fields.pop("id")
