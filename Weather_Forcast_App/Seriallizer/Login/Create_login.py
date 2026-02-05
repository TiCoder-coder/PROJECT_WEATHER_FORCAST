from .Base_login import BaseSerializerLogin

# ============================================================
# LoginLoginCreate: SERIALIZER DÙNG CHO API TẠO TÀI KHOẢN / ĐĂNG KÝ
# ============================================================
# Ý tưởng chính:
# - BaseSerializerLogin (từ Base_login.py) đang là serializer "base" cho LoginModel
#   thường fields='__all__' và các rule mặc định theo model.
#
# - Khi tạo mới user (create/register), bạn muốn BẮT BUỘC một số trường:
#   ["name", "userName", "password", "email", "role"]
#
# - Thay vì khai báo từng field required=True thủ công ở class-level,
#   bạn override __init__ để sau khi DRF tạo self.fields xong,
#   bạn chỉnh thuộc tính required cho từng field mong muốn.
#
# Lưu ý:
# - DRF chỉ tạo self.fields sau khi ModelSerializer khởi tạo
# - Vì vậy phải gọi super().__init__ trước khi chỉnh required
class LoginLoginCreate(BaseSerializerLogin):
    def __init__(self, *args, **kwargs):
        # ============================================================
        # GỌI INIT CỦA LỚP CHA
        # ============================================================
        # - Lớp cha (ModelSerializer) sẽ:
        #   + đọc Meta.model và Meta.fields
        #   + tạo ra self.fields (OrderedDict các field serializer)
        #   + gắn validators, default, required (theo model và serializer config)
        # - Nếu không gọi super() thì self.fields chưa tồn tại -> lỗi
        super().__init__(*args, **kwargs)
        
        # ============================================================
        # DANH SÁCH CÁC FIELD CẦN BẮT BUỘC KHI "CREATE"
        # ============================================================
        # Mục đích:
        # - Khi client gửi request tạo tài khoản, nếu thiếu 1 trong các field này
        #   DRF sẽ báo lỗi validation ngay:
        #   "This field is required."
        required_fields = ["name", "userName", "password", "email", "role"]

        # ============================================================
        # DUYỆT QUA CÁC FIELD VÀ SET required=True NẾU TỒN TẠI
        # ============================================================
        # Vì sao phải check `if field_name in self.fields`?
        # - BaseSerializerLogin có fields='__all__' nên thường sẽ có đủ,
        #   nhưng trong một số tình huống bạn có thể:
        #   + override fields ở serializer khác
        #   + exclude một số field
        # => check tồn tại để tránh KeyError
        for field_name in required_fields:
            if field_name in self.fields:
                # required=True:
                # - Khi deserialize (validate request input),
                #   nếu key field_name không có trong request -> lỗi required
                # - Điều này đặc biệt quan trọng với create/register
                self.fields[field_name].required = True
