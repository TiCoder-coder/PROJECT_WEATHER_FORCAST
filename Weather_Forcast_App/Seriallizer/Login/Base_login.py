from rest_framework import serializers
from Weather_Forcast_App.Models.Login import LoginModel
from bson import ObjectId

# ============================================================
# ObjectIdField: CUSTOM FIELD CHO BSON ObjectId (MongoDB)
# ============================================================
# Trong Django REST Framework (DRF):
# - Serializer dùng để "serialize" (chuyển object -> JSON) và "deserialize" (JSON -> object)
# - MongoDB dùng kiểu _id là ObjectId (không phải int/string bình thường)
# - DRF mặc định KHÔNG hiểu ObjectId => cần custom field để:
#   (1) Khi trả response: ObjectId -> string
#   (2) Khi nhận request: string -> ObjectId
#
# Mục tiêu: làm cho API làm việc "mượt" với MongoDB, frontend chỉ cần xử lý string.
class ObjectIdField(serializers.Field):
    # ============================================================
    # to_representation: OBJECT -> JSON (OUTPUT)
    # ============================================================
    # - Khi DRF trả dữ liệu ra ngoài (response JSON),
    #   nếu gặp trường kiểu ObjectId thì hàm này được gọi.
    # - Ta convert ObjectId -> str để JSON encode được.
    #   Ví dụ: ObjectId("65f...") -> "65f..."
    def to_representation(self, value):
        return str(value)
    
    # ============================================================
    # to_internal_value: JSON -> PYTHON (INPUT)
    # ============================================================
    # - Khi DRF nhận dữ liệu từ client (request JSON),
    #   field này sẽ chuyển "data" (thường là string) thành ObjectId thật.
    # - Nếu string không đúng format ObjectId -> raise ValidationError
    #   để client biết nhập sai.
    def to_internal_value(self, data):
        try:
            # ObjectId(data): convert string => ObjectId
            # Ví dụ: "65f..." -> ObjectId("65f...")
            return ObjectId(data)
        except Exception:
            # Nếu parse thất bại (sai độ dài/format) => báo lỗi validation
            raise serializers.ValidationError("Invalid ObjectId format")

# ============================================================
# BaseSerializerLogin: SERIALIZER CHO LOGIN MODEL
# ============================================================
# - serializers.ModelSerializer: loại serializer tự map theo model,
#   tự sinh field dựa trên model (LoginModel)
# - Bạn cấu hình Meta để chỉ rõ model và fields
class BaseSerializerLogin(serializers.ModelSerializer):
    class Meta:
        # model: model mà serializer sẽ map vào
        model = LoginModel

        # fields = '__all__' nghĩa là:
        # - serialize/deserialze TẤT CẢ các trường trong LoginModel
        # - Dễ dùng, nhanh, nhưng lưu ý:
        #   + Nếu model có field nhạy cảm (password/hash, token, ...) thì
        #     cần cân nhắc loại bỏ khi trả response (tuỳ mục đích serializer)
        fields = '__all__'
