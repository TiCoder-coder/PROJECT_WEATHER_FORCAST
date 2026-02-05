from pymongo import ASCENDING
from bson import ObjectId
from Weather_Forcast_App.db_connection import get_database, create_index_safe

# ============================================================
# KẾT NỐI DATABASE + LẤY COLLECTION
# ============================================================

# get_database(): hàm (bạn tự định nghĩa) để lấy object Database của MongoDB.
# Thường nó sẽ:
# - đọc cấu hình (URI, DB_NAME) từ env/settings
# - tạo MongoClient
# - trả về db = client[DB_NAME]
db = get_database()

# "logins" là tên collection trong MongoDB.
# Collection này thường dùng để lưu thông tin tài khoản đăng nhập:
# ví dụ: userName, email, password_hash, roles, createdAt, ...
login_collection = db["logins"]

# ============================================================
# TẠO INDEX (CHỈ MỤC) AN TOÀN
# ============================================================

# create_index_safe(collection, keys, unique=True/False):
# - Đây là helper bạn tự viết để tạo index mà không làm crash app nếu index đã tồn tại
# - Tránh lỗi khi server chạy nhiều lần, hoặc deploy nhiều lần
# - keys là list các cặp (fieldName, direction)
# - unique=True nghĩa là MongoDB sẽ đảm bảo giá trị của field đó KHÔNG bị trùng
#
# Index userName:
# - Tối ưu truy vấn find_one({"userName": ...})
# - unique=True: chặn 2 user cùng userName
create_index_safe(login_collection, [("userName", ASCENDING)], unique=True)

# Index email:
# - Tối ưu truy vấn find_one({"email": ...}) và truy vấn login bằng email
# - unique=True: chặn 2 user cùng email
create_index_safe(login_collection, [("email", ASCENDING)], unique=True)

# ============================================================
# REPOSITORY LAYER (DATA ACCESS LAYER)
# ============================================================

# LoginRepository là lớp "repository" theo pattern:
# - Đóng gói tất cả thao tác CRUD với collection "logins"
# - Giúp tách bạch logic truy cập DB khỏi service/controller/view
# - Dễ test hơn, dễ thay đổi DB hơn về sau (nếu cần)
#
# Dùng @staticmethod để gọi trực tiếp:
# LoginRepository.find_by_username("abc")
# thay vì phải tạo object LoginRepository()
class LoginRepository:
    @staticmethod
    def insert_one(data: dict, session=None):
        # insert_one:
        # - Thêm 1 document mới vào collection
        # - data là dict Python (sẽ được chuyển thành BSON)
        # - Trả về InsertOneResult (có inserted_id)
        #
        # session:
        # - là ClientSession của MongoDB (nếu bạn đang dùng transaction)
        # - nếu không truyền thì MongoDB chạy bình thường (không transaction)
        #
        # Lưu ý:
        # - do có unique index userName/email, nếu trùng sẽ ném lỗi DuplicateKeyError
        return login_collection.insert_one(data, session=session)

    @staticmethod
    def find_all():
        # find():
        # - Truy vấn toàn bộ document trong collection (không filter)
        # list(...) để convert cursor -> list object Python
        #
        # Lưu ý:
        # - Nếu collection lớn, find_all() sẽ nặng (RAM/time)
        # - Thường nên phân trang (skip/limit) trong thực tế
        return list(login_collection.find())

    @staticmethod
    def find_by_id(user_id):
        # find_one({"_id": ...}):
        # - Tìm 1 document theo _id (primary key của MongoDB)
        #
        # ObjectId(str(user_id)):
        # - MongoDB lưu _id dạng ObjectId
        # - user_id có thể đang là:
        #   + ObjectId
        #   + string (ví dụ: "65f...")
        # => str(user_id) ép về string để chắc chắn
        # => ObjectId(...) convert thành ObjectId đúng kiểu
        #
        # Lưu ý quan trọng:
        # - Nếu user_id không đúng format ObjectId, ObjectId(...) sẽ raise Exception
        return login_collection.find_one({"_id": ObjectId(str(user_id))})

    @staticmethod
    def find_by_username(userName: str):
        # Tìm 1 document theo userName
        # - Có index userName nên truy vấn nhanh
        # - Trả về document dict hoặc None nếu không tìm thấy
        return login_collection.find_one({"userName": userName})

    @staticmethod
    def find_by_username_or_email(identifier: str):
        # Đăng nhập thường cho phép nhập "username hoặc email"
        # => dùng toán tử $or của MongoDB:
        # {"$or": [{"userName": identifier}, {"email": identifier}]}
        #
        # MongoDB sẽ trả về document đầu tiên match (find_one)
        # - Nếu identifier trùng cả userName và email ở 2 user khác nhau thì unique index giúp tránh trường hợp này
        # - Có index ở cả userName và email nên query thường ổn
        return login_collection.find_one({"$or": [{"userName": identifier}, {"email": identifier}]})

    @staticmethod
    def update_by_id(user_id, update_data: dict, session=None):
        # update_one(filter, update, session=?):
        # - filter: {"_id": ObjectId(...)} tìm đúng user
        # - update: {"$set": update_data} chỉ cập nhật các field có trong update_data
        #   (không thay toàn bộ document)
        #
        # session:
        # - nếu có transaction thì truyền session vào để update nằm trong transaction
        #
        # Trả về UpdateResult:
        # - matched_count: số doc match filter
        # - modified_count: số doc thực sự bị sửa
        return login_collection.update_one({"_id": ObjectId(str(user_id))}, {"$set": update_data}, session=session)

    @staticmethod
    def delete_by_id(user_id, session=None):
        # delete_one(filter, session=?):
        # - Xoá 1 document theo _id
        # - Trả về DeleteResult (deleted_count)
        #
        # session:
        # - nếu xoá trong transaction thì truyền session vào
        return login_collection.delete_one({"_id": ObjectId(str(user_id))}, session=session)
