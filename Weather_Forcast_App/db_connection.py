"""
Module quản lý kết nối MongoDB tập trung cho toàn bộ ứng dụng.
Tự động tìm PRIMARY node trong replica set.

Giải thích tổng quan:
- Module này đóng vai trò "DB Connection Manager" (quản lý kết nối DB)
- Hỗ trợ MongoDB Replica Set:
  + PRIMARY: node ghi (write) chính
  + SECONDARY: node đọc (read) phụ
- Vấn đề thực tế:
  + PRIMARY có thể thay đổi (failover)
  + Nếu app đang nối vào node cũ (không còn PRIMARY) -> NotPrimaryError
- Giải pháp:
  + Ping/hello để kiểm tra node hiện tại có phải PRIMARY không
  + Nếu không -> auto-discovery PRIMARY qua danh sách port
  + Có retry, delay để chịu lỗi tốt hơn
- Ngoài ra có:
  + create_index_safe: tạo index an toàn có retry
  + transaction wrapper: transaction chuẩn (snapshot + majority)
  + run_in_transaction: chạy hàm trong transaction và truyền session
"""
from pymongo import MongoClient
from pymongo.errors import NotPrimaryError, ServerSelectionTimeoutError
from decouple import config
import time
from contextlib import contextmanager
from pymongo.read_concern import ReadConcern
from pymongo.write_concern import WriteConcern
from pymongo.read_preferences import ReadPreference

# ============================================================
# CẤU HÌNH PORTS CỦA REPLICA SET LOCAL
# ============================================================
# - Đây là danh sách port mà các node trong replica set đang chạy.
# - Bạn giả định MongoDB replica set chạy trên localhost với 3 node:
#   27108, 27109, 27110
# - find_primary_port() sẽ loop qua đây để tìm PRIMARY
REPLICA_SET_PORTS = [27108, 27109, 27110]


def find_primary_port():
    """Tìm port của PRIMARY node trong replica set

    Cơ chế:
    - Thử kết nối từng port trong REPLICA_SET_PORTS
    - Gọi lệnh admin.command('hello') để lấy thông tin node
      (hello là lệnh mới thay thế isMaster ở các MongoDB mới)
    - Nếu node trả về:
      + isWritablePrimary == True  (chuẩn mới)
      hoặc
      + ismaster == True           (trường legacy)
      => node đó là PRIMARY

    Trả về:
    - port (int) nếu tìm được
    - None nếu không tìm thấy node PRIMARY nào
    """
    for port in REPLICA_SET_PORTS:
        try:
            # directConnection=true:
            # - ép client kết nối trực tiếp vào node ở port đó
            # - không phụ thuộc discovery topology qua seed list
            uri = f"mongodb://localhost:{port}/Login?directConnection=true"

            # serverSelectionTimeoutMS=3000:
            # - tối đa 3s để chọn server, nếu không được sẽ timeout (nhanh fail)
            client = MongoClient(uri, serverSelectionTimeoutMS=3000)

            # hello: lấy info node (role primary/secondary, etc.)
            result = client.admin.command('hello')

            # isWritablePrimary (new) hoặc ismaster (old) => PRIMARY
            if result.get('isWritablePrimary', False) or result.get('ismaster', False):
                client.close()
                print(f"Found PRIMARY at port {port}")
                return port

            # Nếu không phải PRIMARY -> đóng client và thử port khác
            client.close()
        except Exception:
            # Bất kỳ lỗi nào (node down, timeout...) -> bỏ qua và thử tiếp
            continue
    return None


class MongoDBConnection:
    # ============================================================
    # SINGLETON-LIKE CLASS (QUẢN LÝ 1 KẾT NỐI DÙNG CHUNG)
    # ============================================================
    # - _instance: đảm bảo chỉ có 1 instance (pattern Singleton)
    # - _client: MongoClient đang dùng (được cache)
    # - _db: Database object (cache)
    # - _current_port: port PRIMARY hiện tại (nếu auto-discovery bằng port local)
    _instance = None
    _client = None
    _db = None
    _current_port = None

    def __new__(cls):
        # __new__ được gọi khi tạo instance.
        # - Nếu _instance chưa có -> tạo mới
        # - Nếu đã có -> trả lại instance cũ (singleton)
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def reset_connection(cls):
        """Reset connection để tìm PRIMARY mới

        Khi nào cần reset?
        - Node đang kết nối bị mất kết nối
        - Node không còn là PRIMARY (failover)
        - Lỗi NotPrimaryError / ServerSelectionTimeoutError

        Tác dụng:
        - Đóng client cũ (nếu có)
        - Xoá cache _client/_db/_current_port để lần sau get_client() sẽ reconnect
        """
        if cls._client:
            try:
                cls._client.close()
            except:
                # Nếu close lỗi cũng bỏ qua, tránh crash
                pass
        cls._client = None
        cls._db = None
        cls._current_port = None

    @classmethod
    def get_client(cls, max_retries=5, retry_delay=3):
        """Lấy MongoDB client với auto-discovery PRIMARY

        Luồng hoạt động:
        1) Nếu _client đã có:
           - ping thử
           - nếu OK -> trả luôn client cũ (tối ưu performance)
           - nếu lỗi NotPrimary/Timeout -> reset và reconnect
        2) Thử kết nối mới tối đa max_retries lần:
           - Lấy MONGO_URI từ env
           - Kết nối MongoClient
           - hello để check node có phải PRIMARY không
           - Nếu không phải PRIMARY -> tìm PRIMARY bằng find_primary_port()
           - Kết nối lại vào PRIMARY và ping OK thì return client

        Tham số:
        - max_retries: số lần thử reconnect trước khi raise lỗi
        - retry_delay: delay (giây) giữa các lần retry
        """
        # Nếu đã có client cache thì thử ping trước
        if cls._client is not None:
            try:
                cls._client.admin.command('ping')
                return cls._client
            except (NotPrimaryError, ServerSelectionTimeoutError):
                print("Connection lost or node is no longer PRIMARY. Reconnecting...")
                cls.reset_connection()

        # Thử reconnect nhiều lần để chịu lỗi network/node failover
        for attempt in range(max_retries):
            try:
                # Lấy URI từ ENV (python-decouple)
                # - config("MONGO_URI") sẽ đọc từ .env hoặc env system
                mongo_uri = config("MONGO_URI")
                
                try:
                    # ============================================================
                    # THỬ KẾT NỐI THEO URI CẤU HÌNH
                    # ============================================================
                    # Các timeout:
                    # - serverSelectionTimeoutMS: thời gian tối đa chọn server (topology)
                    # - connectTimeoutMS: timeout khi bắt tay TCP
                    # - socketTimeoutMS: timeout khi đọc/ghi socket
                    #
                    # retryWrites=True + w="majority":
                    # - retryWrites: tự retry các thao tác write an toàn (tuỳ server hỗ trợ)
                    # - w=majority: ghi phải được đa số node xác nhận (an toàn hơn)
                    cls._client = MongoClient(
                        mongo_uri,
                        serverSelectionTimeoutMS=5000,
                        connectTimeoutMS=5000,
                        socketTimeoutMS=20000,
                        retryWrites=True,
                        w="majority"
                    )

                    # Gọi hello để "warm up" và lấy thông tin node
                    cls._client.admin.command('hello')
                    result = cls._client.admin.command('hello')

                    # Nếu node không phải PRIMARY -> ném NotPrimaryError để vào except bên dưới
                    if not (result.get('isWritablePrimary', False) or result.get('ismaster', False)):
                        raise NotPrimaryError("Connected node is not PRIMARY")

                except (NotPrimaryError, ServerSelectionTimeoutError):
                    # ============================================================
                    # NẾU URI CẤU HÌNH TRỎ VÀO NODE KHÔNG PHẢI PRIMARY
                    # => TỰ TÌM PRIMARY THẬT SỰ
                    # ============================================================
                    print("Configured URI is not PRIMARY. Auto-discovering PRIMARY...")
                    primary_port = find_primary_port()
                    if primary_port:
                        # Kết nối trực tiếp đến PRIMARY tìm được
                        mongo_uri = f"mongodb://localhost:{primary_port}/Login?directConnection=true"
                        cls._client = MongoClient(
                            mongo_uri,
                            serverSelectionTimeoutMS=5000,
                            connectTimeoutMS=5000,
                            socketTimeoutMS=20000,
                            retryWrites=True,
                            w="majority"
                        )
                        # lưu lại port primary hiện tại (để debug/monitor)
                        cls._current_port = primary_port
                    else:
                        # Không tìm được PRIMARY nào -> raise lỗi để retry/hoặc fail
                        raise Exception("Could not find PRIMARY node in replica set")
                
                # ============================================================
                # PING ĐỂ CHẮC CHẮN KẾT NỐI ĐÃ OK
                # ============================================================
                cls._client.admin.command('ping')
                print("MongoDB connection established successfully!")
                return cls._client
                
            except Exception as e:
                # ============================================================
                # RETRY LOGIC
                # ============================================================
                # - Nếu chưa phải attempt cuối -> in log + sleep rồi thử lại
                # - Nếu attempt cuối -> raise lỗi ra ngoài
                if attempt < max_retries - 1:
                    print(f"MongoDB connection attempt {attempt + 1}/{max_retries} failed: {e}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Failed to connect to MongoDB after {max_retries} attempts")
                    raise e

    @classmethod
    def get_database(cls):
        """Lấy database instance

        - Cache _db để dùng lại (tránh tạo lại object database)
        - Nếu _db chưa có:
          + gọi get_client() để đảm bảo client kết nối OK
          + lấy db theo DB_NAME từ env
        """
        if cls._db is None:
            client = cls.get_client()
            cls._db = client[config("DB_NAME")]
        return cls._db

    @classmethod
    def create_index_safe(cls, collection, keys, **kwargs):
        """Tạo index với retry logic và error handling

        Mục tiêu:
        - Khi chạy app nhiều lần, index có thể đã tồn tại -> không crash
        - Khi PRIMARY đổi trong lúc tạo index -> retry
        - Khi có lỗi tạm thời -> retry vài lần

        Tham số:
        - collection: pymongo collection
        - keys: list các tuple (field, direction) hoặc dạng phù hợp pymongo
        - **kwargs: các option như unique=True, name=..., sparse=True,...

        Trả về:
        - True nếu tạo thành công hoặc index đã tồn tại
        - False nếu thất bại sau nhiều lần thử
        """
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                # create_index: tạo index trên MongoDB
                collection.create_index(keys, **kwargs)
                return True

            except NotPrimaryError:
                # Nếu node đang kết nối không còn PRIMARY:
                # - reset connection để lần sau reconnect đúng PRIMARY
                cls.reset_connection()
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

            except Exception as e:
                error_msg = str(e)

                # Nếu index "already exists" thì coi như OK (idempotent)
                if "already exists" in error_msg.lower():
                    return True

                # Nếu còn lượt retry -> in log và thử lại
                if attempt < max_retries - 1:
                    print(f"Index creation attempt {attempt + 1} failed: {e}")
                    time.sleep(retry_delay)
                else:
                    # Hết retry -> cảnh báo và return False (không crash app)
                    print(f"Warning: Could not create index after {max_retries} attempts: {e}")
                    return False
        return False

    @classmethod
    def start_session(cls):
        # ============================================================
        # TẠO SESSION (dùng cho transaction)
        # ============================================================
        # - MongoDB transaction cần session
        # - start_session() sẽ tạo ClientSession từ client hiện tại
        client = cls.get_client()
        return client.start_session()

    @classmethod
    @contextmanager
    def transaction(cls):
        """
        Transaction chuẩn:
        - majority write concern
        - snapshot read concern
        - primary read

        Giải thích nhanh 3 thành phần:
        - ReadConcern("snapshot"):
          + đảm bảo snapshot nhất quán trong transaction (đọc "ảnh chụp" dữ liệu)
        - WriteConcern("majority"):
          + commit chỉ thành công khi đa số node ack (an toàn dữ liệu)
          + wtimeout=5000: timeout cho write ack (ms)
        - ReadPreference.PRIMARY:
          + đọc từ primary để nhất quán trong transaction
        """
        # with start_session() as session:
        # - context manager đảm bảo session đóng đúng cách
        with cls.start_session() as session:
            try:
                # start_transaction cấu hình transaction
                session.start_transaction(
                    read_concern=ReadConcern("snapshot"),
                    write_concern=WriteConcern("majority", wtimeout=5000),
                    read_preference=ReadPreference.PRIMARY
                )

                # yield session:
                # - trả session ra ngoài để code caller dùng session trong các query
                yield session

                # Nếu không có exception -> commit
                session.commit_transaction()

            except (NotPrimaryError, ServerSelectionTimeoutError):
                # PRIMARY đổi / mất kết nối trong lúc transaction:
                # - abort để rollback
                # - reset connection để lần sau reconnect
                try:
                    session.abort_transaction()
                except:
                    pass
                cls.reset_connection()
                raise

            except Exception:
                # Các exception khác:
                # - abort để rollback
                try:
                    session.abort_transaction()
                except:
                    pass
                raise

    @classmethod
    def run_in_transaction(cls, fn, *args, **kwargs):
        """
        Gọi hàm fn trong transaction.
        fn PHẢI nhận param session=...

        Mục tiêu:
        - Bạn viết logic business trong fn (vd: tạo user + tạo profile)
        - run_in_transaction sẽ:
          + mở transaction
          + gọi fn(..., session=session)
          + commit/abort tự động
        """
        with cls.transaction() as session:
            return fn(*args, session=session, **kwargs)


# ============================================================
# CÁC HÀM "WRAPPER" Ở NGOÀI CLASS (TIỆN IMPORT/USE)
# ============================================================
# - Thay vì gọi MongoDBConnection.get_client() khắp nơi,
#   bạn có thể import get_mongo_client/get_database/create_index_safe...
# - Tách lớp quản lý và API dùng tiện cho code khác.
# ============================================================

# Singleton instances
def get_mongo_client():
    # Trả về MongoClient đã được quản lý (có auto-discovery PRIMARY)
    return MongoDBConnection.get_client()


def get_database():
    # Trả về database object (client[DB_NAME])
    return MongoDBConnection.get_database()

def create_index_safe(collection, keys, **kwargs):
    # Wrapper tạo index an toàn
    return MongoDBConnection.create_index_safe(collection, keys, **kwargs)

def start_session():
    # Wrapper tạo session
    return MongoDBConnection.start_session()

def transaction():
    # Wrapper context manager transaction
    return MongoDBConnection.transaction()

def run_in_transaction(fn, *args, **kwargs):
    # Wrapper chạy hàm trong transaction (tự truyền session)
    return MongoDBConnection.run_in_transaction(fn, *args, **kwargs)
