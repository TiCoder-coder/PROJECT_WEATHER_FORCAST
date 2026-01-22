"""
Module quản lý kết nối MongoDB tập trung cho toàn bộ ứng dụng.
Tự động tìm PRIMARY node trong replica set.
"""
from pymongo import MongoClient
from pymongo.errors import NotPrimaryError, ServerSelectionTimeoutError
from decouple import config
import time


# Danh sách các port của replica set
REPLICA_SET_PORTS = [27108, 27109, 27110]


def find_primary_port():
    """Tìm port của PRIMARY node trong replica set"""
    for port in REPLICA_SET_PORTS:
        try:
            uri = f"mongodb://localhost:{port}/Login?directConnection=true"
            client = MongoClient(uri, serverSelectionTimeoutMS=3000)
            # Kiểm tra xem node này có phải PRIMARY không
            result = client.admin.command('hello')
            if result.get('isWritablePrimary', False) or result.get('ismaster', False):
                client.close()
                print(f"Found PRIMARY at port {port}")
                return port
            client.close()
        except Exception:
            continue
    return None


class MongoDBConnection:
    _instance = None
    _client = None
    _db = None
    _current_port = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def reset_connection(cls):
        """Reset connection để tìm PRIMARY mới"""
        if cls._client:
            try:
                cls._client.close()
            except:
                pass
        cls._client = None
        cls._db = None
        cls._current_port = None

    @classmethod
    def get_client(cls, max_retries=5, retry_delay=3):
        """Lấy MongoDB client với auto-discovery PRIMARY"""
        if cls._client is not None:
            # Kiểm tra connection còn valid không
            try:
                cls._client.admin.command('ping')
                return cls._client
            except (NotPrimaryError, ServerSelectionTimeoutError):
                print("Connection lost or node is no longer PRIMARY. Reconnecting...")
                cls.reset_connection()

        for attempt in range(max_retries):
            try:
                # Thử với MONGO_URI từ config trước
                mongo_uri = config("MONGO_URI")
                
                # Nếu thất bại, tự động tìm PRIMARY
                try:
                    cls._client = MongoClient(
                        mongo_uri,
                        serverSelectionTimeoutMS=5000,
                        connectTimeoutMS=5000,
                        socketTimeoutMS=20000,
                        retryWrites=True,
                        w="majority"
                    )
                    # Test write operation
                    cls._client.admin.command('hello')
                    result = cls._client.admin.command('hello')
                    if not (result.get('isWritablePrimary', False) or result.get('ismaster', False)):
                        raise NotPrimaryError("Connected node is not PRIMARY")
                except (NotPrimaryError, ServerSelectionTimeoutError):
                    print("Configured URI is not PRIMARY. Auto-discovering PRIMARY...")
                    primary_port = find_primary_port()
                    if primary_port:
                        mongo_uri = f"mongodb://localhost:{primary_port}/Login?directConnection=true"
                        cls._client = MongoClient(
                            mongo_uri,
                            serverSelectionTimeoutMS=5000,
                            connectTimeoutMS=5000,
                            socketTimeoutMS=20000,
                            retryWrites=True,
                            w="majority"
                        )
                        cls._current_port = primary_port
                    else:
                        raise Exception("Could not find PRIMARY node in replica set")
                
                cls._client.admin.command('ping')
                print("MongoDB connection established successfully!")
                return cls._client
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"MongoDB connection attempt {attempt + 1}/{max_retries} failed: {e}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Failed to connect to MongoDB after {max_retries} attempts")
                    raise e

    @classmethod
    def get_database(cls):
        """Lấy database instance"""
        if cls._db is None:
            client = cls.get_client()
            cls._db = client[config("DB_NAME")]
        return cls._db

    @classmethod
    def create_index_safe(cls, collection, keys, **kwargs):
        """Tạo index với retry logic và error handling"""
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                collection.create_index(keys, **kwargs)
                return True
            except NotPrimaryError:
                # Reset connection và thử lại
                cls.reset_connection()
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
            except Exception as e:
                error_msg = str(e)
                # Bỏ qua nếu index đã tồn tại
                if "already exists" in error_msg.lower():
                    return True
                if attempt < max_retries - 1:
                    print(f"Index creation attempt {attempt + 1} failed: {e}")
                    time.sleep(retry_delay)
                else:
                    print(f"Warning: Could not create index after {max_retries} attempts: {e}")
                    return False
        return False


# Singleton instances
def get_mongo_client():
    return MongoDBConnection.get_client()


def get_database():
    return MongoDBConnection.get_database()


def create_index_safe(collection, keys, **kwargs):
    return MongoDBConnection.create_index_safe(collection, keys, **kwargs)
