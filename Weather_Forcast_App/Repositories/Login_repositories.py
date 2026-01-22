from pymongo import ASCENDING
from bson import ObjectId
from Weather_Forcast_App.db_connection import get_database, create_index_safe

db = get_database()
login_collection = db["logins"]

# Tạo indexes an toàn
create_index_safe(login_collection, [("userName", ASCENDING)], unique=True)
create_index_safe(login_collection, [("email", ASCENDING)], unique=True)

class LoginRepository:
    @staticmethod
    def insert_one(data: dict):
        return login_collection.insert_one(data)

    @staticmethod
    def find_all():
        return list(login_collection.find())

    @staticmethod
    def find_by_id(user_id):
        return login_collection.find_one({"_id": ObjectId(str(user_id))})

    @staticmethod
    def find_by_username(userName: str):
        return login_collection.find_one({"userName": userName})

    @staticmethod
    def find_by_username_or_email(identifier: str):
        return login_collection.find_one({"$or": [{"userName": identifier}, {"email": identifier}]})

    @staticmethod
    def update_by_id(user_id, update_data: dict):
        return login_collection.update_one({"_id": ObjectId(str(user_id))}, {"$set": update_data})

    @staticmethod
    def delete_by_id(user_id):
        return login_collection.delete_one({"_id": ObjectId(str(user_id))})
