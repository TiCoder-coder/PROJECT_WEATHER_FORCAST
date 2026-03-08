# 🗃️ Repositories — Data Access Layer

## 📁 Overview

This directory implements the **Repository Pattern** — a design pattern that **encapsulates all database operations** and provides a clean API for data access.

---

## 📂 Directory Structure

```
Repositories/
├── __init__.py              # Package initializer  
├── Login_repositories.py    # User/auth data access
└── README.md                # This file
```

---

## 🎯 Purpose

### What is the Repository Pattern?

The Repository Pattern **separates data access logic** from business logic by providing:
- **Single source of truth** for database operations
- **Testable code** (can mock repository in tests)
- **Database agnostic** (easy to switch from MongoDB to PostgreSQL)
- **Reusable queries** (no duplicate code in views)

### Benefits

| Benefit | Description |
|---------|-------------|
| **Separation of Concerns** | Views don't know about MongoDB |
| **Maintainability** | Change all `find_by_username()` calls in one place |
| **Testability** | Mock repository without real database |
| **Performance** | Centralized query optimization |

---

## 📄 Files Explained

### `Login_repositories.py` — User Data Access

**Purpose**: All MongoDB operations for `logins` collection are centralized here.

#### Code Structure

```python
from pymongo import ASCENDING
from bson import ObjectId
from Weather_Forcast_App.db_connection import get_database, create_index_safe

# ==================== LAZY INITIALIZATION ====================

_login_collection = None

def _get_login_collection():
    """
    Lazy getter for login collection.
    Only connects to MongoDB when first called (avoids startup errors).
    """
    global _login_collection
    if _login_collection is None:
        db = get_database()
        _login_collection = db["logins"]
        
        # Create unique indexes for fast lookups
        create_index_safe(_login_collection, [("userName", ASCENDING)], unique=True)
        create_index_safe(_login_collection, [("email", ASCENDING)], unique=True)
    
    return _login_collection
```

---

## 🔧 Repository Methods

### `LoginRepository` Class

All methods are **static** (no need to instantiate):

```python
class LoginRepository:
    # No __init__ needed — all @staticmethod
    pass
```

---

### ✏️ Create Operations

#### `insert_one(data, session=None) → InsertOneResult`

```python
@staticmethod
def insert_one(data: dict, session=None):
    """
    Insert a single user document.
    
    Args:
        data: Dict with user fields (name, userName, email, password, role)
        session: MongoDB session for transactions (optional)
    
    Returns:
        InsertOneResult with .inserted_id
    
    Raises:
        DuplicateKeyError: userName or email already exists
    
    Example:
        result = LoginRepository.insert_one({
            "name": "Võ Anh Nhật",
            "userName": "nhat123",
            "email": "nhat@gmail.com",
            "password": "hashed_password",
            "role": "Admin",
            "is_active": True,
            "failed_attempts": 0,
            "createdAt": datetime.now()
        })
        print(result.inserted_id)  # ObjectId("...")
    """
    return _get_login_collection().insert_one(data, session=session)
```

---

### 📖 Read Operations

#### `find_all() → list[dict]`

```python
@staticmethod
def find_all():
    """
    Get all users in the collection.
    
    ⚠️ WARNING: Can be slow/memory-intensive on large collections!
    Consider using pagination in production.
    
    Returns:
        List of user documents (dicts)
    
    Example:
        users = LoginRepository.find_all()
        for user in users:
            print(user["userName"], user["role"])
    """
    return list(_get_login_collection().find())
```

#### `find_by_username(username: str) → dict | None`

```python
@staticmethod
def find_by_username(username: str):
    """
    Find user by username (exact match, case-sensitive).
    
    Args:
        username: Username to search for
    
    Returns:
        User document dict or None if not found
    
    Example:
        user = LoginRepository.find_by_username("nhat123")
        if user:
            print(user["email"])
        else:
            print("User not found")
    """
    return _get_login_collection().find_one({"userName": username})
```

#### `find_by_email(email: str) → dict | None`

```python
@staticmethod
def find_by_email(email: str):
    """
    Find user by email (exact match, case-sensitive).
    
    Args:
        email: Email to search for
    
    Returns:
        User document dict or None if not found
    
    Example:
        user = LoginRepository.find_by_email("nhat@gmail.com")
        if user:
            print(user["userName"])
    """
    return _get_login_collection().find_one({"email": email})
```

#### `find_by_id(user_id: str) → dict | None`

```python
@staticmethod
def find_by_id(user_id: str):
    """
    Find user by MongoDB ObjectId.
    
    Args:
        user_id: String representation of ObjectId
    
    Returns:
        User document dict or None if not found
    
    Example:
        user = LoginRepository.find_by_id("507f1f77bcf86cd799439011")
        if user:
            print(user["name"])
    """
    return _get_login_collection().find_one({"_id": ObjectId(user_id)})
```

---

### 🔄 Update Operations

#### `update_one(filter_dict, update_dict, session=None) → UpdateResult`

```python
@staticmethod
def update_one(filter_dict: dict, update_dict: dict, session=None):
    """
    Update a single user document.
    
    Args:
        filter_dict: Filter to find document (e.g., {"userName": "nhat123"})
        update_dict: Update operations (e.g., {"$set": {"role": "Manager"}})
        session: MongoDB session for transactions
    
    Returns:
        UpdateResult with .modified_count
    
    Example:
        # Reset failed login attempts
        result = LoginRepository.update_one(
            {"userName": "nhat123"},
            {"$set": {"failed_attempts": 0, "lock_until": None}}
        )
        print(result.modified_count)  # 1 if successful
    """
    return _get_login_collection().update_one(
        filter_dict, 
        update_dict, 
        session=session
    )
```

#### `update_by_id(user_id: str, update_dict, session=None) → UpdateResult`

```python
@staticmethod
def update_by_id(user_id: str, update_dict: dict, session=None):
    """
    Update user by ObjectId.
    
    Args:
        user_id: String ObjectId
        update_dict: Update operations
        session: MongoDB session
    
    Returns:
        UpdateResult
    
    Example:
        LoginRepository.update_by_id(
            "507f1f77bcf86cd799439011",
            {"$set": {"last_login": datetime.now()}}
        )
    """
    return _get_login_collection().update_one(
        {"_id": ObjectId(user_id)},
        update_dict,
        session=session
    )
```

---

### 🗑️ Delete Operations

#### `delete_one(filter_dict, session=None) → DeleteResult`

```python
@staticmethod
def delete_one(filter_dict: dict, session=None):
    """
    Delete a single user document.
    
    Args:
        filter_dict: Filter to find document
        session: MongoDB session
    
    Returns:
        DeleteResult with .deleted_count
    
    Example:
        result = LoginRepository.delete_one({"userName": "test_user"})
        print(result.deleted_count)  # 1 if deleted, 0 if not found
    """
    return _get_login_collection().delete_one(filter_dict, session=session)
```

---

## 💡 Usage Examples

### Example 1: User Registration

```python
from Weather_Forcast_App.Repositories.Login_repositories import LoginRepository
from datetime import datetime

def register_user(data):
    # Check if username exists
    if LoginRepository.find_by_username(data["userName"]):
        return {"error": "Username already taken"}
    
    # Check if email exists
    if LoginRepository.find_by_email(data["email"]):
        return {"error": "Email already registered"}
    
    # Insert new user
    user_doc = {
        "name": data["name"],
        "userName": data["userName"],
        "email": data["email"],
        "password": hash_password(data["password"]),
        "role": "Staff",
        "is_active": True,
        "failed_attempts": 0,
        "createdAt": datetime.now(),
        "updatedAt": datetime.now()
    }
    
    result = LoginRepository.insert_one(user_doc)
    return {"success": True, "user_id": str(result.inserted_id)}
```

### Example 2: Login with Lockout

```python
from datetime import datetime, timedelta

def login_user(username, password):
    # Find user
    user = LoginRepository.find_by_username(username)
    if not user:
        return {"error": "Invalid username or password"}
    
    # Check if locked
    if user.get("lock_until") and user["lock_until"] > datetime.now():
        remaining = (user["lock_until"] - datetime.now()).seconds
        return {"error": f"Account locked. Try again in {remaining}s"}
    
    # Verify password
    if not verify_password(password, user["password"]):
        # Increment failed attempts
        failed_attempts = user.get("failed_attempts", 0) + 1
        update_dict = {"$set": {"failed_attempts": failed_attempts}}
        
        # Lock if threshold reached
        if failed_attempts >= 5:
            update_dict["$set"]["lock_until"] = datetime.now() + timedelta(minutes=5)
        
        LoginRepository.update_one({"userName": username}, update_dict)
        return {"error": "Invalid username or password"}
    
    # Successful login — reset failures
    LoginRepository.update_one(
        {"userName": username},
        {"$set": {
            "failed_attempts": 0,
            "lock_until": None,
            "last_login": datetime.now()
        }}
    )
    
    return {"success": True, "user": user}
```

### Example 3: List All Admins

```python
def get_all_admins():
    all_users = LoginRepository.find_all()
    admins = [u for u in all_users if u["role"] == "Admin"]
    return admins
```

---

## 🔍 Best Practices

### ✅ DO

- **Use repositories** for all database operations (never call MongoDB directly in views)
- **Handle exceptions** (DuplicateKeyError, ConnectionError, etc.)
- **Use transactions** for multi-document operations
- **Add indexes** for frequently queried fields

### ❌ DON'T

- **Don't query in views** — always use repository
- **Don't return full documents** with passwords (sanitize sensitive fields)
- **Don't use `find()` without limits** on large collections
- **Don't hardcode collection names** outside repository

---

## 🚀 Future Enhancements

- [ ] Add **pagination methods** (`find_with_pagination(skip, limit)`)
- [ ] Add **search methods** (`search_by_name(query)`)
- [ ] Add **bulk operations** (`insert_many`, `update_many`)
- [ ] Add **aggregation pipelines** (`get_user_stats()`)
- [ ] Add **soft delete** (`mark_as_deleted()` instead of delete)
- [ ] Add **caching layer** (Redis for frequently accessed users)

---

## 📞 Related Files

- **Models**: `Models/Login.py` (model definition)
- **Views**: `views/View_login.py` (calls repository)
- **Database**: `db_connection.py` (MongoDB connection)
- **Services**: `scripts/Login_services.py` (business logic using repository)

---

## 👨‍💻 Maintainer

**Võ Anh Nhật**  
📧 voanhnhat1612@gmail.com

---

*Last Updated: March 8, 2026*
