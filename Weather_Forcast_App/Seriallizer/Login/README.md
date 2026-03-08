# 🔐 Seriallizer/Login — Authentication Serializers

## 📁 Overview

This directory contains **3 DRF serializers** for the `LoginModel`, each tailored for different API operations:

1. **Base_login.py** — Base serializer with ObjectId handling
2. **Create_login.py** — Registration serializer (strict validation)
3. **Update_login.py** — Profile update serializer (partial validation)

---

## 📂 Directory Structure

```
Seriallizer/Login/
├── __init__.py          # Package initializer
├── Base_login.py        # BaseSerializerLogin + ObjectIdField
├── Create_login.py      # LoginLoginCreate (registration)
├── Update_login.py      # LoginUpdate (profile update)
└── README.md            # This file
```

---

## 📄 Files Explained

### `Base_login.py` — Base Serializer + ObjectId Field

#### Purpose

- Define **base serializer** for `LoginModel`
- Handle **MongoDB's ObjectId** type (not natively supported by DRF)

#### Code Breakdown

```python
from rest_framework import serializers
from bson import ObjectId

# ============================================================
# CUSTOM FIELD FOR MONGODB ObjectId
# ============================================================
class ObjectIdField(serializers.Field):
    """
    Convert between MongoDB ObjectId and JSON string.
    
    - to_representation: ObjectId → str (output)
    - to_internal_value: str → ObjectId (input)
    """
    
    def to_representation(self, value):
        """Called when serializing (Python → JSON)"""
        return str(value)  # ObjectId("507f...") → "507f..."
    
    def to_internal_value(self, data):
        """Called when deserializing (JSON → Python)"""
        try:
            return ObjectId(data)  # "507f..." → ObjectId("507f...")
        except Exception:
            raise serializers.ValidationError("Invalid ObjectId format")

# ============================================================
# BASE SERIALIZER FOR LoginModel
# ============================================================
class BaseSerializerLogin(serializers.ModelSerializer):
    # Override _id field to use custom ObjectIdField
    _id = ObjectIdField(read_only=True)
    
    class Meta:
        model = LoginModel
        fields = "__all__"  # All model fields
        # Optionally: fields = ["_id", "userName", "email", ...]
```

#### Why ObjectIdField?

MongoDB uses `ObjectId("507f1f77bcf86cd799439011")` but JSON only supports strings. This field:
- **Output**: Converts `ObjectId` → `"507f1f77bcf86cd799439011"` in API responses
- **Input**: Converts `"507f..."` → `ObjectId(...)` when receiving requests

---

### `Create_login.py` — Registration Serializer

#### Purpose

Enforce **strict validation** for user registration — all critical fields must be present.

#### Code

```python
from .Base_login import BaseSerializerLogin

class LoginLoginCreate(BaseSerializerLogin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # ============================================================
        # REQUIRED FIELDS FOR REGISTRATION
        # ============================================================
        required_fields = ["name", "userName", "password", "email", "role"]
        
        for field_name in required_fields:
            if field_name in self.fields:
                self.fields[field_name].required = True
```

#### Validation Logic

| Field | Required? | Validation |
|-------|-----------|------------|
| **name** | ✅ Yes | Must be provided |
| **userName** | ✅ Yes | Must be unique (checked in repository) |
| **email** | ✅ Yes | Must be valid email + unique |
| **password** | ✅ Yes | Will be hashed before saving |
| **role** | ✅ Yes | Must be valid Role enum value |

#### Usage Example

```python
from Weather_Forcast_App.Seriallizer.Login.Create_login import LoginLoginCreate

@api_view(['POST'])
def register(request):
    serializer = LoginLoginCreate(data=request.data)
    
    if not serializer.is_valid():
        return Response({
            "error": "Validation failed",
            "details": serializer.errors  # {"userName": ["This field is required."]}
        }, status=400)
    
    # All required fields present
    data = serializer.validated_data
    
    # Hash password
    data["password"] = make_password(data["password"] + PEPPER)
    
    # Save to DB
    LoginRepository.insert_one(data)
    
    return Response({"success": True}, status=201)
```

#### Sample Request

```http
POST /api/register
Content-Type: application/json

{
  "name": "Võ Anh Nhật",
  "userName": "nhat123",
  "email": "nhat@gmail.com",
  "password": "SecurePass123!",
  "role": "Manager"
}
```

#### Sample Error Response (Missing Field)

```json
{
  "error": "Validation failed",
  "details": {
    "email": ["This field is required."]
  }
}
```

---

### `Update_login.py` — Profile Update Serializer

#### Purpose

Allow **partial updates** — users can update only specific fields without sending all data.

#### Code

```python
from .Base_login import BaseSerializerLogin

class LoginUpdate(BaseSerializerLogin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # ============================================================
        # MAKE ALL FIELDS OPTIONAL (PARTIAL UPDATE)
        # ============================================================
        for field in self.fields.values():
            field.required = False
        
        # ============================================================
        # REMOVE "id" FIELD (SECURITY)
        # ============================================================
        # Prevent client from sending/modifying ID in request body.
        # ID should come from URL path parameter, not body.
        self.fields.pop("id", None)  # Remove if exists
```

#### Validation Logic

| Field | Required? | Why? |
|-------|-----------|------|
| **All fields** | ❌ No | Allow partial updates (PATCH) |
| **id** | 🚫 Removed | Prevents ID tampering |

#### Usage Example

```python
from Weather_Forcast_App.Seriallizer.Login.Update_login import LoginUpdate

@api_view(['PATCH'])
def update_profile(request, user_id):
    # Find user
    user = LoginRepository.find_by_id(user_id)
    if not user:
        return Response({"error": "User not found"}, status=404)
    
    # Validate partial data
    serializer = LoginUpdate(data=request.data, partial=True)
    if not serializer.is_valid():
        return Response(serializer.errors, status=400)
    
    # Update only provided fields
    update_data = serializer.validated_data
    LoginRepository.update_by_id(user_id, {"$set": update_data})
    
    return Response({"success": True})
```

#### Sample Request (Partial Update)

```http
PATCH /api/users/507f1f77bcf86cd799439011
Content-Type: application/json

{
  "name": "Võ Anh Nhật (Updated)",
  "email": "new_email@gmail.com"
}
```

**Note**: Only `name` and `email` are updated. Other fields remain unchanged.

---

## 📊 Serializer Comparison

| Feature | BaseSerializerLogin | LoginLoginCreate | LoginUpdate |
|---------|---------------------|------------------|-------------|
| **Purpose** | Base class | User registration | Profile update |
| **Required Fields** | Model defaults | 5 required | All optional |
| **ID Field** | Included | Included | Removed |
| **Use Case** | Not used directly | POST /register | PATCH /users/:id |

---

## 🔧 Common Operations

### Operation 1: Create User (Registration)

```python
serializer = LoginLoginCreate(data={
    "name": "Test User",
    "userName": "testuser",
    "email": "test@example.com",
    "password": "password123",
    "role": "Staff"
})

if serializer.is_valid():
    data = serializer.validated_data
    # Hash password before saving
    data["password"] = make_password(data["password"])
    LoginRepository.insert_one(data)
```

### Operation 2: Update User (Profile Edit)

```python
serializer = LoginUpdate(data={
    "email": "newemail@example.com"
}, partial=True)

if serializer.is_valid():
    LoginRepository.update_one(
        {"userName": "testuser"},
        {"$set": serializer.validated_data}
    )
```

### Operation 3: Serialize User for Response

```python
user = LoginRepository.find_by_username("testuser")
serializer = BaseSerializerLogin(user)

# Remove password before sending
response_data = serializer.data
response_data.pop("password", None)

return Response(response_data)
```

---

## 🐛 Common Issues

### Issue 1: ObjectId Serialization Error

**Error**: `Object of type ObjectId is not JSON serializable`

**Cause**: Forgot to use `ObjectIdField`

**Solution**: Use `BaseSerializerLogin` which includes `ObjectIdField` for `_id`

### Issue 2: Required Field Missing

**Error**: `{"email": ["This field is required."]}`

**Cause**: Using `LoginLoginCreate` but didn't send all required fields

**Solution**: Ensure client sends `name`, `userName`, `email`, `password`, `role`

### Issue 3: Can't Update Specific Field

**Error**: Validation passes but field not updating

**Cause**: Using wrong serializer or not calling `$set` in MongoDB

**Solution**:
```python
# ❌ WRONG (replaces entire document)
LoginRepository.update_one(filter, validated_data)

# ✅ CORRECT (updates specific fields)
LoginRepository.update_one(filter, {"$set": validated_data})
```

---

## 🚀 Future Enhancements

- [ ] Add **password strength validator** (min 8 chars, uppercase, number, special)
- [ ] Add **email format validator** (custom regex for company emails)
- [ ] Add **username sanitization** (no spaces, special chars)
- [ ] Add **role choices validation** (validate against Role enum)
- [ ] Add **nested serializers** for related data (e.g., user profile)

---

## 📞 Related Files

- **Model**: `Models/Login.py` (LoginModel definition)
- **Repository**: `Repositories/Login_repositories.py` (database operations)
- **Views**: `views/View_login.py` (API endpoints using these serializers)
- **Enums**: `Enums/Enums.py` (Role enum)

---

## 👨‍💻 Maintainer

**Võ Anh Nhật**  
📧 voanhnhat1612@gmail.com

---

*Last Updated: March 8, 2026*
