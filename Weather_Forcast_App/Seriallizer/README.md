# 📝 Seriallizer — Data Validation & Transformation Layer

## 📁 Overview

This directory contains **Django REST Framework (DRF) serializers** that handle **data validation**, **transformation**, and **serialization** between Python objects and JSON.

---

## 📂 Directory Structure

```
Seriallizer/
├── __init__.py          # Package initializer
├── Login/               # Login/Auth serializers
│   ├── Base_login.py      # BaseSerializerLogin (base class)
│   ├── Create_login.py    # LoginLoginCreate (registration)
│   ├── Update_login.py    # LoginUpdate (update profile)
│   └── __init__.py
└── README.md            # This file
```

---

## 🎯 Purpose

### What are Serializers?

Serializers are **the bridge between Django models and JSON APIs**. They:
- **Validate** incoming data (requests)
- **Transform** Python objects → JSON (responses)
- **Parse** JSON → Python objects (requests)
- **Define** required/optional fields for each API endpoint

### Why Use Serializers?

| Benefit | Description |
|---------|-------------|
| **Type Safety** | Ensure data types match expectations |
| **Validation** | Reject invalid data before hitting database |
| **Separation** | Keep validation logic out of views |
| **Reusability** | Share serializers across multiple endpoints |

---

## 📄 Files Explained

### `Login/` — Authentication Serializers

This subdirectory contains **3 serializers** for different operations:

1. **Base_login.py** — Base serializer with common fields
2. **Create_login.py** — Serializer for user registration (strict validation)
3. **Update_login.py** — Serializer for profile updates (partial validation)

---

## 🔧 How Serializers Work

### Step 1: Client Sends Request

```http
POST /api/register
Content-Type: application/json

{
  "userName": "nhat123",
  "email": "nhat@gmail.com",
  "password": "SecurePass123!",
  "name": "Võ Anh Nhật",
  "role": "Manager"
}
```

### Step 2: Serializer Validates Data

```python
from Weather_Forcast_App.Seriallizer.Login.Create_login import LoginLoginCreate

serializer = LoginLoginCreate(data=request.data)
if serializer.is_valid():
    # ✅ Data passed validation
    clean_data = serializer.validated_data
else:
    # ❌ Data failed validation
    return Response(serializer.errors, status=400)
```

### Step 3: Serializer Returns Clean Data

```python
print(serializer.validated_data)
# {
#   "userName": "nhat123",
#   "email": "nhat@gmail.com",
#   "password": "SecurePass123!",
#   "name": "Võ Anh Nhật",
#   "role": "Manager"
# }
```

---

## 💡 Usage Examples

### Example 1: User Registration (Create)

```python
from Weather_Forcast_App.Seriallizer.Login.Create_login import LoginLoginCreate

@api_view(['POST'])
def register(request):
    # Create serializer with request data
    serializer = LoginLoginCreate(data=request.data)
    
    # Validate
    if not serializer.is_valid():
        return Response({
            "error": "Validation failed",
            "details": serializer.errors
        }, status=400)
    
    # All required fields present + valid
    data = serializer.validated_data
    
    # Hash password
    data["password"] = make_password(data["password"] + PEPPER)
    
    # Insert into MongoDB
    LoginRepository.insert_one(data)
    
    return Response({"success": True}, status=201)
```

**Validation Rules (Create)**:
- ✅ **Required**: `name`, `userName`, `email`, `password`, `role`
- ✅ **Unique**: `userName`, `email` (checked in repository)
- ✅ **Format**: Email must be valid format

### Example 2: Update Profile (Partial)

```python
from Weather_Forcast_App.Seriallizer.Login.Update_login import LoginUpdate

@api_view(['PATCH'])
def update_profile(request, user_id):
    # Find user
    user = LoginRepository.find_by_id(user_id)
    if not user:
        return Response({"error": "User not found"}, status=404)
    
    # Validate partial data (only fields sent by client)
    serializer = LoginUpdate(data=request.data, partial=True)
    if not serializer.is_valid():
        return Response(serializer.errors, status=400)
    
    # Update only provided fields
    update_data = serializer.validated_data
    LoginRepository.update_by_id(user_id, {"$set": update_data})
    
    return Response({"success": True})
```

**Validation Rules (Update)**:
- ✅ **All fields optional** (partial update)
- ✅ **No `id` field** (prevented in serializer)
- ✅ **Format validation** still applies

---

## 📊 Serializer Comparison

| Serializer | Purpose | Required Fields | Use Case |
|------------|---------|-----------------|----------|
| **BaseSerializerLogin** | Base class | All fields (model default) | Not used directly |
| **LoginLoginCreate** | Registration/Create | `name`, `userName`, `email`, `password`, `role` | POST /register |
| **LoginUpdate** | Profile update | None (all optional) | PATCH /users/:id |

---

## 🔍 Best Practices

### ✅ DO

- **Use specific serializers** for different operations (Create vs Update)
- **Validate in serializer** before database operations
- **Hash passwords** after validation, before saving
- **Check uniqueness** in repository (serializers can't access DB directly)

### ❌ DON'T

- **Don't skip validation** (`serializer.save()` without `is_valid()`)
- **Don't put business logic** in serializers (use services/repositories)
- **Don't expose sensitive fields** in response (e.g., `password`)
- **Don't hardcode validation** (use DRF validators)

---

## 🐛 Common Issues

### Issue 1: Missing Required Field

**Error**:
```json
{
  "userName": ["This field is required."]
}
```

**Cause**: Client didn't send required field in POST request

**Solution**:
```javascript
// Frontend: ensure all required fields are sent
const data = {
  name: "Võ Anh Nhật",
  userName: "nhat123",
  email: "nhat@gmail.com",
  password: "MyPassword",
  role: "Manager"
};
fetch("/api/register", {
  method: "POST",
  body: JSON.stringify(data),
  headers: { "Content-Type": "application/json" }
});
```

### Issue 2: Invalid Email Format

**Error**:
```json
{
  "email": ["Enter a valid email address."]
}
```

**Solution**: Use proper email format (`user@domain.com`)

### Issue 3: Serializer Not Validating

**Problem**: `is_valid()` always returns `True`

**Cause**: Forgot to pass `data=` parameter

```python
# ❌ WRONG
serializer = LoginLoginCreate(request.data)

# ✅ CORRECT
serializer = LoginLoginCreate(data=request.data)
```

---

## 🚀 Future Enhancements

- [ ] Add **password strength validation** (min length, special chars)
- [ ] Add **custom validators** for Vietnamese phone numbers
- [ ] Add **nested serializers** for complex objects
- [ ] Add **serializer methods** for computed fields
- [ ] Add **file upload serializers** for profile photos

---

## 📞 Related Files

- **Models**: `Models/Login.py` (data model)
- **Repository**: `Repositories/Login_repositories.py` (database operations)
- **Views**: `views/View_login.py` (API endpoints)
- **Documentation**: `Seriallizer/Login/README.md` (detailed serializer docs)

---

## 👨‍💻 Maintainer

**Võ Anh Nhật**  
📧 voanhnhat1612@gmail.com

---

*Last Updated: March 8, 2026*
