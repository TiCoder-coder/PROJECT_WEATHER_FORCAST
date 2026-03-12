# 📊 Models — Database Models

## 📁 Overview

This directory contains **Django model definitions** that map to **MongoDB collections**. Models define the structure and behavior of data entities in the application.

---

## 📂 Directory Structure

```
Models/
├── __init__.py          # Package initializer
├── Login.py             # User/Manager model
└── README.md            # This file
```

---

## 🎯 Purpose

### What are Django Models?

Models are **Python classes** that represent database tables/collections. They:
- Define data structure (fields, types, constraints)
- Provide ORM methods (create, read, update, delete)
- Handle data validation and serialization
- Map to MongoDB collections via `djongo`

---

## 📄 Files Explained

### `Login.py` — User/Manager Model

**Purpose**: Define user accounts (managers, admins, staff) stored in MongoDB.

#### Model Definition

```python
from django.db import models
from Weather_Forcast_App.Enums.Enums import Role

class LoginModel(models.Model):
    # ==================== PRIMARY FIELDS ====================
    
    name = models.CharField(max_length=100, unique=True)
    """Full name of the user (e.g., "Võ Anh Nhật")"""
    
    userName = models.CharField(max_length=100, unique=True)
    """Unique username for login"""
    
    email = models.CharField(max_length=100, unique=True)
    """Email address (must be unique, used for OTP)"""
    
    password = models.CharField(max_length=255)
    """Hashed password (pepper + bcrypt)"""
    
    role = models.CharField(
        max_length=100,
        choices=[(choice.name, choice.value) for choice in Role]
    )
    """User role: Admin | Manager | Staff"""
    
    is_active = models.BooleanField(default=True)
    """Account active status (can be deactivated)"""
    
    # ==================== SECURITY FIELDS ====================
    
    failed_attempts = models.IntegerField(default=0)
    """Count of consecutive failed login attempts"""
    
    lock_until = models.DateTimeField(null=True, blank=True)
    """Timestamp until when account is locked (None if not locked)"""
    
    # ==================== TRACKING FIELDS ====================
    
    last_login = models.DateTimeField(null=True, blank=True)
    """Last successful login timestamp"""
    
    createdAt = models.DateTimeField(auto_now_add=True)
    """Account creation timestamp (auto-set)"""
    
    updatedAt = models.DateTimeField(auto_now=True)
    """Last update timestamp (auto-updated)"""
    
    # ==================== META ====================
    
    class Meta:
        db_table = "managers"
        ordering = ["userName"]
    
    def __str__(self):
        return f"{self.name} - ({self.userName}): {self.role}"
```

---

## 📊 Field Details

### Primary Fields

| Field | Type | Constraints | Purpose |
|-------|------|-------------|---------|
| **name** | CharField(100) | Unique | Full display name |
| **userName** | CharField(100) | Unique | Login identifier |
| **email** | CharField(100) | Unique | Email for OTP/recovery |
| **password** | CharField(255) | - | Hashed password |
| **role** | CharField(100) | Enum choices | Access level |
| **is_active** | BooleanField | Default: True | Account status |

### Security Fields

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| **failed_attempts** | IntegerField | 0 | Failed login counter |
| **lock_until** | DateTimeField | NULL | Lockout expiry time |

**Lockout Logic**:
```python
# After 5 failed attempts
if user.failed_attempts >= 5:
    user.lock_until = datetime.now() + timedelta(minutes=5)
    user.save()
```

### Tracking Fields

| Field | Type | Auto? | Purpose |
|-------|------|-------|---------|
| **last_login** | DateTimeField | Manual | Track login activity |
| **createdAt** | DateTimeField | auto_now_add | Account creation |
| **updatedAt** | DateTimeField | auto_now | Last modification |

---

## 🗄️ MongoDB Collection

### Collection Name

```python
class Meta:
    db_table = "managers"  # MongoDB collection name
```

### Document Structure

```json
{
  "_id": ObjectId("507f1f77bcf86cd799439011"),
  "name": "Võ Anh Nhật",
  "userName": "nhat123",
  "email": "voanhnhat1612@gmail.com",
  "password": "pbkdf2_sha256$260000$...",
  "role": "Admin",
  "is_active": true,
  "failed_attempts": 0,
  "lock_until": null,
  "last_login": ISODate("2026-03-08T10:30:00Z"),
  "createdAt": ISODate("2026-01-15T08:00:00Z"),
  "updatedAt": ISODate("2026-03-08T10:30:00Z")
}
```

---

## 🔧 Usage Examples

### Creating a New User

```python
from Weather_Forcast_App.Models.Login import LoginModel
from Weather_Forcast_App.Enums.Enums import Role

user = LoginModel(
    name="Võ Anh Nhật",
    userName="nhat123",
    email="nhat@gmail.com",
    password="hashed_password_here",
    role=Role.Admin.value,
    is_active=True
)
user.save()  # Inserts into MongoDB
```

### Querying Users

```python
# Find by username
user = LoginModel.objects.get(userName="nhat123")

# Find all admins
admins = LoginModel.objects.filter(role=Role.Admin.value)

# Check if email exists
exists = LoginModel.objects.filter(email="test@example.com").exists()
```

### Updating User

```python
user = LoginModel.objects.get(userName="nhat123")
user.failed_attempts = 0
user.lock_until = None
user.last_login = datetime.now()
user.save()
```

### Deleting User

```python
user = LoginModel.objects.get(userName="nhat123")
user.delete()
```

---

## 🔐 Password Security

### Never Store Plain Passwords!

```python
# ❌ WRONG
user.password = "MyPassword123"

# ✅ CORRECT
from django.contrib.auth.hashers import make_password
user.password = make_password(password + PEPPER_SECRET)
```

### Password Verification

```python
from django.contrib.auth.hashers import check_password

is_valid = check_password(
    password + PEPPER_SECRET,
    user.password
)
```

### Pepper Enhancement

```python
# .env file
PASSWORD_PEPPER=yPTp0tlNjhhCmktx_FInwo0bLcu2aquaT3BLVMJaQqw

# In code
peppered_password = password + os.getenv("PASSWORD_PEPPER")
hashed = make_password(peppered_password)
```

---

## 🛡️ Account Lockout System

### Implementation

```python
from datetime import datetime, timedelta
from django.conf import settings

def check_account_lockout(user):
    """Check if account is currently locked"""
    if user.lock_until and user.lock_until > datetime.now():
        remaining = (user.lock_until - datetime.now()).seconds
        return True, f"Account locked. Try again in {remaining} seconds."
    
    # Clear lockout if expired
    if user.lock_until and user.lock_until <= datetime.now():
        user.failed_attempts = 0
        user.lock_until = None
        user.save()
    
    return False, None

def handle_failed_login(user):
    """Increment failed attempts and lock if threshold reached"""
    user.failed_attempts += 1
    
    if user.failed_attempts >= settings.MAX_FAILED_ATTEMPS:
        user.lock_until = datetime.now() + timedelta(
            seconds=settings.LOCKOUT_SECOND
        )
    
    user.save()

def handle_successful_login(user):
    """Reset failed attempts and update last login"""
    user.failed_attempts = 0
    user.lock_until = None
    user.last_login = datetime.now()
    user.save()
```

### Configuration (`.env`)

```env
MAX_FAILED_ATTEMPS=5
LOCKOUT_SECOND=300  # 5 minutes
```

---

## 📊 Model Relationships (Future)

Currently, this app uses **one model** (LoginModel). Future enhancements may add:

### Email Verification Model

```python
class EmailVerificationOTP(models.Model):
    email = models.EmailField()
    otp_hash = models.CharField(max_length=255)
    salt = models.CharField(max_length=255)
    attempts = models.IntegerField(default=0)
    used = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
```

### Password Reset Model

```python
class PasswordResetOTP(models.Model):
    email = models.EmailField()
    otp_hash = models.CharField(max_length=255)
    salt = models.CharField(max_length=255)
    attempts = models.IntegerField(default=0)
    used = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
```

---

## 🐛 Common Issues

### Issue 1: Unique Constraint Violation

**Error**: `DuplicateKeyError: E11000 duplicate key error`

**Cause**: Trying to create user with existing username/email

**Solution**:
```python
# Check before creating
if LoginModel.objects.filter(userName=username).exists():
    return error_response("Username already taken")

if LoginModel.objects.filter(email=email).exists():
    return error_response("Email already registered")
```

### Issue 2: Database Connection Error

**Error**: `ServerSelectionTimeoutError: localhost:27017: [Errno 111] Connection refused`

**Solution**:
```bash
# Start MongoDB
sudo systemctl start mongodb

# Check connection
mongosh --eval "db.version()"
```

### Issue 3: Field Validation Error

**Error**: `ValidationError: Enter a valid email address`

**Solution**:
```python
from django.core.validators import EmailValidator

validator = EmailValidator()
try:
    validator(email)
except ValidationError:
    return error_response("Invalid email format")
```

---

## 🚀 Future Model Enhancements

- [ ] Add **user profile** (photo, bio, preferences)
- [ ] Add **activity log** (track all user actions)
- [ ] Add **session management** (track active logins)
- [ ] Add **permissions** table (granular access control)
- [ ] Add **team/organization** model (multi-tenant)

---

## 📞 Related Files

- **Repository**: `Repositories/Login_repositories.py` (CRUD operations)
- **Serializer**: `Seriallizer/Login/` (data validation)
- **Views**: `views/View_login.py` (login/register logic)
- **Enums**: `Enums/Enums.py` (Role enum)
- **Settings**: `WeatherForcast/settings.py` (database config)

---

## 👨‍💻 Maintainer

**Võ Anh Nhật**  
📧 voanhnhat1612@gmail.com

---

*Last Updated: March 8, 2026*
