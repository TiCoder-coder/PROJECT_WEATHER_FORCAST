# 🔐 middleware — Authentication & Request Processing

## 📁 Overview

This directory contains **Django middleware** components that process every HTTP request/response in the application. Middleware handles **authentication, authorization, and request enrichment**.

---

## 📂 Directory Structure

```
middleware/
├── __init__.py              # Package initializer
├── Auth.py                  # Legacy/basic auth (may be unused)
├── Authentication.py        # JWT authentication middleware
├── Jwt_handler.py           # JWT token generation & verification
└── README.md                # This file
```

---

## 🎯 Purpose

### What is Middleware?

Middleware is **Python code that runs on every request/response** before it reaches your views.

#### Request Flow

```
HTTP Request
    ↓
SessionMiddleware (Django)
    ↓
LangMiddleware (i18n)
    ↓
JWTAuthenticationMiddleware ← YOU ARE HERE
    ↓
View Function (e.g., home_view)
    ↓
HTTP Response
```

### Use Cases

1. **Authentication**: Verify JWT tokens, check user login status
2. **Authorization**: Enforce role-based access control
3. **Request Enrichment**: Attach user object, language, etc.
4. **Logging**: Track requests for debugging/analytics
5. **Security**: CSRF protection, XSS prevention, rate limiting

---

## 📄 Files Explained

### 1️⃣ `Authentication.py` — JWT Authentication Middleware

**Purpose**: Verify JWT tokens on every request and attach authenticated user to `request.user`.

#### Code Structure

```python
class JWTAuthentication(BaseAuthentication):
    """
    DRF (Django Rest Framework) authentication backend.
    Checks for JWT token in Authorization header.
    """
    
    def authenticate(self, request):
        # Get Authorization header
        auth = request.headers.get("Authorization")
        
        # Check format: "Bearer <token>"
        if not auth or not auth.startswith("Bearer "):
            return None
        
        # Extract token
        token = auth.split(" ", 1)[1].strip()
        
        # Verify token (calls Jwt_handler.verify_access_token)
        payload = verify_access_token(token)
        
        # Create user object from payload
        return (TokenUser(payload), None)
```

#### TokenUser Class

```python
class TokenUser(AnonymousUser):
    """
    Lightweight user object created from JWT payload.
    Does NOT query database — stateless authentication.
    """
    
    def __init__(self, payload):
        self.id = payload.get("manager_id")
        self.role = payload.get("role", "guest")
    
    @property
    def is_authenticated(self):
        return True
```

#### Usage in Views

```python
def protected_view(request):
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Unauthorized"}, status=401)
    
    # Access user info from JWT payload
    user_id = request.user.id
    user_role = request.user.role
    
    return JsonResponse({"user_id": user_id, "role": user_role})
```

#### Registration in `settings.py`

```python
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'Weather_Forcast_App.middleware.Authentication.JWTAuthentication',
    ]
}
```

---

### 2️⃣ `Jwt_handler.py` — JWT Token Management

**Purpose**: Generate and verify JWT (JSON Web Tokens) for stateless authentication.

#### Key Functions

##### `create_access_token(payload: dict) → str`

```python
import jwt
from datetime import datetime, timedelta
from django.conf import settings

def create_access_token(payload: dict) -> str:
    """
    Create a JWT access token.
    
    Args:
        payload: Dict with user data (manager_id, role, etc.)
    
    Returns:
        Signed JWT token string
    
    Example:
        token = create_access_token({
            "manager_id": "abc123",
            "role": "Admin"
        })
    """
    # Add expiration time
    payload["exp"] = datetime.utcnow() + timedelta(
        seconds=int(settings.JWT_ACCESS_TTL)  # Default: 900s = 15min
    )
    payload["iat"] = datetime.utcnow()
    
    # Sign token with secret key
    token = jwt.encode(
        payload,
        settings.JWT_SECRET,
        algorithm=settings.JWT_ALGORITHM  # HS256
    )
    
    return token
```

##### `verify_access_token(token: str) → dict`

```python
def verify_access_token(token: str) -> dict:
    """
    Verify and decode JWT token.
    
    Args:
        token: JWT token string
    
    Returns:
        Decoded payload dict
    
    Raises:
        jwt.ExpiredSignatureError: Token expired
        jwt.InvalidTokenError: Invalid token
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM]
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise AuthenticationFailed("Token expired")
    except jwt.InvalidTokenError:
        raise AuthenticationFailed("Invalid token")
```

##### `create_refresh_token(payload: dict) → str`

```python
def create_refresh_token(payload: dict) -> str:
    """
    Create a long-lived refresh token.
    
    Used to obtain new access tokens without re-login.
    TTL: 7 days (default)
    """
    payload["exp"] = datetime.utcnow() + timedelta(
        seconds=int(settings.JWT_REFRESH_TTL)  # 604800s = 7 days
    )
    
    token = jwt.encode(
        payload,
        settings.JWT_SECRET,
        algorithm=settings.JWT_ALGORITHM
    )
    
    return token
```

---

### 3️⃣ `Auth.py` — Legacy Authentication (May Be Unused)

**Status**: ⚠️ May contain legacy code or alternative auth methods.

**Purpose**: Possibly contains:
- Session-based authentication helpers
- Old authentication logic before JWT implementation
- Utility functions for password hashing

**Note**: Check if this file is actually imported anywhere before using.

---

## 🔧 How Authentication Works

### Step-by-Step Flow

#### 1️⃣ User Logs In

```python
# In views/View_login.py
def login_view(request):
    username = request.POST.get("username")
    password = request.POST.get("password")
    
    # Verify credentials (check MongoDB)
    user = LoginRepository.find_by_username(username)
    if not verify_password(password, user.password):
        return error_response("Invalid credentials")
    
    # Create JWT token
    payload = {
        "manager_id": str(user._id),
        "role": user.role,
        "username": user.userName
    }
    
    access_token = create_access_token(payload)
    refresh_token = create_refresh_token(payload)
    
    # Return tokens to client
    return JsonResponse({
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_in": 900  # 15 minutes
    })
```

#### 2️⃣ Client Stores Token

```javascript
// Frontend JavaScript
const response = await fetch('/login', {
    method: 'POST',
    body: JSON.stringify({ username, password })
});

const data = await response.json();

// Store token in localStorage
localStorage.setItem('access_token', data.access_token);
localStorage.setItem('refresh_token', data.refresh_token);
```

#### 3️⃣ Client Sends Token on Requests

```javascript
// Attach token to all API requests
fetch('/api/datasets', {
    headers: {
        'Authorization': `Bearer ${localStorage.getItem('access_token')}`
    }
});
```

#### 4️⃣ Middleware Verifies Token

```python
# JWTAuthentication.authenticate() runs automatically
# Sets request.user to TokenUser object
def datasets_view(request):
    # request.user is already authenticated by middleware
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Unauthorized"}, status=401)
    
    # Access user info
    user_role = request.user.role
    
    return JsonResponse({"datasets": [...]})
```

---

## 🛡️ Security Features

### Token Expiration

| Token Type | TTL | Renewable? |
|------------|-----|------------|
| **Access Token** | 15 minutes | ❌ No (must use refresh token) |
| **Refresh Token** | 7 days | ✅ Yes (can refresh once) |

### Token Payload

```json
{
  "manager_id": "507f1f77bcf86cd799439011",
  "role": "Admin",
  "username": "nhat123",
  "iat": 1709884800,  // Issued at
  "exp": 1709885700   // Expires at
}
```

### Secret Key

**Environment Variable** (`.env`):
```env
JWT_SECRET=MHGtW9YsZcP1O04ScNbiOTVMPS-DCS_NKeenFBzaWXzR2Fk7_3xxnT2vubAMIuXNVybtBsCYifEYHxVW6fRnEQ
JWT_ALGORITHM=HS256
JWT_ACCESS_TTL=900
JWT_REFRESH_TTL=604800
```

**⚠️ NEVER commit** `JWT_SECRET` to git!

---

## 🔑 Role-Based Access Control (RBAC)

### Checking User Role

```python
from Weather_Forcast_App.Enums.Enums import Role

def admin_only_view(request):
    if request.user.role != Role.Admin.value:
        return JsonResponse({"error": "Forbidden"}, status=403)
    
    # Admin-only logic here
    return JsonResponse({"message": "Welcome, Admin!"})
```

### Permission Decorator (Future)

```python
from functools import wraps

def require_role(allowed_roles):
    def decorator(view_func):
        @wraps(view_func)
        def wrapper(request, *args, **kwargs):
            if request.user.role not in allowed_roles:
                return JsonResponse({"error": "Forbidden"}, status=403)
            return view_func(request, *args, **kwargs)
        return wrapper
    return decorator

# Usage
@require_role([Role.Admin.value, Role.Manager.value])
def manage_users_view(request):
    # Only admins and managers can access
    pass
```

---

## 🐛 Common Issues

### Issue 1: Token Expired

**Error**: `{"error": "Token expired"}`

**Solution**:
```javascript
// Use refresh token to get new access token
async function refreshAccessToken() {
    const refreshToken = localStorage.getItem('refresh_token');
    
    const response = await fetch('/api/refresh', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${refreshToken}`
        }
    });
    
    const data = await response.json();
    localStorage.setItem('access_token', data.access_token);
}
```

### Issue 2: Invalid Token

**Error**: `{"error": "Invalid token"}`

**Causes**:
- Token tampered with
- Wrong secret key used
- Malformed token format

**Solution**: Re-login to get new token.

### Issue 3: Missing Authorization Header

**Error**: `request.user` is `AnonymousUser`

**Solution**:
```javascript
// Ensure Authorization header is sent
fetch('/api/endpoint', {
    headers: {
        'Authorization': `Bearer ${token}`  // Don't forget "Bearer " prefix!
    }
});
```

---

## 📊 Middleware Execution Order

```python
# settings.py
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',  # ← Session first
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'Weather_Forcast_App.i18n.middleware.LangMiddleware',  # ← Language detection
    'Weather_Forcast_App.middleware.Authentication.JWTAuthenticationMiddleware',  # ← JWT last
]
```

**Order matters!** JWT middleware needs session to be available first.

---

## 🚀 Future Enhancements

- [ ] **Token blacklisting** (logout invalidates token)
- [ ] **Multi-device sessions** (track active tokens)
- [ ] **OAuth2 integration** (Google, Facebook login)
- [ ] **Two-factor authentication (2FA)**
- [ ] **Rate limiting** (prevent brute force)
- [ ] **Audit logging** (track all auth attempts)

---

## 📞 Related Files

- **Settings**: `WeatherForcast/settings.py` (JWT config)
- **Views**: `views/View_login.py` (login/register)
- **Models**: `Models/Login.py` (user model)
- **Repository**: `Repositories/Login_repositories.py` (user CRUD)
- **Enums**: `Enums/Enums.py` (Role enum)

---

## 👨‍💻 Maintainer

**Võ Anh Nhật**  
📧 voanhnhat1612@gmail.com

---

*Last Updated: March 8, 2026*
