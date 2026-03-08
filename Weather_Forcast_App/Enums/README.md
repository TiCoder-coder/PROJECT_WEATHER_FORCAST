# 📋 Enums — Application Enumerations

## 📁 Overview

This directory contains **enumeration classes** that define **fixed sets of constants** used throughout the application. Enums ensure type safety, prevent typos, and make code more maintainable.

---

## 📂 Directory Structure

```
Enums/
├── __init__.py          # Package initializer
├── Enums.py             # Main enum definitions
└── README.md            # This file
```

---

## 🎯 Purpose

### Why Use Enums?

1. **Type Safety**: Prevent invalid values at compile time
2. **Code Clarity**: Self-documenting constants
3. **Refactoring**: Easy to change values in one place
4. **IDE Support**: Auto-completion and validation

### Use Cases in This Project

- **User Roles**: Admin, Manager, Staff
- **Model Status**: Untrained, Training, Trained, Failed
- **Data Source Types**: API, Selenium, HTML Parser
- **File Types**: CSV, XLSX, JSON, TXT

---

## 📄 Files Explained

### `Enums.py`

Contains all enumeration definitions for the application.

#### **1. CustomEnum (Base Class)**

```python
class CustomEnum(Enum):
    @property
    def description(self):
        """
        Extension point for adding human-readable descriptions
        to enum values. Useful for UI displays and API responses.
        """
        pass
```

**Purpose**: Base class for all custom enums with additional features like `.description` property.

**Future Enhancement**: Map each enum value to Vietnamese/English descriptions for better UX.

---

#### **2. ModelStatus Enum**

```python
class ModelStatus(CustomEnum):
    UNTRAINED = "untrained"   # Model not yet trained
    INIT = "init"              # Model initialized
    TRAINING = "training"      # Currently training
    TRAINED = "trained"        # Training completed successfully
    FAILED = "failed"          # Training failed
    SAVED = "saved"            # Model saved to disk
    LOADED = "loaded"          # Model loaded from disk
```

**Purpose**: Track the lifecycle state of machine learning models.

**Used In**:
- `Machine_learning_model/trainning/train.py` — Update status during training
- `views/View_Train.py` — Display model status in UI
- `Machine_learning_artifacts/Train_info.json` — Store model metadata

**Example Usage**:
```python
from Weather_Forcast_App.Enums.Enums import ModelStatus

# Check model status
if model_status == ModelStatus.TRAINED.value:
    print("Model is ready for predictions!")

# Display in UI
status_text = {
    ModelStatus.UNTRAINED.value: "Chưa huấn luyện",
    ModelStatus.TRAINING.value: "Đang huấn luyện...",
    ModelStatus.TRAINED.value: "Đã huấn luyện ✅"
}
```

---

#### **3. Role Enum**

```python
class Role(CustomEnum):
    Admin = "Admin"       # Full system access
    Staff = "Staff"       # Limited access (view only)
    Manager = "Manager"   # Mid-level access (CRUD operations)
```

**Purpose**: Define user access levels and permissions.

**Used In**:
- `Models/Login.py` — Store user role in database
- `middleware/Authentication.py` — Check permissions
- `views/*.py` — Role-based access control

**Database Mapping**:
```python
# In LoginModel (Models/Login.py)
role = models.CharField(
    max_length=100, 
    choices=[(choice.name, choice.value) for choice in Role]
)
```

**Example Usage**:
```python
from Weather_Forcast_App.Enums.Enums import Role

# Check user role
if user.role == Role.Admin.value:
    # Allow admin-only operations
    pass

# Template rendering
role_badges = {
    Role.Admin.value: "🔑 Administrator",
    Role.Manager.value: "👤 Manager",
    Role.Staff.value: "📋 Staff"
}
```

---

## 🔧 How to Use Enums

### 1️⃣ Import the Enum

```python
from Weather_Forcast_App.Enums.Enums import Role, ModelStatus
```

### 2️⃣ Access Enum Values

```python
# Get the value (string)
admin_role = Role.Admin.value  # "Admin"

# Get the name (attribute name)
admin_name = Role.Admin.name   # "Admin"

# Check if value exists
if user_role in [r.value for r in Role]:
    print("Valid role!")
```

### 3️⃣ Use in Database Models

```python
from Weather_Forcast_App.Enums.Enums import Role

class LoginModel(models.Model):
    role = models.CharField(
        max_length=100,
        choices=[(r.name, r.value) for r in Role],
        default=Role.Staff.value
    )
```

### 4️⃣ Use in Conditionals

```python
from Weather_Forcast_App.Enums.Enums import ModelStatus

if model_status == ModelStatus.TRAINED.value:
    make_predictions()
elif model_status == ModelStatus.TRAINING.value:
    wait_for_training()
else:
    train_model()
```

---

## 🚀 Adding New Enums

### Step 1: Define the Enum Class

```python
# In Enums.py
class DataSourceType(CustomEnum):
    API = "api"
    SELENIUM = "selenium"
    HTML = "html"
    MANUAL = "manual"
```

### Step 2: Use in Code

```python
from Weather_Forcast_App.Enums.Enums import DataSourceType

def crawl_data(source_type):
    if source_type == DataSourceType.API.value:
        return crawl_via_api()
    elif source_type == DataSourceType.SELENIUM.value:
        return crawl_via_selenium()
    # ...
```

### Step 3: Add Descriptions (Optional)

```python
# Extend CustomEnum to return descriptions
class DataSourceType(CustomEnum):
    API = "api"
    SELENIUM = "selenium"
    
    @property
    def description(self):
        descriptions = {
            "api": "REST API Request",
            "selenium": "Web Scraping via Selenium"
        }
        return descriptions.get(self.value, self.name)

# Usage
print(DataSourceType.API.description)  # "REST API Request"
```

---

## 📊 Current Enums Summary

| Enum | Values | Purpose |
|------|--------|---------|
| **ModelStatus** | UNTRAINED, INIT, TRAINING, TRAINED, FAILED, SAVED, LOADED | ML model lifecycle |
| **Role** | Admin, Manager, Staff | User access control |

---

## 🔍 Best Practices

### ✅ DO

- Use enums for **fixed, predefined sets** of values
- Use `.value` when storing to database or comparing
- Use `.name` for logging or debugging
- Document each enum member with comments

### ❌ DON'T

- Don't use enums for **dynamic data** (e.g., city names from DB)
- Don't compare enum objects directly (use `.value`)
- Don't hardcode string values in code (use enum instead)

---

## 🐛 Common Mistakes

### Mistake 1: Direct Comparison

```python
# ❌ WRONG
if user.role == Role.Admin:
    pass

# ✅ CORRECT
if user.role == Role.Admin.value:
    pass
```

### Mistake 2: Typo in String

```python
# ❌ WRONG
if status == "trainned":  # Typo!
    pass

# ✅ CORRECT
if status == ModelStatus.TRAINED.value:
    pass
```

### Mistake 3: Not Importing

```python
# ❌ WRONG
if user.role == "Admin":  # Hardcoded string
    pass

# ✅ CORRECT
from Weather_Forcast_App.Enums.Enums import Role
if user.role == Role.Admin.value:
    pass
```

---

## 🗺️ Future Enhancements

- [ ] Add **FileType** enum (CSV, XLSX, JSON, TXT)
- [ ] Add **CrawlMethod** enum (API, Selenium, HTML)
- [ ] Add **CleaningStep** enum (Remove Duplicates, Fill NaN, etc.)
- [ ] Implement `.description` property with i18n support
- [ ] Add validation methods (e.g., `is_valid_role(value)`)

---

## 📞 Related Files

- **Models**: `Models/Login.py` (uses Role enum)
- **Training**: `Machine_learning_model/trainning/train.py` (uses ModelStatus)
- **Views**: `views/View_Train.py` (displays status)
- **Database**: MongoDB `logins` collection (stores role values)

---

## 👨‍💻 Maintainer

**Võ Anh Nhật**  
📧 voanhnhat1612@gmail.com

---

*Last Updated: March 8, 2026*
