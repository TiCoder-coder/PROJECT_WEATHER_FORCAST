# 🗂️ locales — Translation Files

## 📁 Overview

This directory stores **JSON translation files** for the i18n (internationalization) system. Each file contains translated strings for a specific language.

---

## 📂 Directory Structure

```
locales/
├── vi.json          # Vietnamese translations (250+ keys)
├── en.json          # English translations (250+ keys)
└── README.md        # This file
```

---

## 🎯 Purpose

### Why Separate Translation Files?

1. **Maintainability**: Easy to update translations without touching code
2. **Translation Teams**: Non-developers can edit JSON files
3. **Version Control**: Track translation changes over time
4. **Performance**: Load only the needed language file

---

## 📄 File Format

### Structure

All translation files follow this **nested JSON structure**:

```json
{
  "category": {
    "subcategory": {
      "key": "translated string"
    }
  }
}
```

### Example: `vi.json`

```json
{
  "home": {
    "hero_title": "Dữ Liệu Thời Tiết Việt Nam Theo Thời Gian Thực",
    "hero_desc": "Thu thập, xử lý và dự báo dữ liệu thời tiết với machine learning",
    "btn_crawl": "Bắt Đầu Thu Thập Dữ Liệu",
    "btn_datasets": "Xem Datasets",
    "btn_train": "Huấn Luyện Model",
    "btn_predict": "Dự Báo Thời Tiết"
  },
  "datasets": {
    "tab_recent": "Dữ Liệu Thô Gần Đây",
    "tab_merged": "Dữ Liệu Đã Gộp",
    "tab_cleaned": "Dữ Liệu Đã Làm Sạch",
    "tab_process": "Xử Lý Dữ Liệu",
    "stat_total_files": "Tổng Số File",
    "stat_total_size": "Tổng Dung Lượng",
    "btn_view": "Xem",
    "btn_download": "Tải Xuống",
    "btn_merge": "Gộp",
    "btn_clean": "Làm Sạch"
  },
  "auth": {
    "login_title": "Đăng Nhập Tài Khoản",
    "register_title": "Tạo Tài Khoản Mới",
    "forgot_password": "Quên Mật Khẩu?",
    "email_verification": "Cần Xác Minh Email",
    "otp_sent": "Mã OTP đã được gửi đến email của bạn",
    "otp_placeholder": "Nhập mã OTP 5 chữ số",
    "btn_login": "Đăng Nhập",
    "btn_register": "Đăng Ký",
    "btn_submit": "Gửi",
    "btn_verify": "Xác Minh"
  }
}
```

### Example: `en.json`

```json
{
  "home": {
    "hero_title": "Vietnam Weather Data in Real Time",
    "hero_desc": "Collect, process, and forecast weather data with machine learning",
    "btn_crawl": "Start Data Collection",
    "btn_datasets": "Browse Datasets",
    "btn_train": "Train Model",
    "btn_predict": "Weather Forecast"
  },
  "datasets": {
    "tab_recent": "Recent Raw Data",
    "tab_merged": "Merged Data",
    "tab_cleaned": "Cleaned Data",
    "tab_process": "Process Data",
    "stat_total_files": "Total Files",
    "stat_total_size": "Total Size",
    "btn_view": "View",
    "btn_download": "Download",
    "btn_merge": "Merge",
    "btn_clean": "Clean"
  },
  "auth": {
    "login_title": "Login to Your Account",
    "register_title": "Create New Account",
    "forgot_password": "Forgot Password?",
    "email_verification": "Email Verification Required",
    "otp_sent": "OTP code sent to your email",
    "otp_placeholder": "Enter 5-digit OTP code",
    "btn_login": "Login",
    "btn_register": "Register",
    "btn_submit": "Submit",
    "btn_verify": "Verify"
  }
}
```

---

## 🔧 How to Use

### In Django Templates

```html
<!-- Load translation function (auto-available via context processor) -->
<h1>{{ t("home.hero_title") }}</h1>
<p>{{ t("home.hero_desc") }}</p>
<button>{{ t("home.btn_crawl") }}</button>
```

### In Django Views

```python
def my_view(request):
    title = request.t("home.hero_title")
    return render(request, "template.html", {"title": title})
```

### In JavaScript

```javascript
const t = window.i18n.t;
document.getElementById('title').textContent = t('home.hero_title');
```

---

## ✏️ Adding New Translations

### Step 1: Add to Both Files

**vi.json**:
```json
{
  "forecast": {
    "page_title": "Dự Báo Thời Tiết",
    "select_date": "Chọn Ngày",
    "view_results": "Xem Kết Quả"
  }
}
```

**en.json**:
```json
{
  "forecast": {
    "page_title": "Weather Forecast",
    "select_date": "Select Date",
    "view_results": "View Results"
  }
}
```

### Step 2: Use in Code

```html
<h1>{{ t("forecast.page_title") }}</h1>
<button>{{ t("forecast.view_results") }}</button>
```

### Step 3: Restart Server

```bash
python manage.py runserver
```

---

## 📊 Translation Categories

| Category | Description | Keys Count |
|----------|-------------|------------|
| **home** | Homepage content | 15 |
| **datasets** | Dataset management UI | 25 |
| **auth** | Login, register, password reset | 30 |
| **train** | Model training interface | 20 |
| **forecast** | Weather forecasting page | 18 |
| **nav** | Navigation menu | 12 |
| **errors** | Error messages | 20 |
| **buttons** | Common button labels | 15 |
| **forms** | Form labels and placeholders | 25 |
| **messages** | Success/info/warning messages | 30 |
| **TOTAL** | | **250+** |

---

## 🔍 Key Naming Convention

### Format

```
category.subcategory.element_type
```

### Examples

```json
{
  "auth.login.title": "Login to Your Account",
  "auth.login.btn_submit": "Login",
  "datasets.tab_recent.title": "Recent Raw Data",
  "errors.404.message": "Page not found",
  "nav.menu.home": "Home"
}
```

### Best Practices

- **Use dot notation**: `category.action.element`
- **Be specific**: `btn_login` not just `login`
- **Consistent naming**: All buttons start with `btn_`
- **Lowercase keys**: `page_title` not `PageTitle`

---

## 🌍 Supported Languages

| Language | Code | File | Status |
|----------|------|------|--------|
| 🇻🇳 **Vietnamese** | `vi` | `vi.json` | ✅ 100% |
| 🇬🇧 **English** | `en` | `en.json` | ✅ 100% |

### Adding New Languages (Future)

To add a new language (e.g., French):

1. **Create file**: `locales/fr.json`
2. **Copy structure** from `en.json`
3. **Translate values** to French
4. **Update i18n/__init__.py**:
   ```python
   SUPPORTED_LANGS = ["vi", "en", "fr"]
   ```
5. **Restart server**

---

## 🐛 Common Issues

### Issue 1: Translation Not Showing

**Symptom**: Template shows key instead of translation (e.g., "home.hero_title")

**Causes**:
- ❌ Key doesn't exist in JSON file
- ❌ Typo in key name
- ❌ Invalid JSON syntax
- ❌ File encoding not UTF-8

**Solution**:
```bash
# Validate JSON syntax
python -m json.tool locales/vi.json

# Check if key exists
grep "hero_title" locales/vi.json

# Ensure UTF-8 encoding
file locales/vi.json  # Should show "UTF-8 Unicode text"
```

### Issue 2: Special Characters Broken

**Symptom**: Vietnamese characters show as `�` or `?`

**Solution**:
```bash
# Save file with UTF-8 encoding
# In VS Code: File → Save with Encoding → UTF-8
```

### Issue 3: Nested Key Not Found

**Symptom**: Deep nested keys return undefined

**Solution**:
```json
// ❌ WRONG (missing intermediate levels)
{
  "datasets": {
    "file_count": "10 files"
  }
}

// ✅ CORRECT (full path exists)
{
  "datasets": {
    "stats": {
      "file_count": "10 files"
    }
  }
}
```

---

## 🚀 Performance Tips

### File Loading

Translation files are loaded **once** at server startup and **cached in memory**.

### Optimization

- ✅ Keep files under 100KB each
- ✅ Avoid deeply nested structures (max 3-4 levels)
- ✅ Use consistent key structure for easy lookup

---

## 📞 Related Files

- **i18n Module**: `i18n/__init__.py` (loads these files)
- **Middleware**: `i18n/middleware.py` (detects language)
- **Context Processor**: `i18n/context_processor.py` (injects `t()` in templates)
- **Templates**: All `.html` files use `{{ t() }}`

---

## 👨‍💻 Maintainer

**Võ Anh Nhật**  
📧 voanhnhat1612@gmail.com

---

*Last Updated: March 8, 2026*
