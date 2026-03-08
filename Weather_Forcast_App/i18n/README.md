# 🌍 i18n — Internationalization System

## 📁 Overview

This directory implements the **Internationalization (i18n) system** for VN Weather Hub, enabling seamless **Vietnamese ⇄ English** language switching across the entire application.

---

## 📂 Directory Structure

```
i18n/
├── __init__.py              # Package exports (detect_language, translate, get_t)
├── middleware.py            # Language detection middleware
├── context_processor.py     # Template context processor
├── hooks.ts                 # Frontend i18n hooks (React/TypeScript)
├── index.ts                 # Frontend i18n utilities
└── README.md                # This file
```

---

## 🎯 Purpose

### Why Internationalization?

1. **User Experience**: Users can read UI in their native language
2. **Accessibility**: Broader audience reach (Vietnamese + English speakers)
3. **Professional**: Shows attention to detail and global readiness
4. **Maintenance**: Centralized translations easy to update

### Features

- ✅ **250+ translated strings** covering all UI elements
- ✅ **Live language switching** without page reload
- ✅ **Context-aware translations** via custom template tags
- ✅ **URL persistence** — language saved in session
- ✅ **Backend + Frontend** — Both Django and JS support

---

## 📄 Files Explained

### 1️⃣ `middleware.py` — Language Detection Middleware

**Purpose**: Attach language helpers to every HTTP request so views can access `request.lang` and `request.t()`.

#### Code Structure

```python
class LangMiddleware:
    """
    Lightweight middleware that attaches language helpers to the request.
    
    Runs AFTER SessionMiddleware so session-based preferences are available.
    """
    
    def __call__(self, request):
        # Detect language from: URL params → session → Accept-Language header
        lang = detect_language(request)
        request.lang = lang
        
        # Create translation function bound to detected language
        def _t(key: str) -> str:
            return translate(key, lang)
        
        _t.lang = lang
        request.t = _t
        
        response = self.get_response(request)
        return response
```

#### Usage in Views

```python
def my_view(request):
    # Language code
    current_lang = request.lang  # "vi" or "en"
    
    # Translate a key
    title = request.t("home.hero_title")
    
    return render(request, "template.html", {
        "title": title
    })
```

#### Registration in `settings.py`

```python
MIDDLEWARE = [
    # ...
    'django.contrib.sessions.middleware.SessionMiddleware',  # MUST be before LangMiddleware
    # ...
    'Weather_Forcast_App.i18n.middleware.LangMiddleware',
]
```

---

### 2️⃣ `context_processor.py` — Template Context Processor

**Purpose**: Inject `t()` function and `lang` variable into **every template** automatically.

#### Code Structure

```python
def i18n_context(request):
    """
    Returns context variables available in every template:
    
    - t         callable(key) → translated string
    - lang      current language code ("vi" or "en")
    - languages list of supported codes ["vi", "en"]
    """
    t = get_t(request)
    return {
        "t": t,
        "lang": t.lang,
        "languages": SUPPORTED_LANGS,
    }
```

#### Usage in Templates

```html
<!-- No need to import anything! t() is auto-available -->
<h1>{{ t("home.hero_title") }}</h1>
<p>{{ t("home.hero_desc") }}</p>

<!-- Check current language -->
{% if lang == "vi" %}
  <span>🇻🇳 Tiếng Việt</span>
{% else %}
  <span>🇬🇧 English</span>
{% endif %}

<!-- Language switcher -->
<select>
  {% for l in languages %}
    <option value="{{ l }}" {% if l == lang %}selected{% endif %}>
      {{ l|upper }}
    </option>
  {% endfor %}
</select>
```

#### Registration in `settings.py`

```python
TEMPLATES = [
    {
        "OPTIONS": {
            "context_processors": [
                # ...
                "Weather_Forcast_App.i18n.context_processor.i18n_context",
            ]
        }
    }
]
```

---

### 3️⃣ `__init__.py` — Core i18n Functions

**Purpose**: Central module exporting language detection, translation, and helper functions.

#### Key Functions

##### `detect_language(request) → str`

```python
def detect_language(request):
    """
    Detect user's language from (in priority order):
    1. URL query parameter: ?lang=en
    2. Session variable: request.session.get('lang')
    3. Accept-Language header
    4. Default: "vi"
    
    Returns: "vi" or "en"
    """
    # Check URL param
    if 'lang' in request.GET:
        lang = request.GET['lang']
        if lang in SUPPORTED_LANGS:
            request.session['lang'] = lang  # Save to session
            return lang
    
    # Check session
    if 'lang' in request.session:
        return request.session['lang']
    
    # Check Accept-Language header
    accept_lang = request.META.get('HTTP_ACCEPT_LANGUAGE', '')
    if 'en' in accept_lang:
        return 'en'
    
    # Default
    return 'vi'
```

##### `translate(key: str, lang: str) → str`

```python
def translate(key: str, lang: str) -> str:
    """
    Translate a key to specified language.
    
    Args:
        key: Dot-separated key like "home.hero_title"
        lang: Language code ("vi" or "en")
    
    Returns:
        Translated string or key itself if not found
    
    Example:
        translate("home.hero_title", "en")
        → "Vietnam Weather Data in Real Time"
    """
    translations = load_translations(lang)
    
    # Navigate nested dictionary
    keys = key.split('.')
    result = translations
    for k in keys:
        if isinstance(result, dict) and k in result:
            result = result[k]
        else:
            return key  # Fallback to key if not found
    
    return result
```

##### `get_t(request) → callable`

```python
def get_t(request) -> callable:
    """
    Get translation function bound to request's language.
    
    Returns a callable that can be called like: t("key")
    """
    lang = detect_language(request)
    
    def _t(key: str) -> str:
        return translate(key, lang)
    
    _t.lang = lang
    return _t
```

---

### 4️⃣ `hooks.ts` & `index.ts` — Frontend i18n (TypeScript)

**Purpose**: JavaScript/TypeScript equivalent of backend i18n for client-side rendering.

#### `hooks.ts` (React Hooks)

```typescript
import { useTranslation } from './index';

export function useI18n() {
  const { t, lang, setLang } = useTranslation();
  
  return {
    t,           // Translation function
    lang,        // Current language
    setLang,     // Change language
  };
}

// Usage in React component
function MyComponent() {
  const { t, lang, setLang } = useI18n();
  
  return (
    <div>
      <h1>{t('home.hero_title')}</h1>
      <button onClick={() => setLang(lang === 'vi' ? 'en' : 'vi')}>
        {lang === 'vi' ? 'Switch to English' : 'Chuyển sang Tiếng Việt'}
      </button>
    </div>
  );
}
```

#### `index.ts` (Core Functions)

```typescript
const translations: Record<string, any> = {
  vi: require('../locales/vi.json'),
  en: require('../locales/en.json'),
};

export function translate(key: string, lang: string): string {
  const keys = key.split('.');
  let result = translations[lang];
  
  for (const k of keys) {
    if (result && typeof result === 'object' && k in result) {
      result = result[k];
    } else {
      return key;
    }
  }
  
  return result;
}

export function getCurrentLang(): string {
  return document.documentElement.lang || 'vi';
}

export function setLanguage(lang: string): void {
  document.documentElement.lang = lang;
  localStorage.setItem('lang', lang);
  window.location.reload();  // Reload to apply
}
```

---

## 🔧 How to Use i18n

### Backend (Django Views)

```python
def home_view(request):
    # Get current language
    lang = request.lang  # "vi" or "en"
    
    # Translate keys
    title = request.t("home.hero_title")
    desc = request.t("home.hero_desc")
    
    return render(request, "home.html", {
        "page_title": title,
        "description": desc,
    })
```

### Templates (Django HTML)

```html
{% load static %}

<!-- Simple translation -->
<h1>{{ t("home.hero_title") }}</h1>
<p>{{ t("home.hero_desc") }}</p>

<!-- With variables (if supported) -->
<p>{{ t("datasets.file_count") }} {{ total_files }}</p>

<!-- Language switcher -->
<div class="lang-switcher">
  {% if lang == "vi" %}
    <a href="?lang=en">🇬🇧 English</a>
  {% else %}
    <a href="?lang=vi">🇻🇳 Tiếng Việt</a>
  {% endif %}
</div>

<!-- Check language conditionally -->
{% if lang == "vi" %}
  <span>Xin chào!</span>
{% else %}
  <span>Hello!</span>
{% endif %}
```

### Frontend (JavaScript)

```javascript
// Get translation function
const t = window.i18n.t;

// Translate
const title = t('home.hero_title');
document.getElementById('title').textContent = title;

// Change language
function switchLanguage(lang) {
  window.location.href = `?lang=${lang}`;
}
```

---

## 🗂️ Translation Files

Translations are stored in **`../locales/`** directory:

```
locales/
├── vi.json    # Vietnamese translations
└── en.json    # English translations
```

### Structure Example

```json
{
  "home": {
    "hero_title": "Vietnam Weather Data in Real Time",
    "hero_desc": "Collect, process, and forecast weather data",
    "btn_crawl": "Start Data Collection",
    "btn_datasets": "Browse Datasets"
  },
  "datasets": {
    "tab_recent": "Recent Raw Data",
    "tab_merged": "Merged Data",
    "tab_cleaned": "Cleaned Data",
    "stat_total_files": "Total Files",
    "stat_total_size": "Total Size"
  },
  "auth": {
    "login_title": "Login to Your Account",
    "register_title": "Create New Account",
    "forgot_password": "Forgot Password?",
    "otp_sent": "OTP code sent to your email"
  }
}
```

---

## 🚀 Adding New Translations

### Step 1: Add to JSON Files

```json
// locales/vi.json
{
  "forecast": {
    "page_title": "Dự Báo Thời Tiết",
    "select_date": "Chọn ngày dự báo",
    "view_results": "Xem kết quả"
  }
}

// locales/en.json
{
  "forecast": {
    "page_title": "Weather Forecast",
    "select_date": "Select forecast date",
    "view_results": "View results"
  }
}
```

### Step 2: Use in Template

```html
<h1>{{ t("forecast.page_title") }}</h1>
<button>{{ t("forecast.view_results") }}</button>
```

### Step 3: Restart Django

```bash
# Restart server to reload JSON files
python manage.py runserver
```

---

## 🔍 Language Detection Priority

The system detects language in this order:

1. **URL Parameter**: `?lang=en` (highest priority)
2. **Session Storage**: `request.session['lang']`
3. **Accept-Language Header**: `request.META['HTTP_ACCEPT_LANGUAGE']`
4. **Default**: `"vi"` (Vietnamese)

---

## 🐛 Troubleshooting

### Translation Not Showing

```python
# Check if key exists
print(request.t("home.hero_title"))  # Should print translation

# If returns key itself, check:
# 1. JSON file syntax (valid JSON?)
# 2. Key spelling (case-sensitive!)
# 3. File encoding (UTF-8?)
```

### Language Not Switching

```python
# Verify middleware is registered
# settings.py
MIDDLEWARE = [
    # ...
    'Weather_Forcast_App.i18n.middleware.LangMiddleware',  # Must exist
]

# Check session middleware is BEFORE lang middleware
# SessionMiddleware must come first!
```

### Template `t()` Not Available

```python
# Verify context processor is registered
# settings.py
TEMPLATES = [{
    "OPTIONS": {
        "context_processors": [
            # ...
            "Weather_Forcast_App.i18n.context_processor.i18n_context",
        ]
    }
}]
```

---

## 📊 Current Coverage

| Category | Keys | Status |
|----------|------|--------|
| **Homepage** | 15 | ✅ Complete |
| **Datasets** | 25 | ✅ Complete |
| **Authentication** | 30 | ✅ Complete |
| **Training** | 20 | ✅ Complete |
| **Forecasting** | 18 | ✅ Complete |
| **Navigation** | 12 | ✅ Complete |
| **Errors** | 20 | ✅ Complete |
| **Buttons** | 15 | ✅ Complete |
| **Forms** | 25 | ✅ Complete |
| **TOTAL** | **250+** | ✅ **100%** |

---

## 🗺️ Future Enhancements

- [ ] Add **third language** (e.g., French, Chinese)
- [ ] **Pluralization** support (1 file vs 5 files)
- [ ] **Variable interpolation** in translations (e.g., "Hello, {name}!")
- [ ] **RTL language** support (Arabic, Hebrew)
- [ ] **Translation management** UI for non-developers
- [ ] **Auto-translation** via Google Translate API

---

## 📞 Related Files

- **Translations**: `locales/vi.json`, `locales/en.json`
- **Settings**: `WeatherForcast/settings.py` (middleware/context processor registration)
- **Templates**: `templates/weather/*.html` (use `{{ t() }}`)
- **Views**: `views/*.py` (use `request.t()`)

---

## 👨‍💻 Maintainer

**Võ Anh Nhật**  
📧 voanhnhat1612@gmail.com

---

*Last Updated: March 8, 2026*
