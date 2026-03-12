# 🏷️ templatetags — Custom Django Template Tags & Filters

## 📁 Overview

This directory contains **custom template tags** that extend Django's template language with **internationalization (i18n) utilities**.

---

## 📂 Directory Structure

```
templatetags/
├── __init__.py          # Required for Django to discover template tags
├── i18n_tags.py         # Custom i18n template tags ({% t %}, {% lang_url %})
└── README.md            # This file
```

---

## 🎯 Purpose

### What are Template Tags?

Template tags are **Python functions** that add **custom logic** to Django templates. They allow you to:
- Execute Python code in templates
- Process data before rendering
- Conditionally show/hide content
- Format data (dates, numbers, strings)

### Why Custom Tags?

Django provides built-in tags (`{% if %}`, `{% for %}`), but custom tags add **project-specific functionality**:
- **i18n**: Translate text dynamically
- **Formatting**: Custom date/number formats
- **Utilities**: Complex logic reusable across templates

---

## 📄 Files Explained

### `i18n_tags.py` — Internationalization Template Tags

#### Purpose

Provide **translation utilities** for Django templates, enabling **multi-language support** (Vietnamese/English).

#### Code Structure

```python
from django import template
from Weather_Forcast_App.i18n import translate, detect_language

register = template.Library()  # Required to register tags
```

---

## 🔧 Template Tags Explained

### Tag 1: `{% t %}` — Translate Text

#### Purpose

Translate a **translation key** to the current user's language.

#### Syntax

```django
{% load i18n_tags %}

<!-- Basic usage -->
{% t "auth.login.title" %}

<!-- Assign to variable -->
{% t "nav.home" as home_label %}
{{ home_label }}

<!-- Use with variables -->
{% t some_key_variable %}
```

#### How It Works

```python
class TranslateNode(template.Node):
    def __init__(self, key_expr, var_name=None):
        self.key_expr = key_expr
        self.var_name = var_name
    
    def render(self, context):
        # 1. Resolve translation key from context
        key = self.key_expr.resolve(context)
        
        # 2. Get language from context (set by middleware)
        lang = context.get("lang") or "vi"
        
        # 3. Translate using i18n module
        result = translate(str(key), lang)
        
        # 4. Return result or assign to variable
        if self.var_name:
            context[self.var_name] = result
            return ""
        return result
```

#### Example 1: Translate Login Title

**Template**:
```django
{% load i18n_tags %}
<h1>{% t "auth.login.title" %}</h1>
```

**Output (Vietnamese)**:
```html
<h1>Đăng nhập</h1>
```

**Output (English)**:
```html
<h1>Login</h1>
```

#### Example 2: Assign to Variable

**Template**:
```django
{% load i18n_tags %}
{% t "nav.home" as home_text %}
<a href="/" title="{{ home_text }}">{{ home_text }}</a>
```

**Output (Vietnamese)**:
```html
<a href="/" title="Trang chủ">Trang chủ</a>
```

#### Example 3: Dynamic Key

**Template**:
```django
{% load i18n_tags %}
{% for item in menu_items %}
  <li>{% t item.label_key %}</li>
{% endfor %}
```

**Context**:
```python
context = {
    "menu_items": [
        {"label_key": "nav.home"},
        {"label_key": "nav.datasets"},
        {"label_key": "nav.train"},
    ]
}
```

**Output (Vietnamese)**:
```html
<li>Trang chủ</li>
<li>Bộ dữ liệu</li>
<li>Huấn luyện</li>
```

---

### Tag 2: `{% lang_url %}` — Language Switcher URL

#### Purpose

Generate a URL that **switches the current page to a different language** while preserving existing query parameters.

#### Syntax

```django
{% load i18n_tags %}

<!-- Switch to English -->
<a href="{% lang_url 'en' %}">EN</a>

<!-- Switch to Vietnamese -->
<a href="{% lang_url 'vi' %}">VI</a>
```

#### How It Works

```python
@register.simple_tag(takes_context=True)
def lang_url(context, lang_code):
    """
    Return the current URL with ?lang=<lang_code> appended.
    Preserves existing query parameters.
    """
    request = context.get("request")
    if request is None:
        return f"?lang={lang_code}"
    
    # Copy existing query params
    params = request.GET.copy()
    
    # Override lang parameter
    params["lang"] = lang_code
    
    # Return path + updated query string
    return f"{request.path}?{params.urlencode()}"
```

#### Example 1: Basic Language Switcher

**Template**:
```django
{% load i18n_tags %}
<div class="lang-switcher">
  <a href="{% lang_url 'vi' %}" class="{% if lang == 'vi' %}active{% endif %}">
    🇻🇳 Tiếng Việt
  </a>
  <a href="{% lang_url 'en' %}" class="{% if lang == 'en' %}active{% endif %}">
    🇬🇧 English
  </a>
</div>
```

**Output (on /datasets/ page)**:
```html
<div class="lang-switcher">
  <a href="/datasets/?lang=vi" class="active">🇻🇳 Tiếng Việt</a>
  <a href="/datasets/?lang=en">🇬🇧 English</a>
</div>
```

#### Example 2: Preserve Existing Query Params

**Current URL**: `/search/?q=weather&page=2&lang=vi`

**Template**:
```django
<a href="{% lang_url 'en' %}">Switch to English</a>
```

**Output**:
```html
<a href="/search/?q=weather&page=2&lang=en">Switch to English</a>
```

**Notice**: `q` and `page` parameters are **preserved**, only `lang` is changed.

---

## 💡 Usage Examples

### Example 1: Multilingual Navigation

```django
{% load i18n_tags %}
<nav>
  <ul>
    <li><a href="/">{% t "nav.home" %}</a></li>
    <li><a href="/datasets/">{% t "nav.datasets" %}</a></li>
    <li><a href="/train/">{% t "nav.train" %}</a></li>
    <li><a href="/forecast/">{% t "nav.forecast" %}</a></li>
  </ul>
  
  <div class="lang-toggle">
    <a href="{% lang_url 'vi' %}">VI</a> | 
    <a href="{% lang_url 'en' %}">EN</a>
  </div>
</nav>
```

### Example 2: Translated Form Labels

```django
{% load i18n_tags %}
<form method="post">
  {% csrf_token %}
  
  <label for="username">{% t "auth.login.username" %}</label>
  <input type="text" id="username" name="username" 
         placeholder="{% t 'auth.login.username_placeholder' %}">
  
  <label for="password">{% t "auth.login.password" %}</label>
  <input type="password" id="password" name="password"
         placeholder="{% t 'auth.login.password_placeholder' %}">
  
  <button type="submit">{% t "auth.login.submit" %}</button>
</form>
```

### Example 3: Dynamic Error Messages

```django
{% load i18n_tags %}
{% if error %}
  <div class="alert alert-danger">
    {% t error_key %}
  </div>
{% endif %}
```

**View**:
```python
def login_view(request):
    if user is None:
        return render(request, "login.html", {
            "error": True,
            "error_key": "errors.invalid_credentials"
        })
```

---

## 🐛 Common Issues

### Issue 1: Tag Not Found

**Error**: `Invalid block tag: 't'`

**Cause**: Forgot to load template tag library

**Solution**: Add `{% load i18n_tags %}` at top of template
```django
{% load i18n_tags %}  <!-- Add this -->
<h1>{% t "auth.login.title" %}</h1>
```

### Issue 2: Translation Key Not Found

**Output**: Returns the key itself instead of translation (e.g., `"auth.login.title"`)

**Cause**: Translation key doesn't exist in `locales/vi.json` or `locales/en.json`

**Solution**: Add missing key to translation files
```json
// locales/vi.json
{
  "auth": {
    "login": {
      "title": "Đăng nhập"
    }
  }
}
```

### Issue 3: Language Not Switching

**Symptom**: Clicking language switcher doesn't change language

**Causes**:
1. Middleware not enabled
2. Translation files not loaded
3. Browser cached old page

**Solution**:
```python
# settings.py
MIDDLEWARE = [
    # ...
    'Weather_Forcast_App.i18n.middleware.LangMiddleware',  # Ensure this exists
    # ...
]
```

---

## 🔍 Best Practices

### ✅ DO

- **Load tags once** at the top of each template
- **Use descriptive keys** (`auth.login.title` not `lt`)
- **Nest keys logically** (group by page/feature)
- **Provide fallbacks** (default to Vietnamese if key missing)

### ❌ DON'T

- **Don't hardcode text** that should be translated
- **Don't use `t()` in view** (use `translate()` function instead)
- **Don't overuse `as varname`** (only when needed multiple times)
- **Don't forget to update both** `vi.json` and `en.json`

---

## 🚀 Future Enhancements

- [ ] Add **{% tf %}** tag for formatted strings (e.g., `{% tf "welcome" user=username %}`)
- [ ] Add **{% t_html %}** tag for safe HTML translations
- [ ] Add **{% plural %}** tag for pluralization (1 item vs 2 items)
- [ ] Add **date format** tag for locale-aware dates
- [ ] Add **number format** tag for locale-aware numbers

---

## 📞 Related Files

- **Translations**: `locales/vi.json`, `locales/en.json`
- **i18n Module**: `i18n/__init__.py` (translate function)
- **Middleware**: `i18n/middleware.py` (language detection)
- **Context Processor**: `i18n/context_processor.py` (inject `lang` into context)

---

## 👨‍💻 Maintainer

**Võ Anh Nhật**  
📧 voanhnhat1612@gmail.com

---

*Last Updated: March 8, 2026*
