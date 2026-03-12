# 🎨 static/weather/css — Stylesheet Files

## 📁 Overview

This directory contains **CSS stylesheets** that define the **visual design** of the Weather Forecast application.

---

## 📂 Directory Structure

```
static/weather/css/
├── _common.css                                  # Common styles (resets, variables, utilities)
├── _sidebar.css                                 # Sidebar navigation styles
├── _summary.css                                 # Summary widget styles
├── Auth.css                                     # Login/Register page styles
├── Datasets.css                                 # Dataset management page styles
├── Dataset_preview.css                          # Dataset preview modal styles
├── Home.css                                     # Home page styles
├── Sidebar.css                                  # Sidebar styles (alternative)
├── CSS_Train.css                                # Model training page styles
├── CSS_Predict.css                              # Weather prediction page styles
├── CSS_Crawl_data_by_API.css                    # API data crawling page styles
├── CSS_Crawl_data_from_Vrain_by_API.css         # Vrain API crawling styles
├── CSS_Crawl_data_from_Vrain_by_Selenium.css    # Selenium crawling styles
├── CSS_Crawl_data_from_html_of_Vrain.css        # HTML parsing crawling styles
└── README.md                                    # This file
```

---

## 🎯 Purpose

### CSS Organization Strategy

Stylesheets are organized by **page/feature** rather than component type:

| Strategy | Pros | Cons |
|----------|------|------|
| **Per-Page CSS** (Current) | ✅ Easy to find styles<br>✅ Clear page ownership | ❌ Potential duplication |
| **Component CSS** | ✅ Reusable<br>✅ DRY principle | ❌ Harder to locate |
| **Utility CSS** (Tailwind) | ✅ Fast development | ❌ Verbose HTML |

---

## 📄 Files Explained

### Common Files

#### `_common.css` — Global Styles

**Purpose**: Define **global CSS variables**, **resets**, and **utility classes** used across all pages.

**Contents**:
- CSS custom properties (`:root` variables)
- Typography base styles
- Color palette definitions
- Common animations
- Flexbox/Grid utilities

**Example**:
```css
:root {
  --primary-color: #3b82f6;
  --bg-dark: #0f172a;
  --text-light: #f1f5f9;
  --spacing-sm: 0.5rem;
  --spacing-md: 1rem;
  --spacing-lg: 2rem;
}

*, *::before, *::after {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

.flex-center {
  display: flex;
  justify-content: center;
  align-items: center;
}
```

#### `_sidebar.css` — Sidebar Navigation

**Purpose**: Style the **left sidebar navigation** (common across all pages).

**Features**:
- Fixed position sidebar
- Active link highlighting
- Icon styling
- Hover effects
- Responsive collapse

**Key Classes**:
- `.sidebar` — Main container
- `.sidebar-nav` — Navigation list
- `.sidebar-link` — Navigation link
- `.sidebar-link.active` — Active page indicator

#### `_summary.css` — Summary Widgets

**Purpose**: Style **summary cards** and **KPI widgets** (dashboard widgets).

**Features**:
- Card layout
- Icon integration
- Gradient backgrounds
- Hover effects

---

### Page-Specific Files

#### `Auth.css` — Login/Register Page

**Purpose**: Style authentication pages (login, register, forgot password).

**Features**:
- Centered login form
- Glassmorphism card effect
- Input field styling
- Button animations
- Error message styling

**Key Elements**:
```css
.auth-container { /* Centered full-screen container */ }
.auth-card { /* Glassmorphism login card */ }
.auth-input { /* Styled input fields */ }
.auth-button { /* Gradient button with hover effect */ }
.auth-error { /* Red error message */ }
```

#### `Home.css` — Home Page

**Purpose**: Style the **main dashboard/home page**.

**Features**:
- Hero section with gradient background
- Feature cards grid
- Animated buttons (recently updated by sếp!)
- Responsive layout

**Recent Changes** (Phase 2):
- Added multi-layer box shadows
- Gradient backgrounds
- Transform animations (translateY, scale)
- Hover effects

**Key Classes**:
```css
.hero-section { /* Hero banner */ }
.feature-grid { /* 3-column grid */ }
.feature-card { /* Individual feature card */ }
.cta-button { /* Call-to-action button */ }
```

#### `Datasets.css` — Dataset Management

**Purpose**: Style the **dataset management page** (browse, upload, delete datasets).

**Features**:
- Table layout for dataset list
- Upload form styling
- Background effects (recently reduced by sếp!)
- Action button styling

**Recent Changes** (Phase 1):
- Reduced opacity (14% → 4%)
- Disabled animations (`ambientFloat`, `grainShift`)
- Softer visual effects

**Key Classes**:
```css
.dataset-table { /* Dataset listing table */ }
.dataset-row { /* Table row */ }
.upload-zone { /* Drag-and-drop upload area */ }
.dataset-actions { /* Edit/Delete buttons */ }
```

#### `CSS_Train.css` — Model Training Page

**Purpose**: Style the **ML model training interface**.

**Features**:
- Training configuration form
- Progress indicators
- Model metrics display
- Epoch loss/accuracy charts
- Training logs console

**Key Classes**:
```css
.train-config-form { /* Hyperparameter inputs */ }
.train-progress { /* Progress bar */ }
.train-logs { /* Console-style logs */ }
.metrics-grid { /* Loss/accuracy display */ }
```

#### `CSS_Predict.css` — Weather Prediction Page

**Purpose**: Style the **weather forecast interface**.

**Features**:
- Forecast input form
- Weather card display
- Temperature visualization
- Icon integration
- Responsive grid

**Key Classes**:
```css
.predict-form { /* Input form */ }
.weather-card { /* Forecast result card */ }
.temperature-badge { /* Temperature display */ }
.weather-icon { /* Weather condition icon */ }
```

---

### Data Crawling Files

All `CSS_Crawl_*` files style different **data collection interfaces**:

| File | Purpose |
|------|---------|
| **CSS_Crawl_data_by_API.css** | Generic API crawling interface |
| **CSS_Crawl_data_from_Vrain_by_API.css** | Vrain API crawling page |
| **CSS_Crawl_data_from_Vrain_by_Selenium.css** | Selenium-based crawling |
| **CSS_Crawl_data_from_html_of_Vrain.css** | HTML parsing crawling |

**Common Features**:
- API endpoint input fields
- Parameter configuration forms
- Response preview
- Status indicators
- Error handling displays

---

## 🎨 Design System

### Color Palette

```css
:root {
  /* Primary */
  --primary: #3b82f6;
  --primary-hover: #2563eb;
  
  /* Backgrounds */
  --bg-dark: #0f172a;
  --bg-medium: #1e293b;
  --bg-light: #334155;
  
  /* Text */
  --text-primary: #f1f5f9;
  --text-secondary: #cbd5e1;
  --text-muted: #94a3b8;
  
  /* Status */
  --success: #22c55e;
  --error: #ef4444;
  --warning: #f59e0b;
}
```

### Typography

```css
:root {
  --font-family: 'Inter', 'Segoe UI', sans-serif;
  --font-size-base: 16px;
  --font-size-sm: 0.875rem;
  --font-size-lg: 1.125rem;
  --font-size-xl: 1.5rem;
  --font-size-2xl: 2rem;
}
```

### Spacing

```css
:root {
  --spacing-xs: 0.25rem;  /* 4px */
  --spacing-sm: 0.5rem;   /* 8px */
  --spacing-md: 1rem;     /* 16px */
  --spacing-lg: 1.5rem;   /* 24px */
  --spacing-xl: 2rem;     /* 32px */
  --spacing-2xl: 3rem;    /* 48px */
}
```

---

## 🔧 Usage Examples

### Example 1: Loading Stylesheets in Template

```django
{% load static %}
<!DOCTYPE html>
<html>
<head>
  <!-- Common styles (load first) -->
  <link rel="stylesheet" href="{% static 'weather/css/_common.css' %}">
  <link rel="stylesheet" href="{% static 'weather/css/_sidebar.css' %}">
  
  <!-- Page-specific styles -->
  <link rel="stylesheet" href="{% static 'weather/css/Home.css' %}">
</head>
<body>
  <!-- content -->
</body>
</html>
```

### Example 2: Using CSS Variables

```css
/* In your custom CSS */
.my-button {
  background-color: var(--primary);
  padding: var(--spacing-md) var(--spacing-lg);
  color: var(--text-primary);
  font-size: var(--font-size-base);
}

.my-button:hover {
  background-color: var(--primary-hover);
}
```

---

## 🐛 Common Issues

### Issue 1: Styles Not Applied

**Symptoms**:
- CSS file loaded but styles don't apply
- Browser shows 304 (Not Modified)

**Causes**:
1. Browser cache
2. CSS specificity conflict
3. Wrong file path

**Solutions**:
```bash
# Clear Django static cache
python manage.py collectstatic --clear

# Hard refresh browser
Ctrl + Shift + R (Windows/Linux)
Cmd + Shift + R (Mac)
```

### Issue 2: CSS Variables Not Working

**Error**: Styles using `var()` not rendering

**Cause**: Variables defined in wrong scope or after usage

**Solution**:
```css
/* ❌ WRONG (variables not defined in :root) */
.my-class {
  --my-color: red;
}
.other-class {
  color: var(--my-color); /* Won't work */
}

/* ✅ CORRECT (define in :root) */
:root {
  --my-color: red;
}
.other-class {
  color: var(--my-color); /* Works */
}
```

---

## 🚀 Future Enhancements

- [ ] **Migrate to CSS modules** (scoped styles per component)
- [ ] **Add dark/light theme** toggle (CSS custom properties)
- [ ] **Optimize file sizes** (minify, remove unused rules)
- [ ] **Add print stylesheet** (print-friendly views)
- [ ] **Implement CSS Grid** layouts (replace float/flexbox where appropriate)

---

## 📞 Related Files

- **Theme System**: `static/weather/theme/` (theme CSS files)
- **Templates**: `templates/weather/` (HTML files using these styles)
- **JavaScript**: `static/weather/js/` (interactive styling)

---

## 👨‍💻 Maintainer

**Võ Anh Nhật**  
📧 voanhnhat1612@gmail.com

---

*Last Updated: March 8, 2026*
