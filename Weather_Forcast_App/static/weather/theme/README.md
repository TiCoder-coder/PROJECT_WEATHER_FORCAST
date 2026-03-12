# 🎨 static/weather/theme — Theme System

## 📁 Overview

This directory contains **modular CSS files** that define the **design system** — a centralized, reusable set of **colors, typography, spacing, layouts, and animations**.

---

## 📂 Directory Structure

```
static/weather/theme/
├── index.css            # Theme entry point (imports all theme files)
├── color.css            # Color palette (CSS custom properties)
├── typography.css       # Font styles, sizes, line heights
├── spacing.css          # Margin/padding scale
├── layout.css           # Grid, flexbox, container utilities
├── animation.css        # Keyframes, transitions, transforms
├── effect.css           # Shadows, gradients, glassmorphism
├── themeProvider.css    # Theme switcher logic (dark/light)
└── README.md            # This file
```

---

## 🎯 Purpose

### What is a Design System?

A design system is a **collection of reusable styles** that ensure **consistency** across the application.

| Benefit | Description |
|---------|-------------|
| **Consistency** | Same colors, spacing everywhere |
| **Maintainability** | Change one variable, update entire app |
| **Speed** | Reuse utilities instead of writing custom CSS |
| **Scalability** | Add new pages with consistent design |

---

## 📄 Files Explained

### `index.css` — Theme Entry Point

#### Purpose

Import all theme modules in **correct order**.

#### Code

```css
/* Theme System Entry Point */
@import './color.css';
@import './typography.css';
@import './spacing.css';
@import './layout.css';
@import './animation.css';
@import './effect.css';
@import './themeProvider.css';
```

#### Usage

```html
<!-- Load entire theme system -->
<link rel="stylesheet" href="{% static 'weather/theme/index.css' %}">
```

---

### `color.css` — Color Palette

#### Purpose

Define **all colors** used in the application as **CSS custom properties**.

#### Example Code

```css
:root {
  /* Primary brand colors */
  --color-primary-50: #eff6ff;
  --color-primary-100: #dbeafe;
  --color-primary-200: #bfdbfe;
  --color-primary-500: #3b82f6;
  --color-primary-600: #2563eb;
  --color-primary-700: #1d4ed8;
  
  /* Neutral colors */
  --color-gray-50: #f9fafb;
  --color-gray-100: #f3f4f6;
  --color-gray-500: #6b7280;
  --color-gray-900: #111827;
  
  /* Semantic colors */
  --color-success: #22c55e;
  --color-error: #ef4444;
  --color-warning: #f59e0b;
  --color-info: #3b82f6;
  
  /* Background colors */
  --bg-primary: #0f172a;
  --bg-secondary: #1e293b;
  --bg-tertiary: #334155;
  
  /* Text colors */
  --text-primary: #f1f5f9;
  --text-secondary: #cbd5e1;
  --text-muted: #94a3b8;
}
```

#### Usage

```css
.button {
  background-color: var(--color-primary-500);
  color: var(--text-primary);
}

.button:hover {
  background-color: var(--color-primary-600);
}
```

---

### `typography.css` — Font System

#### Purpose

Define **font families, sizes, weights, line heights** for consistent typography.

#### Example Code

```css
:root {
  /* Font families */
  --font-sans: 'Inter', 'Segoe UI', 'Roboto', sans-serif;
  --font-mono: 'Fira Code', 'Consolas', monospace;
  
  /* Font sizes (fluid scaling) */
  --text-xs: 0.75rem;     /* 12px */
  --text-sm: 0.875rem;    /* 14px */
  --text-base: 1rem;      /* 16px */
  --text-lg: 1.125rem;    /* 18px */
  --text-xl: 1.25rem;     /* 20px */
  --text-2xl: 1.5rem;     /* 24px */
  --text-3xl: 1.875rem;   /* 30px */
  --text-4xl: 2.25rem;    /* 36px */
  
  /* Font weights */
  --font-light: 300;
  --font-normal: 400;
  --font-medium: 500;
  --font-semibold: 600;
  --font-bold: 700;
  
  /* Line heights */
  --leading-tight: 1.25;
  --leading-normal: 1.5;
  --leading-relaxed: 1.75;
}

/* Typography utilities */
.text-xs { font-size: var(--text-xs); }
.text-sm { font-size: var(--text-sm); }
.text-base { font-size: var(--text-base); }
.text-lg { font-size: var(--text-lg); }

.font-bold { font-weight: var(--font-bold); }
.font-medium { font-weight: var(--font-medium); }
```

---

### `spacing.css` — Spacing Scale

#### Purpose

Define **consistent spacing values** for margins and padding.

#### Example Code

```css
:root {
  --space-1: 0.25rem;   /* 4px */
  --space-2: 0.5rem;    /* 8px */
  --space-3: 0.75rem;   /* 12px */
  --space-4: 1rem;      /* 16px */
  --space-5: 1.25rem;   /* 20px */
  --space-6: 1.5rem;    /* 24px */
  --space-8: 2rem;      /* 32px */
  --space-10: 2.5rem;   /* 40px */
  --space-12: 3rem;     /* 48px */
  --space-16: 4rem;     /* 64px */
}

/* Margin utilities */
.m-1 { margin: var(--space-1); }
.m-2 { margin: var(--space-2); }
.m-4 { margin: var(--space-4); }

.mt-4 { margin-top: var(--space-4); }
.mb-4 { margin-bottom: var(--space-4); }

/* Padding utilities */
.p-4 { padding: var(--space-4); }
.px-4 { padding-left: var(--space-4); padding-right: var(--space-4); }
.py-4 { padding-top: var(--space-4); padding-bottom: var(--space-4); }
```

---

### `layout.css` — Layout Utilities

#### Purpose

Provide **flexbox, grid, and container** utilities.

#### Example Code

```css
/* Flexbox */
.flex { display: flex; }
.flex-col { flex-direction: column; }
.items-center { align-items: center; }
.justify-center { justify-content: center; }
.justify-between { justify-content: space-between; }
.gap-4 { gap: var(--space-4); }

/* Grid */
.grid { display: grid; }
.grid-cols-2 { grid-template-columns: repeat(2, 1fr); }
.grid-cols-3 { grid-template-columns: repeat(3, 1fr); }

/* Container */
.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 var(--space-4);
}
```

---

### `animation.css` — Animations & Transitions

#### Purpose

Define **keyframes** and **transition utilities**.

#### Example Code

```css
/* Transitions */
.transition { transition: all 0.3s ease; }
.transition-colors { transition: color, background-color 0.3s; }

/* Keyframe animations */
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

@keyframes slideUp {
  from { transform: translateY(20px); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

/* Animation utilities */
.animate-fade-in {
  animation: fadeIn 0.5s ease-out;
}

.animate-slide-up {
  animation: slideUp 0.6s ease-out;
}
```

---

### `effect.css` — Visual Effects

#### Purpose

Define **shadows, gradients, glassmorphism** effects.

#### Example Code

```css
:root {
  /* Shadow scale */
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
  --shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.1);
  --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.2);
  --shadow-xl: 0 20px 25px rgba(0, 0, 0, 0.3);
}

/* Shadow utilities */
.shadow-sm { box-shadow: var(--shadow-sm); }
.shadow-md { box-shadow: var(--shadow-md); }
.shadow-lg { box-shadow: var(--shadow-lg); }

/* Glassmorphism */
.glass {
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
}

/* Gradient backgrounds */
.gradient-primary {
  background: linear-gradient(135deg, var(--color-primary-500), var(--color-primary-700));
}
```

---

### `themeProvider.css` — Theme Switcher

#### Purpose

Support **dark/light theme switching**.

#### Example Code

```css
/* Default: Dark theme */
:root {
  --bg: var(--bg-primary);
  --text: var(--text-primary);
}

/* Light theme (when [data-theme="light"] is set) */
[data-theme="light"] {
  --bg: #ffffff;
  --text: #111827;
  --color-primary-500: #3b82f6;
}

/* Apply theme variables */
body {
  background-color: var(--bg);
  color: var(--text);
}
```

#### JavaScript (Theme Toggle)

```javascript
// Toggle theme
function toggleTheme() {
  const currentTheme = document.documentElement.getAttribute('data-theme');
  const newTheme = currentTheme === 'light' ? 'dark' : 'light';
  document.documentElement.setAttribute('data-theme', newTheme);
  localStorage.setItem('theme', newTheme);
}

// Load saved theme
const savedTheme = localStorage.getItem('theme') || 'dark';
document.documentElement.setAttribute('data-theme', savedTheme);
```

---

## 💡 Usage Examples

### Example 1: Build a Card Component

```html
<div class="glass p-6 shadow-lg animate-fade-in">
  <h3 class="text-xl font-bold mb-4">Weather Forecast</h3>
  <p class="text-base text-secondary">Temperature: 25°C</p>
</div>
```

### Example 2: Responsive Grid Layout

```html
<div class="container">
  <div class="grid grid-cols-3 gap-4">
    <div class="glass p-4">Card 1</div>
    <div class="glass p-4">Card 2</div>
    <div class="glass p-4">Card 3</div>
  </div>
</div>
```

---

## 🚀 Future Enhancements

- [ ] **CSS-in-JS support** (styled-components)
- [ ] **More color palettes** (green, purple themes)
- [ ] **Accessibility improvements** (focus states, ARIA)
- [ ] **Print styles** (optimize for printing)

---

## 📞 Related Files

- **CSS**: `static/weather/css/` (page-specific styles)
- **JavaScript**: `static/weather/js/` (theme toggle logic)

---

## 👨‍💻 Maintainer

**Võ Anh Nhật**  
📧 voanhnhat1612@gmail.com

---

*Last Updated: March 8, 2026*
