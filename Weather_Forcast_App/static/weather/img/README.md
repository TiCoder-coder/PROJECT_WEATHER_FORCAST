# 🖼️ static/weather/img — Image Assets

## 📁 Overview

This directory contains **image files** used throughout the Weather Forecast application, including **icons, logos, backgrounds, and illustrations**.

---

## 📂 Expected Directory Structure

```
static/weather/img/
├── logo.png                 # Application logo
├── logo-dark.png            # Logo for dark theme
├── logo-light.png           # Logo for light theme
├── favicon.ico              # Browser favicon
├── weather-icons/           # Weather condition icons
│   ├── sunny.svg
│   ├── cloudy.svg
│   ├── rainy.svg
│   ├── stormy.svg
│   └── ...
├── backgrounds/             # Background images
│   ├── hero-bg.jpg
│   ├── pattern.svg
│   └── ...
└── README.md                # This file
```

---

## 🎯 Purpose

### Why Store Images Here?

- **Static files**: Django serves from `static/` via `{% static %}`
- **Caching**: Browser caches static assets
- **CDN-ready**: Can be moved to CDN for production
- **Organization**: Centralized image management

---

## 📄 Image Categories

### 1. **Logos & Branding**

| File | Purpose | Size Recommendation |
|------|---------|---------------------|
| `logo.png` | Main app logo | 200x50px (transparent PNG) |
| `logo-dark.png` | Dark theme logo | 200x50px |
| `logo-light.png` | Light theme logo | 200x50px |
| `favicon.ico` | Browser tab icon | 32x32px |

---

### 2. **Weather Icons**

Weather condition icons (SVG preferred for scalability):

| Icon | Condition | File |
|------|-----------|------|
| ☀️ | Sunny/Clear | `sunny.svg` |
| ⛅ | Partly Cloudy | `partly-cloudy.svg` |
| ☁️ | Cloudy | `cloudy.svg` |
| 🌧️ | Rainy | `rainy.svg` |
| ⛈️ | Stormy | `stormy.svg` |
| 🌫️ | Foggy | `foggy.svg` |
| ❄️ | Snowy | `snowy.svg` |
| 🌪️ | Windy | `windy.svg` |

---

### 3. **Backgrounds**

| File | Purpose |
|------|---------|
| `hero-bg.jpg` | Home page hero background |
| `pattern.svg` | Repeating pattern for sections |
| `gradient-overlay.png` | Overlay for sections |

---

### 4. **Illustrations**

| File | Purpose |
|------|---------|
| `no-data.svg` | Empty state illustration |
| `error-404.svg` | 404 error page |
| `loading.gif` | Loading spinner |

---

## 🔧 Usage Examples

### Example 1: Display Logo

```html
{% load static %}
<img src="{% static 'weather/img/logo.png' %}" alt="Weather Forecast Logo" class="logo">
```

### Example 2: Weather Icon (Dynamic)

```html
{% load static %}
<img src="{% static 'weather/img/weather-icons/' %}{{ forecast.icon }}.svg" 
     alt="{{ forecast.condition }}" 
     class="weather-icon">
```

**Backend (View)**:
```python
forecast = {
    "condition": "Rainy",
    "icon": "rainy",  # Maps to rainy.svg
    "temperature": 25
}
```

### Example 3: Background Image (CSS)

```css
.hero-section {
  background-image: url('/static/weather/img/backgrounds/hero-bg.jpg');
  background-size: cover;
  background-position: center;
}
```

### Example 4: Favicon

```html
<head>
  <link rel="icon" href="{% static 'weather/img/favicon.ico' %}" type="image/x-icon">
</head>
```

---

## 🎨 Image Optimization Best Practices

### ✅ DO

1. **Compress images** before uploading
   - Use TinyPNG, ImageOptim, or Squoosh
   - Target: < 100KB per image

2. **Use SVG for icons**
   - Scalable without quality loss
   - Smaller file size
   - Can be styled with CSS

3. **Use WebP format** for photos (modern browsers)
   ```html
   <picture>
     <source srcset="hero.webp" type="image/webp">
     <img src="hero.jpg" alt="Hero">
   </picture>
   ```

4. **Provide alt text** for accessibility
   ```html
   <img src="logo.png" alt="Weather Forecast Application Logo">
   ```

5. **Lazy load images** (below the fold)
   ```html
   <img src="large-image.jpg" loading="lazy" alt="Forecast chart">
   ```

### ❌ DON'T

- **Don't upload raw photos** (3MB+)
- **Don't use BMP or TIFF** formats
- **Don't forget retina images** (2x resolution for high-DPI screens)
- **Don't hardcode URLs** (use `{% static %}` tag)

---

## 🐛 Common Issues

### Issue 1: Image Not Found (404)

**Error**: Browser shows broken image icon

**Causes**:
1. Wrong file path
2. File not collected (`collectstatic` not run)
3. Typo in filename

**Solution**:
```bash
# Check file exists
ls static/weather/img/logo.png

# Run collectstatic (production)
python manage.py collectstatic

# Verify URL in browser DevTools Network tab
```

### Issue 2: Image Loads Slowly

**Symptoms**: Page renders but images load after 2-3 seconds

**Solutions**:
```html
<!-- 1. Add width/height to prevent layout shift -->
<img src="logo.png" width="200" height="50" alt="Logo">

<!-- 2. Use lazy loading -->
<img src="large.jpg" loading="lazy" alt="Forecast">

<!-- 3. Serve from CDN (production) -->
<img src="https://cdn.example.com/img/logo.png" alt="Logo">
```

### Issue 3: Image Quality Loss

**Problem**: Image looks pixelated/blurry

**Causes**:
1. Over-compressed
2. Wrong dimensions (scaled up)
3. Using raster instead of vector

**Solution**:
- Use **SVG** for icons/logos
- Provide **2x retina images** for photos:
  ```html
  <img src="logo.png" 
       srcset="logo.png 1x, logo@2x.png 2x" 
       alt="Logo">
  ```

---

## 📊 Image Formats Comparison

| Format | Best For | Pros | Cons |
|--------|----------|------|------|
| **SVG** | Icons, logos | Scalable, small size, CSS-stylable | Not for photos |
| **PNG** | Logos with transparency | Lossless, transparency | Larger than JPEG |
| **JPEG** | Photos, backgrounds | Small size, good compression | No transparency |
| **WebP** | Modern photos | Smaller than JPEG, transparency | Not supported in old browsers |
| **GIF** | Animations | Widely supported | Limited colors, large size |

---

## 🚀 Future Enhancements

- [ ] **Image CDN** (CloudFlare, AWS S3 + CloudFront)
- [ ] **Responsive images** (use `srcset` for different screen sizes)
- [ ] **Progressive JPEGs** (load low-res first, then high-res)
- [ ] **Image sprites** (combine small icons into one file)
- [ ] **Automated optimization** (GitHub Actions to compress on commit)

---

## 📞 Related Files

- **Templates**: `templates/weather/` (HTML using these images)
- **CSS**: `static/weather/css/` (background images)
- **Django Settings**: `settings.py` (STATIC_URL, STATIC_ROOT)

---

## 📚 Resources

- **Image Optimization**: https://web.dev/fast/#optimize-your-images
- **SVG Icons**: https://heroicons.com/, https://fontawesome.com/
- **Compression Tools**: https://tinypng.com/, https://squoosh.app/

---

## 👨‍💻 Maintainer

**Võ Anh Nhật**  
📧 voanhnhat1612@gmail.com

---

*Last Updated: March 8, 2026*
