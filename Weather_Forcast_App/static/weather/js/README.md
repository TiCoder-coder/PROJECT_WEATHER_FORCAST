# ⚡ static/weather/js — JavaScript Files

## 📁 Overview

This directory contains **vanilla JavaScript files** that add **interactivity** and **dynamic behavior** to the Weather Forecast application.

---

## 📂 Directory Structure

```
static/weather/js/
├── Home.js                                      # Home page interactions
├── JS_Train.js                                  # Model training page logic
├── JS_Predict.js                                # Weather prediction page logic
├── JS_Crawl_data_by_API.js                      # Generic API crawling logic
├── JS_Crawl_data_from_Vrain_by_API.js           # Vrain API crawling logic
├── JS_Crawl_data_from_Vrain_by_Selenium.js      # Selenium crawling logic
├── JS_Crawl_data_from_html_of_Vrain.js          # HTML parsing crawling logic
└── README.md                                    # This file
```

---

## 🎯 Purpose

### Why Vanilla JavaScript?

This project uses **vanilla JS** (no jQuery/React) for:
- 🚀 **Performance**: No framework overhead
- 📦 **Lightweight**: Smaller bundle sizes
- 🎓 **Learning**: Direct DOM manipulation
- 🔧 **Control**: Full control over behavior

---

## 📄 Files Explained

### `Home.js` — Home Page Interactions

#### Purpose

Add **interactivity** to the home page (dashboard).

#### Features

- **Feature card animations**: Hover effects, click handlers
- **Quick stats updates**: Live KPI updates (if applicable)
- **Navigation**: Smooth scrolling to sections
- **Language switcher**: Client-side language toggle

#### Example Code

```javascript
// Smooth scroll to section
document.querySelectorAll('.nav-link').forEach(link => {
  link.addEventListener('click', (e) => {
    e.preventDefault();
    const target = document.querySelector(link.getAttribute('href'));
    target.scrollIntoView({ behavior: 'smooth' });
  });
});

// Animate feature cards on scroll
const observeCards = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('animate-in');
    }
  });
});

document.querySelectorAll('.feature-card').forEach(card => {
  observeCards.observe(card);
});
```

---

### `JS_Train.js` — Model Training Page

#### Purpose

Handle **ML model training workflow** (configure, train, monitor).

#### Features

1. **Configuration form validation**
2. **AJAX training requests**
3. **Real-time progress updates**
4. **Live metrics display** (loss, accuracy)
5. **Training logs streaming**

#### Example Code

```javascript
// Start training via AJAX
document.getElementById('train-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  
  const formData = new FormData(e.target);
  const config = Object.fromEntries(formData);
  
  // Show progress bar
  document.getElementById('progress-container').style.display = 'block';
  
  try {
    const response = await fetch('/api/train/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': getCookie('csrftoken')
      },
      body: JSON.stringify(config)
    });
    
    const data = await response.json();
    
    if (data.success) {
      // Poll for progress updates
      startProgressPolling(data.task_id);
    }
  } catch (error) {
    showError('Training failed: ' + error.message);
  }
});

// Poll training progress
function startProgressPolling(taskId) {
  const interval = setInterval(async () => {
    const response = await fetch(`/api/train/status/${taskId}/`);
    const data = await response.json();
    
    // Update progress bar
    document.getElementById('progress-bar').style.width = `${data.progress}%`;
    
    // Update metrics
    document.getElementById('loss').textContent = data.loss.toFixed(4);
    document.getElementById('accuracy').textContent = (data.accuracy * 100).toFixed(2) + '%';
    
    // Check if complete
    if (data.status === 'complete') {
      clearInterval(interval);
      showSuccess('Training complete!');
    }
  }, 2000); // Poll every 2 seconds
}
```

#### Key Functions

| Function | Purpose |
|----------|---------|
| `validateConfig()` | Validate hyperparameters |
| `startTraining()` | Submit training request |
| `startProgressPolling()` | Poll training status |
| `updateMetrics()` | Update loss/accuracy display |
| `streamLogs()` | Append training logs to console |

---

### `JS_Predict.js` — Weather Prediction Page

#### Purpose

Handle **weather forecasting** (input, predict, display).

#### Features

1. **Input validation** (date, location)
2. **AJAX prediction requests**
3. **Result display** (weather cards)
4. **Error handling**

#### Example Code

```javascript
// Submit prediction request
document.getElementById('predict-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  
  const date = document.getElementById('date').value;
  const location = document.getElementById('location').value;
  
  // Show loading spinner
  document.getElementById('loading').style.display = 'block';
  
  try {
    const response = await fetch('/api/predict/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': getCookie('csrftoken')
      },
      body: JSON.stringify({ date, location })
    });
    
    const data = await response.json();
    
    if (data.success) {
      displayForecast(data.forecast);
    } else {
      showError(data.error);
    }
  } catch (error) {
    showError('Prediction failed: ' + error.message);
  } finally {
    document.getElementById('loading').style.display = 'none';
  }
});

// Display forecast result
function displayForecast(forecast) {
  const container = document.getElementById('forecast-result');
  container.innerHTML = `
    <div class="weather-card">
      <h3>${forecast.location}</h3>
      <div class="temperature">${forecast.temperature}°C</div>
      <div class="condition">${forecast.condition}</div>
      <div class="humidity">Humidity: ${forecast.humidity}%</div>
      <div class="wind">Wind: ${forecast.wind_speed} km/h</div>
    </div>
  `;
}
```

---

### `JS_Crawl_data_by_API.js` — Generic API Crawling

#### Purpose

Handle **data crawling from APIs** (configure, fetch, preview).

#### Features

1. **API endpoint input**
2. **Parameter configuration**
3. **AJAX fetch requests**
4. **Response preview**
5. **Data storage**

#### Example Code

```javascript
// Fetch data from API
async function fetchApiData() {
  const endpoint = document.getElementById('api-endpoint').value;
  const params = getFormParams();
  
  try {
    const response = await fetch('/api/crawl/api/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': getCookie('csrftoken')
      },
      body: JSON.stringify({ endpoint, params })
    });
    
    const data = await response.json();
    
    if (data.success) {
      previewData(data.records);
      showSuccess(`Fetched ${data.count} records`);
    }
  } catch (error) {
    showError('API fetch failed: ' + error.message);
  }
}

// Preview fetched data
function previewData(records) {
  const table = document.getElementById('data-preview');
  table.innerHTML = `
    <thead>
      <tr>${Object.keys(records[0]).map(k => `<th>${k}</th>`).join('')}</tr>
    </thead>
    <tbody>
      ${records.map(r => `
        <tr>${Object.values(r).map(v => `<td>${v}</td>`).join('')}</tr>
      `).join('')}
    </tbody>
  `;
}
```

---

### `JS_Crawl_data_from_Vrain_by_Selenium.js` — Selenium Crawling

#### Purpose

Handle **Selenium-based web scraping** (start, monitor, stop).

#### Features

1. **Selenium configuration**
2. **Crawl status monitoring**
3. **Real-time log display**
4. **Stop/resume crawling**

#### Example Code

```javascript
// Start Selenium crawling
async function startCrawling() {
  const config = {
    url: document.getElementById('target-url').value,
    selectors: getSelectors(),
    headless: document.getElementById('headless').checked
  };
  
  const response = await fetch('/api/crawl/selenium/start/', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRFToken': getCookie('csrftoken')
    },
    body: JSON.stringify(config)
  });
  
  const data = await response.json();
  
  if (data.success) {
    startLogStreaming(data.task_id);
  }
}

// Stream crawling logs
function startLogStreaming(taskId) {
  const evtSource = new EventSource(`/api/crawl/selenium/logs/${taskId}/`);
  
  evtSource.onmessage = (event) => {
    const log = JSON.parse(event.data);
    appendLog(log.message);
  };
  
  evtSource.onerror = () => {
    evtSource.close();
    showError('Log stream disconnected');
  };
}
```

---

## 🔧 Common Patterns

### Pattern 1: CSRF Token Retrieval

```javascript
// Get CSRF token from cookie
function getCookie(name) {
  let cookieValue = null;
  if (document.cookie && document.cookie !== '') {
    const cookies = document.cookie.split(';');
    for (let i = 0; i < cookies.length; i++) {
      const cookie = cookies[i].trim();
      if (cookie.substring(0, name.length + 1) === (name + '=')) {
        cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
        break;
      }
    }
  }
  return cookieValue;
}
```

### Pattern 2: AJAX Request with Error Handling

```javascript
async function makeRequest(url, data) {
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': getCookie('csrftoken')
      },
      body: JSON.stringify(data)
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    
    const result = await response.json();
    return result;
  } catch (error) {
    console.error('Request failed:', error);
    showError(error.message);
    return null;
  }
}
```

### Pattern 3: Show/Hide Loading Spinner

```javascript
function showLoading() {
  document.getElementById('loading-spinner').style.display = 'block';
}

function hideLoading() {
  document.getElementById('loading-spinner').style.display = 'none';
}
```

---

## 🐛 Common Issues

### Issue 1: CSRF Verification Failed

**Error**: `403 Forbidden - CSRF verification failed`

**Cause**: Missing or invalid CSRF token in AJAX request

**Solution**: Always include CSRF token in headers
```javascript
headers: {
  'X-CSRFToken': getCookie('csrftoken')
}
```

### Issue 2: Script Not Loading

**Error**: `Uncaught ReferenceError: myFunction is not defined`

**Causes**:
1. Script loaded before DOM ready
2. Wrong file path
3. Script blocked by CSP

**Solution**:
```html
<!-- Load at end of body -->
<body>
  <!-- content -->
  <script src="{% static 'weather/js/Home.js' %}"></script>
</body>
```

---

## 🚀 Future Enhancements

- [ ] **Migrate to TypeScript** (type safety)
- [ ] **Add WebSocket support** (real-time updates)
- [ ] **Implement service workers** (offline support)
- [ ] **Add error tracking** (Sentry integration)
- [ ] **Bundle optimization** (Webpack/Vite)

---

## 📞 Related Files

- **CSS**: `static/weather/css/` (styling for elements)
- **Templates**: `templates/weather/` (HTML using these scripts)
- **Views**: `views/` (backend endpoints called by AJAX)

---

## 👨‍💻 Maintainer

**Võ Anh Nhật**  
📧 voanhnhat1612@gmail.com

---

*Last Updated: March 8, 2026*
