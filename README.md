<div align="center">

# 🌦️ VN WEATHER HUB: NỀN TẢNG WEB THUTHẬP - CHUẨN HOÁ - HỢP NHẤT DỮ LIỆU THỜI TIẾT ĐA NGUỒN VÀ DỰ BÁO NGẮN HẠN CHO VIỆT NAM BẰNG MÔ HÌNH ENSEMBLE

**Full-stack Django application** for **weather data collection** → **processing** → **machine learning** → **forecasting** with beautiful glassmorphism UI and multilingual support.

<br/>

<img alt="Python" src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white">
<img alt="Django" src="https://img.shields.io/badge/Django-6.0.1-092E20?logo=django&logoColor=white">
<img alt="MongoDB" src="https://img.shields.io/badge/MongoDB-7.0+-47A248?logo=mongodb&logoColor=white">
<img alt="Pandas" src="https://img.shields.io/badge/Pandas-2.2+-150458?logo=pandas&logoColor=white">
<img alt="Scikit-learn" src="https://img.shields.io/badge/Scikit--learn-1.8+-F7931E?logo=scikit-learn&logoColor=white">
<img alt="XGBoost" src="https://img.shields.io/badge/XGBoost-3.2+-FF6600?logo=xgboost&logoColor=white">
<img alt="LightGBM" src="https://img.shields.io/badge/LightGBM-4.6+-02569B?logo=lightgbm&logoColor=white">
<img alt="CatBoost" src="https://img.shields.io/badge/CatBoost-1.2+-FFCC00?logo=catboost&logoColor=black">
<img alt="UI" src="https://img.shields.io/badge/UI-Glassmorphism-7C3AED">
<img alt="i18n" src="https://img.shields.io/badge/i18n-VI%20%7C%20EN-22C55E">

<br/>
<sub>🌐 Multi-source crawling • 🔗 Smart merging • 🧹 Data cleaning • 🧠 ML training • 🔮 Forecasting • 📊 Dashboard • 🎨 Modern UI • 🌍 Bilingual</sub>

</div>

---

<div align="center">
  
![VN Weather Hub Dashboard](https://nub.news/api/image/681000/article.png)

*Enterprise-grade weather data pipeline with modern glassmorphism interface*

</div>

---

## 📌 Table of Contents

<details open>
<summary><b>📚 Navigation</b></summary>

- [🎯 Overview](#-overview)
- [✨ Key Features](#-key-features)
- [🧠 Machine Learning Pipeline](#-machine-learning-pipeline)
- [🌍 Internationalization (i18n)](#-internationalization-i18n)
- [🎨 UI/UX Features](#-uiux-features)
- [🔐 Authentication System](#-authentication-system)
- [📊 Data Pipeline](#-data-pipeline)
- [� Airflow — Automated Scheduling](#-airflow--automated-scheduling)
- [�🗂️ Project Structure](#️-project-structure)
- [🚀 Installation & Setup](#-installation--setup)
- [📖 User Guide](#-user-guide)
- [🔧 Configuration](#-configuration)
- [🐛 Troubleshooting](#-troubleshooting)
- [🗺️ Roadmap](#️-roadmap)
- [👥 Team](#-team)

</details>

---

## 🎯 Overview

**VN Weather Hub** is a comprehensive **Django-based weather intelligence platform** that provides end-to-end data pipeline capabilities—from multi-source data collection to advanced machine learning forecasting. Built with modern web technologies, it features a stunning glassmorphism UI, bilingual support (Vietnamese/English), and enterprise-grade security.

### 🎯 Core Objectives

- 🌐 **Multi-Source Data Collection**: Crawl weather data from OpenWeather API, Vrain API, Selenium scraping, and HTML parsing
- 🔗 **Intelligent Data Integration**: Merge heterogeneous datasets with conflict resolution and schema validation
- 🧹 **Advanced Data Cleaning**: Automated cleaning wizard with customizable pipelines
- 🧠 **Machine Learning**: Train and deploy forecasting models (Ensemble, XGBoost, LightGBM, CatBoost, RandomForest)
- 🔮 **Weather Forecasting**: Generate multi-day weather predictions with confidence intervals
- 📊 **Interactive Dashboard**: Real-time metrics, visualization, and dataset management
- 🎨 **Modern UX**: Glassmorphism design with weather-themed animations
- 🌍 **Multilingual**: Full Vietnamese and English localization.

---

### 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Django 6.0.1 Web App               │
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ Auth/JWT │  │  i18n    │  │ Login Required   │  │
│  │Middleware│  │Middleware│  │  Middleware       │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │             Views / URL Router               │   │
│  │  Home │ Crawl │ Datasets │ Train │ Predict   │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  ┌──────────────────┐   ┌────────────────────────┐  │
│  │   ML Pipeline    │   │   Data Pipeline        │  │
│  │  ┌────────────┐  │   │  Crawl → Merge → Clean │  │
│  │  │ Ensemble   │  │   │  (4 crawl methods)     │  │
│  │  │ XGBoost    │  │   └────────────────────────┘  │
│  │  │ LightGBM   │  │                               │
│  │  │ CatBoost   │  │   ┌────────────────────────┐  │
│  │  │ RandomForest│ │   │   Data Storage         │  │
│  │  └────────────┘  │   │  SQLite (ORM) + MongoDB │  │
│  └──────────────────┘   └────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

**Luồng dữ liệu chính (End-to-End Pipeline):**

```
[Nguồn dữ liệu] → [Thu thập] → [Gộp file] → [Làm sạch] → [Huấn luyện] → [Dự báo]
 OpenWeather API    Crawl page   Datasets      Datasets      Train page    Predict page
 Vrain API          (4 methods)  (Merge tab)   (Clean tab)   (ML models)   (4 input tabs)
 Vrain Selenium
 Vrain HTML Parser
```

---

### 🖥️ Homepage Structure (System Overview)

Trang Home được chia thành **3 sections chính**:

1. **Hero Section** — Banner chào mừng với hiệu ứng thời tiết động (mưa, gió, sét, aurora, mây).
   - 5 nút hành động nhanh: `🌐 Thu thập dữ liệu` / `📊 Datasets` / `🧠 Train` / `🔮 Dự báo` / `❓ Hướng dẫn`
   - Hiển thị thời gian cập nhật dataset mới nhất
   - Navigation bar: chuyển ngôn ngữ `VI|EN`, links đăng nhập/đăng ký/hồ sơ

2. **System Overview Section** — 4 cards mô tả các tính năng cốt lõi:
   - 🌐 Thu thập qua API (OpenWeather / Vrain API)
   - 🌧️ Thu thập qua Vrain (Selenium / HTML Parser)
   - 📊 Báo cáo & làm sạch dữ liệu
   - 🧠 Dự báo bằng Machine Learning

3. **Activity Dashboard Section** — Thống kê hoạt động thực tế:
   - Số file từ pipeline API Weather
   - Số file từ pipeline Vrain
   - Trạng thái model ML hiện tại (loại model + ready/training)
   - Card dự báo thời tiết (truy cập nhanh)

**Các modal tương tác trên Homepage:**

| Modal | Mở bằng | Nội dung |
|-------|---------|----------|
| 🌐 **Crawl Method Modal** | Nút `🌐 Thu thập dữ liệu` | Chọn 1 trong 4 phương thức crawl |
| ❓ **Help Modal** | Nút `❓ Hướng dẫn` | Hướng dẫn 4 bước sử dụng app |
| 🌦️ **Intro Modal** | Nút `Giới thiệu` trên nav | Giới thiệu hệ thống, luồng xử lý, nguồn dữ liệu, quickstart |

---

## ✨ Key Features

### 📊 Data Management

| Feature | Description |
|---------|-------------|
| 🌐 **Multi-source crawling** | OpenWeather API, Vrain API/Selenium/HTML parser |
| 🔗 **Smart merging** | Automatic schema alignment, conflict resolution, duplicate removal |
| 🧹 **Cleaning wizard** | Step-by-step guided cleaning with preview and validation |
| 👁️ **Dataset preview** | In-browser CSV/Excel/JSON/TXT preview with pagination |
| ⬇️ **Export** | Download in multiple formats (CSV, XLSX, JSON, TXT) |
| 📈 **Statistics** | Real-time metrics, file counts, dataset sizes |

### 🧠 Machine Learning

| Feature | Description |
|---------|-------------|
| 🏗️ **Model training** | Support for 5 algorithms: Ensemble, XGBoost, LightGBM, CatBoost, RandomForest |
| 🔮 **Forecasting** | Multi-day weather prediction with rain detection |
| 📊 **Evaluation** | Comprehensive metrics: MAE, RMSE, MAPE, R², Rain Accuracy |
| 🎛️ **Hyperparameter tuning** | Grid search, random search, Optuna integration |
| 💾 **Model versioning** | Automatic artifact management and versioning |
| 📓 **Notebook pipeline** | Jupyter notebook for entire ML workflow (31 cells) |

### 🌍 Internationalization

| Feature | Description |
|---------|-------------|
| 🇻🇳 **Vietnamese** | Full Vietnamese localization (default) |
| 🇬🇧 **English** | Complete English translation |
| 🔄 **Live switching** | Toggle language without page reload |
| 🎨 **Context-aware** | Dynamic translations with template tags |
| 📝 **250+ strings** | Comprehensive coverage of all UI elements |

### 🔐 Security & Authentication

| Feature | Description |
|---------|-------------|
| 🔑 **Login** | Username or email login with JWT tokens |
| 📝 **Registration** | Email OTP verification before account creation |
| 🔄 **Password reset** | Secure OTP-based password recovery |
| 🛡️ **Account lockout** | Auto-lock after 5 failed attempts (5 min) |
| 📧 **Email validation** | MX record check, disposable email blocking |
| 🔐 **Password security** | Pepper + hash, strength validation |
| 👤 **Profile** | User profile management and settings |

### 🎨 UI/UX Excellence

| Feature | Description |
|---------|-------------|
| 🪟 **Glassmorphism** | Modern frosted glass design with blur effects |
| 🌧️ **Weather animations** | Rain, wind, lightning, mist, aurora effects |
| 🎭 **Responsive** | Mobile-first design, works on all devices |
| 🎨 **Theme effects** | Dynamic gradient backgrounds, neon accents |
| ⚡ **Performance** | Optimized animations with reduced-motion support |
| 🎯 **Accessibility** | ARIA labels, keyboard navigation, screen reader support |

---

## 🧠 Machine Learning Pipeline

### 📋 Complete ML Workflow (6 Phases)

The platform includes a comprehensive Jupyter notebook (`tremblingProcess.ipynb`) that orchestrates the entire ML pipeline:

```mermaid
graph TD
    A[Phase 1: Data Crawl] -->|Raw CSV| B[Phase 2: Merge]
    B -->|Merged CSV| C[Phase 3: Clean]
    C -->|Cleaned CSV| D[Phase 4: Train Model]
    D -->|Model Artifacts| E[Phase 5: Forecast]
    E -->|Predictions| F[Phase 6: Diagnostics]
    
    D -->|Feature_list.json| G[ML Artifacts]
    D -->|Metrics.json| G
    D -->|Train_info.json| G
    D -->|Model.pkl| G
```

### 🔄 Phase Details

#### Phase 1: Data Collection
- Run crawling scripts (API/Selenium/HTML)
- Output: Raw CSV files in `data/data_crawl/`
- Sources: OpenWeather, Vrain (multiple methods)

#### Phase 2: Data Merging
- Execute `scripts/merge_csv.py`
- Align schemas, resolve conflicts
- Output: Merged file in `data/data_merge/`

#### Phase 3: Data Cleaning (4 steps)
1. ~~Column renaming~~ → **SKIPPED** (handled by Cleardata)
2. Run `Cleardata.py` (main cleaning)
3. ~~Additional cleaning~~ → **SKIPPED** (duplicate)
4. Clean NaN datetime with `clean_datetime_nan.py`

#### Phase 4: Model Training
- Update `train_config.json` with latest dataset
- Run `manage.py train_model`
- Generates 4 artifacts:
  - `Feature_list.json` — Feature schema for predictions
  - `Metrics.json` — Model performance metrics
  - `Train_info.json` — Training configuration metadata
  - `Model.pkl` — Serialized trained model

#### Phase 5: Weather Forecasting
- Sử dụng trang **Predict** trên web hoặc gọi `WeatherPredictor.from_artifacts().predict(df)`
- Output: file `predictions.csv` lưu tại `Weather_Forcast_App/runtime/`
- Kết quả bao gồm cột `y_pred` (lượng mưa dự báo) và `status` (Mưa/Không mưa)

#### Phase 6: Diagnostics
- Run `scripts/run_diagnostics.py`
- Generate health check reports
- Validate data quality

### 🎯 Supported Machine Learning Models

| Model | Technology | Use Case | Ghi chú |
|-------|-----------|----------|---------|
| **RandomForest** | Scikit-learn | Baseline, đánh giá tầm quan trọng feature | Ổn định, dễ diễn giải |
| **XGBoost** | XGBoost 3.2 | Gradient boosting hiệu năng cao | Tốc độ nhanh, chống overfitting |
| **LightGBM** | LightGBM 4.6 | Tập dữ liệu lớn, training nhanh | Tối ưu bộ nhớ |
| **CatBoost** | CatBoost 1.2 | Xử lý categorical features | Gradient boosting chuyên sâu |
| **Ensemble** | Voting (tất cả models) | Dự báo ổn định, giảm variance | **Model production — xem metrics thực tế bên dưới** |

> **Ghi chú:** Các metrics R² của từng model riêng lẻ phụ thuộc vào từng lần huấn luyện và tập dữ liệu cụ thể. Số liệu chính xác được ghi nhận từ production ensemble (test set) ở bảng bên dưới.

### 📊 Model Evaluation Metrics (production ensemble — test set)

```json
{
  "model": "ensemble",
  "applied_log_target": true,
  "metrics": {
    "r2_score": 0.619,
    "mae": 0.730,
    "rmse": 1.254,
    "rain_detection_accuracy": 0.641,
    "rain_f1": 0.752
  },
  "training_time": "~90s",
  "dataset_size": 210457,
  "features_count": 120
}
```

> **Lưu ý:** target `rain_total` được áp dụng log1p transform trước khi train. MAE/RMSE tính trên không gian log.

### 🗂️ ML Artifacts Structure

```
Machine_learning_artifacts/
├── latest/                        # Current production model
│   ├── Feature_list.json          # Feature schema (120 features)
│   ├── Metrics.json               # Performance metrics
│   ├── Train_info.json            # Training metadata
│   ├── Model.pkl                  # Trained model (serialized)
│   └── Transform_pipeline.pkl    # Feature transform pipeline
└── old_model/                     # Previous version backup
    ├── Feature_list.json
    ├── Metrics.json
    └── Train_info.json
```

### 🔧 Training Configuration

**File:** `config/train_config.json`

```json
{
  "data": {
    "filename": "merged_vrain_data_cleaned_20260307_223506.clean_final.csv",
    "target_column": "rain_total",
    "date_column": "datetime"
  },
  "model": {
    "type": "ensemble",
    "params": {
      "n_estimators": 100,
      "max_depth": 10,
      "learning_rate": 0.1
    }
  },
  "split": {
    "train_ratio": 0.8,
    "valid_ratio": 0.1,
    "test_ratio": 0.1,
    "shuffle": false
  }
}
```

### 🚀 Training a Model

#### Via Django Management Command
```bash
python manage.py train_model
```

#### Via Direct Script
```bash
python Weather_Forcast_App/Machine_learning_model/trainning/train.py \
  --config Weather_Forcast_App/Machine_learning_model/config/train_config.json
```

#### Via Jupyter Notebook
```bash
jupyter notebook Weather_Forcast_App/Evaluate_accuracy/tremblingProcess.ipynb
```

### 🔮 Making Predictions

#### Via Web Interface
1. Navigate to **🔮 Weather Forecast** page
2. Select date range and location
3. View predictions with confidence intervals

#### Via Python Script
```python
from Weather_Forcast_App.Machine_learning_model.interface.predictor import WeatherPredictor

predictor = WeatherPredictor()
predictions = predictor.predict(date='2026-03-10', city='Ho Chi Minh City')
print(f"Predicted rain: {predictions['rain_total']} mm")
```

---

## 🌍 Internationalization (i18n)

### 🔄 Language Support

VN Weather Hub provides comprehensive bilingual support with seamless language switching.

#### Supported Languages
- 🇻🇳 **Tiếng Việt** (Vietnamese) — Default
- 🇬🇧 **English** — Full translation

#### Features
- ✅ **250+ translated strings** covering all UI elements
- ✅ **Live switching** without page reload
- ✅ **Context-aware** translations via custom template tags
- ✅ **URL persistence** — language preference saved in URL
- ✅ **Session storage** — remembers user language choice

### 📂 Translation Files

```
Weather_Forcast_App/i18n/
├── locales/
│   ├── vi.json                  # Vietnamese translations
│   └── en.json                  # English translations
├── context_processor.py         # Django context processor
├── middleware.py                # Language detection middleware
└── hooks.ts / index.ts          # Frontend i18n utilities
```

### 🔧 Usage in Templates

```html
{% load i18n_tags %}

<!-- Simple translation -->
<h1>{% t "home.hero_title" %}</h1>

<!-- With variables -->
<p>{% t "datasets.file_count" count=total_files %}</p>

<!-- Language switcher -->
{% if lang == 'vi' %}
  <a href="{% lang_url 'en' %}">Switch to English</a>
{% else %}
  <a href="{% lang_url 'vi' %}">Chuyển sang Tiếng Việt</a>
{% endif %}
```

### 📝 Translation Keys (Examples)

#### Homepage (`home.*`)
```json
{
  "home.hero_title": "Vietnam Weather Data in Real Time",
  "home.hero_desc": "Collect, process, and forecast weather data with machine learning",
  "home.btn_crawl": "Start Data Collection",
  "home.btn_datasets": "Browse Datasets",
  "home.btn_train": "Train Model",
  "home.btn_predict": "Weather Forecast"
}
```

#### Datasets Page (`datasets.*`)
```json
{
  "datasets.tab_recent": "Recent Raw Data",
  "datasets.tab_merged": "Merged Data",
  "datasets.tab_cleaned": "Cleaned Data",
  "datasets.tab_process": "Process Data",
  "datasets.stat_total_files": "Total Files",
  "datasets.stat_total_size": "Total Size"
}
```

#### Authentication (`auth.*`)
```json
{
  "auth.login_title": "Login to Your Account",
  "auth.register_title": "Create New Account",
  "auth.forgot_password": "Forgot Password?",
  "auth.email_verification": "Email Verification Required",
  "auth.otp_sent": "OTP code sent to your email"
}
```

### 🌐 Adding New Translations

1. **Add key to JSON files:**
```json
// locales/vi.json
{
  "feature.new_key": "Nội dung tiếng Việt"
}

// locales/en.json
{
  "feature.new_key": "English content"
}
```

2. **Use in template:**
```html
{% load i18n_tags %}
<p>{% t "feature.new_key" %}</p>
```

3. **Restart Django server** to reload translations

---

## 🎨 UI/UX Features

### 🪟 Glassmorphism Design System

The platform uses a modern glassmorphism design language inspired by Apple's design philosophy:

#### Visual Elements
- **Frosted glass cards** with backdrop-filter blur
- **Semi-transparent backgrounds** with gradient overlays
- **Neon accent colors** for CTAs and highlights
- **Soft shadows** with multiple layers
- **Smooth transitions** on all interactive elements

#### Color Palette
```css
/* Primary Colors */
--cyan: #22d3ee;           /* Accent cyan */
--blue: #3b82f6;           /* Primary blue */
--purple: #a78bfa;         /* Secondary purple */
--green: #22c55e;          /* Success green */

/* Background Gradients */
--bg0: #020617;            /* Deep dark blue-black */
--bg1: #0f172a;            /* Dark slate */
--card: rgba(15,23,42,0.7);/* Glass card background */

/* Text Colors */
--text: #e5e7eb;           /* Main text (light gray) */
--muted: #94a3b8;          /* Muted text (slate) */
```

### 🌧️ Weather-Themed Animations

#### Background Effects (CSS-based)
- **Rain effect** — Vertical rain streaks with shimmer animation
- **Wind blow** — Horizontal gradient sweep
- **Mist/Fog** — Floating radial gradients
- **Aurora shimmer** — Large-scale color rotation
- **Lightning flash** — Periodic brightness pulses
- **Wave ripple** — Horizontal sine-wave patterns

#### Interactive Animations
- **Button hover effects** — Scale + glow + lift transforms
- **Card entrance** — Fade-in with slide-up
- **Logo pulse** — Breathing animation with color cycle
- **Floating orbs** — Particle system drifting across viewport

#### Accessibility
```css
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
```

### 📱 Responsive Design

| Breakpoint | Device | Adjustments |
|-----------|--------|-------------|
| `>1200px` | Desktop | Full 3-column grid, side-by-side panels |
| `768px-1200px` | Tablet | 2-column grid, collapsible sidebar |
| `<768px` | Mobile | Single column, stacked layout, bottom nav |

### ✨ Enhanced Button System

#### Primary Button (Main Actions)
```css
.btn.primary {
  background: linear-gradient(135deg, #0ea5e9 0%, #22c55e 100%);
  border: 1.5px solid rgba(34,211,238,0.28);
  box-shadow: 
    0 16px 48px rgba(34,211,238,0.32),
    0 8px 24px rgba(52,211,153,0.22),
    0 0 20px rgba(34,211,238,0.12) inset;
}

.btn.primary:hover {
  transform: translateY(-3px) scale(1.02);
  box-shadow: 
    0 22px 56px rgba(34,211,238,0.4),
    0 14px 32px rgba(52,211,153,0.28),
    0 0 28px rgba(34,211,238,0.2);
}
```

#### Features
- **Gradient backgrounds** with smooth color transitions
- **Multi-layer shadows** for depth perception
- **Scale transforms** on hover (1.02x)
- **Lift effect** with translateY(-3px)
- **Pulse animation** on primary CTAs

---

## 🔐 Authentication System

### 📋 Features Overview

| Feature | Description |
|---------|-------------|
| 🔑 **Login** | Username or email login with JWT tokens |
| 📝 **Registration** | Two-step process with email OTP verification |
| 🔄 **Password Reset** | Three-step OTP-based recovery |
| 🛡️ **Account Lockout** | Auto-lock after 5 failed attempts (5 minutes) |
| 📧 **Email Security** | MX record validation, disposable email blocking |
| 🔐 **Password Strength** | Min 8 chars, uppercase, lowercase, number, special char |
| 👤 **Profile Management** | Update name, email, view activity |

### 🔑 Login Flow

```
User Input (username/email + password)
        ↓
Find account in MongoDB (username OR email)
        ↓
Check account locked? → Yes: Show lockout message
        ↓ No
Check account active? → No: Show activation message
        ↓ Yes
Verify password (hash + pepper)
        ↓
Create JWT token (role, manager_id)
        ↓
Save to session + Set cookie
        ↓
Redirect to Home page ✅
```

#### Security Features
- **Pepper-enhanced hashing:** `hash(password + PEPPER_SECRET)`
- **Failed attempt tracking:** Counter increments on wrong password
- **Auto-lockout:** 5 attempts → lock for 5 minutes
- **JWT tokens:** Stateless authentication with expiry
- **Session management:** Secure session storage

### 📝 Registration Flow (2 Steps)

#### Step 1: Register Form
```
Input: First/Last Name, Username, Email, Password
        ↓
Validation:
├── Email format + MX records check
├── Block disposable emails (tempmail, guerrilla, etc.)
├── Username unique check
├── Email not already registered
├── Password strength validation
        ↓
Generate 5-digit OTP
        ↓
Send OTP email (Gmail SMTP / Resend API)
        ↓
Save registration data to session (temporary)
        ↓
Redirect to OTP verification page
```

#### Step 2: OTP Verification
```
Input: 5-digit OTP code
        ↓
Verify OTP hash (SHA-256 + salt)
        ↓
Check expiry (10 minutes TTL)
        ↓
Check attempts (max 5)
        ↓
Create account in MongoDB
        ↓
Auto-login with JWT
        ↓
Redirect to Home ✅
```

#### Password Requirements
```
✅ Minimum 8 characters
✅ At least 1 lowercase letter (a-z)
✅ At least 1 uppercase letter (A-Z)
✅ At least 1 digit (0-9)
✅ At least 1 special character (!@#$%^&*()-_+=)
```

#### Email Validation
| Check | Description |
|-------|-------------|
| **Syntax** | RFC 5322 format validation |
| **Unicode** | Reject non-ASCII characters |
| **MX Records** | Verify domain can receive email |
| **Disposable** | Block tempmail, mailinator, guerrillamail, etc. |
| **Trusted Domains** | Skip MX check for gmail.com, yahoo.com, outlook.com |

### 🔄 Password Reset Flow (3 Steps)

#### Step 1: Request Reset
```
Input: Email address
        ↓
Check email exists in database
        ↓
Generate 5-digit OTP
        ↓
Send OTP email
        ↓
Redirect to OTP verification
```

#### Step 2: Verify OTP
```
Input: 5-digit OTP
        ↓
Verify OTP hash + expiry + attempts
        ↓
Mark OTP as verified
        ↓
Redirect to password reset form
```

#### Step 3: Set New Password
```
Input: New password + Confirm password
        ↓
Validate password strength
        ↓
Check passwords match
        ↓
Update password in database (hash + pepper)
        ↓
Invalidate all existing OTPs
        ↓
Redirect to Login ✅
```

### 📧 Email OTP System

#### OTP Security
| Feature | Implementation |
|---------|----------------|
| **Generation** | `secrets.randbelow(100000)` (cryptographically secure) |
| **Storage** | SHA-256 hash(otp + salt + secret_key), never plain text |
| **Expiry** | 10 minutes TTL (MongoDB TTL index auto-delete) |
| **Attempts** | Max 5 attempts, then request new OTP |
| **Invalidation** | Old OTPs invalidated when new one generated |

#### Email Providers (Priority Order)
1. **Gmail SMTP** (recommended) — Most reliable
2. **Resend API** — If RESEND_API_KEY configured
3. **Console Mode** — Development fallback (prints to terminal)

#### Gmail SMTP Configuration
```env
EMAIL_BACKEND=django.core.mail.backends.smtp.EmailBackend
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_HOST_USER=your_email@gmail.com
EMAIL_HOST_PASSWORD=your_16_char_app_password
EMAIL_USE_TLS=True
DEFAULT_FROM_EMAIL=VN Weather Hub <your_email@gmail.com>
```

**How to get Gmail App Password:**
1. Go to [Google Account](https://myaccount.google.com/)
2. Enable **2-Step Verification** (Security section)
3. Generate **App Password** for Mail app
4. Copy 16-character password to `EMAIL_HOST_PASSWORD`

### 👤 Profile Management

**Route:** `/profile/`

#### Features
- View account details (name, username, email, role)
- Update first/last name
- Update email (with uniqueness check)
- View registration date and last login
- See account role (Staff/Manager/Admin)

### 🗄️ MongoDB Collections

#### Collection: `logins`
```javascript
{
  "_id": ObjectId("..."),
  "name": "Võ Anh Nhật",
  "userName": "nhat123",
  "email": "nhat@gmail.com",
  "password": "pbkdf2_sha256$...",  // Hashed with pepper
  "role": "Staff",                   // Staff | Manager | Admin
  "is_active": true,
  "failed_attempts": 0,
  "lock_until": null,
  "last_login": ISODate("2026-03-08T10:30:00Z"),
  "createdAt": ISODate("2026-01-15T08:00:00Z"),
  "updatedAt": ISODate("2026-03-08T10:30:00Z")
}
```

#### Collection: `email_verification_otps`
```javascript
{
  "_id": ObjectId("..."),
  "email": "nhat@gmail.com",
  "otpHash": "sha256_hash_string",  // Never stores plain OTP
  "salt": "random_hex_salt",
  "attempts": 0,
  "used": false,
  "createdAt": ISODate("2026-03-08T10:00:00Z"),
  "expiresAt": ISODate("2026-03-08T10:10:00Z"),  // TTL: 10 min
  "verifiedAt": null                              // Set when verified
}
```

#### Collection: `password_reset_otps`
```javascript
{
  "_id": ObjectId("..."),
  "email": "nhat@gmail.com",
  "otpHash": "sha256_hash_string",
  "salt": "random_hex_salt",
  "attempts": 0,
  "used": false,
  "createdAt": ISODate("2026-03-08T10:00:00Z"),
  "expiresAt": ISODate("2026-03-08T10:10:00Z"),
  "verifiedAt": null
}
```

---

## 📊 Data Pipeline

### 🗂️ Directory Structure

```
data/
├── data_crawl/              # Raw data from crawlers (12 files)
│   ├── Bao_cao_20260307_160220.xlsx
│   ├── Bao_cao_20260307_160230.csv
│   └── ...
├── data_merge/              # Merged datasets (2 files + log)
│   ├── merged_vrain_data.xlsx
│   ├── merged_files_log.txt
│   └── ...
├── data_clean/              # Cleaned datasets
│   ├── data_merge_clean/   # Cleaned from merged data (21 files)
│   │   ├── cleaned_merge_merged_vrain_data_20260307.csv
│   │   └── ...
│   └── data_not_merge_clean/  # Cleaned from raw data
│       ├── cleaned_output_Bao_cao_20260307.csv
│       └── ...

```

### 🔄 Data Flow

```mermaid
flowchart LR
  A[🌐 Crawl Modules] --> B[📦 data_crawl/]
  B -->|Merge| C[🔗 data_merge/]
  C -->|Clean| D[🧹 data_clean/data_merge_clean/]
  B -->|Clean| E[🧹 data_clean/data_not_merge_clean/]
  D --> F[📊 Datasets Page]
  E --> F
  C --> F
  F --> G[👁️ Preview/Download]
  D -->|Train| H[🧠 ML Model]
  H -->|Predict| I[🔮 runtime/predictions.csv]
```

### 📥 Data Sources

| Source | Method | File Format | Frequency |
|--------|--------|-------------|-----------|
| **OpenWeather API** | REST API | JSON → CSV | Hourly |
| **Vrain API** | REST API | JSON → CSV | Daily |
| **Vrain Selenium** | Web scraping | HTML → CSV | On-demand |
| **Vrain HTML Parser** | Static parsing | HTML → CSV | On-demand |

### 🔗 Merge Process

#### Features
- **Schema alignment**: Automatically align column names and types
- **Conflict resolution**: Handle duplicate timestamps with configurable strategies
- **Deduplication**: Remove exact duplicate rows
- **Validation**: Check data integrity before/after merge
- **Logging**: Detailed merge log with file info and statistics

#### Merge Strategies
```python
# Configuration in merge script
MERGE_CONFIG = {
    "on": ["datetime", "city"],        # Join keys
    "how": "outer",                    # Join type
    "suffixes": ["_src1", "_src2"],   # Handle column conflicts
    "validate": "1:1"                  # Ensure no duplicates
}
```

### 🧹 Clean Process

#### Cleaning Pipeline (4 Steps)

**Step 1:** Column Renaming (Skipped - handled by Cleardata)

**Step 2:** Main Cleaning (`Cleardata.py`)
- Remove duplicate rows
- Handle missing values (fill/drop)
- Standardize datetime formats
- Remove outliers
- Normalize units (e.g., temperature °C/°F, pressure hPa/mmHg)
- Validate data ranges

**Step 3:** Additional Cleaning (Skipped - duplicate of Step 2)

**Step 4:** DateTime NaN Cleaning (`clean_datetime_nan.py`)
- Fix invalid datetime values
- Fill missing dates with interpolation
- Remove rows with critical datetime errors

#### Clean Wizard UI (3 Steps)

```
Step 1: Select Source
├── 🔗 Merged data (from data_merge/)
└── 📦 Raw data (from data_crawl/)
        ↓
Step 2: Choose File
├── Search/filter files
├── View file metadata (size, date)
└── Select file to clean
        ↓
Step 3: Track Progress
├── Real-time log output
├── Progress percentage
├── Error/warning messages
└── Download cleaned file ✅
```

### 👁️ Dataset Preview

#### CSV/Excel (Table Mode)
- **Pagination**: 50 rows per page
- **Sortable columns**: Click header to sort
- **Search**: Filter by any column value
- **Metadata**: Row count, column count, file size
- **Export**: Download as CSV/XLSX/JSON

#### JSON (Syntax Highlighted)
```json
{
  "datetime": "2026-03-08T10:00:00",
  "temperature": 28.5,
  "humidity": 75,
  "rain_total": 12.3,
  "city": "Ho Chi Minh City"
}
```

#### TXT (Preformatted)
```
=== Weather Data Log ===
Date: 2026-03-08
Temperature: 28.5°C
Humidity: 75%
Rain: 12.3mm
```

### ⬇️ Download Options

| Format | Use Case | Features |
|--------|----------|----------|
| **CSV** | Data analysis, Excel | UTF-8 BOM, comma-delimited |
| **XLSX** | Business reports | Formatted headers, auto-width |
| **JSON** | API integration | Pretty-printed, indented |
| **TXT** | Logs, debugging | Plain text, line-delimited |

---

## � Airflow — Automated Scheduling

### 📌 Overview

VN Weather Hub dùng **Apache Airflow** để tự động crawl dữ liệu thời tiết theo lịch cố định trong ngày. Chỉ sử dụng **HTML parser crawler** để đảm bảo ổn định.

| Thành phần | Chi tiết |
|------------|----------|
| **Crawler** | `Crawl_data_from_html_of_Vrain.py` |
| **Output** | `data/data_crawl/` |
| **Scheduler** | Apache Airflow 2.x |
| **Timezone** | `Asia/Bangkok` (UTC+7) |

### 🐳 Kiến trúc Docker

Airflow dùng chung `docker-compose.yml` với Django:

| Service | Vai trò |
|---------|---------|
| `web` | Django application |
| `postgres` | Airflow metadata database |
| `airflow-init` | Container khởi tạo một lần |
| `airflow-webserver` | Airflow UI (port 8080) |
| `airflow-scheduler` | DAG scheduler |

### 🗓️ DAG và lịch crawl

**File:** `airflow/dags/weather_crawl_schedule.py`

| DAG | Cron | Giờ chạy (UTC+7) |
|-----|------|------------------|
| `weather_crawl_morning` | `0 5-7 * * *` | 05:00, 06:00, 07:00 |
| `weather_crawl_noon` | `0,45 12-14 * * *` | 12:00, 12:45, 13:00, 13:45, 14:00, 14:45 |
| `weather_crawl_evening` | `0 19-21 * * *` | 19:00, 20:00, 21:00 |

Mỗi DAG gồm **2 task** chạy tuần tự:

```
Task 1: crawl_vrain_html
└── Chạy: Weather_Forcast_App/scripts/Crawl_data_from_html_of_Vrain.py
        ↓
Task 2: list_latest_outputs
└── In danh sách file mới nhất trong data/data_crawl/
```

### 🚀 Cài đặt & Khởi động

#### 1️⃣ Chuẩn bị file môi trường
```bash
cp .env.example .env
```

Đảm bảo các biến sau có mặt trong `.env`:
```env
AIRFLOW_ADMIN_USERNAME=admin
AIRFLOW_ADMIN_PASSWORD=admin
AIRFLOW_ADMIN_EMAIL=admin@example.com
```

#### 2️⃣ Khởi tạo Airflow
```bash
docker compose up airflow-init
```

#### 3️⃣ Khởi động toàn bộ services
```bash
docker compose up -d
```

### 🌐 Truy cập

| Service | URL | Credentials mặc định |
|---------|-----|----------------------|
| **Django** | http://localhost:8000 | — |
| **Airflow UI** | http://localhost:8080 | admin / admin |

### 🔍 Kiểm tra hệ thống

**Trạng thái container:**
```bash
docker compose ps
```

**Danh sách file crawl mới nhất:**
```bash
ls -lt data/data_crawl/ | head -10
```

### ⏱️ Lưu ý về timestamp trong CSV

| Cột | Ý nghĩa |
|-----|---------|
| `timestamp` | Thời điểm dữ liệu mưa từ nguồn Vrain (vd: `19:00`) |
| `data_time` | Thời điểm script crawl ghi nhận, đã đổi sang UTC+7 |

### 🎯 Trigger thủ công để test

```bash
docker exec project_weather_forcast-airflow-scheduler-1 \
  airflow dags trigger weather_crawl_evening
```

### 📋 Vị trí log

| Loại log | Đường dẫn |
|----------|-----------|
| **Scheduler DAG parse** | `airflow/logs/scheduler/latest/weather_crawl_schedule.py.log` |
| **Task run** | `airflow/logs/dag_id=<dag_id>/run_id=<run_id>/task_id=<task_id>/attempt=*.log` |

### 🛑 Dừng hệ thống

```bash
docker compose down
```

---

## �🗂️ Project Structure

```
PROJECT_WEATHER_FORECAST/
├── 📁 Weather_Forcast_App/         # Main Django application
│   ├── 📁 Enums/                   # Enumerations and constants
│   ├── 📁 Machine_learning_artifacts/
│   │   ├── 📁 latest/              # Current model artifacts
│   │   │   ├── Feature_list.json   # Feature schema
│   │   │   ├── Metrics.json        # Performance metrics
│   │   │   ├── Train_info.json     # Training metadata
│   │   │   └── Model.pkl           # Trained model
│   │   └── 📁 old_model/           # Previous version backup
│   ├── 📁 Machine_learning_model/  # ML pipeline modules
│   │   ├── 📁 config/              # Training configurations
│   │   │   ├── train_config.json   # Main config
│   │   │   └── default.yaml        # Default parameters
│   │   ├── 📁 data/                # Data loading and validation
│   │   │   ├── Loader.py           # Load CSV/XLSX to DataFrame
│   │   │   ├── Schema.py           # Data schema validation
│   │   │   └── Split.py            # Train/valid/test split
│   │   ├── 📁 evaluation/          # Model evaluation
│   │   │   ├── metrics.py          # Metric calculations
│   │   │   └── report.py           # Report generation
│   │   ├── 📁 features/            # Feature engineering
│   │   │   ├── Build_transfer.py   # Feature construction
│   │   │   └── Transformers.py     # Data transformers
│   │   ├── 📁 interface/           # Prediction interface
│   │   │   └── predictor.py        # WeatherPredictor class
│   │   ├── 📁 Models/              # ML model implementations
│   │   │   ├── Base_model.py         # Abstract base class
│   │   │   ├── Random_Forest_Model.py
│   │   │   ├── XGBoost_Model.py
│   │   │   ├── LightGBM_Model.py
│   │   │   ├── CatBoost_Model.py
│   │   │   ├── Ensemble_Model.py     # Voting/weighted mean ensemble
│   │   │   ├── TwoStage_Model.py     # Two-stage classifier + regressor
│   │   │   └── ml_types.py           # Type definitions
│   │   ├── 📁 trainning/           # Training pipeline
│   │   │   ├── train.py            # Main training script
│   │   │   └── tuning.py           # Hyperparameter tuning
│   │   └── 📁 WeatherForcast/      # Forecasting module
│   │       └── WeatherForcast.py   # Forecast generator
│   ├── 📁 Evaluate_accuracy/       # Evaluation notebooks
│   │   ├── tremblingProcess.ipynb  # Full ML pipeline (31 cells)
│   │   ├── evaluate.ipynb          # Model evaluation
│   │   └── MODEL_EVALUATION_SUMMARY.md
│   ├── 📁 i18n/                    # Internationalization
│   │   ├── 📁 locales/
│   │   │   ├── vi.json             # Vietnamese translations
│   │   │   └── en.json             # English translations
│   │   ├── context_processor.py    # Django context processor
│   │   ├── middleware.py           # Language middleware
│   │   └── hooks.ts / index.ts     # Frontend i18n
│   ├── 📁 middleware/              # Django middleware
│   │   ├── Auth.py                 # Authentication middleware
│   │   ├── Authentication.py       # Session handling
│   │   └── Jwt_handler.py          # JWT token processing
│   ├── 📁 Models/                  # Django models
│   │   └── Login.py                # User model (MongoDB)
│   ├── 📁 Repositories/            # Data access layer
│   │   └── Login_repositories.py   # User CRUD operations
│   ├── 📁 Seriallizer/             # Data serializers
│   │   └── 📁 Login/
│   │       ├── Base_login.py
│   │       ├── Create_login.py
│   │       └── Update_login.py
│   ├── 📁 management/              # Django management commands
│   │   └── 📁 commands/
│   │       ├── train_model.py      # Train ML model command
│   │       └── insert_first_data.py
│   ├── 📁 scripts/                 # Utility scripts
│   │   ├── Crawl_data_by_API.py
│   │   ├── Crawl_data_from_Vrain_by_API.py
│   │   ├── Crawl_data_from_Vrain_by_Selenium.py
│   │   ├── Crawl_data_from_html_of_Vrain.py
│   │   ├── Merge_xlsx.py           # Data merging
│   │   ├── Cleardata.py            # Data cleaning
│   │   ├── clean_datetime_nan.py   # DateTime cleanup
│   │   ├── run_diagnostics.py      # Health check
│   │   ├── Email_validator.py      # Email validation
│   │   ├── Login_services.py       # Auth services
│   │   └── email_templates.py      # Email templates
│   ├── 📁 static/weather/          # Static assets
│   │   ├── 📁 css/                 # Stylesheets
│   │   │   ├── Home.css            # Homepage styles (4500+ lines)
│   │   │   ├── Datasets.css        # Dataset page styles
│   │   │   ├── Auth.css            # Authentication styles
│   │   │   └── ...
│   │   ├── 📁 js/                  # JavaScript modules
│   │   │   ├── Home.js             # Homepage interactions
│   │   │   ├── Datasets.js         # Dataset management
│   │   │   └── ...
│   │   ├── 📁 theme/               # Design system
│   │   │   ├── color.css           # Color variables
│   │   │   ├── effect.css          # Visual effects
│   │   │   ├── animation.css       # Keyframe animations
│   │   │   └── typography.css      # Font system
│   │   └── 📁 img/                 # Images and icons
│   ├── 📁 templates/weather/       # Django templates
│   │   ├── Home.html               # Homepage
│   │   ├── Datasets.html           # Dataset browser
│   │   ├── Dataset_preview.html    # File preview
│   │   ├── HTML_Train.html         # Model training page
│   │   ├── HTML_Predict.html       # Weather forecast page
│   │   ├── 📁 auth/                # Authentication templates
│   │   │   ├── Login.html
│   │   │   ├── Register.html
│   │   │   ├── Verify_email_register.html
│   │   │   ├── Forgot_password.html
│   │   │   ├── Verify_otp.html
│   │   │   ├── Reset_password_otp.html
│   │   │   └── Profile.html
│   │   └── Sidebar_nav.html        # Navigation sidebar
│   ├── 📁 views/                   # Django views
│   │   ├── Home.py                 # Homepage view
│   │   ├── View_Datasets.py        # Dataset browser views
│   │   ├── View_Merge_Data.py      # Merge API endpoint
│   │   ├── View_Clear.py           # Clean wizard endpoints
│   │   ├── View_Crawl_*.py         # Crawl endpoints
│   │   ├── View_login.py           # Auth views
│   │   ├── View_Train.py           # Training views
│   │   └── View_Predict.py         # Forecast views
│   ├── urls.py                     # URL routing (47 patterns)
│   ├── db_connection.py            # MongoDB connection
│   └── paths.py                    # Path constants
├── 📁 WeatherForcast/              # Django project settings
│   ├── settings.py                 # Configuration
│   ├── urls.py                     # Root URL conf
│   ├── wsgi.py                     # WSGI entry point
│   └── asgi.py                     # ASGI entry point
├── 📁 data/                        # Data storage (see Data Pipeline section)
├── 📁 config/                      # Project configurations
│   └── train_config.json           # ML training config
├── manage.py                       # Django management script
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables (gitignored)
├── README.md                       # This file
├── CHANGELOG.md                    # Version history and updates
└── debug_top50_errors.csv          # Error diagnostics
```

### 📦 Key Directories Explained

#### `Machine_learning_artifacts/`
Stores trained model artifacts separated by version:
- `latest/` — Current production model
- `old_model/` — Previous version backup

**Never commit** `.pkl` files to git (large binary files). Use `.gitignore`.

#### `Machine_learning_model/`
Complete ML pipeline with modular architecture:
- **data/**: Loading, validation, splitting
- **features/**: Engineering, transformation
- **Models/**: Algorithm implementations
- **trainning/**: Training orchestration
- **interface/**: Prediction API
- **evaluation/**: Metrics and reporting

#### `i18n/`
Internationalization system:
- **locales/**: JSON translation files (vi.json, en.json)
- **middleware.py**: Auto-detect language from URL/session
- **context_processor.py**: Inject `{% t %}` template tag

#### `static/weather/`
Frontend assets with design system:
- **css/**: Organized by page + theme system
- **js/**: Vanilla JavaScript modules (no framework)
- **theme/**: Centralized design tokens (colors, shadows, spacing)

#### `templates/weather/`
Django templates with template tags:
- Server-side rendering for SEO
- `{% load i18n_tags %}` for translations
- `{% load static %}` for asset URLs

---

## 🚀 Installation & Setup

### 📋 Requirements

| Requirement | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.11+ | Core runtime |
| **Django** | 6.0.1 | Web framework |
| **MongoDB** | 7.0+ | User database |
| **pip** | 24+ | Package manager |
| **virtualenv** | 20+ | Environment isolation |

### 🔧 Step-by-Step Installation

#### 1️⃣ Clone Repository
```bash
git clone https://github.com/TiCoder-coder/PROJECT_WEATHER_FORCAST.git
cd PROJECT_WEATHER_FORCAST
```

#### 2️⃣ Create Virtual Environment
```bash
# Create venv
python3 -m venv .venv

# Activate venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

#### 3️⃣ Install Dependencies
```bash
# Install Python packages
pip install -r requirements.txt

# Verify installation
pip list
```

**Key packages:**
```txt
Django==6.0.1
pymongo==4.16.0
pandas==2.3.3
numpy==2.4.1
scikit-learn==1.8.0
xgboost==3.2.0
lightgbm==4.6.0
catboost==1.2.10
selenium==4.39.0
beautifulsoup4==4.14.3
requests==2.32.5
openpyxl==3.1.5
python-dotenv==1.2.1
python-decouple==3.8
resend==2.19.0
```

#### 4️⃣ Configure Environment Variables

Create `.env` file in project root:

```env
# Django Settings
SECRET_KEY=django-insecure-4$t0@wnk+#qu19m66%a90(d10z69tr$-ei@u_pf_%#m5it@=t+
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

# MongoDB Connection
MONGO_URI=mongodb://localhost:27017/Login?directConnection=true
DB_HOST=mongodb+srv://username:<password>@cluster0.mongodb.net/
DB_NAME=Login
DB_USER=username
DB_PASSWORD=your_password
DB_PORT=27017
DB_AUTH_SOURCE=admin
DB_AUTH_MECHANISM=SCRAM-SHA-256

# Authentication
PASSWORD_PEPPER=your_random_pepper_string
JWT_SECRET=your_jwt_secret_key
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_HOURS=3
REFRESH_TOKEN_EXPIRE_DAYS=1
MAX_FAILED_ATTEMPTS=5

# Email Configuration (Gmail SMTP)
EMAIL_BACKEND=django.core.mail.backends.smtp.EmailBackend
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_HOST_USER=your_email@gmail.com
EMAIL_HOST_PASSWORD=your_16_char_app_password
EMAIL_USE_TLS=True
DEFAULT_FROM_EMAIL=VN Weather Hub <your_email@gmail.com>

# OTP Settings
PASSWORD_RESET_OTP_EXPIRE_SECONDS=600
PASSWORD_RESET_OTP_MAX_ATTEMPTS=5

# Resend API (alternative to Gmail)
RESEND_API_KEY=re_your_resend_api_key
RESEND_FROM_EMAIL=onboarding@resend.dev

# Admin Account (first-time setup)
USER_NAME_ADMIN=admin
ADMIN_PASSWORD=Admin@2026
ADMIN_EMAIL=admin@vnweatherhub.com
```

#### 5️⃣ Set Up MongoDB

**Option A: MongoDB Atlas (Cloud)**
1. Create free cluster at [mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas)
2. Get connection string
3. Update `DB_HOST` in `.env`

**Option B: Local MongoDB**
```bash
# Install MongoDB
# Ubuntu
sudo apt install mongodb

# macOS
brew install mongodb-community

# Start MongoDB
sudo systemctl start mongodb  # Linux
brew services start mongodb-community  # macOS

# Verify running
mongosh --eval "db.version()"
```

#### 6️⃣ Initialize Database
```bash
# Run Django migrations (creates SQLite tables)
python manage.py migrate

# Create first admin user (inserts to MongoDB)
python manage.py insert_first_data

# Verify MongoDB collections
mongosh
> use Login
> db.logins.find()
```

#### 7️⃣ Run Development Server
```bash
python manage.py runserver

# Server starts at http://127.0.0.1:8000
```

#### 8️⃣ Access Application
Open browser and navigate to:
- **Homepage**: http://127.0.0.1:8000/
- **Login**: http://127.0.0.1:8000/login/
- **Register**: http://127.0.0.1:8000/register/
- **Datasets**: http://127.0.0.1:8000/datasets/
- **Admin**: http://127.0.0.1:8000/admin/ (use admin credentials)

---

### 🎯 Quick Start Commands

```bash
# Development workflow
python manage.py runserver           # Start dev server
python manage.py check               # Check Django configuration
python manage.py showurls            # List all URL patterns (if installed)

# ML commands
python manage.py train_model         # Train ML model
python Weather_Forcast_App/Machine_learning_model/trainning/train.py \
  --config Weather_Forcast_App/Machine_learning_model/config/train_config.json

# Database commands
python manage.py makemigrations      # Create migration files
python manage.py migrate             # Apply migrations
python manage.py insert_first_data   # Insert admin user

# Utility commands
python manage.py collectstatic       # Collect static files for production
python manage.py createsuperuser     # Create Django admin superuser
```

---

## 📖 User Guide

---

### 🏠 Homepage

URL: `/` hoặc `/<lang>/` (VD: `/vi/`, `/en/`)

Homepage được chia thành **3 sections chính** và nhiều modal tương tác:

#### Section 1 — Hero (Banner)

Hiệu ứng thời tiết động bao gồm mưa, gió, sét, mây và aurora. Gồm:

- **Thời gian cập nhật mới nhất** — hiển thị `latest_dataset_time` bên dưới các nút hành động
- **5 nút hành động nhanh:**

| Nút | Hành động |
|-----|-----------|
| 🌐 **Thu thập dữ liệu** | Mở Crawl Method Modal — chọn 1 trong 4 phương thức crawl |
| 📊 **Datasets** | Chuyển đến trang quản lý datasets (`/datasets/?tab=recent`) |
| 🧠 **Train Model** | Chuyển đến trang huấn luyện model (`/train/`) |
| 🔮 **Dự báo** | Chuyển đến trang dự báo (`/predict/`) |
| ❓ **Hướng dẫn** | Mở Help Modal — hướng dẫn 4 bước |

- **Navigation bar** — góc trên phải:
  - `VI|EN` — chuyển đổi ngôn ngữ không cần reload trang
  - Avatar + tên user (khi đã đăng nhập) → liên kết đến Profile
  - Nút `🚪 Đăng xuất` (khi đã đăng nhập)
  - Nút `🔑 Đăng nhập` + `✨ Đăng ký` (khi chưa đăng nhập)
  - Nút `Giới thiệu` → mở Intro Modal

#### Section 2 — System Overview (4 Feature Cards)

Hiển thị 4 card mô tả các tính năng cốt lõi của hệ thống:

| Card | Biểu tượng | Nội dung |
|------|-----------|---------|
| **Thu thập qua API** | `api.svg` | Crawl từ OpenWeather API, Vrain API |
| **Thu thập qua Vrain** | `vrain.svg` | Crawl bằng Selenium và HTML Parser |
| **Báo cáo & Làm sạch** | `report.svg` | Gộp, kiểm tra, làm sạch dữ liệu |
| **Dự báo ML** | 🧠 | Huấn luyện và dự báo với Ensemble ML |

#### Section 3 — Activity Dashboard (4 stat cards)

Theo dõi hoạt động thực tế của hệ thống theo thời gian thực:

| Card | Nhãn | Dữ liệu hiển thị |
|------|------|-----------------|
| **API Weather** | `API` badge | Số lượng file pipeline API đã chạy |
| **Vrain Pipeline** | `VRAIN` badge | Số lượng file pipeline Vrain đã chạy |
| **ML Model** | `ML` badge | Loại model hiện tại + trạng thái (Ready/Training) |
| **Forecast** | `PREDICT` badge | Mô tả tính năng dự báo |

Nút `🚀 Quick Crawl` — bắt đầu crawl nhanh mà không cần chọn phương thức.

#### Các Modal trên Homepage

**1. Crawl Method Modal** (mở bằng nút 🌐):

| Phương thức | Key | Mô tả |
|-------------|-----|-------|
| **OpenWeather API** | `api_weather` | Crawl từ API thời tiết OpenWeather |
| **Vrain API** | `vrain_api` | Crawl từ API chính thức của Vrain |
| **Vrain Selenium** | `vrain_selenium` | Crawl bằng trình duyệt tự động (Selenium) |
| **Vrain HTML Parser** | `vrain_html` | Phân tích HTML trang Vrain trực tiếp |

Sau khi chọn phương thức → tự động điều hướng đến trang crawl tương ứng.

**2. Help Modal** (mở bằng nút ❓):

4 card hướng dẫn theo luồng sử dụng:
1. **Thu thập dữ liệu** — Nhấn `🌐 Thu thập`, chọn phương thức, điền tham số, chạy
2. **Quản lý Datasets** — Vào `📊 Datasets`, xem / tải / gộp / làm sạch file
3. **Huấn luyện Model** — Vào `🧠 Train`, chọn dataset, cấu hình, nhấn Train
4. **Dự báo thời tiết** — Vào `🔮 Dự báo`, chọn phương thức nhập, chạy và xem kết quả

**3. Intro Modal** (mở bằng nút `Giới thiệu` trên nav):

Giới thiệu tổng quan hệ thống với 5 card:
- **VN Weather Hub là gì** — Mô tả nền tảng
- **Tính năng chính** — Danh sách 5 tính năng: Thu thập / Xử lý / Huấn luyện / Dự báo / Bảo mật
- **Luồng xử lý** — Sơ đồ: `Thu thập → Gộp → Làm sạch → Huấn luyện → Dự báo`
- **Nguồn dữ liệu** — OpenWeather API, Vrain API, Vrain Selenium, Vrain HTML Parser
- **Quickstart** — 5 bước bắt đầu nhanh

---

### 🌐 Data Crawl Pages

#### 1. OpenWeather API (`/crawl-api/`)

Crawl dữ liệu thời tiết từ API OpenWeather:
- Nhập **API key** (có thể đặt trong `.env`)
- Chọn tỉnh/thành phố hoặc nhập tọa độ lat/lon
- Chọn khoảng thời gian (start date → end date)
- Nhấn **Bắt đầu Crawl** → file CSV xuất ra `data/data_crawl/`
- Tên file dạng: `vietnam_weather_data_*.csv`

#### 2. Vrain API (`/crawl-vrain-api/`)

Crawl báo cáo từ hệ thống Vrain qua API:
- Nhập **access token** Vrain
- Chọn tỉnh/thành phố và thời gian
- Nhấn crawl → dữ liệu trả về dạng CSV
- Tên file dạng: `Bao_cao_*.csv`

#### 3. Vrain Selenium (`/crawl-vrain-selenium/`)

Crawl bằng trình duyệt tự động (yêu cầu ChromeDriver):
- Nhập thông tin đăng nhập Vrain (username/password)
- Selenium tự động điều hướng web, xuất báo cáo
- Phù hợp khi API bị giới hạn hoặc captcha
- Yêu cầu: `selenium` + `ChromeDriver` đã cài đặt

#### 4. Vrain HTML Parser (`/crawl-vrain-html/`)

Phân tích HTML trực tiếp từ trang Vrain (không cần đăng nhập):
- Nhập URL trang báo cáo Vrain
- Parser tự động trích xuất bảng dữ liệu
- Nhanh nhất trong 4 phương thức
- **Đây là phương thức được Airflow scheduler sử dụng tự động**
- Tên file dạng: `Bao_cao_<timestamp>.csv`

> **Lưu ý chung:** Tất cả file crawl được lưu vào `data/data_crawl/`. Sau khi crawl xong, vào **Datasets** để gộp và làm sạch dữ liệu.

---

### 📊 Datasets Page

URL: `/datasets/`

#### Tabs (4 tab)

| Tab | Thư mục | Các actions |
|-----|---------|-------------|
| **📦 Raw Data gần nhất** | `data/data_crawl/` | 👁️ Xem, ⬇️ Tải, 🔗 Gộp |
| **🔗 Merged Data** | `data/data_merge/` | 👁️ Xem, ⬇️ Tải, 🧹 Làm sạch |
| **🧹 Cleaned Data** | `data_merge_clean/` + `data_not_merge_clean/` | 👁️ Xem, ⬇️ Tải |
| **🧪 Process Data** | — | Gộp wizard, Làm sạch wizard |

#### Các actions chi tiết

| Action | Biểu tượng | Mô tả |
|--------|-----------|-------|
| **Xem** | 👁️ VIEW | Preview file ngay trên trình duyệt (bảng phân trang, không cần tải) |
| **Tải xuống** | ⬇️ DOWNLOAD | Tải file về máy tính cục bộ |
| **Gộp** | 🔗 MERGE | Kết hợp nhiều file raw thành 1 file merged |
| **Làm sạch** | 🧹 CLEAN | Chạy pipeline làm sạch tự động trên file đã chọn |

#### Tab Process Data

- **Merge Wizard**: Chọn nhiều file → cấu hình → gộp schema → xuất file merged
- **Clean Wizard**: Chọn file merged → cấu hình cleaning → preview → xuất file clean
- Theo dõi tiến độ realtime qua progress bar

---

### 🧠 Train Model Page

URL: `/train/`

#### Giao diện

Trang train gồm 3 phần chính:

**1. Collapsible Guide Card** — Thu gọn/mở rộng được, hướng dẫn 4 bước + lưu ý kỹ thuật.

**2. Artifacts Section** (hiển thị khi đã có model) — Các card metrics chia theo 3 split:
  - `TRAIN` — R², RMSE, MAE, Rain Detection Accuracy (nếu có)
  - `VALIDATION` — R², RMSE, MAE, Rain Detection Accuracy (nếu có)
  - `TEST` — R², RMSE, MAE, Rain Detection Accuracy (nếu có)

**3. Training Form** — Cấu hình và chạy training:

| Trường | Mô tả |
|--------|-------|
| **Dataset** | Chọn file cleaned từ dropdown |
| **Model type** | `ensemble` / `xgboost` / `lightgbm` / `catboost` / `randomforest` |
| **Train ratio** | Tỉ lệ dữ liệu training (mặc định: 0.70) |
| **Valid ratio** | Tỉ lệ validation (mặc định: 0.15) |
| **Test ratio** | Tỉ lệ test (mặc định: 0.15) |

#### Quy trình Training

1. Chọn **cleaned dataset** (tab Cleaned Data trong Datasets)
2. Chọn **loại model** (hoặc dùng `ensemble` cho kết quả ổn định nhất)
3. Thiết lập tỉ lệ `train/valid/test`
4. Nhấn **🧠 Bắt đầu Training**
5. Theo dõi **progress bar** realtime
6. Sau khi xong → xem **metrics report** (TRAIN / VALIDATION / TEST)
7. Artifacts được lưu tự động vào `Machine_learning_artifacts/latest/`

#### Artifacts sinh ra

| File | Nội dung |
|------|---------|
| `Feature_list.json` | Schema 120 features dùng cho prediction |
| `Metrics.json` | Metrics đầy đủ (train/valid/test) + rain accuracy |
| `Train_info.json` | Metadata huấn luyện (model type, dataset, thời gian) |
| `Model.pkl` | Model đã serialized (dùng cho Predict) |
| `Transform_pipeline.pkl` | Pipeline transform features đầu vào |

> **Cấu hình nâng cao:** Chỉnh sửa `config/train_config.json` trước khi train để thay đổi hyperparameters, feature engineering settings.

---

### 🔮 Weather Forecast Page

URL: `/predict/`

#### Giao diện

Trang Predict gồm 3 card chính:

**1. Guide Card** (collapsible) — Hướng dẫn sử dụng các tab.

**2. Model Info Card** — Thông tin model đang được tải:

| Thông tin | Mô tả |
|-----------|-------|
| Test R² | Hệ số xác định trên tập test |
| Test RMSE | Sai số căn bậc hai trung bình |
| Rain Accuracy | Độ chính xác phát hiện mưa |
| Predict Threshold | Ngưỡng phân loại Mưa/Không mưa |
| Overfit Status | Trạng thái overfitting (`ok` / `warn` / `overfit`) |
| Model Size | Dung lượng file `.pkl` (MB) |
| Trained At | Thời điểm huấn luyện |

**3. Prediction Tabs** — 4 phương thức nhập dữ liệu:

| Tab | Icon | Mô tả |
|-----|------|-------|
| **Forecast Now** | ⚡ | Tự động crawl + dự báo ngay lập tức (Airflow / HTML crawl) |
| **From Dataset** | 📦 | Chọn cleaned dataset có sẵn từ hệ thống |
| **Upload File** | 📤 | Tải lên file CSV tự chuẩn bị |
| **Manual Input** | ✏️ | Nhập thủ công từng dòng dữ liệu |

#### Kết quả dự báo

Bảng kết quả hiển thị **toàn bộ dòng dữ liệu** (không giới hạn), gồm các cột:

| Cột quan trọng | Mô tả |
|---------------|-------|
| `datetime` | Thời gian quan trắc |
| `location` | Vị trí (tỉnh/thành) |
| `y_pred` | Lượng mưa dự báo (đơn vị log1p space) |
| `status` | Phân loại: **Mưa** (rain > threshold) / **Không mưa** |
| Các cột feature | Dữ liệu đầu vào gốc kèm theo |

Kết quả được lưu tự động vào: `Weather_Forcast_App/runtime/predictions.csv`

> **Lưu ý kỹ thuật:** Target `rain_total` được áp dụng `log1p` transform trước khi train. Giá trị `y_pred` trong output là giá trị dự báo ở không gian log. Để chuyển về mm thực tế: `expm1(y_pred)`.

---

### 🔐 Authentication

#### Luồng xác thực

```
Đăng ký → OTP Email → Tạo tài khoản
Đăng nhập → JWT (access + refresh token) → access session
Quên mật khẩu → OTP Email → Đặt lại mật khẩu
```

#### Các trang Auth

| Trang | URL | Mô tả |
|-------|-----|-------|
| **Đăng nhập** | `/login/` | Đăng nhập bằng username hoặc email + mật khẩu |
| **Đăng ký** | `/register/` | Điền thông tin, gửi OTP xác minh email |
| **Xác minh OTP đăng ký** | `/verify-email-register/` | Nhập mã OTP 6 số gửi qua email |
| **Quên mật khẩu** | `/forgot-password/` | Nhập email để nhận OTP đặt lại mật khẩu |
| **OTP đặt lại mật khẩu** | `/reset-password-otp/` | Xác minh OTP trước khi đặt mật khẩu mới |
| **Đặt mật khẩu mới** | `/reset-password/` | Nhập mật khẩu mới (có kiểm tra độ mạnh) |
| **Profile** | `/profile/` | Cập nhật thông tin cá nhân, xem lịch sử |

#### Chi tiết bảo mật

| Tính năng | Mô tả |
|-----------|-------|
| **JWT HS256** | Access token (hết hạn sau `ACCESS_TOKEN_EXPIRE_HOURS=3`) |
| **Refresh Token** | Tự động gia hạn session (hết hạn sau `REFRESH_TOKEN_EXPIRE_DAYS=1`) |
| **OTP Email** | Mã 6 số gửi qua email SMTP, có thời hạn |
| **Khóa tài khoản** | Tự động khóa sau 5 lần đăng nhập sai (mở lại sau 5 phút) |
| **Kiểm tra email** | Xác minh MX record + chặn email disposable |
| **Mật khẩu** | Pepper + hash, yêu cầu độ mạnh tối thiểu |
| **Email validation** | Kiểm tra MX record, chặn email tạm thời (disposable) |

#### Lưu ý cho dev

- Session được lưu trong **Django SQLite session table** (backend mặc định)
- JWT payload chứa: `user_id`, `email`, `role`, `exp`
- `LoginRequiredMiddleware` tự động chặn các route cần đăng nhập và redirect về `/login/`
- `LangMiddleware` đọc `lang` từ URL path (`/vi/` hoặc `/en/`) và set `request.lang`

---

## 🔧 Configuration

### ⚙️ Django Settings

**File:** `WeatherForcast/settings.py`

#### Key Configurations

```python
# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'Weather_Forcast_App',  # Main app
]

# Middleware
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'Weather_Forcast_App.i18n.middleware.LangMiddleware',       # i18n: sets request.lang
    'Weather_Forcast_App.middleware.LoginRequired.LoginRequiredMiddleware',  # route protection
]

# Database: SQLite cho Django ORM (admin, sessions, migrations)
# MongoDB được kết nối trực tiếp qua pymongo (db_connection.py)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'
STATICFILES_DIRS = [
    BASE_DIR / 'Weather_Forcast_App' / 'static',
]
STATIC_ROOT = BASE_DIR / 'staticfiles'

# Email configuration
EMAIL_BACKEND = os.getenv('EMAIL_BACKEND')
EMAIL_HOST = os.getenv('EMAIL_HOST')
EMAIL_PORT = int(os.getenv('EMAIL_PORT', 587))
EMAIL_HOST_USER = os.getenv('EMAIL_HOST_USER')
EMAIL_HOST_PASSWORD = os.getenv('EMAIL_HOST_PASSWORD')
EMAIL_USE_TLS = os.getenv('EMAIL_USE_TLS', 'True') == 'True'
DEFAULT_FROM_EMAIL = os.getenv('DEFAULT_FROM_EMAIL')

# Security
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SECURE = False  # Set True in production with HTTPS
CSRF_COOKIE_HTTPONLY = True
CSRF_COOKIE_SECURE = False  # Set True in production
```

### 🔐 Security Settings

#### Production Checklist
```python
# settings.py (production overrides)
DEBUG = False
ALLOWED_HOSTS = ['yourdomain.com', 'www.yourdomain.com']
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_HSTS_SECONDS = 31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'DENY'
```

### 📊 ML Training Configuration

**File:** `config/train_config.json`

```json
{
  "data": {
    "filename": "merged_vrain_data_cleaned_20260307_223506.clean_final.csv",
    "target_column": "rain_total",
    "date_column": "datetime",
    "features": ["temperature", "humidity", "pressure", "wind_speed", "..."]
  },
  "model": {
    "type": "ensemble",
    "params": {
      "n_estimators": 100,
      "max_depth": 10,
      "learning_rate": 0.1
    }
  },
  "split": {
    "train_ratio": 0.8,
    "valid_ratio": 0.1,
    "test_ratio": 0.1,
    "shuffle": false,
    "random_state": 42
  },
  "feature_engineering": {
    "lag_features": [1, 2, 3, 6, 12, 24],
    "rolling_windows": [3, 6, 12, 24],
    "time_features": ["hour", "day", "month", "season"],
    "location_features": ["lat", "lon", "elevation"]
  },
  "evaluation": {
    "metrics": ["mae", "rmse", "mape", "r2", "rain_accuracy"],
    "save_predictions": true,
    "save_feature_importance": true
  }
}
```

---

## 🐛 Troubleshooting

### ❌ Common Errors

#### 1. `UnicodeEncodeError` on Windows
**Error:**
```
UnicodeEncodeError: 'charmap' codec can't encode character '\u2500'
```

**Solution:**
Set console encoding to UTF-8:
```python
# Add to top of script
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
```

Or use ASCII characters instead of Unicode:
```python
# Before
print("──────")  # Box-drawing characters
# After
print("------")  # ASCII hyphens
```

#### 2. MongoDB Connection Failed
**Error:**
```
pymongo.errors.ServerSelectionTimeoutError: localhost:27017: [Errno 111] Connection refused
```

**Solutions:**
```bash
# Check if MongoDB is running
sudo systemctl status mongodb  # Linux
brew services list             # macOS

# Start MongoDB
sudo systemctl start mongodb   # Linux
brew services start mongodb-community  # macOS

# Verify connection string in .env
MONGO_URI=mongodb://localhost:27017/Login?directConnection=true
```

#### 3. Email OTP Not Sending
**Error:**
```
SMTPAuthenticationError: (535, b'5.7.8 Username and Password not accepted')
```

**Solutions:**
1. **Check Gmail App Password:**
   - Go to Google Account → Security → 2-Step Verification → App passwords
   - Generate new app password
   - Update `EMAIL_HOST_PASSWORD` in `.env`

2. **Verify settings:**
   ```env
   EMAIL_HOST=smtp.gmail.com
   EMAIL_PORT=587
   EMAIL_USE_TLS=True
   EMAIL_HOST_USER=your_email@gmail.com
   EMAIL_HOST_PASSWORD=16_char_app_password
   ```

3. **Use console backend for testing:**
   ```env
   EMAIL_BACKEND=django.core.mail.backends.console.EmailBackend
   ```

#### 4. 404 on Static Files
**Error:**
```
Not Found: /static/weather/css/Home.css
```

**Solutions:**
```bash
# Collect static files
python manage.py collectstatic

# Verify STATIC_URL in settings.py
STATIC_URL = '/static/'

# Check file exists
ls Weather_Forcast_App/static/weather/css/Home.css
```

#### 5. Import Error: `ModuleNotFoundError`
**Error:**
```
ModuleNotFoundError: No module named 'xgboost'
```

**Solution:**
```bash
# Activate virtual environment
source .venv/bin/activate

# Install missing package
pip install xgboost

# Or reinstall all requirements
pip install -r requirements.txt
```

#### 6. Model Training Fails
**Error:**
```
KeyError: 'rain_total'
```

**Solutions:**
1. **Check dataset has target column:**
   ```python
   import pandas as pd
   df = pd.read_csv('data.csv')
   print(df.columns)
   ```

2. **Update `train_config.json`:**
   ```json
   {
     "data": {
       "target_column": "rain_total"  // Must match column name exactly
     }
   }
   ```

3. **Verify cleaned data:**
   - Open cleaned CSV file
   - Confirm column exists and has data

#### 7. JWT Token Expires Too Quickly
**Problem:**
User gets logged out after a few minutes.

**Solution:**
Increase token TTL in `.env`:
```env
# Access token (default: 3 hours)
ACCESS_TOKEN_EXPIRE_HOURS=8

# Refresh token (default: 1 day)
REFRESH_TOKEN_EXPIRE_DAYS=7
```

#### 8. Language Not Switching
**Problem:**
Clicking language button doesn't change UI text.

**Solutions:**
1. **Check translations exist:**
   ```bash
   cat Weather_Forcast_App/i18n/locales/en.json
   cat Weather_Forcast_App/i18n/locales/vi.json
   ```

2. **Verify template tag:**
   ```html
   {% load i18n_tags %}
   {% t "home.hero_title" %}
   ```

3. **Clear browser cache:**
   ```
   Ctrl+Shift+Delete (Chrome/Firefox)
   ```

4. **Restart Django server:**
   ```bash
   python manage.py runserver
   ```

#### 9. `ModuleNotFoundError: selenium` trong Airflow
**Error:**
```
ModuleNotFoundError: No module named 'selenium'
```

**Solution:**
Image Airflow chưa được rebuild sau khi thêm dependency. Rebuild lại:
```bash
docker compose build airflow-scheduler
docker compose up -d
```

#### 10. Airflow DAG không chạy đúng giờ
**Problem:**
DAG trigger vào giờ không mong muốn.

**Solutions:**
1. Kiểm tra cron expression và timezone (`Asia/Bangkok`) trong file DAG
2. Mở **Airflow UI** → chọn DAG → xem mục **Schedule** để xác nhận
3. Đảm bảo container scheduler đang chạy: `docker compose ps`

#### 11. `NameError: VN_TZ is not defined` trong Airflow task
**Problem:**
Script HTML crawler chưa được đồng bộ phiên bản mới.

**Solution:**
```bash
docker compose restart airflow-scheduler
```

---

## 🗺️ Roadmap

### 🔜 Short-term (Q1-Q2 2026)

- [ ] 📈 **Enhanced Dashboard**
  - Interactive charts with Chart.js/D3.js
  - Real-time metrics updates via WebSocket
  - Export dashboard as PDF report

- [ ] 🔐 **Advanced Authentication**
  - OAuth2 integration (Google, Facebook, GitHub)
  - Two-factor authentication (2FA)
  - Role-based access control (RBAC)
  - Activity logging and audit trail

- [ ] ✅ **Robust Data Validation**
  - JSON Schema validation for all datasets
  - Auto-fix common data issues
  - Data quality score dashboard
  - Anomaly detection in raw data

- [ ] 🧠 **ML Model Improvements**
  - AutoML with Optuna hyperparameter tuning
  - Model explainability with SHAP values
  - A/B testing framework for models
  - Online learning capabilities

### 🚀 Medium-term (Q3-Q4 2026)

- [ ] 🌐 **RESTful API**
  - Complete REST API for all features
  - API authentication with API keys
  - Rate limiting and quota management
  - Interactive Swagger/OpenAPI documentation

- [ ] 🐳 **Containerization & Deployment**
  - Docker Compose setup for development
  - Kubernetes manifests for production
  - CI/CD pipeline with GitHub Actions
  - Automated testing suite

- [ ] 💾 **Data Storage Enhancements**
  - S3/MinIO for large file storage
  - Redis caching for frequently accessed data
  - PostgreSQL for relational data
  - Automated backup and restore

- [ ] 📱 **Mobile App**
  - React Native mobile application
  - Push notifications for forecast alerts
  - Offline mode with data sync
  - Mobile-optimized UI

### 🌟 Long-term (2027+)

- [ ] 🧪 **Advanced ML Pipelines**
  - Deep learning models (LSTM, Transformer)
  - Ensemble stacking with meta-learner
  - Transfer learning from global weather models
  - Multi-task learning (temperature + rain + wind)

- [ ] 🌍 **Geographic Expansion**
  - Support for multiple countries/regions
  - Multi-language support (10+ languages)
  - Timezone handling for global users
  - Regional weather pattern analysis

- [ ] 🔬 **Research Features**
  - Climate change trend analysis
  - Extreme weather event prediction
  - Integration with satellite data
  - Collaboration with meteorological agencies

- [ ] 🎨 **UI/UX Enhancements**
  - Dark mode with system preference detection
  - Customizable dashboard layouts
  - Accessibility compliance (WCAG 2.1 AA)
  - Interactive data visualization playground

---

## 👥 Team

### 🧑‍💻 Maintainers & Contributors

<table>
  <tr>
    <td align="center">
      <b>Võ Anh Nhật</b><br>
      <i>Team Lead</i><br>
      📧 voanhnhat1612@gmail.com<br>
      📞 +84 335 052 899
    </td>
    <td align="center">
      <b>Võ Huỳnh Anh Tuấn</b><br>
      <i>Full-stack Developer</i><br>
      📧 vohuynhanhtuan0512@gmail.com
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>Trương Hoài Tú</b><br>
      <i>Frontend Developer</i><br>
      📧 hoaitu163@gmail.com
    </td>
    <td align="center">
      <b>Dư Quốc Việt</b><br>
      <i>Data Engineer</i><br>
      📧 duviet720@gmail.com
    </td>
  </tr>
</table>

### 🎓 Institution
**Ho Chi Minh city University of Transport (UTH)**  
*Computer Science Department*

### 📅 Project Timeline
- **Started:** January 2026
- **Last Updated:** March 2026
- **Status:** Active Development

---

## 📜 License & Acknowledgments

### 📄 License
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

### 🙏 Acknowledgments

#### Data Sources
- **OpenWeather** — [openweathermap.org](https://openweathermap.org) — Free weather API
- **Vrain** — Vietnamese weather data provider

#### Technologies & Frameworks
- **Django** — Web framework by Django Software Foundation
- **MongoDB** — NoSQL database by MongoDB Inc.
- **Scikit-learn** — Machine learning library
- **XGBoost** — Gradient boosting framework by DMLC
- **LightGBM** — Fast gradient boosting by Microsoft
- **CatBoost** — Gradient boosting by Yandex
- **Pandas** — Data manipulation library
- **NumPy** — Numerical computing library

#### Design Inspiration
- **Glassmorphism** — Design trend by Michal Malewicz
- **Meteorological UI** — Inspired by Weather.com and Apple Weather

---

<div align="center">

### 🌦️ VN Weather Hub

**Making Weather Data Accessible Through Technology**

---

<sub>
Made with ☕ + ⛈️ by VN Weather Hub Team  
© 2026 Ho Chi Minh city University of Transport (UTH)
</sub>

<br/>

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-6.0.1-092E20?logo=django&logoColor=white)](https://www.djangoproject.com/)
[![MongoDB](https://img.shields.io/badge/MongoDB-7.0+-47A248?logo=mongodb&logoColor=white)](https://www.mongodb.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

</div>
