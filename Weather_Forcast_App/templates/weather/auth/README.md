# 🔐 templates/weather/auth — Authentication Templates

## 📁 Overview

This directory contains **Django HTML templates** for **authentication pages** (login, register, logout, password reset).

---

## 📂 Directory Structure

```
templates/weather/auth/
├── login.html               # Login page
├── register.html            # Registration page
├── logout.html              # Logout confirmation page
├── forgot_password.html     # Password reset request
├── reset_password.html      # Password reset form
└── README.md                # This file
```

---

## 🎯 Purpose

### What are Authentication Templates?

Templates that handle **user authentication flows**:
- **Login**: Existing users sign in
- **Register**: New users create accounts
- **Logout**: Users end sessions
- **Password Reset**: Forgot password recovery

---

## 📄 Files Explained

### `login.html` — Login Page

#### Purpose

Display **login form** for existing users.

#### Features

- Username/email input
- Password input (hidden)
- "Remember me" checkbox
- Forgot password link
- Error message display
- Language switcher

#### Example Code

```django
{% extends "weather/base.html" %}
{% load static %}
{% load i18n_tags %}

{% block title %}{% t "auth.login.title" %}{% endblock %}

{% block content %}
<div class="auth-container">
  <div class="auth-card glass">
    <h1 class="auth-title">{% t "auth.login.title" %}</h1>
    
    {% if error %}
      <div class="auth-error">
        {% t error_key %}
      </div>
    {% endif %}
    
    <form method="post" action="{% url 'login' %}" class="auth-form">
      {% csrf_token %}
      
      <div class="form-group">
        <label for="username">{% t "auth.login.username" %}</label>
        <input 
          type="text" 
          id="username" 
          name="userName" 
          class="auth-input"
          placeholder="{% t 'auth.login.username_placeholder' %}"
          required
        >
      </div>
      
      <div class="form-group">
        <label for="password">{% t "auth.login.password" %}</label>
        <input 
          type="password" 
          id="password" 
          name="password" 
          class="auth-input"
          placeholder="{% t 'auth.login.password_placeholder' %}"
          required
        >
      </div>
      
      <div class="form-options flex justify-between">
        <label class="checkbox">
          <input type="checkbox" name="remember_me">
          <span>{% t "auth.login.remember_me" %}</span>
        </label>
        
        <a href="{% url 'forgot_password' %}" class="forgot-link">
          {% t "auth.login.forgot_password" %}
        </a>
      </div>
      
      <button type="submit" class="auth-button gradient-primary">
        {% t "auth.login.submit" %}
      </button>
    </form>
    
    <div class="auth-footer">
      <p>{% t "auth.login.no_account" %} 
        <a href="{% url 'register' %}">{% t "auth.login.register_link" %}</a>
      </p>
    </div>
  </div>
</div>
{% endblock %}
```

#### Associated Files

- **CSS**: `static/weather/css/Auth.css`
- **View**: `views/View_login.py` (login logic)
- **URL**: `urls.py` (route: `/auth/login/`)

---

### `register.html` — Registration Page

#### Purpose

Display **registration form** for new users.

#### Features

- Full name input
- Username input (unique)
- Email input (unique)
- Password input (with strength indicator)
- Password confirmation
- Terms of service agreement
- Error/success messages

#### Example Code

```django
{% extends "weather/base.html" %}
{% load static %}
{% load i18n_tags %}

{% block title %}{% t "auth.register.title" %}{% endblock %}

{% block content %}
<div class="auth-container">
  <div class="auth-card glass">
    <h1 class="auth-title">{% t "auth.register.title" %}</h1>
    
    <form method="post" action="{% url 'register' %}" class="auth-form">
      {% csrf_token %}
      
      <div class="form-group">
        <label for="name">{% t "auth.register.full_name" %}</label>
        <input 
          type="text" 
          id="name" 
          name="name" 
          class="auth-input"
          placeholder="{% t 'auth.register.full_name_placeholder' %}"
          required
        >
      </div>
      
      <div class="form-group">
        <label for="username">{% t "auth.register.username" %}</label>
        <input 
          type="text" 
          id="username" 
          name="userName" 
          class="auth-input"
          placeholder="{% t 'auth.register.username_placeholder' %}"
          required
        >
      </div>
      
      <div class="form-group">
        <label for="email">{% t "auth.register.email" %}</label>
        <input 
          type="email" 
          id="email" 
          name="email" 
          class="auth-input"
          placeholder="{% t 'auth.register.email_placeholder' %}"
          required
        >
      </div>
      
      <div class="form-group">
        <label for="password">{% t "auth.register.password" %}</label>
        <input 
          type="password" 
          id="password" 
          name="password" 
          class="auth-input"
          placeholder="{% t 'auth.register.password_placeholder' %}"
          required
        >
        <div class="password-strength" id="password-strength"></div>
      </div>
      
      <div class="form-group">
        <label for="confirm_password">{% t "auth.register.confirm_password" %}</label>
        <input 
          type="password" 
          id="confirm_password" 
          name="confirm_password" 
          class="auth-input"
          placeholder="{% t 'auth.register.confirm_password_placeholder' %}"
          required
        >
      </div>
      
      <label class="checkbox">
        <input type="checkbox" name="terms" required>
        <span>{% t "auth.register.terms_agree" %} 
          <a href="{% url 'terms' %}">{% t "auth.register.terms_link" %}</a>
        </span>
      </label>
      
      <button type="submit" class="auth-button gradient-primary">
        {% t "auth.register.submit" %}
      </button>
    </form>
    
    <div class="auth-footer">
      <p>{% t "auth.register.have_account" %} 
        <a href="{% url 'login' %}">{% t "auth.register.login_link" %}</a>
      </p>
    </div>
  </div>
</div>

<script>
// Password strength indicator
document.getElementById('password').addEventListener('input', (e) => {
  const password = e.target.value;
  const strength = calculatePasswordStrength(password);
  const indicator = document.getElementById('password-strength');
  
  indicator.className = `password-strength strength-${strength.level}`;
  indicator.textContent = strength.text;
});

function calculatePasswordStrength(password) {
  let score = 0;
  if (password.length >= 8) score++;
  if (/[a-z]/.test(password)) score++;
  if (/[A-Z]/.test(password)) score++;
  if (/[0-9]/.test(password)) score++;
  if (/[^a-zA-Z0-9]/.test(password)) score++;
  
  if (score <= 2) return { level: 'weak', text: 'Weak' };
  if (score === 3) return { level: 'medium', text: 'Medium' };
  return { level: 'strong', text: 'Strong' };
}
</script>
{% endblock %}
```

---

### `logout.html` — Logout Confirmation

#### Purpose

Confirm user logout and display success message.

#### Example Code

```django
{% extends "weather/base.html" %}
{% load i18n_tags %}

{% block title %}{% t "auth.logout.title" %}{% endblock %}

{% block content %}
<div class="auth-container">
  <div class="auth-card glass">
    <h1 class="auth-title">{% t "auth.logout.title" %}</h1>
    
    <div class="auth-message success">
      <p>{% t "auth.logout.success_message" %}</p>
    </div>
    
    <a href="{% url 'login' %}" class="auth-button gradient-primary">
      {% t "auth.logout.login_again" %}
    </a>
    
    <a href="{% url 'home' %}" class="auth-link">
      {% t "auth.logout.go_home" %}
    </a>
  </div>
</div>
{% endblock %}
```

---

### `forgot_password.html` — Password Reset Request

#### Purpose

Allow users to request password reset via email.

#### Example Code

```django
{% extends "weather/base.html" %}
{% load i18n_tags %}

{% block title %}{% t "auth.forgot_password.title" %}{% endblock %}

{% block content %}
<div class="auth-container">
  <div class="auth-card glass">
    <h1 class="auth-title">{% t "auth.forgot_password.title" %}</h1>
    
    <p class="auth-description">
      {% t "auth.forgot_password.description" %}
    </p>
    
    <form method="post" action="{% url 'forgot_password' %}" class="auth-form">
      {% csrf_token %}
      
      <div class="form-group">
        <label for="email">{% t "auth.forgot_password.email" %}</label>
        <input 
          type="email" 
          id="email" 
          name="email" 
          class="auth-input"
          placeholder="{% t 'auth.forgot_password.email_placeholder' %}"
          required
        >
      </div>
      
      <button type="submit" class="auth-button gradient-primary">
        {% t "auth.forgot_password.submit" %}
      </button>
    </form>
    
    <a href="{% url 'login' %}" class="auth-link">
      {% t "auth.forgot_password.back_to_login" %}
    </a>
  </div>
</div>
{% endblock %}
```

---

## 🎨 Design Features

### Glassmorphism Effect

All auth templates use **glassmorphism**:

```css
.auth-card {
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 16px;
}
```

### Gradient Buttons

```css
.auth-button {
  background: linear-gradient(135deg, #3b82f6, #2563eb);
  color: white;
  padding: 12px 24px;
  border-radius: 8px;
  transition: transform 0.3s;
}

.auth-button:hover {
  transform: translateY(-2px);
}
```

---

## 🔍 Best Practices

### ✅ DO

- **Use CSRF tokens** in all POST forms
- **Translate all text** using `{% t %}` tags
- **Validate client-side** (prevent unnecessary server requests)
- **Show clear errors** (field-specific error messages)
- **Implement CAPTCHA** (prevent bot registrations)

### ❌ DON'T

- **Don't hardcode text** (use translation keys)
- **Don't expose sensitive errors** ("Invalid username or password" not "Username doesn't exist")
- **Don't skip CSRF protection**
- **Don't store passwords plainly** (always hash)

---

## 🚀 Future Enhancements

- [ ] **OAuth integration** (Google, Facebook login)
- [ ] **Two-factor authentication** (2FA via OTP)
- [ ] **Email verification** (verify email before activation)
- [ ] **Password strength meter** (visual indicator)
- [ ] **Rate limiting** (prevent brute force attacks)

---

## 📞 Related Files

- **CSS**: `static/weather/css/Auth.css`
- **Views**: `views/View_login.py`
- **Serializers**: `Seriallizer/Login/`
- **i18n**: `locales/vi.json`, `locales/en.json`

---

## 👨‍💻 Maintainer

**Võ Anh Nhật**  
📧 voanhnhat1612@gmail.com

---

*Last Updated: March 8, 2026*
