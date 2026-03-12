# ⚙️ management — Django Management Commands

## 📁 Overview

This directory contains **custom Django management commands** — CLI tools that extend `python manage.py` functionality.

---

## 📂 Directory Structure

```
management/
├── __init__.py          # Required for Django to recognize this as commands package
├── commands/            # All custom commands go here
│   ├── __init__.py
│   ├── train_model.py      # Train ML model command
│   ├── insert_first_data.py  # Seed admin account command
│   └── README.md
└── README.md            # This file
```

---

## 🎯 Purpose

### What are Management Commands?

Management commands are **Python scripts** that run via `python manage.py <command>`. They provide:
- **CLI access** to application functionality
- **Automation** for repetitive tasks (seeding data, training models)
- **Cron job integration** (schedule commands with cron/systemd timers)
- **DevOps tools** (deployment scripts, database migrations)

### Built-in vs Custom Commands

| Type | Examples | Location |
|------|----------|----------|
| **Built-in** | `runserver`, `migrate`, `makemigrations` | Django core |
| **Custom** | `train_model`, `insert_first_data` | `management/commands/` |

---

## 📄 Files Explained

### `__init__.py`

**Purpose**: Makes `management/` a Python package. **Must exist** for Django to recognize custom commands.

---

### `commands/` Directory

**Purpose**: Container for all custom management commands.

**Naming Convention**:
- File name = command name
- `train_model.py` → `python manage.py train_model`
- `insert_first_data.py` → `python manage.py insert_first_data`

---

## 🔧 How Management Commands Work

### Step 1: Create Command File

```python
# management/commands/my_command.py
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = "Description shown in --help"
    
    def add_arguments(self, parser):
        # Add optional/required arguments
        parser.add_argument('--option', type=str, help='Some option')
    
    def handle(self, *args, **options):
        # Your command logic here
        self.stdout.write(self.style.SUCCESS("Command executed!"))
```

### Step 2: Run Command

```bash
python manage.py my_command --option value
```

### Step 3: View Help

```bash
python manage.py my_command --help
```

---

## 💡 Usage Examples

### Example 1: List All Commands

```bash
python manage.py help
```

**Output**:
```
Type 'manage.py help <subcommand>' for help on a specific subcommand.

Available subcommands:

[Weather_Forcast_App]
    insert_first_data
    train_model

[auth]
    changepassword
    createsuperuser

[django]
    check
    migrate
    runserver
    ...
```

### Example 2: Get Command Help

```bash
python manage.py train_model --help
```

**Output**:
```
usage: manage.py train_model [-h] [--config CONFIG] [--dry-run]

Train the weather forecast ML model using train_config.json

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Path to train config JSON/YAML. Default: ...
  --dry-run             Load config và validate nhưng không chạy training thật.
```

### Example 3: Run Command

```bash
# Train model with default config
python manage.py train_model

# Train with custom config
python manage.py train_model --config /path/to/custom_config.json

# Dry run (validate without training)
python manage.py train_model --dry-run
```

---

## 📊 Available Commands

| Command | Purpose | Arguments |
|---------|---------|-----------|
| **train_model** | Train ML weather forecast model | `--config`, `--dry-run` |
| **insert_first_data** | Seed admin account into MongoDB | None |

*For detailed documentation, see [commands/README.md](commands/README.md)*

---

## 🚀 Creating a New Command

### Template

```python
# management/commands/cleanup_logs.py
from django.core.management.base import BaseCommand
from pathlib import Path

class Command(BaseCommand):
    help = "Delete old log files"
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--days',
            type=int,
            default=30,
            help='Delete logs older than N days'
        )
    
    def handle(self, *args, **options):
        days = options['days']
        
        # Your logic
        log_dir = Path("logs/")
        deleted = 0
        
        for log_file in log_dir.glob("*.log"):
            # Check file age and delete if needed
            # ...
            deleted += 1
        
        self.stdout.write(
            self.style.SUCCESS(f"Deleted {deleted} log files")
        )
```

### Run Your Command

```bash
python manage.py cleanup_logs --days 7
```

---

## 🐛 Common Issues

### Issue 1: Command Not Found

**Error**: `Unknown command: 'my_command'`

**Causes**:
1. Missing `__init__.py` in `management/` or `commands/`
2. File name doesn't match command name
3. Command file not in `commands/` directory

**Solution**:
```bash
# Check directory structure
ls management/
# Should see: __init__.py  commands/

ls management/commands/
# Should see: __init__.py  my_command.py  ...
```

### Issue 2: Import Errors in Command

**Error**: `ModuleNotFoundError: No module named 'Weather_Forcast_App'`

**Cause**: Running command from wrong directory

**Solution**: Always run from project root (where `manage.py` is):
```bash
# ❌ WRONG
cd Weather_Forcast_App/management/commands
python train_model.py

# ✅ CORRECT
cd /path/to/PROJECT_WEATHER_FORECAST
python manage.py train_model
```

### Issue 3: Command Runs But Does Nothing

**Cause**: `handle()` method not defined or not returning anything

**Solution**: Check `handle()` exists and executes your logic:
```python
def handle(self, *args, **options):
    # ❌ WRONG (empty — does nothing)
    pass
    
    # ✅ CORRECT
    self.stdout.write("Running command...")
    # Your actual logic here
    self.stdout.write(self.style.SUCCESS("Done!"))
```

---

## 🔍 Best Practices

### ✅ DO

- **Add help text** — users should understand command purpose via `--help`
- **Use self.stdout.write()** — proper Django logging for command output
- **Add arguments** — make commands flexible with `add_arguments()`
- **Handle errors gracefully** — wrap in try/except, return meaningful messages

### ❌ DON'T

- **Don't use print()** — use `self.stdout.write()` instead
- **Don't hardcode paths** — use settings or arguments
- **Don't run heavy tasks synchronously** — consider Celery for long-running jobs
- **Don't skip validation** — validate arguments before executing

---

## 🚀 Future Enhancements

- [ ] Add **backup_database** command (dump MongoDB to JSON)
- [ ] Add **clear_expired_sessions** command
- [ ] Add **send_daily_report** command (email weather summary)
- [ ] Add **optimize_database** command (reindex MongoDB)
- [ ] Add **sync_weather_data** command (fetch latest from API)

---

## 📞 Related Files

- **Commands**: `management/commands/README.md` (detailed command docs)
- **Settings**: `WeatherForcast/settings.py` (Django configuration)
- **Models**: `Models/Login.py` (used in `insert_first_data`)
- **ML**: `Machine_learning_model/trainning/train.py` (used in `train_model`)

---

## 👨‍💻 Maintainer

**Võ Anh Nhật**  
📧 voanhnhat1612@gmail.com

---

*Last Updated: March 8, 2026*
