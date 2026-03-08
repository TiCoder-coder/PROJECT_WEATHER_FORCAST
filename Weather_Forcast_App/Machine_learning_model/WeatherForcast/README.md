# 🔮 WeatherForcast/ — Prediction Runner Script

## 📁 Overview

Module `WeatherForcast/` chứa **script chạy dự đoán (prediction runner)** sử dụng artifacts đã train. Đây là **command-line tool** để:
- Load model, pipeline, features từ `Machine_learning_artifacts/latest/`
- Nhận dữ liệu mới từ CSV hoặc database
- Thực hiện predictions batch
- Lưu kết quả vào file `predictions.csv`

Module này là **production-ready prediction pipeline**, tách biệt khỏi Django web app, có thể chạy qua:
- **Command line**: `python WeatherForcast.py --input data.csv --output predictions.csv`
- **Cron jobs**: Scheduled daily/hourly forecasts
- **Docker containers**: Isolated prediction service


## 📂 Directory Structure

```
WeatherForcast/
├── WeatherForcast.py              # 🔮 Main prediction runner script
└── predictions.csv                # 📄 Output file (generated after running)
```


## 🎯 Purpose

### ❓ Tại sao cần folder này?

| Vấn đề | Giải pháo |
|--------|-----------|
| **Cần chạy predictions từ command line** | `python WeatherForcast.py` thay vì phải viết Python script riêng |
| **Muốn schedule predictions hàng ngày** | Dùng cron job gọi script này |
| **Tách biệt inference khỏi Django web app** | Prediction runner độc lập, không phụ thuộc Django server |
| **Lưu predictions vào file CSV** | Script tự động export `predictions.csv` |

### ✅ Lợi ích

- **CLI-friendly**: Chạy từ command line với arguments
- **Batch processing**: Process lớn datasets hiệu quả
- **Artifact reuse**: Tự động load artifacts từ latest training run
- **Production-ready**: Sẵn sàng deploy vào production workflow


## 📄 Files Explained

### 1. `WeatherForcast.py` — Main Prediction Runner

**Mục đích**: Script chính để thực hiện batch predictions với trained model.

#### 🏗️ Architecture

```
WeatherForcast.py Flow:
1. Parse command-line arguments (--input, --output, --config)
2. Load training artifacts từ Machine_learning_artifacts/latest/
   - Model.pkl
   - Transform_pipeline.pkl
   - Feature_list.json
   - Train_info.json
3. Load input data từ CSV hoặc database
4. Drop known bad rows (từ debug_top50_errors.csv nếu có)
5. Build features (WeatherFeatureBuilder)
6. Transform data (WeatherTransformPipeline)
7. Predict
8. Save results to predictions.csv
```

#### 🔧 Command-Line Arguments

| Argument | Required? | Default | Mô tả |
|----------|-----------|---------|-------|
| `--input` | ⚠️ Tùy chọn | Database query | Path to input CSV file |
| `--output` | ⚠️ Tùy chọn | `predictions.csv` | Path to output file |
| `--artifacts-dir` | ⚠️ Tùy chọn | `Machine_learning_artifacts/latest` | Artifacts folder path |

#### 🔧 Usage Examples

**Example 1: Predict from CSV file**

```bash
# Chạy prediction với input CSV
python WeatherForcast.py --input new_weather_data.csv

# Output: predictions.csv (generated in current directory)
```

**Example 2: Specify custom output path**

```bash
# Save predictions to custom path
python WeatherForcast.py \
    --input new_data.csv \
    --output /path/to/output/forecast_2024_01_15.csv
```

**Example 3: Use old model artifacts**

```bash
# Use artifacts từ old_model folder
python WeatherForcast.py \
    --input data.csv \
    --artifacts-dir Machine_learning_artifacts/old_model
```

**Example 4: Run from Django management command**

```bash
# Nếu có management command wrapper
python manage.py run_forecast --input data.csv
```


#### 📊 Input Data Format

**CSV file requirements**:

```csv
Date,Temperature_C,Humidity_%,Pressure_hPa,Wind_Speed_km_h,Cloud_Cover_%
2024-01-15,25.3,78.5,1013.2,12.5,60
2024-01-16,26.1,75.2,1012.8,15.3,55
2024-01-17,24.8,80.1,1014.5,10.2,70
```

**Required columns**:
- Phải có tất cả features được sử dụng lúc train (xem `Feature_list.json`)
- Column names phải match exactly (case-sensitive)
- Date column (nếu có) nên ở format ISO 8601: `YYYY-MM-DD`

**Optional columns**:
- Target column (Precipitation_mm) không cần có (vì đang predict)
- Extra columns sẽ bị ignore


#### 📊 Output Data Format

**predictions.csv**:

```csv
Date,Temperature_C,Humidity_%,Pressure_hPa,Wind_Speed_km_h,Cloud_Cover_%,Predicted_Precipitation_mm
2024-01-15,25.3,78.5,1013.2,12.5,60,15.3
2024-01-16,26.1,75.2,1012.8,15.3,55,8.7
2024-01-17,24.8,80.1,1014.5,10.2,70,22.1
```

**Output columns**:
- Tất cả input columns (giữ nguyên)
- `Predicted_{target_column}` (VD: `Predicted_Precipitation_mm`)


### 2. `predictions.csv` — Output File

**Mục đích**: File kết quả predictions (generated sau khi chạy script).

**Sử dụng**:
- Import vào Excel/Google Sheets cho visualization
- Load vào database cho storage
- Input cho downstream analytics pipelines
- Validation predictions với ground truth (nếu có)


## 🔧 How to Use

### 1️⃣ Basic Workflow

```bash
# Step 1: Prepare input data
# - CSV file với các columns cần thiết
# - Hoặc query từ database (script tự động)

# Step 2: Run prediction
cd /path/to/PROJECT_WEATHER_FORECAST/Weather_Forcast_App/Machine_learning_model/WeatherForcast
python WeatherForcast.py --input new_data.csv

# Step 3: Check output
cat predictions.csv
```


### 2️⃣ Schedule Daily Forecasts with Cron

**Cron job** để chạy predictions hàng ngày lúc 6:00 AM:

```bash
# Edit crontab
crontab -e

# Add this line
0 6 * * * cd /path/to/PROJECT_WEATHER_FORECAST && python Weather_Forcast_App/Machine_learning_model/WeatherForcast/WeatherForcast.py --input /data/daily_weather.csv --output /forecasts/forecast_$(date +\%Y\%m\%d).csv
```

**Giải thích**:
- `0 6 * * *`: Run daily at 6:00 AM
- `cd /path/to/PROJECT`: Change to project root
- `--output /forecasts/forecast_$(date +\%Y\%m\%d).csv`: Save with date in filename (forecast_20240115.csv)


### 3️⃣ Integration with Django Management Command

**Create management command** để wrap WeatherForcast.py:

```python
# Weather_Forcast_App/management/commands/run_forecast.py

from django.core.management.base import BaseCommand
import subprocess
from pathlib import Path

class Command(BaseCommand):
    help = 'Run weather forecast predictions'

    def add_arguments(self, parser):
        parser.add_argument(
            '--input',
            type=str,
            help='Input CSV file path'
        )
        parser.add_argument(
            '--output',
            type=str,
            default='predictions.csv',
            help='Output CSV file path'
        )

    def handle(self, *args, **options):
        input_file = options['input']
        output_file = options['output']
        
        # Path to WeatherForcast.py
        script_path = Path(__file__).resolve().parents[2] / 'Machine_learning_model' / 'WeatherForcast' / 'WeatherForcast.py'
        
        # Run script
        cmd = ['python', str(script_path)]
        if input_file:
            cmd += ['--input', input_file]
        cmd += ['--output', output_file]
        
        self.stdout.write(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            self.stdout.write(self.style.SUCCESS(f'✅ Predictions saved to {output_file}'))
            self.stdout.write(result.stdout)
        else:
            self.stdout.write(self.style.ERROR('❌ Prediction failed'))
            self.stderr.write(result.stderr)
```

**Usage**:
```bash
python manage.py run_forecast --input data.csv --output forecast.csv
```


### 4️⃣ Docker Container for Predictions

**Dockerfile**:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY Weather_Forcast_App/ /app/Weather_Forcast_App/
COPY WeatherForcast/ /app/WeatherForcast/
COPY Machine_learning_artifacts/ /app/Machine_learning_artifacts/

# Run predictions
CMD ["python", "Weather_Forcast_App/Machine_learning_model/WeatherForcast/WeatherForcast.py"]
```

**Build and run**:
```bash
# Build image
docker build -t weather-forecast .

# Run prediction
docker run -v $(pwd)/data:/data -v $(pwd)/output:/output weather-forecast \
    python Weather_Forcast_App/Machine_learning_model/WeatherForcast/WeatherForcast.py \
    --input /data/new_weather.csv \
    --output /output/predictions.csv
```


## 🛠️ Internal Functions

### `_drop_known_bad_rows()`

**Signature**:
```python
def _drop_known_bad_rows(df: pd.DataFrame) -> pd.DataFrame:
```

**Mục đích**: Lọc bỏ các dòng dữ liệu xấu (bad rows) đã được xác định trong `debug_top50_errors.csv`.

**Process**:
1. Check nếu `ROOT/debug_top50_errors.csv` tồn tại
2. Load bad rows từ CSV
3. So khớp rows trong input DataFrame với bad rows (fingerprint matching)
4. Remove matched rows
5. Return cleaned DataFrame

**Why needed?**: Script `scripts/run_diagnostics.py` tạo ra `debug_top50_errors.csv` chứa top 50 predictions có sai số lớn nhất. Những rows này có thể là:
- Outliers
- Data quality issues
- Mislabelled ground truth

Lọc bỏ chúng giúp predictions stability tốt hơn.

**Example**:
```python
# Before
df = pd.DataFrame(...)  # 1000 rows

# After dropping bad rows
df_clean = _drop_known_bad_rows(df)  # 950 rows (removed 50 bad rows)
```


### `load_train_info()`

**Signature**:
```python
def load_train_info(path: Path) -> Dict[str, Any]:
```

**Mục đích**: Load metadata từ `Train_info.json`.

**Returns**:
```python
{
    "target_column": "Precipitation_mm",
    "group_by": "Date",
    "train_date": "2024-01-10",
    "train_samples": 10000,
    "train_duration": 120.5,
    "model_type": "XGBoost"
}
```


### `_get_feature_names_from_estimator()`

**Signature**:
```python
def _get_feature_names_from_estimator(estimator: Any) -> list[str] | None:
```

**Mục đích**: Cố gắng lấy danh sách feature names từ model object.

**Process**:
1. Try `estimator.feature_names_in_` (sklearn 1.0+)
2. Try `estimator.feature_name_` (older sklearn)
3. Try `estimator.get_booster().feature_names` (XGBoost)
4. Return None nếu không tìm được

**Why needed?**: Đảm bảo predictions sử dụng đúng features như lúc train, tránh feature mismatch errors.


## 🐛 Common Issues

### ❌ Issue 1: FileNotFoundError: Artifacts dir not found

**Triệu chứng**:
```
FileNotFoundError: Artifacts dir not found: /path/to/Machine_learning_artifacts/latest
```

**Nguyên nhân**: Chưa train model hoặc đang chạy script từ sai directory.

**Giải pháp**:
```bash
# Option 1: Train model first
python manage.py train_model

# Option 2: Specify full path to artifacts
python WeatherForcast.py \
    --artifacts-dir /full/path/to/Machine_learning_artifacts/latest
```


### ❌ Issue 2: KeyError: 'Humidity_%'

**Triệu chứng**:
```
KeyError: 'Humidity_%'
```

**Nguyên nhân**: Input CSV thiếu column cần thiết.

**Giải pháp**:
```python
# Kiểm tra required columns trong Feature_list.json
import json

with open("Machine_learning_artifacts/latest/Feature_list.json") as f:
    features = json.load(f)
    required_cols = features["all_feature_columns"]
    print("Required columns:", required_cols)

# Đảm bảo input CSV có tất cả columns này
```


### ❌ Issue 3: django.core.exceptions.ImproperlyConfigured

**Triệu chứng**:
```
django.core.exceptions.ImproperlyConfigured: Requested setting DATABASES, but settings are not configured...
```

**Nguyên nhân**: Script chạy Django setup nhưng không tìm được settings.

**Giải pháp**:
```bash
# Set DJANGO_SETTINGS_MODULE environment variable
export DJANGO_SETTINGS_MODULE=WeatherForcast.settings

# Hoặc trong script, đảm bảo có:
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "WeatherForcast.settings")
# django.setup()
```


### ❌ Issue 4: Script chạy nhưng predictions.csv rỗng

**Triệu chứng**:
```bash
python WeatherForcast.py --input data.csv
# Script exits successfully
cat predictions.csv
# Empty file or only headers
```

**Nguyên nhân**:
- Input data bị filter hết do bad rows
- Validation errors

**Giải pháp**:
```bash
# Check script logs
python WeatherForcast.py --input data.csv 2>&1 | tee forecast.log

# Look for warnings:
# "Removed 1000 known bad rows from input data"
# → Input data có vấn đề

# Debug với smaller dataset
head -100 data.csv > sample.csv
python WeatherForcast.py --input sample.csv
```


### ❌ Issue 5: Predictions = NaN

**Triệu chứng**:
```csv
Date,Predicted_Precipitation_mm
2024-01-15,nan
2024-01-16,nan
```

**Nguyên nhân**:
- Input features có NaN values
- Transform pipeline tạo ra NaN
- Feature engineering logic lỗi

**Giải pháp**:
```python
# Add debugging trong script
df_before = df.copy()
print("NaN before transform:", df.isnull().sum().sum())

# Apply transform
df_transformed = pipeline.transform(df)
print("NaN after transform:", df_transformed.isnull().sum().sum())

# Check predictions
predictions = model.predict(df_transformed)
print("NaN in predictions:", np.isnan(predictions).sum())

# Identify problem step
```


## 🚀 Future Enhancements

- [ ] **Add logging**: Structured logging với log levels (INFO, WARNING, ERROR)
- [ ] **Progress bar**: Show progress khi process large datasets (với tqdm)
- [ ] **Validation mode**: Compare predictions với ground truth nếu có
- [ ] **Multiple output formats**: JSON, Parquet, SQLite ngoài CSV
- [ ] **API server mode**: Run như HTTP API server (FastAPI/Flask)
- [ ] **Streaming predictions**: Process data theo chunks cho big files
- [ ] **Confidence intervals**: Return prediction intervals (min/max)
- [ ] **Feature drift detection**: Warn nếu input features khác xa training distribution
- [ ] **Auto-retry**: Retry với fallback model nếu prediction fails


## 📞 Related Files

**Imports**:
```python
# Trong WeatherForcast.py
from Weather_Forcast_App.Machine_learning_model.features.Build_transfer import (
    WeatherFeatureBuilder,
)

from Weather_Forcast_App.Machine_learning_model.interface import WeatherPredictor
```

**Reads từ**:
- `Machine_learning_artifacts/latest/Model.pkl`
- `Machine_learning_artifacts/latest/Transform_pipeline.pkl`
- `Machine_learning_artifacts/latest/Feature_list.json`
- `Machine_learning_artifacts/latest/Train_info.json`
- `debug_top50_errors.csv` (optional)

**Writes to**:
- `predictions.csv` (output file)

**Called by**:
- Cron jobs (scheduled forecasts)
- Django management commands (`run_forecast`)
- Docker containers
- Shell scripts


## 👨‍💻 Maintainer

**Võ Anh Nhật** - voanhnhat1612@gmail.com

*Last Updated: March 8, 2026*
