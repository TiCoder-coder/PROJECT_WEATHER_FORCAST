# 📊 Dataset_after_split/ — Train/Test/Validation Datasets

## 📁 Overview

Module `Dataset_after_split/` chứa **datasets đã được split** thành Train/Test/Validation sets cho Machine Learning training. Đây là **output của data splitting process**, được tổ chức thành:
- **Dataset_merge/**: Datasets đã merge (combined data từ nhiều nguồn)
- **Dataset_not_merge/**: Datasets riêng lẻ (mỗi nguồn data một folder)
- **split_log.json**: Metadata về quá trình split (số lượng rows, tỉ lệ split)

Folder này là **input chính cho ML training pipeline**, đảm bảo:
- Không có data leakage giữa train/test sets
- Reproducible splits (cùng random seed → cùng split)
- Tracking metadata (biết được data đến từ đâu, split như thế nào)


## 📂 Directory Structure

```
Dataset_after_split/
├── split_log.json                 # 📋 Log file: metadata về split process
├── Dataset_merge/                 # 📦 Merged datasets (combined from all sources)
│   ├── Train/                     # Training set folder
│   ├── Test/                      # Test set folder
│   ├── Validate/                  # Validation set folder
│   ├── merge_train.csv            # 📄 Training CSV (70%)
│   ├── merge_test.csv             # 📄 Test CSV (10%)
│   └── merge_valid.csv            # 📄 Validation CSV (10%)
└── Dataset_not_merge/             # 📦 Individual datasets (per source)
    ├── Train/                     # Training sets per source
    ├── Test/                      # Test sets per source
    └── Validate/                  # Validation sets per source
```


## 🎯 Purpose

### ❓ Tại sao cần folder này?

| Vấn đề | Giải pháp |
|--------|-----------|
| **Dataset gốc quá lớn, cần split cho training** | Tách thành Train (70%), Test (10%), Validate (20%) |
| **Cần đảm bảo không data leakage** | Split một lần, lưu vào file, reuse cho consistency |
| **Training script cần biết path đến train/test sets** | Chuẩn hóa structure: `Dataset_after_split/Dataset_merge/merge_train.csv` |
| **Tracking metadata về split process** | `split_log.json` lưu số lượng rows, output paths |

### ✅ Lợi ích

- **Reproducibility**: Cùng split được reuse nhiều lần, đảm bảo kết quả nhất quán
- **No data leakage**: Train/test splits được tách biệt hoàn toàn
- **Easy access**: Training scripts chỉ cần load `merge_train.csv`, `merge_test.csv`
- **Metadata tracking**: `split_log.json` giúp debug và audit


## 📄 Files & Folders Explained

### 1. `split_log.json` — Split Metadata

**Mục đích**: Log file lưu metadata về quá trình split dataset.

**Format**:
```json
[
  {
    "file": "/path/to/original/cleaned_merge_merged_vrain_data_20260216_121532.csv",
    "rows_total": 7356,
    "rows_train": 5884,
    "rows_validate": 735,
    "rows_test": 737,
    "out_dir": "/path/to/Dataset_after_split/Dataset_merge"
  }
]
```

**Fields**:

| Field | Mô tả |
|-------|-------|
| `file` | Path to original dataset file |
| `rows_total` | Total rows trong dataset gốc |
| `rows_train` | Rows trong training set (70% = 5884/7356) |
| `rows_validate` | Rows trong validation set (10% = 735/7356) |
| `rows_test` | Rows trong test set (10% = 737/7356) |
| `out_dir` | Output directory chứa split datasets |

**Usage**:
```python
import json

# Load split log
with open("Dataset_after_split/split_log.json") as f:
    split_info = json.load(f)

# Check split ratios
for entry in split_info:
    total = entry["rows_total"]
    train = entry["rows_train"]
    test = entry["rows_test"]
    valid = entry["rows_validate"]
    
    print(f"Dataset: {entry['file']}")
    print(f"  Train: {train}/{total} ({train/total*100:.1f}%)")
    print(f"  Test: {test}/{total} ({test/total*100:.1f}%)")
    print(f"  Valid: {valid}/{total} ({valid/total*100:.1f}%)")

# Output:
# Dataset: cleaned_merge_merged_vrain_data_20260216_121532.csv
#   Train: 5884/7356 (80.0%)
#   Test: 737/7356 (10.0%)
#   Valid: 735/7356 (10.0%)
```


### 2. `Dataset_merge/` — Merged Datasets

**Mục đích**: Chứa datasets đã merge (combined) từ nhiều nguồn data (Vrain, API, Web scraping).

#### 📂 Folder Structure

```
Dataset_merge/
├── Train/                 # Folder for training set (may contain subfolders)
├── Test/                  # Folder for test set
├── Validate/              # Folder for validation set
├── merge_train.csv        # 📄 Main training CSV (70% of data)
├── merge_test.csv         # 📄 Main test CSV (10% of data)
└── merge_valid.csv        # 📄 Main validation CSV (20% of data)
```

#### 📄 CSV Files

**merge_train.csv**:
- **Rows**: 5884 (80% of 7356)
- **Purpose**: Training data cho model fitting
- **Usage**: `pd.read_csv("Dataset_after_split/Dataset_merge/merge_train.csv")`

**merge_test.csv**:
- **Rows**: 737 (10% of 7356)
- **Purpose**: Final evaluation sau khi train xong
- **Usage**: Model evaluation, báo cáo metrics cuối cùng

**merge_valid.csv**:
- **Rows**: 735 (10% of 7356)
- **Purpose**: Validation during training (hyperparameter tuning, early stopping)
- **Usage**: Monitor overfitting, select best model

#### 🔧 Data Split Strategy

**Typical split ratios**:

| Set | Percentage | Rows (from 7356) | Purpose |
|-----|------------|------------------|---------|
| **Train** | 80% | 5884 | Model fitting, learning patterns |
| **Validation** | 10% | 735 | Hyperparameter tuning, early stopping |
| **Test** | 10% | 737 | Final evaluation (báo cáo cuối cùng) |

**Splitting method**: 
- **Stratified split** (nếu có): Đảm bảo tỉ lệ target values giống nhau giữa train/test
- **Temporal split** (time series): Train = old data, Test = recent data
- **Random split**: Shuffle data với fixed random seed


### 3. `Dataset_not_merge/` — Individual Datasets

**Mục đích**: Chứa datasets riêng lẻ (per source) thay vì merged.

#### 📂 Folder Structure

```
Dataset_not_merge/
├── Train/
│   ├── vrain_train.csv
│   ├── api_train.csv
│   └── web_train.csv
├── Test/
│   ├── vrain_test.csv
│   ├── api_test.csv
│   └── web_test.csv
└── Validate/
    ├── vrain_valid.csv
    ├── api_valid.csv
    └── web_valid.csv
```

**Khi nào dùng Dataset_not_merge?**:
- ✅ Training separate models cho từng data source
- ✅ Comparing model performance per source
- ✅ Data source analysis (which source has better quality?)
- ❌ Không dùng cho main training pipeline (dùng `Dataset_merge` thay thế)


## 🔧 How to Use

### 1️⃣ Load Training Data

```python
import pandas as pd

# Load train/test/validation sets
train_df = pd.read_csv("Weather_Forcast_App/Machine_learning_model/Dataset_after_split/Dataset_merge/merge_train.csv")
test_df = pd.read_csv("Weather_Forcast_App/Machine_learning_model/Dataset_after_split/Dataset_merge/merge_test.csv")
valid_df = pd.read_csv("Weather_Forcast_App/Machine_learning_model/Dataset_after_split/Dataset_merge/merge_valid.csv")

print(f"Train: {len(train_df)} rows")
print(f"Test: {len(test_df)} rows")
print(f"Valid: {len(valid_df)} rows")

# Output:
# Train: 5884 rows
# Test: 737 rows
# Valid: 735 rows
```


### 2️⃣ Training Workflow

```python
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Step 1: Load data
train_df = pd.read_csv("Dataset_after_split/Dataset_merge/merge_train.csv")
valid_df = pd.read_csv("Dataset_after_split/Dataset_merge/merge_valid.csv")
test_df = pd.read_csv("Dataset_after_split/Dataset_merge/merge_test.csv")

# Step 2: Split features and target
target_col = "Precipitation_mm"
feature_cols = [col for col in train_df.columns if col != target_col and col != "Date"]

X_train = train_df[feature_cols]
y_train = train_df[target_col]

X_valid = valid_df[feature_cols]
y_valid = valid_df[target_col]

X_test = test_df[feature_cols]
y_test = test_df[target_col]

# Step 3: Train model with validation set for early stopping
model = XGBRegressor(
    learning_rate=0.1,
    max_depth=5,
    n_estimators=1000,
    early_stopping_rounds=50
)

model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=True
)

# Step 4: Evaluate on test set (final evaluation)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(f"Test MAE: {mae:.3f} mm")
```


### 3️⃣ Cross-Validation on Training Set

```python
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMRegressor

# Load training data only
train_df = pd.read_csv("Dataset_after_split/Dataset_merge/merge_train.csv")

X_train = train_df.drop(columns=["Precipitation_mm", "Date"])
y_train = train_df["Precipitation_mm"]

# 5-fold cross-validation on training set
model = LGBMRegressor(learning_rate=0.1, num_leaves=31)

cv_scores = cross_val_score(
    model, X_train, y_train,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)

print(f"CV MAE: {-cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
```


### 4️⃣ Re-split Dataset (if needed)

**Khi nào cần re-split?**:
- Original dataset updated (thêm data mới)
- Muốn thay đổi split ratio (70/20/10 → 80/10/10)
- Phát hiện data leakage trong split hiện tại

**Example split script**:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
import json
from pathlib import Path

# Load original dataset
df_original = pd.read_csv("data/data_clean/data_merge_clean/cleaned_merge_data.csv")

# Split: 80% train, 10% validation, 10% test
train_df, temp_df = train_test_split(df_original, test_size=0.2, random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Save splits
output_dir = Path("Weather_Forcast_App/Machine_learning_model/Dataset_after_split/Dataset_merge")
output_dir.mkdir(parents=True, exist_ok=True)

train_df.to_csv(output_dir / "merge_train.csv", index=False)
valid_df.to_csv(output_dir / "merge_valid.csv", index=False)
test_df.to_csv(output_dir / "merge_test.csv", index=False)

# Log split info
split_log = [{
    "file": str(Path("data/data_clean/data_merge_clean/cleaned_merge_data.csv").resolve()),
    "rows_total": len(df_original),
    "rows_train": len(train_df),
    "rows_validate": len(valid_df),
    "rows_test": len(test_df),
    "out_dir": str(output_dir.resolve())
}]

with open("Weather_Forcast_App/Machine_learning_model/Dataset_after_split/split_log.json", "w") as f:
    json.dump(split_log, f, indent=2)

print(f"✅ Split completed:")
print(f"  Train: {len(train_df)} rows")
print(f"  Valid: {len(valid_df)} rows")
print(f"  Test: {len(test_df)} rows")
```


## 📊 Split Ratios Guide

### 📐 Common Split Strategies

| Strategy | Train | Validation | Test | Khi nào dùng |
|----------|-------|------------|------|--------------|
| **70/20/10** | 70% | 20% | 10% | Small datasets (< 10k rows) |
| **80/10/10** | 80% | 10% | 10% | Medium datasets (10k-100k rows) — **project hiện tại** |
| **90/5/5** | 90% | 5% | 5% | Large datasets (> 100k rows) |
| **80/20 (no valid)** | 80% | - | 20% | No hyperparameter tuning |

### 🎯 Role of Each Set

**Training Set** (80%):
- **Purpose**: Fit model parameters
- **Usage**: `model.fit(X_train, y_train)`
- **Size**: Càng lớn càng tốt (more data = better learning)

**Validation Set** (10%):
- **Purpose**: Tune hyperparameters, early stopping
- **Usage**: `model.fit(..., eval_set=[(X_valid, y_valid)])`
- **Size**: Đủ lớn để estimate performance reliably

**Test Set** (10%):
- **Purpose**: Final evaluation (model chưa nhìn thấy data này bao giờ)
- **Usage**: `y_pred = model.predict(X_test)` → báo cáo metrics cuối cùng
- **Size**: Đủ lớn để representative, nhưng không cần quá lớn


## 🐛 Common Issues

### ❌ Issue 1: Train/Test Leakage

**Triệu chứng**:
```python
# Train accuracy = 99%
# Test accuracy = 45%
# → Có thể bị leakage hoặc overfitting
```

**Nguyên nhân**:
- Test data bị leak vào training (duplicate rows)
- Feature engineering sử dụng thông tin từ test set
- Temporal leakage (dùng future data để predict past)

**Giải pháp**:
```python
# Check for duplicate rows between train/test
train_df = pd.read_csv("merge_train.csv")
test_df = pd.read_csv("merge_test.csv")

# Compare fingerprints
train_fingerprints = set(train_df.astype(str).apply("|".join, axis=1))
test_fingerprints = set(test_df.astype(str).apply("|".join, axis=1))

overlap = train_fingerprints & test_fingerprints

if overlap:
    print(f"❌ Data leakage detected: {len(overlap)} duplicate rows!")
else:
    print("✅ No overlap between train/test")
```


### ❌ Issue 2: Imbalanced Target Distribution

**Triệu chứng**:
```python
# Train set: 90% values < 10mm, 10% values > 10mm
# Test set: 50% values < 10mm, 50% values > 10mm
# → Test set not representative of train set
```

**Nguyên nhân**: Random split không bảo toàn distribution của target variable.

**Giải pháo**:
```python
# Use stratified split
from sklearn.model_selection import train_test_split

# Binning target for stratification
df["target_bin"] = pd.cut(df["Precipitation_mm"], bins=5, labels=False)

# Stratified split
train_df, test_df = train_test_split(
    df, 
    test_size=0.2, 
    stratify=df["target_bin"],  # Preserve distribution
    random_state=42
)

# Drop temporary column
train_df = train_df.drop(columns=["target_bin"])
test_df = test_df.drop(columns=["target_bin"])
```


### ❌ Issue 3: Temporal Leakage in Time Series

**Triệu chứng**:
```python
# Training với data từ 2024
# Testing với data từ 2023
# → Model nhìn thấy "future" data!
```

**Nguyên nhân**: Random split không phù hợp với time series data.

**Giải pháo**:
```python
# Temporal split: train = past, test = future
df = df.sort_values("Date")

split_date = "2024-01-01"
train_df = df[df["Date"] < split_date]
test_df = df[df["Date"] >= split_date]

print(f"Train: {train_df['Date'].min()} to {train_df['Date'].max()}")
print(f"Test: {test_df['Date'].min()} to {test_df['Date'].max()}")
```


### ❌ Issue 4: Empty CSV Files

**Triệu chứng**:
```python
train_df = pd.read_csv("merge_train.csv")
print(len(train_df))  # 0 rows
```

**Nguyên nhân**: Split script failed hoặc empty original dataset.

**Giải pháp**:
```bash
# Check file sizes
ls -lh Dataset_after_split/Dataset_merge/

# If files are empty/very small:
# - Re-run split script
# - Check original dataset has data
# - Check split script logic
```


## 🚀 Future Enhancements

- [ ] **Automated split validation**: Script to validate train/test splits (no overlap, correct ratios)
- [ ] **Stratified time series split**: Combine temporal ordering với stratification
- [ ] **K-fold splits**: Generate multiple train/test splits cho cross-validation
- [ ] **Data versioning**: Track dataset versions với DVC (Data Version Control)
- [ ] **Split visualization**: Plot target distribution across train/test/valid sets
- [ ] **Imbalance handling**: SMOTE, undersampling cho imbalanced datasets
- [ ] **Holdout set**: Additional holdout set (5%) never used until final deployment


## 📞 Related Files

**Generated by**:
- Data splitting scripts (custom scripts hoặc training pipeline)

**Used by**:
- `trainning/` — Training scripts load datasets từ folder này
- `evaluation/` — Evaluation scripts đánh giá trên test set
- `WeatherForcast/` — Có thể dùng để validate predictions

**Related data**:
- Original datasets: `data/data_clean/data_merge_clean/`
- Cleaned datasets: `data/data_clean/data_not_merge_clean/`


## 👨‍💻 Maintainer

**Võ Anh Nhật** - voanhnhat1612@gmail.com

*Last Updated: March 8, 2026*
