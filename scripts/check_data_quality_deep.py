"""Deep quality check: Selenium + HTML Parse + OpenWeather."""
import pandas as pd
import numpy as np
from pathlib import Path

crawl_dir = Path(r"d:\PROJECT_WEATHER_FORCAST\data\data_crawl")
xlsx_files = sorted([f for f in crawl_dir.iterdir() if f.suffix == ".xlsx" and not f.name.startswith("~$")])
csv_files = sorted([f for f in crawl_dir.iterdir() if f.suffix == ".csv"])

# =====================================================
# PART 1: ALL SELENIUM FILES (43-col xlsx) DEEP CHECK
# =====================================================
print("=" * 70)
print("PART 1: SELENIUM FILES (43-col) - COMPREHENSIVE")
print("=" * 70)

selenium_files = []
vrain_api_files = []
for f in xlsx_files:
    df = pd.read_excel(f, nrows=2)
    if df.shape[1] == 43:
        selenium_files.append(f)
    elif df.shape[1] == 8:
        vrain_api_files.append(f)

print(f"\nSelenium files: {len(selenium_files)}")
print(f"Vrain API files: {len(vrain_api_files)} (SKIPPED)")

# Load all Selenium data
all_selenium = []
file_stats = []
for f in selenium_files:
    df = pd.read_excel(f)
    missing_pct = df.isna().mean().mean() * 100
    file_stats.append({
        "file": f.name,
        "rows": len(df),
        "missing_pct": missing_pct,
        "stations": df["station_name"].nunique() if "station_name" in df.columns else 0,
        "provinces": df["province"].nunique() if "province" in df.columns else 0,
        "rain_mean": df["rain_total"].mean() if "rain_total" in df.columns else None,
        "temp_mean": df["temperature_current"].mean() if "temperature_current" in df.columns else None,
    })
    all_selenium.append(df)

# Per-file summary
print("\n--- Per-file summary (Selenium) ---")
for s in file_stats:
    print(f"  {s['file']}: {s['rows']} rows, {s['stations']} stations, {s['provinces']} provinces, "
          f"missing={s['missing_pct']:.1f}%, rain_mean={s['rain_mean']:.2f}, temp_mean={s['temp_mean']:.2f}")

# Check consistency
row_counts = [s['rows'] for s in file_stats]
station_counts = [s['stations'] for s in file_stats]
print(f"\n--- Consistency ---")
print(f"  Row counts: min={min(row_counts)}, max={max(row_counts)}, all same={len(set(row_counts))==1}")
print(f"  Station counts: min={min(station_counts)}, max={max(station_counts)}, all same={len(set(station_counts))==1}")

# Merge all selenium data for overall analysis
df_sel = pd.concat(all_selenium, ignore_index=True)
print(f"\n--- Combined Selenium data: {df_sel.shape} ---")

# Check all numeric columns for anomalies
numeric_cols = df_sel.select_dtypes(include=[np.number]).columns
print(f"\n--- Numeric column ranges ---")
for col in numeric_cols:
    vals = df_sel[col].dropna()
    if len(vals) == 0:
        print(f"  {col}: ALL NULL")
        continue
    print(f"  {col}: min={vals.min():.2f}, max={vals.max():.2f}, mean={vals.mean():.2f}, "
          f"std={vals.std():.2f}, nulls={df_sel[col].isna().sum()}/{len(df_sel)}")

# Check for negative values where they shouldn't be
print(f"\n--- Anomaly checks ---")
neg_cols = {
    "rain_total": "should be >= 0",
    "rain_current": "should be >= 0",
    "humidity_current": "should be 0-100",
    "pressure_current": "should be 900-1100 hPa",
    "temperature_current": "should be -10 to 50°C",
    "wind_speed_current": "should be >= 0",
    "visibility_current": "should be >= 0",
    "cloud_cover_current": "should be 0-100",
}
for col, desc in neg_cols.items():
    if col in df_sel.columns:
        vals = df_sel[col].dropna()
        if col == "rain_total" or col == "rain_current" or col == "wind_speed_current" or col == "visibility_current":
            bad = (vals < 0).sum()
            print(f"  {col} ({desc}): {bad} negative values")
        elif col == "humidity_current" or col == "cloud_cover_current":
            bad = ((vals < 0) | (vals > 100)).sum()
            print(f"  {col} ({desc}): {bad} out-of-range values")
        elif col == "pressure_current":
            bad = ((vals < 800) | (vals > 1200)).sum()
            print(f"  {col} ({desc}): {bad} out-of-range values")
        elif col == "temperature_current":
            bad = ((vals < -10) | (vals > 55)).sum()
            print(f"  {col} ({desc}): {bad} out-of-range values")

# Check timestamp format
print(f"\n--- Timestamp format check ---")
ts = df_sel["timestamp"].dropna()
print(f"  Sample timestamps: {ts.iloc[:3].tolist()}")
print(f"  Unique timestamps: {ts.nunique()}")

# Check data_source values
if "data_source" in df_sel.columns:
    print(f"\n--- data_source values ---")
    print(df_sel["data_source"].value_counts().to_string())

# Check data_quality values
if "data_quality" in df_sel.columns:
    print(f"\n--- data_quality values ---")
    print(df_sel["data_quality"].value_counts().to_string())

# Check duplicate stations within same timestamp
print(f"\n--- Duplicate check (same station + timestamp) ---")
if "station_name" in df_sel.columns and "timestamp" in df_sel.columns:
    dups = df_sel.duplicated(subset=["station_name", "timestamp"]).sum()
    print(f"  Duplicate (station_name + timestamp): {dups}/{len(df_sel)}")

# =====================================================
# PART 2: HTML PARSE FILES (CSV) DEEP CHECK
# =====================================================
print("\n" + "=" * 70)
print("PART 2: HTML PARSE FILES (CSV) - COMPREHENSIVE")
print("=" * 70)
print(f"\nCSV files: {len(csv_files)}")

csv_file_stats = []
all_csv = []
for f in csv_files:
    df = pd.read_csv(f, encoding="utf-8-sig")
    csv_file_stats.append({
        "file": f.name,
        "rows": len(df),
        "stations": df["station_name"].nunique() if "station_name" in df.columns else 0,
        "provinces": df["province"].nunique() if "province" in df.columns else 0,
        "rain_mean": df["rain_total"].mean() if "rain_total" in df.columns else None,
        "rain_zeros": (df["rain_total"] == 0).sum() if "rain_total" in df.columns else 0,
        "dups": df.duplicated().sum(),
    })
    all_csv.append(df)

# Per-file summary
print("\n--- Per-file summary (HTML Parse CSV) ---")
for s in csv_file_stats:
    print(f"  {s['file']}: {s['rows']} rows, {s['stations']} stations, {s['provinces']} provinces, "
          f"rain_mean={s['rain_mean']:.2f}, rain_zeros={s['rain_zeros']}, dups={s['dups']}")

df_csv = pd.concat(all_csv, ignore_index=True)
print(f"\n--- Combined CSV data: {df_csv.shape} ---")

# Check status values
if "status" in df_csv.columns:
    print(f"\n--- status values ---")
    print(df_csv["status"].value_counts().to_string())

# Check province coverage
if "province" in df_csv.columns:
    print(f"\n--- Province coverage (top 15) ---")
    print(df_csv["province"].value_counts().head(15).to_string())

# Check station_id bug across ALL csv files
if "station_id" in df_csv.columns and "station_name" in df_csv.columns:
    same = (df_csv["station_id"] == df_csv["station_name"]).sum()
    print(f"\n--- station_id == station_name bug ---")
    print(f"  All CSVs: {same}/{len(df_csv)} ({same/len(df_csv)*100:.1f}%)")

# Check total overlap across csv files (how many unique data points)
print(f"\n--- Data overlap across CSV files ---")
if "station_name" in df_csv.columns and "timestamp" in df_csv.columns:
    total_rows = len(df_csv)
    unique_rows = df_csv.drop_duplicates(subset=["station_name", "timestamp"]).shape[0]
    print(f"  Total rows across all CSVs: {total_rows}")
    print(f"  Unique (station_name + timestamp): {unique_rows}")
    print(f"  Overlap/duplicates: {total_rows - unique_rows} ({(total_rows-unique_rows)/total_rows*100:.1f}%)")

# Check rain_total distribution across all CSVs
if "rain_total" in df_csv.columns:
    rt = pd.to_numeric(df_csv["rain_total"], errors="coerce")
    print(f"\n--- rain_total distribution (all CSVs) ---")
    print(f"  zeros: {(rt == 0).sum()} ({(rt == 0).sum()/len(rt)*100:.1f}%)")
    print(f"  > 0 and <= 5: {((rt > 0) & (rt <= 5)).sum()}")
    print(f"  > 5 and <= 20: {((rt > 5) & (rt <= 20)).sum()}")
    print(f"  > 20 and <= 50: {((rt > 20) & (rt <= 50)).sum()}")
    print(f"  > 50 and <= 100: {((rt > 50) & (rt <= 100)).sum()}")
    print(f"  > 100: {(rt > 100).sum()}")
    print(f"  negative: {(rt < 0).sum()}")

# =====================================================
# PART 3: CHECK FOR OPENWEATHER API FILES
# =====================================================
print("\n" + "=" * 70)
print("PART 3: OPENWEATHER API FILES CHECK")
print("=" * 70)

# OpenWeather script outputs different columns than Selenium/HTML Parse
# Check if any xlsx file has a different schema
other_xlsx = [f for f in xlsx_files if f not in selenium_files and f not in vrain_api_files]
print(f"\nXLSX files not matching Selenium (43-col) or Vrain API (8-col): {len(other_xlsx)}")
for f in other_xlsx:
    df = pd.read_excel(f, nrows=2)
    print(f"  {f.name}: {df.shape[1]} cols - {list(df.columns)}")

# Check if any CSV has different columns from HTML Parse
html_parse_cols = set(df_csv.columns) if len(all_csv) > 0 else set()
other_csv = []
for f in csv_files:
    df = pd.read_csv(f, encoding="utf-8-sig", nrows=2)
    if set(df.columns) != html_parse_cols:
        other_csv.append(f)
        print(f"  Different CSV: {f.name}: {list(df.columns)}")

if len(other_csv) == 0:
    print("  All CSV files have identical column structure")

# Check data_source if exists in selenium data
if "data_source" in df_sel.columns:
    sources = df_sel["data_source"].unique()
    ow_data = df_sel[df_sel["data_source"].str.contains("openweather|api|weather", case=False, na=False)]
    if len(ow_data) > 0:
        print(f"\n  OpenWeather data found within Selenium xlsx: {len(ow_data)} rows")
    else:
        print(f"\n  No OpenWeather-tagged data found in xlsx files")
        print(f"  data_source values: {sources}")

print("\n" + "=" * 70)
print("SUMMARY OF ISSUES")
print("=" * 70)
print("""
[SELENIUM - 43 col xlsx]:
  - error_reason column is 100% NULL in every file
  - Check other columns for missing data patterns
  
[HTML PARSE - 8 col csv]:
  - station_id = station_name (BUG - not a real ID)
  - Only 8 columns vs 43 in Selenium (missing weather details)
  - Might have significant data overlap between files
  
[OPENWEATHER API]:
  - Check if any data was crawled at all
""")

print("Done.")
