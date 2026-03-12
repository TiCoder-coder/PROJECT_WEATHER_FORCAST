"""Check data quality of crawled files."""
import pandas as pd
import numpy as np
import os
from pathlib import Path

crawl_dir = Path(r"d:\PROJECT_WEATHER_FORCAST\data\data_crawl")
xlsx_files = sorted([f for f in crawl_dir.iterdir() if f.suffix == ".xlsx" and not f.name.startswith("~$")])
csv_files = sorted([f for f in crawl_dir.iterdir() if f.suffix == ".csv"])
print(f"XLSX: {len(xlsx_files)} | CSV: {len(csv_files)}")

# ===== Categorize XLSX by column count =====
xlsx_groups = {}
for f in xlsx_files:
    df = pd.read_excel(f)
    nc = df.shape[1]
    if nc not in xlsx_groups:
        xlsx_groups[nc] = {"count": 0, "cols": list(df.columns), "rows": [], "file": f.name}
    xlsx_groups[nc]["count"] += 1
    xlsx_groups[nc]["rows"].append(df.shape[0])

for nc, g in sorted(xlsx_groups.items()):
    print(f"\n=== XLSX {nc}-col: {g['count']} files, rows range: {min(g['rows'])}-{max(g['rows'])} ===")
    print(f"  Columns: {g['cols'][:15]}")
    if len(g['cols']) > 15:
        print(f"  ... + {len(g['cols'])-15} more")

# ===== Categorize CSV by column count =====
csv_groups = {}
for f in csv_files:
    df = pd.read_csv(f, encoding="utf-8-sig", nrows=5)
    nc = df.shape[1]
    if nc not in csv_groups:
        csv_groups[nc] = {"count": 0, "cols": list(df.columns), "file": f.name}
    csv_groups[nc]["count"] += 1

for nc, g in sorted(csv_groups.items()):
    print(f"\n=== CSV {nc}-col: {g['count']} files ===")
    print(f"  Columns: {g['cols']}")

# ===== Deep quality check for each category =====
print("\n" + "=" * 70)
print("DEEP QUALITY CHECK")
print("=" * 70)

# Check one 43-col xlsx (Selenium output)
for nc, g in xlsx_groups.items():
    if nc == 43:
        print(f"\n--- SELENIUM OUTPUT (43-col xlsx): {g['file']} ---")
        df = pd.read_excel(crawl_dir / g["file"])
        print(f"Shape: {df.shape}")
        print(f"Missing values per column:")
        missing = df.isna().sum()
        for col in df.columns:
            m = missing[col]
            pct = m / len(df) * 100
            dtype = df[col].dtype
            if m > 0:
                print(f"  {col}: {m} ({pct:.1f}%) - dtype={dtype}")
        
        # Check key columns
        print(f"\nKey column checks:")
        for col in ["station_name", "province", "rain_total", "temperature_current", "timestamp"]:
            if col in df.columns:
                non_null = df[col].notna().sum()
                unique = df[col].nunique()
                print(f"  {col}: {non_null}/{len(df)} non-null, {unique} unique, dtype={df[col].dtype}")
                if df[col].dtype in ["float64", "int64"]:
                    print(f"    min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}")
            else:
                print(f"  {col}: MISSING COLUMN!")
        
        # Check rain_total specifically
        if "rain_total" in df.columns:
            rt = df["rain_total"]
            print(f"\n  rain_total stats:")
            print(f"    zeros: {(rt == 0).sum()} ({(rt == 0).sum()/len(rt)*100:.1f}%)")
            print(f"    negative: {(rt < 0).sum()}")
            print(f"    > 100: {(rt > 100).sum()}")
            print(f"    NaN: {rt.isna().sum()}")
        break

# Check 8-col xlsx (Vrain API output - skip per user request)
for nc, g in xlsx_groups.items():
    if nc == 8:
        print(f"\n--- VRAIN API OUTPUT (8-col xlsx): SKIPPED per user request ---")
        print(f"  ({g['count']} files)")
        break

# Check CSV files (HTML Parse output)
for nc, g in csv_groups.items():
    print(f"\n--- HTML PARSE OUTPUT (CSV {nc}-col): {g['file']} ---")
    df = pd.read_csv(crawl_dir / g["file"], encoding="utf-8-sig")
    print(f"Shape: {df.shape}")
    print(f"Missing values per column:")
    missing = df.isna().sum()
    for col in df.columns:
        m = missing[col]
        pct = m / len(df) * 100
        dtype = df[col].dtype
        print(f"  {col}: {m} ({pct:.1f}%) - dtype={dtype}")
    
    print(f"\nSample data (first 3 rows):")
    print(df.head(3).to_string())
    
    # Check data types and ranges
    print(f"\nData type checks:")
    for col in df.columns:
        if col == "rain_total":
            vals = pd.to_numeric(df[col], errors="coerce")
            print(f"  {col}: min={vals.min()}, max={vals.max()}, zeros={( vals==0).sum()}, NaN={vals.isna().sum()}")
        elif col in ["station_name", "province", "district"]:
            print(f"  {col}: {df[col].nunique()} unique, sample={df[col].dropna().iloc[:3].tolist()}")
        elif col == "station_id":
            print(f"  {col}: {df[col].nunique()} unique (should match station count)")
    
    # Check for duplicate rows
    dups = df.duplicated().sum()
    print(f"\nDuplicate rows: {dups} ({dups/len(df)*100:.1f}%)")
    
    # Check station_id == station_name (known bug)
    if "station_id" in df.columns and "station_name" in df.columns:
        same = (df["station_id"] == df["station_name"]).sum()
        print(f"station_id == station_name: {same}/{len(df)} ({same/len(df)*100:.1f}%)")
        if same > len(df) * 0.5:
            print("  ⚠️ BUG CONFIRMED: station_id is just a copy of station_name!")

# ===== Cross-check: compare schemas =====
print("\n" + "=" * 70)
print("SCHEMA COMPARISON ACROSS FORMATS")
print("=" * 70)

# Get 43-col columns
selenium_cols = set()
for nc, g in xlsx_groups.items():
    if nc == 43:
        selenium_cols = set(g["cols"])
        break

csv_cols = set()
for nc, g in csv_groups.items():
    csv_cols = set(g["cols"])

common = selenium_cols & csv_cols
only_selenium = selenium_cols - csv_cols
only_csv = csv_cols - selenium_cols

print(f"\nSelenium (43-col) vs HTML Parse ({len(csv_cols)}-col) CSV:")
print(f"  Common columns: {len(common)}")
print(f"  Only in Selenium: {len(only_selenium)} -> {sorted(only_selenium)}")
print(f"  Only in CSV: {len(only_csv)} -> {sorted(only_csv)}")

# ===== Consistency check: same timestamp data across files =====
print("\n" + "=" * 70)
print("DATA CONSISTENCY CHECK")
print("=" * 70)

# Check if CSVs contain overlapping data
if len(csv_files) >= 2:
    df1 = pd.read_csv(csv_files[0], encoding="utf-8-sig")
    df2 = pd.read_csv(csv_files[-1], encoding="utf-8-sig")
    print(f"\nComparing first vs last CSV:")
    print(f"  File 1: {csv_files[0].name} ({df1.shape})")
    print(f"  File 2: {csv_files[-1].name} ({df2.shape})")
    
    if "station_name" in df1.columns and "station_name" in df2.columns:
        s1 = set(df1["station_name"].dropna().unique())
        s2 = set(df2["station_name"].dropna().unique())
        print(f"  Stations in file1: {len(s1)}")
        print(f"  Stations in file2: {len(s2)}")
        print(f"  Common stations: {len(s1 & s2)}")
        
    if "rain_total" in df1.columns and "rain_total" in df2.columns:
        r1 = pd.to_numeric(df1["rain_total"], errors="coerce")
        r2 = pd.to_numeric(df2["rain_total"], errors="coerce")
        print(f"  rain_total file1: mean={r1.mean():.3f}, zeros={( r1==0).sum()}")
        print(f"  rain_total file2: mean={r2.mean():.3f}, zeros={( r2==0).sum()}")
        
        # Check if data is identical (copy)
        if df1.shape == df2.shape:
            identical = (df1.fillna("") == df2.fillna("")).all().all()
            if identical:
                print("  ⚠️ WARNING: Files are IDENTICAL copies!")
            else:
                diff_cols = [c for c in df1.columns if not (df1[c].fillna("") == df2[c].fillna("")).all()]
                print(f"  Differences in columns: {diff_cols}")

print("\nDone.")
