import pandas as pd
import sys
import os

def clean_datetime_nan(input_path, output_path=None):
    df = pd.read_csv(input_path)
    before = len(df)
    df_clean = df.dropna(subset=["timestamp", "data_time"])
    after = len(df_clean)
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = base + "_cleaned" + ext
    df_clean.to_csv(output_path, index=False)
    print(f"Removed {before-after} rows with NaN in 'timestamp' or 'data_time'. Cleaned file saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_datetime_nan.py <input_csv> [output_csv]")
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    clean_datetime_nan(input_path, output_path)
