"""Test the full WeatherPredictor.predict() pipeline to reproduce training messages."""
import sys, os, io, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "WeatherForcast.settings")

import django
django.setup()

import numpy as np
import pandas as pd

# Use the full predictor pipeline
from Weather_Forcast_App.Machine_learning_model.interface.predictor import WeatherPredictor
from Weather_Forcast_App.paths import ML_ARTIFACTS_LATEST

print("Loading predictor from artifacts...")
predictor = WeatherPredictor.from_artifacts(str(ML_ARTIFACTS_LATEST))
print(f"Model: {type(predictor.model).__name__}")
print(f"Target: {predictor.target_column}")
print(f"Features: {len(predictor.feature_columns)}")
print(f"Has feature_builder: {predictor.feature_builder is not None}")
print(f"Has pipeline: {predictor.pipeline is not None}")
print()

# Create a small fake dataset resembling crawled data (5 rows)
np.random.seed(42)
df = pd.DataFrame({
    'station_id': ['S001', 'S002', 'S003', 'S004', 'S005'],
    'station_name': ['Ha Noi', 'HCM', 'Da Nang', 'Hue', 'Can Tho'],
    'province': ['Ha Noi', 'Ho Chi Minh', 'Da Nang', 'Thua Thien Hue', 'Can Tho'],
    'district': ['Cau Giay', 'Quan 1', 'Hai Chau', 'TP Hue', 'Ninh Kieu'],
    'latitude': [21.03, 10.82, 16.07, 16.46, 10.03],
    'longitude': [105.85, 106.63, 108.22, 107.6, 105.73],
    'temperature_current': [28.5, 32.1, 29.3, 27.8, 31.5],
    'temperature_max': [32.0, 35.0, 33.0, 30.0, 34.0],
    'temperature_min': [24.0, 26.0, 25.0, 23.0, 27.0],
    'temperature_avg': [28.0, 30.5, 29.0, 26.5, 30.5],
    'humidity_current': [75, 80, 78, 82, 79],
    'humidity_max': [90, 95, 92, 88, 93],
    'humidity_min': [60, 65, 62, 70, 64],
    'humidity_avg': [75, 80, 77, 79, 78],
    'pressure_current': [1013, 1012, 1014, 1013, 1011],
    'pressure_max': [1015, 1014, 1016, 1015, 1013],
    'pressure_min': [1010, 1009, 1011, 1010, 1008],
    'pressure_avg': [1012, 1011, 1013, 1012, 1010],
    'wind_speed_current': [5.5, 3.2, 4.1, 6.0, 2.8],
    'wind_speed_max': [8.0, 6.0, 7.0, 9.0, 5.0],
    'wind_speed_min': [2.0, 1.0, 1.5, 3.0, 0.5],
    'wind_speed_avg': [5.0, 3.5, 4.0, 6.0, 2.5],
    'wind_direction_current': [180, 90, 270, 45, 135],
    'wind_direction_avg': [175, 95, 265, 50, 130],
    'cloud_cover_current': [60, 40, 55, 70, 45],
    'cloud_cover_max': [80, 60, 75, 90, 65],
    'cloud_cover_min': [40, 20, 35, 50, 25],
    'cloud_cover_avg': [60, 40, 55, 70, 45],
    'visibility_current': [10, 12, 11, 8, 13],
    'visibility_max': [15, 15, 14, 12, 15],
    'visibility_min': [5, 8, 7, 4, 9],
    'visibility_avg': [10, 12, 11, 8, 13],
    'thunder_probability': [0.2, 0.1, 0.15, 0.3, 0.05],
    'rain_total': [2.5, 0.0, 1.2, 5.0, 0.0],
    'timestamp': ['2026-03-10 12:00:00'] * 5,
})

print(f"Input DataFrame: {len(df)} rows, {len(df.columns)} cols")
print()

# Capture stdout
captured = io.StringIO()
old_stdout = sys.stdout
sys.stdout = captured

start = time.time()
try:
    result = predictor.predict(df)
    elapsed = time.time() - start
    sys.stdout = old_stdout
    print(f"Prediction completed in {elapsed:.2f}s")
    print(f"Predictions: {result['predictions']}")
except Exception as e:
    sys.stdout = old_stdout
    elapsed = time.time() - start
    print(f"ERROR after {elapsed:.2f}s: {e}")

output = captured.getvalue()
print()
print("=== CAPTURED STDOUT during predict() ===")
if output.strip():
    # Show first 3000 chars
    print(output[:3000])
else:
    print("(empty)")
print("=== END ===")

if "Training" in output:
    print("\n*** CONFIRMED: Training messages detected during prediction! ***")
    # Find lines containing "Training"
    for line in output.split('\n'):
        if 'Training' in line or 'training' in line:
            print(f"  >>> {line}")
