"""Quick diagnostic: count features created by default vs training config."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "WeatherForcast.settings")

import pandas as pd
import numpy as np
import json

from Weather_Forcast_App.Machine_learning_model.features.Build_transfer import WeatherFeatureBuilder
from Weather_Forcast_App.paths import ML_ARTIFACTS_LATEST

# What training produced
feat_data = json.loads((ML_ARTIFACTS_LATEST / "Feature_list.json").read_text(encoding="utf-8"))
created_during_training = feat_data.get("created_features", [])
print(f"Features created during TRAINING: {len(created_during_training)}")
print(f"  -> Only: time features + interaction features")
print(f"  -> NO lag, rolling, or difference features")
print()

# What the default builder config enables
builder = WeatherFeatureBuilder()  # Default config (same as predictor creates)
cfg = builder.config
print("Default builder config (used during PREDICTION):")
print(f"  lag_features.enabled: {cfg.get('lag_features', {}).get('enabled', True)}")
print(f"    periods: {cfg.get('lag_features', {}).get('periods', [])}")
print(f"  rolling_features.enabled: {cfg.get('rolling_features', {}).get('enabled', True)}")
print(f"    windows: {cfg.get('rolling_features', {}).get('windows', [])}")
print(f"    functions: {cfg.get('rolling_features', {}).get('functions', [])}")
print(f"  difference_features.enabled: {cfg.get('difference_features', {}).get('enabled', True)}")
print(f"    periods: {cfg.get('difference_features', {}).get('periods', [])}")
print()

# Simulate the cascade with a minimal dataset
df = pd.DataFrame({
    'temperature_current': [28.5] * 10,
    'humidity_current': [75.0] * 10,
    'pressure_current': [1013.0] * 10,
    'wind_speed_current': [5.5] * 10,
    'rain_total': [2.5] * 10,
    'timestamp': pd.date_range('2026-03-10', periods=10, freq='h'),
})

print("Simulating feature cascade with 10 rows, 5 weather columns:")
n_weather = len(builder._get_numeric_weather_columns(df))
print(f"  Initial weather columns detected: {n_weather}")

# After lag
n_lag_periods = len(cfg.get('lag_features', {}).get('periods', [1, 3, 6, 12, 24, 168]))
n_after_lag = n_weather + n_weather * n_lag_periods
print(f"  After lag ({n_lag_periods} periods): {n_after_lag} weather columns")

# Rolling picks up lag cols too (they match keywords)
n_windows = len(cfg.get('rolling_features', {}).get('windows', [3, 6, 12, 24, 168]))
n_functions = len(cfg.get('rolling_features', {}).get('functions', ['mean', 'std', 'min', 'max']))
n_rolling_new = n_after_lag * n_windows * n_functions
n_after_rolling = n_after_lag + n_rolling_new
print(f"  After rolling ({n_windows} windows x {n_functions} funcs): {n_after_rolling} weather columns")

# Difference picks up all
n_diff_periods = len(cfg.get('difference_features', {}).get('periods', [1, 6, 24]))
n_diff_new = n_after_rolling * n_diff_periods * 2  # diff + pct_change
n_total = n_after_rolling + n_diff_new
print(f"  After difference ({n_diff_periods} periods x 2): {n_total} total columns!")
print()
print(f"  With 427 rows: {427 * n_total:,} cells to compute!")
