"""Test prediction to see if training messages appear."""
import sys, os, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "WeatherForcast.settings")

import django
django.setup()

import joblib
import numpy as np
import pandas as pd

# Load ensemble
model = joblib.load('Weather_Forcast_App/Machine_learning_artifacts/latest/Model.pkl')

# Create dummy data with 104 features (matching n_features_in_)
np.random.seed(42)
X = pd.DataFrame(np.random.randn(5, 104), columns=[f"f_{i}" for i in range(104)])

# Capture stdout
captured = io.StringIO()
old_stdout = sys.stdout
sys.stdout = captured

try:
    print("=== CALLING model.predict(X) ===")
    preds = model.predict(X)
    print(f"=== DONE. preds shape: {preds.shape} ===")
except Exception as e:
    print(f"=== ERROR: {e} ===")
finally:
    sys.stdout = old_stdout

output = captured.getvalue()
print("=== CAPTURED STDOUT ===")
print(output)
print("=== END ===")

if "Training" in output:
    print("\n*** WARNING: Training messages detected during prediction! ***")
else:
    print("\n OK: No training messages during direct ensemble prediction.")
