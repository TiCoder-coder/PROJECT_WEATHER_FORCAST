import numpy as np
import pandas as pd
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

print("PROJECT_ROOT =", PROJECT_ROOT)
print("sys.path[0]  =", sys.path[0])

import numpy as np
import pandas as pd

from Weather_Forcast_App.Machine_learning_model.features.Transformers import (
    WeatherFeatureTransformer,
)

def main():
    print("=" * 70)
    print("🧪 TEST WeatherFeatureTransformer")
    print("=" * 70)

    # 1) Tạo dữ liệu giả (có numeric + categorical + datetime + missing)
    n = 200
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        "temp": np.random.uniform(10, 40, n),
        "humidity": np.random.uniform(30, 100, n),
        "wind_speed": np.random.uniform(0, 25, n),
        "season": np.random.choice(["Spring", "Summer", "Fall", "Winter"], n),
        "location": np.random.choice(["Hanoi", "HCMC", "Danang"], n),
    })

    # add missing
    df.loc[5:10, "humidity"] = np.nan
    df.loc[20:25, "season"] = None
    df.loc[30:31, "timestamp"] = None

    # 2) Fit + transform train
    tfm = WeatherFeatureTransformer(
        numeric_scaler="standard",
        categorical_encoder="onehot",
        add_datetime_features=True,
    )

    X_train = df.iloc[:160].copy()
    X_test = df.iloc[160:].copy()

    X_train_t = tfm.fit_transform(X_train)
    X_test_t = tfm.transform(X_test)

    print("\n✅ Fit/Transform OK")
    print("Train shape:", X_train_t.shape)
    print("Test  shape:", X_test_t.shape)

    # 3) Check: số cột train == test + không NaN toàn bộ
    assert X_train_t.shape[1] == X_test_t.shape[1], "❌ Mismatch number of features!"
    assert list(X_train_t.columns) == list(X_test_t.columns), "❌ Column order mismatch!"
    print("✅ Same feature columns between train & test")

    # 4) Test case: predict thiếu cột (drop cột)
    X_missing_col = X_test.drop(columns=["wind_speed"])
    X_missing_t = tfm.transform(X_missing_col)
    assert list(X_missing_t.columns) == list(X_train_t.columns), "❌ Missing column should still align!"
    print("✅ Missing-column input still aligns schema")

    # 5) Test case: predict có category mới (unseen)
    X_unseen = X_test.copy()
    X_unseen.loc[X_unseen.index[:5], "location"] = "NEW_CITY"
    X_unseen_t = tfm.transform(X_unseen)
    assert list(X_unseen_t.columns) == list(X_train_t.columns), "❌ Unseen category should not break!"
    print("✅ Unseen category does NOT break (handle_unknown=ignore)")

    # 6) Save/Load
    tmp = Path("/tmp/weather_transformer_test.joblib")
    tfm.save(tmp)
    tfm2 = WeatherFeatureTransformer.load(tmp)

    X_test_t2 = tfm2.transform(X_test)
    assert list(X_test_t2.columns) == list(X_test_t.columns), "❌ Save/Load columns mismatch!"
    assert X_test_t2.shape == X_test_t.shape, "❌ Save/Load shape mismatch!"
    print("✅ Save/Load OK")

    # 7) So sánh output gần giống nhau sau load (sai số float nhỏ)
    diff = np.abs(X_test_t.values - X_test_t2.values).max()
    print("Max abs diff after load:", diff)
    assert diff < 1e-9, "❌ Transformed values changed after load!"
    print("\n🎉 ALL TRANSFORMER TESTS PASSED!")

if __name__ == "__main__":
    main()
