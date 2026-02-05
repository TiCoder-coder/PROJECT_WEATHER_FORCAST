"""
Test script cho 4 model:
- Random Forest
- CatBoost
- XGBoost
- LightGBM
"""

import os
import sys
from pathlib import Path

# ===================== DJANGO BOOTSTRAP =====================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

print("PROJECT_ROOT =", PROJECT_ROOT)
print("sys.path[0]  =", sys.path[0])

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "WeatherForcast.settings")

import django
django.setup()
print("✅ django.setup() OK")

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

print("=" * 70)
print("🧪 TEST WEATHER ML MODELS (RandomForest / CatBoost / XGBoost / LightGBM)")
print("=" * 70)


# ===================== TEST RANDOM FOREST =====================
def test_random_forest():
    print("\n🌲 TEST 1: Random Forest Model")
    print("-" * 50)

    try:
        from Weather_Forcast_App.Machine_learning_model.Models.Random_Forest_Model import WeatherRandomForest
    except Exception as e:
        print(f"   ❌ Cannot import Random Forest model: {e}")
        return False

    # ---- 1.1 Classification ----
    print("\n📌 1.1 Random Forest - Classification")
    X_cls, y_cls = make_classification(
        n_samples=600, n_features=10, n_informative=6, random_state=42
    )
    X_cls = pd.DataFrame(X_cls, columns=[f"feature_{i}" for i in range(10)])

    rf_clf = WeatherRandomForest(task_type="classification", n_estimators=80)
    result = rf_clf.train(X_cls, y_cls, validation_split=0.2, verbose=False)

    print(f"   ✅ Training: {result.success}")
    if not result.success:
        print(f"   ❌ Message: {result.message}")
        return False

    print(f"   📊 Accuracy: {result.metrics['accuracy']:.4f}")
    print(f"   📊 F1 Score: {result.metrics['f1_score']:.4f}")

    pred_result = rf_clf.predict(X_cls.iloc[:5])
    print(f"   🔮 Predictions: {pred_result.predictions[:5]}")

    pred_proba = rf_clf.predict(X_cls.iloc[:3], return_proba=True)
    if pred_proba.probabilities is not None:
        print(f"   📈 Probabilities shape: {pred_proba.probabilities.shape}")

    # ---- 1.2 Regression ----
    print("\n📌 1.2 Random Forest - Regression")
    X_reg, y_reg = make_regression(
        n_samples=600, n_features=12, noise=12, random_state=42
    )
    X_reg = pd.DataFrame(X_reg, columns=[f"feature_{i}" for i in range(12)])

    rf_reg = WeatherRandomForest(task_type="regression", n_estimators=120)
    result = rf_reg.train(X_reg, y_reg, validation_split=0.2, verbose=False)

    print(f"   ✅ Training: {result.success}")
    if not result.success:
        print(f"   ❌ Message: {result.message}")
        return False

    print(f"   📊 R2 Score: {result.metrics['r2_score']:.4f}")
    print(f"   📊 RMSE:     {result.metrics['rmse']:.4f}")
    print(f"   📊 MAE:      {result.metrics['mae']:.4f}")

    importances = rf_reg.get_feature_importance(top_n=3)
    print(f"   🔝 Top 3 Features: {list(importances.keys())}")

    # ---- 1.3 Cross validation ----
    print("\n📌 1.3 Random Forest - Cross Validation")
    cv_result = rf_reg.cross_validate(X_reg, y_reg, cv=3)
    print(f"   📊 CV Mean Score: {cv_result['mean_score']:.4f}")
    print(f"   📊 CV Std Score:  {cv_result['std_score']:.4f}")

    # ---- 1.4 Save/Load ----
    print("\n📌 1.4 Random Forest - Save/Load")
    filepath = rf_reg.save()
    print(f"   💾 Saved to: {filepath}")

    rf_loaded = WeatherRandomForest.load(filepath)
    print(f"   📂 Loaded is_trained: {rf_loaded.is_trained}")

    reload_preds = rf_loaded.predict(X_reg.iloc[:3]).predictions
    print(f"   🔁 Reload predictions: {reload_preds}")

    os.remove(filepath)
    print("   🗑️ Cleaned up test file")

    print("\n✅ Random Forest tests PASSED!")
    return True


# ===================== TEST CATBOOST =====================
def test_catboost():
    print("\n🐱 TEST 2: CatBoost Model")
    print("-" * 50)

    try:
        from Weather_Forcast_App.Machine_learning_model.Models.CatBoost_Model import WeatherCatBoost, CATBOOST_AVAILABLE
    except Exception as e:
        print(f"   ❌ Cannot import CatBoost model: {e}")
        return False

    if not CATBOOST_AVAILABLE:
        print("   ⚠️ CatBoost not installed -> SKIP")
        return True

    try:
        # ---- 2.1 Classification with categorical ----
        print("\n📌 2.1 CatBoost - Classification with Categorical Features")
        np.random.seed(42)
        n = 600
        data = {
            "temperature": np.random.uniform(15, 35, n),
            "humidity": np.random.uniform(40, 95, n),
            "wind_speed": np.random.uniform(0, 30, n),
            "season": np.random.choice(["Spring", "Summer", "Fall", "Winter"], n),
            "location": np.random.choice(["Hanoi", "HCMC", "Danang"], n),
        }
        X_cat = pd.DataFrame(data)

        # target: Sunny/Cloudy/Rainy
        condition = (X_cat["temperature"] > 28).astype(int) + (X_cat["humidity"] > 70).astype(int)
        y_cat = np.where(condition == 0, "Sunny", np.where(condition == 1, "Cloudy", "Rainy"))

        cb_clf = WeatherCatBoost(
            task_type="classification",
            loss_function="MultiClass",
            iterations=120,
            depth=5,
            verbose=0
        )

        result = cb_clf.train(
            X_cat, y_cat,
            cat_features=["season", "location"],
            validation_split=0.2,
            verbose=False
        )

        print(f"   ✅ Training: {result.success}")
        if not result.success:
            print(f"   ❌ Message: {result.message}")
            return False

        print(f"   📊 Accuracy: {result.metrics['accuracy']:.4f}")
        print(f"   📊 F1 Score: {result.metrics['f1_score']:.4f}")
        print(f"   ⏱️ Training time: {result.training_time:.2f}s")

        pred_result = cb_clf.predict(X_cat.iloc[:5])
        print(f"   🔮 Predictions: {pred_result.predictions[:5]}")

        # ---- 2.2 Regression ----
        print("\n📌 2.2 CatBoost - Regression")
        X_reg = X_cat.drop(columns=["temperature"])
        y_reg = X_cat["temperature"]

        cb_reg = WeatherCatBoost(
            task_type="regression",
            loss_function="RMSE",
            iterations=120,
            depth=5,
            verbose=0
        )

        result = cb_reg.train(
            X_reg, y_reg,
            cat_features=["season", "location"],
            validation_split=0.2,
            verbose=False
        )

        print(f"   ✅ Training: {result.success}")
        if not result.success:
            print(f"   ❌ Message: {result.message}")
            return False

        print(f"   📊 R2 Score: {result.metrics['r2_score']:.4f}")
        print(f"   📊 RMSE:     {result.metrics['rmse']:.4f}")
        print(f"   📊 MAE:      {result.metrics['mae']:.4f}")

        importances = cb_reg.get_feature_importance(top_n=3)
        print(f"   🔝 Top 3 Features: {list(importances.keys())}")

        # ---- 2.3 Save/Load ----
        print("\n📌 2.3 CatBoost - Save/Load")
        filepath = cb_reg.save()
        metadata_path = filepath.replace(".cbm", "_metadata.json")
        print(f"   💾 Saved to: {filepath}")

        cb_loaded = WeatherCatBoost.load(filepath)
        print(f"   📂 Loaded is_trained: {cb_loaded.is_trained}")

        # cleanup
        os.remove(filepath)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        print("   🗑️ Cleaned up test files")

        print("\n✅ CatBoost tests PASSED!")
        return True

    except Exception as e:
        print(f"   ❌ CatBoost test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ===================== TEST XGBOOST =====================
def test_xgboost():
    print("\n🚀 TEST 3: XGBoost Model (Classification + Regression)")
    print("-" * 50)

    try:
        # Import theo kiểu “chịu mọi phiên bản”
        import Weather_Forcast_App.Machine_learning_model.Models.XGBoost_Model as xgb_mod

        WeatherXGBoostModel = getattr(xgb_mod, "WeatherXGBoostModel", None) or getattr(xgb_mod, "WeatherXGBoost", None)
        if WeatherXGBoostModel is None:
            raise ImportError("Không tìm thấy class WeatherXGBoostModel / WeatherXGBoost trong XGBoost_Model.py")

        XGBOOST_AVAILABLE = getattr(xgb_mod, "XGBOOST_AVAILABLE", True)

        if not XGBOOST_AVAILABLE:
            print("   ⚠️ XGBoost not installed -> SKIP")
            return True

        # ===================== 3.1 Classification =====================
        print("\n📌 3.1 XGBoost - Classification")
        X_cls, y_cls = make_classification(
            n_samples=600, n_features=12, n_informative=6, random_state=42
        )
        X_cls = pd.DataFrame(X_cls, columns=[f"feature_{i}" for i in range(X_cls.shape[1])])

        clf = WeatherXGBoostModel(task_type="classification")
        res = clf.train(X_cls, y_cls, val_size=0.2, shuffle=True, stratify=True, verbose=False)

        print(f"   ✅ Training: {res.success}")
        if not res.success:
            print(f"   ❌ Message: {res.message}")
            return False

        print(f"   📊 Accuracy: {res.metrics.get('accuracy', 0):.4f}")
        print(f"   📊 F1(macro): {res.metrics.get('f1_macro', 0):.4f}")

        pred = clf.predict(X_cls.iloc[:5])
        print(f"   🔮 Predictions: {pred.predictions[:5]}")

        proba = clf.predict(X_cls.iloc[:3], return_proba=True).probabilities
        print(f"   📈 Probabilities shape: {None if proba is None else proba.shape}")

        # ===================== 3.2 Regression =====================
        print("\n📌 3.2 XGBoost - Regression")
        X_reg, y_reg = make_regression(
            n_samples=600, n_features=12, noise=12, random_state=42
        )
        X_reg = pd.DataFrame(X_reg, columns=[f"feature_{i}" for i in range(X_reg.shape[1])])

        reg = WeatherXGBoostModel(task_type="regression")
        res = reg.train(X_reg, y_reg, val_size=0.2, shuffle=False, verbose=False)

        print(f"   ✅ Training: {res.success}")
        if not res.success:
            print(f"   ❌ Message: {res.message}")
            return False

        print(f"   📊 R2:   {res.metrics.get('r2', 0):.4f}")
        print(f"   📊 RMSE: {res.metrics.get('rmse', 0):.4f}")
        print(f"   📊 MAE:  {res.metrics.get('mae', 0):.4f}")

        importances = reg.get_feature_importance(top_k=3)
        print(f"   🔝 Top 3 Features: {list(importances.keys())}")

        # ===================== 3.3 CV =====================
        print("\n📌 3.3 XGBoost - Cross Validation")
        cv_res = reg.cross_validate(X_reg, y_reg, cv=3, scoring="r2")
        print(f"   📊 CV Mean: {cv_res['mean']:.4f}")
        print(f"   📊 CV Std:  {cv_res['std']:.4f}")

        # ===================== 3.4 Save/Load =====================
        print("\n📌 3.4 XGBoost - Save/Load")
        filepath = reg.save()
        print(f"   💾 Saved to: {filepath}")

        loaded = WeatherXGBoostModel.load(filepath)
        print(f"   📂 Loaded is_trained: {loaded.is_trained}")

        re_pred = loaded.predict(X_reg.iloc[:3]).predictions
        print(f"   🔁 Reload predictions: {re_pred}")

        os.remove(filepath)
        print("   🗑️ Cleaned up test file")

        print("\n✅ XGBoost tests PASSED!")
        return True

    except Exception as e:
        print(f"   ❌ XGBoost test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ===================== TEST LIGHTGBM =====================
def test_lightgbm():
    print("\n💡 TEST 4: LightGBM Model (Classification + Regression)")
    print("-" * 50)

    try:
        from Weather_Forcast_App.Machine_learning_model.Models.LightGBM_Model import WeatherLightGBM
    except Exception as e:
        print(f"   ❌ Cannot import LightGBM model: {e}")
        return False

    try:
        np.random.seed(42)
        n = 800
        X = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),  # dùng "h" để tránh warning
            "temp": np.random.uniform(10, 40, n),
            "humidity": np.random.uniform(30, 100, n),
            "wind_speed": np.random.uniform(0, 25, n),
            "season": np.random.choice(["Spring", "Summer", "Fall", "Winter"], n),
            "location": np.random.choice(["Hanoi", "HCMC", "Danang"], n),
        })

        # ---- 4.1 Classification ----
        print("\n📌 4.1 LightGBM - Classification (datetime + categorical)")
        score = (X["temp"] > 30).astype(int) + (X["humidity"] > 75).astype(int)
        y_cls = np.where(score == 0, "Sunny", np.where(score == 1, "Cloudy", "Rainy"))

        clf = WeatherLightGBM(
            task_type="classification",
            params={"n_estimators": 800, "learning_rate": 0.05, "num_leaves": 63},
        )
        res = clf.train(
            X, y_cls,
            target_name="weather_type",
            cat_features=["season", "location"],
            datetime_cols=["timestamp"],
            val_size=0.2,
            shuffle=True,
            stratify=True,
            early_stopping_rounds=50,
            verbose_eval=200,
        )
        print(f"   ✅ Training: {res.success}")
        if not res.success:
            print(f"   ❌ Message: {res.message}")
            return False

        print(f"   📊 Accuracy: {res.metrics.get('accuracy', None)}")
        pred = clf.predict(X.iloc[:5])
        print(f"   🔮 Predictions: {pred.predictions[:5]}")
        proba = clf.predict(X.iloc[:3], return_proba=True)
        if proba.probabilities is not None:
            print(f"   📈 Probabilities shape: {proba.probabilities.shape}")

        # ---- 4.2 Regression ----
        print("\n📌 4.2 LightGBM - Regression (datetime + categorical)")
        noise = np.random.normal(0, 3, n)
        y_reg = np.clip(0.35 * X["humidity"].values - 0.15 * X["temp"].values + noise, 0, None)

        reg = WeatherLightGBM(
            task_type="regression",
            params={"n_estimators": 800, "learning_rate": 0.05, "num_leaves": 63},
        )
        res2 = reg.train(
            X, y_reg,
            target_name="rain_mm",
            cat_features=["season", "location"],
            datetime_cols=["timestamp"],
            val_size=0.2,
            shuffle=False,   # time-series style
            early_stopping_rounds=50,
            verbose_eval=200,
        )
        print(f"   ✅ Training: {res2.success}")
        if not res2.success:
            print(f"   ❌ Message: {res2.message}")
            return False

        print(f"   📊 RMSE: {res2.metrics.get('rmse', None)}")
        pred2 = reg.predict(X.iloc[:5])
        print(f"   🔮 Pred rain_mm: {pred2.predictions[:5]}")

        # Feature importance
        fi = reg.get_feature_importance(top_k=5)
        print(f"   🔝 Top features: {list(fi.keys())[:5]}")

        # Save/Load
        print("\n📌 4.3 LightGBM - Save/Load")
        p = reg.save()
        print(f"   💾 Saved to: {p}")
        reg2 = WeatherLightGBM.load(p)
        print(f"   📂 Loaded OK, status: {reg2.status.value}")

        os.remove(p)
        print("   🗑️ Cleaned up LightGBM file")

        print("\n✅ LightGBM tests PASSED!")
        return True

    except Exception as e:
        print(f"   ❌ LightGBM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ===================== MAIN =====================
if __name__ == "__main__":
    rf_passed = test_random_forest()
    cb_passed = test_catboost()
    xgb_passed = test_xgboost()
    lgb_passed = test_lightgbm()

    print("\n" + "=" * 70)
    print("📋 TEST SUMMARY")
    print("=" * 70)
    print(f"   🌲 Random Forest: {'✅ PASSED' if rf_passed else '❌ FAILED'}")
    print(f"   🐱 CatBoost:      {'✅ PASSED' if cb_passed else '❌ FAILED'}")
    print(f"   🚀 XGBoost:       {'✅ PASSED' if xgb_passed else '❌ FAILED'}")
    print(f"   💡 LightGBM:      {'✅ PASSED' if lgb_passed else '❌ FAILED'}")
    print("=" * 70)

    if rf_passed and cb_passed and xgb_passed and lgb_passed:
        print("\n🎉 All tests passed successfully!")
    else:
        print("\n⚠️ Some tests failed! Check logs above.")
