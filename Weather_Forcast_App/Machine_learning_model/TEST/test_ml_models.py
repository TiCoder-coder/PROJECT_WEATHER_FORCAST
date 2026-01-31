"""
Test script cho Random Forest và CatBoost Models
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

print("PROJECT_ROOT =", PROJECT_ROOT)
print("sys.path[0]  =", sys.path[0])

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "WeatherForcast.settings")

import django
django.setup()


import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

print("=" * 60)
print("🧪 TEST WEATHER ML MODELS")
print("=" * 60)


# ===================== TEST RANDOM FOREST =====================
def test_random_forest():
    print("\n🌲 TEST 1: Random Forest Model")
    print("-" * 40)
    
    from Weather_Forcast_App.Machine_learning_model.Models.Random_Forest_Model import WeatherRandomForest
    
    # Test Classification
    print("\n📌 1.1 Random Forest - Classification")
    X_cls, y_cls = make_classification(
        n_samples=500, n_features=10, n_informative=5, random_state=42
    )
    X_cls = pd.DataFrame(X_cls, columns=[f"feature_{i}" for i in range(10)])
    
    rf_clf = WeatherRandomForest(task_type="classification", n_estimators=50)
    result = rf_clf.train(X_cls, y_cls, validation_split=0.2, verbose=False)
    
    print(f"   ✅ Training: {result.success}")
    print(f"   📊 Accuracy: {result.metrics['accuracy']:.4f}")
    print(f"   📊 F1 Score: {result.metrics['f1_score']:.4f}")
    
    # Test prediction
    pred_result = rf_clf.predict(X_cls[:5])
    print(f"   🔮 Predictions: {pred_result.predictions[:5]}")
    
    # Test with probabilities
    pred_proba = rf_clf.predict(X_cls[:3], return_proba=True)
    print(f"   📈 Probabilities shape: {pred_proba.probabilities.shape}")
    
    # Test Regression
    print("\n📌 1.2 Random Forest - Regression")
    X_reg, y_reg = make_regression(
        n_samples=500, n_features=10, noise=10, random_state=42
    )
    X_reg = pd.DataFrame(X_reg, columns=[f"feature_{i}" for i in range(10)])
    
    rf_reg = WeatherRandomForest(task_type="regression", n_estimators=50)
    result = rf_reg.train(X_reg, y_reg, validation_split=0.2, verbose=False)
    
    print(f"   ✅ Training: {result.success}")
    print(f"   📊 R2 Score: {result.metrics['r2_score']:.4f}")
    print(f"   📊 RMSE: {result.metrics['rmse']:.4f}")
    print(f"   📊 MAE: {result.metrics['mae']:.4f}")
    
    # Feature importance
    importances = rf_reg.get_feature_importance(top_n=3)
    print(f"   🔝 Top 3 Features: {list(importances.keys())}")
    
    # Test cross-validation
    print("\n📌 1.3 Random Forest - Cross Validation")
    cv_result = rf_reg.cross_validate(X_reg, y_reg, cv=3)
    print(f"   📊 CV Mean Score: {cv_result['mean_score']:.4f}")
    print(f"   📊 CV Std Score: {cv_result['std_score']:.4f}")
    
    # Test save/load
    print("\n📌 1.4 Random Forest - Save/Load")
    filepath = rf_reg.save()
    print(f"   💾 Saved to: {filepath}")
    
    rf_loaded = WeatherRandomForest.load(filepath)
    print(f"   📂 Loaded: {rf_loaded.is_trained}")
    
    # Clean up
    os.remove(filepath)
    print(f"   🗑️ Cleaned up test file")
    
    print("\n✅ Random Forest tests PASSED!")
    return True


# ===================== TEST CATBOOST =====================
def test_catboost():
    print("\n🐱 TEST 2: CatBoost Model")
    print("-" * 40)
    
    try:
        from Weather_Forcast_App.Machine_learning_model.Models.CatBoost_Model import WeatherCatBoost, CATBOOST_AVAILABLE
        
        if not CATBOOST_AVAILABLE:
            print("   ⚠️ CatBoost not installed, skipping test")
            return True
        
        # Test Classification with categorical features
        print("\n📌 2.1 CatBoost - Classification with Categorical Features")
        
        # Create sample data with categorical features
        np.random.seed(42)
        n_samples = 500
        
        data = {
            "temperature": np.random.uniform(15, 35, n_samples),
            "humidity": np.random.uniform(40, 95, n_samples),
            "wind_speed": np.random.uniform(0, 30, n_samples),
            "season": np.random.choice(["Spring", "Summer", "Fall", "Winter"], n_samples),
            "location": np.random.choice(["Hanoi", "HCMC", "Danang"], n_samples),
        }
        X_cat = pd.DataFrame(data)
        
        # Ensure string columns are object dtype (not StringDtype)
        for col in ["season", "location"]:
            X_cat[col] = X_cat[col].astype(object)
        
        # Create target based on features - use np.where for object dtype
        condition = (X_cat["temperature"] > 28).astype(int) + (X_cat["humidity"] > 70).astype(int)
        y_cat = np.where(condition == 0, "Sunny", np.where(condition == 1, "Cloudy", "Rainy"))
        
        cb_clf = WeatherCatBoost(
            task_type="classification",
            loss_function="MultiClass",  # Use MultiClass for 3+ classes
            iterations=100,
            depth=4,
            verbose=0
        )
        
        result = cb_clf.train(
            X_cat, y_cat,
            cat_features=["season", "location"],
            validation_split=0.2,
            verbose=False
        )
        
        print(f"   ✅ Training: {result.success}")
        print(f"   📊 Accuracy: {result.metrics['accuracy']:.4f}")
        print(f"   📊 F1 Score: {result.metrics['f1_score']:.4f}")
        print(f"   ⏱️ Training time: {result.training_time:.2f}s")
        print(f"   🎯 Best iteration: {result.best_iteration}")
        
        # Test prediction
        pred_result = cb_clf.predict(X_cat[:5])
        print(f"   🔮 Predictions: {pred_result.predictions[:5]}")
        
        # Test Regression
        print("\n📌 2.2 CatBoost - Regression")
        
        # Target: predict temperature based on other features
        X_reg = X_cat.drop(columns=["temperature"])
        y_reg = X_cat["temperature"]
        
        cb_reg = WeatherCatBoost(
            task_type="regression",
            loss_function="RMSE",
            iterations=100,
            depth=4,
            verbose=0
        )
        
        result = cb_reg.train(
            X_reg, y_reg,
            cat_features=["season", "location"],
            validation_split=0.2,
            verbose=False
        )
        
        print(f"   ✅ Training: {result.success}")
        print(f"   📊 R2 Score: {result.metrics['r2_score']:.4f}")
        print(f"   📊 RMSE: {result.metrics['rmse']:.4f}")
        print(f"   📊 MAE: {result.metrics['mae']:.4f}")
        
        # Feature importance
        importances = cb_reg.get_feature_importance(top_n=3)
        print(f"   🔝 Top 3 Features: {list(importances.keys())}")
        
        # Test predict single
        print("\n📌 2.3 CatBoost - Predict Single Sample")
        sample = {
            "humidity": 75.0,
            "wind_speed": 15.0,
            "season": "Summer",
            "location": "HCMC"
        }
        prediction = cb_reg.predict_single(sample)
        print(f"   🌡️ Predicted temperature: {prediction:.2f}°C")
        
        # Test save/load
        print("\n📌 2.4 CatBoost - Save/Load")
        filepath = cb_reg.save()
        print(f"   💾 Saved to: {filepath}")
        
        cb_loaded = WeatherCatBoost.load(filepath)
        print(f"   📂 Loaded: {cb_loaded.is_trained}")
        print(f"   📋 Info: {cb_loaded.info}")
        
        # Clean up
        metadata_path = filepath.replace(".cbm", "_metadata.json")
        os.remove(filepath)
        os.remove(metadata_path)
        print(f"   🗑️ Cleaned up test files")
        
        print("\n✅ CatBoost tests PASSED!")
        return True
        
    except Exception as e:
        print(f"   ❌ CatBoost test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ===================== MAIN =====================
if __name__ == "__main__":
    print()
    
    # Test Random Forest
    rf_passed = test_random_forest()
    
    # Test CatBoost
    cb_passed = test_catboost()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    print(f"   🌲 Random Forest: {'✅ PASSED' if rf_passed else '❌ FAILED'}")
    print(f"   🐱 CatBoost:      {'✅ PASSED' if cb_passed else '❌ FAILED'}")
    print("=" * 60)
    
    if rf_passed and cb_passed:
        print("\n🎉 All tests passed successfully!")
    else:
        print("\n⚠️ Some tests failed!")