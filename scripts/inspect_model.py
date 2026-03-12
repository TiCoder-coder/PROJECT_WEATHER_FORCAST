"""One-time diagnostic script to inspect Model.pkl state."""
import joblib
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "WeatherForcast.settings")

model = joblib.load('Weather_Forcast_App/Machine_learning_artifacts/latest/Model.pkl')

print("=== Ensemble State ===")
print(f"is_trained: {model.is_trained}")
print(f"models count: {len(model.models)}")
print(f"base_models_cfg count: {len(model.base_models_cfg)}")
print()

for i, m in enumerate(model.models):
    print(f"--- Base model {i} ---")
    print(f"  Wrapper type: {type(m).__name__}")
    inner = getattr(m, 'model', None)
    print(f"  Inner model type: {type(inner).__name__ if inner is not None else 'None'}")
    status = getattr(m, 'status', 'N/A')
    print(f"  Status: {status}")
    
    # Check if inner model is fitted
    if inner is not None:
        # sklearn models have n_features_in_ after fit
        n_feat = getattr(inner, 'n_features_in_', None)
        print(f"  n_features_in_: {n_feat}")
        
        # Check for tree-based fitted attributes
        has_estimators = hasattr(inner, 'estimators_')
        print(f"  has estimators_ (fitted): {has_estimators}")
        
        # For CatBoost
        if hasattr(inner, 'is_fitted'):
            try:
                print(f"  is_fitted(): {inner.is_fitted()}")
            except Exception as e:
                print(f"  is_fitted() error: {e}")
        
        # For XGBoost
        if hasattr(inner, 'get_booster'):
            try:
                booster = inner.get_booster()
                print(f"  has booster: {booster is not None}")
            except Exception as e:
                print(f"  get_booster() error: {e}")

print()
print("=== Config ===")
for i, cfg in enumerate(model.base_models_cfg):
    print(f"  cfg[{i}]: type={cfg.get('type', '?')}, params_keys={list(cfg.get('params', {}).keys())}")
