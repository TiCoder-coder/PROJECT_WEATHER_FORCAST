# ----------------------------- TEST FILE CHO TUNING MODULE -----------------------------------------------------------
"""
test_tuning.py - Bộ test toàn diện cho tuning.py

Bao gồm:
    1. Unit tests - Test các function riêng lẻ
    2. Integration tests - Test pipeline tuning đầu cuối
    3. Manual tests - Test nhanh với sample data
    4. Benchmark tests - Đo lường hiệu năng

Cách chạy:
    # Chạy tất cả tests
    pytest test_tuning.py -v
    
    # Chạy riêng từng loại test
    pytest test_tuning.py::TestUnitTests -v
    pytest test_tuning.py::TestIntegration -v
    pytest test_tuning.py::TestBenchmark -v
    
    # Chạy với coverage
    pytest test_tuning.py --cov=tuning --cov-report=html
    
    # Chạy manual tests
    python test_tuning.py --manual
"""

import sys
import json
import time
import logging
from pathlib import Path
from typing import Tuple
import argparse

import numpy as np
import pandas as pd
import pytest

# =====================================================================================
# (1) FIX IMPORT PATH
# =====================================================================================
THIS_FILE = Path(__file__).resolve()
project_root = THIS_FILE
for _ in range(6):
    if (project_root / "Weather_Forcast_App").exists():
        break
    project_root = project_root.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# =====================================================================================
# (2) IMPORT MODULES
# =====================================================================================
from Weather_Forcast_App.Machine_learning_model.trainning.tuning import (
    HyperparameterTuner,
    TuningConfig,
    TuningResult,
    TuningMethod,
    _load_config,
    _ensure_dir,
    _now_tag,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =====================================================================================
# (3) FIXTURES - DỮ LIỆU SAMPLE CHO TEST
# =====================================================================================

@pytest.fixture
def sample_data() -> Tuple[np.ndarray, np.ndarray]:
    """Tạo sample data cho test."""
    np.random.seed(42)
    n_samples = 200
    n_features = 15
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    
    return X, y


@pytest.fixture
def small_sample_data() -> Tuple[np.ndarray, np.ndarray]:
    """Tạo sample data nhỏ cho test nhanh."""
    np.random.seed(42)
    n_samples = 50
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    
    return X, y


@pytest.fixture
def tuning_config(tmp_path) -> TuningConfig:
    """Tạo config cho test."""
    return TuningConfig(
        model_type="xgboost",
        tuning_method=TuningMethod.RANDOM_SEARCH,
        n_trials=5,
        cv_folds=3,
        n_jobs=1,
        random_state=42,
        metric="neg_mean_squared_error",
        verbose=0,
        save_results=True,
        output_dir=tmp_path,
    )


@pytest.fixture
def output_dir(tmp_path) -> Path:
    """Tạo output directory cho test."""
    return tmp_path


# =====================================================================================
# (4) UNIT TESTS
# =====================================================================================

class TestUnitTests:
    """Unit tests cho các function riêng lẻ."""
    
    def test_ensure_dir(self, tmp_path):
        """Test hàm _ensure_dir."""
        test_dir = tmp_path / "test_dir"
        assert not test_dir.exists()
        
        _ensure_dir(test_dir)
        assert test_dir.exists()
    
    def test_now_tag_format(self):
        """Test hàm _now_tag() format."""
        tag = _now_tag()
        assert isinstance(tag, str)
        assert len(tag) > 0
        assert "_" in tag
    
    def test_tuning_config_defaults(self):
        """Test TuningConfig defaults."""
        config = TuningConfig()
        assert config.model_type == "xgboost"
        assert config.n_trials == 50
        assert config.cv_folds == 5
        assert config.random_state == 42
    
    def test_tuning_config_custom(self):
        """Test TuningConfig với custom values."""
        config = TuningConfig(
            model_type="lightgbm",
            n_trials=100,
            cv_folds=10,
        )
        assert config.model_type == "lightgbm"
        assert config.n_trials == 100
        assert config.cv_folds == 10
    
    def test_tuning_result_creation(self):
        """Test TuningResult creation."""
        result = TuningResult(
            model_type="xgboost",
            best_params={"learning_rate": 0.05, "max_depth": 6},
            best_score=0.95,
            cv_mean=0.94,
            cv_std=0.02,
            n_iter=50,
            tuning_time=120.5,
            method="random_search",
        )
        
        assert result.model_type == "xgboost"
        assert result.best_score == 0.95
        assert result.n_iter == 50
    
    def test_tuning_result_to_dict(self):
        """Test TuningResult.to_dict()."""
        result = TuningResult(
            model_type="xgboost",
            best_params={"learning_rate": 0.05},
            best_score=0.95,
            cv_mean=0.94,
            cv_std=0.02,
            n_iter=50,
            tuning_time=120.5,
            method="random_search",
        )
        
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["model_type"] == "xgboost"
        assert result_dict["best_score"] == 0.95
    
    def test_tuning_result_to_json(self, tmp_path):
        """Test TuningResult.to_json()."""
        result = TuningResult(
            model_type="xgboost",
            best_params={"learning_rate": 0.05},
            best_score=0.95,
            cv_mean=0.94,
            cv_std=0.02,
            n_iter=50,
            tuning_time=120.5,
            method="random_search",
        )
        
        output_file = tmp_path / "test_result.json"
        result.to_json(output_file)
        
        assert output_file.exists()
        
        # Verify saved content
        with open(output_file, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data["model_type"] == "xgboost"
        assert saved_data["best_score"] == 0.95


# =====================================================================================
# (5) INTEGRATION TESTS
# =====================================================================================

class TestIntegration:
    """Integration tests cho pipeline tuning."""
    
    @pytest.mark.slow
    def test_random_search_xgboost(self, small_sample_data, tuning_config):
        """Test RandomSearch tuning với XGBoost."""
        X, y = small_sample_data
        
        config = TuningConfig(
            model_type="xgboost",
            tuning_method=TuningMethod.RANDOM_SEARCH,
            n_trials=3,
            cv_folds=2,
            n_jobs=1,
            verbose=0,
            save_results=False,
        )
        
        tuner = HyperparameterTuner(config)
        result = tuner.tune(X, y)
        
        assert result is not None
        assert result.model_type == "xgboost"
        assert result.method == "random_search"
        assert result.n_iter == 3
        assert "learning_rate" in result.best_params or "max_depth" in result.best_params
    
    @pytest.mark.slow
    def test_tuner_initialization(self, tuning_config):
        """Test HyperparameterTuner initialization."""
        tuner = HyperparameterTuner(tuning_config)
        
        assert tuner.config is not None
        assert tuner.config.model_type == "xgboost"
        assert tuner.best_params is None
        assert tuner.search_result is None
    
    @pytest.mark.slow
    def test_tune_with_results_saving(self, small_sample_data, tuning_config, tmp_path):
        """Test tuning with results saving."""
        X, y = small_sample_data
        
        config = TuningConfig(
            model_type="xgboost",
            tuning_method=TuningMethod.RANDOM_SEARCH,
            n_trials=2,
            cv_folds=2,
            n_jobs=1,
            verbose=0,
            save_results=True,
            output_dir=tmp_path,
        )
        
        tuner = HyperparameterTuner(config)
        result = tuner.tune(X, y)
        
        # Check if result file was created
        result_files = list(tmp_path.glob("tuning_result_*.json"))
        assert len(result_files) > 0
        
        # Verify saved result
        with open(result_files[0], 'r') as f:
            saved_result = json.load(f)
        
        assert saved_result["model_type"] == "xgboost"


# =====================================================================================
# (6) MANUAL TESTS (CHỈ CHẠY KHI GỌI TRỰC TIẾP)
# =====================================================================================

class ManualTests:
    """Manual tests - có thể chạy ngoài pytest."""
    
    @staticmethod
    def test_quick_tune():
        """Quick test - tuning nhanh khỏi với RandomSearch."""
        print("\n" + "="*60)
        print("MANUAL TEST 1: Quick RandomSearch Tuning")
        print("="*60)
        
        # Tạo sample data
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        
        print(f"Data shape: X={X.shape}, y={y.shape}")
        
        # Tuning config
        config = TuningConfig(
            model_type="xgboost",
            tuning_method=TuningMethod.RANDOM_SEARCH,
            n_trials=3,
            cv_folds=2,
            n_jobs=1,
            verbose=1,
            save_results=False,
        )
        
        print(f"Tuning config: {config}")
        
        # Chạy tuning
        tuner = HyperparameterTuner(config)
        result = tuner.tune(X, y)
        
        print("\n✅ Tuning completed!")
        print(f"Best params: {result.best_params}")
        print(f"Best score: {result.best_score:.6f}")
        print(f"Time: {result.tuning_time:.2f}s")
    
    @staticmethod
    def test_compare_methods():
        """So sánh các tuning methods."""
        print("\n" + "="*60)
        print("MANUAL TEST 2: So sánh Tuning Methods")
        print("="*60)
        
        # Tạo sample data
        np.random.seed(42)
        X = np.random.randn(150, 12)
        y = np.random.randn(150)
        
        print(f"Data shape: X={X.shape}, y={y.shape}")
        print()
        
        methods = [
            TuningMethod.RANDOM_SEARCH,
            TuningMethod.GRID_SEARCH,
        ]
        
        results = {}
        
        for method in methods:
            print(f"Testing {method}...")
            
            config = TuningConfig(
                model_type="xgboost",
                tuning_method=method,
                n_trials=3,
                cv_folds=2,
                n_jobs=1,
                verbose=0,
                save_results=False,
            )
            
            tuner = HyperparameterTuner(config)
            start_time = time.time()
            result = tuner.tune(X, y)
            elapsed = time.time() - start_time
            
            results[method] = {
                "best_score": result.best_score,
                "time": elapsed,
                "best_params": result.best_params,
            }
            
            print(f"  ✅ Score: {result.best_score:.6f}, Time: {elapsed:.2f}s")
            print()
        
        # So sánh
        print("-" * 60)
        print("COMPARISON:")
        print("-" * 60)
        for method, data in results.items():
            print(f"{method}:")
            print(f"  Score: {data['best_score']:.6f}")
            print(f"  Time: {data['time']:.2f}s")
            print()
    
    @staticmethod
    def test_different_models():
        """Test với các models khác nhau."""
        print("\n" + "="*60)
        print("MANUAL TEST 3: Test Các Models Khác Nhau")
        print("="*60)
        
        # Tạo sample data
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        
        print(f"Data shape: X={X.shape}, y={y.shape}")
        print()
        
        models = ["xgboost", "random_forest"]
        
        for model_type in models:
            print(f"Testing {model_type}...")
            
            try:
                config = TuningConfig(
                    model_type=model_type,
                    tuning_method=TuningMethod.RANDOM_SEARCH,
                    n_trials=2,
                    cv_folds=2,
                    n_jobs=1,
                    verbose=0,
                    save_results=False,
                )
                
                tuner = HyperparameterTuner(config)
                result = tuner.tune(X, y)
                
                print(f"  ✅ {model_type} - Score: {result.best_score:.6f}")
                print(f"     Best params keys: {list(result.best_params.keys())}")
                print()
            
            except Exception as e:
                print(f"  ❌ {model_type} failed: {e}")
                print()
    
    @staticmethod
    def test_data_variations():
        """Test với dữ liệu có kích thước khác nhau."""
        print("\n" + "="*60)
        print("MANUAL TEST 4: Test Với Dữ Liệu Khác Nhau")
        print("="*60)
        
        np.random.seed(42)
        
        sizes = [
            (50, 5, "Nhỏ"),
            (200, 15, "Vừa"),
            (500, 30, "Lớn"),
        ]
        
        for n_samples, n_features, label in sizes:
            print(f"\nTesting with {label} data: ({n_samples} samples, {n_features} features)")
            
            X = np.random.randn(n_samples, n_features)
            y = np.random.randn(n_samples)
            
            config = TuningConfig(
                model_type="xgboost",
                tuning_method=TuningMethod.RANDOM_SEARCH,
                n_trials=2,
                cv_folds=2,
                n_jobs=1,
                verbose=0,
                save_results=False,
            )
            
            tuner = HyperparameterTuner(config)
            start_time = time.time()
            result = tuner.tune(X, y)
            elapsed = time.time() - start_time
            
            print(f"  ✅ Score: {result.best_score:.6f}, Time: {elapsed:.2f}s")


# =====================================================================================
# (7) BENCHMARK TESTS
# =====================================================================================

class TestBenchmark:
    """Benchmark tests - đo lường hiệu năng."""
    
    @pytest.mark.benchmark
    def test_random_search_speed(self, small_sample_data, benchmark):
        """Benchmark RandomSearch speed."""
        X, y = small_sample_data
        
        config = TuningConfig(
            model_type="xgboost",
            tuning_method=TuningMethod.RANDOM_SEARCH,
            n_trials=2,
            cv_folds=2,
            n_jobs=1,
            verbose=0,
            save_results=False,
        )
        
        def run_tuning():
            tuner = HyperparameterTuner(config)
            result = tuner.tune(X, y)
            return result
        
        result = benchmark(run_tuning)
        assert result is not None
    
    @pytest.mark.benchmark
    def test_config_creation_speed(self, benchmark):
        """Benchmark TuningConfig creation speed."""
        def create_config():
            return TuningConfig(
                model_type="xgboost",
                n_trials=50,
                cv_folds=5,
            )
        
        result = benchmark(create_config)
        assert result is not None


# =====================================================================================
# (8) HELPER SCRIPT - RUN MANUAL TESTS
# =====================================================================================

def run_manual_tests():
    """Chạy tất cả manual tests."""
    manual = ManualTests()
    
    try:
        manual.test_quick_tune()
    except Exception as e:
        print(f"❌ Test 1 failed: {e}\n")
    
    try:
        manual.test_compare_methods()
    except Exception as e:
        print(f"❌ Test 2 failed: {e}\n")
    
    try:
        manual.test_different_models()
    except Exception as e:
        print(f"❌ Test 3 failed: {e}\n")
    
    try:
        manual.test_data_variations()
    except Exception as e:
        print(f"❌ Test 4 failed: {e}\n")
    
    print("\n" + "="*60)
    print("✅ All manual tests completed!")
    print("="*60)


# =====================================================================================
# (9) MAIN
# =====================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test suite for tuning.py")
    parser.add_argument("--manual", action="store_true", help="Run manual tests")
    parser.add_argument("--test", type=str, default="all", help="Specific test to run")
    
    args = parser.parse_args()
    
    if args.manual:
        run_manual_tests()
    else:
        # Pytest
        print("Chạy pytest...")
        pytest.main([__file__, "-v"])