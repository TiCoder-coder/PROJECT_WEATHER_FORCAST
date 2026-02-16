# ----------------------------- HYPERPARAMETER TUNING - CÀI ĐẶT THAM SỐ TỐI ƯU -----------------------------------------------------------
# ===============================
# File này dùng để điều chỉnh (tuning) các tham số hyperparameters cho các mô hình Machine Learning.
# Mục tiêu: Tìm ra bộ tham số tối ưu nhất cho các mô hình như XGBoost, LightGBM, RandomForest, CatBoost.
# Hỗ trợ nhiều phương pháp tuning: GridSearch, RandomSearch, Optuna.
# Sử dụng Cross-Validation để đánh giá hiệu năng.
# Lưu lại kết quả tuning và best parameters.
# Tích hợp với pipeline training.
# ===============================
# ===============================
# Ví dụ sử dụng:
# python -m Weather_Forcast_App.Machine_learning_model.trainning.tuning \
#     --config config/train_config.json \
#     --method optuna \
#     --n_trials 100
# Hoặc import trực tiếp:
# from Weather_Forcast_App.Machine_learning_model.trainning.tuning import HyperparameterTuner
# tuner = HyperparameterTuner(model_type='xgboost', tuning_method='optuna', n_trials=100)
# best_params = tuner.tune(X_train, y_train)
# ===============================


# Đảm bảo tương thích với các phiên bản Python cũ (annotations)
from __future__ import annotations


# Các thư viện chuẩn và ML cần thiết
import argparse  # Xử lý tham số dòng lệnh
import json      # Đọc/ghi file json
import logging   # Ghi log
import sys       # Xử lý sys.path
import warnings  # Ẩn cảnh báo
from dataclasses import asdict, dataclass, field  # Dùng cho dataclass
from datetime import datetime  # Xử lý thời gian
from pathlib import Path       # Xử lý đường dẫn
from typing import Any, Callable, Dict, List, Optional, Tuple, Union  # Kiểu dữ liệu

import numpy as np             # Xử lý số học
import pandas as pd            # Xử lý dataframe
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, KFold  # Tuning & CV
from sklearn.preprocessing import StandardScaler  # Chuẩn hóa dữ liệu

# Ẩn các cảnh báo không cần thiết để log sạch hơn
warnings.filterwarnings('ignore')

# =====================================================================================
# (1) IMPORT PATH
# =====================================================================================
# Đảm bảo import đúng các module trong project (thêm project_root vào sys.path)
THIS_FILE = Path(__file__).resolve()
project_root = THIS_FILE
for _ in range(5):
    if (project_root / "Weather_Forcast_App").exists():
        break
    project_root = project_root.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# =====================================================================================
# (2) IMPORT MODULES
# =====================================================================================
# Import các module custom trong project
from Weather_Forcast_App.Machine_learning_model.data.Loader import DataLoader
from Weather_Forcast_App.Machine_learning_model.data.Schema import validate_weather_dataframe
from Weather_Forcast_App.Machine_learning_model.data.Split import SplitConfig, split_dataframe
from Weather_Forcast_App.Machine_learning_model.features.Build_transfer import WeatherFeatureBuilder
from Weather_Forcast_App.Machine_learning_model.features.Transformers import WeatherTransformPipeline
from Weather_Forcast_App.Machine_learning_model.evaluation.metrics import calculate_all_metrics

# Setup logging để theo dõi quá trình tuning
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



# =====================================================================================
# (3) CONSTANTS & ENUMS
# =====================================================================================
# Định nghĩa các phương pháp tuning và search space cho từng model

class TuningMethod:
    """Phương pháp tuning hyperparameters."""
    GRID_SEARCH = "grid_search"      # Tìm toàn bộ grid
    RANDOM_SEARCH = "random_search"  # Tìm ngẫu nhiên
    OPTUNA = "optuna"                # Tìm thông minh (Optuna)
    BAYESIAN = "bayesian"            # Tìm theo Bayesian optimization (tương lai)

# Các search space cho từng model (dùng cho GridSearch/RandomSearch)
HYPERPARAMETER_SPACES = {
    "xgboost": {
        "n_estimators": [100, 200, 500, 1000],
        "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 4, 5, 6, 7, 8, 10],
        "min_child_weight": [1, 3, 5, 7],
        "subsample": [0.5, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.5, 0.7, 0.8, 0.9],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "reg_lambda": [0.5, 1.0, 1.5, 2.0],
    },
    "lightgbm": {
        "n_estimators": [100, 200, 500, 1000],
        "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.2],
        "num_leaves": [20, 31, 50, 100],
        "max_depth": [-1, 5, 7, 10],
        "min_child_samples": [10, 20, 30, 50],
        "subsample": [0.5, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.5, 0.7, 0.8, 0.9],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "reg_lambda": [0.0, 0.1, 0.5, 1.0],
    },
    "random_forest": {
        "n_estimators": [50, 100, 200, 500],
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
    },
    "catboost": {
        "iterations": [100, 200, 500, 1000],
        "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.2],
        "max_depth": [4, 6, 8, 10],
        "l2_leaf_reg": [1, 3, 5, 7, 9],
        "subsample": [0.5, 0.7, 0.8, 0.9],
        "colsample_bylevel": [0.5, 0.7, 0.8, 0.9],
    },
}

# Các range cho Optuna (dùng cho Optuna trial)
OPTUNA_SEARCH_SPACES = {
    "xgboost": {
        "n_estimators": (100, 1000),
        "learning_rate": (0.001, 0.2),
        "max_depth": (3, 10),
        "min_child_weight": (1, 7),
        "subsample": (0.5, 1.0),
        "colsample_bytree": (0.5, 1.0),
        "reg_alpha": (0.0, 2.0),
        "reg_lambda": (0.0, 2.0),
    },
    "lightgbm": {
        "n_estimators": (100, 1000),
        "learning_rate": (0.001, 0.2),
        "num_leaves": (20, 150),
        "max_depth": (-1, 15),
        "min_child_samples": (10, 100),
        "subsample": (0.5, 1.0),
        "colsample_bytree": (0.5, 1.0),
        "reg_alpha": (0.0, 2.0),
        "reg_lambda": (0.0, 2.0),
    },
    "random_forest": {
        "n_estimators": (50, 500),
        "max_depth": (5, 30),
        "min_samples_split": (2, 20),
        "min_samples_leaf": (1, 10),
    },
    "catboost": {
        "iterations": (100, 1000),
        "learning_rate": (0.001, 0.2),
        "max_depth": (4, 10),
        "l2_leaf_reg": (1, 9),
        "subsample": (0.5, 1.0),
    },
}


# =====================================================================================
# (4) DATA CLASSES
# =====================================================================================
# Định nghĩa các dataclass để lưu kết quả tuning và cấu hình tuning

@dataclass
class TuningResult:
    """
    Kết quả tuning hyperparameters.
    - model_type: Loại model (xgboost, lightgbm, ...)
    - best_params: Bộ tham số tốt nhất tìm được
    - best_score: Điểm số tốt nhất
    - cv_mean: Trung bình điểm cross-validation
    - cv_std: Độ lệch chuẩn CV
    - n_iter: Số lần thử
    - tuning_time: Thời gian tuning
    - method: Phương pháp tuning
    - timestamp: Thời gian thực hiện
    """
    model_type: str
    best_params: Dict[str, Any]
    best_score: float
    cv_mean: float
    cv_std: float
    n_iter: int
    tuning_time: float
    method: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)



@dataclass
class TuningConfig:
    """
    Cấu hình cho quá trình tuning.
    - model_type: Loại model
    - tuning_method: Phương pháp tuning
    - n_trials: Số lần thử (Optuna/RandomSearch)
    - cv_folds: Số fold cross-validation
    - n_jobs: Số luồng chạy song song
    - random_state: Seed random
    - test_size: Tỉ lệ test
    - metric: Chỉ số đánh giá (rmse, mae, ...)
    - verbose: Mức độ log
    - save_results: Có lưu kết quả không
    - output_dir: Thư mục lưu kết quả
    """
    model_type: str = "xgboost"
    tuning_method: str = TuningMethod.OPTUNA
    n_trials: int = 50
    cv_folds: int = 5
    n_jobs: int = -1
    random_state: int = 42
    test_size: float = 0.2
    metric: str = "rmse"  # mse, mae, rmse, r2, neg_mean_squared_error
    verbose: int = 1
    save_results: bool = True
    output_dir: Optional[Path] = None



# =====================================================================================
# (5) HELPER FUNCTIONS
# =====================================================================================
# Các hàm tiện ích hỗ trợ cho quá trình tuning

def _load_config(path: Path) -> Dict[str, Any]:
    """
    Load config từ file JSON/YAML.
    Trả về dict cấu hình.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    ext = path.suffix.lower()
    if ext == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    elif ext in [".yml", ".yaml"]:
        try:
            import yaml
            return yaml.safe_load(path.read_text(encoding="utf-8"))
        except ImportError:
            raise RuntimeError("PyYAML not installed. Use JSON config hoặc install pyyaml.")
    
    raise ValueError(f"Unsupported config extension: {ext}")

def _load_df_via_loader(app_root: Path, folder_key: str, filename: str) -> pd.DataFrame:
    """
    Load data từ folder bằng DataLoader.
    Trả về dataframe.
    """
    loader = DataLoader(base_path=str(app_root))
    result = loader.load_all(folder_key, filename)
    
    if not result.is_success or result.data is None:
        raise FileNotFoundError(f"Cannot load data: {folder_key}/{filename}")
    
    if not isinstance(result.data, pd.DataFrame):
        raise ValueError(f"Loaded data is not DataFrame: {type(result.data)}")
    
    return result.data

def _ensure_dir(p: Path) -> None:
    """
    Tạo thư mục nếu chưa tồn tại.
    """
    p.mkdir(parents=True, exist_ok=True)

def _now_tag() -> str:
    """
    Lấy timestamp hiện tại dạng string.
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")



# =====================================================================================
# (6) IMPORT OPTUNA (nếu khả dụng)
# =====================================================================================
# Kiểm tra xem optuna đã cài chưa, nếu chưa thì chỉ dùng GridSearch/RandomSearch
OPTUNA_AVAILABLE = False
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    logger.warning("Optuna not installed. GridSearch/RandomSearch only. Install: pip install optuna")



# =====================================================================================
# (7) MAIN TUNING CLASS
# =====================================================================================
# Class chính để thực hiện tuning hyperparameters cho các mô hình ML

class HyperparameterTuner:

    def _bayesian_search(self, X: np.ndarray, y: np.ndarray) -> TuningResult:
        """
        Bayesian Optimization - Tối ưu hóa hyperparameters bằng scikit-optimize (skopt).
        Sử dụng BayesSearchCV để tìm bộ tham số tốt nhất.
        """
        try:
            from skopt import BayesSearchCV
            from skopt.space import Integer, Real
        except ImportError:
            logger.error("Bạn cần cài đặt scikit-optimize: pip install scikit-optimize")
            raise

        logger.info("Starting Bayesian Optimization (BayesSearchCV)...")
        start_time = datetime.now()

        # Chuyển đổi search space sang format cho skopt
        param_space = {}
        search_space = OPTUNA_SEARCH_SPACES.get(self.config.model_type.lower(), {})
        for param, rng in search_space.items():
            if isinstance(rng, tuple) and len(rng) == 2:
                # Nếu là int, dùng Integer, nếu là float, dùng Real
                if param in ["n_estimators", "iterations", "num_leaves", "max_depth", "min_child_samples", "min_samples_split", "min_samples_leaf", "l2_leaf_reg", "min_child_weight"]:
                    param_space[param] = Integer(rng[0], rng[1])
                else:
                    param_space[param] = Real(rng[0], rng[1], prior="uniform")


        # Lấy estimator gốc cho BayesSearchCV
        model_type = self.config.model_type.lower()
        if model_type in ["xgb", "xgboost"]:
            try:
                from xgboost import XGBRegressor
                estimator = XGBRegressor(random_state=self.config.random_state)
            except ImportError:
                logger.error("Bạn cần cài đặt xgboost: pip install xgboost")
                raise
        elif model_type in ["lgbm", "lightgbm"]:
            try:
                from lightgbm import LGBMRegressor
                estimator = LGBMRegressor(random_state=self.config.random_state)
            except ImportError:
                logger.error("Bạn cần cài đặt lightgbm: pip install lightgbm")
                raise
        elif model_type in ["rf", "random_forest"]:
            from sklearn.ensemble import RandomForestRegressor
            estimator = RandomForestRegressor(random_state=self.config.random_state)
        elif model_type in ["cat", "catboost"]:
            try:
                from catboost import CatBoostRegressor
                estimator = CatBoostRegressor(random_state=self.config.random_state, verbose=0)
            except ImportError:
                logger.error("Bạn cần cài đặt catboost: pip install catboost")
                raise
        else:
            raise ValueError(f"Unknown model type for Bayesian search: {self.config.model_type}")

        # Khởi tạo BayesSearchCV
        bayes_search = BayesSearchCV(
            estimator=estimator,
            search_spaces=param_space,
            n_iter=self.config.n_trials,
            cv=self.config.cv_folds,
            scoring=self.config.metric,
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_state,
            verbose=self.config.verbose,
        )

        # Fit
        bayes_search.fit(X, y)

        # Kết quả
        tuning_time = (datetime.now() - start_time).total_seconds()

        result = TuningResult(
            model_type=self.config.model_type,
            best_params=bayes_search.best_params_,
            best_score=bayes_search.best_score_,
            cv_mean=bayes_search.cv_results_['mean_test_score'].mean(),
            cv_std=bayes_search.cv_results_['std_test_score'].mean(),
            n_iter=len(bayes_search.cv_results_['params']),
            tuning_time=tuning_time,
            method="bayesian",
        )

        logger.info(f"Bayesian Optimization completed in {tuning_time:.2f}s")
        logger.info(f"Best params: {result.best_params}")
        logger.info(f"Best score: {result.best_score:.6f}")

        return result

    def __init__(self, config: Optional[TuningConfig] = None):
        """
        Khởi tạo tuner.
        Args:
            config: TuningConfig instance
        """
        self.config = config or TuningConfig()
        self.model = None
        self.best_params = None
        self.search_result = None

    def _get_model(self):
        """
        Lấy model instance dựa trên model_type.
        Import đúng class model theo loại model.
        """
        try:
            if self.config.model_type.lower() in ["xgb", "xgboost"]:
                from Weather_Forcast_App.Machine_learning_model.Models.XGBoost_Model import WeatherXGBoost
                return WeatherXGBoost(task_type="regression")
            elif self.config.model_type.lower() in ["lgbm", "lightgbm"]:
                from Weather_Forcast_App.Machine_learning_model.Models.LightGBM_Model import WeatherLightGBM
                return WeatherLightGBM(task_type="regression")
            elif self.config.model_type.lower() in ["rf", "random_forest"]:
                from Weather_Forcast_App.Machine_learning_model.Models.Random_Forest_Model import WeatherRandomForest
                return WeatherRandomForest(task_type="regression")
            elif self.config.model_type.lower() in ["cat", "catboost"]:
                from Weather_Forcast_App.Machine_learning_model.Models.CatBoost_Model import WeatherCatBoost
                return WeatherCatBoost(task_type="regression")
            else:
                raise ValueError(f"Unknown model type: {self.config.model_type}")
        except ImportError as e:
            logger.error(f"Failed to import model: {e}")
            raise

    def _grid_search(self, X: np.ndarray, y: np.ndarray) -> TuningResult:
        """
        GridSearchCV - tìm kiếm toàn bộ grid.
        Duyệt hết tất cả các tổ hợp tham số.
        """
        logger.info("Starting GridSearchCV...")
        start_time = datetime.now()

        # Lấy search space
        param_grid = HYPERPARAMETER_SPACES.get(
            self.config.model_type.lower(), 
            {}
        )
        if not param_grid:
            raise ValueError(f"No search space defined for {self.config.model_type}")

        # Lấy model
        estimator = self._get_model()

        # GridSearchCV
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=self.config.cv_folds,
            scoring=self.config.metric,
            n_jobs=self.config.n_jobs,
            verbose=self.config.verbose,
        )

        # Fit
        grid_search.fit(X, y)

        # Kết quả
        tuning_time = (datetime.now() - start_time).total_seconds()

        result = TuningResult(
            model_type=self.config.model_type,
            best_params=grid_search.best_params_,
            best_score=grid_search.best_score_,
            cv_mean=grid_search.cv_results_['mean_test_score'].mean(),
            cv_std=grid_search.cv_results_['std_test_score'].mean(),
            n_iter=len(grid_search.cv_results_['params']),
            tuning_time=tuning_time,
            method="grid_search",
        )

        logger.info(f"GridSearch completed in {tuning_time:.2f}s")
        logger.info(f"Best params: {result.best_params}")
        logger.info(f"Best score: {result.best_score:.6f}")

        return result

    def _random_search(self, X: np.ndarray, y: np.ndarray) -> TuningResult:
        """
        RandomizedSearchCV - tìm kiếm ngẫu nhiên.
        Chọn ngẫu nhiên các tổ hợp tham số.
        """
        logger.info("Starting RandomizedSearchCV...")
        start_time = datetime.now()

        # Lấy search space
        param_dist = HYPERPARAMETER_SPACES.get(
            self.config.model_type.lower(), 
            {}
        )
        if not param_dist:
            raise ValueError(f"No search space defined for {self.config.model_type}")

        # Lấy model
        estimator = self._get_model()

        # RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_dist,
            n_iter=self.config.n_trials,
            cv=self.config.cv_folds,
            scoring=self.config.metric,
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_state,
            verbose=self.config.verbose,
        )

        # Fit
        random_search.fit(X, y)

        # Kết quả
        tuning_time = (datetime.now() - start_time).total_seconds()

        result = TuningResult(
            model_type=self.config.model_type,
            best_params=random_search.best_params_,
            best_score=random_search.best_score_,
            cv_mean=random_search.cv_results_['mean_test_score'].mean(),
            cv_std=random_search.cv_results_['std_test_score'].mean(),
            n_iter=random_search.n_iter_,
            tuning_time=tuning_time,
            method="random_search",
        )

        logger.info(f"RandomSearch completed in {tuning_time:.2f}s")
        logger.info(f"Best params: {result.best_params}")
        logger.info(f"Best score: {result.best_score:.6f}")

        return result

    def _optuna_search(self, X: np.ndarray, y: np.ndarray) -> TuningResult:
        """
        Optuna - tìm kiếm thông minh (TPE sampler + pruning).
        Dùng Optuna để tối ưu hóa hyperparameters.
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available. Falling back to RandomSearch.")
            return self._random_search(X, y)

        logger.info("Starting Optuna optimization...")
        start_time = datetime.now()

        # Tạo study
        sampler = TPESampler(seed=self.config.random_state)
        pruner = MedianPruner()

        study = optuna.create_study(
            sampler=sampler,
            pruner=pruner,
            direction="maximize",  # Maximize score
        )

        # Objective function: Hàm mục tiêu cho Optuna
        def objective(trial: optuna.Trial) -> float:
            # Tạo params từ trial
            params = self._create_trial_params(trial)

            # Train model với CV
            model = self._get_model()

            try:
                cv = KFold(
                    n_splits=self.config.cv_folds,
                    shuffle=True,
                    random_state=self.config.random_state,
                )

                scores = cross_val_score(
                    model,
                    X, y,
                    cv=cv,
                    scoring=self.config.metric,
                    n_jobs=1,
                )

                score = scores.mean()

                # Report to trial (for pruning)
                trial.report(score, step=0)

                if trial.should_prune():
                    raise optuna.TrialPruned()

                return score

            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return -np.inf

        # Optimize
        study.optimize(objective, n_trials=self.config.n_trials, show_progress_bar=True)

        # Kết quả
        tuning_time = (datetime.now() - start_time).total_seconds()

        best_params = study.best_params
        best_score = study.best_value

        result = TuningResult(
            model_type=self.config.model_type,
            best_params=best_params,
            best_score=best_score,
            cv_mean=best_score,
            cv_std=0.0,
            n_iter=len(study.trials),
            tuning_time=tuning_time,
            method="optuna",
        )

        logger.info(f"Optuna completed in {tuning_time:.2f}s")
        logger.info(f"Best params: {result.best_params}")
        logger.info(f"Best score: {result.best_score:.6f}")

        return result

    def _create_trial_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Tạo hyperparameters từ Optuna trial.
        Dựa vào OPTUNA_SEARCH_SPACES.
        """
        spaces = OPTUNA_SEARCH_SPACES.get(self.config.model_type.lower(), {})
        params = {}

        for param_name, space in spaces.items():
            if isinstance(space, tuple) and len(space) == 2:
                # Numeric range
                if param_name in ["n_estimators", "iterations", "num_leaves", "max_depth", 
                                  "min_child_samples", "min_samples_split", "min_samples_leaf",
                                  "l2_leaf_reg", "min_child_weight"]:
                    params[param_name] = trial.suggest_int(param_name, space[0], space[1])
                else:
                    params[param_name] = trial.suggest_float(param_name, space[0], space[1])
        return params

    def tune(self, X: np.ndarray, y: np.ndarray) -> TuningResult:
        """
        Thực hiện tuning hyperparameters.
        Args:
            X: Features (n_samples, n_features)
            y: Target (n_samples,)
        Returns:
            TuningResult
        """
        logger.info(f"Tuning {self.config.model_type} with {self.config.tuning_method}")
        logger.info(f"Data shape: X={X.shape}, y={y.shape}")

        # Tiêu chuẩn hóa features (quan trọng cho một số models)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Chọn phương pháp tuning
        if self.config.tuning_method == TuningMethod.GRID_SEARCH:
            result = self._grid_search(X_scaled, y)
        elif self.config.tuning_method == TuningMethod.RANDOM_SEARCH:
            result = self._random_search(X_scaled, y)
        elif self.config.tuning_method == TuningMethod.OPTUNA:
            result = self._optuna_search(X_scaled, y)
        elif self.config.tuning_method == TuningMethod.BAYESIAN:
            result = self._bayesian_search(X_scaled, y)
        else:
            raise ValueError(f"Unknown tuning method: {self.config.tuning_method}")

        self.best_params = result.best_params
        self.search_result = result

        # Lưu kết quả tuning ra file nếu cần
        if self.config.save_results and self.config.output_dir:
            output_file = (
                self.config.output_dir / 
                f"tuning_result_{self.config.model_type}_{_now_tag()}.json"
            )
            result.to_json(output_file)
            logger.info(f"Results saved to {output_file}")

        return result


# =====================================================================================
# (8) MAIN SCRIPT
# =====================================================================================
# Đoạn script chạy trực tiếp khi gọi file này bằng python

def main():
    """
    Main entry point.
    Xử lý các tham số dòng lệnh, load config, tạo tuner và thực hiện tuning.
    """
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning cho ML models"
    )

    # Thêm các tham số dòng lệnh
    parser.add_argument("--config", type=Path, help="Config file path")
    parser.add_argument("--model", type=str, default="xgboost", 
                       help="Model type: xgboost, lightgbm, random_forest, catboost")
    parser.add_argument("--method", type=str, default=TuningMethod.OPTUNA,
                       help="Tuning method: grid_search, random_search, optuna")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--metric", type=str, default="neg_mean_squared_error", help="Metric to optimize")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel jobs")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")

    args = parser.parse_args()

    # Load config nếu có
    if args.config:
        config_dict = _load_config(args.config)
        tuning_config = TuningConfig(
            model_type=args.model,
            tuning_method=args.method,
            n_trials=args.n_trials,
            cv_folds=args.cv_folds,
            metric=args.metric,
            n_jobs=args.n_jobs,
            output_dir=args.output_dir or Path(config_dict.get("artifacts", {}).get("root_dir", "artifacts")),
        )
    else:
        tuning_config = TuningConfig(
            model_type=args.model,
            tuning_method=args.method,
            n_trials=args.n_trials,
            cv_folds=args.cv_folds,
            metric=args.metric,
            n_jobs=args.n_jobs,
            output_dir=args.output_dir or Path("Weather_Forcast_App/Machine_learning_artifacts/latest"),
        )

    # Load data (dummy) - demo, thực tế sẽ load từ file
    logger.info("Loading data...")
    X_dummy = np.random.randn(100, 20)
    y_dummy = np.random.randn(100)

    # Tuning
    tuner = HyperparameterTuner(tuning_config)
    result = tuner.tune(X_dummy, y_dummy)

    logger.info("Tuning completed!")
    logger.info(f"Best params: {result.best_params}")
    logger.info(f"Best score: {result.best_score:.6f}")


if __name__ == "__main__":
    main()