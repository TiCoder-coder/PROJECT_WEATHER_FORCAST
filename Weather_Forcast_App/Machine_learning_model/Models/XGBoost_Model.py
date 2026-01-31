import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

class WeatherXGBoostModel:
    """
    XGBoost model cho dự báo thời tiết
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

        # Tham số mặc định cho XGBoost
        self.default_params = {
            'objective': 'reg:squarederror',  # cho regression
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbosity': 0
        }

        # Cập nhật với config người dùng
        self.params = {**self.default_params, **self.config}

    def prepare_data(self, data_path, target_column, test_size=0.2):
        """
        Chuẩn bị dữ liệu cho training
        """
        # Đọc dữ liệu
        df = pd.read_csv(data_path)

        # Xử lý missing values cơ bản
        df = df.dropna()

        # Tách features và target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Mã hóa categorical features nếu có
        X = pd.get_dummies(X, drop_first=True)

        # Chia train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train model XGBoost
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Tạo DMatrix cho XGBoost
        dtrain = xgb.DMatrix(X_train_scaled, label=y_train)

        # Validation set nếu có
        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            dval = xgb.DMatrix(X_val_scaled, label=y_val)
            evals.append((dval, 'validation'))

        # Train model
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.params['n_estimators'],
            evals=evals,
            early_stopping_rounds=10,
            verbose_eval=False
        )

        self.is_trained = True
        print("Model XGBoost đã được train thành công!")
        return self.model

    def predict(self, X):
        """
        Dự đoán với model đã train
        """
        if not self.is_trained:
            raise ValueError("Model chưa được train!")

        # Scale input
        X_scaled = self.scaler.transform(X)

        # Tạo DMatrix và predict
        dtest = xgb.DMatrix(X_scaled)
        predictions = self.model.predict(dtest)

        return predictions

    def evaluate(self, X_test, y_test):
        """
        Đánh giá model
        """
        if not self.is_trained:
            raise ValueError("Model chưa được train!")

        predictions = self.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }

        print("\nKết quả đánh giá model:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")

        return metrics

    def save_model(self, filepath):
        """
        Lưu model đã train
        """
        if not self.is_trained:
            raise ValueError("Model chưa được train!")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Lưu model và scaler
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'params': self.params
        }

        joblib.dump(model_data, filepath)
        print(f"Model đã được lưu tại: {filepath}")

    def load_model(self, filepath):
        """
        Load model đã lưu
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Không tìm thấy file model: {filepath}")

        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.params = model_data['params']
        self.is_trained = True

        print(f"Model đã được load từ: {filepath}")

    def get_feature_importance(self):
        """
        Lấy feature importance
        """
        if not self.is_trained:
            raise ValueError("Model chưa được train!")

        importance = self.model.get_score(importance_type='gain')
        return importance

    def plot_feature_importance(self, top_n=20):
        """
        Vẽ biểu đồ feature importance
        """
        if not self.is_trained:
            raise ValueError("Model chưa được train!")

        try:
            import matplotlib.pyplot as plt

            importance = self.get_feature_importance()
            importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

            if len(importance) > top_n:
                importance = importance[:top_n]

            features, scores = zip(*importance)

            plt.figure(figsize=(10, 6))
            plt.barh(range(len(features)), scores)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Importance Score')
            plt.title('Feature Importance (XGBoost)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()

        except ImportError:
            print("Cần cài matplotlib để vẽ biểu đồ")
            print("Chạy: pip install matplotlib")


if __name__ == "__main__":
    # Khởi tạo model
    model = WeatherXGBoostModel()

    # data_path = "path/to/weather_data.csv"
    # target_column = "temperature"  # hoặc "humidity", "rainfall", etc.

    # Chuẩn bị dữ liệu
    # X_train, X_test, y_train, y_test = model.prepare_data(data_path, target_column)

    # Train model
    # model.train(X_train, y_train)

    # Đánh giá
    # metrics = model.evaluate(X_test, y_test)

    # Lưu model
    # model.save_model("xgboost_weather_model.pkl")

    print("XGBoost model cho dự báo thời tiết đã sẵn sàng!")