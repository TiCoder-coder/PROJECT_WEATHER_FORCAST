"""
📊 Metrics Module - Định nghĩa các chỉ số đánh giá cho ML models
=================================================================

Nơi định nghĩa các metric: MAE, RMSE, MAPE, R2...
Dùng chung cho mọi model trong hệ thống dự báo thời tiết.

Author: Weather Forecast Team
"""

import numpy as np
from typing import Dict, Union, Optional, List
from dataclasses import dataclass


# =============================================================================
# 📦 DATA CLASSES
# =============================================================================

@dataclass
class MetricResult:
    """Kết quả của một metric đơn lẻ"""
    name: str
    value: float
    description: str
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "value": round(self.value, 6),
            "description": self.description
        }


@dataclass  
class EvaluationReport:
    """Báo cáo tổng hợp tất cả metrics"""
    metrics: Dict[str, float]
    target_column: str
    n_samples: int
    
    def to_dict(self) -> Dict:
        return {
            "target_column": self.target_column,
            "n_samples": self.n_samples,
            "metrics": {k: round(v, 6) for k, v in self.metrics.items()}
        }


# =============================================================================
# 📐 REGRESSION METRICS - Các metric cho bài toán hồi quy
# =============================================================================

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MAE - Mean Absolute Error (Sai số tuyệt đối trung bình)
    
    Công thức: MAE = (1/n) * Σ|y_true - y_pred|
    
    Ý nghĩa:
    - Đơn vị giống với target (mm mưa, độ C, ...)
    - MAE càng nhỏ càng tốt
    - Không phạt nặng outliers như MSE/RMSE
    
    Args:
        y_true: Giá trị thực tế
        y_pred: Giá trị dự đoán
        
    Returns:
        float: Giá trị MAE
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"Độ dài không khớp: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MSE - Mean Squared Error (Sai số bình phương trung bình)
    
    Công thức: MSE = (1/n) * Σ(y_true - y_pred)²
    
    Ý nghĩa:
    - Đơn vị là bình phương của target
    - Phạt nặng các sai số lớn (outliers)
    - MSE càng nhỏ càng tốt
    
    Args:
        y_true: Giá trị thực tế
        y_pred: Giá trị dự đoán
        
    Returns:
        float: Giá trị MSE
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"Độ dài không khớp: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    RMSE - Root Mean Squared Error (Căn bậc hai sai số bình phương trung bình)
    
    Công thức: RMSE = √MSE = √[(1/n) * Σ(y_true - y_pred)²]
    
    Ý nghĩa:
    - Đơn vị giống với target (dễ diễn giải)
    - Phạt nặng các sai số lớn hơn MAE
    - RMSE >= MAE (luôn luôn)
    - RMSE càng nhỏ càng tốt
    
    Args:
        y_true: Giá trị thực tế
        y_pred: Giá trị dự đoán
        
    Returns:
        float: Giá trị RMSE
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_percentage_error(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """
    MAPE - Mean Absolute Percentage Error (Sai số phần trăm tuyệt đối trung bình)
    
    Công thức: MAPE = (100/n) * Σ|((y_true - y_pred) / y_true)|
    
    Ý nghĩa:
    - Đơn vị: phần trăm (%)
    - Dễ hiểu và so sánh giữa các dataset khác nhau
    - ⚠️ Không dùng được khi y_true có giá trị 0 hoặc gần 0
    
    Args:
        y_true: Giá trị thực tế
        y_pred: Giá trị dự đoán
        epsilon: Giá trị nhỏ để tránh chia cho 0
        
    Returns:
        float: Giá trị MAPE (đơn vị %)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"Độ dài không khớp: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    # Tránh chia cho 0 bằng cách thêm epsilon
    denominator = np.abs(y_true) + epsilon
    
    # Cảnh báo nếu có nhiều giá trị gần 0
    near_zero_count = np.sum(np.abs(y_true) < 0.01)
    if near_zero_count > len(y_true) * 0.1:  # > 10% giá trị gần 0
        import warnings
        warnings.warn(
            f"⚠️ {near_zero_count}/{len(y_true)} giá trị y_true gần 0. "
            "MAPE có thể không đáng tin cậy!"
        )
    
    mape = np.mean(np.abs((y_true - y_pred) / denominator)) * 100
    return float(mape)


def symmetric_mean_absolute_percentage_error(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> float:
    """
    sMAPE - Symmetric Mean Absolute Percentage Error
    
    Công thức: sMAPE = (100/n) * Σ(2 * |y_true - y_pred| / (|y_true| + |y_pred|))
    
    Ý nghĩa:
    - Phiên bản cân đối của MAPE
    - Xử lý tốt hơn khi y_true hoặc y_pred gần 0
    - Giới hạn trong khoảng [0, 200]%
    
    Args:
        y_true: Giá trị thực tế
        y_pred: Giá trị dự đoán
        
    Returns:
        float: Giá trị sMAPE (đơn vị %)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"Độ dài không khớp: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    # Xử lý trường hợp cả y_true và y_pred đều = 0
    mask = denominator != 0
    smape = np.zeros_like(numerator)
    smape[mask] = numerator[mask] / denominator[mask]
    
    return float(np.mean(smape) * 100)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R² - Coefficient of Determination (Hệ số xác định)
    
    Công thức: R² = 1 - (SS_res / SS_tot)
              SS_res = Σ(y_true - y_pred)²
              SS_tot = Σ(y_true - mean(y_true))²
    
    Ý nghĩa:
    - Tỷ lệ phương sai của y được giải thích bởi model
    - R² = 1: Model hoàn hảo
    - R² = 0: Model dự đoán bằng mean(y)
    - R² < 0: Model tệ hơn cả baseline mean
    
    Args:
        y_true: Giá trị thực tế
        y_pred: Giá trị dự đoán
        
    Returns:
        float: Giá trị R² (thường trong khoảng [0, 1])
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"Độ dài không khớp: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    ss_res = np.sum((y_true - y_pred) ** 2)  # Residual sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # Total sum of squares
    
    if ss_tot == 0:
        return 0.0  # Tất cả y_true đều bằng nhau
    
    return float(1 - (ss_res / ss_tot))


def adjusted_r2_score(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    n_features: int
) -> float:
    """
    Adjusted R² - R² điều chỉnh theo số features
    
    Công thức: Adj_R² = 1 - [(1 - R²) * (n - 1) / (n - p - 1)]
    
    Ý nghĩa:
    - Phạt model có quá nhiều features
    - Tránh overfitting do thêm feature vô nghĩa
    - Nên dùng khi so sánh models với số features khác nhau
    
    Args:
        y_true: Giá trị thực tế
        y_pred: Giá trị dự đoán
        n_features: Số lượng features trong model
        
    Returns:
        float: Giá trị Adjusted R²
    """
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    
    if n <= n_features + 1:
        raise ValueError(
            f"Số samples ({n}) phải lớn hơn số features + 1 ({n_features + 1})"
        )
    
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - n_features - 1))
    return float(adj_r2)


# =============================================================================
# 🌧️ WEATHER-SPECIFIC METRICS - Metrics đặc thù cho dự báo thời tiết
# =============================================================================

def rainfall_accuracy(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    threshold: float = 0.1
) -> float:
    """
    Accuracy cho dự báo mưa/không mưa (binary classification từ regression)
    
    Ý nghĩa:
    - Chuyển đổi lượng mưa thành có mưa (>threshold) / không mưa
    - Tính accuracy của việc dự đoán có mưa hay không
    
    Args:
        y_true: Lượng mưa thực tế (mm)
        y_pred: Lượng mưa dự đoán (mm)
        threshold: Ngưỡng xác định có mưa (mặc định 0.1mm)
        
    Returns:
        float: Accuracy (0-1)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Chuyển thành binary: có mưa = 1, không mưa = 0
    true_rain = (y_true > threshold).astype(int)
    pred_rain = (y_pred > threshold).astype(int)
    
    accuracy = np.mean(true_rain == pred_rain)
    return float(accuracy)


def rainfall_precision_recall(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    threshold: float = 0.1
) -> Dict[str, float]:
    """
    Precision và Recall cho dự báo mưa
    
    Returns:
        Dict với keys: precision, recall, f1_score
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    true_rain = (y_true > threshold).astype(int)
    pred_rain = (y_pred > threshold).astype(int)
    
    # True Positives, False Positives, False Negatives
    tp = np.sum((true_rain == 1) & (pred_rain == 1))
    fp = np.sum((true_rain == 0) & (pred_rain == 1))
    fn = np.sum((true_rain == 1) & (pred_rain == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }


def critical_success_index(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    threshold: float = 0.1
) -> float:
    """
    CSI - Critical Success Index (Threat Score)
    
    Công thức: CSI = TP / (TP + FP + FN)
    
    Ý nghĩa:
    - Metric phổ biến trong dự báo thời tiết
    - Đánh giá khả năng dự báo sự kiện (mưa)
    - Không tính True Negatives (ngày không mưa đúng)
    
    Args:
        y_true: Lượng mưa thực tế
        y_pred: Lượng mưa dự đoán
        threshold: Ngưỡng xác định có mưa
        
    Returns:
        float: CSI score (0-1)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    true_rain = (y_true > threshold).astype(int)
    pred_rain = (y_pred > threshold).astype(int)
    
    tp = np.sum((true_rain == 1) & (pred_rain == 1))
    fp = np.sum((true_rain == 0) & (pred_rain == 1))
    fn = np.sum((true_rain == 1) & (pred_rain == 0))
    
    denominator = tp + fp + fn
    if denominator == 0:
        return 0.0
    
    return float(tp / denominator)


def bias_score(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    threshold: float = 0.1
) -> float:
    """
    Bias Score (Frequency Bias Index)
    
    Công thức: Bias = (TP + FP) / (TP + FN)
    
    Ý nghĩa:
    - Bias = 1: Model không thiên lệch
    - Bias > 1: Model dự báo mưa quá nhiều (over-forecast)
    - Bias < 1: Model dự báo mưa quá ít (under-forecast)
    
    Args:
        y_true: Lượng mưa thực tế
        y_pred: Lượng mưa dự đoán
        threshold: Ngưỡng xác định có mưa
        
    Returns:
        float: Bias score
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    true_rain = (y_true > threshold).astype(int)
    pred_rain = (y_pred > threshold).astype(int)
    
    tp = np.sum((true_rain == 1) & (pred_rain == 1))
    fp = np.sum((true_rain == 0) & (pred_rain == 1))
    fn = np.sum((true_rain == 1) & (pred_rain == 0))
    
    denominator = tp + fn
    if denominator == 0:
        return 0.0
    
    return float((tp + fp) / denominator)


# =============================================================================
# 🔧 UTILITY FUNCTIONS - Hàm tiện ích
# =============================================================================

def calculate_all_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    n_features: Optional[int] = None,
    include_weather_metrics: bool = False,
    rain_threshold: float = 0.1
) -> Dict[str, float]:
    """
    Tính tất cả metrics một lần
    
    Args:
        y_true: Giá trị thực tế
        y_pred: Giá trị dự đoán
        n_features: Số features (để tính Adjusted R²)
        include_weather_metrics: Có tính metrics đặc thù thời tiết không
        rain_threshold: Ngưỡng mưa (nếu include_weather_metrics=True)
        
    Returns:
        Dict chứa tất cả metrics
    """
    results = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "sMAPE": symmetric_mean_absolute_percentage_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }
    
    # Adjusted R² nếu có số features
    if n_features is not None:
        try:
            results["Adjusted_R2"] = adjusted_r2_score(y_true, y_pred, n_features)
        except ValueError:
            results["Adjusted_R2"] = None
    
    # Weather-specific metrics
    if include_weather_metrics:
        results["Rain_Accuracy"] = rainfall_accuracy(y_true, y_pred, rain_threshold)
        results["CSI"] = critical_success_index(y_true, y_pred, rain_threshold)
        results["Bias"] = bias_score(y_true, y_pred, rain_threshold)
        
        pr_metrics = rainfall_precision_recall(y_true, y_pred, rain_threshold)
        results["Rain_Precision"] = pr_metrics["precision"]
        results["Rain_Recall"] = pr_metrics["recall"]
        results["Rain_F1"] = pr_metrics["f1_score"]
    
    return results


def compare_models(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    primary_metric: str = "RMSE"
) -> Dict[str, Dict]:
    """
    So sánh nhiều models với nhau
    
    Args:
        y_true: Giá trị thực tế
        predictions: Dict {model_name: y_pred}
        primary_metric: Metric chính để xếp hạng
        
    Returns:
        Dict với metrics của từng model + ranking
    """
    results = {}
    
    for model_name, y_pred in predictions.items():
        metrics = calculate_all_metrics(y_true, y_pred)
        results[model_name] = metrics
    
    # Xếp hạng theo primary_metric
    # (lower is better for MAE/MSE/RMSE/MAPE, higher is better for R²)
    lower_is_better = primary_metric in ["MAE", "MSE", "RMSE", "MAPE", "sMAPE"]
    
    sorted_models = sorted(
        results.keys(),
        key=lambda m: results[m].get(primary_metric, float('inf')),
        reverse=not lower_is_better
    )
    
    for rank, model_name in enumerate(sorted_models, 1):
        results[model_name]["rank"] = rank
    
    return results


def format_metrics_report(metrics: Dict[str, float]) -> str:
    """
    Format metrics thành string đẹp để in ra console/log
    
    Args:
        metrics: Dict chứa các metrics
        
    Returns:
        String formatted
    """
    lines = [
        "=" * 50,
        "📊 EVALUATION METRICS REPORT",
        "=" * 50,
    ]
    
    # Group metrics
    regression_metrics = ["MAE", "MSE", "RMSE", "MAPE", "sMAPE", "R2", "Adjusted_R2"]
    weather_metrics = ["Rain_Accuracy", "CSI", "Bias", "Rain_Precision", "Rain_Recall", "Rain_F1"]
    
    lines.append("\n📐 Regression Metrics:")
    lines.append("-" * 30)
    for m in regression_metrics:
        if m in metrics and metrics[m] is not None:
            value = metrics[m]
            if m in ["MAPE", "sMAPE"]:
                lines.append(f"  {m:15} : {value:>10.4f} %")
            else:
                lines.append(f"  {m:15} : {value:>10.6f}")
    
    # Weather metrics nếu có
    weather_present = any(m in metrics for m in weather_metrics)
    if weather_present:
        lines.append("\n🌧️ Weather-Specific Metrics:")
        lines.append("-" * 30)
        for m in weather_metrics:
            if m in metrics and metrics[m] is not None:
                value = metrics[m]
                if "Accuracy" in m or "Precision" in m or "Recall" in m or "F1" in m:
                    lines.append(f"  {m:15} : {value:>10.4f} ({value*100:.2f}%)")
                else:
                    lines.append(f"  {m:15} : {value:>10.4f}")
    
    lines.append("=" * 50)
    
    return "\n".join(lines)


# =============================================================================
# 🧪 SIMPLE TEST / DEMO
# =============================================================================

if __name__ == "__main__":
    # Demo với dữ liệu giả
    np.random.seed(42)
    
    # Giả lập dữ liệu lượng mưa (mm)
    y_true = np.array([0.0, 0.0, 5.2, 12.3, 0.0, 8.7, 25.1, 0.0, 3.4, 15.8])
    y_pred = np.array([0.5, 0.0, 4.8, 10.1, 0.2, 9.5, 22.0, 0.0, 4.1, 14.2])
    
    print("\n🌧️ Demo: Đánh giá dự báo lượng mưa")
    print("=" * 50)
    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")
    
    # Tính tất cả metrics
    all_metrics = calculate_all_metrics(
        y_true, y_pred, 
        n_features=5,
        include_weather_metrics=True
    )
    
    # In report
    print(format_metrics_report(all_metrics))
    
    # So sánh nhiều models
    print("\n📊 So sánh 3 models:")
    predictions = {
        "XGBoost": y_pred,
        "LightGBM": y_pred + np.random.randn(10) * 0.5,
        "CatBoost": y_pred - np.random.randn(10) * 0.3,
    }
    
    comparison = compare_models(y_true, predictions, primary_metric="RMSE")
    
    for model, metrics in comparison.items():
        print(f"\n🔹 {model} (Rank #{metrics['rank']}):")
        print(f"   RMSE: {metrics['RMSE']:.4f}")
        print(f"   MAE:  {metrics['MAE']:.4f}")
        print(f"   R²:   {metrics['R2']:.4f}")
