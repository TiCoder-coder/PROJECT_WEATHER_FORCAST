"""
Report Module - Xuất báo cáo đánh giá ML Models


Mục đích:
    - Xuất báo cáo: bảng so sánh model, lưu biểu đồ, lưu file report csv/json
    - Đây là phần cực hợp để "bỏ vào báo cáo nghiên cứu"
    - Tạo visualizations cho training results
    - Export reports ở nhiều định dạng

Tính năng:
    - So sánh nhiều models trên cùng dataset
    - Vẽ biểu đồ: actual vs predicted, residuals, feature importance
    - Xuất report dạng CSV, JSON, HTML, Markdown
    - Tạo summary tables cho báo cáo khoa học

Author: Weather Forecast Team
"""

import json
import csv
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd

# Import metrics từ cùng package
from .metrics import (
    calculate_all_metrics,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    format_metrics_report
)

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print(" matplotlib not installed. Charts will not be available.")
    print("   Run: pip install matplotlib")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


#  DATA CLASSES


@dataclass
class ModelEvaluationResult:
    """Kết quả đánh giá của một model."""
    model_name: str
    metrics: Dict[str, float]
    y_true: Optional[np.ndarray] = None
    y_pred: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None
    training_time: float = 0.0
    n_samples: int = 0
    n_features: int = 0
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "model_name": self.model_name,
            "metrics": {k: round(v, 6) if v is not None else None 
                       for k, v in self.metrics.items()},
            "training_time": self.training_time,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "hyperparameters": self.hyperparameters,
            "notes": self.notes
        }
        if self.feature_importance:
            result["feature_importance"] = {
                k: round(v, 6) for k, v in self.feature_importance.items()
            }
        return result


@dataclass
class ComparisonReport:
    """Báo cáo so sánh nhiều models."""
    title: str
    description: str
    dataset_name: str
    target_column: str
    created_at: datetime = field(default_factory=datetime.now)
    models: List[ModelEvaluationResult] = field(default_factory=list)
    best_model: Optional[str] = None
    ranking_metric: str = "RMSE"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "dataset_name": self.dataset_name,
            "target_column": self.target_column,
            "created_at": self.created_at.isoformat(),
            "models": [m.to_dict() for m in self.models],
            "best_model": self.best_model,
            "ranking_metric": self.ranking_metric,
            "summary": self.get_summary()
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Tạo summary statistics."""
        if not self.models:
            return {}
        
        return {
            "total_models": len(self.models),
            "metrics_compared": list(self.models[0].metrics.keys()) if self.models else [],
            "best_model": self.best_model,
            "ranking_metric": self.ranking_metric
        }



#  REPORT GENERATOR CLASS


class EvaluationReportGenerator:
    """
    Class chính để tạo báo cáo đánh giá ML models.
    
    Tính năng:
        - So sánh nhiều models
        - Vẽ biểu đồ (charts)
        - Xuất report CSV/JSON/HTML/Markdown
        - Tạo bảng so sánh cho báo cáo nghiên cứu
    
    Example:
        >>> generator = EvaluationReportGenerator(output_dir="reports/")
        >>> 
        >>> # Thêm kết quả của các models
        >>> generator.add_model_result("XGBoost", y_true, y_pred_xgb)
        >>> generator.add_model_result("CatBoost", y_true, y_pred_cat)
        >>> generator.add_model_result("LightGBM", y_true, y_pred_lgb)
        >>> 
        >>> # Tạo báo cáo
        >>> report = generator.generate_comparison_report(
        ...     title="Weather Prediction Model Comparison",
        ...     dataset_name="Vietnam Weather 2024-2025"
        ... )
        >>> 
        >>> # Xuất các định dạng
        >>> generator.export_to_json("comparison_report.json")
        >>> generator.export_to_csv("metrics_table.csv")
        >>> generator.export_to_markdown("report.md")
        >>> generator.plot_comparison_charts()
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path] = "reports",
        fig_dpi: int = 150,
        fig_format: str = "png"
    ):
        """
        Khởi tạo Report Generator.
        
        Args:
            output_dir: Thư mục lưu reports
            fig_dpi: DPI cho hình ảnh
            fig_format: Định dạng hình (png, jpg, svg, pdf)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.fig_dpi = fig_dpi
        self.fig_format = fig_format
        
        # Store model results
        self.model_results: List[ModelEvaluationResult] = []
        self.comparison_report: Optional[ComparisonReport] = None
        
        # Figure style
        if MATPLOTLIB_AVAILABLE:
            plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'ggplot')
    
    
    # ADD MODEL RESULTS
   
    
    def add_model_result(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        feature_importance: Optional[Dict[str, float]] = None,
        training_time: float = 0.0,
        n_features: int = 0,
        hyperparameters: Optional[Dict[str, Any]] = None,
        include_weather_metrics: bool = False,
        notes: str = ""
    ) -> ModelEvaluationResult:
        """
        Thêm kết quả đánh giá của một model.
        
        Args:
            model_name: Tên model
            y_true: Giá trị thực
            y_pred: Giá trị dự đoán
            feature_importance: Dict {feature_name: importance_score}
            training_time: Thời gian train (seconds)
            n_features: Số features
            hyperparameters: Các hyperparameters đã dùng
            include_weather_metrics: Có tính weather-specific metrics không
            notes: Ghi chú thêm
            
        Returns:
            ModelEvaluationResult object
        """
        # Calculate all metrics
        metrics = calculate_all_metrics(
            y_true, y_pred,
            n_features=n_features if n_features > 0 else None,
            include_weather_metrics=include_weather_metrics
        )
        
        result = ModelEvaluationResult(
            model_name=model_name,
            metrics=metrics,
            y_true=np.asarray(y_true),
            y_pred=np.asarray(y_pred),
            feature_importance=feature_importance,
            training_time=training_time,
            n_samples=len(y_true),
            n_features=n_features,
            hyperparameters=hyperparameters or {},
            notes=notes
        )
        
        self.model_results.append(result)
        print(f" Added model: {model_name} (RMSE: {metrics['RMSE']:.4f}, R²: {metrics['R2']:.4f})")
        
        return result
    
    def add_model_from_dict(
        self,
        model_name: str,
        metrics: Dict[str, float],
        **kwargs
    ) -> ModelEvaluationResult:
        """Thêm model từ dict metrics có sẵn."""
        result = ModelEvaluationResult(
            model_name=model_name,
            metrics=metrics,
            **kwargs
        )
        self.model_results.append(result)
        return result
    
    def clear_results(self):
        """Xóa tất cả kết quả đã lưu."""
        self.model_results = []
        self.comparison_report = None
    
    
    # GENERATE REPORTS
    
    
    def generate_comparison_report(
        self,
        title: str = "Model Comparison Report",
        description: str = "",
        dataset_name: str = "Unknown",
        target_column: str = "Unknown",
        ranking_metric: str = "RMSE"
    ) -> ComparisonReport:
        """
        Tạo báo cáo so sánh các models.
        
        Args:
            title: Tiêu đề báo cáo
            description: Mô tả
            dataset_name: Tên dataset
            target_column: Tên cột target
            ranking_metric: Metric để xếp hạng (RMSE, MAE, R2, ...)
            
        Returns:
            ComparisonReport object
        """
        if not self.model_results:
            raise ValueError("No model results added. Use add_model_result() first.")
        
        # Determine best model
        lower_is_better = ranking_metric in ["MAE", "MSE", "RMSE", "MAPE", "sMAPE"]
        
        sorted_models = sorted(
            self.model_results,
            key=lambda m: m.metrics.get(ranking_metric, float('inf')),
            reverse=not lower_is_better
        )
        
        best_model = sorted_models[0].model_name if sorted_models else None
        
        self.comparison_report = ComparisonReport(
            title=title,
            description=description,
            dataset_name=dataset_name,
            target_column=target_column,
            models=self.model_results,
            best_model=best_model,
            ranking_metric=ranking_metric
        )
        
        print(f"\n Generated comparison report: {title}")
        print(f"    Best model ({ranking_metric}): {best_model}")
        
        return self.comparison_report
    
    def get_metrics_dataframe(self) -> pd.DataFrame:
        """
        Lấy DataFrame chứa metrics của tất cả models.
        
        Returns:
            DataFrame với columns là metrics, rows là models
        """
        if not self.model_results:
            return pd.DataFrame()
        
        data = []
        for result in self.model_results:
            row = {"Model": result.model_name}
            row.update(result.metrics)
            row["Training Time (s)"] = result.training_time
            row["N Samples"] = result.n_samples
            row["N Features"] = result.n_features
            data.append(row)
        
        df = pd.DataFrame(data)
        df = df.set_index("Model")
        
        return df
    
    def get_ranking_table(
        self,
        metrics: Optional[List[str]] = None,
        ascending: bool = True
    ) -> pd.DataFrame:
        """
        Tạo bảng xếp hạng models theo từng metric.
        
        Args:
            metrics: List metrics cần xếp hạng. None = tất cả
            ascending: True = giá trị nhỏ hơn xếp trước
            
        Returns:
            DataFrame với ranking của từng model theo từng metric
        """
        df = self.get_metrics_dataframe()
        
        if metrics is None:
            metrics = ["MAE", "RMSE", "MAPE", "R2"]
        
        rankings = {}
        for metric in metrics:
            if metric in df.columns:
                # R² cao hơn = tốt hơn
                asc = ascending if metric != "R2" else not ascending
                rankings[f"{metric}_Rank"] = df[metric].rank(ascending=asc).astype(int)
        
        ranking_df = pd.DataFrame(rankings, index=df.index)
        ranking_df["Average_Rank"] = ranking_df.mean(axis=1)
        ranking_df = ranking_df.sort_values("Average_Rank")
        
        return ranking_df
    
    
    # EXPORT METHODS
    
    
    def export_to_json(
        self,
        filename: str = "evaluation_report.json",
        include_predictions: bool = False
    ) -> Path:
        """
        Xuất báo cáo ra file JSON.
        
        Args:
            filename: Tên file
            include_predictions: Có lưu y_true, y_pred không (file sẽ lớn)
            
        Returns:
            Path đến file đã lưu
        """
        filepath = self.output_dir / filename
        
        if self.comparison_report:
            data = self.comparison_report.to_dict()
        else:
            data = {
                "models": [m.to_dict() for m in self.model_results],
                "created_at": datetime.now().isoformat()
            }
        
        # Optionally include predictions
        if include_predictions:
            for i, result in enumerate(self.model_results):
                if result.y_true is not None:
                    data["models"][i]["y_true"] = result.y_true.tolist()
                if result.y_pred is not None:
                    data["models"][i]["y_pred"] = result.y_pred.tolist()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f" Saved JSON report: {filepath}")
        return filepath
    
    def export_to_csv(
        self,
        filename: str = "metrics_comparison.csv"
    ) -> Path:
        """
        Xuất bảng metrics ra file CSV.
        
        Args:
            filename: Tên file
            
        Returns:
            Path đến file đã lưu
        """
        filepath = self.output_dir / filename
        
        df = self.get_metrics_dataframe()
        df.to_csv(filepath, encoding='utf-8-sig')
        
        print(f" Saved CSV report: {filepath}")
        return filepath
    
    def export_to_markdown(
        self,
        filename: str = "evaluation_report.md",
        include_charts: bool = True
    ) -> Path:
        """
        Xuất báo cáo ra file Markdown (dễ đọc, dễ paste vào docs).
        
        Args:
            filename: Tên file
            include_charts: Có tham chiếu đến charts không
            
        Returns:
            Path đến file đã lưu
        """
        filepath = self.output_dir / filename
        
        lines = []
        
        # Header
        title = self.comparison_report.title if self.comparison_report else "Model Evaluation Report"
        lines.append(f"#  {title}\n")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        if self.comparison_report:
            lines.append(f"**Dataset:** {self.comparison_report.dataset_name}\n")
            lines.append(f"**Target:** {self.comparison_report.target_column}\n")
            if self.comparison_report.description:
                lines.append(f"\n{self.comparison_report.description}\n")
        
        lines.append("\n---\n")
        
        # Summary
        lines.append("## 🏆 Summary\n")
        if self.comparison_report and self.comparison_report.best_model:
            lines.append(f"- **Best Model:** {self.comparison_report.best_model}\n")
            lines.append(f"- **Ranking Metric:** {self.comparison_report.ranking_metric}\n")
        lines.append(f"- **Total Models Compared:** {len(self.model_results)}\n")
        
        lines.append("\n---\n")
        
        # Metrics Table
        lines.append("##  Metrics Comparison\n")
        
        df = self.get_metrics_dataframe()
        if not df.empty:
            # Format numbers
            formatted_df = df.copy()
            for col in formatted_df.columns:
                if formatted_df[col].dtype in ['float64', 'float32']:
                    formatted_df[col] = formatted_df[col].apply(
                        lambda x: f"{x:.4f}" if pd.notna(x) else "-"
                    )
            
            lines.append(formatted_df.to_markdown())
            lines.append("\n")
        
        # Ranking Table
        lines.append("\n## 🥇 Model Rankings\n")
        ranking_df = self.get_ranking_table()
        if not ranking_df.empty:
            lines.append(ranking_df.to_markdown())
            lines.append("\n")
        
        # Individual Model Details
        lines.append("\n##  Model Details\n")
        for result in self.model_results:
            lines.append(f"\n### {result.model_name}\n")
            lines.append(f"- **Samples:** {result.n_samples}\n")
            lines.append(f"- **Features:** {result.n_features}\n")
            lines.append(f"- **Training Time:** {result.training_time:.2f}s\n")
            
            if result.hyperparameters:
                lines.append("\n**Hyperparameters:**\n```json\n")
                lines.append(json.dumps(result.hyperparameters, indent=2))
                lines.append("\n```\n")
            
            if result.notes:
                lines.append(f"\n**Notes:** {result.notes}\n")
        
        # Charts reference
        if include_charts:
            lines.append("\n---\n")
            lines.append("##  Visualizations\n")
            lines.append("See generated chart files in the same directory.\n")
        
        # Write file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f" Saved Markdown report: {filepath}")
        return filepath
    
    def export_to_latex_table(
        self,
        filename: str = "metrics_table.tex",
        caption: str = "Model Comparison Results",
        label: str = "tab:model_comparison"
    ) -> Path:
        """
        Xuất bảng metrics ra LaTeX (cho papers/thesis).
        
        Args:
            filename: Tên file
            caption: Caption của bảng
            label: Label để reference trong LaTeX
            
        Returns:
            Path đến file đã lưu
        """
        filepath = self.output_dir / filename
        
        df = self.get_metrics_dataframe()
        
        # Select key metrics for paper
        key_metrics = ["MAE", "RMSE", "MAPE", "R2"]
        cols_to_use = [c for c in key_metrics if c in df.columns]
        df_paper = df[cols_to_use].copy()
        
        # Rename for LaTeX
        rename_map = {
            "MAE": "MAE",
            "RMSE": "RMSE", 
            "MAPE": "MAPE (\\%)",
            "R2": "$R^2$"
        }
        df_paper = df_paper.rename(columns=rename_map)
        
        # Generate LaTeX
        latex_content = df_paper.to_latex(
            float_format="%.4f",
            caption=caption,
            label=label,
            escape=False
        )
        
        # Add booktabs style
        latex_content = latex_content.replace("\\toprule", "\\hline\\hline")
        latex_content = latex_content.replace("\\midrule", "\\hline")
        latex_content = latex_content.replace("\\bottomrule", "\\hline\\hline")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        print(f" Saved LaTeX table: {filepath}")
        return filepath
    
   
    # VISUALIZATION METHODS
    
    
    def plot_metrics_comparison(
        self,
        metrics: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 6),
        save: bool = True
    ) -> Optional[plt.Figure]:
        """
        Vẽ biểu đồ so sánh metrics giữa các models.
        
        Args:
            metrics: List metrics cần vẽ
            figsize: Kích thước figure
            save: Có lưu file không
            
        Returns:
            matplotlib Figure
        """
        if not MATPLOTLIB_AVAILABLE:
            print(" matplotlib not available")
            return None
        
        if not self.model_results:
            print(" No model results to plot")
            return None
        
        if metrics is None:
            metrics = ["MAE", "RMSE", "R2"]
        
        df = self.get_metrics_dataframe()
        metrics = [m for m in metrics if m in df.columns]
        
        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
        if len(metrics) == 1:
            axes = [axes]
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(df)))
        
        for ax, metric in zip(axes, metrics):
            values = df[metric].values
            models = df.index.tolist()
            
            bars = ax.bar(models, values, color=colors)
            ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            
            # Highlight best
            if metric == "R2":
                best_idx = np.argmax(values)
            else:
                best_idx = np.argmin(values)
            bars[best_idx].set_color('green')
            bars[best_idx].set_edgecolor('darkgreen')
            bars[best_idx].set_linewidth(2)
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Model Metrics Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f"metrics_comparison.{self.fig_format}"
            plt.savefig(filepath, dpi=self.fig_dpi, bbox_inches='tight')
            print(f" Saved chart: {filepath}")
        
        return fig
    
    def plot_actual_vs_predicted(
        self,
        model_name: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
        save: bool = True
    ) -> Optional[plt.Figure]:
        """
        Vẽ biểu đồ Actual vs Predicted cho model.
        
        Args:
            model_name: Tên model. None = vẽ tất cả
            figsize: Kích thước figure
            save: Có lưu file không
        """
        if not MATPLOTLIB_AVAILABLE:
            print(" matplotlib not available")
            return None
        
        # Filter models with predictions
        models_with_data = [m for m in self.model_results 
                          if m.y_true is not None and m.y_pred is not None]
        
        if model_name:
            models_with_data = [m for m in models_with_data if m.model_name == model_name]
        
        if not models_with_data:
            print(" No models with prediction data")
            return None
        
        n_models = len(models_with_data)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(figsize[0], figsize[1] * rows / 2))
        if n_models == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for ax, result in zip(axes, models_with_data):
            y_true = result.y_true
            y_pred = result.y_pred
            
            # Scatter plot
            ax.scatter(y_true, y_pred, alpha=0.5, s=20)
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
            
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title(f'{result.model_name}\n(R²={result.metrics["R2"]:.4f})')
            ax.legend()
        
        # Hide empty subplots
        for idx in range(len(models_with_data), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Actual vs Predicted Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f"actual_vs_predicted.{self.fig_format}"
            plt.savefig(filepath, dpi=self.fig_dpi, bbox_inches='tight')
            print(f" Saved chart: {filepath}")
        
        return fig
    
    def plot_residuals(
        self,
        model_name: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 4),
        save: bool = True
    ) -> Optional[plt.Figure]:
        """
        Vẽ biểu đồ residuals (errors) cho model.
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        models_with_data = [m for m in self.model_results 
                          if m.y_true is not None and m.y_pred is not None]
        
        if model_name:
            models_with_data = [m for m in models_with_data if m.model_name == model_name]
        
        if not models_with_data:
            return None
        
        n_models = len(models_with_data)
        fig, axes = plt.subplots(1, n_models, figsize=(figsize[0], figsize[1]))
        if n_models == 1:
            axes = [axes]
        
        for ax, result in zip(axes, models_with_data):
            residuals = result.y_true - result.y_pred
            
            ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
            ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
            ax.set_xlabel('Residual (Actual - Predicted)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{result.model_name}\n(Mean: {residuals.mean():.4f})')
        
        plt.suptitle('Residual Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f"residuals.{self.fig_format}"
            plt.savefig(filepath, dpi=self.fig_dpi, bbox_inches='tight')
            print(f" Saved chart: {filepath}")
        
        return fig
    
    def plot_feature_importance(
        self,
        model_name: str,
        top_n: int = 20,
        figsize: Tuple[int, int] = (10, 8),
        save: bool = True
    ) -> Optional[plt.Figure]:
        """
        Vẽ biểu đồ feature importance cho model.
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        # Find model
        result = next((m for m in self.model_results if m.model_name == model_name), None)
        
        if not result or not result.feature_importance:
            print(f" No feature importance data for {model_name}")
            return None
        
        # Sort by importance
        importance = result.feature_importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        if len(sorted_features) > top_n:
            sorted_features = sorted_features[:top_n]
        
        features, values = zip(*sorted_features)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        y_pos = np.arange(len(features))
        ax.barh(y_pos, values, color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Feature Importance - {model_name} (Top {top_n})')
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f"feature_importance_{model_name}.{self.fig_format}"
            plt.savefig(filepath, dpi=self.fig_dpi, bbox_inches='tight')
            print(f"📊 Saved chart: {filepath}")
        
        return fig
    
    def plot_all_charts(self):
        """Vẽ tất cả biểu đồ có thể."""
        print("\n Generating all charts...")
        
        self.plot_metrics_comparison()
        self.plot_actual_vs_predicted()
        self.plot_residuals()
        
        # Feature importance for each model that has it
        for result in self.model_results:
            if result.feature_importance:
                self.plot_feature_importance(result.model_name)
        
        print(" All charts generated!")
    
    def generate_full_report(
        self,
        title: str = "Weather Prediction Model Evaluation",
        dataset_name: str = "Vietnam Weather Dataset",
        target_column: str = "rainfall"
    ):
        """
        Tạo báo cáo đầy đủ: comparison report + tất cả exports + tất cả charts.
        """
        print("\n" + "="*60)
        print(" GENERATING FULL EVALUATION REPORT")
        print("="*60)
        
        # Generate comparison report
        self.generate_comparison_report(
            title=title,
            dataset_name=dataset_name,
            target_column=target_column
        )
        
        # Export all formats
        self.export_to_json("evaluation_report.json")
        self.export_to_csv("metrics_comparison.csv")
        self.export_to_markdown("evaluation_report.md")
        self.export_to_latex_table("metrics_table.tex")
        
        # Generate all charts
        if MATPLOTLIB_AVAILABLE:
            self.plot_all_charts()
        
        print("\n" + "="*60)
        print(f" Full report generated in: {self.output_dir}")
        print("="*60)



#  UTILITY FUNCTIONS

def quick_compare_models(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    output_dir: str = "reports",
    include_weather_metrics: bool = False
) -> ComparisonReport:
    """
    Quick helper để so sánh nhiều models.
    
    Args:
        y_true: Giá trị thực
        predictions: Dict {model_name: y_pred}
        output_dir: Thư mục lưu output
        include_weather_metrics: Có tính weather metrics không
        
    Returns:
        ComparisonReport
        
    Example:
        >>> report = quick_compare_models(
        ...     y_true,
        ...     {
        ...         "XGBoost": y_pred_xgb,
        ...         "CatBoost": y_pred_cat,
        ...         "LightGBM": y_pred_lgb
        ...     }
        ... )
    """
    generator = EvaluationReportGenerator(output_dir=output_dir)
    
    for model_name, y_pred in predictions.items():
        generator.add_model_result(
            model_name=model_name,
            y_true=y_true,
            y_pred=y_pred,
            include_weather_metrics=include_weather_metrics
        )
    
    report = generator.generate_comparison_report()
    generator.export_to_json()
    generator.export_to_csv()
    
    return report


def create_evaluation_summary(
    metrics: Dict[str, float],
    model_name: str = "Model"
) -> str:
    """
    Tạo summary text từ metrics.
    
    Args:
        metrics: Dict các metrics
        model_name: Tên model
        
    Returns:
        String summary
    """
    lines = [
        f" Evaluation Summary: {model_name}",
        "=" * 40,
        f"MAE:   {metrics.get('MAE', 'N/A'):.4f}" if isinstance(metrics.get('MAE'), (int, float)) else f"MAE:   N/A",
        f"RMSE:  {metrics.get('RMSE', 'N/A'):.4f}" if isinstance(metrics.get('RMSE'), (int, float)) else f"RMSE:  N/A",
        f"MAPE:  {metrics.get('MAPE', 'N/A'):.2f}%" if isinstance(metrics.get('MAPE'), (int, float)) else f"MAPE:  N/A",
        f"R²:    {metrics.get('R2', 'N/A'):.4f}" if isinstance(metrics.get('R2'), (int, float)) else f"R²:    N/A",
        "=" * 40
    ]
    return "\n".join(lines)


 
#  DEMO / TEST


if __name__ == "__main__":
    print("\n Demo: Weather Prediction Model Evaluation Report")
    print("=" * 60)
    
    # Simulate predictions from 3 models
    np.random.seed(42)
    n_samples = 100
    
    # True rainfall values (mm)
    y_true = np.random.exponential(scale=5, size=n_samples)
    
    # Simulated predictions
    y_pred_xgb = y_true + np.random.normal(0, 1.5, n_samples)
    y_pred_cat = y_true + np.random.normal(0, 1.2, n_samples)
    y_pred_rf = y_true + np.random.normal(0, 2.0, n_samples)
    
    # Create report generator
    generator = EvaluationReportGenerator(output_dir="demo_reports")
    
    # Add model results
    generator.add_model_result(
        model_name="XGBoost",
        y_true=y_true,
        y_pred=y_pred_xgb,
        training_time=45.2,
        n_features=25,
        hyperparameters={"max_depth": 6, "learning_rate": 0.1},
        include_weather_metrics=True
    )
    
    generator.add_model_result(
        model_name="CatBoost",
        y_true=y_true,
        y_pred=y_pred_cat,
        training_time=38.5,
        n_features=25,
        hyperparameters={"depth": 6, "iterations": 1000},
        include_weather_metrics=True
    )
    
    generator.add_model_result(
        model_name="Random Forest",
        y_true=y_true,
        y_pred=y_pred_rf,
        training_time=12.3,
        n_features=25,
        hyperparameters={"n_estimators": 100, "max_depth": None},
        include_weather_metrics=True
    )
    
    # Generate full report
    generator.generate_full_report(
        title="Weather Prediction Model Comparison",
        dataset_name="Vietnam Weather 2024-2025",
        target_column="rainfall (mm)"
    )
    
    # Print metrics table
    print("\n Metrics DataFrame:")
    print(generator.get_metrics_dataframe())
    
    # Print ranking
    print("\n Model Rankings:")
    print(generator.get_ranking_table())