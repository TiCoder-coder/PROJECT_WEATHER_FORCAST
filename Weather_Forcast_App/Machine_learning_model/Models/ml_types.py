from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class TrainingResult:
    success: bool = True
    message: str = ""
    metrics: Optional[Dict[str, Any]] = None
    best_params: Optional[Dict[str, Any]] = None
    history: Optional[List[Dict[str, Any]]] = None
    training_time: Optional[float] = None
    n_samples: Optional[int] = None
    n_features: Optional[int] = None
    feature_names: Optional[List[str]] = None
    feature_importances: Optional[Any] = None
    best_iteration: Optional[int] = None

@dataclass
class PredictionResult:
    predictions: Any = None
    probabilities: Any = None
    message: str = ""
    timestamp: Optional[float] = None
