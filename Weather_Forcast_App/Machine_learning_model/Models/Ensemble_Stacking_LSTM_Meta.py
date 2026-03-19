# Weather_Forcast_App.Machine_learning_model.Models.Ensemble_Stacking_LSTM_Meta
# =============================================================================
# STACKING ENSEMBLE — LSTM META-CLASSIFIER + LINEAR REGRESSION META-REGRESSOR
# =============================================================================
#
# Mở rộng StackingEnsemble bằng cách thay thế meta-models bằng:
#   • Meta-Classifier : LSTM (học temporal patterns trong Z_cls OOF predictions)
#   • Meta-Regressor  : LinearRegression + interaction features (thay Ridge alpha=1.0)
#
# Architecture:
#
#  ┌──────────────────────────────────────────────────────────────────────────┐
#  │  BASE LAYER  (kế thừa từ StackingEnsemble)                              │
#  │                                                                          │
#  │  XGBCls │ RFCls │ CatCls │ LGBMCls                                      │
#  │       └──────────┬──────────┘                                            │
#  │          Z_cls (OOF TimeSeriesSplit, shape n × 4)                        │
#  │                  │                                                        │
#  │          sliding window: (n, seq_len=24, 4)  ← temporal context          │
#  │                  └──→ Bidirectional LSTM → Dense → sigmoid  → p_rain     │
#  ├──────────────────────────────────────────────────────────────────────────┤
#  │  XGBReg │ RFReg │ CatReg │ LGBMReg                                      │
#  │       └──────────┬──────────┘                                            │
#  │          Z_reg (OOF rainy-only, shape n_rainy × 4)                       │
#  │          augment: [Z, Z², cross-products Z_i*Z_j] → shape (n_rainy, 14)  │
#  │                  └──→ LinearRegression (OLS) ─────────→ log1p(mm)        │
#  ├──────────────────────────────────────────────────────────────────────────┤
#  │  INFERENCE (kế thừa từ StackingEnsemble)                                 │
#  │  p_rain > threshold → expm1(linear_reg(augment(Z_reg(X))))               │
#  │  else               → 0.0 mm                                             │
#  └──────────────────────────────────────────────────────────────────────────┘
#
# Giải thích lý do chọn từng component:
#
#   LSTM meta-cls:
#     - Z_cls có 75 278 rows theo thứ tự thời gian (TimeSeriesSplit)
#     - Mỗi row i là vector 4 xác suất từ 4 base classifiers tại thời điểm i
#     - LSTM học temporal pattern: "XGB + LGBM cùng tăng liên tiếp 24h → signal mạnh"
#     - seq_len=24 → mỗi prediction nhìn back 24 time steps
#
#   LinearRegression (thay Ridge):
#     - Ridge alpha=1.0 → R²=-3.28 (validation) — tệ hơn predict mean
#     - Ridge shrink coef → bias khi base models đã aligned cao
#     - OLS (LinearRegression) là BLUE estimator (Gauss-Markov) — unbiased
#     - Interaction terms Z_i*Z_j capture "khi xgb VÀ lgbm cùng cao → boost"
#
# LSTM Backend (ưu tiên theo thứ tự):
#   1. TensorFlow / Keras LSTM (nếu có)
#   2. PyTorch LSTM (nếu có)
#   3. sklearn MLPClassifier với temporal features (fallback luôn hoạt động)
#
# Cách sử dụng:
#
#   # Option 1: Train từ đầu (thay StackingEnsemble)
#   from Weather_Forcast_App.Machine_learning_model.Models.Ensemble_Stacking_LSTM_Meta import (
#       LSTMStackingEnsemble
#   )
#   stacking = LSTMStackingEnsemble(n_splits=5, predict_threshold=0.4,
#                                   lstm_seq_len=24, lstm_units=64)
#   result = stacking.fit(X_train, y_train, X_val=X_val, y_val=y_val)
#   y_pred = stacking.predict(X_test)
#   stacking.save("path/to/model.pkl")
#
#   # Option 2: Retrofit meta-models lên base models đã train (KHÔNG retrain base)
#   from Weather_Forcast_App.Machine_learning_model.Models.Ensemble_Stacking_LSTM_Meta import (
#       LSTMMetaPredictor
#   )
#   predictor = LSTMMetaPredictor.from_artifacts(
#       artifacts_dir="Weather_Forcast_App/Machine_learning_artifacts/stacking_ensemble/latest",
#       train_csv="...Dataset_after_split/Dataset_merge/merge_train.csv",
#   )
#   predictor.save_meta_models("lstm_meta_models/")
#   y_pred = predictor.predict(X_new)
#
# =============================================================================

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from Weather_Forcast_App.Machine_learning_model.Models.Ensemble_Stacking_Model import (
    StackingEnsemble,
    StackingResult,
)
from Weather_Forcast_App.Machine_learning_model.evaluation.metrics import RAIN_THRESHOLD

logger = logging.getLogger(__name__)

# =============================================================================
# Deep-learning backend detection
# =============================================================================

_TF_OK: bool = False
_TORCH_OK: bool = False

try:
    import tensorflow as tf  # type: ignore
    _TF_OK = True
    logger.debug("LSTMMetaClassifier: TensorFlow %s available", tf.__version__)
except ImportError:
    pass

if not _TF_OK:
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        import torch.optim as optim  # type: ignore
        from torch.utils.data import DataLoader, TensorDataset  # type: ignore
        _TORCH_OK = True
        logger.debug("LSTMMetaClassifier: PyTorch %s available", torch.__version__)
    except ImportError:
        pass

if not _TF_OK and not _TORCH_OK:
    logger.warning(
        "LSTMMetaClassifier: không tìm thấy TensorFlow hoặc PyTorch. "
        "Dùng sklearn MLPClassifier làm fallback (temporal features thủ công). "
        "Để có LSTM thực sự: pip install tensorflow  hoặc  pip install torch"
    )


# =============================================================================
# CLASS 1 — LSTMMetaClassifier (sklearn-compatible)
# =============================================================================

class LSTMMetaClassifier:
    """
    LSTM meta-classifier cho Stacking Ensemble.

    Input : Z_cls (n, n_models) — OOF predictions từ 4 base classifiers
    Output: predict_proba returns (n, 2) — cột 1 là p_rain

    Kỹ thuật:
    - Xây sliding-window sequences: row i → Z_cls[i-seq_len+1 : i+1] (left-zero-pad)
    - LSTM học temporal dependency giữa các thời điểm liên tiếp
    - Class imbalance: pos_weight = n_neg / n_pos   (dry >> rain)
    - EarlyStopping trên val_loss (10% last samples) → patience=8

    sklearn API:
        fit(Z_cls, y_cls) → self
        predict_proba(Z_cls) → ndarray (n, 2)

    Context buffer (streaming inference):
        Gọi push_to_buffer(z_row) sau mỗi prediction để tích lũy context.
        Buffer tự giữ tối đa seq_len-1 rows gần nhất.

    Backend priority:
        1. TensorFlow Keras (nếu có) → BiLSTM + LSTM → sigmoid
        2. PyTorch (nếu có)          → BiLSTM + LSTM → sigmoid
        3. sklearn MLPClassifier     → temporal statistics features

    Fallback (sklearn) luôn hoạt động ngay cả khi không có DL framework.
    """

    def __init__(
        self,
        seq_len: int = 24,
        lstm_units: int = 64,
        dropout: float = 0.2,
        epochs: int = 50,
        batch_size: int = 512,
        patience: int = 8,
        seed: int = 42,
        verbose: int = 0,
    ):
        self.seq_len = seq_len
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed
        self.verbose = verbose

        self._model: Any = None
        self._scaler: Optional[StandardScaler] = None
        self._backend: str = ""
        self._n_features: int = 0
        self._is_fitted: bool = False
        self._torch_device: Any = None

        # Context buffer cho single-sample streaming inference
        self._context_buffer: Deque[np.ndarray] = deque(maxlen=max(seq_len - 1, 1))
        self._buffer_enabled: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # Sequence utilities
    # ─────────────────────────────────────────────────────────────────────────

    def _build_sequences(
        self,
        Z: np.ndarray,
        prepend: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Chuyển Z (n, k) → sequences (n, seq_len, k) với zero left-padding.

        Mỗi output row i chứa tối đa seq_len rows gần nhất kết thúc tại i.
        Nếu i < seq_len, phần đầu được zero-pad.

        Args:
            Z:       Input OOF matrix (n, k)
            prepend: Optional context rows (c, k) đặt trước Z
                     Dùng để cấp context khi predict batch nhỏ (ví dụ fold-5 eval)
        Returns:
            seqs: np.ndarray shape (n, seq_len, k), dtype float32
        """
        if prepend is not None and len(prepend) > 0:
            Z_full = np.vstack([prepend, Z]).astype(np.float32)
            offset = len(prepend)
        else:
            Z_full = Z.astype(np.float32)
            offset = 0

        n_out = len(Z)
        k = Z_full.shape[1]
        seqs = np.zeros((n_out, self.seq_len, k), dtype=np.float32)

        for i in range(n_out):
            pos = i + offset                          # vị trí trong Z_full
            start = max(0, pos - self.seq_len + 1)
            length = pos - start + 1
            seqs[i, self.seq_len - length:] = Z_full[start: pos + 1]

        return seqs

    def _extract_temporal_features(
        self,
        Z: np.ndarray,
        prepend: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Sklearn fallback: trích xuất temporal statistics từ Z.

        Với mỗi vị trí i, tính trên cửa sổ Z[max(0,i-seq_len+1):i+1]:
          [last_val, mean, std, trend(last-first), min, max] × k_models
          + agreement pairs: last_val[a] - last_val[b] với mọi a<b

        Returns:
            X_feat: shape (n, k*6 + C(k,2))
        """
        if prepend is not None and len(prepend) > 0:
            Z_full = np.vstack([prepend, Z])
            offset = len(prepend)
        else:
            Z_full = Z
            offset = 0

        n_out = len(Z)
        k = Z_full.shape[1]
        n_stats = 6
        n_pairs = k * (k - 1) // 2
        X_feat = np.zeros((n_out, k * n_stats + n_pairs), dtype=np.float32)

        for i in range(n_out):
            pos = i + offset
            start = max(0, pos - self.seq_len + 1)
            window = Z_full[start: pos + 1]           # (window_len, k)
            for j in range(k):
                w = window[:, j]
                off = j * n_stats
                X_feat[i, off + 0] = float(w[-1])
                X_feat[i, off + 1] = float(w.mean())
                X_feat[i, off + 2] = float(w.std()) if len(w) > 1 else 0.0
                X_feat[i, off + 3] = float(w[-1] - w[0]) if len(w) > 1 else 0.0
                X_feat[i, off + 4] = float(w.min())
                X_feat[i, off + 5] = float(w.max())
            # Agreement pairs (disagreement signal giữa base models)
            last_vals = Z_full[pos]
            pair_off = k * n_stats
            idx = 0
            for a in range(k):
                for b in range(a + 1, k):
                    X_feat[i, pair_off + idx] = float(last_vals[a] - last_vals[b])
                    idx += 1

        return X_feat

    # ─────────────────────────────────────────────────────────────────────────
    # TensorFlow backend
    # ─────────────────────────────────────────────────────────────────────────

    def _build_tf_model(self, n_features: int):
        """Xây Bidirectional LSTM → LSTM → Dense(sigmoid) với Keras."""
        import tensorflow as tf  # type: ignore

        tf.random.set_seed(self.seed)
        os.environ.setdefault("PYTHONHASHSEED", str(self.seed))

        inp = tf.keras.Input(shape=(self.seq_len, n_features), name="Z_cls_seq")

        # Lớp 1: Bidirectional LSTM để capture cả forward và backward temporal patterns
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                self.lstm_units,
                return_sequences=True,
                dropout=self.dropout,
                recurrent_dropout=0.1,
                kernel_initializer="glorot_uniform",
            ),
            name="bilstm_1",
        )(inp)

        # Lớp 2: LSTM để reduce xuống hidden state cuối
        x = tf.keras.layers.LSTM(
            self.lstm_units // 2,
            dropout=self.dropout,
            name="lstm_2",
        )(x)

        # Dense head
        x = tf.keras.layers.Dense(32, activation="relu", name="dense_1")(x)
        x = tf.keras.layers.BatchNormalization(name="bn_1")(x)
        x = tf.keras.layers.Dropout(self.dropout, name="dropout_1")(x)
        out = tf.keras.layers.Dense(1, activation="sigmoid", name="p_rain")(x)

        model = tf.keras.Model(inputs=inp, outputs=out, name="LSTMMetaCls")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def _fit_tf(self, Z_cls: np.ndarray, y_cls: np.ndarray) -> None:
        import tensorflow as tf  # type: ignore

        seqs = self._build_sequences(Z_cls.astype(np.float32))

        # Class imbalance handling
        n_pos = int(y_cls.sum())
        n_neg = len(y_cls) - n_pos
        class_weight = {0: 1.0, 1: max(float(n_neg) / max(n_pos, 1), 1.0)}

        self._model = self._build_tf_model(n_features=Z_cls.shape[1])

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                restore_best_weights=True,
                verbose=0,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=self.patience // 2,
                min_lr=1e-5,
                verbose=0,
            ),
        ]

        self._model.fit(
            seqs,
            y_cls.astype(np.float32),
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,       # 10% cuối làm val (không shuffle → time-ordered)
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=self.verbose,
            shuffle=False,              # QUAN TRỌNG: time-series, không shuffle
        )

    def _predict_tf(
        self, Z_cls: np.ndarray, prepend: Optional[np.ndarray] = None
    ) -> np.ndarray:
        seqs = self._build_sequences(Z_cls.astype(np.float32), prepend=prepend)
        p = self._model.predict(seqs, verbose=0, batch_size=self.batch_size)
        return p.reshape(-1)

    # ─────────────────────────────────────────────────────────────────────────
    # PyTorch backend
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_torch_device():
        import torch  # type: ignore
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _fit_torch(self, Z_cls: np.ndarray, y_cls: np.ndarray) -> None:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        import torch.optim as optim  # type: ignore
        from torch.utils.data import DataLoader, TensorDataset  # type: ignore

        device = self._get_torch_device()
        self._torch_device = device
        torch.manual_seed(self.seed)

        n_feat = Z_cls.shape[1]
        units = self.lstm_units

        # ─── Định nghĩa module bên trong hàm để tránh circular dependency ───
        class _LSTMClsNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.bilstm = nn.LSTM(
                    n_feat,
                    units,
                    batch_first=True,
                    bidirectional=True,
                    num_layers=1,
                )
                self.lstm2 = nn.LSTM(
                    units * 2,
                    units // 2,
                    batch_first=True,
                )
                self.fc1 = nn.Linear(units // 2, 32)
                self.bn = nn.BatchNorm1d(32)
                self.drop = nn.Dropout(0.2)
                self.fc2 = nn.Linear(32, 1)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):                         # x: (batch, seq, n_feat)
                out, _ = self.bilstm(x)                   # (batch, seq, units*2)
                out, (h, _) = self.lstm2(out)             # h: (1, batch, units//2)
                h = h.squeeze(0)                          # (batch, units//2)
                out = self.drop(self.relu(self.bn(self.fc1(h))))
                logit = self.fc2(out).squeeze(-1)         # (batch,)
                return self.sigmoid(logit)

        seqs_np = self._build_sequences(Z_cls.astype(np.float32))
        n = len(seqs_np)
        n_val = max(1, int(n * 0.1))
        X_tr_t = torch.tensor(seqs_np[:-n_val], dtype=torch.float32)
        y_tr_t = torch.tensor(y_cls[:-n_val].astype(np.float32), dtype=torch.float32)
        X_val_t = torch.tensor(seqs_np[-n_val:], dtype=torch.float32)
        y_val_t = torch.tensor(y_cls[-n_val:].astype(np.float32), dtype=torch.float32)

        n_pos = int(y_tr_t.sum().item())
        n_neg = int(len(y_tr_t)) - n_pos
        pos_weight = torch.tensor([float(n_neg) / max(n_pos, 1)]).to(device)

        model = _LSTMClsNet().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=self.patience // 2, factor=0.5
        )

        train_ds = TensorDataset(X_tr_t, y_tr_t)
        loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=False)

        best_val_loss = float("inf")
        best_state: Optional[Dict] = None
        patience_cnt = 0

        for epoch in range(self.epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                p = model(xb)
                # Chuyển sigmoid output → logit để dùng BCEWithLogitsLoss
                eps = 1e-7
                logit = torch.log(p.clamp(eps, 1 - eps) / (1 - p.clamp(eps, 1 - eps)))
                loss = criterion(logit, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                p_val = model(X_val_t.to(device))
                eps = 1e-7
                logit_val = torch.log(
                    p_val.clamp(eps, 1 - eps) / (1 - p_val.clamp(eps, 1 - eps))
                )
                val_loss = criterion(logit_val, y_val_t.to(device)).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self._model = model

    def _predict_torch(
        self, Z_cls: np.ndarray, prepend: Optional[np.ndarray] = None
    ) -> np.ndarray:
        import torch  # type: ignore

        device = self._torch_device or self._get_torch_device()
        seqs_np = self._build_sequences(Z_cls.astype(np.float32), prepend=prepend)
        seqs_t = torch.tensor(seqs_np, dtype=torch.float32).to(device)
        with torch.no_grad():
            p = self._model(seqs_t).cpu().numpy().reshape(-1)
        return p

    # ─────────────────────────────────────────────────────────────────────────
    # sklearn MLP fallback
    # ─────────────────────────────────────────────────────────────────────────

    def _fit_sklearn(self, Z_cls: np.ndarray, y_cls: np.ndarray) -> None:
        X_feat = self._extract_temporal_features(Z_cls)
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_feat)

        self._model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            max_iter=400,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=self.patience,
            random_state=self.seed,
            verbose=False,
        )
        self._model.fit(X_scaled, y_cls)

    def _predict_sklearn(
        self, Z_cls: np.ndarray, prepend: Optional[np.ndarray] = None
    ) -> np.ndarray:
        X_feat = self._extract_temporal_features(Z_cls, prepend=prepend)
        X_scaled = self._scaler.transform(X_feat)
        return self._model.predict_proba(X_scaled)[:, 1]

    # ─────────────────────────────────────────────────────────────────────────
    # Public sklearn API
    # ─────────────────────────────────────────────────────────────────────────

    def fit(self, Z_cls: np.ndarray, y_cls: np.ndarray) -> "LSTMMetaClassifier":
        """
        Train LSTM meta-classifier.

        Args:
            Z_cls: OOF predictions shape (n, n_models), range 0-1
            y_cls: Binary labels (0=khô, 1=mưa), shape (n,)

        Returns:
            self (sklearn-compatible)
        """
        Z_cls = np.asarray(Z_cls, dtype=np.float64)
        y_cls = np.asarray(y_cls, dtype=np.int32)
        self._n_features = Z_cls.shape[1]

        t0 = time.time()
        if _TF_OK:
            self._backend = "tensorflow"
            self._fit_tf(Z_cls, y_cls)
        elif _TORCH_OK:
            self._backend = "torch"
            self._fit_torch(Z_cls, y_cls)
        else:
            self._backend = "sklearn_mlp"
            self._fit_sklearn(Z_cls, y_cls)

        self._is_fitted = True
        logger.info(
            "LSTMMetaClassifier.fit() OK — backend=%s, n=%d, seq_len=%d, time=%.1fs",
            self._backend, len(Z_cls), self.seq_len, time.time() - t0,
        )
        return self

    def predict_proba(self, Z_cls: np.ndarray) -> np.ndarray:
        """
        Predict xác suất có/không mưa.

        Args:
            Z_cls: shape (n, n_models)

        Returns:
            proba: shape (n, 2) — cột 0 = p_dry, cột 1 = p_rain
                   (tương thích với sklearn predict_proba interface)
        """
        if not self._is_fitted:
            raise RuntimeError("LSTMMetaClassifier chưa fit(). Gọi fit() trước.")

        Z_cls = np.asarray(Z_cls, dtype=np.float64)
        prepend = (
            np.array(list(self._context_buffer), dtype=np.float64)
            if self._buffer_enabled and len(self._context_buffer) > 0
            else None
        )

        if self._backend == "tensorflow":
            p = self._predict_tf(Z_cls, prepend=prepend)
        elif self._backend == "torch":
            p = self._predict_torch(Z_cls, prepend=prepend)
        else:
            p = self._predict_sklearn(Z_cls, prepend=prepend)

        p = np.clip(p, 1e-7, 1.0 - 1e-7)
        return np.column_stack([1.0 - p, p])

    def push_to_buffer(self, z_row: np.ndarray) -> None:
        """
        Đẩy 1 hàng Z vào context buffer (dùng khi inference từng sample).
        Buffer giữ tối đa seq_len-1 rows gần nhất (FIFO).

        Ví dụ:
            for i, x in enumerate(X_stream):
                z = get_base_predictions(x)  # (n_models,)
                p = predictor.meta_cls.predict_proba(z.reshape(1, -1))[0, 1]
                predictor.meta_cls.push_to_buffer(z)
        """
        self._buffer_enabled = True
        self._context_buffer.append(np.asarray(z_row, dtype=np.float64))

    def clear_buffer(self) -> None:
        """Xóa context buffer (bắt đầu sequence mới)."""
        self._context_buffer.clear()
        self._buffer_enabled = False

    def save(self, path: Union[str, Path]) -> None:
        """Lưu model ra file pickle (hoặc SavedModel cho TF)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self._backend == "tensorflow":
            import tensorflow as tf  # type: ignore
            tf_dir = path.with_suffix("")
            self._model.save(str(tf_dir))
            meta = {
                "backend": "tensorflow",
                "seq_len": self.seq_len,
                "lstm_units": self.lstm_units,
                "dropout": self.dropout,
                "n_features": self._n_features,
                "tf_model_dir": str(tf_dir),
            }
            with open(str(path) + ".meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        else:
            joblib.dump(self, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LSTMMetaClassifier":
        """Load model từ file."""
        path = Path(path)
        meta_path = Path(str(path) + ".meta.json")

        if meta_path.exists() and _TF_OK:
            import tensorflow as tf  # type: ignore
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            obj = cls(
                seq_len=meta["seq_len"],
                lstm_units=meta["lstm_units"],
                dropout=meta["dropout"],
            )
            obj._backend = "tensorflow"
            obj._n_features = meta["n_features"]
            obj._model = tf.keras.models.load_model(meta["tf_model_dir"])
            obj._is_fitted = True
            return obj

        return joblib.load(path)


# =============================================================================
# CLASS 2 — LinearInteractionMetaRegressor (sklearn-compatible)
# =============================================================================

class LinearInteractionMetaRegressor:
    """
    OLS Linear Regression meta-regressor với interaction features.

    Input : Z_reg (n, n_models) — OOF predictions log1p(mm) từ base regressors
    Output: log1p(mm) predictions

    Feature augmentation (n, k) → (n, 2k + C(k,2)):
        k=4: (n, 4) → (n, 4+4+6) = (n, 14)
        Columns: [Z (4)] + [Z² (4)] + [Z_i*Z_j for i<j (6)]

    Lý do dùng OLS thay Ridge:
        Ridge alpha=1.0 → coef shrinkage → bias khi base models đã aligned
        → R² validation = -3.28 (tệ hơn predict mean)
        OLS là BLUE (Best Linear Unbiased Estimator) theo Gauss-Markov

    Interaction terms giải thích:
        Z_i * Z_j ≈ "cả XGB lẫn LGBM đều dự báo cao" → tín hiệu mạnh hơn
        Giúp model học synergistic agreement giữa base models

    sklearn API:
        fit(Z_reg, y_reg) → self
        predict(Z_reg)    → ndarray (n,)
        coef_             → property (tương thích StackingResult)
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        normalize_inputs: bool = True,
    ):
        self.fit_intercept = fit_intercept
        self.normalize_inputs = normalize_inputs

        self._reg: Optional[LinearRegression] = None
        self._scaler: Optional[StandardScaler] = None
        self._n_features: int = 0
        self._n_aug_features: int = 0
        self._is_fitted: bool = False

    def _augment(self, Z: np.ndarray) -> np.ndarray:
        """
        Augment Z với squared terms và cross-product interactions.

        (n, k) → (n, 2k + C(k,2))
        Columns:
            - Z[:,j]           : k original features
            - Z[:,j]**2        : k squared features
            - Z[:,i]*Z[:,j]    : C(k,2) cross-products (i < j)
        """
        n, k = Z.shape
        parts: List[np.ndarray] = [Z, Z ** 2]
        for i in range(k):
            for j in range(i + 1, k):
                parts.append((Z[:, i] * Z[:, j]).reshape(-1, 1))
        return np.hstack(parts)

    def fit(
        self, Z_reg: np.ndarray, y_reg: np.ndarray
    ) -> "LinearInteractionMetaRegressor":
        """
        Train LinearRegression trên augmented Z_reg.

        Args:
            Z_reg: OOF predictions shape (n, n_models), đơn vị log1p(mm)
            y_reg: Target log1p(mm), shape (n,)

        Returns:
            self
        """
        Z_reg = np.asarray(Z_reg, dtype=np.float64)
        y_reg = np.asarray(y_reg, dtype=np.float64).reshape(-1)
        self._n_features = Z_reg.shape[1]

        Z_aug = self._augment(Z_reg)
        self._n_aug_features = Z_aug.shape[1]

        if self.normalize_inputs:
            self._scaler = StandardScaler()
            Z_aug = self._scaler.fit_transform(Z_aug)

        self._reg = LinearRegression(fit_intercept=self.fit_intercept)
        self._reg.fit(Z_aug, y_reg)
        self._is_fitted = True

        train_r2 = self._reg.score(Z_aug, y_reg)
        logger.info(
            "LinearInteractionMetaRegressor.fit() OK — n=%d, aug_features=%d, train_R²=%.4f",
            len(y_reg), self._n_aug_features, train_r2,
        )
        return self

    def predict(self, Z_reg: np.ndarray) -> np.ndarray:
        """
        Predict log1p(mm) từ Z_reg.

        Args:
            Z_reg: shape (n, n_models)

        Returns:
            preds: shape (n,), đơn vị log1p(mm)
        """
        if not self._is_fitted:
            raise RuntimeError("LinearInteractionMetaRegressor chưa fit().")

        Z_reg = np.asarray(Z_reg, dtype=np.float64)
        Z_aug = self._augment(Z_reg)

        if self.normalize_inputs and self._scaler is not None:
            Z_aug = self._scaler.transform(Z_aug)

        return self._reg.predict(Z_aug)

    @property
    def coef_(self) -> Optional[np.ndarray]:
        """sklearn compatibility: trả về coef của LinearRegression bên trong."""
        if self._is_fitted and self._reg is not None:
            return self._reg.coef_
        return None

    @property
    def intercept_(self) -> Optional[float]:
        """sklearn compatibility: intercept của LinearRegression."""
        if self._is_fitted and self._reg is not None:
            return float(self._reg.intercept_)
        return None

    @property
    def feature_names_augmented(self) -> List[str]:
        """Tên các augmented features (debug / explainability)."""
        k = self._n_features
        names: List[str] = [f"z{i}" for i in range(k)]
        names += [f"z{i}^2" for i in range(k)]
        for i in range(k):
            for j in range(i + 1, k):
                names.append(f"z{i}*z{j}")
        return names

    def coef_summary(self) -> str:
        """In tóm tắt hệ số hồi quy (debug / explainability)."""
        if not self._is_fitted:
            return "LinearInteractionMetaRegressor chưa fit()"
        coefs = self.coef_
        names = self.feature_names_augmented
        lines = ["LinearInteractionMetaRegressor — coefficients:"]
        for name, c in zip(names, coefs):
            lines.append(f"  {name:20s}: {c:+.6f}")
        return "\n".join(lines)

    def save(self, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LinearInteractionMetaRegressor":
        return joblib.load(path)


# =============================================================================
# CLASS 3 — LSTMStackingEnsemble (kế thừa StackingEnsemble)
# =============================================================================

class LSTMStackingEnsemble(StackingEnsemble):
    """
    Stacking Ensemble với LSTM meta-classifier + Linear Regression meta-regressor.

    Kế thừa toàn bộ từ StackingEnsemble:
        - Stage 6: Verify base models
        - Stage 7: OOF Classification → build Z_cls
        - Stage 8: OOF Regression     → build Z_reg
        - Stage 9: Refit base models trên full train
        - predict(), predict_full(), evaluate(), save(), load()

    Thay đổi DUY NHẤT:
        self.meta_cls: LogisticRegression  → LSTMMetaClassifier
        self.meta_reg: Ridge(alpha=1.0)    → LinearInteractionMetaRegressor

    Compatibility:
        - parent.fit() gọi meta_cls.fit(Z_cls[mask], y_cls[mask]) → ✓ tương thích
        - parent._get_cls_stack() gọi meta_cls.predict_proba(Z)[:,1] → ✓ tương thích
        - parent._get_reg_stack() gọi meta_reg.predict(Z) → ✓ tương thích
        - StackingResult.meta_cls_coef: hasattr(meta_cls, 'coef_') = False → None ✓
        - StackingResult.meta_reg_coef: meta_reg.coef_ → ndarray ✓ (property)

    Khi nào nên dùng:
        - Dữ liệu train lớn (>50K samples) → LSTM có đủ data để học
        - Base models đã ổn định, muốn cải thiện combination layer
        - Khi Ridge cho R² âm → LinearInteraction giải quyết vấn đề bias

    Khi nào KHÔNG nên dùng:
        - Dữ liệu nhỏ (<10K) → LSTM overfit, dùng LogisticRegression tốt hơn
        - Không có GPU và cần train nhanh → sklearn fallback vẫn hoạt động
        - Production environment không có TF/PyTorch → sklearn MLP fallback

    Example:
        stacking = LSTMStackingEnsemble(
            n_splits=5,
            predict_threshold=0.4,
            lstm_seq_len=24,
            lstm_units=64,
            lstm_epochs=50,
        )
        result = stacking.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        print(stacking.summary())
        y_pred = stacking.predict(X_test)
        stacking.save("path/to/model.pkl")
    """

    def __init__(
        self,
        n_splits: int = 5,
        predict_threshold: float = 0.4,
        rain_threshold: float = RAIN_THRESHOLD,
        seed: int = 42,
        cls_params: Optional[Dict[str, Any]] = None,
        reg_params: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        # LSTM meta-classifier params
        lstm_seq_len: int = 24,
        lstm_units: int = 64,
        lstm_dropout: float = 0.2,
        lstm_epochs: int = 50,
        lstm_batch_size: int = 512,
        lstm_patience: int = 8,
        # Linear meta-regressor params
        linear_normalize_inputs: bool = True,
        linear_fit_intercept: bool = True,
    ):
        # Gọi parent.__init__ (thiết lập base models, meta_{cls,reg} ban đầu)
        super().__init__(
            n_splits=n_splits,
            predict_threshold=predict_threshold,
            rain_threshold=rain_threshold,
            seed=seed,
            cls_params=cls_params,
            reg_params=reg_params,
            meta_cls_params=None,   # sẽ bị override ngay bên dưới
            meta_reg_params=None,
            verbose=verbose,
        )

        # Override meta-models bằng LSTM + Linear
        self.meta_cls = LSTMMetaClassifier(
            seq_len=lstm_seq_len,
            lstm_units=lstm_units,
            dropout=lstm_dropout,
            epochs=lstm_epochs,
            batch_size=lstm_batch_size,
            patience=lstm_patience,
            seed=seed,
            verbose=1 if verbose else 0,
        )
        self.meta_reg = LinearInteractionMetaRegressor(
            fit_intercept=linear_fit_intercept,
            normalize_inputs=linear_normalize_inputs,
        )

        # Lưu params để serialize / summary
        self.lstm_seq_len = lstm_seq_len
        self.lstm_units = lstm_units
        self.lstm_dropout = lstm_dropout
        self.lstm_epochs = lstm_epochs
        self.linear_normalize_inputs = linear_normalize_inputs

    def summary(self) -> str:
        """Mở rộng summary() của parent với thông tin LSTM + Linear."""
        base_lines = super().summary()
        # Thay tên class trong dòng header
        base_lines = base_lines.replace(
            "StackingEnsemble Summary", "LSTMStackingEnsemble Summary"
        )
        # Thêm LSTM / Linear info vào cuối (trước dòng "=" cuối)
        extra = (
            f"\n  LSTM backend       : {self.meta_cls._backend or 'not fitted'}"
            f"\n  LSTM seq_len       : {self.lstm_seq_len}"
            f"\n  LSTM units         : {self.lstm_units}"
            f"\n  LinearReg features : {self.meta_reg._n_aug_features or '(not fitted)'}"
        )
        # Chèn trước dòng "=" cuối
        sep = "=" * 70
        if base_lines.endswith(sep):
            return base_lines[: -len(sep)] + extra + "\n" + sep
        return base_lines + extra

    def __repr__(self) -> str:
        status = "trained" if self.is_trained else "untrained"
        backend = getattr(self.meta_cls, "_backend", "?")
        return (
            f"LSTMStackingEnsemble({status}, "
            f"n_splits={self.n_splits}, "
            f"threshold={self.predict_threshold}, "
            f"lstm_backend={backend!r}, "
            f"seq_len={self.lstm_seq_len})"
        )


# =============================================================================
# CLASS 4 — LSTMMetaPredictor (predictor độc lập từ artifacts)
# =============================================================================

@dataclass
class LSTMMetaArtifacts:
    """Thông tin sau khi train/load LSTM meta-models từ artifacts."""

    artifacts_dir: str = ""
    train_csv_path: str = ""
    n_features: int = 0
    n_cls_oof_samples: int = 0
    n_reg_oof_samples: int = 0
    meta_cls_backend: str = ""
    meta_cls_seq_len: int = 24
    # Metrics trên fold-5 holdout
    fold5_roc_auc: float = 0.0
    fold5_f1: float = 0.0
    fold5_pr_auc: float = 0.0
    fold5_reg_mae_mm: float = 0.0
    fold5_reg_r2: float = 0.0
    trained_at: str = ""
    note: str = ""


class LSTMMetaPredictor:
    """
    Predictor độc lập: đọc artifacts từ StackingEnsemble đã train,
    rebuild OOF meta-features, rồi train LSTM cls + Linear reg.

    Use-case chính:
        Retrofit meta-models tốt hơn lên base models đã train (tiết kiệm ~3h training).
        Không cần retrain XGB/RF/CatBoost/LGBM — chỉ retrain combination layer.

    Workflow:
        1. Load Model.pkl → extract final_cls_models (4 classifiers), final_reg_models (4 regressors)
        2. Load Feature_list.json → lấy feature columns, target_column
        3. Load Transform_pipeline.pkl → WeatherTransformPipeline
        4. Load train CSV → apply pipeline → build OOF Z_cls (n, 4), Z_reg (n_rainy, 4)
        5. Train LSTMMetaClassifier trên Z_cls
        6. Train LinearInteractionMetaRegressor trên Z_reg
        7. Sẵn sàng predict(X_new)

    Save / Load:
        predictor.save_meta_models("lstm_meta/")  → lưu 2 meta-models
        predictor = LSTMMetaPredictor.load_meta_models(
            artifacts_dir="...", meta_dir="lstm_meta/"
        )

    Predict interface (tương thích WeatherPredictor):
        y_pred_mm = predictor.predict(X_new)        # np.ndarray shape (n,)
        full      = predictor.predict_full(X_new)   # dict như StackingEnsemble

    Compare với original meta-models:
        metrics = predictor.compare_with_original(X_test, y_test, dataset_name="test")
        # → dict với keys "original" và "lstm" để so sánh side-by-side
    """

    # Tên file trong artifacts dir
    _MODEL_PKL = "Model.pkl"
    _PIPELINE_PKL = "Transform_pipeline.pkl"
    _FEATURE_JSON = "Feature_list.json"
    _METRICS_JSON = "Metrics.json"
    _TRAIN_INFO_JSON = "Train_info.json"

    # Tên file meta-models (trong meta_dir)
    _META_CLS_FILE = "lstm_meta_cls.pkl"
    _META_REG_FILE = "linear_meta_reg.pkl"
    _ARTIFACTS_INFO = "lstm_meta_artifacts.json"

    def __init__(self):
        self._stacking: Optional[StackingEnsemble] = None  # loaded base + meta skeleton
        self._pipeline: Any = None                          # WeatherTransformPipeline
        self._feature_columns: List[str] = []
        self._target_column: str = "rain_total"
        self._rain_threshold: float = RAIN_THRESHOLD
        self._predict_threshold: float = 0.4
        self._meta_cls: Optional[LSTMMetaClassifier] = None
        self._meta_reg: Optional[LinearInteractionMetaRegressor] = None
        self._artifacts_info: Optional[LSTMMetaArtifacts] = None
        self._is_ready: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # Factory: xây từ artifacts + train CSV
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_artifacts(
        cls,
        artifacts_dir: Union[str, Path],
        train_csv: Optional[Union[str, Path]] = None,
        n_splits: int = 5,
        lstm_seq_len: int = 24,
        lstm_units: int = 64,
        lstm_dropout: float = 0.2,
        lstm_epochs: int = 50,
        lstm_patience: int = 8,
        rain_threshold: float = RAIN_THRESHOLD,
        predict_threshold: float = 0.4,
        seed: int = 42,
        verbose: bool = True,
    ) -> "LSTMMetaPredictor":
        """
        Tải artifacts → rebuild OOF Z_cls/Z_reg → train LSTM meta-models.

        Args:
            artifacts_dir:    Đường dẫn thư mục chứa artifacts StackingEnsemble
                              (chứa Model.pkl, Transform_pipeline.pkl, Feature_list.json, ...)
            train_csv:        Đường dẫn train CSV để rebuild OOF.
                              Nếu None, sẽ thử đọc từ Train_info.json.
            n_splits:         Số fold TimeSeriesSplit để rebuild OOF
            lstm_seq_len:     Độ dài sequence LSTM
            lstm_units:       Số neurons LSTM
            lstm_dropout:     Dropout rate
            lstm_epochs:      Số epochs tối đa
            lstm_patience:    EarlyStopping patience
            rain_threshold:   Ngưỡng mm để coi là có mưa
            predict_threshold: Ngưỡng p_rain để predict mưa
            seed:             Random seed
            verbose:          In progress log

        Returns:
            LSTMMetaPredictor sẵn sàng predict
        """
        obj = cls()
        obj._rain_threshold = rain_threshold
        obj._predict_threshold = predict_threshold
        artifacts_dir = Path(artifacts_dir)

        def _log(msg: str) -> None:
            if verbose:
                print(msg)
            logger.info(msg)

        _log("\n" + "=" * 70)
        _log("LSTMMetaPredictor.from_artifacts()")
        _log("=" * 70)

        # ─── 1. Load artifacts ───
        _log(f"\n[1/6] Loading artifacts from: {artifacts_dir}")
        obj._stacking = obj._load_stacking(artifacts_dir)
        obj._pipeline = obj._load_pipeline(artifacts_dir)
        feature_data = obj._load_feature_list(artifacts_dir)
        obj._feature_columns = feature_data.get("all_feature_columns", [])
        obj._target_column = feature_data.get("target_column", "rain_total")
        train_info = obj._load_train_info(artifacts_dir)
        _log(f"  ✓ Loaded base models: {obj._stacking.cls_model_names} / {obj._stacking.reg_model_names}")
        _log(f"  ✓ Feature columns   : {len(obj._feature_columns)} features")

        # ─── 2. Resolve train CSV path ───
        _log("\n[2/6] Resolving train CSV path...")
        if train_csv is None:
            train_csv = obj._resolve_train_csv(train_info, artifacts_dir)
        if train_csv is None:
            raise FileNotFoundError(
                "Không tìm thấy train CSV. Truyền train_csv= hoặc kiểm tra Train_info.json."
            )
        train_csv = Path(train_csv)
        if not train_csv.exists():
            raise FileNotFoundError(f"Train CSV không tồn tại: {train_csv}")
        _log(f"  ✓ Train CSV: {train_csv}")

        # ─── 3. Load + transform train data ───
        _log("\n[3/6] Loading and transforming train data...")
        X_train, y_train = obj._load_and_transform(train_csv, verbose=verbose)
        _log(f"  ✓ X_train shape: {X_train.shape}, rainy: {(y_train > rain_threshold).sum()}")

        # ─── 4. Rebuild OOF meta-features ───
        _log(f"\n[4/6] Rebuilding OOF meta-features (n_splits={n_splits})...")
        t4 = time.time()
        Z_cls, Z_reg, y_cls, y_reg_log = obj._rebuild_oof(
            X_train, y_train, n_splits=n_splits, verbose=verbose
        )
        _log(f"  ✓ Z_cls: {Z_cls.shape}, Z_reg: {Z_reg.shape}  ({time.time()-t4:.1f}s)")

        # ─── 5. Train LSTM meta-classifier ───
        _log("\n[5/6] Training LSTM meta-classifier...")
        t5 = time.time()

        # Dùng folds 1..n_splits-1 để train meta, fold n (cuối) để eval
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cls_folds = list(tscv.split(Z_cls))
        fold_last_val_cls = cls_folds[-1][1]
        meta_train_mask_cls = np.ones(len(y_cls), dtype=bool)
        meta_train_mask_cls[fold_last_val_cls] = False

        obj._meta_cls = LSTMMetaClassifier(
            seq_len=lstm_seq_len,
            lstm_units=lstm_units,
            dropout=lstm_dropout,
            epochs=lstm_epochs,
            patience=lstm_patience,
            seed=seed,
            verbose=1 if verbose else 0,
        )
        obj._meta_cls.fit(Z_cls[meta_train_mask_cls], y_cls[meta_train_mask_cls])

        # Evaluate trên fold cuối
        eval_proba = obj._meta_cls.predict_proba(Z_cls[fold_last_val_cls])[:, 1]
        eval_y = y_cls[fold_last_val_cls]
        roc_auc = roc_auc_score(eval_y, eval_proba) if len(np.unique(eval_y)) > 1 else float("nan")
        pr_auc = average_precision_score(eval_y, eval_proba)
        f1 = f1_score(eval_y, (eval_proba > predict_threshold).astype(int), zero_division=0)
        _log(
            f"  ✓ Meta-cls fold-{n_splits} holdout: "
            f"ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}, F1={f1:.4f}  ({time.time()-t5:.1f}s)"
        )

        # ─── 6. Train Linear meta-regressor ───
        _log("\n[6/6] Training LinearInteraction meta-regressor...")
        t6 = time.time()

        reg_folds = list(tscv.split(Z_reg))
        fold_last_val_reg = reg_folds[-1][1]
        meta_train_mask_reg = np.ones(len(y_reg_log), dtype=bool)
        meta_train_mask_reg[fold_last_val_reg] = False

        obj._meta_reg = LinearInteractionMetaRegressor(
            normalize_inputs=True, fit_intercept=True
        )
        obj._meta_reg.fit(Z_reg[meta_train_mask_reg], y_reg_log[meta_train_mask_reg])

        # Evaluate trên fold cuối
        pred_log = obj._meta_reg.predict(Z_reg[fold_last_val_reg])
        pred_mm = np.expm1(pred_log).clip(min=0)
        true_mm = np.expm1(y_reg_log[fold_last_val_reg])
        mae_mm = mean_absolute_error(true_mm, pred_mm)
        r2 = r2_score(true_mm, pred_mm)
        _log(
            f"  ✓ Meta-reg fold-{n_splits} holdout (rainy): "
            f"MAE={mae_mm:.4f}mm, R²={r2:.4f}  ({time.time()-t6:.1f}s)"
        )

        # ─── Done ───
        obj._artifacts_info = LSTMMetaArtifacts(
            artifacts_dir=str(artifacts_dir),
            train_csv_path=str(train_csv),
            n_features=X_train.shape[1],
            n_cls_oof_samples=len(y_cls),
            n_reg_oof_samples=len(y_reg_log),
            meta_cls_backend=obj._meta_cls._backend,
            meta_cls_seq_len=lstm_seq_len,
            fold5_roc_auc=round(roc_auc, 4),
            fold5_f1=round(f1, 4),
            fold5_pr_auc=round(pr_auc, 4),
            fold5_reg_mae_mm=round(mae_mm, 4),
            fold5_reg_r2=round(r2, 4),
            trained_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        obj._is_ready = True

        _log(f"\n{'=' * 70}")
        _log("✅ LSTMMetaPredictor ready!")
        _log(f"{'=' * 70}\n")
        return obj

    # ─────────────────────────────────────────────────────────────────────────
    # Factory: load meta-models đã save (không train lại)
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def load_meta_models(
        cls,
        artifacts_dir: Union[str, Path],
        meta_dir: Union[str, Path],
    ) -> "LSTMMetaPredictor":
        """
        Load meta-models đã save (LSTM cls + Linear reg) kèm artifacts.

        Args:
            artifacts_dir: Thư mục chứa Model.pkl, Transform_pipeline.pkl, Feature_list.json
            meta_dir:      Thư mục chứa lstm_meta_cls.pkl, linear_meta_reg.pkl

        Returns:
            LSTMMetaPredictor sẵn sàng predict (không cần train CSV)
        """
        obj = cls()
        artifacts_dir = Path(artifacts_dir)
        meta_dir = Path(meta_dir)

        obj._stacking = obj._load_stacking(artifacts_dir)
        obj._pipeline = obj._load_pipeline(artifacts_dir)
        feature_data = obj._load_feature_list(artifacts_dir)
        obj._feature_columns = feature_data.get("all_feature_columns", [])
        obj._target_column = feature_data.get("target_column", "rain_total")

        obj._meta_cls = LSTMMetaClassifier.load(meta_dir / cls._META_CLS_FILE)
        obj._meta_reg = LinearInteractionMetaRegressor.load(meta_dir / cls._META_REG_FILE)

        info_path = meta_dir / cls._ARTIFACTS_INFO
        if info_path.exists():
            with open(info_path, encoding="utf-8") as f:
                data = json.load(f)
            obj._artifacts_info = LSTMMetaArtifacts(**data)
            obj._rain_threshold = data.get("rain_threshold", RAIN_THRESHOLD)
            obj._predict_threshold = data.get("predict_threshold", 0.4)

        obj._is_ready = True
        logger.info("LSTMMetaPredictor loaded from %s", meta_dir)
        return obj

    # ─────────────────────────────────────────────────────────────────────────
    # Save meta-models
    # ─────────────────────────────────────────────────────────────────────────

    def save_meta_models(self, meta_dir: Union[str, Path]) -> Path:
        """
        Lưu LSTM cls + Linear reg vào meta_dir.
        Không lưu base models (đã có trong artifacts_dir/Model.pkl).

        Args:
            meta_dir: Thư mục đích

        Returns:
            Path của meta_dir
        """
        self._check_ready()
        meta_dir = Path(meta_dir)
        meta_dir.mkdir(parents=True, exist_ok=True)

        self._meta_cls.save(meta_dir / self._META_CLS_FILE)
        self._meta_reg.save(meta_dir / self._META_REG_FILE)

        info = {}
        if self._artifacts_info is not None:
            info = {
                "artifacts_dir": self._artifacts_info.artifacts_dir,
                "train_csv_path": self._artifacts_info.train_csv_path,
                "n_features": self._artifacts_info.n_features,
                "n_cls_oof_samples": self._artifacts_info.n_cls_oof_samples,
                "n_reg_oof_samples": self._artifacts_info.n_reg_oof_samples,
                "meta_cls_backend": self._artifacts_info.meta_cls_backend,
                "meta_cls_seq_len": self._artifacts_info.meta_cls_seq_len,
                "fold5_roc_auc": self._artifacts_info.fold5_roc_auc,
                "fold5_f1": self._artifacts_info.fold5_f1,
                "fold5_pr_auc": self._artifacts_info.fold5_pr_auc,
                "fold5_reg_mae_mm": self._artifacts_info.fold5_reg_mae_mm,
                "fold5_reg_r2": self._artifacts_info.fold5_reg_r2,
                "trained_at": self._artifacts_info.trained_at,
                "rain_threshold": self._rain_threshold,
                "predict_threshold": self._predict_threshold,
            }
        with open(meta_dir / self._ARTIFACTS_INFO, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        logger.info("LSTMMetaPredictor meta-models saved → %s", meta_dir)
        return meta_dir

    # ─────────────────────────────────────────────────────────────────────────
    # Prediction
    # ─────────────────────────────────────────────────────────────────────────

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Dự báo lượng mưa (mm) sử dụng LSTM cls + Linear reg meta-models.

        Tương thích với WeatherPredictor.predict() và StackingEnsemble.predict().

        Args:
            X: Feature matrix (n, n_features) — đã preprocessed (numpy hoặc DataFrame)

        Returns:
            rain_mm: shape (n,), >= 0
        """
        full = self.predict_full(X)
        return full["predictions"]

    def predict_full(
        self, X: Union[np.ndarray, pd.DataFrame]
    ) -> Dict[str, np.ndarray]:
        """
        Dự báo đầy đủ, trả về dict:
            - "predictions":     lượng mưa cuối cùng (mm), qua classification gate
            - "p_rain":          xác suất có mưa (0–1)
            - "has_rain":        nhãn binary 0/1 theo predict_threshold
            - "rain_mm_ungated": lượng mưa từ regression không qua gate (debug)

        Returns:
            dict với 4 keys, mỗi key là ndarray shape (n,)
        """
        self._check_ready()
        X_np = self._to_numpy(X)

        # Bước 1: Xây Z_cls từ base classifiers
        n_cls = len(self._stacking.final_cls_models)
        Z_cls = np.zeros((len(X_np), n_cls), dtype=np.float64)
        for i, model in enumerate(self._stacking.final_cls_models):
            Z_cls[:, i] = model.predict_proba(X_np)[:, 1]

        # Bước 2: LSTM meta-classifier → p_rain
        p_rain = self._meta_cls.predict_proba(Z_cls)[:, 1]
        has_rain = p_rain > self._predict_threshold

        # Bước 3: Xây Z_reg từ base regressors → Linear meta-regressor
        n_reg = len(self._stacking.final_reg_models)
        Z_reg = np.zeros((len(X_np), n_reg), dtype=np.float64)
        for i, model in enumerate(self._stacking.final_reg_models):
            Z_reg[:, i] = model.predict(X_np)

        rain_mm_ungated = np.expm1(self._meta_reg.predict(Z_reg)).clip(min=0)

        # Bước 4: Gate
        predictions = np.where(has_rain, rain_mm_ungated, 0.0)

        return {
            "predictions": predictions,
            "p_rain": p_rain,
            "has_rain": has_rain.astype(np.int32),
            "rain_mm_ungated": rain_mm_ungated,
        }

    def predict_from_df(
        self, df: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """
        Predict từ DataFrame raw (chưa transform).
        Tự động apply Transform_pipeline trước khi predict.

        Args:
            df: DataFrame với đúng feature columns (chưa scaled)

        Returns:
            dict giống predict_full()
        """
        self._check_ready()
        X_np = self._transform_df(df)
        return self.predict_full(X_np)

    # ─────────────────────────────────────────────────────────────────────────
    # Compare với original meta-models
    # ─────────────────────────────────────────────────────────────────────────

    def compare_with_original(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y_true: Union[np.ndarray, pd.Series],
        dataset_name: str = "holdout",
    ) -> Dict[str, Dict]:
        """
        So sánh side-by-side: meta-models gốc (Ridge/LR) vs LSTM/Linear.

        Args:
            X:            Feature matrix (đã preprocessed)
            y_true:       Target lượng mưa (mm), shape (n,)
            dataset_name: Tên dataset để hiển thị

        Returns:
            dict với keys "original" và "lstm", mỗi key chứa:
                - classification: {roc_auc, f1, recall, pr_auc}
                - regression_rainy: {mae_mm, rmse_mm, r2}
        """
        self._check_ready()
        X_np = self._to_numpy(X)
        y_np = np.asarray(y_true, dtype=np.float64).reshape(-1)

        # Original meta-models (từ StackingEnsemble đã load)
        original_pred = self._stacking.evaluate(X_np, y_np, dataset_name=f"{dataset_name}_original")

        # LSTM meta-models
        full = self.predict_full(X_np)
        p_rain = full["p_rain"]
        y_pred = full["predictions"]
        y_pred_cls = (p_rain > self._predict_threshold).astype(int)
        y_cls_true = (y_np > self._rain_threshold).astype(int)

        rainy_mask = y_np > self._rain_threshold
        lstm_cls = {}
        if len(np.unique(y_cls_true)) > 1:
            lstm_cls["roc_auc"] = round(roc_auc_score(y_cls_true, p_rain), 4)
            lstm_cls["pr_auc"] = round(average_precision_score(y_cls_true, p_rain), 4)
        lstm_cls["f1"] = round(f1_score(y_cls_true, y_pred_cls, zero_division=0), 4)
        lstm_cls["recall"] = round(recall_score(y_cls_true, y_pred_cls, zero_division=0), 4)

        lstm_reg = {}
        if rainy_mask.sum() > 0:
            yr_true = y_np[rainy_mask]
            yr_pred = y_pred[rainy_mask]
            lstm_reg["mae_mm"] = round(mean_absolute_error(yr_true, yr_pred), 4)
            lstm_reg["rmse_mm"] = round(
                float(np.sqrt(mean_squared_error(yr_true, yr_pred))), 4
            )
            lstm_reg["r2"] = round(r2_score(yr_true, yr_pred), 4)

        print(f"\n{'=' * 60}")
        print(f"Compare [{dataset_name}]: Original vs LSTM/Linear")
        print(f"{'=' * 60}")
        print(f"{'Metric':<25} {'Original':>12} {'LSTM/Linear':>12}")
        print(f"{'-' * 50}")
        for key in ["roc_auc", "pr_auc", "f1", "recall"]:
            orig_val = original_pred.get("classification", {}).get(key, "n/a")
            lstm_val = lstm_cls.get(key, "n/a")
            if isinstance(orig_val, float) and isinstance(lstm_val, float):
                delta = lstm_val - orig_val
                delta_str = f"({delta:+.4f})"
            else:
                delta_str = ""
            print(f"  CLS {key:<20} {str(orig_val):>12} {str(lstm_val):>12}  {delta_str}")
        for key in ["mae_mm", "rmse_mm", "r2"]:
            orig_val = original_pred.get("regression_rainy", {}).get(key, "n/a")
            lstm_val = lstm_reg.get(key, "n/a")
            if isinstance(orig_val, float) and isinstance(lstm_val, float):
                delta = lstm_val - orig_val
                delta_str = f"({delta:+.4f})"
            else:
                delta_str = ""
            print(f"  REG {key:<20} {str(orig_val):>12} {str(lstm_val):>12}  {delta_str}")
        print(f"{'=' * 60}\n")

        return {
            "original": original_pred,
            "lstm": {"classification": lstm_cls, "regression_rainy": lstm_reg},
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _load_stacking(self, artifacts_dir: Path) -> StackingEnsemble:
        """Load StackingEnsemble từ Model.pkl."""
        model_path = artifacts_dir / self._MODEL_PKL
        if not model_path.exists():
            raise FileNotFoundError(f"Model.pkl không tìm thấy: {model_path}")
        payload = joblib.load(model_path)
        # Có thể là dict (joblib dump) hoặc StackingEnsemble trực tiếp
        if isinstance(payload, StackingEnsemble):
            return payload
        if isinstance(payload, dict):
            stacking = StackingEnsemble(
                n_splits=payload.get("n_splits", 5),
                predict_threshold=payload.get("predict_threshold", 0.4),
                rain_threshold=payload.get("rain_threshold", RAIN_THRESHOLD),
                seed=payload.get("seed", 42),
                verbose=False,
            )
            stacking.meta_cls = payload["meta_cls"]
            stacking.meta_reg = payload["meta_reg"]
            stacking.final_cls_models = payload.get("final_cls_models", [])
            stacking.final_reg_models = payload.get("final_reg_models", [])
            stacking.cls_model_names = payload.get("cls_model_names", [])
            stacking.reg_model_names = payload.get("reg_model_names", [])
            stacking.oof_cls_shape = payload.get("oof_cls_shape", (0, 0))
            stacking.oof_reg_shape = payload.get("oof_reg_shape", (0, 0))
            stacking.stage_metrics = payload.get("stage_metrics", {})
            stacking.is_trained = True
            return stacking
        raise ValueError(f"Model.pkl không đúng format: {type(payload)}")

    def _load_pipeline(self, artifacts_dir: Path) -> Any:
        """Load WeatherTransformPipeline từ Transform_pipeline.pkl."""
        pipeline_path = artifacts_dir / self._PIPELINE_PKL
        if not pipeline_path.exists():
            logger.warning("Transform_pipeline.pkl không tìm thấy. Sẽ không apply transform.")
            return None
        return joblib.load(pipeline_path)

    def _load_feature_list(self, artifacts_dir: Path) -> Dict:
        """Đọc Feature_list.json."""
        json_path = artifacts_dir / self._FEATURE_JSON
        if not json_path.exists():
            return {}
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)

    def _load_train_info(self, artifacts_dir: Path) -> Dict:
        """Đọc Train_info.json."""
        json_path = artifacts_dir / self._TRAIN_INFO_JSON
        if not json_path.exists():
            return {}
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)

    def _resolve_train_csv(
        self, train_info: Dict, artifacts_dir: Path
    ) -> Optional[Path]:
        """Lấy đường dẫn train CSV từ Train_info.json."""
        paths = train_info.get("split_saved_paths", {})
        train_path_str = (
            paths.get("train")
            or paths.get("train_path")
            or paths.get("merge_train")
        )
        if not train_path_str:
            return None

        # Thử resolve relative từ nhiều gốc
        candidates = [
            Path(train_path_str),                                      # absolute
            artifacts_dir / train_path_str,                            # relative to artifacts_dir
            artifacts_dir.parent.parent / train_path_str,              # relative to app root
            Path(__file__).parent.parent / train_path_str,            # relative to model dir
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def _load_and_transform(
        self,
        train_csv: Path,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load train CSV → align columns → apply pipeline → (X_np, y_np).

        Returns:
            X_np: shape (n, n_features), float64
            y_np: shape (n,), mm values
        """
        df = pd.read_csv(train_csv)

        # Lấy target trước khi transform
        target_col = self._target_column
        if target_col not in df.columns:
            # Thử một số tên phổ biến khác
            for alt in ["rain_total", "rainfall_mm", "precipitation", "rain"]:
                if alt in df.columns:
                    target_col = alt
                    break
            else:
                raise ValueError(
                    f"Không tìm thấy target column '{self._target_column}' trong CSV. "
                    f"Columns: {list(df.columns)[:20]}"
                )

        y_np = np.asarray(df[target_col], dtype=np.float64)

        # Apply transform pipeline nếu có
        if self._pipeline is not None:
            try:
                df_transformed = self._pipeline.transform(df)
            except Exception as e:
                logger.warning("Pipeline transform thất bại: %s. Dùng raw DataFrame.", e)
                df_transformed = df
        else:
            df_transformed = df

        # Chọn feature columns
        if self._feature_columns:
            available = [c for c in self._feature_columns if c in df_transformed.columns]
            if len(available) < len(self._feature_columns):
                missing = set(self._feature_columns) - set(df_transformed.columns)
                logger.warning(
                    "Thiếu %d feature columns: %s...",
                    len(missing), list(missing)[:5],
                )
            X_np = df_transformed[available].select_dtypes(include=[np.number]).values.astype(np.float64)
        else:
            X_np = df_transformed.select_dtypes(include=[np.number]).drop(
                columns=[target_col], errors="ignore"
            ).values.astype(np.float64)

        # Xử lý NaN / Inf còn sót
        X_np = np.nan_to_num(X_np, nan=0.0, posinf=1e6, neginf=-1e6)

        if verbose:
            print(f"  Dataset: {len(X_np)} rows, {X_np.shape[1]} features")
        return X_np, y_np

    def _rebuild_oof(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_splits: int = 5,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Rebuild OOF Z_cls và Z_reg từ base models đã train.

        Cách hoạt động:
        - final_cls_models được train trên TOÀN BỘ train data (Stage 9)
        - Để rebuild OOF KHÔNG bị leakage, ta dùng TimeSeriesSplit:
          trong mỗi fold, chỉ dùng model.predict_proba(X_val) — không retrain
        - ĐÂY LÀ XẤP XỈ: base models đã thấy X_val khi train (Stage 9)
          Do đó Z_cls sẽ optimistic một phần, nhưng pattern temporal vẫn hữu ích
          cho LSTM học
        - Nếu muốn OOF thực sự unbiased, cần train lại từ đầu với LSTMStackingEnsemble

        Returns:
            Z_cls:     (n, n_cls_models) — OOF classification predictions
            Z_reg:     (n_rainy, n_reg_models) — OOF regression predictions
            y_cls:     (n,) binary — has rain
            y_reg_log: (n_rainy,) — log1p(rain_mm) for rainy-only samples
        """
        n = len(X_train)
        n_cls_models = len(self._stacking.final_cls_models)
        n_reg_models = len(self._stacking.final_reg_models)

        y_cls = (y_train > self._rain_threshold).astype(np.int32)
        rainy_mask = y_train > self._rain_threshold
        X_rainy = X_train[rainy_mask]
        y_reg_log = np.log1p(y_train[rainy_mask])

        # Classification OOF
        Z_cls = np.zeros((n, n_cls_models), dtype=np.float64)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for _, val_idx in tscv.split(X_train):
            X_val = X_train[val_idx]
            for m_idx, model in enumerate(self._stacking.final_cls_models):
                Z_cls[val_idx, m_idx] = model.predict_proba(X_val)[:, 1]

        # Điền rows không thuộc val fold nào (đầu array) bằng mean
        zero_rows = np.all(Z_cls == 0, axis=1)
        if zero_rows.sum() > 0:
            col_means = Z_cls[~zero_rows].mean(axis=0)
            Z_cls[zero_rows] = col_means

        # Regression OOF (rainy-only)
        n_rainy = len(X_rainy)
        Z_reg = np.zeros((n_rainy, n_reg_models), dtype=np.float64)
        for _, val_idx_rainy in tscv.split(X_rainy):
            X_val_rainy = X_rainy[val_idx_rainy]
            for m_idx, model in enumerate(self._stacking.final_reg_models):
                Z_reg[val_idx_rainy, m_idx] = model.predict(X_val_rainy)

        zero_rows_reg = np.all(Z_reg == 0, axis=1)
        if zero_rows_reg.sum() > 0:
            col_medians = np.median(Z_reg[~zero_rows_reg], axis=0) if (~zero_rows_reg).sum() > 0 else np.zeros(n_reg_models)
            Z_reg[zero_rows_reg] = col_medians

        if verbose:
            print(f"  Z_cls: {Z_cls.shape}, Z_reg: {Z_reg.shape}")

        return Z_cls, Z_reg, y_cls, y_reg_log

    def _transform_df(self, df: pd.DataFrame) -> np.ndarray:
        """Apply transform pipeline lên DataFrame raw → numpy."""
        if self._pipeline is not None:
            try:
                df = self._pipeline.transform(df)
            except Exception as e:
                logger.warning("Pipeline transform thất bại: %s", e)

        if self._feature_columns:
            available = [c for c in self._feature_columns if c in df.columns]
            X = df[available].select_dtypes(include=[np.number]).values.astype(np.float64)
        else:
            X = df.select_dtypes(include=[np.number]).values.astype(np.float64)

        return np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    @staticmethod
    def _to_numpy(X: Any) -> np.ndarray:
        """Chuyển DataFrame/array về numpy float64."""
        if hasattr(X, "select_dtypes"):
            X = X.select_dtypes(include=[np.number]).values
        arr = np.asarray(X)
        if arr.dtype != np.float64:
            arr = arr.astype(np.float64)
        return arr

    def _check_ready(self) -> None:
        if not self._is_ready:
            raise RuntimeError(
                "LSTMMetaPredictor chưa sẵn sàng. "
                "Gọi from_artifacts() hoặc load_meta_models() trước."
            )

    def summary(self) -> str:
        """In tóm tắt trạng thái LSTMMetaPredictor."""
        info = self._artifacts_info
        lines = [
            "=" * 70,
            "  LSTMMetaPredictor Summary",
            "=" * 70,
            f"  ready              : {self._is_ready}",
            f"  rain_threshold     : {self._rain_threshold} mm",
            f"  predict_threshold  : {self._predict_threshold}",
        ]
        if info is not None:
            lines += [
                f"  artifacts_dir      : {info.artifacts_dir}",
                f"  n_features         : {info.n_features}",
                f"  n_cls_oof          : {info.n_cls_oof_samples}",
                f"  n_reg_oof          : {info.n_reg_oof_samples}",
                f"  meta_cls_backend   : {info.meta_cls_backend}",
                f"  meta_cls_seq_len   : {info.meta_cls_seq_len}",
                f"  fold5_roc_auc      : {info.fold5_roc_auc}",
                f"  fold5_f1           : {info.fold5_f1}",
                f"  fold5_reg_r2       : {info.fold5_reg_r2}",
                f"  trained_at         : {info.trained_at}",
            ]
        if self._meta_reg is not None and self._meta_reg._is_fitted:
            lines.append(f"  linear_coef_norm   : {np.linalg.norm(self._meta_reg.coef_):.4f}")
        lines.append("=" * 70)
        return "\n".join(lines)

    def __repr__(self) -> str:
        status = "ready" if self._is_ready else "not_ready"
        backend = getattr(self._meta_cls, "_backend", "?") if self._meta_cls else "none"
        return f"LSTMMetaPredictor({status}, backend={backend!r})"
