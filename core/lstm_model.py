import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from core.indicators import add_indicators

FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "rsi", "macd", "macd_signal", "macd_hist",
    "bb_pct", "volume_ratio", "returns", "returns_5", "ema_cross",
]


class LSTMPredictor:
    def __init__(self, seq_len: int = 60):
        self.seq_len = seq_len
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False
        self.val_accuracy = 0.0

    # ------------------------------------------------------------------ build

    def _build_model(self, input_shape):
        # Lazy import so startup is fast if TF not needed immediately
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
        from tensorflow.keras.optimizers import Adam

        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            BatchNormalization(),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # ------------------------------------------------------------------ data

    def _prepare(self, df: pd.DataFrame):
        df = add_indicators(df)
        df = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).dropna()
        return df

    def _make_sequences(self, X: np.ndarray, y: np.ndarray):
        Xs, ys = [], []
        for i in range(self.seq_len, len(X)):
            Xs.append(X[i - self.seq_len: i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    # ------------------------------------------------------------------ train

    def train(self, df: pd.DataFrame, epochs: int = 50, progress_cb=None):
        from tensorflow.keras.callbacks import EarlyStopping, Callback

        features = self._prepare(df)

        # Binary target: 1 if NEXT close > CURRENT close
        raw_close = features["close"].values
        y_full = (raw_close[1:] > raw_close[:-1]).astype(np.float32)
        features = features.iloc[:-1]

        X_scaled = self.scaler.fit_transform(features.values)
        X_seq, y_seq = self._make_sequences(X_scaled, y_full)

        split = int(0.8 * len(X_seq))
        X_tr, X_val = X_seq[:split], X_seq[split:]
        y_tr, y_val = y_seq[:split], y_seq[split:]

        self.model = self._build_model((self.seq_len, X_seq.shape[2]))

        class ProgressCallback(Callback):
            def __init__(self, total, cb):
                super().__init__()
                self.total = total
                self.cb = cb

            def on_epoch_end(self, epoch, logs=None):
                if self.cb:
                    pct = int((epoch + 1) / self.total * 100)
                    acc = logs.get("val_accuracy", 0)
                    self.cb(pct, f"Epoch {epoch+1}/{self.total}  val_acc={acc:.3f}")

        callbacks = [EarlyStopping(patience=7, restore_best_weights=True, monitor="val_accuracy")]
        if progress_cb:
            callbacks.append(ProgressCallback(epochs, progress_cb))

        history = self.model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=0,
        )

        self.is_trained = True
        val_accs = history.history.get("val_accuracy", [0])
        self.val_accuracy = float(max(val_accs))
        return self.val_accuracy, len(history.history["loss"])

    # ----------------------------------------------------------------- predict

    def predict(self, df: pd.DataFrame) -> tuple[str, float]:
        """Return (direction, probability) for the NEXT candle."""
        if not self.is_trained or self.model is None:
            return "WAIT", 0.5

        features = self._prepare(df)
        if len(features) < self.seq_len:
            return "WAIT", 0.5

        window = features.values[-self.seq_len:]
        X_scaled = self.scaler.transform(window)
        X_seq = X_scaled.reshape(1, self.seq_len, -1)

        prob = float(self.model.predict(X_seq, verbose=0)[0][0])
        direction = "LONG" if prob >= 0.5 else "SHORT"
        return direction, prob
