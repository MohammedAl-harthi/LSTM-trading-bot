import sys
from datetime import datetime

import pandas as pd
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPalette
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QLabel, QPushButton, QProgressBar,
    QSplitter, QStatusBar, QFrame, QLineEdit, QSpinBox,
    QDoubleSpinBox, QGroupBox, QMessageBox, QSizePolicy,
)

from core.data_fetcher import DataFetcher
from core.indicators import add_indicators
from core.lstm_model import LSTMPredictor
from core.paper_trader import PaperTrader
from ui.chart_widget import ChartWidget
from ui.trading_panel import TradingPanel


# ------------------------------------------------------------------- workers

class SymbolsWorker(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def run(self):
        try:
            fetcher = DataFetcher()
            symbols = fetcher.get_futures_symbols()
            self.finished.emit(symbols)
        except Exception as exc:
            self.error.emit(str(exc))


class FetchWorker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, symbol: str, timeframe: str, limit: int):
        super().__init__()
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit

    def run(self):
        try:
            fetcher = DataFetcher()
            df = fetcher.fetch_ohlcv(self.symbol, self.timeframe, self.limit)
            df = add_indicators(df)
            self.finished.emit(df)
        except Exception as exc:
            self.error.emit(str(exc))


class TrainWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(float, int)
    error = pyqtSignal(str)

    def __init__(self, model: LSTMPredictor, df: pd.DataFrame, epochs: int):
        super().__init__()
        self.model = model
        self.df = df
        self.epochs = epochs

    def run(self):
        try:
            def cb(pct, msg):
                self.progress.emit(pct, msg)

            val_acc, n_epochs = self.model.train(self.df, self.epochs, progress_cb=cb)
            self.finished.emit(val_acc, n_epochs)
        except Exception as exc:
            self.error.emit(str(exc))


class LiveTickerWorker(QThread):
    tick = pyqtSignal(float)
    error = pyqtSignal(str)

    def __init__(self, symbol: str):
        super().__init__()
        self.symbol = symbol

    def run(self):
        try:
            fetcher = DataFetcher()
            ticker = fetcher.fetch_ticker(self.symbol)
            price = float(ticker.get("last", 0))
            self.tick.emit(price)
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------- main window

DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #0f0f1e;
    color: #e0e0e0;
}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 4px;
    padding: 4px 8px;
    color: #e0e0e0;
    font-size: 11px;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    color: #e0e0e0;
    selection-background-color: #16213e;
}
QPushButton {
    background: #1565c0;
    color: #ffffff;
    border: none;
    border-radius: 5px;
    padding: 6px 16px;
    font-size: 11px;
    font-weight: bold;
}
QPushButton:hover { background: #1976d2; }
QPushButton:pressed { background: #0d47a1; }
QPushButton:disabled { background: #2a2a4a; color: #666; }
QPushButton#btn_start { background: #2e7d32; }
QPushButton#btn_start:hover { background: #388e3c; }
QPushButton#btn_stop { background: #b71c1c; }
QPushButton#btn_stop:hover { background: #c62828; }
QPushButton#btn_reset { background: #4a148c; }
QPushButton#btn_reset:hover { background: #6a1b9a; }
QProgressBar {
    border: 1px solid #2a2a4a;
    border-radius: 4px;
    background: #1a1a2e;
    text-align: center;
    color: #e0e0e0;
    height: 16px;
}
QProgressBar::chunk { background: #1565c0; border-radius: 3px; }
QGroupBox {
    color: #9e9e9e;
    border: 1px solid #2a2a4a;
    border-radius: 6px;
    margin-top: 10px;
    font-size: 11px;
}
QGroupBox::title { subcontrol-origin: margin; left: 8px; color: #9e9e9e; }
QLabel { color: #e0e0e0; }
QStatusBar { background: #0a0a14; color: #9e9e9e; font-size: 10px; }
QSplitter::handle { background: #2a2a4a; }
"""


class PredictionCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("QFrame { background: #16213e; border: 1px solid #2a2a4a; border-radius:6px; }")
        self.setFixedWidth(200)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        title = QLabel("LSTM Prediction")
        title.setStyleSheet("color: #9e9e9e; font-size: 10px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.lbl_direction = QLabel("—")
        self.lbl_direction.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        self.lbl_direction.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.lbl_confidence = QLabel("Confidence: —")
        self.lbl_confidence.setStyleSheet("color: #9e9e9e; font-size: 11px;")
        self.lbl_confidence.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.conf_bar = QProgressBar()
        self.conf_bar.setRange(0, 100)
        self.conf_bar.setValue(0)
        self.conf_bar.setTextVisible(False)
        self.conf_bar.setFixedHeight(6)

        self.lbl_status = QLabel("Not trained")
        self.lbl_status.setStyleSheet("color: #616161; font-size: 10px;")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(title)
        layout.addWidget(self.lbl_direction)
        layout.addWidget(self.lbl_confidence)
        layout.addWidget(self.conf_bar)
        layout.addStretch()
        layout.addWidget(self.lbl_status)

    def update(self, direction: str, prob: float, status: str = ""):
        if direction == "LONG":
            color = "#26a69a"
            conf = prob
        elif direction == "SHORT":
            color = "#ef5350"
            conf = 1 - prob
        else:
            color = "#9e9e9e"
            conf = 0.5

        self.lbl_direction.setText(direction)
        self.lbl_direction.setStyleSheet(f"color: {color}; font-size: 24px; font-weight: bold;")
        pct = int(conf * 100)
        self.lbl_confidence.setText(f"Confidence: {pct}%")
        self.conf_bar.setValue(pct)
        if status:
            self.lbl_status.setText(status)


class MainWindow(QMainWindow):
    TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"]
    DEFAULT_TF = "1h"
    DEFAULT_LIMIT = 2000
    UPDATE_INTERVAL_MS = 60_000  # 1 min default; resized when TF changes

    def __init__(self):
        super().__init__()
        self.setWindowTitle("LSTM Binance Futures Bot")
        self.resize(1400, 900)
        self.setStyleSheet(DARK_STYLE)

        self._df: pd.DataFrame | None = None
        self._model = LSTMPredictor(seq_len=60)
        self._trader = PaperTrader()
        self._signals: list[tuple] = []
        self._current_price: float | None = None
        self._bot_running = False

        self._workers: list[QThread] = []

        self._setup_ui()
        self._setup_timer()
        self._load_symbols()

    # ------------------------------------------------------------------- UI

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        root.addLayout(self._build_toolbar())
        root.addWidget(self._build_train_bar())

        # Main splitter: chart + bottom panel
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)

        self._chart = ChartWidget()
        splitter.addWidget(self._chart)

        # Bottom: prediction card + trading panel
        bottom = QWidget()
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(8)

        self._pred_card = PredictionCard()
        self._trading_panel = TradingPanel()

        bottom_layout.addWidget(self._pred_card, stretch=0)
        bottom_layout.addWidget(self._trading_panel, stretch=1)
        splitter.addWidget(bottom)
        splitter.setSizes([560, 240])

        root.addWidget(splitter, stretch=1)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_label = QLabel("Ready")
        self._price_label = QLabel("")
        self._price_label.setStyleSheet("color: #ffffff; font-weight: bold;")
        self._status_bar.addWidget(self._status_label)
        self._status_bar.addPermanentWidget(self._price_label)

    def _build_toolbar(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.setSpacing(8)

        # Symbol search + combo
        self._symbol_search = QLineEdit()
        self._symbol_search.setPlaceholderText("Filter symbols…")
        self._symbol_search.setFixedWidth(130)
        self._symbol_search.textChanged.connect(self._filter_symbols)

        self._symbol_combo = QComboBox()
        self._symbol_combo.setFixedWidth(180)
        self._symbol_combo.setEditable(False)

        # Timeframe
        self._tf_combo = QComboBox()
        self._tf_combo.addItems(self.TIMEFRAMES)
        self._tf_combo.setCurrentText(self.DEFAULT_TF)
        self._tf_combo.setFixedWidth(70)
        self._tf_combo.currentTextChanged.connect(self._on_tf_changed)

        # Candle limit
        self._limit_spin = QSpinBox()
        self._limit_spin.setRange(200, 5000000)
        self._limit_spin.setValue(self.DEFAULT_LIMIT)
        self._limit_spin.setSuffix(" bars")
        self._limit_spin.setSingleStep(500)
        self._limit_spin.setFixedWidth(110)

        # Epochs
        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(10, 200)
        self._epochs_spin.setValue(50)
        self._epochs_spin.setSuffix(" epochs")
        self._epochs_spin.setFixedWidth(110)

        # Buttons
        self.btn_fetch = QPushButton("⬇ Fetch")
        self.btn_fetch.setToolTip("Download historical OHLCV data")
        self.btn_fetch.clicked.connect(self._fetch_data)

        self.btn_train = QPushButton("⚡ Train LSTM")
        self.btn_train.setToolTip("Train LSTM model on fetched data")
        self.btn_train.clicked.connect(self._train_model)
        self.btn_train.setEnabled(False)

        self.btn_start = QPushButton("▶ Start Bot")
        self.btn_start.setObjectName("btn_start")
        self.btn_start.clicked.connect(self._start_bot)
        self.btn_start.setEnabled(False)

        self.btn_stop = QPushButton("⏹ Stop")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.clicked.connect(self._stop_bot)
        self.btn_stop.setEnabled(False)

        self.btn_reset = QPushButton("↺ Reset")
        self.btn_reset.setObjectName("btn_reset")
        self.btn_reset.clicked.connect(self._reset_trader)

        layout.addWidget(QLabel("Symbol:"))
        layout.addWidget(self._symbol_search)
        layout.addWidget(self._symbol_combo)
        layout.addWidget(QLabel("TF:"))
        layout.addWidget(self._tf_combo)
        layout.addWidget(self._limit_spin)
        layout.addWidget(self.btn_fetch)
        layout.addWidget(self._epochs_spin)
        layout.addWidget(self.btn_train)
        layout.addSpacing(16)
        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_stop)
        layout.addWidget(self.btn_reset)
        layout.addStretch()

        return layout

    def _build_train_bar(self) -> QWidget:
        container = QWidget()
        container.setFixedHeight(28)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)

        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet("color: #9e9e9e; font-size: 10px;")

        layout.addWidget(self._progress_bar, stretch=1)
        layout.addWidget(self._progress_label, stretch=0)
        return container

    # ---------------------------------------------------------------- symbols

    def _load_symbols(self):
        self._set_status("Loading futures symbols…")
        worker = SymbolsWorker()
        worker.finished.connect(self._on_symbols_loaded)
        worker.error.connect(lambda e: self._set_status(f"Error loading symbols: {e}"))
        self._workers.append(worker)
        worker.start()

    def _on_symbols_loaded(self, symbols: list[str]):
        self._all_symbols = symbols
        self._symbol_combo.clear()
        self._symbol_combo.addItems(symbols)
        # Default to BTC/USDT:USDT
        idx = next((i for i, s in enumerate(symbols) if "BTC/USDT:USDT" in s), 0)
        self._symbol_combo.setCurrentIndex(idx)
        self._set_status(f"Loaded {len(symbols)} futures symbols.")

    def _filter_symbols(self, text: str):
        if not hasattr(self, "_all_symbols"):
            return
        filtered = [s for s in self._all_symbols if text.upper() in s.upper()]
        current = self._symbol_combo.currentText()
        self._symbol_combo.blockSignals(True)
        self._symbol_combo.clear()
        self._symbol_combo.addItems(filtered)
        if current in filtered:
            self._symbol_combo.setCurrentText(current)
        self._symbol_combo.blockSignals(False)

    # ------------------------------------------------------------------ fetch

    def _fetch_data(self):
        symbol = self._symbol_combo.currentText()
        if not symbol:
            return
        tf = self._tf_combo.currentText()
        limit = self._limit_spin.value()
        self._set_status(f"Fetching {limit} × {tf} candles for {symbol}…")
        self.btn_fetch.setEnabled(False)
        self.btn_train.setEnabled(False)

        worker = FetchWorker(symbol, tf, limit)
        worker.finished.connect(self._on_data_fetched)
        worker.error.connect(self._on_fetch_error)
        self._workers.append(worker)
        worker.start()

    def _on_data_fetched(self, df: pd.DataFrame):
        self._df = df
        self._chart.update_chart(df, signals=self._signals)
        self._set_status(
            f"Fetched {len(df)} candles | "
            f"Last close: {df['close'].iloc[-1]:.4f} | "
            f"{df.index[-1].strftime('%Y-%m-%d %H:%M')}"
        )
        self.btn_fetch.setEnabled(True)
        self.btn_train.setEnabled(True)

    def _on_fetch_error(self, msg: str):
        self._set_status(f"Fetch error: {msg}")
        self.btn_fetch.setEnabled(True)
        QMessageBox.warning(self, "Fetch Error", msg)

    # ------------------------------------------------------------------ train

    def _train_model(self):
        if self._df is None:
            QMessageBox.information(self, "No Data", "Fetch data first.")
            return

        self._model = LSTMPredictor(seq_len=60)
        epochs = self._epochs_spin.value()
        self._set_status("Training LSTM…")
        self.btn_train.setEnabled(False)
        self.btn_fetch.setEnabled(False)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)

        worker = TrainWorker(self._model, self._df, epochs)
        worker.progress.connect(self._on_train_progress)
        worker.finished.connect(self._on_train_finished)
        worker.error.connect(self._on_train_error)
        self._workers.append(worker)
        worker.start()

    def _on_train_progress(self, pct: int, msg: str):
        self._progress_bar.setValue(pct)
        self._progress_label.setText(msg)

    def _on_train_finished(self, val_acc: float, n_epochs: int):
        self._progress_bar.setVisible(False)
        self._progress_label.setText("")
        self.btn_train.setEnabled(True)
        self.btn_fetch.setEnabled(True)
        self.btn_start.setEnabled(True)

        self._set_status(
            f"Training complete — {n_epochs} epochs | "
            f"Val accuracy: {val_acc * 100:.1f}%"
        )
        self._pred_card.update("—", 0.5, f"Trained  acc={val_acc*100:.1f}%")

        # Run first prediction immediately
        self._run_prediction()

    def _on_train_error(self, msg: str):
        self._progress_bar.setVisible(False)
        self.btn_train.setEnabled(True)
        self.btn_fetch.setEnabled(True)
        self._set_status(f"Training error: {msg}")
        QMessageBox.critical(self, "Training Error", msg)

    # --------------------------------------------------------------- live bot

    def _setup_timer(self):
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer_tick)

    def _on_tf_changed(self, tf: str):
        mapping = {
            "1m": 60_000, "3m": 180_000, "5m": 300_000,
            "15m": 900_000, "30m": 1_800_000,
            "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
        }
        self.UPDATE_INTERVAL_MS = mapping.get(tf, 60_000)
        if self._bot_running:
            self._timer.setInterval(self.UPDATE_INTERVAL_MS)

    def _start_bot(self):
        if not self._model.is_trained:
            QMessageBox.information(self, "Not Trained", "Train the LSTM model first.")
            return
        self._bot_running = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._set_status("Bot running…")
        self._timer.start(self.UPDATE_INTERVAL_MS)
        self._on_timer_tick()

    def _stop_bot(self):
        self._bot_running = False
        self._timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._set_status("Bot stopped.")

    def _on_timer_tick(self):
        symbol = self._symbol_combo.currentText()
        tf = self._tf_combo.currentText()

        # Fetch latest candles
        self._set_status("Refreshing data…")
        worker = FetchWorker(symbol, tf, self._limit_spin.value())
        worker.finished.connect(self._on_live_data)
        worker.error.connect(lambda e: self._set_status(f"Live fetch error: {e}"))
        self._workers.append(worker)
        worker.start()

    def _on_live_data(self, df: pd.DataFrame):
        self._df = df
        self._current_price = float(df["close"].iloc[-1])
        self._price_label.setText(
            f"{self._symbol_combo.currentText()} | ${self._current_price:,.4f}"
        )
        self._run_prediction()
        self._chart.update_chart(df, signals=self._signals, current_price=self._current_price)
        self._trading_panel.update_account(self._trader, self._current_price)

    def _run_prediction(self):
        if self._df is None or not self._model.is_trained:
            return

        direction, prob = self._model.predict(self._df)
        self._pred_card.update(direction, prob, f"acc={self._model.val_accuracy*100:.1f}%")

        if self._bot_running and self._current_price:
            ts = datetime.now()
            msg = self._trader.process_signal(
                direction, prob, self._current_price, ts
            )
            if msg:
                for line in msg.split("\n"):
                    self._trading_panel.append_log(line)
                # Record signal on chart
                if direction in ("LONG", "SHORT") and self._df is not None:
                    last_ts = self._df.index[-1]
                    self._signals.append((last_ts, self._current_price, direction))
                    self._chart.update_chart(self._df, signals=self._signals,
                                             current_price=self._current_price)

            self._trading_panel.update_account(self._trader, self._current_price)

        self._set_status(
            f"Prediction: {direction}  |  prob={prob:.3f}  |  "
            f"{datetime.now().strftime('%H:%M:%S')}"
        )

    # ------------------------------------------------------------------ reset

    def _reset_trader(self):
        reply = QMessageBox.question(
            self, "Reset Paper Trader",
            "Reset balance to $10,000 and clear all trades?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._trader.reset()
            self._signals.clear()
            self._trading_panel.clear_log()
            self._trading_panel.update_account(self._trader)
            if self._df is not None:
                self._chart.update_chart(self._df, signals=[])
            self._set_status("Paper trader reset.")

    # ----------------------------------------------------------------- helpers

    def _set_status(self, msg: str):
        self._status_label.setText(msg)

    def closeEvent(self, event):
        self._timer.stop()
        for w in self._workers:
            if w.isRunning():
                w.quit()
                w.wait(2000)
        event.accept()
