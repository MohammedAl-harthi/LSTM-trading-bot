import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QPicture, QPainter, QColor, QPen, QBrush
from PyQt6.QtWidgets import QWidget, QVBoxLayout


# ---------------------------------------------------------------- candlestick

class CandlestickItem(pg.GraphicsObject):
    """Custom pyqtgraph item that draws OHLC candlesticks."""

    def __init__(self):
        super().__init__()
        self._picture = QPicture()
        self._data = []

    def set_data(self, data: list[tuple]):
        """data: list of (x_index, open, high, low, close)"""
        self._data = data
        self._generate()
        self.update()
        self.prepareGeometryChange()
        self.informViewBoundsChanged()

    def _generate(self):
        self._picture = QPicture()
        if not self._data:
            return
        p = QPainter(self._picture)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        w = 0.4  # half-width of body
        for (x, o, h, l, c) in self._data:
            # Wick
            wick_pen = QPen(QColor("#888888"), 1)
            p.setPen(wick_pen)
            p.drawLine(QPointF(x, l), QPointF(x, h))

            # Body
            if c >= o:
                body_color = QColor("#26a69a")   # bullish green
            else:
                body_color = QColor("#ef5350")   # bearish red

            p.setPen(QPen(body_color, 1))
            p.setBrush(QBrush(body_color))
            top = max(o, c)
            bot = min(o, c)
            height = top - bot if top != bot else 0.0001
            p.drawRect(QRectF(x - w, bot, w * 2, height))
        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self._picture)

    def boundingRect(self) -> QRectF:
        return QRectF(self._picture.boundingRect())


# ---------------------------------------------------------------- chart widget

class ChartWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._signals: list[tuple] = []  # (x, price, direction)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        pg.setConfigOption("background", "#1a1a2e")
        pg.setConfigOption("foreground", "#e0e0e0")

        # Price plot
        self.price_plot = pg.PlotWidget()
        self.price_plot.setMenuEnabled(False)
        self.price_plot.showGrid(x=True, y=True, alpha=0.15)
        self.price_plot.getAxis("left").setTextPen(pg.mkPen("#e0e0e0"))
        self.price_plot.getAxis("bottom").setStyle(showValues=False)

        # Volume plot
        self.vol_plot = pg.PlotWidget()
        self.vol_plot.setMenuEnabled(False)
        self.vol_plot.showGrid(x=True, y=False, alpha=0.10)
        self.vol_plot.getAxis("left").setTextPen(pg.mkPen("#9e9e9e"))
        self.vol_plot.setFixedHeight(90)
        self.vol_plot.setXLink(self.price_plot)

        layout.addWidget(self.price_plot, stretch=1)
        layout.addWidget(self.vol_plot, stretch=0)

        # Persistent items
        self._candle_item = CandlestickItem()
        self.price_plot.addItem(self._candle_item)

        self._ema20_curve = self.price_plot.plot(pen=pg.mkPen("#42a5f5", width=1.5), name="EMA20")
        self._ema50_curve = self.price_plot.plot(pen=pg.mkPen("#ff7043", width=1.5), name="EMA50")
        self._bb_upper_curve = self.price_plot.plot(pen=pg.mkPen("#7e57c2", width=1, style=Qt.PenStyle.DashLine))
        self._bb_lower_curve = self.price_plot.plot(pen=pg.mkPen("#7e57c2", width=1, style=Qt.PenStyle.DashLine))

        self._vol_bars = pg.BarGraphItem(x=[], height=[], width=0.7, brush="#37474f")
        self.vol_plot.addItem(self._vol_bars)

        # Signal scatter plots
        self._long_signals = pg.ScatterPlotItem(
            symbol="t1", size=14, brush=pg.mkBrush("#00e676"), pen=pg.mkPen(None))
        self._short_signals = pg.ScatterPlotItem(
            symbol="t", size=14, brush=pg.mkBrush("#ff1744"), pen=pg.mkPen(None))
        self.price_plot.addItem(self._long_signals)
        self.price_plot.addItem(self._short_signals)

        # Prediction line
        self._pred_line = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen("#ffeb3b", width=2, style=Qt.PenStyle.DashDotLine),
        )
        self.price_plot.addItem(self._pred_line)
        self._pred_line.setVisible(False)

        # Current price line
        self._price_line = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen("#ffffff", width=1, style=Qt.PenStyle.DotLine),
        )
        self.price_plot.addItem(self._price_line)

        # Legend
        legend = pg.LegendItem(offset=(10, 10))
        legend.setParentItem(self.price_plot.graphicsItem())
        legend.addItem(self._ema20_curve, "EMA20")
        legend.addItem(self._ema50_curve, "EMA50")

    # ----------------------------------------------------------------- update

    def update_chart(self, df: pd.DataFrame, signals: list[tuple] | None = None,
                     current_price: float | None = None):
        if df is None or df.empty:
            return

        xs = np.arange(len(df))

        # Candlesticks
        candle_data = list(zip(
            xs,
            df["open"].values,
            df["high"].values,
            df["low"].values,
            df["close"].values,
        ))
        self._candle_item.set_data(candle_data)

        # Overlay lines
        if "ema20" in df.columns:
            self._ema20_curve.setData(xs, df["ema20"].values)
        if "ema50" in df.columns:
            self._ema50_curve.setData(xs, df["ema50"].values)
        if "bb_upper" in df.columns:
            self._bb_upper_curve.setData(xs, df["bb_upper"].values)
            self._bb_lower_curve.setData(xs, df["bb_lower"].values)

        # Volume
        vol = df["volume"].values
        max_vol = vol.max() if vol.max() > 0 else 1
        colors = [
            "#26a69a" if c >= o else "#ef5350"
            for o, c in zip(df["open"].values, df["close"].values)
        ]
        brushes = [pg.mkBrush(c) for c in colors]
        self._vol_bars.setOpts(x=xs, height=vol, width=0.7, brushes=brushes)
        self.vol_plot.setYRange(0, max_vol * 1.2)

        # Current price line
        if current_price is not None:
            self._price_line.setValue(current_price)
            self._price_line.setVisible(True)

        # Trade signals
        if signals is not None:
            self._signals = signals
            self._draw_signals(df)

        # Prediction marker at right edge
        self._pred_line.setValue(len(df) - 0.5)
        self._pred_line.setVisible(True)

        self.price_plot.autoRange()

    def _draw_signals(self, df: pd.DataFrame):
        long_xs, long_ys = [], []
        short_xs, short_ys = [], []
        for (ts, price, direction) in self._signals:
            if ts in df.index:
                idx = df.index.get_loc(ts)
            else:
                continue
            low = df["low"].iloc[idx]
            high = df["high"].iloc[idx]
            if direction == "LONG":
                long_xs.append(idx)
                long_ys.append(low * 0.9995)
            else:
                short_xs.append(idx)
                short_ys.append(high * 1.0005)

        self._long_signals.setData(x=long_xs, y=long_ys)
        self._short_signals.setData(x=short_xs, y=short_ys)

    def clear(self):
        self._candle_item.set_data([])
        self._ema20_curve.setData([], [])
        self._ema50_curve.setData([], [])
        self._bb_upper_curve.setData([], [])
        self._bb_lower_curve.setData([], [])
        self._vol_bars.setOpts(x=[], height=[])
        self._long_signals.setData([], [])
        self._short_signals.setData([], [])
        self._pred_line.setVisible(False)
