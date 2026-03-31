from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox, QTextEdit,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont

from core.paper_trader import PaperTrader

_MONO = QFont("Menlo", 10)
_MONO_SM = QFont("Menlo", 9)


def _label(text: str, color: str = "#e0e0e0", bold: bool = False) -> QLabel:
    lbl = QLabel(text)
    font = QFont("Segoe UI", 10)
    font.setBold(bold)
    lbl.setFont(font)
    lbl.setStyleSheet(f"color: {color};")
    return lbl


def _value_label(text: str = "—", color: str = "#ffffff") -> QLabel:
    lbl = QLabel(text)
    lbl.setFont(_MONO)
    lbl.setStyleSheet(f"color: {color};")
    lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return lbl


class StatCard(QFrame):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background: #16213e;
                border: 1px solid #2a2a4a;
                border-radius: 6px;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(2)

        self._title = _label(title, color="#9e9e9e")
        self._value = _value_label()
        layout.addWidget(self._title)
        layout.addWidget(self._value)

    def set_value(self, text: str, color: str = "#ffffff"):
        self._value.setText(text)
        self._value.setStyleSheet(f"color: {color};")


class TradingPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # ---- left: account + position ----
        left = QVBoxLayout()
        left.setSpacing(6)

        account_box = QGroupBox("Account")
        account_box.setStyleSheet("QGroupBox { color: #9e9e9e; border: 1px solid #2a2a4a; border-radius:6px; margin-top:10px; } QGroupBox::title { subcontrol-origin: margin; left:8px; }")
        acct_grid = QVBoxLayout(account_box)
        acct_grid.setSpacing(4)

        row1 = QHBoxLayout()
        row1.addWidget(_label("Balance"))
        self.lbl_balance = _value_label("$10,000.00", "#ffffff")
        row1.addWidget(self.lbl_balance)

        row2 = QHBoxLayout()
        row2.addWidget(_label("Equity"))
        self.lbl_equity = _value_label("$10,000.00", "#ffffff")
        row2.addWidget(self.lbl_equity)

        row3 = QHBoxLayout()
        row3.addWidget(_label("Total PnL"))
        self.lbl_total_pnl = _value_label("$0.00", "#9e9e9e")
        row3.addWidget(self.lbl_total_pnl)

        acct_grid.addLayout(row1)
        acct_grid.addLayout(row2)
        acct_grid.addLayout(row3)
        left.addWidget(account_box)

        pos_box = QGroupBox("Open Position")
        pos_box.setStyleSheet(account_box.styleSheet())
        pos_layout = QVBoxLayout(pos_box)
        pos_layout.setSpacing(4)

        self.lbl_pos_side = _value_label("None", "#9e9e9e")
        self.lbl_pos_entry = _value_label("—")
        self.lbl_pos_sl = _value_label("—", "#ef5350")
        self.lbl_pos_tp = _value_label("—", "#26a69a")
        self.lbl_pos_upnl = _value_label("$0.00", "#9e9e9e")

        for title, widget in [
            ("Side", self.lbl_pos_side),
            ("Entry", self.lbl_pos_entry),
            ("Stop Loss", self.lbl_pos_sl),
            ("Take Profit", self.lbl_pos_tp),
            ("Unrealized PnL", self.lbl_pos_upnl),
        ]:
            row = QHBoxLayout()
            row.addWidget(_label(title))
            row.addWidget(widget)
            pos_layout.addLayout(row)

        left.addWidget(pos_box)

        # ---- middle: stats ----
        stats_box = QGroupBox("Stats")
        stats_box.setStyleSheet(account_box.styleSheet())
        stats_layout = QVBoxLayout(stats_box)
        stats_layout.setSpacing(4)

        self.lbl_trades = _value_label("0")
        self.lbl_winrate = _value_label("0.0%")
        self.lbl_best = _value_label("—", "#26a69a")
        self.lbl_worst = _value_label("—", "#ef5350")
        self.lbl_pf = _value_label("—")

        for title, widget in [
            ("Total Trades", self.lbl_trades),
            ("Win Rate", self.lbl_winrate),
            ("Best Trade", self.lbl_best),
            ("Worst Trade", self.lbl_worst),
            ("Profit Factor", self.lbl_pf),
        ]:
            row = QHBoxLayout()
            row.addWidget(_label(title))
            row.addWidget(widget)
            stats_layout.addLayout(row)

        # ---- right: log ----
        log_box = QGroupBox("Trade Log")
        log_box.setStyleSheet(account_box.styleSheet())
        log_layout = QVBoxLayout(log_box)
        self.trade_log = QTextEdit()
        self.trade_log.setReadOnly(True)
        self.trade_log.setFont(_MONO_SM)
        self.trade_log.setStyleSheet("background: #0f0f1e; color: #b0bec5; border: none;")
        self.trade_log.setMinimumWidth(300)
        log_layout.addWidget(self.trade_log)

        layout.addLayout(left, stretch=0)
        layout.addWidget(stats_box, stretch=0)
        layout.addWidget(log_box, stretch=1)

    # ----------------------------------------------------------------- update

    def update_account(self, trader: PaperTrader, current_price: float | None = None):
        equity = trader.equity_with_open(current_price) if current_price else trader.balance
        pnl = equity - trader.initial_balance
        pnl_color = "#26a69a" if pnl >= 0 else "#ef5350"
        sign = "+" if pnl >= 0 else ""

        self.lbl_balance.setText(f"${trader.balance:,.2f}")
        self.lbl_equity.setText(f"${equity:,.2f}")
        self.lbl_total_pnl.setText(f"{sign}${pnl:,.2f}")
        self.lbl_total_pnl.setStyleSheet(f"color: {pnl_color};")

        if trader.position and current_price:
            pos = trader.position
            upnl = pos.unrealized_pnl(current_price)
            upnl_color = "#26a69a" if upnl >= 0 else "#ef5350"
            side_color = "#26a69a" if pos.side == "long" else "#ef5350"
            sign2 = "+" if upnl >= 0 else ""

            self.lbl_pos_side.setText(pos.side.upper())
            self.lbl_pos_side.setStyleSheet(f"color: {side_color};")
            self.lbl_pos_entry.setText(f"{pos.entry_price:.4f}")
            self.lbl_pos_sl.setText(f"{pos.stop_loss:.4f}")
            self.lbl_pos_tp.setText(f"{pos.take_profit:.4f}")
            self.lbl_pos_upnl.setText(f"{sign2}${upnl:.2f}")
            self.lbl_pos_upnl.setStyleSheet(f"color: {upnl_color};")
        else:
            self.lbl_pos_side.setText("None")
            self.lbl_pos_side.setStyleSheet("color: #9e9e9e;")
            self.lbl_pos_entry.setText("—")
            self.lbl_pos_sl.setText("—")
            self.lbl_pos_tp.setText("—")
            self.lbl_pos_upnl.setText("$0.00")
            self.lbl_pos_upnl.setStyleSheet("color: #9e9e9e;")

        stats = trader.get_stats()
        self.lbl_trades.setText(str(stats["total_trades"]))
        self.lbl_winrate.setText(f"{stats['win_rate']:.1f}%")
        best = stats["best_trade"]
        worst = stats["worst_trade"]
        pf = stats["profit_factor"]
        self.lbl_best.setText(f"+${best:.2f}" if stats["total_trades"] else "—")
        self.lbl_worst.setText(f"${worst:.2f}" if stats["total_trades"] else "—")
        self.lbl_pf.setText(f"{pf:.2f}" if pf != float("inf") else "∞")

    def append_log(self, msg: str):
        if not msg:
            return
        self.trade_log.append(msg)
        sb = self.trade_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def clear_log(self):
        self.trade_log.clear()
