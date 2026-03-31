from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Trade:
    side: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    entry_time: datetime
    exit_time: datetime

    @property
    def pnl_pct(self) -> float:
        return (self.pnl / (self.entry_price * self.size)) * 100

    @property
    def duration(self) -> str:
        delta = self.exit_time - self.entry_time
        h, rem = divmod(int(delta.total_seconds()), 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"


@dataclass
class Position:
    side: str            # "long" or "short"
    entry_price: float
    size: float          # in base asset
    entry_time: datetime
    stop_loss: float
    take_profit: float

    def unrealized_pnl(self, current_price: float) -> float:
        if self.side == "long":
            return (current_price - self.entry_price) * self.size
        return (self.entry_price - current_price) * self.size

    def pnl_pct(self, current_price: float) -> float:
        cost = self.entry_price * self.size
        return (self.unrealized_pnl(current_price) / cost) * 100 if cost else 0


class PaperTrader:
    def __init__(self, initial_balance: float = 10_000.0, risk_pct: float = 0.05,
                 sl_pct: float = 0.015, tp_pct: float = 0.03, confidence_threshold: float = 0.60):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_pct = risk_pct
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.confidence_threshold = confidence_threshold

        self.position: Optional[Position] = None
        self.trades: list[Trade] = []
        self.equity_curve: list[float] = [initial_balance]
        self.log: list[str] = []

    # ----------------------------------------------------------------- equity

    @property
    def total_equity(self) -> float:
        return self.balance

    def equity_with_open(self, current_price: float) -> float:
        if self.position:
            return self.balance + self.position.unrealized_pnl(current_price)
        return self.balance

    # ----------------------------------------------------------------- action

    def process_signal(self, direction: str, confidence: float,
                       current_price: float, timestamp: datetime | None = None) -> str | None:
        if timestamp is None:
            timestamp = datetime.now()

        messages = []

        # Check SL / TP first
        if self.position:
            msg = self._check_sl_tp(current_price, timestamp)
            if msg:
                messages.append(msg)

        # React to new signal
        if direction == "WAIT":
            return "\n".join(messages) or None

        if self.position:
            pos_side = self.position.side
            reverse = (pos_side == "long" and direction == "SHORT") or \
                      (pos_side == "short" and direction == "LONG")
            if reverse:
                msg = self._close_position(current_price, timestamp, reason="Signal Reversed")
                if msg:
                    messages.append(msg)

        if not self.position and confidence >= self.confidence_threshold:
            msg = self._open_position(direction.lower(), current_price, timestamp)
            if msg:
                messages.append(msg)

        return "\n".join(messages) or None

    def _open_position(self, side: str, price: float, ts: datetime) -> str:
        risk_amount = self.balance * self.risk_pct
        size = risk_amount / price

        if side == "long":
            sl = price * (1 - self.sl_pct)
            tp = price * (1 + self.tp_pct)
        else:
            sl = price * (1 + self.sl_pct)
            tp = price * (1 - self.tp_pct)

        self.position = Position(
            side=side, entry_price=price, size=size,
            entry_time=ts, stop_loss=sl, take_profit=tp,
        )
        msg = f"[{ts.strftime('%H:%M:%S')}] OPEN {side.upper()}  @ {price:.4f}  SL={sl:.4f}  TP={tp:.4f}"
        self.log.append(msg)
        return msg

    def _close_position(self, price: float, ts: datetime, reason: str = "") -> str | None:
        if not self.position:
            return None

        pnl = self.position.unrealized_pnl(price)
        self.balance += pnl

        trade = Trade(
            side=self.position.side,
            entry_price=self.position.entry_price,
            exit_price=price,
            size=self.position.size,
            pnl=pnl,
            entry_time=self.position.entry_time,
            exit_time=ts,
        )
        self.trades.append(trade)
        self.equity_curve.append(self.balance)

        sign = "+" if pnl >= 0 else ""
        msg = (f"[{ts.strftime('%H:%M:%S')}] CLOSE {self.position.side.upper()} "
               f"@ {price:.4f}  PnL={sign}{pnl:.2f} USDT  ({reason})")
        self.log.append(msg)
        self.position = None
        return msg

    def _check_sl_tp(self, price: float, ts: datetime) -> str | None:
        if not self.position:
            return None
        pos = self.position
        if pos.side == "long":
            if price <= pos.stop_loss:
                return self._close_position(price, ts, reason="Stop Loss")
            if price >= pos.take_profit:
                return self._close_position(price, ts, reason="Take Profit")
        else:
            if price >= pos.stop_loss:
                return self._close_position(price, ts, reason="Stop Loss")
            if price <= pos.take_profit:
                return self._close_position(price, ts, reason="Take Profit")
        return None

    # ------------------------------------------------------------------ stats

    def get_stats(self) -> dict:
        if not self.trades:
            return {
                "total_trades": 0, "win_rate": 0.0,
                "total_pnl": 0.0, "best_trade": 0.0, "worst_trade": 0.0,
                "avg_pnl": 0.0, "profit_factor": 0.0,
            }
        pnls = [t.pnl for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        return {
            "total_trades": len(self.trades),
            "win_rate": len(wins) / len(pnls) * 100,
            "total_pnl": sum(pnls),
            "best_trade": max(pnls),
            "worst_trade": min(pnls),
            "avg_pnl": sum(pnls) / len(pnls),
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        }

    def reset(self):
        self.balance = self.initial_balance
        self.position = None
        self.trades.clear()
        self.equity_curve = [self.initial_balance]
        self.log.clear()
