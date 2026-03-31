# 🤖 LSTM Binance Futures Bot

> A PyQt6 desktop trading bot that uses a multi-layer LSTM neural network to predict price direction on Binance Futures and paper trades automatically in real time.

---

## ✨ Features

| Feature | Details |
|---|---|
| 📈 **Live Candlestick Chart** | Real-time OHLCV chart with EMA20/50, Bollinger Bands, volume bars and buy/sell signal markers |
| 🧠 **LSTM Predictor** | 3-layer LSTM (128→64→32) trained on 14 technical features per candle |
| 🔍 **Symbol Picker** | Search and select any USDT-margined Binance Futures pair |
| ⏱ **Multi-Timeframe** | 1m · 3m · 5m · 15m · 30m · 1h · 4h · 1d |
| 📦 **Deep History** | Paginated fetcher — pull up to **5000 candles** across multiple API pages |
| 💰 **Paper Trading** | Auto position management with configurable SL / TP, 5% risk per trade |
| 📊 **Live Stats** | Win rate, profit factor, best/worst trade, equity tracking, full trade log |

---

## 🖥️ Screenshot

```
┌─────────────────────────────────────────────────────────────────────┐
│ Symbol: [BTC/USDT ▼]  TF: [1h ▼]  2000 bars  ⬇ Fetch  ⚡ Train   │
│                          50 epochs  ▶ Start Bot  ⏹ Stop  ↺ Reset   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   🕯 Candlestick chart  ── EMA20  ── EMA50  ── BB Upper/Lower      │
│   ▲ Long signals (green)  ▼ Short signals (red)                    │
│   ▬ Volume bars                                                     │
│                                                                     │
├──────────────┬──────────────────────────────────────────────────────┤
│ LSTM         │ Account · Position · Stats · Trade Log               │
│ Prediction   │                                                      │
│  LONG  72%   │ Balance: $10,432   Win Rate: 58.3%                  │
└──────────────┴──────────────────────────────────────────────────────┘
```

---

## 🚀 Setup

### 1. Clone / download

```bash
git clone https://github.com/MohammedAl-harthi/LSTM-trading-bot.git
cd lstm-futures-bot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run

```bash
python main.py
```

> No API key required — all market data is fetched from public Binance endpoints.

---

## 📦 Requirements

```
Python  3.11+
PyQt6   6.6+
pyqtgraph
ccxt
numpy · pandas · scikit-learn
tensorflow  2.15+
```

---

## 🗂️ Project Structure

```
LSTM/
├── main.py                  # Entry point
├── requirements.txt
│
├── core/
│   ├── data_fetcher.py      # Paginated Binance Futures OHLCV downloader
│   ├── indicators.py        # RSI, MACD, Bollinger Bands, ATR, EMA
│   ├── lstm_model.py        # Keras LSTM model — train & predict
│   └── paper_trader.py      # Position manager with SL / TP engine
│
└── ui/
    ├── main_window.py       # Main PyQt6 window + background QThread workers
    ├── chart_widget.py      # Live candlestick chart (pyqtgraph)
    └── trading_panel.py     # Account · position · stats · trade log widgets
```

---

## 🧠 LSTM Architecture

```
Input  →  (sequence_length=60, features=14)
          ↓
     LSTM  128  (return_sequences=True)
     Dropout 0.2  +  BatchNorm
          ↓
     LSTM  64   (return_sequences=True)
     Dropout 0.2  +  BatchNorm
          ↓
     LSTM  32
     Dropout 0.2
          ↓
     Dense 16  (ReLU)
          ↓
     Dense  1  (Sigmoid)  →  P(next candle UP)
```

**14 input features per candle:**
`open · high · low · close · volume · RSI-14 · MACD · MACD signal · MACD histogram · Bollinger %B · volume ratio · 1-bar return · 5-bar return · EMA cross`

Training uses **Early Stopping** (patience=7) on validation accuracy with an 80/20 train-val split.

---

## 💹 Paper Trading Logic

| Rule | Value |
|---|---|
| Enter **LONG** | prediction probability ≥ 60% |
| Enter **SHORT** | prediction probability ≤ 40% |
| Exit | opposite signal **or** SL/TP hit |
| Stop Loss | 1.5% from entry |
| Take Profit | 3.0% from entry |
| Position size | 5% of current balance |
| Starting balance | $10,000 USDT |

---

## 🔄 Workflow

```
1. Select symbol  →  BTC/USDT:USDT, ETH/USDT:USDT, SOL/USDT:USDT …
2. Choose timeframe and bar count (200 – 5000)
3. ⬇  Fetch   — downloads history (paginates automatically)
4. ⚡  Train   — LSTM trains in background thread, shows epoch progress
5. ▶  Start   — bot auto-refreshes each new candle, predicts, trades
6. ↺  Reset   — clears trades and resets balance to $10,000
```

---

## ⚠️ Disclaimer

This project is for **educational purposes only**.  
Paper trading results do not guarantee real-world performance.  
Never trade with money you cannot afford to lose.

---

## 📄 License

MIT
