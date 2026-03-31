import ccxt
import pandas as pd
import numpy as np


class DataFetcher:
    def __init__(self):
        self.exchange = ccxt.binance({
            "options": {"defaultType": "future"},
            "enableRateLimit": True,
        })
        self._symbols_cache = []

    def get_futures_symbols(self) -> list[str]:
        markets = self.exchange.load_markets()
        symbols = [
            s for s, m in markets.items()
            if m.get("active") and m.get("type") == "swap" and s.endswith(":USDT")
        ]
        self._symbols_cache = sorted(symbols)
        return self._symbols_cache

    # Binance Futures max candles per request
    _PAGE_SIZE = 1500

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 2000) -> pd.DataFrame:
        """Fetch up to `limit` candles, paginating backwards through history."""
        PAGE = self._PAGE_SIZE
        tf_ms = self._tf_ms(timeframe)

        # --- first page: most recent candles ---
        rows = self.exchange.fetch_ohlcv(symbol, timeframe, limit=min(limit, PAGE))
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        all_rows: list = rows

        # --- walk backwards until we have enough ---
        while len(all_rows) < limit:
            need = limit - len(all_rows)
            page = min(need, PAGE)
            oldest_ts = all_rows[0][0]
            since = oldest_ts - tf_ms * page
            rows = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=page)
            if not rows:
                break
            # keep only rows strictly older than what we already have
            new_rows = [r for r in rows if r[0] < oldest_ts]
            if not new_rows:
                break
            all_rows = new_rows + all_rows

        df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df.astype(float)
        return df.iloc[-limit:]

    @staticmethod
    def _tf_ms(timeframe: str) -> int:
        """Return the duration of one candle in milliseconds."""
        units = {"m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}
        mul = int(timeframe[:-1])
        unit = timeframe[-1]
        return mul * units.get(unit, 60_000)

    def fetch_latest_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 5) -> pd.DataFrame:
        return self.fetch_ohlcv(symbol, timeframe, limit=limit)

    def fetch_ticker(self, symbol: str) -> dict:
        return self.exchange.fetch_ticker(symbol)
