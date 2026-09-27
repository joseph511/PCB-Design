from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:  # lets non-MT5 environments import safely
    mt5 = None

TIMEFRAME_MAP = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
}

@dataclass
class MT5Config:
    login: Optional[int] = None
    password: Optional[str] = None
    server: Optional[str] = None
    path: Optional[str] = None


def _require_mt5():
    if mt5 is None:
        raise ImportError("MetaTrader5 package is not installed. Run: pip install MetaTrader5")


def connect(config: MT5Config | None = None) -> None:
    """Initialize MT5. Leave credentials empty if terminal is already logged in."""
    _require_mt5()
    config = config or MT5Config()
    kwargs = {}
    if config.path:
        kwargs["path"] = config.path
    if config.login and config.password and config.server:
        kwargs.update(login=config.login, password=config.password, server=config.server)
    if not mt5.initialize(**kwargs):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")


def shutdown() -> None:
    if mt5 is not None:
        mt5.shutdown()


def timeframe_code(tf: str):
    _require_mt5()
    key = TIMEFRAME_MAP.get(tf.upper())
    if key is None:
        raise ValueError(f"Unsupported timeframe: {tf}")
    return getattr(mt5, key)


def fetch_rates(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch OHLCV bars from MT5 between UTC datetimes."""
    _require_mt5()
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"Could not select symbol {symbol}: {mt5.last_error()}")

    rates = mt5.copy_rates_range(symbol, timeframe_code(timeframe), start, end)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No data returned for {symbol} {timeframe}: {mt5.last_error()}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"tick_volume": "volume"})
    df["symbol"] = symbol
    df["timeframe"] = timeframe.upper()
    cols = ["time", "symbol", "timeframe", "open", "high", "low", "close", "volume", "spread", "real_volume"]
    return df[cols].sort_values("time").reset_index(drop=True)


def save_rates(df: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    symbol = df["symbol"].iloc[0]
    tf = df["timeframe"].iloc[0]
    path = out_dir / f"{symbol}_{tf}.parquet"
    df.to_parquet(path, index=False)
    return path
