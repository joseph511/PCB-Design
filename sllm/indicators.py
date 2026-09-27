from __future__ import annotations
import numpy as np
import pandas as pd


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    out = df.copy()
    prev_close = out["close"].shift(1)
    tr = pd.concat([
        out["high"] - out["low"],
        (out["high"] - prev_close).abs(),
        (out["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["tr"] = tr
    out["atr"] = tr.ewm(alpha=1 / period, adjust=False).mean()
    return out


def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rng = (out["high"] - out["low"]).replace(0, np.nan)
    body = (out["close"] - out["open"]).abs()
    out["body"] = body
    out["range"] = rng
    out["upper_wick"] = out["high"] - out[["open", "close"]].max(axis=1)
    out["lower_wick"] = out[["open", "close"]].min(axis=1) - out["low"]
    out["body_atr"] = body / out["atr"].replace(0, np.nan)
    out["upper_wick_ratio"] = out["upper_wick"] / rng
    out["lower_wick_ratio"] = out["lower_wick"] / rng
    out["close_pos"] = (out["close"] - out["low"]) / rng
    out["bullish"] = (out["close"] > out["open"]).astype(int)
    out["bearish"] = (out["close"] < out["open"]).astype(int)
    out["displacement_score"] = out["body_atr"] * (out["close_pos"] - 0.5).abs() * 2
    return out
