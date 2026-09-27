from __future__ import annotations
import numpy as np
import pandas as pd


def add_liquidity_features(df: pd.DataFrame, equal_tol_atr: float = 0.12, sweep_min_atr: float = 0.05) -> pd.DataFrame:
    out = df.copy()
    atr = out["atr"].replace(0, np.nan)
    prev_high = out["last_swing_high"].shift(1)
    prev_low = out["last_swing_low"].shift(1)

    out["sweep_high"] = ((out["high"] > prev_high) & (out["close"] < prev_high) & ((out["high"] - prev_high) / atr > sweep_min_atr)).astype(int)
    out["sweep_low"] = ((out["low"] < prev_low) & (out["close"] > prev_low) & ((prev_low - out["low"]) / atr > sweep_min_atr)).astype(int)
    out["sweep_high_depth_atr"] = (out["high"] - prev_high) / atr
    out["sweep_low_depth_atr"] = (prev_low - out["low"]) / atr

    out["equal_high_zone"] = ((out["high"] - prev_high).abs() / atr <= equal_tol_atr).astype(int)
    out["equal_low_zone"] = ((out["low"] - prev_low).abs() / atr <= equal_tol_atr).astype(int)

    session = out["time"].dt.hour
    out["asian_session"] = session.between(0, 6).astype(int)
    out["london_session"] = session.between(7, 12).astype(int)
    out["ny_session"] = session.between(13, 20).astype(int)
    return out


def add_fvg_features(df: pd.DataFrame, min_atr: float = 0.10) -> pd.DataFrame:
    out = df.copy()
    atr = out["atr"].replace(0, np.nan)
    out["bullish_fvg"] = ((out["low"] > out["high"].shift(2)) & ((out["low"] - out["high"].shift(2)) / atr > min_atr)).astype(int)
    out["bearish_fvg"] = ((out["high"] < out["low"].shift(2)) & ((out["low"].shift(2) - out["high"]) / atr > min_atr)).astype(int)
    out["bullish_fvg_size_atr"] = (out["low"] - out["high"].shift(2)) / atr
    out["bearish_fvg_size_atr"] = (out["low"].shift(2) - out["high"]) / atr
    return out
