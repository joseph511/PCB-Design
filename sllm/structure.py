from __future__ import annotations
import numpy as np
import pandas as pd


def detect_swings(df: pd.DataFrame, atr_mult: float = 0.8) -> pd.DataFrame:
    """Adaptive zigzag-like swing detector using ATR threshold."""
    out = df.copy()
    n = len(out)
    out["swing_high"] = False
    out["swing_low"] = False
    out["structure_label"] = None

    if n < 20:
        return out

    direction = 0
    last_pivot_idx = 0
    last_pivot_price = out.loc[0, "close"]
    highs, lows = [], []

    for i in range(1, n):
        atr = out.loc[i, "atr"] if not pd.isna(out.loc[i, "atr"]) else out["range"].iloc[:i+1].mean()
        threshold = atr * atr_mult
        high, low = out.loc[i, "high"], out.loc[i, "low"]

        if direction >= 0:
            if high >= last_pivot_price:
                last_pivot_price = high
                last_pivot_idx = i
            elif last_pivot_price - low >= threshold:
                out.loc[last_pivot_idx, "swing_high"] = True
                highs.append((last_pivot_idx, last_pivot_price))
                direction = -1
                last_pivot_price = low
                last_pivot_idx = i
        if direction <= 0:
            if low <= last_pivot_price:
                last_pivot_price = low
                last_pivot_idx = i
            elif high - last_pivot_price >= threshold:
                out.loc[last_pivot_idx, "swing_low"] = True
                lows.append((last_pivot_idx, last_pivot_price))
                direction = 1
                last_pivot_price = high
                last_pivot_idx = i

    # HH/LH and HL/LL labeling
    prev_high = None
    prev_low = None
    for idx in range(n):
        if out.loc[idx, "swing_high"]:
            price = out.loc[idx, "high"]
            out.loc[idx, "structure_label"] = "HH" if prev_high is not None and price > prev_high else "LH"
            prev_high = price
        if out.loc[idx, "swing_low"]:
            price = out.loc[idx, "low"]
            out.loc[idx, "structure_label"] = "HL" if prev_low is not None and price > prev_low else "LL"
            prev_low = price

    return out


def add_bos_choch(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["last_swing_high"] = out["high"].where(out["swing_high"]).ffill()
    out["last_swing_low"] = out["low"].where(out["swing_low"]).ffill()
    out["bos_up"] = (out["close"] > out["last_swing_high"].shift(1)).astype(int)
    out["bos_down"] = (out["close"] < out["last_swing_low"].shift(1)).astype(int)

    trend = []
    state = "range"
    for _, r in out.iterrows():
        if r["bos_up"]:
            state = "up"
        elif r["bos_down"]:
            state = "down"
        trend.append(state)
    out["structure_state"] = trend
    out["choch_up"] = ((out["bos_up"] == 1) & (out["structure_state"].shift(1) == "down")).astype(int)
    out["choch_down"] = ((out["bos_down"] == 1) & (out["structure_state"].shift(1) == "up")).astype(int)
    return out
