from __future__ import annotations
import pandas as pd

from .indicators import add_atr, add_candle_features
from .structure import detect_swings, add_bos_choch
from .liquidity import add_liquidity_features, add_fvg_features
from .labeling import add_trade_labels


def build_feature_frame(df: pd.DataFrame, atr_period=14, swing_atr_mult=0.8) -> pd.DataFrame:
    out = add_atr(df, atr_period)
    out = add_candle_features(out)
    out = detect_swings(out, swing_atr_mult)
    out = add_bos_choch(out)
    out = add_liquidity_features(out)
    out = add_fvg_features(out)
    return out


def build_training_frame(df: pd.DataFrame, forward_bars=96, rr=2.0, stop_atr_mult=1.0) -> pd.DataFrame:
    out = build_feature_frame(df)
    out = add_trade_labels(out, forward_bars, rr, stop_atr_mult)
    return out

FEATURE_COLUMNS = [
    "atr", "body_atr", "upper_wick_ratio", "lower_wick_ratio", "close_pos",
    "displacement_score", "bos_up", "bos_down", "choch_up", "choch_down",
    "sweep_high", "sweep_low", "sweep_high_depth_atr", "sweep_low_depth_atr",
    "equal_high_zone", "equal_low_zone", "bullish_fvg", "bearish_fvg",
    "bullish_fvg_size_atr", "bearish_fvg_size_atr", "asian_session", "london_session", "ny_session",
]
