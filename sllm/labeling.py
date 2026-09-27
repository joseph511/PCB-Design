from __future__ import annotations
import numpy as np
import pandas as pd


def add_trade_labels(df: pd.DataFrame, forward_bars: int = 96, rr: float = 2.0, stop_atr_mult: float = 1.0) -> pd.DataFrame:
    """Label whether long/short hits +RR before -1R over future bars."""
    out = df.copy()
    out["long_win_2r"] = np.nan
    out["short_win_2r"] = np.nan
    out["long_mfe_r"] = np.nan
    out["long_mae_r"] = np.nan
    out["short_mfe_r"] = np.nan
    out["short_mae_r"] = np.nan

    for i in range(len(out) - forward_bars - 1):
        entry = out.loc[i, "close"]
        risk = out.loc[i, "atr"] * stop_atr_mult
        if pd.isna(risk) or risk <= 0:
            continue
        future = out.iloc[i+1:i+1+forward_bars]
        long_tp, long_sl = entry + rr * risk, entry - risk
        short_tp, short_sl = entry - rr * risk, entry + risk

        long_result = 0
        short_result = 0
        for _, r in future.iterrows():
            if r["low"] <= long_sl:
                break
            if r["high"] >= long_tp:
                long_result = 1
                break
        for _, r in future.iterrows():
            if r["high"] >= short_sl:
                break
            if r["low"] <= short_tp:
                short_result = 1
                break

        out.loc[i, "long_win_2r"] = long_result
        out.loc[i, "short_win_2r"] = short_result
        out.loc[i, "long_mfe_r"] = (future["high"].max() - entry) / risk
        out.loc[i, "long_mae_r"] = (entry - future["low"].min()) / risk
        out.loc[i, "short_mfe_r"] = (entry - future["low"].min()) / risk
        out.loc[i, "short_mae_r"] = (future["high"].max() - entry) / risk
    return out
