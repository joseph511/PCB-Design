from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None
from sklearn.ensemble import HistGradientBoostingClassifier

from .pipeline import FEATURE_COLUMNS


def make_classifier():
    if LGBMClassifier is not None:
        return LGBMClassifier(
            n_estimators=500, learning_rate=0.03, max_depth=-1,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=42, class_weight="balanced"
        )
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.03, random_state=42)


def train_direction_model(df: pd.DataFrame, target: str):
    clean = df.dropna(subset=FEATURE_COLUMNS + [target]).copy()
    X = clean[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = clean[target].astype(int)
    model = make_classifier()
    model.fit(X, y)
    return model, clean


def walk_forward_report(df: pd.DataFrame, target: str, splits: int = 5) -> list[dict]:
    clean = df.dropna(subset=FEATURE_COLUMNS + [target]).copy()
    X = clean[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = clean[target].astype(int)
    tscv = TimeSeriesSplit(n_splits=splits)
    rows = []
    for fold, (tr, te) in enumerate(tscv.split(X), start=1):
        model = make_classifier()
        model.fit(X.iloc[tr], y.iloc[tr])
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X.iloc[te])[:, 1]
        else:
            proba = model.predict(X.iloc[te])
        pred = (proba >= 0.62).astype(int)
        auc = roc_auc_score(y.iloc[te], proba) if y.iloc[te].nunique() > 1 else np.nan
        rows.append({
            "fold": fold,
            "train_rows": len(tr),
            "test_rows": len(te),
            "auc": auc,
            "signals": int(pred.sum()),
            "signal_win_rate": float(y.iloc[te][pred == 1].mean()) if pred.sum() else np.nan,
        })
    return rows


def save_model(model, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path
