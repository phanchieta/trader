"""Painel (features + rótulo) para o modelo, com purga temporal.

label='return'         -> retorno futuro de `horizon` dias > cost_hurdle
label='triple_barrier' -> ver trader.ml.labels

`add_xs_ranks`: adiciona o percentil de algumas features NO CORTE TRANSVERSAL
(entre as ações do painel naquele dia) — informação relativa.
Purga: descarta as últimas (horizon + embargo) datas de treino.
"""
from __future__ import annotations

import pandas as pd

from .features import build_features
from .labels import triple_barrier

_META = {"symbol", "_fwd", "_y"}
_XS_COLS = ["mom_12_1", "mom_6_1", "ret_20", "ret_60", "vol_yz_20", "adx_14",
            "sma10_50", "dist_sma50_atr"]


def feature_columns(panel: pd.DataFrame) -> list[str]:
    return [c for c in panel.columns if c not in _META]


def build_panel(frames: dict, *, horizon: int = 10, label: str = "return",
                cost_hurdle: float = 0.004, pt_atr: float = 2.0, sl_atr: float = 1.5,
                add_xs_ranks: bool = True, drop_cols=("month",)) -> pd.DataFrame:
    parts = []
    for sym, df in frames.items():
        feats = build_features(df)
        if drop_cols:
            feats = feats.drop(columns=[c for c in drop_cols if c in feats.columns])
        fwd = df["close"].shift(-horizon) / df["close"] - 1.0
        if label == "triple_barrier":
            y = triple_barrier(df, horizon=horizon, pt_atr=pt_atr, sl_atr=sl_atr)
        else:
            y = (fwd > cost_hurdle).astype("float64")
        parts.append(feats.assign(symbol=sym, _fwd=fwd, _y=y))

    panel = pd.concat(parts).sort_index()
    if add_xs_ranks:
        for col in _XS_COLS:
            if col in panel.columns:
                panel[f"xs_{col}"] = panel.groupby(level=0)[col].rank(pct=True)
    panel = panel.dropna(subset=feature_columns(panel))
    return panel.dropna(subset=["_y"])


def purge(panel: pd.DataFrame, test_start: pd.Timestamp, *, horizon: int,
          embargo: int) -> pd.DataFrame:
    train_dates = panel.index[panel.index < test_start].unique().sort_values()
    if len(train_dates) <= horizon + embargo:
        return panel.iloc[:0]
    cutoff = train_dates[-(horizon + embargo)]
    return panel[panel.index < cutoff]
