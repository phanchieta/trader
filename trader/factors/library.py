"""Biblioteca de fatores. Cada função devolve, por nome, uma série "raw" onde
MAIOR = MAIS ATRATIVO (para ficar comprado). O cross-section é rankeado depois.

Só fatores de PREÇO (sem fundamentos): momentum, baixa-vol, baixo-beta,
reversão de longo prazo (proxy de valor), reversão de curto prazo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _mom_12_1(close: pd.Series, **_) -> pd.Series:
    return close.shift(21) / close.shift(252) - 1.0            # maior = melhor


def _low_vol(close: pd.Series, window: int = 120, **_) -> pd.Series:
    return -(close.pct_change().rolling(window).std())         # menos vol = melhor


def _low_beta(close: pd.Series, mkt: pd.Series, window: int = 252, **_) -> pd.Series:
    r, rm = close.pct_change(), mkt.reindex(close.index).pct_change()
    cov = r.rolling(window).cov(rm)
    var = rm.rolling(window).var()
    return -(cov / var.where(var > 0))                         # beta baixo = melhor


def _lt_reversal(close: pd.Series, **_) -> pd.Series:
    return -(close.shift(252) / close.shift(252 * 4) - 1.0)    # perdedor de 3a = melhor


def _st_reversal(close: pd.Series, **_) -> pd.Series:
    return -(close / close.shift(21) - 1.0)                    # perdedor do mês = melhor


FACTORS = {
    "momentum": _mom_12_1,
    "low_vol": _low_vol,
    "low_beta": _low_beta,
    "lt_reversal": _lt_reversal,
    "st_reversal": _st_reversal,
}


def factor_score(frames: dict, name: str, *, mkt: pd.Series | None = None) -> pd.DataFrame:
    """DataFrame (datas × nomes) com o PERCENTIL cross-sectional do fator `name`
    (0..1; maior = mais atrativo). NaN onde não há dados suficientes."""
    fn = FACTORS[name]
    idx = None
    for f in frames.values():
        idx = f.index if idx is None else idx.union(f.index)
    idx = idx.sort_values()
    raw = {}
    for sym, f in frames.items():
        c = f["close"].reindex(idx)
        raw[sym] = fn(c, mkt=mkt) if name == "low_beta" else fn(c)
    R = pd.DataFrame(raw)
    return R.rank(axis=1, pct=True)


def combined_score(frames: dict, names: list[str], *, mkt: pd.Series | None = None
                   ) -> pd.DataFrame:
    """Média dos percentis dos fatores em `names`."""
    parts = [factor_score(frames, n, mkt=mkt) for n in names]
    return sum(parts) / len(parts)
