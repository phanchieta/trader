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


def _low_beta(close: pd.Series, *, mkt: pd.Series, window: int = 252, **_) -> pd.Series:
    r, rm = close.pct_change(), mkt.reindex(close.index).pct_change()
    cov = r.rolling(window).cov(rm)
    var = rm.rolling(window).var()
    return -(cov / var.where(var > 0))                         # beta baixo = melhor


def _lt_reversal(close: pd.Series, **_) -> pd.Series:
    return -(close.shift(252) / close.shift(252 * 4) - 1.0)    # perdedor de 3a = melhor


def _st_reversal(close: pd.Series, **_) -> pd.Series:
    return -(close / close.shift(21) - 1.0)                    # perdedor do mês = melhor


def _div_yield(close: pd.Series, *, symbol=None, dividends=None, **_) -> pd.Series:
    if dividends is None or symbol is None:
        return pd.Series(np.nan, index=close.index)
    d = dividends.loc[dividends["symbol"] == symbol].set_index("ex_date")["amount"]
    if d.empty:
        return pd.Series(0.0, index=close.index)
    d = d[~d.index.duplicated()].reindex(close.index).fillna(0.0)
    return d.rolling(252).sum() / close.where(close > 0)       # DY 12m; maior = melhor


FACTORS = {
    "momentum": _mom_12_1,
    "low_vol": _low_vol,
    "low_beta": _low_beta,
    "lt_reversal": _lt_reversal,
    "st_reversal": _st_reversal,
    "div_yield": _div_yield,
}


def factor_score(frames: dict, name: str, *, mkt: pd.Series | None = None,
                 dividends: pd.DataFrame | None = None) -> pd.DataFrame:
    """DataFrame (datas × nomes) com o PERCENTIL cross-sectional do fator `name`
    (0..1; maior = mais atrativo). NaN onde não há dados suficientes."""
    fn = FACTORS[name]
    idx = None
    for f in frames.values():
        idx = f.index if idx is None else idx.union(f.index)
    idx = idx.sort_values()
    raw = {sym: fn(f["close"].reindex(idx), mkt=mkt, symbol=sym, dividends=dividends)
           for sym, f in frames.items()}
    return pd.DataFrame(raw).rank(axis=1, pct=True)


def combined_score(frames: dict, names: list[str], *, mkt: pd.Series | None = None,
                   dividends: pd.DataFrame | None = None) -> pd.DataFrame:
    """Média dos percentis dos fatores em `names`. Fatores sem dados (ex.:
    div_yield sem o parquet de proventos) são ignorados."""
    parts = [factor_score(frames, n, mkt=mkt, dividends=dividends) for n in names]
    parts = [p for p in parts if p.notna().to_numpy().any()]
    if not parts:
        raise ValueError("nenhum fator com dados")
    return sum(parts) / len(parts)
