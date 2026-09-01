"""Overlay de vol-target: multiplicador de exposição [0, max_leverage] por barra.

= vol_alvo / vol_realizada da carteira equal-weight. Corta exposição quando o
mercado fica volátil (é o que dói nos drawdowns de −40%+). Nunca alavanca:
o multiplicador é limitado em `max_leverage` (default 1.0).

Causal: usa retornos até t−1 (shift), warmup -> 0.
"""
from __future__ import annotations

import pandas as pd

from ..backtest.metrics import _periods_per_year
from ..indicators import adx, garman_klass, parkinson, yang_zhang

_ESTIMATORS = {"parkinson": parkinson, "garman_klass": garman_klass,
               "yang_zhang": yang_zhang}


def common_index(frames: dict) -> pd.DatetimeIndex:
    idx = None
    for f in frames.values():
        idx = f.index if idx is None else idx.intersection(f.index)
    return idx.sort_values()


def vol_target_multiplier(frames: dict, *, target_ann_vol: float = 0.15,
                          lookback: int = 20, max_leverage: float = 1.0,
                          estimator: str = "close") -> pd.Series:
    """`estimator`: 'close' (close-a-close, padrão) ou 'parkinson' /
    'garman_klass' / 'yang_zhang' (usam o OHLC inteiro)."""
    idx = common_index(frames)
    if estimator == "close":
        rets = pd.DataFrame({s: frames[s]["close"].reindex(idx).pct_change()
                             for s in frames})
        ppy = _periods_per_year(idx)
        realized = rets.mean(axis=1).rolling(lookback).std() * (ppy ** 0.5)
    else:
        fn = _ESTIMATORS[estimator]
        vols = pd.DataFrame({s: fn(frames[s].reindex(idx), window=lookback)
                             for s in frames})
        realized = vols.mean(axis=1)
    mult = (target_ann_vol / realized).clip(upper=max_leverage)
    return mult.shift(1).reindex(idx).fillna(0.0).rename("mult")


def inv_vol_weights(frames: dict, *, lookback: int = 20,
                    estimator: str = "yang_zhang") -> pd.DataFrame:
    """Pesos por ação ∝ 1 / volatilidade (dá mais capital aos nomes mais calmos).
    Somam 1 por dia. Causal (shift 1). Fallback: equal-weight no aquecimento."""
    idx = common_index(frames)
    fn = _ESTIMATORS.get(estimator)
    if fn is None:
        vols = pd.DataFrame({s: frames[s]["close"].reindex(idx).pct_change()
                             .rolling(lookback).std() for s in frames})
    else:
        vols = pd.DataFrame({s: fn(frames[s].reindex(idx), window=lookback)
                             for s in frames})
    inv = 1.0 / vols.where(vols > 0)
    w = inv.div(inv.sum(axis=1), axis=0)
    return w.shift(1).reindex(idx).ffill().fillna(1.0 / len(frames))


def adx_scale_multiplier(frames: dict, *, period: int = 14, adx_floor: float = 15.0,
                         adx_cap: float = 40.0) -> pd.Series:
    """Escala de exposição [0, 1] pela força da tendência (ADX médio da cesta):
    0 quando ADX <= floor (sem direção), 1 quando ADX >= cap. Versão contínua
    do filtro de regime — o gate binário atrapalhou, escalar não."""
    idx = common_index(frames)
    a = pd.DataFrame({s: adx(frames[s].reindex(idx), period) for s in frames}).mean(axis=1)
    m = ((a - adx_floor) / (adx_cap - adx_floor)).clip(0.0, 1.0)
    return m.shift(1).reindex(idx).fillna(0.0).rename("mult")
