"""Pesos-alvo decididos no último pregão de cada mês (só com preços até ele).

Todas devolvem um DataFrame indexado pelos fins de mês, colunas = ativos, soma ≤ 1;
o que falta para 1 fica no CDI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).max().to_numpy())


def _me_prices(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.reindex(month_ends(prices.index))


def static_weights(prices: pd.DataFrame, weights: dict) -> pd.DataFrame:
    me = _me_prices(prices)
    w = pd.DataFrame({a: float(weights.get(a, 0.0)) for a in prices.columns}, index=me.index)
    return w.where(me.notna(), 0.0)


def faber_sma(prices: pd.DataFrame, *, months: int = 10) -> pd.DataFrame:
    """Faber (2007): cada ativo recebe 1/N se o fechamento do mês está acima da média
    dos últimos `months` fechamentos mensais; senão sua fatia vai para o CDI. N = ativos
    com histórico suficiente naquele mês."""
    me = _me_prices(prices)
    sma = me.rolling(months, min_periods=months).mean()
    ready = sma.notna() & me.notna()
    n = ready.sum(axis=1).replace(0, np.nan)
    on = (me > sma) & ready
    return on.astype(float).div(n, axis=0).fillna(0.0)


def dual_momentum(prices: pd.DataFrame, cdi: pd.Series, *, months: int = 12) -> pd.DataFrame:
    """Antonacci (GEM): todo o capital no ativo de maior retorno em `months` meses, se esse
    retorno superar o do CDI no mesmo período; senão, CDI."""
    me = _me_prices(prices)
    mom = me / me.shift(months) - 1.0
    cdi_idx = (1.0 + cdi.reindex(prices.index).fillna(0.0)).cumprod().reindex(me.index)
    cdi_mom = cdi_idx / cdi_idx.shift(months) - 1.0
    best = mom.dropna(how="all").idxmax(axis=1)
    w = pd.DataFrame(0.0, index=me.index, columns=prices.columns)
    for d, b in best.items():
        if mom.loc[d, b] > cdi_mom.loc[d]:
            w.loc[d, b] = 1.0
    return w
