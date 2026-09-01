"""Backtest de trend multi-ativo baseado em RETORNOS (não em lotes/ações).

Adequado para misturar ações e futuros (WIN/WDO): permite long/short e
alavancagem implícita. Vol-target interno: escala o livro pela vol realizada
dos próprios retornos não-alavancados da estratégia. Caixa livre rende CDI.

  retorno_dia = Σ peso_i·retorno_i  +  (1 − exposição_bruta)·CDI  −  custo_de_giro
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import summarize


@dataclass
class TrendResult:
    equity: pd.Series
    stats: dict
    exposure_media: float
    turnover_medio: float


def run_trend_portfolio(frames: dict, signals: pd.DataFrame, *,
                        base_weights: dict | None = None, target_vol: float = 0.15,
                        vol_lookback: int = 40, max_leverage: float = 2.0,
                        cash_rate: pd.Series | None = None,
                        cost_bps: dict | None = None,
                        initial_equity: float = 100_000.0) -> TrendResult:
    cols = list(signals.columns)
    idx = signals.index
    bw = base_weights or {c: 1.0 / len(cols) for c in cols}
    cb = cost_bps or {c: 8.0 for c in cols}

    rets = pd.DataFrame({c: frames[c]["close"].reindex(idx).pct_change().fillna(0.0)
                         for c in cols})
    s = signals.reindex(idx).fillna(0.0).shift(1).fillna(0.0)

    w_unlev = s.mul(pd.Series(bw), axis=1)
    unlev_ret = (w_unlev * rets).sum(axis=1)
    realized = unlev_ret.rolling(vol_lookback).std() * np.sqrt(252.0)
    mult = (target_vol / realized.where(realized > 0)).clip(upper=max_leverage)
    mult = mult.shift(1).fillna(0.0)

    w = w_unlev.mul(mult, axis=0)
    turnover = w.diff().abs()
    turnover.iloc[0] = w.iloc[0].abs()
    cost = turnover.mul(pd.Series({c: cb[c] / 1e4 for c in cols}), axis=1).sum(axis=1)

    cdi = (cash_rate.reindex(idx).ffill().fillna(0.0) if cash_rate is not None
           else pd.Series(0.0, index=idx))
    gross_exp = w.abs().sum(axis=1)
    port_ret = (w * rets).sum(axis=1) + (1.0 - gross_exp).clip(lower=0.0) * cdi - cost

    equity = (initial_equity * (1.0 + port_ret).cumprod()).rename("equity")
    stats = summarize(equity, pd.DataFrame(columns=["price", "shares", "fee"]),
                      initial_equity)
    return TrendResult(equity, stats, round(float(gross_exp.mean()), 3),
                       round(float(turnover.sum(axis=1).mean()), 4))
