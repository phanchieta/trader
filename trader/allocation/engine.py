"""Simulação diária de uma carteira de ETFs com rebalanceamento mensal.

Pesos decididos no fechamento do fim de mês t são executados no fechamento do pregão
seguinte (um dia de defasagem, causal). Entre rebalanceamentos os pesos flutuam com os
preços; a parte não investida rende CDI. Custo por rebalanceamento: `cost_bps` sobre o
giro (taxa B3 + slippage) mais `order_cost` reais por ordem, sobre o patrimônio.
`band`: só negocia um ativo se o sinal mudou (entrou/saiu) ou se o desvio do alvo
passou dessa fração do alvo.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backtest.metrics import vs_benchmark
from ..taxes import after_tax_summary


@dataclass
class AllocResult:
    equity: pd.Series
    weights: pd.DataFrame
    stats: dict


def run_allocation(prices: pd.DataFrame, targets: pd.DataFrame, cdi: pd.Series, *,
                   cost_bps: float = 5.0, order_cost: float = 0.0, capital: float = 100_000.0,
                   band: float = 0.0) -> AllocResult:
    idx = prices.index
    if (idx.tz is None) != (cdi.index.tz is None):              # alinha o fuso do CDI aos preços
        cdi = cdi.set_axis(cdi.index.tz_convert(None).normalize() if idx.tz is None
                           else cdi.index.tz_localize(idx.tz))
    rets = prices.pct_change().fillna(0.0).to_numpy()
    rate = cdi.reindex(idx).ffill().fillna(0.0).to_numpy()
    n = len(idx)
    exec_day = {}
    pos = idx.get_indexer(targets.index)
    for d, p in zip(targets.index, pos):
        if 0 <= p < n - 1:
            exec_day[p + 1] = targets.loc[d].reindex(prices.columns).fillna(0.0).to_numpy()
    w = np.zeros(prices.shape[1])
    eq = np.empty(n)
    wts = np.empty((n, prices.shape[1]))
    v = capital
    orders, turnover = 0, 0.0
    for i in range(n):
        if i > 0:
            gross = w * (1.0 + rets[i])
            cash = (1.0 - w.sum()) * (1.0 + rate[i])
            total = gross.sum() + cash
            v *= total
            w = gross / total
        if i in exec_day:
            tgt = exec_day[i]
            flip = (tgt > 0) != (w > 1e-12)
            dev = np.abs(tgt - w) > band * np.maximum(tgt, 1e-12)
            trade = flip | (dev if band > 0 else np.abs(tgt - w) > 1e-9)
            new = np.where(trade, tgt, w)
            if new.sum() > 1.0:
                new = new / new.sum()
            dw = np.abs(new - w)
            k = int((dw > 1e-9).sum())
            cost = dw.sum() * cost_bps / 1e4 + (k * order_cost / v if v > 0 else 0.0)
            v *= 1.0 - cost
            orders += k
            turnover += dw.sum()
            w = new
        eq[i] = v
        wts[i] = w
    equity = pd.Series(eq, index=idx, name="equity")
    utc = lambda ix: ix.tz_localize("UTC") if ix.tz is None else ix.tz_convert("UTC")
    eq_utc = equity.set_axis(utc(idx))
    cdi_utc = cdi.set_axis(utc(cdi.index))                 # métricas exigem o mesmo fuso
    b = vs_benchmark(eq_utc, cdi_utc)
    years = max((idx[-1] - idx[0]).days / 365.25, 1e-9)
    stats = {"cagr": b["cagr_estrategia"], "alpha": b["alpha_aa"],
             "alpha_pos_ir": after_tax_summary(eq_utc, cdi_utc)["alpha_pos_ir"],
             "sharpe_cdi": b["sharpe_sobre_cdi"],
             "max_dd": round(float((equity / equity.cummax() - 1.0).min()), 4),
             "exposicao": round(float(wts.sum(axis=1).mean()), 3),
             "ordens_aa": round(orders / years, 1), "giro_aa": round(turnover / years, 2)}
    return AllocResult(equity, pd.DataFrame(wts, index=idx, columns=prices.columns), stats)
