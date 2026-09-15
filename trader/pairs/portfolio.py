"""Carteira de pares com número limitado de posições abertas (slots).

Cada par aberto usa +c numa ponta e −c na outra, c = gross/(2·slots): com todos os
slots ocupados a exposição bruta é `gross`. Novos sinais entram por ordem de |z|
enquanto houver slot; posição aberta não é desalojada.

Retorno diário com execução na ABERTURA seguinte ao sinal: o peso decidido no
fechamento de t vale do open de t+1 em diante (o gap noturno de t+1 ainda é do
peso anterior). A carteira é neutra em valor: o dinheiro da venda a descoberto
paga a compra, então o patrimônio inteiro fica aplicado no CDI; paga-se custo por
ponta sobre o giro e aluguel sobre o lado vendido.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backtest.metrics import summarize, vs_benchmark
from ..taxes import after_tax_summary


@dataclass
class PairsResult:
    equity: pd.Series
    excess: pd.Series
    contrib: pd.DataFrame
    turnover: pd.Series
    stats: dict


def build_weights(positions: dict, zscores: dict, legs: dict, index: pd.DatetimeIndex, *,
                  slots: int = 8, gross: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """positions/zscores: {par: array alinhado a `index`}; legs: {par: (A, B)}.
    Devolve (pesos por ação, posição efetiva por par) após a alocação de slots."""
    names = list(positions)
    P = np.column_stack([positions[p] for p in names]) if names else np.zeros((len(index), 0))
    Z = np.column_stack([np.nan_to_num(zscores[p]) for p in names]) if names else P
    held = np.zeros_like(P)
    cur = np.zeros(len(names))
    for t in range(len(index)):
        want = P[t]
        cur = np.where(np.sign(cur) == np.sign(want), cur, 0.0)
        free = slots - int((cur != 0).sum())
        if free > 0:
            cand = np.nonzero((cur == 0) & (want != 0))[0]
            cand = cand[np.argsort(-np.abs(Z[t, cand]))][:free]
            cur[cand] = want[cand]
        held[t] = cur
    c = gross / (2.0 * slots)
    syms = sorted({s for ab in legs.values() for s in ab})
    W = pd.DataFrame(0.0, index=index, columns=syms)
    for j, p in enumerate(names):
        a, b = legs[p]
        W[a] += c * held[:, j]
        W[b] -= c * held[:, j]
    return W, pd.DataFrame(held, index=index, columns=names)


def pairs_portfolio(weights: pd.DataFrame, held: pd.DataFrame, opens: pd.DataFrame,
                    closes: pd.DataFrame, legs: dict, cdi: pd.Series, *,
                    cost_side: float = 0.0008, borrow_aa: float = 0.02,
                    equity0: float = 1_000_000.0) -> PairsResult:
    idx = weights.index
    O, C = opens.reindex(idx)[weights.columns], closes.reindex(idx)[weights.columns]
    gap = (O / C.shift(1) - 1.0).fillna(0.0)
    intra = (C / O - 1.0).fillna(0.0)
    w_now, w_prev = weights.shift(1).fillna(0.0), weights.shift(2).fillna(0.0)
    contrib = w_prev * gap + w_now * intra
    turnover = (w_now - w_prev).abs().sum(axis=1)
    short = (-w_now).clip(lower=0.0).sum(axis=1)
    rate = cdi.reindex(idx).ffill().fillna(0.0)
    borrow_d = (1.0 + borrow_aa) ** (1.0 / 252.0) - 1.0
    excess = contrib.sum(axis=1) - turnover * cost_side - short * borrow_d
    equity = equity0 * (1.0 + rate + excess).cumprod()

    h = held.shift(1).fillna(0.0)
    entries = int(((h != 0) & (h.shift(1).fillna(0.0) != h)).to_numpy().sum())
    years = max((idx[-1] - idx[0]) / pd.Timedelta(days=365.25), 1e-9)
    b = vs_benchmark(equity, cdi)
    stats = {
        "cagr": b["cagr_estrategia"], "alpha": b["alpha_aa"],
        "alpha_pos_ir": after_tax_summary(equity, cdi)["alpha_pos_ir"],
        "sharpe_cdi": b["sharpe_sobre_cdi"],
        "max_dd": summarize(equity, pd.DataFrame(columns=["fee"]), equity0)["max_drawdown"],
        "expo_bruta": round(float(weights.abs().sum(axis=1).mean()), 3),
        "giro_aa": round(float(turnover.sum() / years), 2),
        "trades_aa": round(entries / years, 1),
    }
    return PairsResult(equity, excess, contrib, turnover, stats)
