"""Retorno de estratégia de "estar ou não estar posicionado" num ativo, com CDI no
caixa, custo por ponta sobre o giro e dois modos de execução.

`hold[t]` = exposição desejada durante o pregão t (retorno do fechamento de t−1 ao
fechamento de t), decidida até o fechamento de t−1.
  execution="close": ordem no leilão de fechamento de t−1 → pega o dia t inteiro.
  execution="open":  ordem na abertura de t → o gap noturno de t ainda é da
                      posição anterior.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backtest.metrics import vs_benchmark
from ..taxes import after_tax_summary


@dataclass
class TimingResult:
    equity: pd.Series
    returns: pd.Series
    stats: dict


def timing_returns(hold: pd.Series, opens: pd.Series, closes: pd.Series, cdi: pd.Series, *,
                   cost_side: float = 0.0004, execution: str = "close",
                   equity0: float = 1_000_000.0) -> TimingResult:
    idx = hold.index
    h = hold.astype(float).fillna(0.0)
    h_prev = h.shift(1).fillna(0.0)
    o, c = opens.reindex(idx), closes.reindex(idx)
    rate = cdi.reindex(idx).ffill().fillna(0.0)
    if execution == "close":
        risky = h * (c / c.shift(1) - 1.0).fillna(0.0)
    elif execution == "open":
        gap = (o / c.shift(1) - 1.0).fillna(0.0)
        intra = (c / o - 1.0).fillna(0.0)
        risky = h_prev * gap + h * intra
    else:
        raise ValueError(f"execution {execution!r}: use 'close' ou 'open'")
    ret = risky + (1.0 - h) * rate - (h - h_prev).abs() * cost_side
    equity = equity0 * (1.0 + ret).cumprod()
    b = vs_benchmark(equity, cdi)
    years = max((idx[-1] - idx[0]) / pd.Timedelta(days=365.25), 1e-9)
    stats = {
        "cagr": b["cagr_estrategia"], "alpha": b["alpha_aa"],
        "alpha_pos_ir": after_tax_summary(equity, cdi)["alpha_pos_ir"],
        "sharpe_cdi": b["sharpe_sobre_cdi"],
        "max_dd": round(float((equity / equity.cummax() - 1.0).min()), 4),
        "tempo_posicionado": round(float(h.mean()), 3),
        "idas_voltas_aa": round(float((h - h_prev).abs().sum() / 2.0 / years), 1),
    }
    return TimingResult(equity, ret, stats)
