"""Backtest orientado a barra, sem look-ahead.

Sinal decidido no fechamento de t  ->  executado na ABERTURA de t+1.
Posição em lotes de `lot_size`, dimensionada como fração do patrimônio ('tudo
ou nada' no alvo {-1,0,+1}). Custos por ordem via `B3Costs`.

Simplificações desta versão (a trocar na Fase 1):
  - execução sempre no open, sem fila / fill parcial;
  - dinheiro parado não rende (sem CDI);
  - profit factor é aproximado (nível de barra, não round-trip).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..costs.b3 import B3Costs
from ..strategy.base import Strategy
from .metrics import summarize


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    stats: dict


def run(df: pd.DataFrame, strategy: Strategy, costs: B3Costs,
        initial_equity: float = 100_000.0, lot_size: int = 100) -> BacktestResult:
    df = df.sort_index()
    target = strategy.generate_signals(df).reindex(df.index).fillna(0.0)
    target_exec = target.shift(1).fillna(0.0)      # decidido em t, vale a partir de t+1

    open_ = df["open"].to_numpy(dtype="float64")
    close = df["close"].to_numpy(dtype="float64")
    tgt = target_exec.to_numpy(dtype="float64")
    n = len(df)

    cash = float(initial_equity)
    shares = 0.0
    equity = np.empty(n, dtype="float64")
    trades: list[dict] = []

    for i in range(n):
        price_open = open_[i]
        mtm = cash + shares * price_open
        desired = 0.0
        if tgt[i] != 0.0 and price_open > 0.0:
            raw = mtm * tgt[i] / price_open
            desired = np.trunc(raw / lot_size) * lot_size
        delta = desired - shares
        if delta != 0.0:
            fee = costs.order_cost(price_open, delta)
            cash -= delta * price_open + fee
            shares = desired
            trades.append({"time": df.index[i], "price": price_open,
                           "shares": delta, "fee": fee})
        equity[i] = cash + shares * close[i]

    eq = pd.Series(equity, index=df.index, name="equity")
    tr = (pd.DataFrame(trades).set_index("time")
          if trades else pd.DataFrame(columns=["price", "shares", "fee"]))
    return BacktestResult(eq, tr, summarize(eq, tr, initial_equity))
