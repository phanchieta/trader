"""O mesmo alvo de carteira executado em horários diferentes.

`weights[t]` = peso-alvo por ação decidido até o fechamento de t (sinal × overlay / N).
Quando executar cada lado (compra = aumentar posição, venda = reduzir):
  "open_next"   abertura de t+1 — o padrão do projeto
  "close_same"  leilão de fechamento de t — o sinal usa o próprio fechamento de t,
                que não se conhece a tempo: é limite superior
  "close_next"  leilão de fechamento de t+1 — totalmente causal
Tamanho da ordem = patrimônio no instante da execução × peso / preço, em lotes.
Caixa rende CDI; custo B3 por ordem.
`close_same_weights`: pesos usados nas execuções "close_same" (ex.: sinal calculado
com o preço de minutos antes do leilão, em vez do fechamento final).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..costs.b3 import B3Costs
from .metrics import summarize

_WHEN = ("open_next", "close_same", "close_next")


@dataclass
class ExecResult:
    equity: pd.Series
    stats: dict


def run_exec_timing(frames: dict, weights: pd.DataFrame, costs: B3Costs, *,
                    buy_at: str = "open_next", sell_at: str = "open_next",
                    cash_rate: pd.Series | None = None, initial_equity: float = 1_000_000.0,
                    lot_size: int = 100,
                    close_same_weights: pd.DataFrame | None = None) -> ExecResult:
    if buy_at not in _WHEN or sell_at not in _WHEN:
        raise ValueError(f"buy_at/sell_at devem ser um de {_WHEN}")
    syms = list(weights.columns)
    idx = weights.index
    n = len(idx)
    op = {s: np.nan_to_num(frames[s]["open"].reindex(idx).to_numpy("float64")) for s in syms}
    cl = {s: np.nan_to_num(frames[s]["close"].reindex(idx).to_numpy("float64")) for s in syms}
    W = weights.fillna(0.0).to_numpy("float64")
    Wc = (W if close_same_weights is None
          else close_same_weights.reindex(index=idx, columns=syms).fillna(0.0).to_numpy("float64"))
    rate = (cash_rate.reindex(idx).ffill().fillna(0.0).to_numpy("float64")
            if cash_rate is not None else np.zeros(n))

    cash = float(initial_equity)
    shares = dict.fromkeys(syms, 0.0)
    equity, expo = np.empty(n), np.empty(n)
    orders, fees = 0, 0.0

    def trade(j: int, px: dict, w_row: np.ndarray, side_rule: dict):
        nonlocal cash, orders, fees
        mtm = cash + sum(shares[s] * px[s][j] for s in syms)
        for k, s in enumerate(syms):
            p = px[s][j]
            if p <= 0.0:
                continue
            desired = np.trunc(mtm * w_row[k] / p / lot_size) * lot_size
            delta = desired - shares[s]
            if (delta > 0 and side_rule["buy"]) or (delta < 0 and side_rule["sell"]):
                fee = costs.order_cost(p, delta)
                cash -= delta * p + fee
                fees += fee
                orders += 1
                shares[s] = desired

    for i in range(n):
        cash *= 1.0 + rate[i]
        if i > 0:
            trade(i, op, W[i - 1], {"buy": buy_at == "open_next", "sell": sell_at == "open_next"})
            trade(i, cl, W[i - 1], {"buy": buy_at == "close_next", "sell": sell_at == "close_next"})
        trade(i, cl, Wc[i], {"buy": buy_at == "close_same", "sell": sell_at == "close_same"})
        pos = sum(shares[s] * cl[s][i] for s in syms)
        equity[i] = cash + pos
        expo[i] = pos / equity[i] if equity[i] > 0 else 0.0

    eq = pd.Series(equity, index=idx, name="equity")
    stats = summarize(eq, pd.DataFrame({"fee": [fees]}), initial_equity)
    stats.update({"n_ordens": orders, "exposure_media": round(float(expo.mean()), 3)})
    return ExecResult(eq, stats)
