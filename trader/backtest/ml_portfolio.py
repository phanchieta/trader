"""Backtest multi-ativo a partir de uma matriz de alvos já pronta (índice =
datas, colunas = ações, valores em [-1, 1]). Serve para ML, ensembles, ou
qualquer coisa que produza um alvo por ação/dia.

Mesma mecânica do backtest.portfolio: alvo de t executa na abertura de t+1;
capital dividido igualmente; overlay de exposição e trailing-stop opcionais.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..costs.b3 import B3Costs
from .metrics import summarize
from .portfolio import PortfolioResult


def run_target_portfolio(frames: dict, target: pd.DataFrame, costs: B3Costs, *,
                         initial_equity: float = 100_000.0, lot_size: int = 100,
                         overlay_mult: pd.Series | None = None,
                         trailing_stop: float | None = None) -> PortfolioResult:
    syms = list(frames)
    idx = target.index
    n = len(syms)

    open_ = {s: frames[s]["open"].reindex(idx).to_numpy(dtype="float64") for s in syms}
    close = {s: frames[s]["close"].reindex(idx).to_numpy(dtype="float64") for s in syms}
    tgt = {s: target[s].reindex(idx).fillna(0.0).shift(1).fillna(0.0).to_numpy()
           for s in syms}
    mult = (overlay_mult.reindex(idx).fillna(0.0).to_numpy(dtype="float64")
            if overlay_mult is not None else np.ones(len(idx)))

    cash = float(initial_equity)
    shares = {s: 0.0 for s in syms}
    entry = {s: 0.0 for s in syms}
    peak = {s: 0.0 for s in syms}
    blocked = {s: False for s in syms}
    equity = np.empty(len(idx))
    trades = {s: 0 for s in syms}
    fees_total = 0.0

    for i in range(len(idx)):
        mtm = cash + sum(shares[s] * open_[s][i] for s in syms)
        per_name = mtm / n
        for s in syms:
            po = open_[s][i]
            raw_t = tgt[s][i]
            if raw_t == 0.0:
                blocked[s] = False
            eff_t = 0.0 if blocked[s] else raw_t

            desired = 0.0
            if eff_t != 0.0 and po > 0.0 and mult[i] > 0.0:
                desired = np.trunc(per_name * eff_t * mult[i] / po / lot_size) * lot_size
            delta = desired - shares[s]
            if delta != 0.0:
                fee = costs.order_cost(po, delta)
                cash -= delta * po + fee
                fees_total += fee
                if shares[s] == 0.0 and desired != 0.0:
                    entry[s] = peak[s] = po
                shares[s] = desired
                trades[s] += 1
            if shares[s] == 0.0:
                entry[s] = peak[s] = 0.0

            pc = close[s][i]
            if trailing_stop and shares[s] > 0.0:
                peak[s] = max(peak[s], pc)
                if peak[s] > entry[s] and pc <= peak[s] * (1.0 - trailing_stop):
                    blocked[s] = True

        equity[i] = cash + sum(shares[s] * close[s][i] for s in syms)

    eq = pd.Series(equity, index=idx, name="equity")
    fee_frame = (pd.DataFrame({"fee": [fees_total]}) if fees_total
                 else pd.DataFrame(columns=["price", "shares", "fee"]))
    stats = summarize(eq, fee_frame, initial_equity)
    stats["n_trades"] = int(sum(trades.values()))
    stats["trades_por_ativo"] = dict(trades)
    return PortfolioResult(eq, dict(trades), stats)
