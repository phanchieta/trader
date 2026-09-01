"""Momentum cross-sectional: rankeia as ações do universo e fica comprado nas
melhores (e opcionalmente vendido nas piores). É o fator que comprovadamente
funciona em ações — e diferente por natureza dos sinais trend por nome.

A cada `rebalance` dias: score = retorno de (lookback → skip) dias atrás.
Long equal-weight nos `n_long` maiores; short nos `n_short` menores.
Universo com histórias diferentes: usa a UNIÃO das datas; um nome só entra no
ranking quando já tem `lookback` barras.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from ..costs.b3 import B3Costs
from .metrics import summarize
from .split import WalkForward

_DIAS_ANO = 252.0


def _union_index(frames: dict) -> pd.DatetimeIndex:
    idx = None
    for f in frames.values():
        idx = f.index if idx is None else idx.union(f.index)
    return idx.sort_values()


def _vol_target(frames: dict, target_ann_vol: float, lookback: int,
                max_leverage: float) -> pd.Series:
    """vol-target close-a-close no universo (média dos retornos disponíveis por dia)."""
    idx = _union_index(frames)
    rets = pd.DataFrame({s: frames[s]["close"].reindex(idx).pct_change() for s in frames})
    realized = rets.mean(axis=1).rolling(lookback).std() * (_DIAS_ANO ** 0.5)
    mult = (target_ann_vol / realized).clip(upper=max_leverage)
    return mult.shift(1).reindex(idx).fillna(0.0)


@dataclass
class CrossSectionalResult:
    equity: pd.Series
    stats: dict


def run_cross_sectional(frames: dict, costs: B3Costs, *, lookback: int = 120,
                        skip: int = 20, n_long: int = 3, n_short: int = 0,
                        rebalance: int = 5, borrow_annual: float = 0.03,
                        overlay_mult: pd.Series | None = None,
                        initial_equity: float = 100_000.0, lot_size: int = 100
                        ) -> CrossSectionalResult:
    syms = list(frames)
    idx = _union_index(frames)
    n, k = len(idx), len(syms)

    close_r = np.column_stack([frames[s]["close"].reindex(idx).to_numpy() for s in syms])
    open_r = np.column_stack([frames[s]["open"].reindex(idx).to_numpy() for s in syms])
    close = np.nan_to_num(close_r)
    open_ = np.nan_to_num(open_r)

    score = np.full((n, k), np.nan)
    if lookback < n:
        with np.errstate(invalid="ignore", divide="ignore"):
            score[lookback:] = close_r[lookback - skip: n - skip] / close_r[:n - lookback] - 1.0

    mult = (overlay_mult.reindex(idx).fillna(0.0).to_numpy()
            if overlay_mult is not None else np.ones(n))

    cash = float(initial_equity)
    shares = np.zeros(k)
    weights = np.zeros(k)
    equity = np.empty(n)
    fees = 0.0
    trades = 0

    for i in range(n):
        if i % rebalance == 0 and i > 0:
            row = score[i - 1]
            ok = np.isfinite(row) & (open_[i] > 0.0)
            if ok.sum() >= n_long + max(n_short, 0):
                ranked = np.where(ok, row, -np.inf)
                order = np.argsort(ranked)[::-1]
                w = np.zeros(k)
                w[order[:n_long]] = 1.0 / n_long
                if n_short:
                    w[order[-n_short:]] = -1.0 / n_short
                weights = w

        m = mult[i]
        mtm = cash + float((shares * open_[i]).sum())
        for j in range(k):
            po = open_[i, j]
            tgt = 0.0
            if weights[j] != 0.0 and po > 0.0 and m > 0.0:
                tgt = np.trunc(mtm * weights[j] * m / po / lot_size) * lot_size
            delta = tgt - shares[j]
            if delta != 0.0:
                fee = costs.order_cost(po, delta)
                cash -= delta * po + fee
                fees += fee
                shares[j] = tgt
                trades += 1

        short_notional = float(np.where(shares < 0.0, -shares * close[i], 0.0).sum())
        cash -= short_notional * borrow_annual / _DIAS_ANO
        equity[i] = cash + float((shares * close[i]).sum())

    eq = pd.Series(equity, index=idx, name="equity")
    fee_frame = (pd.DataFrame({"fee": [fees]}) if fees
                 else pd.DataFrame(columns=["price", "shares", "fee"]))
    stats = summarize(eq, fee_frame, initial_equity)
    stats["n_trades"] = int(trades)
    return CrossSectionalResult(eq, stats)


def walk_forward_cs(frames: dict, costs: B3Costs, param_grid: dict, *,
                    wf: WalkForward | None = None, overlay: dict | None = None,
                    initial_equity: float = 100_000.0, lot_size: int = 100) -> dict:
    wf = wf or WalkForward()
    idx = _union_index(frames)
    keys = list(param_grid)
    combos = [dict(zip(keys, c)) for c in product(*(param_grid[k] for k in keys))]

    curves, folds, is_sharpes = [], [], []
    for tr_sl, te_sl in wf.splits(len(idx)):
        tr = {s: f.loc[idx[tr_sl.start]:idx[tr_sl.stop - 1]] for s, f in frames.items()}
        te = {s: f.loc[idx[te_sl.start]:idx[te_sl.stop - 1]] for s, f in frames.items()}
        m_tr = _vol_target(tr, **overlay) if overlay else None
        m_te = _vol_target(te, **overlay) if overlay else None

        best, best_sh = None, -np.inf
        for params in combos:
            sh = run_cross_sectional(tr, costs, overlay_mult=m_tr,
                                     initial_equity=initial_equity, lot_size=lot_size,
                                     **params).stats.get("sharpe")
            if sh is not None and sh > best_sh:
                best, best_sh = params, sh
        best = best or combos[0]
        res = run_cross_sectional(te, costs, overlay_mult=m_te,
                                  initial_equity=initial_equity, lot_size=lot_size, **best)
        curves.append(res.equity)
        is_sharpes.append(best_sh)
        folds.append({"params": best, "oos": res.stats})

    level, parts = initial_equity, []
    for c in curves:
        parts.append(c * (level / float(c.iloc[0])))
        level = float(parts[-1].iloc[-1])
    stitched = pd.concat(parts)
    oos = summarize(stitched, pd.DataFrame(columns=["price", "shares", "fee"]),
                    initial_equity)
    oos["n_trades"] = int(sum(f["oos"]["n_trades"] for f in folds))
    return {"is_sharpe_mean": round(float(np.nanmean(is_sharpes)), 3),
            "oos_stats": oos, "oos_equity": stitched, "folds": folds}
