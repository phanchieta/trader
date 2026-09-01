"""Backtest OOS de um book de fatores: top n_frac (quintil/decil), long-only,
equal-weight, rebalance periódico, vol-target + CDI no caixa.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.ml_portfolio import run_target_portfolio
from ..costs.b3 import B3Costs


def build_target(score: pd.DataFrame, win, *, n_frac: float = 0.20,
                 rebal: int = 21) -> pd.DataFrame:
    """Alvo (pesos) equal-weight no topo `n_frac` do score, congelado entre
    rebalances de `rebal` dias."""
    sc = score.reindex(win)
    keep = pd.Series(sc.index.isin(sc.index[::rebal]), index=sc.index)
    sc = sc.where(keep, np.nan).ffill()
    thr = sc.quantile(1.0 - n_frac, axis=1)
    picks = sc.ge(thr, axis=0) & sc.notna()
    npick = picks.sum(axis=1).replace(0, np.nan)
    return picks.div(npick, axis=0).fillna(0.0)


def _stitch(parts, eq0):
    lvl, out = eq0, []
    for c in parts:
        out.append(c * (lvl / float(c.iloc[0])))
        lvl = float(out[-1].iloc[-1])
    return pd.concat(out)


def factor_oos(frames: dict, score: pd.DataFrame, *, wf, cdi: pd.Series,
               ibov_mult: pd.Series, n_frac: float = 0.20, rebal: int = 21,
               overlay_mode: str = "ibov", target_vol: float = 0.15,
               vol_lookback: int = 40, max_leverage: float = 2.0,
               eq0: float = 100_000.0, lot: int = 100,
               costs: B3Costs | None = None) -> pd.Series:
    """overlay_mode: 'ibov' (vol-target pela vol do Ibov), 'book' (pela vol dos
    retornos do próprio book, 2 passadas), 'none'."""
    costs = costs or B3Costs()
    idx = score.index
    parts = []
    for _, te_sl in wf.splits(len(idx)):
        win = idx[te_sl.start:te_sl.stop]
        tgt = build_target(score, win, n_frac=n_frac, rebal=rebal)

        if overlay_mode == "none":
            mult = None
        elif overlay_mode == "ibov":
            mult = ibov_mult.reindex(win)
        else:  # book
            base = run_target_portfolio(frames, tgt, costs, cash_rate=cdi,
                                        target_is_weight=True, initial_equity=eq0,
                                        lot_size=lot).equity
            r = base.pct_change().fillna(0.0)
            realized = r.rolling(vol_lookback).std() * np.sqrt(252.0)
            mult = (target_vol / realized.where(realized > 0)).clip(
                upper=max_leverage).shift(1).fillna(0.0)

        eq = run_target_portfolio(frames, tgt, costs, overlay_mult=mult,
                                  cash_rate=cdi, target_is_weight=True,
                                  initial_equity=eq0, lot_size=lot).equity
        parts.append(eq)
    return _stitch(parts, eq0)


def beta_hedge(book_equity: pd.Series, ibov_close: pd.Series, *, window: int = 60,
               cost_bps: float = 1.0) -> pd.Series:
    """Curva do book com hedge de índice: short beta·Ibov (beta rolante).
    Aproximação pós-hoc — assume o hedge executável a custo baixo (WIN)."""
    r = book_equity.pct_change()
    rm = ibov_close.reindex(r.index).pct_change()
    beta = (r.rolling(window).cov(rm) / rm.rolling(window).var()).shift(1).clip(0.0, 2.0)
    hedged = (r - beta * rm - (beta.diff().abs().fillna(0.0) * cost_bps / 1e4)).fillna(0.0)
    return book_equity.iloc[0] * (1.0 + hedged).cumprod()
