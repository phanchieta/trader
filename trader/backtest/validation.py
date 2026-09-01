"""Avaliação out-of-sample.

REGRA DO PROJETO: nenhuma estratégia é aprovada por desempenho in-sample.
Os parâmetros são escolhidos SÓ no período de treino; o número que vale é o
do período de teste, que a estratégia nunca viu. Degradação grande de IS para
OOS = overfitting.

  holdout(...)       -> um corte treino/teste (ex.: 70/30)
  walk_forward(...)  -> vários folds; re-escolhe parâmetros em cada treino e
                        mede no teste seguinte; costura a curva OOS.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Callable

import numpy as np
import pandas as pd

from ..costs.b3 import B3Costs
from ..risk.overlay import adx_scale_multiplier, common_index, vol_target_multiplier
from .engine import run
from .metrics import summarize
from .portfolio import run_portfolio
from .split import WalkForward, train_test_split

Factory = Callable[..., object]
_EMPTY_TRADES = pd.DataFrame(columns=["price", "shares", "fee"])


def _grid(param_grid: dict):
    keys = list(param_grid)
    for combo in product(*(param_grid[k] for k in keys)):
        yield dict(zip(keys, combo))


def grid_search(make: Factory, param_grid: dict, train: pd.DataFrame, costs: B3Costs,
                *, min_trades: int = 10, initial_equity: float = 100_000.0,
                lot_size: int = 100):
    """Melhor combinação por Sharpe no TREINO. Exige `min_trades`; se nenhuma
    combinação atingir, relaxa o filtro para não voltar vazio."""
    def evaluate(require_trades: bool):
        best, best_score, best_stats = None, -np.inf, None
        for params in _grid(param_grid):
            stats = run(train, make(**params), costs,
                        initial_equity=initial_equity, lot_size=lot_size).stats
            if require_trades and stats["n_trades"] < min_trades:
                continue
            s = stats.get("sharpe")
            score = float(s) if s is not None else -np.inf
            if score > best_score:
                best, best_score, best_stats = params, score, stats
        return best, best_stats

    best, stats = evaluate(require_trades=True)
    if best is None:
        best, stats = evaluate(require_trades=False)
    return best, stats


def _stitch(curves: list[pd.Series], initial: float) -> pd.Series:
    """Concatena curvas de equity OOS de folds contíguos, cada uma reancorada
    no nível em que a anterior terminou."""
    parts, level = [], initial
    for c in curves:
        if len(c) == 0:
            continue
        scaled = c * (level / float(c.iloc[0]))
        parts.append(scaled)
        level = float(scaled.iloc[-1])
    return pd.concat(parts) if parts else pd.Series(dtype="float64")


@dataclass
class OOSReport:
    best_params: object            # dict (holdout) ou list[dict] (walk-forward)
    is_stats: dict
    oos_stats: dict
    oos_equity: pd.Series
    folds: list = field(default_factory=list)

    def degradacao(self) -> dict:
        """OOS - IS por métrica. Negativo = piorou no teste."""
        out = {}
        for k in ("sharpe", "cagr", "retorno_total", "max_drawdown"):
            a, b = self.is_stats.get(k), self.oos_stats.get(k)
            out[k] = None if (a is None or b is None) else round(b - a, 4)
        return out


def holdout(make: Factory, param_grid: dict, df: pd.DataFrame, costs: B3Costs, *,
            test_frac: float = 0.3, embargo: int = 0, min_trades: int = 10,
            initial_equity: float = 100_000.0, lot_size: int = 100) -> OOSReport:
    train, test = train_test_split(df, test_frac=test_frac, embargo=embargo)
    best, is_stats = grid_search(make, param_grid, train, costs, min_trades=min_trades,
                                 initial_equity=initial_equity, lot_size=lot_size)
    oos = run(test, make(**best), costs,
              initial_equity=initial_equity, lot_size=lot_size)
    return OOSReport(best, is_stats, oos.stats, oos.equity)


def walk_forward(make: Factory, param_grid: dict, df: pd.DataFrame, costs: B3Costs, *,
                 wf: WalkForward | None = None, min_trades: int = 10,
                 initial_equity: float = 100_000.0, lot_size: int = 100) -> OOSReport:
    wf = wf or WalkForward()
    folds, curves, chosen = [], [], []
    for tr_sl, te_sl in wf.splits(len(df)):
        best, is_stats = grid_search(make, param_grid, df.iloc[tr_sl], costs,
                                     min_trades=min_trades,
                                     initial_equity=initial_equity, lot_size=lot_size)
        oos = run(df.iloc[te_sl], make(**best), costs,
                  initial_equity=initial_equity, lot_size=lot_size)
        folds.append({"params": best, "is": is_stats, "oos": oos.stats})
        curves.append(oos.equity)
        chosen.append(best)

    stitched = _stitch(curves, initial_equity)
    oos_stats = summarize(stitched, _EMPTY_TRADES, initial_equity)
    fees = float(sum((f["oos"]["custo_total_R$"] or 0.0) for f in folds))
    gross = (float(stitched.iloc[-1]) - initial_equity + fees) if len(stitched) else 0.0
    oos_stats["n_trades"] = int(sum(f["oos"]["n_trades"] for f in folds))
    oos_stats["custo_total_R$"] = round(fees, 2)
    oos_stats["custo_sobre_lucro_bruto"] = round(fees / gross, 3) if gross > 0 else None

    is_agg = {}
    for k in ("sharpe", "cagr", "retorno_total", "max_drawdown", "custo_sobre_lucro_bruto"):
        vals = [f["is"].get(k) for f in folds if f["is"].get(k) is not None]
        is_agg[k] = round(float(np.mean(vals)), 4) if vals else None
    is_agg["n_trades"] = int(round(float(np.mean([f["is"]["n_trades"] for f in folds]))))
    return OOSReport(chosen, is_agg, oos_stats, stitched, folds)


# ---------------------------------------------------------------------------
# Versões multi-ativo (cesta). Um único conjunto de parâmetros para todas as
# ações — escolher parâmetros por ação seria overfitting.
# ---------------------------------------------------------------------------

def _slice_frames(frames: dict, start, end) -> dict:
    return {s: f.loc[start:end] for s, f in frames.items()}


def _mult(frames: dict, overlay_cfg: dict | None):
    """overlay_cfg é o kwargs de vol_target_multiplier; uma chave opcional 'adx'
    (dict) multiplica pelo escalador de ADX."""
    if not overlay_cfg:
        return None
    cfg = dict(overlay_cfg)
    adx_cfg = cfg.pop("adx", None)
    m = vol_target_multiplier(frames, **cfg)
    if adx_cfg:
        m = m * adx_scale_multiplier(frames, **adx_cfg)
    return m


def grid_search_portfolio(make: Factory, param_grid: dict, train: dict, costs: B3Costs,
                          *, overlay_cfg: dict | None = None, trailing_stop: float | None = None,
                          min_trades: int = 10, initial_equity: float = 100_000.0,
                          lot_size: int = 100):
    mult = _mult(train, overlay_cfg)

    def evaluate(require_trades: bool):
        best, best_score, best_stats = None, -np.inf, None
        for params in _grid(param_grid):
            stats = run_portfolio(train, make(**params), costs, overlay_mult=mult,
                                  trailing_stop=trailing_stop,
                                  initial_equity=initial_equity, lot_size=lot_size).stats
            if require_trades and stats["n_trades"] < min_trades:
                continue
            s = stats.get("sharpe")
            score = float(s) if s is not None else -np.inf
            if score > best_score:
                best, best_score, best_stats = params, score, stats
        return best, best_stats

    best, stats = evaluate(True)
    if best is None:
        best, stats = evaluate(False)
    return best, stats


def holdout_portfolio(make: Factory, param_grid: dict, frames: dict, costs: B3Costs, *,
                      test_frac: float = 0.3, embargo: int = 0,
                      overlay_cfg: dict | None = None, trailing_stop: float | None = None,
                      min_trades: int = 10, initial_equity: float = 100_000.0,
                      lot_size: int = 100) -> OOSReport:
    idx = common_index(frames)
    cut = int(len(idx) * (1.0 - test_frac))
    train = _slice_frames(frames, idx[0], idx[max(1, cut - embargo) - 1])
    test = _slice_frames(frames, idx[cut], idx[-1])
    best, is_stats = grid_search_portfolio(make, param_grid, train, costs,
                                           overlay_cfg=overlay_cfg, trailing_stop=trailing_stop,
                                           min_trades=min_trades,
                                           initial_equity=initial_equity, lot_size=lot_size)
    res = run_portfolio(test, make(**best), costs, overlay_mult=_mult(test, overlay_cfg),
                        trailing_stop=trailing_stop,
                        initial_equity=initial_equity, lot_size=lot_size)
    return OOSReport(best, is_stats, res.stats, res.equity)


def walk_forward_portfolio(make: Factory, param_grid: dict, frames: dict, costs: B3Costs,
                           *, wf: WalkForward | None = None, overlay_cfg: dict | None = None,
                           trailing_stop: float | None = None, min_trades: int = 10,
                           initial_equity: float = 100_000.0, lot_size: int = 100) -> OOSReport:
    wf = wf or WalkForward()
    idx = common_index(frames)
    folds, curves, chosen = [], [], []
    for tr_sl, te_sl in wf.splits(len(idx)):
        train = _slice_frames(frames, idx[tr_sl.start], idx[tr_sl.stop - 1])
        test = _slice_frames(frames, idx[te_sl.start], idx[te_sl.stop - 1])
        best, is_stats = grid_search_portfolio(make, param_grid, train, costs,
                                               overlay_cfg=overlay_cfg, trailing_stop=trailing_stop,
                                               min_trades=min_trades,
                                               initial_equity=initial_equity, lot_size=lot_size)
        res = run_portfolio(test, make(**best), costs, overlay_mult=_mult(test, overlay_cfg),
                            trailing_stop=trailing_stop,
                            initial_equity=initial_equity, lot_size=lot_size)
        folds.append({"params": best, "is": is_stats, "oos": res.stats})
        curves.append(res.equity)
        chosen.append(best)

    stitched = _stitch(curves, initial_equity)
    oos_stats = summarize(stitched, _EMPTY_TRADES, initial_equity)
    fees = float(sum((f["oos"]["custo_total_R$"] or 0.0) for f in folds))
    gross = (float(stitched.iloc[-1]) - initial_equity + fees) if len(stitched) else 0.0
    oos_stats["n_trades"] = int(sum(f["oos"]["n_trades"] for f in folds))
    oos_stats["custo_total_R$"] = round(fees, 2)
    oos_stats["custo_sobre_lucro_bruto"] = round(fees / gross, 3) if gross > 0 else None

    is_agg = {}
    for k in ("sharpe", "cagr", "retorno_total", "max_drawdown", "custo_sobre_lucro_bruto"):
        vals = [f["is"].get(k) for f in folds if f["is"].get(k) is not None]
        is_agg[k] = round(float(np.mean(vals)), 4) if vals else None
    is_agg["n_trades"] = int(round(float(np.mean([f["is"]["n_trades"] for f in folds]))))
    return OOSReport(chosen, is_agg, oos_stats, stitched, folds)
