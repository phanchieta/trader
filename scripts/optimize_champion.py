"""Otimização do SIZING do campeão (SuperTrend 10/3 fixo). Walk-forward, CDI no
caixa, objetivo = Sharpe sobre o CDI. Cada combo é aplicado fixo em todos os
folds (sem grid-search por fold) e medido no OOS costurado — número honesto.

Knobs: vol-alvo, janela da vol (Yang-Zhang), alavancagem máx, peso por ação.
Pré-requisitos: pull_basket.py ; pull_cdi.py
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from trader.backtest.metrics import summarize, vs_benchmark
from trader.backtest.portfolio import run_portfolio
from trader.backtest.split import WalkForward
from trader.costs.b3 import B3Costs
from trader.data.store import load
from trader.risk.overlay import inv_vol_weights, vol_target_multiplier
from trader.strategy import SuperTrend

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
DATA = Path(cfg["data"]["dir"])
EQ0, LOT = cfg["backtest"]["initial_equity"], cfg["backtest"]["lot_size"]
COSTS = B3Costs.from_config(cfg)

basket = {s: load(DATA / f"{s.lower()}_d1.parquet") for s in cfg["basket"]["symbols"]}
CDI = pd.read_parquet(DATA / "cdi_d1.parquet")["rate"]
idx = None
for f in basket.values():
    idx = f.index if idx is None else idx.intersection(f.index)
idx = idx.sort_values()
WF = WalkForward(n_folds=8, test_frac=0.60)

GRID = dict(target_vol=[0.12, 0.18, 0.25, 0.35],
            vol_lb=[15, 25, 40],
            max_lev=[1.0, 1.5],
            weights=["equal", "inv_vol"])


def evaluate(target_vol, vol_lb, max_lev, weights):
    lvl, parts = EQ0, []
    for _, te_sl in WF.splits(len(idx)):
        te = {s: f.loc[idx[te_sl.start]:idx[te_sl.stop - 1]] for s, f in basket.items()}
        mult = vol_target_multiplier(te, target_ann_vol=target_vol, lookback=vol_lb,
                                     max_leverage=max_lev, estimator="yang_zhang")
        nw = inv_vol_weights(te, lookback=vol_lb) if weights == "inv_vol" else None
        eq = run_portfolio(te, SuperTrend(10, 3.0), COSTS, overlay_mult=mult,
                           name_weights=nw, cash_rate=CDI,
                           initial_equity=EQ0, lot_size=LOT).equity
        parts.append(eq * (lvl / float(eq.iloc[0])))
        lvl = float(parts[-1].iloc[-1])
    stitched = pd.concat(parts)
    b = vs_benchmark(stitched, CDI)
    s = summarize(stitched, pd.DataFrame(columns=["price", "shares", "fee"]), EQ0)
    return {**b, "maxDD": s["max_drawdown"]}


rows = []
for tv, lb, lev, w in product(*GRID.values()):
    r = evaluate(tv, lb, lev, w)
    rows.append((tv, lb, lev, w, r))

rows.sort(key=lambda x: -(x[4]["sharpe_sobre_cdi"] or -9))
print(f"CDI no período: {rows[0][4]['cagr_cdi']:.1%} a.a.\n")
print(f"{'vol':>5} {'lb':>4} {'lev':>4} {'peso':>8} {'CAGR':>7} {'alpha':>7} "
      f"{'Sh/CDI':>7} {'maxDD':>8}")
print("-" * 60)
for tv, lb, lev, w, r in rows[:16]:
    print(f"{tv:>5} {lb:>4} {lev:>4} {w:>8} {r['cagr_estrategia']:>7.1%} "
          f"{r['alpha_aa']:>+7.1%} {str(r['sharpe_sobre_cdi']):>7} {r['maxDD']:>8.1%}")

cur = next(r for tv, lb, lev, w, r in rows if (tv, lb, lev, w) == (0.18, 25, 1.0, "equal"))
print(f"\n(config próxima da atual — vol 0,15/lb20 não está no grid; ref vol0,18/lb25: "
      f"alpha {cur['alpha_aa']:+.1%}, Sh/CDI {cur['sharpe_sobre_cdi']}, maxDD {cur['maxDD']:.1%})")
