"""Momentum cross-sectional no universo grande (~24 ações). Walk-forward OOS,
long-only e long-short, com e sem vol-target.

Pré-requisito: python scripts/pull_universe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from trader.backtest.cross_sectional import walk_forward_cs
from trader.backtest.split import WalkForward
from trader.costs.b3 import B3Costs
from trader.data.store import load

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
DATA = Path(cfg["data"]["dir"])
EQ0, LOT = cfg["backtest"]["initial_equity"], cfg["backtest"]["lot_size"]
SYMS = cfg["universe"]["symbols"]
COSTS = B3Costs.from_config(cfg)
VOL = {"target_ann_vol": cfg["risk_overlay"]["target_ann_vol"],
       "lookback": cfg["risk_overlay"]["vol_lookback"],
       "max_leverage": cfg["risk_overlay"]["max_leverage"]}

frames = {}
for s in SYMS:
    p = DATA / f"{s.lower()}_d1.parquet"
    if p.exists():
        frames[s] = load(p)
if len(frames) < 8:
    sys.exit(f"só {len(frames)} ações — rode scripts/pull_universe.py")

starts = sorted(f.index[0].date() for f in frames.values())
print(f"{len(frames)} ações | início mais antigo {starts[0]} | mais novo {starts[-1]}\n")

WF = WalkForward(n_folds=8, test_frac=0.60)
LONG_GRID = {"lookback": [120, 180, 250], "skip": [5, 20],
             "n_long": [3, 5, 8], "n_short": [0], "rebalance": [10, 20]}
LS_GRID = {"lookback": [120, 180, 250], "skip": [20],
           "n_long": [3, 5], "n_short": [3, 5], "rebalance": [10, 20]}

CONFIGS = [
    ("long-only",              LONG_GRID, None),
    ("long-only + vol-target", LONG_GRID, VOL),
    ("long-short",             LS_GRID,   None),
    ("long-short + vol-target", LS_GRID,  VOL),
]

for label, grid, overlay in CONFIGS:
    rep = walk_forward_cs(frames, COSTS, grid, wf=WF, overlay=overlay,
                          initial_equity=EQ0, lot_size=LOT)
    o = rep["oos_stats"]
    print(f"== {label} ==")
    print(f"  IS Sharpe médio {rep['is_sharpe_mean']} | OOS Sharpe {o.get('sharpe')} | "
          f"ret {o.get('retorno_total')} | maxDD {o.get('max_drawdown')} | "
          f"custo/gross {o.get('custo_sobre_lucro_bruto')} | trades {o.get('n_trades')}")
    print(f"  params por fold: {[f['params'] for f in rep['folds']]}")
    eq = rep["oos_equity"]
    py = {int(y): round(float(s.iloc[-1] / s.iloc[0] - 1), 3)
          for y, s in eq.groupby(eq.index.year)}
    print(f"  retorno por ano: {py}\n")
