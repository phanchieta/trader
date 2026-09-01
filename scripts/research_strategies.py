"""Testa OOS (walk-forward na cesta) as estratégias do research do blog MQL5,
mais um ensemble ponderado por qualidade. Comparação contra o campeão atual.

Pré-requisito: python scripts/pull_basket.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from trader.backtest.split import WalkForward
from trader.backtest.validation import walk_forward_portfolio
from trader.costs.b3 import B3Costs
from trader.data.store import load
from trader.strategy import (Donchian, Keltner, SmaCrossover, SuperTrend, Vortex)

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
DATA = Path(cfg["data"]["dir"])
EQ0, LOT = cfg["backtest"]["initial_equity"], cfg["backtest"]["lot_size"]
SYMS = cfg["basket"]["symbols"]
COSTS = B3Costs.from_config(cfg)
TRAIL = cfg["risk_overlay"]["trailing_stop_pct"]

frames = {}
for s in SYMS:
    p = DATA / f"{s.lower()}_d1.parquet"
    if not p.exists():
        sys.exit(f"falta {p} — rode: python scripts/pull_basket.py")
    frames[s] = load(p)

start = max(f.index[0] for f in frames.values())
end = min(f.index[-1] for f in frames.values())
print(f"cesta: {', '.join(SYMS)} | janela {start.date()}..{end.date()}\n")

WF = WalkForward(n_folds=8, test_frac=0.60)


def make_st(period=10, multiplier=3.0, allow_short=False):
    return SuperTrend(period=period, multiplier=multiplier, allow_short=allow_short)


def make_don(entry_lookback=20, exit_lookback=10, allow_short=False):
    return Donchian(entry_lookback, exit_lookback, allow_short)


def make_kelt(ema_period=20, atr_period=20, mult=2.0, allow_short=False):
    return Keltner(ema_period, atr_period, mult, allow_short)


def make_vortex(period=14, allow_short=False):
    return Vortex(period=period, allow_short=allow_short)


ST_GRID = {"period": [7, 10, 14], "multiplier": [2.0, 3.0, 4.0], "allow_short": [False]}
DON_GRID = {"entry_lookback": [20, 40, 55], "exit_lookback": [10, 20], "allow_short": [False]}
KELT_GRID = {"ema_period": [20, 40], "atr_period": [20], "mult": [1.5, 2.0, 3.0],
             "allow_short": [False]}
VORTEX_GRID = {"period": [14, 21, 35], "allow_short": [False]}

YZ = {"target_ann_vol": cfg["risk_overlay"]["target_ann_vol"],
      "lookback": cfg["risk_overlay"]["vol_lookback"],
      "max_leverage": cfg["risk_overlay"]["max_leverage"], "estimator": "yang_zhang"}
YZ_ADX = {**YZ, "adx": {"period": 14, "adx_floor": 15.0, "adx_cap": 40.0}}

CONFIGS = [
    ("supertrend + YZ (campeão)",   make_st,     ST_GRID,     YZ),
    ("supertrend + YZ + ADX-scale", make_st,     ST_GRID,     YZ_ADX),
    ("donchian + YZ",               make_don,    DON_GRID,    YZ),
    ("keltner",                     make_kelt,   KELT_GRID,   None),
    ("keltner + YZ vol-target",     make_kelt,   KELT_GRID,   YZ),
    ("vortex",                      make_vortex, VORTEX_GRID, None),
    ("vortex + YZ vol-target",      make_vortex, VORTEX_GRID, YZ),
]

print(f"{'config':30} {'Sharpe':>7} {'retorno':>9} {'maxDD':>8} {'custo/gr':>9} "
      f"{'trades':>7} {'ΔSharpe':>8}")
print("-" * 84)
for label, make, grid, overlay in CONFIGS:
    rep = walk_forward_portfolio(make, grid, frames, COSTS, wf=WF,
                                 overlay_cfg=overlay, min_trades=20,
                                 initial_equity=EQ0, lot_size=LOT)
    o = rep.oos_stats
    print(f"{label:30} {str(o.get('sharpe')):>7} {str(o.get('retorno_total')):>9} "
          f"{str(o.get('max_drawdown')):>8} {str(o.get('custo_sobre_lucro_bruto')):>9} "
          f"{str(o.get('n_trades')):>7} {str(rep.degradacao().get('sharpe')):>8}")
