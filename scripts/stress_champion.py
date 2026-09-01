"""Stress-test do campeão: SuperTrend + vol-target Yang-Zhang.

Roda walk-forward com mais folds e com OOS cobrindo mais história (empurra o
ΔSharpe favorável). Mostra Sharpe/retorno por fold e retorno por ano.

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
from trader.strategy import SuperTrend

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
DATA = Path(cfg["data"]["dir"])
EQ0, LOT = cfg["backtest"]["initial_equity"], cfg["backtest"]["lot_size"]
SYMS = cfg["basket"]["symbols"]
COSTS = B3Costs.from_config(cfg)
YZ = {"target_ann_vol": cfg["risk_overlay"]["target_ann_vol"],
      "lookback": cfg["risk_overlay"]["vol_lookback"],
      "max_leverage": cfg["risk_overlay"]["max_leverage"], "estimator": "yang_zhang"}

frames = {}
for s in SYMS:
    p = DATA / f"{s.lower()}_d1.parquet"
    if not p.exists():
        sys.exit(f"falta {p} — rode: python scripts/pull_basket.py")
    frames[s] = load(p)


def make_st(period=10, multiplier=3.0, allow_short=False):
    return SuperTrend(period=period, multiplier=multiplier, allow_short=allow_short)


GRID = {"period": [7, 10, 14], "multiplier": [2.0, 3.0, 4.0], "allow_short": [False]}

for tag, wf in [("padrão  (8 folds, OOS=60%)", WalkForward(8, 0.60)),
                ("largo   (12 folds, OOS=75%)", WalkForward(12, 0.75)),
                ("máximo  (12 folds, OOS=85%)", WalkForward(12, 0.85))]:
    rep = walk_forward_portfolio(make_st, GRID, frames, COSTS, wf=wf, overlay_cfg=YZ,
                                 min_trades=20, initial_equity=EQ0, lot_size=LOT)
    o = rep.oos_stats
    print(f"\n== {tag} ==")
    print(f"  IS Sharpe médio : {rep.is_stats.get('sharpe')}   ->  ΔSharpe {rep.degradacao().get('sharpe')}")
    print(f"  OOS Sharpe .....: {o.get('sharpe')} | ret {o.get('retorno_total')} | "
          f"maxDD {o.get('max_drawdown')} | custo/gross {o.get('custo_sobre_lucro_bruto')}")
    print(f"  Sharpe por fold : {[f['oos'].get('sharpe') for f in rep.folds]}")
    eq = rep.oos_equity
    per_year = {int(y): round(float(s.iloc[-1] / s.iloc[0] - 1), 3)
                for y, s in eq.groupby(eq.index.year)}
    print(f"  retorno por ano : {per_year}")
