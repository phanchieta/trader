"""Backtest da cesta de 5 ações, com e sem o overlay de vol-target.

Horizonte swing, custos swing. Um único conjunto de parâmetros para as 5 ações
(por ação seria overfitting). Walk-forward 8 folds + holdout como cross-check.
Mostra o efeito do overlay no drawdown.

Pré-requisito: python scripts/pull_basket.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from trader.backtest.split import WalkForward
from trader.backtest.validation import holdout_portfolio, walk_forward_portfolio
from trader.costs.b3 import B3Costs
from trader.data.store import load
from trader.strategy import Momentum, SmaCrossover

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
DATA = Path(cfg["data"]["dir"])
EQ0, LOT = cfg["backtest"]["initial_equity"], cfg["backtest"]["lot_size"]
SYMS = cfg["basket"]["symbols"]
COSTS = B3Costs.from_config(cfg)
OVERLAY = {"target_ann_vol": cfg["risk_overlay"]["target_ann_vol"],
           "lookback": cfg["risk_overlay"]["vol_lookback"],
           "max_leverage": cfg["risk_overlay"]["max_leverage"]}
TRAIL = cfg["risk_overlay"]["trailing_stop_pct"]

frames = {}
for s in SYMS:
    p = DATA / f"{s.lower()}_d1.parquet"
    if not p.exists():
        sys.exit(f"falta {p} — rode: python scripts/pull_basket.py")
    frames[s] = load(p)

start = max(f.index[0] for f in frames.values())
end = min(f.index[-1] for f in frames.values())
print(f"cesta: {', '.join(SYMS)}")
print(f"janela comum: {start.date()} .. {end.date()}\n")

STRATS = {
    "sma_crossover": (SmaCrossover, {"fast": [10, 20, 50], "slow": [50, 100, 200],
                                     "allow_short": [False]}),
    "momentum": (Momentum, {"lookback": [40, 120, 200], "band": [0.0, 0.02],
                            "trend_filter": [0, 200], "allow_short": [False]}),
}
WF = WalkForward(n_folds=8, test_frac=0.60)
_ROWS = [("Sharpe", "sharpe"), ("retorno", "retorno_total"),
         ("max_drawdown", "max_drawdown"), ("custo/gross", "custo_sobre_lucro_bruto"),
         ("trades", "n_trades")]


def show(title, rep):
    print(f"  {title}")
    print(f"    {'métrica':14} {'IS':>12} {'OOS':>12}")
    for label, key in _ROWS:
        print(f"    {label:14} {str(rep.is_stats.get(key)):>12} {str(rep.oos_stats.get(key)):>12}")
    print(f"    ΔSharpe OOS-IS: {rep.degradacao().get('sharpe')}")


VARIANTS = [
    ("base (só sinal)", None, None),
    ("+ trailing-stop", None, TRAIL),
    ("+ vol-target", OVERLAY, None),
    ("+ trailing + vol-target", OVERLAY, TRAIL),
]

for name, (cls, grid) in STRATS.items():
    print(f"{'=' * 60}\n{name} — cesta de {len(SYMS)}\n{'=' * 60}")
    for tag, ocfg, ts in VARIANTS:
        rep = walk_forward_portfolio(cls, grid, frames, COSTS, wf=WF,
                                     overlay_cfg=ocfg, trailing_stop=ts, min_trades=20,
                                     initial_equity=EQ0, lot_size=LOT)
        show(f"walk-forward · {tag}", rep)
        eq = rep.oos_equity
        per_year = {int(y): round(float(s.iloc[-1] / s.iloc[0] - 1), 3)
                    for y, s in eq.groupby(eq.index.year)}
        print(f"    retorno por ano: {per_year}\n")
