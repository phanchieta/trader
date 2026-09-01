"""Validação out-of-sample das estratégias de regra.

Escolhe parâmetros SÓ no treino e mede em dados nunca vistos.
Compara in-sample (IS) x out-of-sample (OOS): degradação grande = overfitting.

Pré-requisito: python scripts/pull_daily.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from trader.backtest.split import WalkForward
from trader.backtest.validation import holdout, walk_forward
from trader.costs.b3 import B3Costs
from trader.data.store import load
from trader.strategy import MeanReversion, Momentum

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
df = load(Path(cfg["data"]["dir"]) / cfg["data"]["daily_file"])
costs = B3Costs.from_config(cfg)
eq0 = cfg["backtest"]["initial_equity"]
lot = cfg["backtest"]["lot_size"]

SPECS = {
    "mean_reversion": (MeanReversion, {
        "lookback": [10, 20, 40], "entry_z": [1.0, 1.5, 2.0],
        "exit_z": [0.0, 0.5], "allow_short": [False]}),
    "momentum": (Momentum, {
        "lookback": [40, 60, 120, 200], "band": [0.0, 0.02],
        "trend_filter": [0, 200], "allow_short": [False]}),
}

_ROWS = [("Sharpe", "sharpe"), ("CAGR", "cagr"), ("retorno_total", "retorno_total"),
         ("max_drawdown", "max_drawdown"), ("n_trades", "n_trades"),
         ("custo/lucro_bruto", "custo_sobre_lucro_bruto")]


def _fmt_params(p) -> str:
    if isinstance(p, list):
        from collections import Counter
        counts = Counter(tuple(sorted(d.items())) for d in p)
        return " | ".join(f"{dict(k)} (x{n})" for k, n in counts.items())
    return str(p)


def show(title: str, rep) -> None:
    print(f"\n== {title} ==")
    print(f"  parâmetros (escolhidos no treino): {_fmt_params(rep.best_params)}")
    print(f"  {'métrica':18} {'IS (treino)':>13} {'OOS (novo)':>13}")
    for label, key in _ROWS:
        print(f"  {label:18} {str(rep.is_stats.get(key)):>13} {str(rep.oos_stats.get(key)):>13}")
    print(f"  degradação OOS-IS: {rep.degradacao()}")


for name, (cls, grid) in SPECS.items():
    print(f"\n{'#' * 60}\n# {name}\n{'#' * 60}")
    show("holdout 70/30",
         holdout(cls, grid, df, costs, test_frac=0.30,
                 initial_equity=eq0, lot_size=lot))
    show("walk-forward 4 folds (OOS = últimos 40%)",
         walk_forward(cls, grid, df, costs,
                      wf=WalkForward(n_folds=4, test_frac=0.40),
                      initial_equity=eq0, lot_size=lot))

print("\nopening_range_breakout: pendente — precisa de histórico intraday "
      "(rodar scripts/collect_intraday.py durante o pregão).")
