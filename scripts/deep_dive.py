"""Walk-forward aprofundado das combinações que a triagem apontou.

Sharpe por fold é o diagnóstico-chave: se pula de +2 para −1 entre folds, é
ruído; se fica em torno de um valor, é sinal. Idem params por fold — se mudam
a cada fold, a estratégia não tem um regime estável.

Pré-requisitos: pull_daily.py + pull_mt5_history.py D1 M15 M5
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from trader.backtest.split import WalkForward
from trader.backtest.validation import walk_forward
from trader.costs.b3 import B3Costs
from trader.data.store import load
from trader.strategy import MeanReversion, Momentum, SmaCrossover

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
DATA = Path(cfg["data"]["dir"])
EQ0, LOT = cfg["backtest"]["initial_equity"], cfg["backtest"]["lot_size"]
c = cfg["costs"]

SWING = B3Costs.from_config(cfg)
DAYTRADE = B3Costs(emolumentos_pct=c["emolumentos_pct_daytrade"],
                   corretagem_por_ordem=c["corretagem_por_ordem"],
                   iss_sobre_corretagem=c["iss_sobre_corretagem"],
                   slippage_ticks=c["slippage_ticks"], tick_size=c["tick_size"])

JOBS = [
    ("D1 momentum (Yahoo 2000-2026)", "petr4_d1.parquet", Momentum,
     {"lookback": [40, 120, 200], "band": [0.0, 0.02],
      "trend_filter": [0, 200], "allow_short": [False]},
     SWING, WalkForward(n_folds=8, test_frac=0.60)),
    ("D1 sma_crossover (Yahoo 2000-2026)", "petr4_d1.parquet", SmaCrossover,
     {"fast": [10, 20, 50], "slow": [50, 100, 200], "allow_short": [False]},
     SWING, WalkForward(n_folds=8, test_frac=0.60)),
    ("M15 mean_reversion (MT5 2021-2026)", "petr4_mt5_m15.parquet", MeanReversion,
     {"lookback": [28, 56, 112], "entry_z": [1.0, 1.5, 2.0],
      "exit_z": [0.0, 0.5], "allow_short": [False]},
     DAYTRADE, WalkForward(n_folds=6, test_frac=0.50)),
    ("M5 mean_reversion (MT5 2022-2026)", "petr4_mt5_m5.parquet", MeanReversion,
     {"lookback": [84, 168, 336], "entry_z": [1.0, 1.5, 2.0],
      "exit_z": [0.0, 0.5], "allow_short": [False]},
     DAYTRADE, WalkForward(n_folds=6, test_frac=0.50)),
]

for name, fname, cls, grid, costs, wf in JOBS:
    p = DATA / fname
    print(f"\n{'#' * 70}\n### {name}")
    if not p.exists():
        print(f"  sem {fname} — pule 'pull_*'")
        continue
    df = load(p)
    rep = walk_forward(cls, grid, df, costs, wf=wf, min_trades=10,
                       initial_equity=EQ0, lot_size=LOT)
    o = rep.oos_stats
    print(f"  {len(df)} barras | {wf.n_folds} folds | OOS = últimos {int(wf.test_frac * 100)}%")
    print(f"  WF OOS : Sharpe {o.get('sharpe')} | ret {o.get('retorno_total')} | "
          f"maxDD {o.get('max_drawdown')} | custo/gross {o.get('custo_sobre_lucro_bruto')} "
          f"| trades {o.get('n_trades')}")
    print(f"  IS méd : Sharpe {rep.is_stats.get('sharpe')}  ->  ΔSharpe {rep.degradacao().get('sharpe')}")
    print(f"  Sharpe por fold : {[f['oos'].get('sharpe') for f in rep.folds]}")
    print(f"  params por fold : {[f['params'] for f in rep.folds]}")
    eq = rep.oos_equity
    per_year = {int(y): round(float(s.iloc[-1] / s.iloc[0] - 1), 3)
                for y, s in eq.groupby(eq.index.year)}
    print(f"  retorno por ano : {per_year}")
