"""Roda o backtest da estratégia de regra sobre o histórico diário. Rodar da raiz do repo.

Pré-requisito: `python scripts/pull_daily.py` (gera o Parquet).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # roda sem pip install -e .

import yaml

from trader.backtest.engine import run
from trader.costs.b3 import B3Costs
from trader.data.store import load
from trader.strategy.sma_crossover import SmaCrossover

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
df = load(Path(cfg["data"]["dir"]) / cfg["data"]["daily_file"])

strat = SmaCrossover(fast=20, slow=60)
costs = B3Costs.from_config(cfg)
res = run(df, strat, costs,
          initial_equity=cfg["backtest"]["initial_equity"],
          lot_size=cfg["backtest"]["lot_size"])

print(f"\n{strat.name}  |  {df.index[0].date()} .. {df.index[-1].date()}  ({len(df)} barras)")
print("-" * 48)
for k, v in res.stats.items():
    print(f"  {k:26} {v}")

bh = df["close"].iloc[-1] / df["close"].iloc[0] - 1.0
print(f"\n  {'buy&hold_retorno_total':26} {round(bh, 4)}   <- referência")
