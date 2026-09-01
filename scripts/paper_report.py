"""Compara o paper-trade real (data/paper_journal.csv) com o backtest da mesma
config no mesmo período. Mostra tracking error.

Pré-requisito: python scripts/pull_basket.py (backtest usa data/<sym>_d1.parquet).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml

from trader.backtest.portfolio import run_portfolio
from trader.costs.b3 import B3Costs
from trader.data.store import load
from trader.live.journal import read as read_journal
from trader.risk.overlay import vol_target_multiplier
from trader.strategy import SuperTrend

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
DATA = Path(cfg["data"]["dir"])
SYMS = cfg["basket"]["symbols"]
LOT = cfg["backtest"]["lot_size"]
YZ = dict(target_ann_vol=cfg["risk_overlay"]["target_ann_vol"],
          lookback=cfg["risk_overlay"]["vol_lookback"],
          max_leverage=cfg["risk_overlay"]["max_leverage"], estimator="yang_zhang")

jr = read_journal(DATA / "paper_journal.csv")
if len(jr) < 2:
    sys.exit(f"diário com {len(jr)} linha(s) — precisa de pelo menos 2 execuções.")

jr = jr.sort_values("ts")
live = jr.set_index("ts")["equity"].astype(float)
live_ret = live.iloc[-1] / live.iloc[0] - 1.0
live_dd = (live / live.cummax() - 1.0).min()
dias = (live.index[-1] - live.index[0]).days
print(f"PAPER (real)   {live.index[0].date()} .. {live.index[-1].date()}  ({dias} dias, "
      f"{len(live)} marcações)")
print(f"  retorno {live_ret:+.3%} | drawdown {live_dd:.3%} | "
      f"equity {live.iloc[0]:,.0f} -> {live.iloc[-1]:,.0f}")

start = live.index[0].tz_localize(None) if live.index[0].tzinfo else live.index[0]
frames = {}
for s in SYMS:
    p = DATA / f"{s.lower()}_d1.parquet"
    if not p.exists():
        sys.exit(f"falta {p} — rode scripts/pull_basket.py")
    frames[s] = load(p).loc[str(start.date()):]

yz = vol_target_multiplier(frames, **YZ)
res = run_portfolio(frames, SuperTrend(10, 3.0), B3Costs.from_config(cfg),
                    initial_equity=float(live.iloc[0]), lot_size=LOT, overlay_mult=yz)
bt = res.equity
bt_ret = bt.iloc[-1] / bt.iloc[0] - 1.0
bt_dd = (bt / bt.cummax() - 1.0).min()
print(f"\nBACKTEST (mesma config, mesma janela)")
print(f"  retorno {bt_ret:+.3%} | drawdown {bt_dd:.3%} | "
      f"equity {bt.iloc[0]:,.0f} -> {bt.iloc[-1]:,.0f}")

# tracking error: desvio-padrão da diferença de retornos diários alinhados por data
a = live.copy()
a.index = a.index.tz_localize(None) if a.index.tzinfo else a.index
b = bt.copy()
b.index = b.index.tz_localize(None) if b.index.tz is not None else b.index
j = pd.concat([a.rename("live"), b.rename("bt")], axis=1).dropna()
if len(j) >= 3:
    d = j["live"].pct_change() - j["bt"].pct_change()
    te = d.std() * np.sqrt(252)
    print(f"\ntracking error anualizado: {te:.2%}  "
          f"(diferença média diária {d.mean():+.4%})")
else:
    print("\npoucos pontos alinhados para tracking error ainda.")
