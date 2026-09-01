"""O overlay de eleição ajuda? Roda o campeão com e sem o haircut de exposição
(15/set–20/nov dos anos 2010/14/18/22) no walk-forward, e mostra o que aconteceu
em out–dez de 2018 e 2022 especificamente.

Pré-requisitos: pull_basket.py ; pull_cdi.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from trader.backtest.metrics import summarize, vs_benchmark
from trader.backtest.portfolio import run_portfolio
from trader.backtest.split import WalkForward
from trader.costs.b3 import B3Costs
from trader.data.store import load
from trader.risk.election import haircut_series
from trader.risk.overlay import vol_target_multiplier
from trader.strategy import SuperTrend

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
DATA = Path(cfg["data"]["dir"])
EQ0, LOT = cfg["backtest"]["initial_equity"], cfg["backtest"]["lot_size"]
COSTS = B3Costs.from_config(cfg)
TV, LB, MAXLEV = (cfg["risk_overlay"]["target_ann_vol"],
                  cfg["risk_overlay"]["vol_lookback"], cfg["risk_overlay"]["max_leverage"])

basket = {s: load(DATA / f"{s.lower()}_d1.parquet") for s in cfg["basket"]["symbols"]}
CDI = pd.read_parquet(DATA / "cdi_d1.parquet")["rate"]
YZ = vol_target_multiplier(basket, target_ann_vol=TV, lookback=LB,
                           max_leverage=MAXLEV, estimator="yang_zhang")
HC = haircut_series(YZ.index, factor=0.5)

idx = None
for f in basket.values():
    idx = f.index if idx is None else idx.intersection(f.index)
idx = idx.sort_values()
WF = WalkForward(n_folds=8, test_frac=0.60)


def run(mult):
    lvl, parts = EQ0, []
    for _, te_sl in WF.splits(len(idx)):
        w = idx[te_sl.start:te_sl.stop]
        fr = {s: f.loc[w[0]:w[-1]] for s, f in basket.items()}
        e = run_portfolio(fr, SuperTrend(10, 3.0), COSTS, overlay_mult=mult,
                          cash_rate=CDI, initial_equity=EQ0, lot_size=LOT).equity
        parts.append(e * (lvl / float(e.iloc[0])))
        lvl = float(parts[-1].iloc[-1])
    return pd.concat(parts)


base = run(YZ)
hedged = run(YZ * HC)

print(f"{'config':22} {'CAGR':>7} {'alpha':>7} {'Sh/CDI':>7} {'maxDD':>8}")
print("-" * 56)
for name, eq in [("campeão", base), ("campeão + haircut eleição", hedged)]:
    b = vs_benchmark(eq, CDI)
    s = summarize(eq, pd.DataFrame(columns=["price", "shares", "fee"]), EQ0)
    print(f"{name:22} {b['cagr_estrategia']:>7.1%} {b['alpha_aa']:>+7.1%} "
          f"{str(b['sharpe_sobre_cdi']):>7} {s['max_drawdown']:>8.1%}")

for y in (2018, 2022):
    lo, hi = f"{y}-09-15", f"{y}-12-31"
    for name, eq in [("sem", base), ("com haircut", hedged)]:
        seg = eq.loc[lo:hi]
        if len(seg) > 1:
            print(f"  {y} 15/set–31/dez [{name:11}]: {seg.iloc[-1] / seg.iloc[0] - 1:+.2%}")
