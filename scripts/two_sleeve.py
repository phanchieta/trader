"""Portfólio de duas sleeves: campeão (SuperTrend + YZ, 5 ações, drawdown baixo)
+ modelo de fatores (combinado, retorno alto). Mistura os retornos e mede vs CDI.

Pré-requisitos: pull_basket.py ; pull_factor_universe.py ; pull_futures_proxy.py ; pull_cdi.py
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
from trader.factors import FACTORS, combined_score, factor_oos
from trader.risk.overlay import vol_target_multiplier
from trader.strategy import SuperTrend

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
DATA = Path(cfg["data"]["dir"])
EQ0, LOT = cfg["backtest"]["initial_equity"], cfg["backtest"]["lot_size"]
COSTS = B3Costs.from_config(cfg)
TV, LB, MAXLEV = (cfg["risk_overlay"]["target_ann_vol"],
                  cfg["risk_overlay"]["vol_lookback"], cfg["risk_overlay"]["max_leverage"])

basket = {s: load(DATA / f"{s.lower()}_d1.parquet") for s in cfg["basket"]["symbols"]}
uni = {s: load(DATA / f"{s.lower()}_d1.parquet")
       for s in cfg["factor_universe"]["symbols"]
       if (DATA / f"{s.lower()}_d1.parquet").exists()}
IBOV = load(DATA / "ibov_d1.parquet")
CDI = pd.read_parquet(DATA / "cdi_d1.parquet")["rate"]

WF = WalkForward(n_folds=6, test_frac=0.55)
champ_yz = vol_target_multiplier(basket, target_ann_vol=TV, lookback=LB,
                                 max_leverage=MAXLEV, estimator="yang_zhang")
ibov_yz = vol_target_multiplier({"IBOV": IBOV}, target_ann_vol=TV, lookback=LB,
                                max_leverage=MAXLEV, estimator="yang_zhang")

# Índice comum: o do universo de fatores (OOS mais longo e honesto, ~2011+).
score = combined_score(uni, list(FACTORS), mkt=IBOV["close"])
IDX = score.index

# Sleeve A: campeão (5 ações), mesma janela de walk-forward.
lvl, parts = EQ0, []
for _, te_sl in WF.splits(len(IDX)):
    w = IDX[te_sl.start:te_sl.stop]
    fr = {s: f.loc[w[0]:w[-1]] for s, f in basket.items()
          if len(f.loc[w[0]:w[-1]]) > 20}
    if len(fr) < len(basket):
        continue
    e = run_portfolio(fr, SuperTrend(10, 3.0), COSTS, overlay_mult=champ_yz,
                      cash_rate=CDI, initial_equity=EQ0, lot_size=LOT).equity
    parts.append(e * (lvl / float(e.iloc[0])))
    lvl = float(parts[-1].iloc[-1])
A = pd.concat(parts)

# Sleeve B: fatores combinados, quintil, TRIMESTRAL (melhor no tuning), ibov vol-target
B = factor_oos(uni, score, wf=WF, cdi=CDI, ibov_mult=ibov_yz, n_frac=0.20,
               rebal=63, overlay_mode="ibov", target_vol=TV, vol_lookback=40,
               max_leverage=MAXLEV, eq0=EQ0)

common = A.index.intersection(B.index)
rA, rB = A.reindex(common).pct_change().fillna(0.0), B.reindex(common).pct_change().fillna(0.0)


def curve(w):
    return EQ0 * (1.0 + (1.0 - w) * rA + w * rB).cumprod()


print(f"{'config':22} {'CAGR':>7} {'alpha':>7} {'Sh/CDI':>7} {'maxDD':>8}")
print("-" * 56)
for label, eq in [("campeão só (A)", A.reindex(common)),
                  ("fatores só (B)", B.reindex(common)),
                  ("30% fatores", curve(0.30)),
                  ("50% fatores", curve(0.50)),
                  ("70% fatores", curve(0.70))]:
    b = vs_benchmark(eq, CDI)
    s = summarize(eq, pd.DataFrame(columns=["price", "shares", "fee"]), EQ0)
    print(f"{label:22} {b['cagr_estrategia']:>7.1%} {b['alpha_aa']:>+7.1%} "
          f"{str(b['sharpe_sobre_cdi']):>7} {s['max_drawdown']:>8.1%}")
print(f"\ncorrelação diária A×B: {rA.corr(rB):.2f}")
