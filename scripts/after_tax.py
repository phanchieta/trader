"""Os programas batem o CDI DEPOIS do imposto? Recomputa campeão / fatores /
blend 50-50 na janela OOS honesta e aplica o modelo de IR (trader/taxes.py).

Pré-requisitos: pull_basket.py ; pull_factor_universe.py ; pull_futures_proxy.py ; pull_cdi.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from trader.backtest.portfolio import run_portfolio
from trader.backtest.split import WalkForward
from trader.costs.b3 import B3Costs
from trader.data.store import load
from trader.factors import FACTORS, combined_score, factor_oos
from trader.risk.overlay import vol_target_multiplier
from trader.strategy import SuperTrend
from trader.taxes import after_tax_summary

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
score = combined_score(uni, list(FACTORS), mkt=IBOV["close"])
IDX = score.index

lvl, parts = EQ0, []
for _, te_sl in WF.splits(len(IDX)):
    w = IDX[te_sl.start:te_sl.stop]
    fr = {s: f.loc[w[0]:w[-1]] for s, f in basket.items() if len(f.loc[w[0]:w[-1]]) > 20}
    if len(fr) < len(basket):
        continue
    e = run_portfolio(fr, SuperTrend(10, 3.0), COSTS, overlay_mult=champ_yz,
                      cash_rate=CDI, initial_equity=EQ0, lot_size=LOT).equity
    parts.append(e * (lvl / float(e.iloc[0])))
    lvl = float(parts[-1].iloc[-1])
A = pd.concat(parts)
B = factor_oos(uni, score, wf=WF, cdi=CDI, ibov_mult=ibov_yz, n_frac=0.20,
               rebal=63, overlay_mode="ibov", target_vol=TV, vol_lookback=40,
               max_leverage=MAXLEV, eq0=EQ0)
common = A.index.intersection(B.index)
rA, rB = A.reindex(common).pct_change().fillna(0.0), B.reindex(common).pct_change().fillna(0.0)
BLEND = EQ0 * (1.0 + 0.5 * rA + 0.5 * rB).cumprod()

print(f"{'programa':16} {'CAGR pré':>9} {'CAGR pós':>9} {'CDI':>7} "
      f"{'alpha pré':>10} {'alpha pós':>10}")
print("-" * 66)
for name, eq in [("campeão", A.reindex(common)), ("fatores", B.reindex(common)),
                 ("blend 50/50", BLEND)]:
    r = after_tax_summary(eq, CDI, rate=0.155)
    print(f"{name:16} {r['cagr_pre_ir']:>9.1%} {r['cagr_pos_ir']:>9.1%} "
          f"{r['cagr_cdi']:>7.1%} {r['alpha_pre_ir']:>+10.1%} {r['alpha_pos_ir']:>+10.1%}")
print("\n(IR aprox.: 15,5% mensal sobre ganho líquido, prejuízo compensável, "
      "sem isenção de R$20k. Estimativa conservadora do haircut.)")
