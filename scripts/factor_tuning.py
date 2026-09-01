"""Tuning OOS do modelo de fatores B3: quintil/decil, frequência de rebalance,
conjunto de fatores, vol-target (Ibov vs book), hedge de índice (WIN).

Pré-requisitos: pull_factor_universe.py ; pull_futures_proxy.py ; pull_cdi.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from trader.backtest.metrics import summarize, vs_benchmark
from trader.backtest.split import WalkForward
from trader.data.store import load
from trader.factors import FACTORS, beta_hedge, combined_score, factor_oos, factor_score
from trader.risk.overlay import vol_target_multiplier

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
DATA = Path(cfg["data"]["dir"])
EQ0 = cfg["backtest"]["initial_equity"]
TV, LB, MAXLEV = (cfg["risk_overlay"]["target_ann_vol"],
                  cfg["risk_overlay"]["vol_lookback"], cfg["risk_overlay"]["max_leverage"])

frames = {}
for s in cfg["factor_universe"]["symbols"]:
    p = DATA / f"{s.lower()}_d1.parquet"
    if p.exists():
        frames[s] = load(p)
IBOV = load(DATA / "ibov_d1.parquet")
CDI = pd.read_parquet(DATA / "cdi_d1.parquet")["rate"]
IBOV_MULT = vol_target_multiplier({"IBOV": IBOV}, target_ann_vol=TV, lookback=LB,
                                  max_leverage=MAXLEV, estimator="yang_zhang")
WF = WalkForward(n_folds=6, test_frac=0.55)
mkt = IBOV["close"]

SCORE_MOM = factor_score(frames, "momentum", mkt=mkt)
SCORE_TREND = combined_score(frames, ["momentum", "low_vol", "low_beta"], mkt=mkt)
SCORE_ALL = combined_score(frames, list(FACTORS), mkt=mkt)

# (label, score, n_frac, rebal, overlay_mode, hedge)
CONFIGS = [
    ("all5 quintil mensal ibov",     SCORE_ALL,   0.20, 21, "ibov",  False),
    ("all5 decil mensal ibov",       SCORE_ALL,   0.10, 21, "ibov",  False),
    ("all5 quintil trim ibov",       SCORE_ALL,   0.20, 63, "ibov",  False),
    ("all5 quintil mensal book",     SCORE_ALL,   0.20, 21, "book",  False),
    ("all5 quintil mensal book+hedge", SCORE_ALL, 0.20, 21, "book",  True),
    ("mom quintil mensal ibov",      SCORE_MOM,   0.20, 21, "ibov",  False),
    ("mom quintil mensal book+hedge", SCORE_MOM,  0.20, 21, "book",  True),
    ("mom+lv+lb quintil mensal book", SCORE_TREND, 0.20, 21, "book", False),
    ("mom+lv+lb quintil book+hedge", SCORE_TREND, 0.20, 21, "book",  True),
]

print(f"{len(frames)} ações\n")
print(f"{'config':32} {'CAGR':>7} {'alpha':>7} {'Sh/CDI':>7} {'maxDD':>8}")
print("-" * 66)
for label, sc, nf, rb, mode, hedge in CONFIGS:
    eq = factor_oos(frames, sc, wf=WF, cdi=CDI, ibov_mult=IBOV_MULT, n_frac=nf,
                    rebal=rb, overlay_mode=mode, target_vol=TV, vol_lookback=40,
                    max_leverage=MAXLEV, eq0=EQ0)
    if hedge:
        eq = beta_hedge(eq, IBOV["close"])
    b = vs_benchmark(eq, CDI)
    s = summarize(eq, pd.DataFrame(columns=["price", "shares", "fee"]), EQ0)
    print(f"{label:32} {b['cagr_estrategia']:>7.1%} {b['alpha_aa']:>+7.1%} "
          f"{str(b['sharpe_sobre_cdi']):>7} {s['max_drawdown']:>8.1%}")
