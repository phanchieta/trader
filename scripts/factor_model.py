"""Modelo de fatores cross-sectional na B3 (~universo grande). Cada rebalance
(mensal) rankeia os nomes pelo score do fator, fica LONG no quintil de cima
(equal-weight), com vol-target (vol Yang-Zhang do Ibov) e CDI no caixa.
Walk-forward OOS, objetivo = Sharpe sobre o CDI.

Pré-requisitos: pull_factor_universe.py ; pull_futures_proxy.py ; pull_cdi.py ;
                pull_basket.py (referência do campeão)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml

from trader.backtest.metrics import summarize, vs_benchmark
from trader.backtest.ml_portfolio import run_target_portfolio
from trader.backtest.portfolio import run_portfolio
from trader.backtest.split import WalkForward
from trader.costs.b3 import B3Costs
from trader.data.store import load
from trader.factors.library import FACTORS, combined_score, factor_score
from trader.risk.overlay import vol_target_multiplier
from trader.strategy import SuperTrend

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
DATA = Path(cfg["data"]["dir"])
EQ0, LOT = cfg["backtest"]["initial_equity"], cfg["backtest"]["lot_size"]
COSTS = B3Costs.from_config(cfg)
TV, LB, MAXLEV = (cfg["risk_overlay"]["target_ann_vol"],
                  cfg["risk_overlay"]["vol_lookback"], cfg["risk_overlay"]["max_leverage"])
REBAL, QUINTILE = 63, 0.20      # trimestral (melhor no tuning); quintil de cima

frames = {}
for s in cfg["factor_universe"]["symbols"]:
    p = DATA / f"{s.lower()}_d1.parquet"
    if p.exists():
        frames[s] = load(p)
IBOV = load(DATA / "ibov_d1.parquet")
CDI = pd.read_parquet(DATA / "cdi_d1.parquet")["rate"]
YZ = vol_target_multiplier({"IBOV": IBOV}, target_ann_vol=TV, lookback=LB,
                           max_leverage=MAXLEV, estimator="yang_zhang")
print(f"{len(frames)} ações no universo de fatores\n")

# índice comum (interseção com o Ibov, a partir de quando há nomes suficientes)
idx = IBOV.index
for f in frames.values():
    idx = idx.union(f.index)
idx = idx.sort_values()
WF = WalkForward(n_folds=6, test_frac=0.55)


def target_from_score(score: pd.DataFrame, win) -> pd.DataFrame:
    sc = score.reindex(win)
    # congela entre rebalances
    keep = sc.index.isin(sc.index[::REBAL])
    sc = sc.where(pd.Series(keep, index=sc.index), np.nan).ffill()
    thr = sc.quantile(1.0 - QUINTILE, axis=1)
    picks = sc.ge(thr, axis=0) & sc.notna()
    n_pick = picks.sum(axis=1).replace(0, np.nan)
    return picks.div(n_pick, axis=0).fillna(0.0)      # equal-weight no quintil


def evaluate(score_full: pd.DataFrame):
    lvl, parts = EQ0, []
    for _, te_sl in WF.splits(len(idx)):
        win = idx[te_sl.start:te_sl.stop]
        tgt = target_from_score(score_full, win)
        fr = {s: frames[s] for s in frames}
        r = run_target_portfolio(fr, tgt, COSTS, overlay_mult=YZ, cash_rate=CDI,
                                 target_is_weight=True, initial_equity=EQ0, lot_size=LOT)
        parts.append(r.equity * (lvl / float(r.equity.iloc[0])))
        lvl = float(parts[-1].iloc[-1])
    return pd.concat(parts)


mkt = IBOV["close"]
CONFIGS = {f: factor_score(frames, f, mkt=mkt) for f in FACTORS}
CONFIGS["COMBINADO"] = combined_score(frames, list(FACTORS), mkt=mkt)

# referências
champ_basket = {s: load(DATA / f"{s.lower()}_d1.parquet") for s in cfg["basket"]["symbols"]}
champ_yz = vol_target_multiplier(champ_basket, target_ann_vol=TV, lookback=LB,
                                 max_leverage=MAXLEV, estimator="yang_zhang")

def _pct(x):
    return f"{x:>7.1%}" if isinstance(x, (int, float)) else f"{'--':>7}"


def _row(name, eq):
    b = vs_benchmark(eq, CDI)
    s = summarize(eq, pd.DataFrame(columns=["price", "shares", "fee"]), EQ0)
    print(f"{name:16} {_pct(b['cagr_estrategia'])} {_pct(b['cagr_cdi'])} "
          f"{_pct(b['alpha_aa'])} {str(b['sharpe_sobre_cdi']):>7} {_pct(s['max_drawdown'])}")


print(f"{'config':16} {'CAGR':>7} {'CAGR CDI':>7} {'alpha':>7} {'Sh/CDI':>7} {'maxDD':>7}")
print("-" * 60)
for name, sc in CONFIGS.items():
    _row(name, evaluate(sc))

# campeão (5 ações, SuperTrend + YZ) na mesma janela OOS
lvl, parts = EQ0, []
for _, te_sl in WF.splits(len(idx)):
    w = idx[te_sl.start:te_sl.stop]
    fr = {s: f.loc[w[0]:w[-1]] for s, f in champ_basket.items()}
    e = run_portfolio(fr, SuperTrend(10, 3.0), COSTS, overlay_mult=champ_yz,
                      cash_rate=CDI, initial_equity=EQ0, lot_size=LOT).equity
    if len(e) == 0:
        continue
    parts.append(e * (lvl / float(e.iloc[0])))
    lvl = float(parts[-1].iloc[-1])
_row("CAMPEÃO (5 ações)", pd.concat(parts))
