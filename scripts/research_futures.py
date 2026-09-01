"""Adiciona WIN/WDO (proxies ^BVSP e USDBRL=X) ao programa de trend.
Walk-forward, CDI no caixa, objetivo = Sharpe sobre o CDI.

Pré-requisitos: pull_basket.py ; pull_futures_proxy.py ; pull_cdi.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from trader.backtest.metrics import summarize, vs_benchmark
from trader.backtest.split import WalkForward
from trader.backtest.trend_portfolio import run_trend_portfolio
from trader.data.store import load
from trader.strategy import SuperTrend

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
DATA = Path(cfg["data"]["dir"])
EQ0 = cfg["backtest"]["initial_equity"]
BPS_S, BPS_F = cfg["futures"]["cost_bps_stock"], cfg["futures"]["cost_bps_future"]
TARGET_VOL = cfg["risk_overlay"]["target_ann_vol"]

STOCKS = cfg["basket"]["symbols"]
frames = {s: load(DATA / f"{s.lower()}_d1.parquet") for s in STOCKS}
frames["IBOV"] = load(DATA / "ibov_d1.parquet")
frames["USDBRL"] = load(DATA / "usdbrl_d1.parquet")
CDI = pd.read_parquet(DATA / "cdi_d1.parquet")["rate"]

# sinais: ações long-only; futuros long/short
SIG = {}
for s in STOCKS:
    SIG[s] = SuperTrend(10, 3.0, allow_short=False).generate_signals(frames[s])
SIG["IBOV"] = SuperTrend(10, 3.0, allow_short=True).generate_signals(frames["IBOV"])
SIG["USDBRL"] = SuperTrend(10, 3.0, allow_short=True).generate_signals(frames["USDBRL"])

WF = WalkForward(n_folds=8, test_frac=0.60)

CONFIGS = {
    "só ações (ref)":          (STOCKS, 1.0),
    "ações + USDBRL":          (STOCKS + ["USDBRL"], 1.0),
    "ações + IBOV + USDBRL":   (STOCKS + ["IBOV", "USDBRL"], 1.0),
    "IBOV + USDBRL (macro)":   (["IBOV", "USDBRL"], 1.0),
    "ações + USDBRL, lev 3x":  (STOCKS + ["USDBRL"], 3.0),
}


def evaluate(assets, maxlev):
    common = None
    for a in assets:
        common = frames[a].index if common is None else common.intersection(frames[a].index)
    common = common.sort_values()
    sig = pd.DataFrame({a: SIG[a] for a in assets}).reindex(common)
    bw = {a: 1.0 / len(assets) for a in assets}
    cost = {a: (BPS_F if a in ("IBOV", "USDBRL") else BPS_S) for a in assets}

    lvl, parts, expo = EQ0, [], []
    for _, te_sl in WF.splits(len(common)):
        w = common[te_sl.start:te_sl.stop]
        fr = {a: frames[a].loc[w[0]:w[-1]] for a in assets}
        r = run_trend_portfolio(fr, sig.loc[w[0]:w[-1]], base_weights=bw,
                                target_vol=TARGET_VOL, max_leverage=maxlev,
                                cash_rate=CDI, cost_bps=cost, initial_equity=EQ0)
        parts.append(r.equity * (lvl / float(r.equity.iloc[0])))
        lvl = float(parts[-1].iloc[-1])
        expo.append(r.exposure_media)
    return pd.concat(parts), float(pd.Series(expo).mean())


print(f"{'config':24} {'CAGR':>7} {'CAGR CDI':>9} {'alpha':>7} {'Sh/CDI':>7} "
      f"{'maxDD':>8} {'expo':>6}")
print("-" * 74)
for name, (assets, lev) in CONFIGS.items():
    eq, ex = evaluate(assets, lev)
    b = vs_benchmark(eq, CDI)
    s = summarize(eq, pd.DataFrame(columns=["price", "shares", "fee"]), EQ0)
    print(f"{name:24} {b['cagr_estrategia']:>7.1%} {b['cagr_cdi']:>9.1%} "
          f"{b['alpha_aa']:>+7.1%} {str(b['sharpe_sobre_cdi']):>7} "
          f"{s['max_drawdown']:>8.1%} {ex:>6.2f}")
