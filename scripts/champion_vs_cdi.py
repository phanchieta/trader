"""O campeão bate o CDI? Walk-forward do SuperTrend + vol-target Yang-Zhang com
JUROS NO CAIXA (série CDI do BACEN), em vários níveis de vol-alvo (menos vol-alvo
= mais caixa parado rendendo CDI; mais vol-alvo = mais exposição, mais risco).

Pré-requisitos: python scripts/pull_basket.py ; python scripts/pull_cdi.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml

from trader.backtest.metrics import summarize, vs_benchmark
from trader.backtest.portfolio import run_portfolio
from trader.backtest.split import WalkForward
from trader.costs.b3 import B3Costs
from trader.data.store import load
from trader.risk.overlay import vol_target_multiplier
from trader.strategy import SuperTrend

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
DATA = Path(cfg["data"]["dir"])
EQ0, LOT = cfg["backtest"]["initial_equity"], cfg["backtest"]["lot_size"]
COSTS = B3Costs.from_config(cfg)
LB, MAXLEV = cfg["risk_overlay"]["vol_lookback"], cfg["risk_overlay"]["max_leverage"]

basket = {}
for s in cfg["basket"]["symbols"]:
    basket[s] = load(DATA / f"{s.lower()}_d1.parquet")
cdi_path = DATA / "cdi_d1.parquet"
if not cdi_path.exists():
    sys.exit("falta data/cdi_d1.parquet — rode: python scripts/pull_cdi.py")
CDI = load_cdi = pd.read_parquet(cdi_path)["rate"]

idx = None
for f in basket.values():
    idx = f.index if idx is None else idx.intersection(f.index)
idx = idx.sort_values()
WF = WalkForward(n_folds=8, test_frac=0.60)


def _stitch(parts):
    lvl, out = EQ0, []
    for c in parts:
        out.append(c * (lvl / float(c.iloc[0])))
        lvl = float(out[-1].iloc[-1])
    return pd.concat(out)


def run_champion(target_vol):
    curves, expo = [], []
    for tr_sl, te_sl in WF.splits(len(idx)):
        te = {s: f.loc[idx[te_sl.start]:idx[te_sl.stop - 1]] for s, f in basket.items()}
        mult = None
        if target_vol is not None:
            mult = vol_target_multiplier(te, target_ann_vol=target_vol, lookback=LB,
                                        max_leverage=MAXLEV, estimator="yang_zhang")
        res = run_portfolio(te, SuperTrend(10, 3.0), COSTS, overlay_mult=mult,
                            cash_rate=CDI, initial_equity=EQ0, lot_size=LOT)
        curves.append(res.equity)
        expo.append(res.stats["exposure_media"])
    return _stitch(curves), float(np.mean(expo))


# referência: cesta comprada e segura (equal-weight), sem caixa
def buy_hold():
    parts = []
    for tr_sl, te_sl in WF.splits(len(idx)):
        w = idx[te_sl.start:te_sl.stop]
        norm = pd.DataFrame({s: basket[s]["close"].reindex(w) for s in basket})
        norm = norm / norm.iloc[0]
        parts.append(norm.mean(axis=1) * EQ0)
    return _stitch(parts)


print(f"janela OOS: {idx[int(len(idx) * 0.4)].date()} .. {idx[-1].date()}\n")
print(f"{'config':22} {'CAGR estr':>10} {'CAGR CDI':>9} {'alpha a.a.':>11} "
      f"{'Sh/CDI':>7} {'maxDD':>8} {'expo méd':>9}")
print("-" * 82)

rows = [("buy&hold cesta", buy_hold(), 1.0)]
for tv in (0.12, 0.15, 0.20, 0.30, 0.50, None):
    eq, ex = run_champion(tv)
    rows.append((f"SuperTrend vol={tv or 'sem'}", eq, ex))

for name, eq, ex in rows:
    b = vs_benchmark(eq, CDI)
    s = summarize(eq, pd.DataFrame(columns=["price", "shares", "fee"]), EQ0)
    print(f"{name:22} {str(b['cagr_estrategia']):>10} {str(b['cagr_cdi']):>9} "
          f"{str(b['alpha_aa']):>11} {str(b['sharpe_sobre_cdi']):>7} "
          f"{str(s['max_drawdown']):>8} {ex:>9.2f}")
