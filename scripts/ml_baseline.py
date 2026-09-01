"""Fase C — baseline de ML. Walk-forward OOS na cesta de 5, modelo treinado no
universo de ~19 ações. Compara:

  champion       — SuperTrend + Yang-Zhang vol-target (referência)
  ml_only        — long onde a probabilidade do LGBM >= threshold
  ml_and_champ   — long só quando ML E SuperTrend concordam (filtro)
  ensemble       — voto de {ML, SuperTrend, Vortex} >= 2

Todos com o overlay Yang-Zhang. Regra: tem que bater o champion OOS.

Pré-requisitos: python scripts/pull_basket.py ; python scripts/pull_universe.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml

from trader.backtest.ml_portfolio import run_target_portfolio
from trader.backtest.split import WalkForward
from trader.costs.b3 import B3Costs
from trader.data.store import load
from trader.ml.dataset import build_panel, feature_columns, purge
from trader.ml.features import build_features
from trader.ml.model import LgbmSignal
from trader.risk.overlay import vol_target_multiplier
from trader.strategy import SuperTrend, Vortex

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
DATA = Path(cfg["data"]["dir"])
EQ0, LOT = cfg["backtest"]["initial_equity"], cfg["backtest"]["lot_size"]
COSTS = B3Costs.from_config(cfg)
YZ = dict(target_ann_vol=cfg["risk_overlay"]["target_ann_vol"],
          lookback=cfg["risk_overlay"]["vol_lookback"],
          max_leverage=cfg["risk_overlay"]["max_leverage"], estimator="yang_zhang")

BASKET = cfg["basket"]["symbols"]
UNIVERSE = cfg["universe"]["symbols"]
HORIZON, COST_HURDLE, EMBARGO, THRESHOLD = 10, 0.004, 5, 0.55


def _load(syms):
    out = {}
    for s in syms:
        p = DATA / f"{s.lower()}_d1.parquet"
        if p.exists():
            out[s] = load(p)
    return out


basket = _load(BASKET)
universe = _load(UNIVERSE)
if len(basket) < len(BASKET) or len(universe) < 8:
    sys.exit("faltam parquets — rode pull_basket.py e pull_universe.py")

idx = None
for f in basket.values():
    idx = f.index if idx is None else idx.intersection(f.index)
idx = idx.sort_values()

# sinais de regra na cesta (long-only 0/1), causais, calculados no frame inteiro
champ_sig = {s: (SuperTrend(10, 3.0).generate_signals(basket[s]) > 0).astype("float64")
             for s in basket}
vortex_sig = {s: (Vortex(14).generate_signals(basket[s]) > 0).astype("float64")
              for s in basket}

WF = WalkForward(n_folds=6, test_frac=0.55)
curves = defaultdict(list)
imp_acc: dict = defaultdict(float)
n_folds = 0

for tr_sl, te_sl in WF.splits(len(idx)):
    te_start, te_end = idx[te_sl.start], idx[te_sl.stop - 1]
    tr_univ = {s: f.loc[:te_start] for s, f in universe.items()}

    panel = build_panel(tr_univ, horizon=HORIZON, cost_hurdle=COST_HURDLE)
    panel = purge(panel, te_start, horizon=HORIZON, embargo=EMBARGO)
    cols = feature_columns(panel)
    if len(panel) < 5000:
        continue
    model = LgbmSignal().fit(panel[cols], panel["_y"])
    for k, val in model.importances().items():
        imp_acc[k] += val
    n_folds += 1

    # scores na cesta, janela de teste
    scores = {}
    for s in basket:
        feats = build_features(basket[s]).loc[te_start:te_end].dropna()
        if feats.empty:
            continue
        scores[s] = pd.Series(model.predict(feats[cols]), index=feats.index)
    sc = pd.DataFrame(scores).reindex(idx[te_sl.start:te_sl.stop])
    win = sc.index

    yz = vol_target_multiplier({s: basket[s] for s in basket}, **YZ)

    ml_only = (sc >= THRESHOLD).astype("float64")
    champ = pd.DataFrame({s: champ_sig[s].reindex(win).fillna(0.0) for s in basket})
    vtx = pd.DataFrame({s: vortex_sig[s].reindex(win).fillna(0.0) for s in basket})
    targets = {
        "champion": champ,
        "ml_only": ml_only,
        "ml_and_champ": (ml_only * champ),
        "ensemble": ((ml_only + champ + vtx) >= 2).astype("float64"),
    }
    for name, tgt in targets.items():
        res = run_target_portfolio(basket, tgt.reindex(win), COSTS, overlay_mult=yz,
                                   initial_equity=EQ0, lot_size=LOT)
        curves[name].append(res.equity)

print(f"walk-forward {n_folds} folds | horizon {HORIZON}d | threshold {THRESHOLD}\n")
print(f"{'config':16} {'OOS Sharpe':>11} {'retorno':>9} {'maxDD':>8} {'custo/gr':>9} {'trades':>7}")
print("-" * 64)
for name, parts in curves.items():
    level, stitched = EQ0, []
    for c in parts:
        stitched.append(c * (level / float(c.iloc[0])))
        level = float(stitched[-1].iloc[-1])
    eq = pd.concat(stitched)
    from trader.backtest.metrics import summarize
    st = summarize(eq, pd.DataFrame(columns=["price", "shares", "fee"]), EQ0)
    print(f"{name:16} {str(st['sharpe']):>11} {str(st['retorno_total']):>9} "
          f"{str(st['max_drawdown']):>8} {'-':>9} {'-':>7}")

top = sorted(imp_acc.items(), key=lambda kv: -kv[1])[:15]
print("\ntop features (importância média LGBM):")
for k, val in top:
    print(f"  {k:18} {val / max(n_folds, 1):.0f}")
