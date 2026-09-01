"""Fase C — ML v2 (investigação mais funda). vs ml_baseline:
  - rótulo triple-barrier (não 'retorno > custo')
  - features de rank cross-sectional (posição relativa entre as ~19 ações)
  - remove 'month' (estava decorando o calendário)
  - treina E avalia no universo inteiro (~19 nomes), long nos top-K por score
  - reporta AUC out-of-sample por fold (se AUC ~0,5, o modelo não tem sinal)

Pré-requisito: python scripts/pull_universe.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from trader.backtest.cross_sectional import _vol_target
from trader.backtest.metrics import summarize
from trader.backtest.ml_portfolio import run_target_portfolio
from trader.backtest.split import WalkForward
from trader.costs.b3 import B3Costs
from trader.data.store import load
from trader.ml.dataset import build_panel, feature_columns, purge
from trader.ml.model import LgbmSignal
from trader.strategy import SuperTrend

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
DATA = Path(cfg["data"]["dir"])
EQ0, LOT = cfg["backtest"]["initial_equity"], cfg["backtest"]["lot_size"]
COSTS = B3Costs.from_config(cfg)
VOL = dict(target_ann_vol=cfg["risk_overlay"]["target_ann_vol"],
           lookback=cfg["risk_overlay"]["vol_lookback"],
           max_leverage=cfg["risk_overlay"]["max_leverage"])
HORIZON, EMBARGO, TOP_K, REBAL = 10, 5, 8, 5

frames = {}
for s in cfg["universe"]["symbols"]:
    p = DATA / f"{s.lower()}_d1.parquet"
    if p.exists():
        frames[s] = load(p)
print(f"{len(frames)} ações. Montando painel triple-barrier...")

FULL = build_panel(frames, horizon=HORIZON, label="triple_barrier",
                   pt_atr=2.0, sl_atr=1.5, add_xs_ranks=True, drop_cols=("month",))
COLS = feature_columns(FULL)
print(f"painel: {len(FULL)} linhas, {len(COLS)} features\n")

idx = FULL.index.unique().sort_values()
WF = WalkForward(n_folds=6, test_frac=0.55)
curves = defaultdict(list)
aucs, imp = [], defaultdict(float)

for tr_sl, te_sl in WF.splits(len(idx)):
    te_start, te_end = idx[te_sl.start], idx[te_sl.stop - 1]
    train = purge(FULL, te_start, horizon=HORIZON, embargo=EMBARGO)
    # teste: exclui as últimas HORIZON datas (rótulo truncado)
    safe_end = idx[max(te_sl.stop - 1 - HORIZON, te_sl.start)]
    test = FULL.loc[te_start:safe_end]
    if len(train) < 20000 or test.empty:
        continue

    model = LgbmSignal().fit(train[COLS], train["_y"])
    pred = model.predict(test[COLS])
    aucs.append(roc_auc_score(test["_y"].to_numpy(), pred))
    for k, v in model.importances().items():
        imp[k] += v

    wide = (test.assign(score=pred).reset_index()
            .pivot_table(index="time", columns="symbol", values="score"))
    st_ref = pd.DataFrame({s: (SuperTrend(10, 3.0).generate_signals(frames[s]) > 0)
                           .astype("float64") for s in frames}).reindex(wide.index)
    yz = _vol_target({s: frames[s] for s in frames}, **VOL).reindex(wide.index)

    # rebalance a cada REBAL dias: congela o score entre rebalanceamentos
    frozen = wide.copy()
    frozen.loc[~wide.index.isin(wide.index[::REBAL])] = np.nan
    frozen = frozen.ffill()
    ml_top = (frozen.rank(axis=1, ascending=False) <= TOP_K).astype("float64")

    for name, tgt in {"ml_topk": ml_top,
                      "supertrend_ref": st_ref.fillna(0.0),
                      "ml_and_st": (ml_top * st_ref.fillna(0.0))}.items():
        res = run_target_portfolio(frames, tgt, COSTS, overlay_mult=yz,
                                   initial_equity=EQ0, lot_size=LOT)
        curves[name].append(res.equity)

print(f"OOS AUC por fold: {[round(a, 3) for a in aucs]}  (média {np.mean(aucs):.3f})")
print(f"\n{'config':16} {'OOS Sharpe':>11} {'retorno':>9} {'maxDD':>8}")
print("-" * 48)
for name, parts in curves.items():
    level, stitched = EQ0, []
    for c in parts:
        stitched.append(c * (level / float(c.iloc[0])))
        level = float(stitched[-1].iloc[-1])
    eq = pd.concat(stitched)
    st = summarize(eq, pd.DataFrame(columns=["price", "shares", "fee"]), EQ0)
    print(f"{name:16} {str(st['sharpe']):>11} {str(st['retorno_total']):>9} "
          f"{str(st['max_drawdown']):>8}")

print("\ntop features:")
for k, v in sorted(imp.items(), key=lambda kv: -kv[1])[:15]:
    print(f"  {k:20} {v / max(len(aucs), 1):.0f}")
