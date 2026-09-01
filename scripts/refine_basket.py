"""Direção A: refinar o sistema de regras na cesta.

  - varredura do % do trailing-stop
  - confirmação por volume (só entra com volume acima da média)
  - ensemble por votação (SMA + reversão + momentum)

Tudo walk-forward OOS na cesta. Pré-requisito: python scripts/pull_basket.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from trader.backtest.split import WalkForward
from trader.backtest.validation import walk_forward_portfolio
from trader.costs.b3 import B3Costs
from trader.data.store import load
from trader.strategy import Ensemble, MeanReversion, Momentum, SmaCrossover, VolumeConfirmed

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
DATA = Path(cfg["data"]["dir"])
EQ0, LOT = cfg["backtest"]["initial_equity"], cfg["backtest"]["lot_size"]
SYMS = cfg["basket"]["symbols"]
COSTS = B3Costs.from_config(cfg)

frames = {}
for s in SYMS:
    p = DATA / f"{s.lower()}_d1.parquet"
    if not p.exists():
        sys.exit(f"falta {p} — rode: python scripts/pull_basket.py")
    frames[s] = load(p)

start = max(f.index[0] for f in frames.values())
end = min(f.index[-1] for f in frames.values())
print(f"cesta: {', '.join(SYMS)} | janela {start.date()}..{end.date()}\n")

WF = WalkForward(n_folds=8, test_frac=0.60)


# ---- factories de estratégia a partir de params planos (p/ o grid_search) ----

def make_sma(fast=10, slow=100, allow_short=False):
    return SmaCrossover(fast=fast, slow=slow, allow_short=allow_short)


def make_sma_vc(fast=10, slow=100, vol_min=1.2):
    return VolumeConfirmed(SmaCrossover(fast, slow), lookback=20, min_rel_volume=vol_min)


def make_ensemble(sma_slow=100, mr_lookback=20, min_agree=2):
    return Ensemble([SmaCrossover(10, sma_slow),
                     MeanReversion(mr_lookback, 1.5, 0.5),
                     Momentum(120)], min_agree=min_agree)


def make_ensemble_vc(sma_slow=100, mr_lookback=20, min_agree=2, vol_min=1.2):
    return VolumeConfirmed(make_ensemble(sma_slow, mr_lookback, min_agree),
                           lookback=20, min_rel_volume=vol_min)


SMA_GRID = {"fast": [10, 20], "slow": [50, 100, 200], "allow_short": [False]}
SMA_VC_GRID = {"fast": [10, 20], "slow": [50, 100, 200], "vol_min": [1.0, 1.3]}
ENS_GRID = {"sma_slow": [50, 100, 200], "mr_lookback": [20, 40], "min_agree": [2, 3]}
ENS_VC_GRID = {"sma_slow": [50, 100, 200], "mr_lookback": [20, 40],
               "min_agree": [2], "vol_min": [1.0, 1.3]}

CONFIGS = [
    ("SMA base",                     make_sma,        SMA_GRID,    None),
    ("SMA + trail 8%",               make_sma,        SMA_GRID,    0.08),
    ("SMA + trail 10%",              make_sma,        SMA_GRID,    0.10),
    ("SMA + trail 12%",              make_sma,        SMA_GRID,    0.12),
    ("SMA + trail 15%",              make_sma,        SMA_GRID,    0.15),
    ("SMA + trail 20%",              make_sma,        SMA_GRID,    0.20),
    ("SMA + trail 12% + vol-conf",   make_sma_vc,     SMA_VC_GRID, 0.12),
    ("Ensemble(SMA,MR,Mom)",         make_ensemble,   ENS_GRID,    None),
    ("Ensemble + trail 12%",         make_ensemble,   ENS_GRID,    0.12),
    ("Ensemble + trail 12% + v-conf", make_ensemble_vc, ENS_VC_GRID, 0.12),
]

print(f"{'config':32} {'Sharpe':>7} {'retorno':>9} {'maxDD':>8} {'custo/gr':>9} "
      f"{'trades':>7} {'ΔSharpe':>8}")
print("-" * 86)
best = None
for label, make, grid, trail in CONFIGS:
    rep = walk_forward_portfolio(make, grid, frames, COSTS, wf=WF, trailing_stop=trail,
                                 min_trades=20, initial_equity=EQ0, lot_size=LOT)
    o = rep.oos_stats
    print(f"{label:32} {str(o.get('sharpe')):>7} {str(o.get('retorno_total')):>9} "
          f"{str(o.get('max_drawdown')):>8} {str(o.get('custo_sobre_lucro_bruto')):>9} "
          f"{str(o.get('n_trades')):>7} {str(rep.degradacao().get('sharpe')):>8}")
    sh = o.get("sharpe")
    if sh is not None and (best is None or sh > best[1]):
        best = (label, sh, rep)

if best:
    _, _, rep = best
    eq = rep.oos_equity
    per_year = {int(y): round(float(s.iloc[-1] / s.iloc[0] - 1), 3)
                for y, s in eq.groupby(eq.index.year)}
    print(f"\nmelhor: {best[0]}  (Sharpe {best[1]})")
    print(f"  params por fold: {best[2].best_params}")
    print(f"  retorno por ano: {per_year}")
