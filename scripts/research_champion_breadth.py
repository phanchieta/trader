"""Campeão com mais ações. Trend-following melhora com o número de mercados
independentes (Hurst, Ooi & Pedersen 2017); o campeão usa só 5.

Mesma regra (SuperTrend(10,3) long-only, vol-target Yang-Zhang da cesta 0,15, peso
1/N), cesta causal de N empresas mais negociadas no início de cada ano (§23),
R$4,90/ordem, banda de 15%, R$1 mi (com R$100 mil e N grande o lote de 100 distorce).
Todos os N mostrados — não é escolha de parâmetro, é ver se a curva melhora com N.

Pré-requisitos: build_stocks_db.py, pull_etfs.py, pull_cdi.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from trader.backtest.champion import causal_members, champion_targets, run_champion
from trader.backtest.metrics import vs_benchmark
from trader.costs.b3 import B3Costs
from trader.data.b3_panel import build_panel, load_rows
from trader.data.store import load
from trader.taxes import after_tax_summary

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
DATA = Path(cfg["data"]["dir"])
CDI = pd.read_parquet(DATA / "cdi_d1.parquet")["rate"]
CDI_N = CDI.set_axis(CDI.index.tz_convert(None).normalize())
RO = cfg["risk_overlay"]
COSTS = B3Costs.from_config(cfg)


def _pct(x, fmt="{:+.1%}"):
    return "—" if x is None or x != x else fmt.format(x)


rows = load_rows(DATA / "b3")
CAL = pd.DatetimeIndex(sorted(rows["date"].unique()))
bova = load(DATA / "bova11_d1.parquet")
bova.index = bova.index.tz_convert(None).normalize()
bova = bova[(bova["open"] > 0) & (bova["volume"] > 0)]
mgap = (bova["open"] / bova["close"].shift(1) - 1.0).reindex(CAL).fillna(0.0)
panel = build_panel(rows, CAL, market_gap=mgap, total_return=True)
FR = {i: f.reindex(CAL).ffill(limit=10) for i, f in panel.items()}

print("=" * 96)
print("CAMPEÃO x NÚMERO DE AÇÕES — cesta causal, R$4,90/ordem, banda 15%, R$1 mi")
print("=" * 96)
print(f"{'N':>3} {'janela':10} | {'CAGR':>6} {'alpha':>6} {'pós-IR':>7} {'Sh/CDI':>6} {'maxDD':>6} {'expo':>5} "
      f"{'ordens/a':>8} {'anos>CDI':>8}")
for n in (5, 10, 15, 20):
    members, _ = causal_members(panel, CAL, n=n)
    cols, tgt, mult = champion_targets(members, FR, CAL, n=n, target_vol=RO["target_ann_vol"],
                                       lookback=RO["vol_lookback"], max_leverage=RO["max_leverage"])
    for lo, lab in (("2010-01-01", "2010–2026"), ("2014-01-01", "2014–2026")):
        win = CAL[CAL >= lo]
        eq, st = run_champion(cols, tgt, mult, FR, win, COSTS, CDI_N, capital=1_000_000.0, band=0.15)
        b = vs_benchmark(eq, CDI)
        yrs = (win[-1] - win[0]).days / 365.25
        r = eq.pct_change().fillna(0.0)
        c = CDI.reindex(eq.index).ffill().fillna(0.0)
        ye = (1 + r).groupby(eq.index.year).prod() - (1 + c).groupby(eq.index.year).prod()
        print(f"{n:>3} {lab:10} | {_pct(b['cagr_estrategia']):>6} {_pct(b['alpha_aa']):>6} "
              f"{_pct(after_tax_summary(eq, CDI)['alpha_pos_ir']):>7} {b['sharpe_sobre_cdi']:>+6.2f} "
              f"{_pct(float((eq / eq.cummax() - 1).min()), '{:.0%}'):>6} {st['exposure_media']:>5.2f} "
              f"{st['n_trades'] / yrs:>8.0f} {f'{(ye > 0).mean():.0%}':>8}")
