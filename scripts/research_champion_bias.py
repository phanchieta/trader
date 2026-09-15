"""O campeão também tem viés de escolha? A cesta (PETR4, VALE3, ITUB4, BBDC4, ABEV3)
foi escolhida olhando para as maiores de HOJE.

Mesma regra — SuperTrend(10,3) long-only por ação, vol-target Yang-Zhang da cesta
(alvo 0,15, janela 20, sem alavancagem), peso igual, CDI no caixa, custo B3, R$1 mi —
em três cestas:
  Yahoo fixa  as 5 de hoje, dados do Yahoo (o campeão como está no §CAMPEÃO)
  B3 fixa     as 5 de hoje, dados da B3 com provento estimado
  B3 causal   no 1º pregão de cada ano, as 5 empresas mais negociadas (mediana de R$
              em 250 pregões, uma classe por empresa) — só o que se sabia na data

Pré-requisitos: build_stocks_db.py, pull_basket.py, pull_etfs.py, pull_cdi.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml

from trader.backtest.metrics import vs_benchmark
from trader.backtest.ml_portfolio import run_target_portfolio
from trader.costs.b3 import B3Costs
from trader.data.b3_panel import build_panel, load_rows
from trader.data.store import load
from trader.indicators import yang_zhang
from trader.strategy import SuperTrend
from trader.taxes import after_tax_summary

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
DATA = Path(cfg["data"]["dir"])
CDI = pd.read_parquet(DATA / "cdi_d1.parquet")["rate"]
CDI_N = CDI.set_axis(CDI.index.tz_convert(None).normalize())
COSTS = B3Costs.from_config(cfg)
RO = cfg["risk_overlay"]
EQ0, LOT, N = 1_000_000.0, 100, 5
ST = SuperTrend(10, 3.0)


def _pct(x, fmt="{:+.1%}"):
    return "—" if x is None or x != x else fmt.format(x)


rows = load_rows(DATA / "b3")
CAL = pd.DatetimeIndex(sorted(rows["date"].unique()))
bova = load(DATA / "bova11_d1.parquet")
bova.index = bova.index.tz_convert(None).normalize()
bova = bova[(bova["open"] > 0) & (bova["volume"] > 0)]
MGAP = (bova["open"] / bova["close"].shift(1) - 1.0).reindex(CAL).fillna(0.0)
panel = build_panel(rows, CAL, market_gap=MGAP, total_return=True)
FR = {i: f.reindex(CAL).ffill(limit=10) for i, f in panel.items()}
last_trade = {i: f.index[f["traded"]].max() for i, f in panel.items()}
code2isin = {}
for isin, f in panel.items():
    code = f["codneg"].dropna().iloc[-1]
    if code not in code2isin or last_trade[isin] > last_trade[code2isin[code]]:
        code2isin[code] = isin

# ---- cesta causal: 5 empresas mais negociadas no início de cada ano
liq = pd.DataFrame({i: f["volume"].reindex(CAL).fillna(0.0).rolling(250, min_periods=200).median()
                    for i, f in panel.items()})
ok = pd.DataFrame({i: (f["raw_close"].reindex(CAL) >= 1.0)
                   & (f["traded"].reindex(CAL).fillna(False).astype(float).rolling(5, min_periods=1).max() > 0)
                   for i, f in panel.items()}).fillna(False)
company = {i: (f["codneg"].dropna().iloc[-1])[:4] for i, f in panel.items()}
member = pd.DataFrame(0.0, index=CAL, columns=list(panel))
history = []
years = sorted(set(CAL.year))
for y in years[1:]:
    days = CAL[CAL.year == y]
    decide = CAL[CAL < days[0]][-1]
    cand = liq.loc[decide].where(ok.loc[decide]).dropna().sort_values(ascending=False)
    picks, seen = [], set()
    for isin in cand.index:
        if company[isin] in seen:
            continue
        picks.append(isin)
        seen.add(company[isin])
        if len(picks) == N:
            break
    member.loc[days, picks] = 1.0
    history.append((y, [panel[i]["codneg"].reindex(CAL).loc[:decide].dropna().iloc[-1] for i in picks]))


def champion(members: pd.DataFrame, frames: dict, win: pd.DatetimeIndex):
    cols = [c for c in members.columns if members.loc[win, c].any()]
    sig = pd.DataFrame({c: ST.generate_signals(frames[c].dropna(subset=["close"])).reindex(CAL) for c in cols}).fillna(0.0)
    vol = pd.DataFrame({c: yang_zhang(frames[c].dropna(subset=["close"]), window=RO["vol_lookback"]).reindex(CAL)
                        for c in cols})
    basket_vol = vol.where(members[cols] > 0).mean(axis=1)
    mult = (RO["target_ann_vol"] / basket_vol).clip(upper=RO["max_leverage"]).shift(1).fillna(0.0)
    tgt = (sig * members[cols] / N).reindex(win)
    res = run_target_portfolio({c: frames[c] for c in cols}, tgt, COSTS, overlay_mult=mult.reindex(win),
                               cash_rate=CDI_N, target_is_weight=True, initial_equity=EQ0, lot_size=LOT)
    eq = res.equity.set_axis(res.equity.index.tz_localize("UTC"))
    b = vs_benchmark(eq, CDI)
    return {"cagr": b["cagr_estrategia"], "alpha": b["alpha_aa"], "ir": after_tax_summary(eq, CDI)["alpha_pos_ir"],
            "sh": b["sharpe_sobre_cdi"], "dd": float((eq / eq.cummax() - 1).min()),
            "expo": res.stats["exposure_media"]}


print("=" * 100)
print("CESTA CAUSAL — 5 empresas mais negociadas no início de cada ano (uma classe por empresa)")
print("=" * 100)
for y, names in history:
    if y >= 2010:
        print(f"  {y}: {', '.join(names)}")

fixed = [code2isin[s] for s in cfg["basket"]["symbols"] if s in code2isin]
member_fixed = pd.DataFrame(0.0, index=CAL, columns=list(panel))
member_fixed[fixed] = 1.0
yahoo = {s: load(DATA / f"{s.lower()}_d1.parquet") for s in cfg["basket"]["symbols"]}
for s in yahoo:
    yahoo[s].index = yahoo[s].index.tz_convert(None).normalize()
YF = {s: f.reindex(CAL).ffill(limit=10) for s, f in yahoo.items()}
member_y = pd.DataFrame(1.0, index=CAL, columns=list(yahoo))

print("\n" + "=" * 100)
print("CAMPEÃO — mesma regra em cestas diferentes, R$1 mi, CDI no caixa")
print("=" * 100)
for lo, hi in (("2014-01-01", "2026-12-31"), ("2010-01-01", "2026-12-31")):
    win = CAL[(CAL >= lo) & (CAL <= hi)]
    print(f"\n {win[0].date()}..{win[-1].date()}")
    print(f"  {'cesta':30} {'CAGR':>6} {'alpha':>6} {'pós-IR':>7} {'Sh/CDI':>6} {'maxDD':>6} {'expo':>5}")
    rows_out = [("Yahoo fixa (as 5 de hoje)", champion(member_y, YF, win))]
    if lo >= "2014":
        rows_out.append(("B3 fixa (as 5 de hoje)", champion(member_fixed, FR, win)))
    rows_out.append(("B3 causal (5 mais negociadas)", champion(member, FR, win)))
    for name, r in rows_out:
        print(f"  {name:30} {_pct(r['cagr']):>6} {_pct(r['alpha']):>6} {_pct(r['ir']):>7} {r['sh']:>+6.2f} "
              f"{_pct(r['dd'], '{:.0%}'):>6} {r['expo']:>5.2f}")
