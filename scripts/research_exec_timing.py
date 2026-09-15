"""Campeão executando no leilão de fechamento em vez da abertura seguinte.

Motivação (§20b): na B3 a noite rende e o pregão perde na média. O campeão hoje
decide no fechamento e executa tudo na abertura seguinte — cada compra perde a
noite que vem logo depois do sinal.

Variantes (mesmo alvo: SuperTrend(10,3) × vol-target Yang-Zhang / 5 ações):
  A  atual — compra e venda na abertura seguinte
  B  tudo no leilão de fechamento, com o próprio fechamento no sinal (ideal: não se
     conhece o fechamento a tempo)
  C  compra no leilão de fechamento (ideal), venda na abertura seguinte
  D  venda na abertura seguinte, compra no leilão de fechamento do dia SEGUINTE
     (causal só com dado diário)
  F  espelho de C: compra na abertura seguinte, venda no fechamento do dia seguinte
  E  (só no MT5, 2021–2026) compra no leilão de fechamento com o sinal recalculado
     pelo que se sabia às 16h50/17h50 — o jeito real de fazer C

Parte 1: Yahoo 2008–2026. Parte 2: MT5 (preço bruto, sem proventos — só serve para
comparar as variantes entre si) com barras de 5 minutos.

Pré-requisitos: pull_basket.py, pull_cdi.py, pull_mt5_intraday.py D1 M5 --symbols ...
e o M1 da PETR4 (horário de fechamento do à vista).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml

from trader.backtest.exec_timing import run_exec_timing
from trader.backtest.metrics import vs_benchmark
from trader.costs.b3 import B3Costs
from trader.data.store import load
from trader.indicators import yang_zhang
from trader.intraday import BarIndex, cash_close_minute
from trader.risk.overlay import vol_target_multiplier
from trader.strategy import SuperTrend
from trader.taxes import after_tax_summary

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
DATA = Path(cfg["data"]["dir"])
CDI = pd.read_parquet(DATA / "cdi_d1.parquet")["rate"]
EQ0, LOT = 1_000_000.0, cfg["backtest"]["lot_size"]
COSTS = B3Costs.from_config(cfg)
RO = cfg["risk_overlay"]
SYMS = cfg["basket"]["symbols"]
N = len(SYMS)
ST = SuperTrend(10, 3.0)

VARIANTS = [("A atual: tudo na abertura seguinte", "open_next", "open_next"),
            ("B tudo no fechamento (ideal)", "close_same", "close_same"),
            ("C compra fech. / vende abertura (ideal)", "close_same", "open_next"),
            ("D compra fech. dia seguinte / vende abert.", "close_next", "open_next"),
            ("F espelho: compra abert. / vende fech.", "open_next", "close_next")]


def _pct(x, fmt="{:+.1%}"):
    return "—" if x is None or x != x else fmt.format(x)


def weights_from(frames: dict, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Peso-alvo decidido com dados até o fechamento de t."""
    sig = pd.DataFrame({s: ST.generate_signals(frames[s]).reindex(idx) for s in SYMS}).fillna(0.0)
    mult = vol_target_multiplier({s: frames[s] for s in SYMS}, target_ann_vol=RO["target_ann_vol"],
                                 lookback=RO["vol_lookback"], max_leverage=RO["max_leverage"],
                                 estimator="yang_zhang")
    mult_t = mult.shift(-1).reindex(idx).ffill().fillna(0.0)          # overlay devolve com shift(1)
    return sig.mul(mult_t, axis=0) / N


def stats(res, win_years: float) -> dict:
    eq = res.equity
    b = vs_benchmark(eq, CDI)
    return {"cagr": b["cagr_estrategia"], "alpha": b["alpha_aa"],
            "ir": after_tax_summary(eq, CDI)["alpha_pos_ir"], "sh": b["sharpe_sobre_cdi"],
            "dd": float((eq / eq.cummax() - 1.0).min()), "ordens": res.stats["n_ordens"] / win_years,
            "expo": res.stats["exposure_media"]}


def years(idx):
    return (idx[-1] - idx[0]) / pd.Timedelta(days=365.25)


# ================================================================== PARTE 1: Yahoo
yahoo = {s: load(DATA / f"{s.lower()}_d1.parquet") for s in SYMS}
idx = None
for f in yahoo.values():
    idx = f.index if idx is None else idx.intersection(f.index)
idx = idx.sort_values()
W = weights_from(yahoo, idx)
cut = idx[int(len(idx) * 0.40)]
WINS = {"2008–2015": idx[idx < cut], "2015–2026": idx[idx >= cut]}

print("=" * 104)
print(f"PARTE 1 — Yahoo, cesta de 5, CDI no caixa, custo B3 | {idx[0].date()}..{idx[-1].date()}")
print("=" * 104)
print(f"{'variante':44} | {'2008–15':^15} | {'2015–2026':^46}")
print(f"{'':44} | {'alpha':>7} {'Sh/CDI':>6} | {'CAGR':>6} {'alpha':>6} {'pós-IR':>7} {'Sh/CDI':>6} "
      f"{'maxDD':>6} {'ordens/a':>8}")
print("-" * 104)
results1 = {}
for name, buy, sell in VARIANTS:
    row = {}
    for wname, win in WINS.items():
        r = run_exec_timing(yahoo, W.reindex(win), COSTS, buy_at=buy, sell_at=sell, cash_rate=CDI,
                            initial_equity=EQ0, lot_size=LOT)
        row[wname] = (r, stats(r, years(win)))
    results1[name] = row
    a, b = row["2008–2015"][1], row["2015–2026"][1]
    print(f"{name:44} | {_pct(a['alpha']):>7} {a['sh'] or 0:>+6.2f} | {_pct(b['cagr']):>6} {_pct(b['alpha']):>6} "
          f"{_pct(b['ir']):>7} {b['sh'] or 0:>+6.2f} {_pct(b['dd'], '{:.0%}'):>6} {b['ordens']:>8.0f}")

base = results1[VARIANTS[0][0]]["2015–2026"][0].equity
for name, *_ in VARIANTS[2:4]:
    eq = results1[name]["2015–2026"][0].equity
    diff = ((1 + eq.pct_change()).groupby(eq.index.year).prod() - (1 + base.pct_change()).groupby(base.index.year).prod())
    print(f"  {name[:1]} menos A, por ano: " + " ".join(f"{y % 100:02d}:{v:+.1%}" for y, v in diff.items()))

# ================================================================== PARTE 2: MT5 + pré-fechamento
print("\n" + "=" * 104)
print("PARTE 2 — MT5 (preço bruto), sinal de compra recalculado com o que se sabia antes do leilão")
print("=" * 104)
mt5 = {}
for s in SYMS:
    f = load(DATA / f"{s.lower()}_mt5_d1.parquet")
    f.index = f.index.normalize()
    mt5[s] = f
idx2 = None
for f in mt5.values():
    idx2 = f.index if idx2 is None else idx2.intersection(f.index)
idx2 = idx2.sort_values()
W2 = weights_from(mt5, idx2)

CC = cash_close_minute(load(DATA / "petr4_mt5_m1.parquet").index)
partial = {}
for s in SYMS:
    m5 = load(DATA / f"{s.lower()}_mt5_m5.parquet")
    bi = BarIndex(m5.index)
    cutoff = CC.reindex(bi.day).to_numpy(dtype="float64") - 9          # barra das 16h45 fecha 16h50
    m = bi.minute <= cutoff
    g = pd.DataFrame({"day": bi.day[m], "high": m5["high"].to_numpy()[m], "low": m5["low"].to_numpy()[m],
                      "close": m5["close"].to_numpy()[m]})
    agg = g.groupby("day").agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
    agg.index = pd.to_datetime(agg.index.astype(str), format="%Y%m%d").tz_localize("UTC")
    partial[s] = agg

HIST = 250
sig_pre = pd.DataFrame(0.0, index=idx2, columns=SYMS)
sig_chk = pd.DataFrame(0.0, index=idx2, columns=SYMS)
yz_pre = pd.DataFrame(np.nan, index=idx2, columns=SYMS)
for s in SYMS:
    d1 = mt5[s].reindex(idx2)
    pa = partial[s].reindex(idx2)
    for k in range(30, len(idx2)):
        hist = d1.iloc[max(0, k - HIST):k]
        today = d1.iloc[[k]].copy()
        sig_chk.iloc[k, SYMS.index(s)] = ST.generate_signals(pd.concat([hist, today])).iloc[-1]
        if pa.iloc[k].isna().any():
            continue
        today[["high", "low", "close"]] = pa.iloc[k][["high", "low", "close"]].to_numpy()
        today["high"] = max(today["high"].iloc[0], today["open"].iloc[0])
        today["low"] = min(today["low"].iloc[0], today["open"].iloc[0])
        frame = pd.concat([hist, today])
        sig_pre.iloc[k, SYMS.index(s)] = ST.generate_signals(frame).iloc[-1]
        yz_pre.iloc[k, SYMS.index(s)] = yang_zhang(frame, window=RO["vol_lookback"]).iloc[-1]
mult_pre = (RO["target_ann_vol"] / yz_pre.mean(axis=1)).clip(upper=RO["max_leverage"]).fillna(0.0)
W_pre = sig_pre.mul(mult_pre, axis=0) / N
sig_full = pd.DataFrame({s: ST.generate_signals(mt5[s]).reindex(idx2) for s in SYMS}).fillna(0.0)
live = idx2[60:]
print(f"checagem: sinal com {HIST} pregões de histórico = sinal com histórico completo em "
      f"{(sig_chk.loc[live] == sig_full.loc[live]).to_numpy().mean():.1%} dos casos")
print(f"sinal às 16h50 igual ao do fechamento em {(sig_pre.loc[live] == sig_full.loc[live]).to_numpy().mean():.1%} "
      f"dos casos")

VAR2 = [("A atual: tudo na abertura seguinte", "open_next", "open_next", None),
        ("C compra fech. / vende abertura (ideal)", "close_same", "open_next", None),
        ("D compra fech. dia seguinte / vende abert.", "close_next", "open_next", None),
        ("E compra fech. c/ sinal das 16h50 / vende abert.", "close_same", "open_next", W_pre)]
yrs2 = years(idx2)
print(f"\n{idx2[0].date()}..{idx2[-1].date()}")
print(f"{'variante':50} | {'CAGR':>6} {'alpha':>6} {'pós-IR':>7} {'Sh/CDI':>6} {'maxDD':>6} {'ordens/a':>8}")
print("-" * 96)
res2 = {}
for name, buy, sell, cw in VAR2:
    r = run_exec_timing(mt5, W2, COSTS, buy_at=buy, sell_at=sell, cash_rate=CDI, initial_equity=EQ0,
                        lot_size=LOT, close_same_weights=cw)
    st = stats(r, yrs2)
    res2[name] = r
    print(f"{name:50} | {_pct(st['cagr']):>6} {_pct(st['alpha']):>6} {_pct(st['ir']):>7} "
          f"{st['sh'] or 0:>+6.2f} {_pct(st['dd'], '{:.0%}'):>6} {st['ordens']:>8.0f}")
base2 = res2[VAR2[0][0]].equity
eqE = res2[VAR2[3][0]].equity
diff2 = ((1 + eqE.pct_change()).groupby(eqE.index.year).prod() - (1 + base2.pct_change()).groupby(base2.index.year).prod())
print("  E menos A, por ano: " + " ".join(f"{y}:{v:+.1%}" for y, v in diff2.items()))
