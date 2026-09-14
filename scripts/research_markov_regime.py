"""Regimes de Markov — o "hedge fund method" do vídeo, testado out-of-sample.

Fonte: github.com/jackson-video-resources/markov-hedge-fund-method (framework de
Roan/@RohOnChain). Estado pelo retorno de 20 pregões (±5%), matriz 3×3 por
contagem re-estimada todo dia (janela expansível, mín. 252 transições), sinal =
P(bull amanhã) − P(bear amanhã). O código do autor opera sign(sinal) long/short;
o vídeo usa a magnitude como tamanho. Testamos os dois e as versões long-only.

A) Diagnóstico — a persistência da matriz é do mercado ou da janela sobreposta?
   O estado de hoje prevê o RETORNO de amanhã (que é o que paga)?
B) Backtest na cesta: escolha de variante em 2010–2015, OOS 2015–2026, CDI no
   caixa, custos B3, aluguel de 2% a.a. no short. Referências: buy & hold, campeão,
   momentum 20d com banda de 5% (equivalência) e HMM gaussiano de 2 estados
   (Hamilton 1989 — o regime de Markov da literatura).

    python scripts/research_markov_regime.py [--refit-hmm]
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
from trader.costs.b3 import B3Costs
from trader.data.store import load
from trader.regime import (BEAR, BULL, SIDE, calm_probability_next, expanding_matrices,
                           label_states, markov_signal, transition_matrix)
from trader.risk.overlay import vol_target_multiplier
from trader.strategy import Momentum, SuperTrend
from trader.taxes import after_tax_summary

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
DATA = Path(cfg["data"]["dir"])
SYMS = cfg["basket"]["symbols"]
EQ0, LOT = 1_000_000.0, cfg["backtest"]["lot_size"]
COSTS = B3Costs.from_config(cfg)
BORROW = 0.02
RO = cfg["risk_overlay"]
STATES = ((BEAR, "bear"), (SIDE, "lateral"), (BULL, "bull"))

basket = {s: load(DATA / f"{s.lower()}_d1.parquet") for s in SYMS}
closes = {s: f["close"] for s, f in basket.items()}
CDI = pd.read_parquet(DATA / "cdi_d1.parquet")["rate"]

IDX = None
for f in basket.values():
    IDX = f.index if IDX is None else IDX.intersection(f.index)
IDX = IDX.sort_values()
OOS_START = IDX[int(len(IDX) * 0.40)]
SEL_IDX = IDX[(IDX >= pd.Timestamp("2010-01-04", tz=IDX.tz)) & (IDX < OOS_START)]
OOS_IDX = IDX[IDX >= OOS_START]


def _p(x, fmt="{:+.1%}"):
    return "—" if x is None or x != x else fmt.format(x)


# ----------------------------------------------------------------------------- A
print("=" * 88)
print("A) DIAGNÓSTICO — toda a história de cada ação")
print("=" * 88)
print("\nDiagonal da matriz (bear, lateral, bull) = a tal 'persistência':")
print(f"{'':7} {'real':>20} {'retornos embaralhados':>24} {'sem sobreposição':>20}")
rng = np.random.default_rng(7)
for s in SYMS:
    c = closes[s]
    st = label_states(c)
    r = np.log(c).diff().dropna().to_numpy()
    shuf = [np.diag(transition_matrix(label_states(pd.Series(
        np.exp(np.r_[0.0, np.cumsum(rng.permutation(r))]), index=c.index))))
        for _ in range(20)]
    d_real = np.diag(transition_matrix(st))
    d_shuf = np.mean(shuf, axis=0)
    d_noov = np.diag(transition_matrix(st.iloc[::20]))
    f = lambda d: " ".join(f"{v:.2f}" for v in d)
    print(f"{s:7} {f(d_real):>20} {f(d_shuf):>24} {f(d_noov):>20}")
print("  embaralhado = mesma distribuição de retornos, ordem aleatória (zero regime).")
print("  sem sobreposição = estado a cada 20 pregões (janelas disjuntas).")

P = transition_matrix(label_states(closes["PETR4"]))
print("\nMatriz PETR4 (linha = hoje, coluna = amanhã):")
print(f"{'':9}" + "".join(f"{n:>9}" for _, n in STATES))
for k, n in STATES:
    print(f"{n:9}" + "".join(f"{P[k, j]:>9.2f}" for j, _ in STATES))

hit_mat, hit_naive, tot = 0, 0, 0
p1, p20 = [], []
print("\nMapa estado -> sinal P(bull)-P(bear) no último dia, e % dos dias OOS em cada estado:")
for s in SYMS:
    c = closes[s]
    st = label_states(c)
    Pt, n = expanding_matrices(st)
    sv = st.to_numpy()
    oos = np.nonzero((c.index >= OOS_START)[:-1] & ~np.isnan(sv[:-1]) & ~np.isnan(sv[1:]))[0]
    cur = sv[oos].astype(int)
    hit_mat += int((Pt[oos, cur].argmax(axis=1) == sv[oos + 1]).sum())
    hit_naive += int((cur == sv[oos + 1]).sum())
    tot += len(oos)
    sig = Pt[-1][:, BULL] - Pt[-1][:, BEAR]
    share = st[st.index >= OOS_START].value_counts(normalize=True)
    print(f"  {s:6} sinal: bear {sig[BEAR]:+.2f} | lateral {sig[SIDE]:+.2f} | bull {sig[BULL]:+.2f}"
          f"   dias: bear {share.get(BEAR, 0):.0%} lateral {share.get(SIDE, 0):.0%} "
          f"bull {share.get(BULL, 0):.0%}")
    d = pd.DataFrame({"st": st, "r1": c.shift(-1) / c - 1.0, "r20": c.shift(-20) / c - 1.0})
    d = d[d.index >= OOS_START].dropna()
    p1.append(d[["st", "r1"]])
    p20.append(d[["st", "r20"]].iloc[::20])
print(f"\nAcerto do ESTADO de amanhã (OOS, 5 ações): matriz {hit_mat / tot:.1%} | "
      f"'amanhã = hoje' {hit_naive / tot:.1%}")

p1, p20 = pd.concat(p1), pd.concat(p20)
print("\nO que o estado de hoje prevê de RETORNO (OOS 2015–2026, 5 ações juntas):")
print(f"{'estado':9} {'n':>6} {'dia seguinte':>13} {'t':>6} {'n':>5} {'próx. 20 dias':>14} {'t':>6}")


def _row(name, a, b):
    t1 = a.mean() / (a.std() / np.sqrt(len(a))) if len(a) > 2 else np.nan
    t2 = b.mean() / (b.std() / np.sqrt(len(b))) if len(b) > 2 else np.nan
    print(f"{name:9} {len(a):>6} {a.mean() * 1e4:>+10.1f}bps {t1:>6.2f} {len(b):>5} "
          f"{b.mean():>+13.2%} {t2:>6.2f}")


for k, n in STATES:
    _row(n, p1.loc[p1["st"] == k, "r1"], p20.loc[p20["st"] == k, "r20"])
_row("todos", p1["r1"], p20["r20"])

# ----------------------------------------------------------------------------- B
print("\n" + "=" * 88)
print(f"B) BACKTEST NA CESTA — escolha {SEL_IDX[0].date()}..{SEL_IDX[-1].date()} | "
      f"OOS {OOS_IDX[0].date()}..{OOS_IDX[-1].date()}")
print(f"   R$1M, CDI no caixa, custos B3, aluguel {BORROW:.0%} a.a. no short")
print("=" * 88)

cache = DATA / "hmm_p_calmo.parquet"
last = min(c.index[-1] for c in closes.values())
if cache.exists() and "--refit-hmm" not in sys.argv and pd.read_parquet(cache).index[-1] >= last:
    HMM = pd.read_parquet(cache)
else:
    print("ajustando HMM (walk-forward anual, 5 ações)...", flush=True)
    HMM = pd.DataFrame({s: calm_probability_next(closes[s], start_year=2010,
                                                 min_train_bars=400).reindex(IDX)
                        for s in SYMS})
    HMM.to_parquet(cache)

YZ = {tv: vol_target_multiplier(basket, target_ann_vol=tv, lookback=RO["vol_lookback"],
                                max_leverage=RO["max_leverage"], estimator="yang_zhang")
      for tv in (RO["target_ann_vol"], 0.25, 0.35)}
TV0 = RO["target_ann_vol"]


def _tgt(fn) -> pd.DataFrame:
    return pd.DataFrame({s: fn(s).reindex(IDX) for s in SYMS}).fillna(0.0)


TG = {
    "st": _tgt(lambda s: SuperTrend(10, 3.0).generate_signals(basket[s])),
    "ls_sign": _tgt(lambda s: markov_signal(closes[s], sizing="sign")),
    "ls_prob": _tgt(lambda s: markov_signal(closes[s], sizing="prob")),
    "lo_sign": _tgt(lambda s: markov_signal(closes[s], sizing="sign", long_only=True)),
    "lo_prob": _tgt(lambda s: markov_signal(closes[s], sizing="prob", long_only=True)),
    "lo2_sign": _tgt(lambda s: markov_signal(closes[s], threshold=0.02, sizing="sign",
                                             long_only=True)),
    "mom20": _tgt(lambda s: Momentum(lookback=20, band=0.05).generate_signals(basket[s])),
    "hmm": HMM.fillna(0.0),
}

r20 = pd.DataFrame({s: (closes[s] / closes[s].shift(20) - 1.0).reindex(IDX) for s in SYMS})
live = r20.notna() & (TG["ls_sign"] != 0)
print("\nEquivalência (dias OOS com sinal):")
o = live.index >= OOS_START
m = live[o].to_numpy()
print(f"  markov long 'sinal' == (r20 >= -5%):        "
      f"{((TG['lo_sign'][o] > 0) == (r20[o] >= -0.05)).to_numpy()[m].mean():.1%}")
print(f"  markov long 'prob' > 0,5 == (r20 > +5%):     "
      f"{((TG['lo_prob'][o] > 0.5) == (r20[o] > 0.05)).to_numpy()[m].mean():.1%}")

CONFIGS = [
    ("campeão SuperTrend +YZ (ref)", "st", TV0),
    ("campeão vol-alvo 0,25 (ref)", "st", 0.25),
    ("campeão vol-alvo 0,35 (ref)", "st", 0.35),
    ("markov L/S sinal (autor)", "ls_sign", None),
    ("markov L/S prob (vídeo)", "ls_prob", None),
    ("markov L/S sinal +YZ", "ls_sign", TV0),
    ("markov long sinal", "lo_sign", None),
    ("markov long prob", "lo_prob", None),
    ("markov long sinal +YZ", "lo_sign", TV0),
    ("markov long prob +YZ", "lo_prob", TV0),
    ("markov ±2% long sinal +YZ", "lo2_sign", TV0),
    ("momentum 20d banda 5% +YZ", "mom20", TV0),
    ("HMM P(calmo)", "hmm", None),
    ("HMM P(calmo) +YZ", "hmm", TV0),
]


def evaluate(key, tv, win):
    res = run_target_portfolio(basket, TG[key].reindex(win), COSTS, initial_equity=EQ0,
                               lot_size=LOT,
                               overlay_mult=YZ[tv].reindex(win) if tv is not None else None,
                               cash_rate=CDI, borrow_rate_aa=BORROW)
    b = vs_benchmark(res.equity, CDI)
    return {"sh": b["sharpe_sobre_cdi"], "cagr": b["cagr_estrategia"], "alpha": b["alpha_aa"],
            "alpha_ir": after_tax_summary(res.equity, CDI)["alpha_pos_ir"],
            "dd": res.stats["max_drawdown"], "expo": res.stats["exposure_media"],
            "trades": res.stats["n_trades"]}


def buy_hold(win):
    n = pd.DataFrame({s: closes[s].reindex(win) for s in SYMS})
    eq = (n / n.iloc[0]).mean(axis=1) * EQ0
    b = vs_benchmark(eq, CDI)
    return {"sh": b["sharpe_sobre_cdi"], "cagr": b["cagr_estrategia"], "alpha": b["alpha_aa"],
            "alpha_ir": None, "dd": summarize(eq, pd.DataFrame(columns=["fee"]), EQ0)["max_drawdown"],
            "expo": 1.0, "trades": 5}


cdi_oos = vs_benchmark(pd.Series(EQ0, index=OOS_IDX), CDI)["cagr_cdi"]
print(f"\nCDI no OOS: {cdi_oos:.1%} a.a.")
print(f"{'config':30} {'Sh/CDI':>7} | {'CAGR':>6} {'alpha':>6} {'pós-IR':>7} {'Sh/CDI':>7} "
      f"{'maxDD':>6} {'expo':>5} {'trades':>6}")
print(f"{'':30} {'10–15':>7} | {'— OOS 2015–2026 —':^52}")
print("-" * 88)
rows = [("buy & hold cesta", buy_hold(SEL_IDX), buy_hold(OOS_IDX))]
rows += [(name, evaluate(k, tv, SEL_IDX), evaluate(k, tv, OOS_IDX)) for name, k, tv in CONFIGS]
for name, sel, oos in rows:
    print(f"{name:30} {_p(sel['sh'], '{:+.2f}'):>7} | {_p(oos['cagr']):>6} {_p(oos['alpha']):>6} "
          f"{_p(oos['alpha_ir']):>7} {_p(oos['sh'], '{:+.2f}'):>7} {_p(oos['dd'], '{:.0%}'):>6} "
          f"{oos['expo']:>5.2f} {oos['trades']:>6}")

mk = [r for r in rows if r[0].startswith("markov")]
pick = max(mk, key=lambda r: r[1]["sh"] if r[1]["sh"] is not None else -9)
print(f"\nEscolhida na janela 2010–15 (melhor Sh/CDI entre as markov): {pick[0]} "
      f"({_p(pick[1]['sh'], '{:+.2f}')}) -> OOS Sh/CDI {_p(pick[2]['sh'], '{:+.2f}')}, "
      f"alpha pós-IR {_p(pick[2]['alpha_ir'])}, maxDD {_p(pick[2]['dd'], '{:.0%}')}")
