"""Arbitragem entre classes de ação da mesma empresa (ON x PN, unit x PN) na B3.

Spread = log(A) − log(B) com preços ajustados por proventos, hedge 1:1 em valor.
z-score móvel do spread; entra além de ±entry, sai ao voltar para dentro de
±exit, stop em |z| > 4 e prazo máximo de 60 ou 250 pregões. Carteira com até 8 pares
abertos, exposição bruta ≤ 100%, neutra em valor (o patrimônio fica no CDI),
execução na abertura seguinte, custo por ponta e aluguel no lado vendido.

Dois universos definidos antes de rodar:
  líquido: R$5 mi/dia nas duas pontas, 8 bps por ponta, aluguel 2% a.a.
  amplo:   R$1 mi/dia nas duas pontas, 20 bps por ponta, aluguel 4% a.a.
Parâmetros (janela do z, entrada, saída, prazo) escolhidos no TREINO (40% iniciais desde
2006), medidos no TESTE; depois walk-forward rolante (treino 3 anos, teste 1 ano).

Pré-requisitos: pull_share_classes.py, pull_basket.py, pull_cdi.py, pull_futures_proxy.py.
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml

from trader.backtest.metrics import vs_benchmark
from trader.backtest.ml_portfolio import run_target_portfolio
from trader.costs.b3 import B3Costs
from trader.data.store import load
from trader.pairs import (adf_tstat, build_weights, eligibility, half_life, log_spread,
                          pair_positions, pairs_portfolio, rolling_z)
from trader.risk.overlay import vol_target_multiplier
from trader.strategy import SuperTrend
from trader.taxes import after_tax_summary

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
DATA = Path(cfg["data"]["dir"])
CDI = pd.read_parquet(DATA / "cdi_d1.parquet")["rate"]
EQ0 = 1_000_000.0
EXCLUDE = {"OIBR3/OIBR4": "Oi em recuperação judicial, dados com saltos",
           "VALE3/VALE5": "VALE5 extinta em 2017, ajuste do Yahoo inconsistente"}
UNIVERSES = {"líquido": dict(min_value=5e6, cost=0.0008, borrow=0.02),
             "amplo": dict(min_value=1e6, cost=0.0020, borrow=0.04)}
# prazo 250 e janela 250: no TREINO a meia-vida dos pares ON/PN líquidos deu 100–230 pregões
GRID = [dict(lookback=L, entry=e, exit_z=x, max_hold=h) for L, e, x, h in
        product((20, 60, 120, 250), (1.5, 2.0, 2.5), (0.0, 0.5), (60, 250))]
SLOTS, STOP = 8, 4.0

legs, frames = {}, {}
for a, b in cfg["share_classes"]["pairs"]:
    name = f"{a}/{b}"
    pa, pb = DATA / f"{a.lower()}_d1.parquet", DATA / f"{b.lower()}_d1.parquet"
    if name in EXCLUDE or not (pa.exists() and pb.exists()):
        continue
    frames.setdefault(a, load(pa))
    frames.setdefault(b, load(pb))
    legs[name] = (a, b)

IDX = frames["PETR4"].index[frames["PETR4"].index >= pd.Timestamp("2006-01-02", tz="UTC")]
OPENS = pd.DataFrame({s: f["open"].reindex(IDX) for s, f in frames.items()})
CLOSES = pd.DataFrame({s: f["close"].reindex(IDX) for s, f in frames.items()})
CUT = IDX[int(len(IDX) * 0.40)]
TRAIN, TEST = IDX[IDX < CUT], IDX[IDX >= CUT]

spread, elig = {}, {u: {} for u in UNIVERSES}
for p, (a, b) in legs.items():
    A, B = frames[a], frames[b]
    common = A.index.intersection(B.index)
    spread[p] = log_spread(A["close"].reindex(common), B["close"].reindex(common))
    for u, par in UNIVERSES.items():
        elig[u][p] = eligibility(A, B, min_traded_value=par["min_value"]).reindex(IDX).fillna(False)


def _pct(x, fmt="{:+.1%}"):
    return "—" if x is None or x != x else fmt.format(x)


# ------------------------------------------------------------------ A) diagnóstico
print("=" * 100)
print(f"A) PARES — {len(legs)} com dados (excluídos: {', '.join(EXCLUDE)}); dias desde 2006")
print("=" * 100)
print(f"{'par':14} {'elegível líq.':>13} {'amplo':>6} | {'prêmio médio':>12} {'desvio':>7} | "
      f"{'ADF t':>6} {'meia-vida':>10}")
for p in legs:
    s = spread[p][spread[p].index >= IDX[0]]
    ok = elig["amplo"][p]
    s_ok = s.reindex(IDX)[ok]
    adf = adf_tstat(s_ok) if len(s_ok) > 250 else float("nan")
    hl = half_life(s_ok) if len(s_ok) > 250 else float("nan")
    print(f"{p:14} {elig['líquido'][p].mean():>13.0%} {ok.mean():>6.0%} | "
          f"{np.expm1(s.mean()):>+12.0%} {s.std():>7.2f} | {adf:>6.1f} {hl:>8.0f} d")
print("  prêmio = quanto A vale a mais que B em média; ADF < -2,86 = spread estacionário (5%).")
print("  ADF e meia-vida só nos dias do universo amplo — juntar trechos soltos infla a reversão.")


# ------------------------------------------------------------------ B) grid
def run(u: str, p: dict):
    par = UNIVERSES[u]
    pos, zs = {}, {}
    for name in legs:
        z = rolling_z(spread[name], p["lookback"]).reindex(IDX).to_numpy()
        pos[name] = pair_positions(z, elig[u][name].to_numpy(), entry=p["entry"],
                                   exit_z=p["exit_z"], stop_z=STOP, max_hold=p["max_hold"])
        zs[name] = z
    W, held = build_weights(pos, zs, legs, IDX, slots=SLOTS)
    return W, held, pairs_portfolio(W, held, OPENS, CLOSES, legs, CDI, cost_side=par["cost"],
                                    borrow_aa=par["borrow"], equity0=EQ0)


def window(res, W, held, win, extra_excess=None):
    ex = res.excess.reindex(win)
    if extra_excess is not None:
        ex = ex + extra_excess.reindex(win).fillna(0.0)
    rate = CDI.reindex(win).ffill().fillna(0.0)
    eq = EQ0 * (1.0 + rate + ex).cumprod()
    b = vs_benchmark(eq, CDI)
    h = held.reindex(win).shift(1).fillna(0.0)
    years = (win[-1] - win[0]) / pd.Timedelta(days=365.25)
    return {"sh": b["sharpe_sobre_cdi"], "alpha": b["alpha_aa"],
            "alpha_ir": after_tax_summary(eq, CDI)["alpha_pos_ir"],
            "dd": float((eq / eq.cummax() - 1.0).min()),
            "expo": float(W.reindex(win).abs().sum(axis=1).mean()),
            "trades": float(((h != 0) & (h.shift(1).fillna(0.0) != h)).to_numpy().sum() / years),
            "eq": eq, "ex": ex}


print("\n" + "=" * 100)
print(f"B) TREINO {TRAIN[0].date()}..{TRAIN[-1].date()} | TESTE {TEST[0].date()}..{TEST[-1].date()}")
print(f"   até {SLOTS} pares abertos, bruto <= 100%, stop |z|>{STOP:.0f}; prazo máximo 60 ou 250 pregões")
print("=" * 100)
cache, chosen = {}, {}
for u in UNIVERSES:
    for i, p in enumerate(GRID):
        cache[(u, i)] = run(u, p)
    scores = [(window(cache[(u, i)][2], *cache[(u, i)][:2], TRAIN)["sh"] or -9.0, i)
              for i in range(len(GRID))]
    chosen[u] = max(scores)[1]

print(f"{'universo':9} {'escolhido no treino':28} {'treino':>6} | {'alpha':>6} {'pós-IR':>7} "
      f"{'Sh/CDI':>6} {'maxDD':>6} {'expo':>5} {'trades/a':>8}")
print(f"{'':9} {'':28} {'Sh/CDI':>6} | {'— TESTE —':^45}")
for u in UNIVERSES:
    i = chosen[u]
    W, held, res = cache[(u, i)]
    tr, te = window(res, W, held, TRAIN), window(res, W, held, TEST)
    ptxt = f"L={GRID[i]['lookback']} e={GRID[i]['entry']} s={GRID[i]['exit_z']} prazo={GRID[i]['max_hold']}"
    print(f"{u:9} {ptxt:28} {tr['sh'] or 0:>+6.2f} | {_pct(te['alpha']):>6} {_pct(te['alpha_ir']):>7} "
          f"{te['sh'] or 0:>+6.2f} {_pct(te['dd'], '{:.0%}'):>6} {te['expo']:>5.2f} {te['trades']:>8.1f}")

print("\nGrid inteiro no TESTE (só para ver estabilidade, não vale como resultado):")
for u in UNIVERSES:
    shs = [window(cache[(u, i)][2], *cache[(u, i)][:2], TEST)["sh"] or 0.0 for i in range(len(GRID))]
    print(f"  {u:9} Sh/CDI no teste: mediana {np.median(shs):+.2f} | min {min(shs):+.2f} | "
          f"max {max(shs):+.2f} | positivos {sum(s > 0 for s in shs)}/{len(shs)}")

print("\nContribuição por par no TESTE (config escolhida, antes de custo e aluguel, % do patrimônio):")
for u in UNIVERSES:
    W, held, res = cache[(u, chosen[u])]
    c = res.contrib.reindex(TEST)
    by = pd.Series({p: float(c[[a, b]].sum(axis=1).sum()) for p, (a, b) in legs.items()})
    by = by[by.abs() > 1e-4].sort_values(ascending=False)
    print(f"  {u:9} " + " | ".join(f"{p} {v:+.1%}" for p, v in by.items()))

# ------------------------------------------------------------------ C) walk-forward rolante
TR_D, TE_D = 756, 252
print("\n" + "=" * 100)
print(f"C) WALK-FORWARD ROLANTE — re-escolhe a cada {TE_D} pregões com os {TR_D} anteriores")
print("=" * 100)
wf_ex = {}
for u in UNIVERSES:
    parts, picks = [], []
    for s in range(TR_D, len(IDX) - 20, TE_D):
        tr_w, te_w = IDX[s - TR_D:s], IDX[s:s + TE_D]
        best = max(range(len(GRID)),
                   key=lambda i: window(cache[(u, i)][2], *cache[(u, i)][:2], tr_w)["sh"] or -9.0)
        picks.append(best)
        parts.append(cache[(u, best)][2].excess.reindex(te_w))
    ex = pd.concat(parts)
    wf_ex[u] = ex
    rate = CDI.reindex(ex.index).ffill().fillna(0.0)
    eq = EQ0 * (1.0 + rate + ex).cumprod()
    b = vs_benchmark(eq, CDI)
    changes = sum(1 for x, y in zip(picks, picks[1:]) if x != y)
    print(f"  {u:9} {ex.index[0].date()}..{ex.index[-1].date()} | alpha {_pct(b['alpha_aa'])} | "
          f"pós-IR {_pct(after_tax_summary(eq, CDI)['alpha_pos_ir'])} | Sh/CDI "
          f"{b['sharpe_sobre_cdi'] or 0:+.2f} | maxDD {_pct(float((eq / eq.cummax() - 1).min()), '{:.0%}')} | "
          f"trocou de parâmetro {changes} em {len(picks)} janelas")
    yearly = (1.0 + ex).groupby(ex.index.year).prod() - 1.0
    print("            excesso sobre o CDI por ano: " + " ".join(f"{y}:{v:+.1%}" for y, v in yearly.items()))

# ------------------------------------------------------------------ D) custo, campeão, Ibov
print("\n" + "=" * 100)
print("D) SENSIBILIDADE A CUSTO, CORRELAÇÃO E COMBINAÇÃO COM O CAMPEÃO (TESTE)")
print("=" * 100)
basket = {s: load(DATA / f"{s.lower()}_d1.parquet") for s in cfg["basket"]["symbols"]}
bidx = None
for f in basket.values():
    bidx = f.index if bidx is None else bidx.intersection(f.index)
TE_B = bidx[bidx >= TEST[0]]
yz = vol_target_multiplier(basket, target_ann_vol=cfg["risk_overlay"]["target_ann_vol"],
                           lookback=cfg["risk_overlay"]["vol_lookback"],
                           max_leverage=cfg["risk_overlay"]["max_leverage"], estimator="yang_zhang")
st = pd.DataFrame({s: SuperTrend(10, 3.0).generate_signals(f).reindex(bidx) for s, f in basket.items()}).fillna(0.0)
champ = run_target_portfolio(basket, st.reindex(TE_B), B3Costs.from_config(cfg), initial_equity=EQ0,
                             lot_size=cfg["backtest"]["lot_size"], overlay_mult=yz.reindex(TE_B),
                             cash_rate=CDI).equity
champ_ex = (champ.pct_change() - CDI.reindex(champ.index).ffill().fillna(0.0)).dropna()
ibov = load(DATA / "ibov_d1.parquet")["close"].pct_change()

for u in UNIVERSES:
    i = chosen[u]
    W, held, _ = cache[(u, i)]
    print(f"  {u}: L={GRID[i]['lookback']} entra={GRID[i]['entry']} sai={GRID[i]['exit_z']} prazo={GRID[i]['max_hold']}")
    for label, cost, borrow in (("otimista", 0.0004, 0.01), ("base", UNIVERSES[u]["cost"], UNIVERSES[u]["borrow"]),
                                ("pessimista", UNIVERSES[u]["cost"] + 0.0008, UNIVERSES[u]["borrow"] + 0.02)):
        r = pairs_portfolio(W, held, OPENS, CLOSES, legs, CDI, cost_side=cost, borrow_aa=borrow, equity0=EQ0)
        w = window(r, W, held, TEST)
        print(f"     custo {label:10} ({cost * 1e4:.0f} bps/ponta, aluguel {borrow:.0%}): alpha {_pct(w['alpha'])}"
              f" | pós-IR {_pct(w['alpha_ir'])} | Sh/CDI {w['sh'] or 0:+.2f}")
    ex = cache[(u, i)][2].excess.reindex(champ_ex.index).fillna(0.0)
    both = pd.concat([ex, champ_ex, ibov.reindex(champ_ex.index)], axis=1, sort=True).dropna()
    c_ch, c_ib = both.corr().iloc[0, 1], both.corr().iloc[0, 2]
    combo = window(cache[(u, i)][2], W, held, TE_B, extra_excess=champ_ex)
    solo = vs_benchmark(champ, CDI)
    print(f"     correlação diária com o campeão {c_ch:+.2f} | com o Ibovespa {c_ib:+.2f}")
    print(f"     campeão sozinho Sh/CDI {solo['sharpe_sobre_cdi']:+.2f}, alpha {_pct(solo['alpha_aa'])} | "
          f"campeão + pares por cima: Sh/CDI {combo['sh'] or 0:+.2f}, alpha {_pct(combo['alpha'])}, "
          f"pós-IR {_pct(combo['alpha_ir'])}, maxDD {_pct(combo['dd'], '{:.0%}')}")
