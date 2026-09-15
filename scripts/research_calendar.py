"""Efeito virada de mês e retorno overnight na B3 — pesquisa + OOS.

1) VIRADA DE MÊS (McConnell & Xu 2008: 31 de 35 países). Diagnóstico no ^BVSP
   fechamento-a-fechamento 2000–2026; janela escolhida no TREINO (2000–2012) e
   testada no BOVA11 real (2013–2026), executando no leilão de fechamento ou na
   abertura seguinte, CDI no caixa, custo 4 bps por ponta. Placebo: os mesmos N
   pregões por mês sorteados ao acaso, 500 vezes. Walk-forward anual.
2) OVERNIGHT x PREGÃO (Cliff, Cooper & Gulen 2008). Decompõe o retorno de BOVA11 e
   das ações da cesta em noite (fechamento -> abertura) e dia (abertura ->
   fechamento); no mini-índice (M1 do MT5) separa noite, 9h–10h, pregão do à vista
   e pós-fechamento. Testa "comprado só à noite" no WIN com custo e IR de swing.

Pré-requisitos: pull_futures_proxy.py, pull_etfs.py, pull_basket.py, pull_cdi.py,
pull_mt5_intraday.py.
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
from trader.intraday import WIN, BarIndex, cash_close_minute, run_intraday
from trader.intraday.costs import Instrument
from trader.risk.overlay import vol_target_multiplier
from trader.seasonal import month_offsets, timing_returns, tom_flag
from trader.strategy import SuperTrend
from trader.taxes import after_tax_summary

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
DATA = Path(cfg["data"]["dir"])
CDI = pd.read_parquet(DATA / "cdi_d1.parquet")["rate"]
EQ0, COST = 1_000_000.0, 0.0004
TRAIN_END, TEST_START = pd.Timestamp("2012-12-31", tz="UTC"), pd.Timestamp("2013-01-01", tz="UTC")


def _pct(x, fmt="{:+.1%}"):
    return "—" if x is None or x != x else fmt.format(x)


def _t(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / (x.std() / np.sqrt(len(x)))) if len(x) > 2 and x.std() > 0 else float("nan")


ibov = load(DATA / "ibov_d1.parquet")
bova = load(DATA / "bova11_d1.parquet")
live = (bova["volume"].rolling(20).median() > 0)
bova = bova[live.idxmax():]

# =============================================================================== 1
print("=" * 100)
print("1) VIRADA DE MÊS")
print("=" * 100)
r = ibov["close"].pct_change()
off = month_offsets(ibov.index)
labels = pd.Series("meio do mês", index=ibov.index)
for k in (-3, -2, -1):
    labels[off["from_end"] == k] = f"fim {k}"
for k in (1, 2, 3, 4, 5):
    labels[(off["from_start"] == k) & (labels == "meio do mês")] = f"início +{k}"
order = ["fim -3", "fim -2", "fim -1", "início +1", "início +2", "início +3", "início +4",
         "início +5", "meio do mês"]
print("\nIbovespa: retorno médio por pregão do mês (fechamento a fechamento)")
print(f"{'pregão':12} | {'2000–2012 (treino)':^22} | {'2013–2026 (teste)':^22}")
print(f"{'':12} | {'bps/dia':>9} {'t':>5} {'n':>6} | {'bps/dia':>9} {'t':>5} {'n':>6}")
for lab in order:
    cells = []
    for lo, hi in ((None, TRAIN_END), (TEST_START, None)):
        m = labels == lab
        if lo is not None:
            m &= ibov.index >= lo
        if hi is not None:
            m &= ibov.index <= hi
        x = r[m]
        cells.append(f"{x.mean() * 1e4:>+9.1f} {_t(x):>5.1f} {x.count():>6}")
    print(f"{lab:12} | {cells[0]} | {cells[1]}")
for lo, hi, name in ((None, TRAIN_END, "treino"), (TEST_START, None, "teste")):
    m = pd.Series(True, index=ibov.index)
    if lo is not None:
        m &= ibov.index >= lo
    if hi is not None:
        m &= ibov.index <= hi
    tom = tom_flag(off, start=-1, end=3) & m
    lr = np.log1p(r)
    yrs = (ibov.index[m][-1] - ibov.index[m][0]) / pd.Timedelta(days=365.25)
    print(f"  {name}: janela [-1,+3] = {tom[m].mean():.0%} dos pregões, "
          f"{lr[tom].sum() / yrs:+.1%} a.a. | resto do mês {lr[m & ~tom].sum() / yrs:+.1%} a.a. (log)")

# ---- escolha da janela no treino (^BVSP, execução no fechamento)
GRID = [(s, e) for s, e in product((-4, -3, -2, -1, 0), (0, 1, 2, 3, 4, 5)) if (s, e) != (0, 0)]
tr_idx = ibov.index[ibov.index <= TRAIN_END]


def tom_hold(index, s, e):
    return tom_flag(month_offsets(index), start=s if s < 0 else 99, end=e).astype(float)


scores = []
for s, e in GRID:
    h = tom_hold(ibov.index, s, e).reindex(tr_idx)
    res = timing_returns(h, ibov["close"], ibov["close"], CDI, cost_side=COST, execution="close")
    scores.append((res.stats["sharpe_cdi"] or -9.0, (s, e)))
best_sh, BEST = max(scores)
print(f"\nJanela escolhida no treino (Ibovespa 2000–2012): último(s) {abs(BEST[0]) if BEST[0] < 0 else 0} "
      f"pregão(ões) do mês + primeiros {BEST[1]} (Sh/CDI treino {best_sh:+.2f})")

# ---- teste no BOVA11
te = bova.index[bova.index >= TEST_START]
rows = []
bh = timing_returns(pd.Series(1.0, index=te), bova["open"], bova["close"], CDI, cost_side=COST)
rows.append(("buy & hold BOVA11", bh))
for (s, e), label in ((( -1, 3), "literatura [-1,+3]"), (BEST, f"escolhida [{BEST[0]},+{BEST[1]}]")):
    for ex in ("close", "open"):
        h = tom_hold(bova.index, s, e).reindex(te)
        rows.append((f"{label} {'fech.' if ex == 'close' else 'abert.'}",
                     timing_returns(h, bova["open"], bova["close"], CDI, cost_side=COST, execution=ex)))
print(f"\nTESTE no BOVA11 {te[0].date()}..{te[-1].date()} (custo {COST * 1e4:.0f} bps/ponta, CDI no caixa)")
print(f"{'':30} {'CAGR':>6} {'alpha':>6} {'pós-IR':>7} {'Sh/CDI':>6} {'maxDD':>6} {'no mercado':>10} {'idas/ano':>8}")
for name, res in rows:
    st = res.stats
    print(f"{name:30} {_pct(st['cagr']):>6} {_pct(st['alpha']):>6} {_pct(st['alpha_pos_ir']):>7} "
          f"{st['sharpe_cdi'] or 0:>+6.2f} {_pct(st['max_dd'], '{:.0%}'):>6} {st['tempo_posicionado']:>10.0%} "
          f"{st['idas_voltas_aa']:>8.1f}")

# ---- placebo: mesmo número de pregões por mês, sorteados
rc = bova["close"].pct_change().reindex(te).fillna(0.0).to_numpy()
cdi_te = CDI.reindex(te).ffill().fillna(0.0).to_numpy()
months = np.asarray(te.year * 100 + te.month)


def sharpe_hold(h: np.ndarray) -> float:
    dh = np.abs(np.diff(np.r_[0.0, h]))
    ex = h * rc + (1 - h) * cdi_te - dh * COST - cdi_te
    return float(ex.mean() / ex.std() * np.sqrt(252)) if ex.std() > 0 else 0.0


tom_te = tom_hold(bova.index, -1, 3).reindex(te).to_numpy()
k_per_month = pd.Series(tom_te).groupby(months).sum().to_dict()
rng = np.random.default_rng(42)
first_pos = {}
placebo = []
for _ in range(500):
    u = rng.random(len(te))
    order_ = np.lexsort((u, months))
    sm = months[order_]
    starts = np.r_[0, np.nonzero(sm[1:] != sm[:-1])[0] + 1]
    rank = np.empty(len(te), dtype=int)
    rank[order_] = np.arange(len(te)) - np.repeat(starts, np.diff(np.r_[starts, len(te)]))
    k = np.array([k_per_month[m] for m in months])
    placebo.append(sharpe_hold((rank < k).astype(float)))
placebo = np.array(placebo)
real = sharpe_hold(tom_te)
print(f"\nPlacebo (500 sorteios de {int(np.mean(list(k_per_month.values())))} pregões/mês no BOVA11, "
      f"fechamento): Sh/CDI mediana {np.median(placebo):+.2f}, 95% em "
      f"[{np.percentile(placebo, 2.5):+.2f}, {np.percentile(placebo, 97.5):+.2f}] | virada [-1,+3] "
      f"{real:+.2f} -> melhor que {np.mean(placebo < real):.0%} dos sorteios")

# ---- walk-forward anual (janela re-escolhida com todo o Ibovespa anterior)
parts = []
picks = []
for y in range(2013, te[-1].year + 1):
    hist = ibov.index[ibov.index < pd.Timestamp(f"{y}-01-01", tz="UTC")]
    sc = []
    for s, e in GRID:
        h = tom_hold(ibov.index, s, e).reindex(hist)
        sc.append((timing_returns(h, ibov["close"], ibov["close"], CDI, cost_side=COST).stats["sharpe_cdi"] or -9, (s, e)))
    s, e = max(sc)[1]
    picks.append((y, s, e))
    yr = bova.index[bova.index.year == y]
    parts.append(tom_hold(bova.index, s, e).reindex(yr))
wf = timing_returns(pd.concat(parts), bova["open"], bova["close"], CDI, cost_side=COST)
print(f"Walk-forward anual {te[0].year}–{te[-1].year}: alpha {_pct(wf.stats['alpha'])} | pós-IR "
      f"{_pct(wf.stats['alpha_pos_ir'])} | Sh/CDI {wf.stats['sharpe_cdi'] or 0:+.2f} | maxDD "
      f"{_pct(wf.stats['max_dd'], '{:.0%}')} | janelas: "
      + " ".join(f"{y % 100:02d}:[{s},+{e}]" for y, s, e in picks))

# ---- correlação com o campeão
basket = {s: load(DATA / f"{s.lower()}_d1.parquet") for s in cfg["basket"]["symbols"]}
bidx = None
for f in basket.values():
    bidx = f.index if bidx is None else bidx.intersection(f.index)
bte = bidx[bidx >= TEST_START]
yz = vol_target_multiplier(basket, target_ann_vol=cfg["risk_overlay"]["target_ann_vol"],
                           lookback=cfg["risk_overlay"]["vol_lookback"],
                           max_leverage=cfg["risk_overlay"]["max_leverage"], estimator="yang_zhang")
sig = pd.DataFrame({s: SuperTrend(10, 3.0).generate_signals(f).reindex(bidx) for s, f in basket.items()}).fillna(0.0)
champ = run_target_portfolio(basket, sig.reindex(bte), B3Costs.from_config(cfg), initial_equity=EQ0,
                             lot_size=cfg["backtest"]["lot_size"], overlay_mult=yz.reindex(bte),
                             cash_rate=CDI).equity
cr = CDI.reindex(bte).ffill().fillna(0.0)
champ_ex = champ.pct_change().fillna(0.0) - cr
tom_ex = rows[2][1].returns.reindex(bte).fillna(0.0) - cr
half = EQ0 * (1 + cr + 0.5 * champ_ex + 0.5 * tom_ex).cumprod()
print(f"Correlação diária (excesso sobre CDI) virada x campeão: {champ_ex.corr(tom_ex):+.2f} | "
      f"campeão sozinho Sh/CDI {vs_benchmark(champ, CDI)['sharpe_sobre_cdi']:+.2f} | "
      f"50% campeão + 50% virada Sh/CDI {vs_benchmark(half, CDI)['sharpe_sobre_cdi']:+.2f}")

# =============================================================================== 2
print("\n" + "=" * 100)
print("2) OVERNIGHT x PREGÃO")
print("=" * 100)
print("\nRetorno médio anualizado (média diária x 252) — noite = fechamento->abertura, dia = abertura->fechamento")
print(f"{'ativo':8} {'período':11} | {'noite':>7} {'t':>5} | {'dia':>7} {'t':>5} | {'noite, teste':>12} {'dia, teste':>10}")
assets = {"BOVA11": bova} | {s: basket[s] for s in cfg["basket"]["symbols"]}
for name, f in assets.items():
    f = f[(f["open"] > 0) & (f["volume"] > 0)]
    night = (f["open"] / f["close"].shift(1) - 1.0).dropna()
    day = (f["close"] / f["open"] - 1.0).dropna()
    print(f"{name:8} {f.index[0].year}–{f.index[-1].year:<6} | {night.mean() * 252:>+7.1%} {_t(night):>5.1f} | "
          f"{day.mean() * 252:>+7.1%} {_t(day):>5.1f} | {night[night.index >= TEST_START].mean() * 252:>+12.1%} "
          f"{day[day.index >= TEST_START].mean() * 252:>+10.1%}")

win = load(DATA / "win_d_mt5_m1.parquet")
winn = load(DATA / "win_n_mt5_m1.parquet")
petr = load(DATA / "petr4_mt5_m1.parquet")
CC = cash_close_minute(petr.index)
bi = BarIndex(win.index)
days, s0, e0 = bi.day_bounds()
keep = np.isin(days, CC.index)
days, s0, e0 = days[keep], s0[keep], e0[keep]
o, c = win["open"].to_numpy(), win["close"].to_numpy()
i10 = bi.at_or_after(days, 600)
icc = bi.at_or_after(days, CC.reindex(days).to_numpy() + 1)
ok = (i10 > s0) & (icc > i10) & (icc < e0)
seg = pd.DataFrame({
    "noite (18h25->9h)": np.r_[np.nan, o[s0[1:]] - c[e0[:-1] - 1]],
    "9h->10h": o[i10] - o[s0],
    "pregão do à vista": o[icc] - o[i10],
    "pós-fechamento": c[e0 - 1] - o[icc],
}, index=days)[ok]
wb = BarIndex(winn.index).day_bounds()
price = pd.Series(winn["close"].to_numpy()[wb[2] - 1], index=wb[0]).reindex(seg.index)
seg_days = pd.to_datetime(pd.Series(seg.index).astype(str), format="%Y%m%d").dt.tz_localize("UTC")
carry = float(CDI.reindex(seg_days).ffill().mean() * 252)
print(f"\nMini-índice (WIN, M1 do MT5) {seg.index[0]}..{seg.index[-1]} — pontos por dia e % a.a. do preço")
print(f"{'trecho':20} | {'pts/dia':>8} {'t':>5} {'% a.a.':>7}")
for col in seg.columns:
    x = seg[col]
    print(f"{col:20} | {x.mean():>+8.1f} {_t(x):>5.1f} {(x / price).mean() * 252:>+7.1%}")
print(f"  o futuro desconta ~1 dia de CDI a cada virada de pregão ({carry:.1%} a.a.): noite do WIN + CDI = "
      f"{(seg.iloc[:, 0] / price).mean() * 252 + carry:+.1%} a.a. — é o que o à vista 'ganha' à noite no índice.")

# ---- ações: compra no leilão de fechamento, vende no leilão de abertura
univ = sorted(set(cfg["factor_universe"]["symbols"]) | set(cfg["basket"]["symbols"]))
panel = {s: load(DATA / f"{s.lower()}_d1.parquet") for s in univ
         if (DATA / f"{s.lower()}_d1.parquet").exists()}
cal = ibov.index[ibov.index >= pd.Timestamp("2003-01-01", tz="UTC")]
O = pd.DataFrame({s: f["open"].reindex(cal) for s, f in panel.items()})
C = pd.DataFrame({s: f["close"].reindex(cal) for s, f in panel.items()})
V = pd.DataFrame({s: f["volume"].reindex(cal) for s, f in panel.items()})
gap = O / C.shift(1) - 1.0
NIGHT = gap.where((O > 0) & (V > 0) & (C.shift(1) > 0))
DAY = (C / O - 1.0).where((O > 0) & (V > 0))
stale = gap.abs().lt(1e-9).astype(float).rolling(60, min_periods=40).mean()
liq = (C * V).rolling(60, min_periods=40).median()
RANK = liq.where(stale < 0.10).shift(2).rank(axis=1, ascending=False)     # conhecido antes do leilão
bmask = (bova["open"] > 0) & (bova["volume"] > 0)
bnight = (bova["open"] / bova["close"].shift(1) - 1.0).where(bmask).reindex(cal)
bday = (bova["close"] / bova["open"] - 1.0).where(bmask).reindex(cal)
UNIV = {"BOVA11": (bnight.to_frame("BOVA11"), bday.to_frame("BOVA11"), 1),
        "cesta 5": (NIGHT[cfg["basket"]["symbols"]], DAY[cfg["basket"]["symbols"]], 5),
        "top 20 liquidez": (NIGHT.where(RANK <= 20), DAY.where(RANK <= 20), 20),
        "top 50 liquidez": (NIGHT.where(RANK <= 50), DAY.where(RANK <= 50), 50)}
FEE, FEE_DT, SLIP, CORR = 0.0003, 0.00023, 0.0002, 4.90


def overnight(univ_name, win, *, slip=False, corret=False, with_cdi=False, day=False):
    night_f, day_f, n = UNIV[univ_name]
    g = (day_f if day else night_f).reindex(win).mean(axis=1)
    traded = g.notna().astype(float)
    cost = 2 * ((FEE_DT if day else FEE) + (SLIP if slip else 0.0)) + (2 * n * CORR / EQ0 if corret else 0.0)
    rate = CDI.reindex(win).ffill().fillna(0.0)
    ret = g.fillna(0.0) - cost * traded + (rate if (with_cdi or day) else 0.0)
    eq = EQ0 * (1.0 + ret).cumprod()
    b = vs_benchmark(eq, CDI)
    return {"alpha": b["alpha_aa"], "ir": after_tax_summary(eq, CDI)["alpha_pos_ir"],
            "sh": b["sharpe_sobre_cdi"], "dd": float((eq / eq.cummax() - 1.0).min()),
            "pior": float(g.min()), "ret": ret}


TR_W, TE_W = cal[cal <= TRAIN_END], cal[cal >= TEST_START]
print("\nCOMPRA NO LEILÃO DE FECHAMENTO, VENDE NO LEILÃO DE ABERTURA — todo pregão, peso igual")
print("  top N = N mais líquidas pela mediana de 60 pregões conhecida 2 dias antes; IR swing 15%")
print("  custo: taxa B3 swing 3 bps/ponta | +slip: +2 bps/ponta | +corret.: R$4,90/ordem (pendente) em R$1 mi")
print("  caixa 'sem CDI' = o dinheiro fica na ação à noite e não rende CDI (conservador)")
print("  caixa 'com CDI' = liquidação D+2 compensada, patrimônio segue no CDI (otimista)")
print(f"\n{'':16} | {'TREINO 2003–12':^14} | {'TESTE 2013–26, sem CDI':^49} | {'TESTE, com CDI':^21}")
print(f"{'universo':16} | {'Sh/CDI':>6} {'alpha':>7} | {'só taxa alpha':>13} {'pós-IR':>7} {'Sh/CDI':>6} {'maxDD':>6} "
      f"{'+slip':>6} {'+corret.':>8} | {'alpha':>6} {'pós-IR':>7} {'+slip':>6}")
for u in UNIV:
    tr = overnight(u, TR_W)
    te = overnight(u, TE_W)
    te_s = overnight(u, TE_W, slip=True)
    te_c = overnight(u, TE_W, slip=True, corret=True)
    te_cdi = overnight(u, TE_W, with_cdi=True)
    te_cdi_s = overnight(u, TE_W, with_cdi=True, slip=True)
    print(f"{u:16} | {tr['sh'] or 0:>+6.2f} {_pct(tr['alpha']):>7} | {_pct(te['alpha']):>13} {_pct(te['ir']):>7} "
          f"{te['sh'] or 0:>+6.2f} {_pct(te['dd'], '{:.0%}'):>6} {_pct(te_s['alpha']):>6} {_pct(te_c['alpha']):>8} | "
          f"{_pct(te_cdi['alpha']):>6} {_pct(te_cdi['ir']):>7} {_pct(te_cdi_s['alpha']):>6}")

best_u = max(UNIV, key=lambda u: overnight(u, TR_W)["sh"] or -9.0)
bb = overnight(best_u, TE_W)
yearly = (1.0 + bb["ret"]).groupby(bb["ret"].index.year).prod() - 1.0
cdi_y = (1.0 + CDI.reindex(TE_W).ffill().fillna(0.0)).groupby(TE_W.year).prod() - 1.0
print(f"\nEscolhido no treino: {best_u}. No teste, por ano (só taxa, sem CDI), menos o CDI do ano:")
print("  " + " ".join(f"{y}:{v - cdi_y[y]:+.0%}" for y, v in yearly.items()))
print(f"  pior noite da carteira no teste: {bb['pior']:+.1%} | correlação diária com o Ibovespa "
      f"{bb['ret'].corr(ibov['close'].pct_change().reindex(TE_W)):+.2f}")
dt = overnight(best_u, TE_W, day=True)
print(f"  espelho — comprado só DURANTE o pregão (day trade, CDI à noite): alpha {_pct(dt['alpha'])}, "
      f"Sh/CDI {dt['sh'] or 0:+.2f}")

mt5d = load(DATA / "petr4_mt5_d1.parquet")
mt5d.index = mt5d.index.normalize()
for label, f in (("MT5 (bruto)", mt5d), ("Yahoo (ajustado)", basket["PETR4"].reindex(mt5d.index))):
    g = (f["open"] / f["close"].shift(1) - 1.0).dropna()
    eq = EQ0 * (1 + g - 2 * FEE).cumprod()
    print(f"  checagem PETR4 {mt5d.index[0].date()}..{mt5d.index[-1].date()} {label:17}: noite "
          f"{g.mean() * 252:+.1%} a.a. | só taxa, sem CDI: alpha {_pct(vs_benchmark(eq, CDI)['alpha_aa'])}")
