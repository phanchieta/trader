"""Alocação tática entre classes de ativo listadas na B3, com CDI como porto seguro.

Ativos (retorno total em reais; referência antes do ETF, descontada da diferença medida):
  IBOV  BOVA11  (^BVSP antes de 2009-06; -0,09% a.a.)
  SPX   IVVB11  (^SP500TR × USDBRL antes de 2014-06; -1,04% a.a.)
  OURO  GOLD11  (GLD × USDBRL antes de 2021-02; -0,42% a.a.)
  SMLL  SMAL11  (só de 2009-06 em diante)

Estratégia PRINCIPAL registrada antes de rodar: Faber (2007) — 1/3 em cada um de IBOV,
SPX e OURO se o fechamento do mês está acima da média de 10 meses, senão a fatia vai
para o CDI; rebalanceamento mensal, execução no pregão seguinte. Parâmetros da
literatura, nada ajustado aqui. Robustez: mesma regra com small caps; dual momentum de
Antonacci; carteira fixa 1/3 (o que o filtro de tendência acrescenta = Faber − fixa).

Custos: 5 bps por ponta (taxa B3 + slippage) + R$4,90 por ordem, R$100 mil, banda de
20% (só negocia se o sinal mudou ou o peso desviou mais que isso). IR 15,5% no mesmo
modelo do resto da pesquisa.

Também: campeão com a cesta causal (§23) nas mesmas janelas, correlação e uma carteira
meio a meio.

Pré-requisitos: pull_taa.py, pull_cdi.py, build_stocks_db.py, pull_etfs.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml

from trader.allocation import dual_momentum, faber_sma, run_allocation, splice, static_weights
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
CAPITAL, ORDER, BPS, BAND = 100_000.0, 4.90, 5.0, 0.20


def _pct(x, fmt="{:+.1%}"):
    return "—" if x is None or x != x else fmt.format(x)


L = lambda n: pd.read_parquet(DATA / f"taa_{n}.parquet")["close"]
usd = L("usdbrl")
bvsp = L("bvsp")
CAL = bvsp.index[bvsp.index >= "2004-11-18"]
in_brl = lambda s: (s * usd.reindex(s.index).ffill()).dropna()
PX = pd.DataFrame({
    "IBOV": splice(L("bova11"), bvsp, CAL, drag_aa=0.0009, etf_start="2009-06-01"),
    "SPX": splice(L("ivvb11"), in_brl(L("sp500tr")), CAL, drag_aa=0.0104, etf_start="2014-06-01"),
    "OURO": splice(L("gold11"), in_brl(L("gld")), CAL, drag_aa=0.0042, etf_start="2021-02-01"),
    "SMLL": splice(L("smal11"), None, CAL, etf_start="2009-06-01"),
})

CORE = ["IBOV", "SPX", "OURO"]
STRATS = {
    "CDI": lambda: static_weights(PX[CORE], {}),
    "fixa 1/3 (IBOV, SPX, OURO)": lambda: static_weights(PX[CORE], {a: 1 / 3 for a in CORE}),
    "só Ibovespa (buy & hold)": lambda: static_weights(PX[CORE], {"IBOV": 1.0}),
    "PRINCIPAL Faber 10m (IBOV, SPX, OURO)": lambda: faber_sma(PX[CORE], months=10),
    "  robustez: Faber 8m": lambda: faber_sma(PX[CORE], months=8),
    "  robustez: Faber 12m": lambda: faber_sma(PX[CORE], months=12),
    "Faber 10m + small caps": lambda: faber_sma(PX[CORE + ["SMLL"]], months=10),
    "dual momentum (IBOV, SPX)": lambda: dual_momentum(PX[["IBOV", "SPX"]], CDI_N, months=12),
    "dual momentum (IBOV, SPX, OURO)": lambda: dual_momentum(PX[CORE], CDI_N, months=12),
}
TARGETS = {k: f() for k, f in STRATS.items()}


def run(name, start, end):
    tgt = TARGETS[name]
    cols = list(tgt.columns)
    win = CAL[(CAL >= start) & (CAL <= end)]
    px = PX[cols].reindex(win)
    t = tgt[(tgt.index >= win[0]) & (tgt.index <= win[-1])]
    return run_allocation(px, t, CDI_N, cost_bps=BPS, order_cost=ORDER, capital=CAPITAL, band=BAND)


def yearly_excess(eq: pd.Series) -> pd.Series:
    r = eq.pct_change().fillna(0.0)
    c = CDI_N.reindex(eq.index).fillna(0.0)
    return (1 + r).groupby(eq.index.year).prod() - (1 + c).groupby(eq.index.year).prod()


print("=" * 110)
print("VALIDAÇÃO DAS SÉRIES — CAGR em reais no período de cada ETF (depois da emenda)")
print("=" * 110)
for a in PX:
    s = PX[a].dropna()
    y = (s.index[-1] - s.index[0]).days / 365.25
    print(f"  {a:5} {s.index[0].date()}..{s.index[-1].date()} CAGR {(s.iloc[-1] / s.iloc[0]) ** (1 / y) - 1:+.1%} | "
          f"vol {s.pct_change().std() * np.sqrt(252):.0%} | pior queda {(s / s.cummax() - 1).min():.0%}")
cors = PX[CORE].resample("ME").last().pct_change().corr()
print("  correlação mensal: " + " | ".join(f"{a}x{b} {cors.loc[a, b]:+.2f}" for a, b in (("IBOV", "SPX"), ("IBOV", "OURO"), ("SPX", "OURO"))))

WINDOWS = [("2005-10-01", "2026-12-31", "2005–2026 (tudo)"), ("2005-10-01", "2015-12-31", "2005–2015"),
           ("2016-01-01", "2026-12-31", "2016–2026")]
RES = {}
for lo, hi, lab in WINDOWS:
    print("\n" + "=" * 110)
    print(f"{lab} — R$100 mil, 5 bps + R$4,90/ordem, banda 20%, CDI no caixa")
    print("=" * 110)
    print(f"{'estratégia':40} {'CAGR':>6} {'alpha':>6} {'pós-IR':>7} {'Sh/CDI':>6} {'maxDD':>6} {'expo':>5} "
          f"{'ord/a':>5} {'anos>CDI':>8} {'pior ano':>8}")
    for name in STRATS:
        if name == "Faber 10m + small caps" and lo < "2010":
            continue
        r = run(name, lo, hi)
        RES[(lab, name)] = r
        st = r.stats
        ye = yearly_excess(r.equity)
        ye = ye[ye.index > r.equity.index[0].year] if len(ye) > 2 else ye
        print(f"{name:40} {_pct(st['cagr']):>6} {_pct(st['alpha']):>6} {_pct(st['alpha_pos_ir']):>7} "
              f"{st['sharpe_cdi'] or 0:>+6.2f} {_pct(st['max_dd'], '{:.0%}'):>6} {st['exposicao']:>5.2f} "
              f"{st['ordens_aa']:>5.0f} {f'{(ye > 0).mean():.0%}':>8} {_pct(ye.min()):>8}")

prim = RES[("2005–2026 (tudo)", "PRINCIPAL Faber 10m (IBOV, SPX, OURO)")]
ye = yearly_excess(prim.equity)
print("\nPRINCIPAL — excesso sobre o CDI por ano: " + " ".join(f"{y % 100:02d}:{v:+.0%}" for y, v in ye.items()))
w_avg = prim.weights.mean()
print("  peso médio: " + " | ".join(f"{a} {v:.0%}" for a, v in w_avg.items()) + f" | CDI {1 - w_avg.sum():.0%}")

# ---------------------------------------------------------------- campeão causal e combinação
print("\n" + "=" * 110)
print("COMPARAÇÃO COM O CAMPEÃO (cesta causal, R$4,90/ordem, banda 15%) E CARTEIRA MEIO A MEIO")
print("=" * 110)
rows = load_rows(DATA / "b3")
BCAL = pd.DatetimeIndex(sorted(rows["date"].unique()))
bova = load(DATA / "bova11_d1.parquet")
bova.index = bova.index.tz_convert(None).normalize()
bova = bova[(bova["open"] > 0) & (bova["volume"] > 0)]
mgap = (bova["open"] / bova["close"].shift(1) - 1.0).reindex(BCAL).fillna(0.0)
panel = build_panel(rows, BCAL, market_gap=mgap, total_return=True)
FR = {i: f.reindex(BCAL).ffill(limit=10) for i, f in panel.items()}
members, _ = causal_members(panel, BCAL, n=5)
RO = cfg["risk_overlay"]
cols, tgt, mult = champion_targets(members, FR, BCAL, n=5, target_vol=RO["target_ann_vol"],
                                   lookback=RO["vol_lookback"], max_leverage=RO["max_leverage"])
COSTS = B3Costs.from_config(cfg)
print(f"{'janela':12} {'carteira':34} {'alpha':>6} {'pós-IR':>7} {'Sh/CDI':>6} {'maxDD':>6} {'anos>CDI':>8}")
for lo, lab in (("2010-01-01", "2010–2026"), ("2014-01-01", "2014–2026")):
    win_b = BCAL[BCAL >= lo]
    ch_eq, _ = run_champion(cols, tgt, mult, FR, win_b, COSTS, CDI_N, capital=CAPITAL, band=0.15)
    taa = run("PRINCIPAL Faber 10m (IBOV, SPX, OURO)", lo, "2026-12-31").equity
    taa = taa.set_axis(taa.index.tz_localize("UTC"))
    common = ch_eq.index.intersection(taa.index)
    a, b = ch_eq.reindex(common), taa.reindex(common)
    half = 0.5 * a / a.iloc[0] + 0.5 * b / b.iloc[0]
    exc = lambda e: e.pct_change().dropna() - CDI.reindex(e.index).ffill().fillna(0.0).iloc[1:]
    corr = exc(a).corr(exc(b))
    for name, e in (("campeão causal (5 ações)", a), ("Faber 10m (IBOV, SPX, OURO)", b), ("meio a meio", half)):
        bm = vs_benchmark(e, CDI)
        ye = yearly_excess(e.set_axis(e.index.tz_convert(None)))
        print(f"{lab:12} {name:34} {_pct(bm['alpha_aa']):>6} {_pct(after_tax_summary(e, CDI)['alpha_pos_ir']):>7} "
              f"{bm['sharpe_sobre_cdi']:>+6.2f} {_pct(float((e / e.cummax() - 1).min()), '{:.0%}'):>6} "
              f"{f'{(ye > 0).mean():.0%}':>8}")
    print(f"{'':12} correlação diária do excesso sobre o CDI, campeão x Faber: {corr:+.2f}")
