"""Modelo de fatores sem viés de sobrevivência — universo reconstruído do COTAHIST.

O §12 usou ~98 ações que existem HOJE no Yahoo: quem quebrou, foi comprada ou
fechou capital sumiu da amostra. Aqui o universo de cada data é montado com o
histórico da B3 (todas as ações, inclusive deslistadas), com ajuste de
desdobramento/grupamento/bonificação/cisão pelas marcas "ex" do COTAHIST
(`trader/data/b3_panel.py`).

Mesmo modelo: 5 fatores (momentum 12-1, baixa vol, baixo beta, reversão 3a,
reversão 1m), quintil de cima, rebalance a cada 63 pregões, vol-target Yang-Zhang
do Ibov, CDI no caixa, custo B3, R$1 mi. Janela 2013–2026 (reversão de 3 anos precisa
de 4 anos de histórico e o COTAHIST começa em 2008). Parâmetros fixados antes.

  Y     98 ações do Yahoo (retorno total) — como no §12
  Y-B3  as mesmas 98 reconstruídas da B3 (só preço) + filtro de liquidez
  S     TODAS as sobreviventes da B3 que passam no filtro (só preço)
  T     TODAS as ações que passam no filtro, inclusive as que depois sumiram
Filtro (conhecido na data): mediana de R$ negociados em 60 pregões ≥ R$5 mi, preço
≥ R$1, 1 ano de histórico, negociou nos últimos 5 pregões. Papel que para de
negociar é vendido pelo último preço (target zera).
Viés de sobrevivência = S - T.

Pré-requisitos: pull_cotahist.py, build_stocks_db.py, pull_factor_universe.py,
pull_futures_proxy.py, pull_cdi.py.
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
from trader.factors.backtest import build_target
from trader.factors.library import combined_score
from trader.risk.overlay import vol_target_multiplier
from trader.taxes import after_tax_summary

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
DATA = Path(cfg["data"]["dir"])
CDI = pd.read_parquet(DATA / "cdi_d1.parquet")["rate"]
COSTS = B3Costs.from_config(cfg)
RO = cfg["risk_overlay"]
EQ0, LOT, REBAL, QUINT = 1_000_000.0, 100, 63, 0.20
FACT5 = ["momentum", "low_vol", "low_beta", "lt_reversal", "st_reversal"]
MIN_LIQ, MIN_PX = 5e6, 1.0
START, END = pd.Timestamp("2013-01-02"), pd.Timestamp("2026-12-31")


def utc(idx):
    return idx.tz_localize("UTC") if idx.tz is None else idx


def naive(idx):
    return idx.tz_convert(None).normalize() if idx.tz is not None else idx


def _pct(x, fmt="{:+.1%}"):
    return "—" if x is None or x != x else fmt.format(x)


# ------------------------------------------------------------------ painel B3
rows = load_rows(DATA / "b3")
CAL = pd.DatetimeIndex(sorted(rows["date"].unique()))
panel = build_panel(rows, CAL)
last_trade = {i: fr.index[fr["traded"]].max() for i, fr in panel.items()}
SURVIVOR = {i for i, d in last_trade.items() if d >= CAL[-1] - pd.Timedelta(days=30)}
code2isin = {}
for isin, fr in panel.items():
    code = fr["codneg"].dropna().iloc[-1]
    if code not in code2isin or last_trade[isin] > last_trade[code2isin[code]]:
        code2isin[code] = isin

ibov = load(DATA / "ibov_d1.parquet")
ibov.index = naive(ibov.index)
MKT = ibov["close"].reindex(CAL).ffill()

print("=" * 100)
print("PAINEL DA B3")
print("=" * 100)
print(f"{len(panel)} papéis (ISIN) com negócio em lote padrão 2008–2026 | sobreviventes (negociando no fim): "
      f"{len(SURVIVOR)} | sumiram: {len(panel) - len(SURVIVOR)}")

# ---- validação contra o Yahoo nas sobreviventes
exd = rows[(rows["ev"] != rows.groupby("codisi")["ev"].shift(1)) & rows["ev"].str.contains("D|J|R|C")]
exset = set(zip(exd["codisi"], exd["date"]))
tot, bad = 0, 0
yahoo_frames = {}
for s in cfg["factor_universe"]["symbols"]:
    p = DATA / f"{s.lower()}_d1.parquet"
    if not p.exists():
        continue
    y = load(p)
    y.index = naive(y.index)
    yahoo_frames[s] = y
    if s not in code2isin:
        continue
    fr = panel[code2isin[s]]
    common = fr.index.intersection(y.index)
    rc, ry = fr["close"].reindex(common).pct_change(), y["close"].reindex(common).pct_change()
    ok = fr["traded"].reindex(common) & fr["traded"].reindex(common).shift(1, fill_value=False)
    noex = np.array([(code2isin[s], d) not in exset for d in common])
    d = (rc - ry)[ok & noex].dropna()
    tot += len(d)
    bad += int((d.abs() > 0.05).sum())
print(f"validação: {len([s for s in yahoo_frames if s in code2isin])} ações também no Yahoo | {tot} dias sem "
      f"provento comparados | diferença > 5% em {bad} ({bad / tot:.2%}) — a maioria é erro do Yahoo")

# ---- elegibilidade (conhecida na data)
def eligibility(isins) -> pd.DataFrame:
    cols = {}
    for i in isins:
        fr = panel[i].reindex(CAL)
        liq = fr["volume"].fillna(0.0).rolling(60, min_periods=40).median()
        age = fr["close"].notna().cumsum()
        alive = fr["traded"].fillna(False).astype(float).rolling(5, min_periods=1).max() > 0
        cols[i] = (liq >= MIN_LIQ) & (fr["raw_close"] >= MIN_PX) & (age >= 252) & alive
    return pd.DataFrame(cols).fillna(False)


ALL = list(panel)
ELIG = eligibility(ALL)
SURV_COLS = [i for i in ALL if i in SURVIVOR]
Y98 = [code2isin[s] for s in yahoo_frames if s in code2isin]
ALIVE = pd.DataFrame({i: panel[i]["traded"].reindex(CAL).fillna(False).astype(float)
                      .rolling(5, min_periods=1).max() > 0 for i in ALL}).fillna(False)

cnt = ELIG.sum(axis=1)
cnt_s = ELIG[SURV_COLS].sum(axis=1)
yrs = [y for y in range(2010, 2027, 2)]
print("elegíveis por ano (todas / sobreviventes / das 98 do Yahoo): " + " ".join(
    f"{y}: {int(cnt[CAL.year == y].mean())}/{int(cnt_s[CAL.year == y].mean())}/"
    f"{int(ELIG[Y98][CAL.year == y].sum(axis=1).mean())}" for y in yrs))

# ------------------------------------------------------------------ backtests
WIN = CAL[(CAL >= START) & (CAL <= END)]
bova = load(DATA / "bova11_d1.parquet")
bova.index = naive(bova.index)
bova = bova[(bova["open"] > 0) & (bova["volume"] > 0)]
MGAP = (bova["open"] / bova["close"].shift(1) - 1.0).reindex(CAL).fillna(0.0)
panel_tr = build_panel(rows, CAL, market_gap=MGAP, total_return=True)

FR_PX = {i: panel[i].reindex(CAL).ffill(limit=10) for i in ALL}
FR_TR = {i: panel_tr[i].reindex(CAL).ffill(limit=10) for i in ALL}
SCORE_PX = combined_score({i: panel[i] for i in ALL}, FACT5, mkt=MKT).reindex(CAL)
SCORE_TR = combined_score({i: panel_tr[i] for i in ALL}, FACT5, mkt=MKT).reindex(CAL)
yz_frame = ibov[["open", "high", "low", "close", "volume"]].reindex(CAL).ffill()
YZ = vol_target_multiplier({"IBOV": yz_frame}, target_ann_vol=RO["target_ann_vol"],
                           lookback=RO["vol_lookback"], max_leverage=RO["max_leverage"],
                           estimator="yang_zhang")

# provento implícito x Yahoo nas 98 (dia ex: retorno total Yahoo - retorno só preço B3)
recs = []
for s in yahoo_frames:
    if s not in code2isin:
        continue
    i = code2isin[s]
    rp = panel[i]["close"].pct_change()
    rt = panel_tr[i]["close"].pct_change()
    ry = yahoo_frames[s]["close"].pct_change().reindex(rp.index)
    for d in panel_tr[i].index[(panel_tr[i]["provento"] != 0) & (panel_tr[i].index >= START)]:
        a, b, c = ry.get(d), rp.get(d), rt.get(d)
        if all(v == v for v in (a, b, c)) and abs(a - b) < 0.2:
            recs.append((s, d.year, a - b, c - b))
ev = pd.DataFrame(recs, columns=["tk", "ano", "yahoo", "b3"]).groupby(["tk", "ano"]).sum()
print(f"provento implícito (B3) x Yahoo, por ação-ano desde {START.year}: yield médio {ev['b3'].mean():.1%} x "
      f"{ev['yahoo'].mean():.1%} | correlação {ev['b3'].corr(ev['yahoo']):.2f}")


def run(name, score, frames, alive=None):
    tgt = build_target(score, WIN, n_frac=QUINT, rebal=REBAL)
    if alive is not None:
        tgt = tgt.where(alive.reindex(index=WIN, columns=tgt.columns).fillna(False), 0.0)
    used = [c for c in tgt.columns if (tgt[c] > 0).any()]
    tgt = tgt[used]
    res = run_target_portfolio({c: frames[c] for c in used}, tgt, COSTS, overlay_mult=YZ.reindex(WIN),
                               cash_rate=CDI.set_axis(naive(CDI.index)), target_is_weight=True,
                               initial_equity=EQ0, lot_size=LOT)
    eq = res.equity.set_axis(utc(res.equity.index))
    b = vs_benchmark(eq, CDI)
    held = (tgt > 0).sum(axis=1)
    return {"name": name, "eq": eq, "cagr": b["cagr_estrategia"], "alpha": b["alpha_aa"],
            "ir": after_tax_summary(eq, CDI)["alpha_pos_ir"], "sh": b["sharpe_sobre_cdi"],
            "dd": float((eq / eq.cummax() - 1).min()), "n": float(held[held > 0].mean()), "tgt": tgt}


YF = {s: f.reindex(CAL).ffill(limit=10) for s, f in yahoo_frames.items()}
y_score = combined_score(yahoo_frames, FACT5, mkt=ibov["close"]).reindex(CAL)
tick = [s for s in yahoo_frames if s in code2isin]
elig_y = pd.DataFrame({s: ELIG[code2isin[s]] for s in tick}).reindex(columns=list(yahoo_frames), fill_value=False)
alive_y = pd.DataFrame({s: ALIVE[code2isin[s]] for s in tick}).reindex(columns=list(yahoo_frames), fill_value=False)

R = {}
R["Y"] = run("Y      98 do Yahoo, sem filtro (§12)", y_score, YF)
R["YF"] = run("Y+F    98 do Yahoo + filtro de liquidez", y_score.where(elig_y), YF, alive_y)
R["YB"] = run("Y-B3   mesmas 98 pela B3 (provento estimado)", SCORE_TR[Y98].where(ELIG[Y98]), FR_TR, ALIVE)
R["S"] = run("S      todas as sobreviventes (B3)", SCORE_TR[SURV_COLS].where(ELIG[SURV_COLS]), FR_TR, ALIVE)
R["T"] = run("T      TODAS as ações (B3, sem viés)", SCORE_TR.where(ELIG), FR_TR, ALIVE)
R["Spx"] = run("S px   sobreviventes, só preço", SCORE_PX[SURV_COLS].where(ELIG[SURV_COLS]), FR_PX, ALIVE)
R["Tpx"] = run("T px   todas, só preço", SCORE_PX.where(ELIG), FR_PX, ALIVE)

print("\n" + "=" * 100)
print(f"MODELO DE FATORES — {WIN[0].date()}..{WIN[-1].date()}, R$1 mi, CDI no caixa")
print("=" * 100)
print(f"{'versão':46} {'CAGR':>6} {'alpha':>6} {'pós-IR':>7} {'Sh/CDI':>6} {'maxDD':>6} {'nomes':>6}")
for r in R.values():
    print(f"{r['name']:46} {_pct(r['cagr']):>6} {_pct(r['alpha']):>6} {_pct(r['ir']):>7} {r['sh']:>+6.2f} "
          f"{_pct(r['dd'], '{:.0%}'):>6} {r['n']:>6.0f}")

d = lambda a, b: f"alpha {R[a]['alpha'] - R[b]['alpha']:+.1%} a.a., Sh/CDI {R[a]['sh'] - R[b]['sh']:+.2f}"
print("\ndecomposição (de Y até T):")
print(f"  filtro de liquidez           Y   - Y+F : {d('Y', 'YF')}")
print(f"  fonte Yahoo x B3             Y+F - Y-B3: {d('YF', 'YB')}")
print(f"  mais sobreviventes           Y-B3 - S  : {d('YB', 'S')}")
print(f"  viés de sobrevivência        S   - T   : {d('S', 'T')}   (só preço: {d('Spx', 'Tpx')})")

# ---- quanto do quintil foi ocupado por quem sumiu, e como foram
tgt = R["T"]["tgt"]
dead_cols = [c for c in tgt.columns if c not in SURVIVOR]
share = (tgt[dead_cols] > 0).sum(axis=1) / (tgt > 0).sum(axis=1).replace(0, np.nan)
print(f"\nno universo T, papéis que depois sumiram ocuparam em média {share.mean():.0%} do quintil "
      f"(pico {share.max():.0%})")
reb = WIN[::REBAL]
rets = {"sobreviventes": [], "sumiram": []}
for a, b in zip(reb[:-1], reb[1:]):
    for c in tgt.columns[(tgt.loc[a] > 0)]:
        px = FR_TR[c]["close"]
        p0, p1 = px.asof(a), px.asof(b)
        if p0 and p0 > 0 and p1 == p1:
            rets["sumiram" if c in dead_cols else "sobreviventes"].append(p1 / p0 - 1)
for k, v in rets.items():
    if v:
        print(f"  retorno médio por trimestre no quintil — {k}: {np.mean(v):+.1%} ({len(v)} posições)")
names = sorted({panel[c]["codneg"].dropna().iloc[-1] for c in dead_cols})
print("  exemplos que passaram pelo quintil e sumiram: " + ", ".join(names[:30]))

# ---- de onde vem a diferença Yahoo x B3? dias em que as duas fontes discordam > 5%
tgtF = R["YF"]["tgt"]
mult = YZ.reindex(WIN).fillna(0.0)
contrib = {"Yahoo acima da B3": [], "B3 acima do Yahoo": []}
events = []
for s in tgtF.columns:
    if s not in code2isin:
        continue
    i = code2isin[s]
    ry = YF[s]["close"].reindex(WIN).pct_change()
    rb = FR_TR[i]["close"].reindex(WIN).pct_change()
    w = tgtF[s].shift(1).fillna(0.0) * mult
    diff = (ry - rb).where(w > 0)
    big = diff[diff.abs() > 0.05].dropna()
    for dt, v in big.items():
        key = "Yahoo acima da B3" if v > 0 else "B3 acima do Yahoo"
        contrib[key].append(float(w.loc[dt] * v))
        events.append((s, dt.date(), round(float(ry.loc[dt]), 3), round(float(rb.loc[dt]), 3), round(float(w.loc[dt]), 3)))
yrs_w = (WIN[-1] - WIN[0]).days / 365.25
print("\nY+F: contribuição dos dias em que Yahoo e B3 discordam > 5% (peso da ação na carteira × diferença):")
for k, v in contrib.items():
    print(f"  {k:18}: {len(v):3} eventos em posições da carteira, somam {sum(v):+.1%} ({sum(v) / yrs_w:+.2%} a.a.)")
ev_df = pd.DataFrame(events, columns=["ação", "data", "ret_yahoo", "ret_b3", "peso"])
ev_df["efeito"] = (ev_df["ret_yahoo"] - ev_df["ret_b3"]) * ev_df["peso"]
print("  maiores efeitos:")
print(ev_df.reindex(ev_df["efeito"].abs().sort_values(ascending=False).index).head(10).to_string(index=False))

# ---- híbridos: separa "que ações escolheu" de "como mediu o retorno"
FR_TR_T = {s: FR_TR[code2isin[s]] for s in tick}
YF_T = {s: YF[s] for s in tick}
score_b3_t = pd.DataFrame({s: SCORE_TR[code2isin[s]] for s in tick}).where(elig_y[tick])
R["H1"] = run("H1     escolha pelo Yahoo, retorno pela B3", y_score[tick].where(elig_y[tick]), FR_TR_T, alive_y[tick])
R["H2"] = run("H2     escolha pela B3, retorno pelo Yahoo", score_b3_t, YF_T, alive_y[tick])
R["YF2"] = run("Y+F    (só as 95 mapeadas)", y_score[tick].where(elig_y[tick]), YF_T, alive_y[tick])
R["YB2"] = run("Y-B3   (só as 95 mapeadas)", score_b3_t, FR_TR_T, alive_y[tick])
print("\nhíbridos (95 ações mapeadas, mesmo filtro):")
for k in ("YF2", "H1", "H2", "YB2"):
    r = R[k]
    print(f"  {r['name']:46} alpha {_pct(r['alpha']):>6} | Sh/CDI {r['sh']:+.2f} | maxDD {_pct(r['dd'], '{:.0%}')}")
print(f"  medir o retorno (Yahoo x B3), mesma escolha Yahoo : {d('YF2', 'H1')}")
print(f"  medir o retorno (Yahoo x B3), mesma escolha B3    : {d('H2', 'YB2')}")
print(f"  escolher as ações (Yahoo x B3), retorno B3         : {d('H1', 'YB2')}")

# diferença média diária de retorno Yahoo - B3 nas posições da carteira Y+F (todos os dias)
tgt2 = R["YF2"]["tgt"]
acc, days_w = 0.0, 0
for s in tgt2.columns:
    w = tgt2[s].shift(1).fillna(0.0) * mult
    ry = YF_T[s]["close"].reindex(WIN).pct_change()
    rb = FR_TR_T[s]["close"].reindex(WIN).pct_change()
    dd_ = ((ry - rb) * w).where(w > 0).dropna()
    acc += float(dd_.clip(-0.05, 0.05).sum())
print(f"  soma peso x (ret Yahoo - ret B3) nos dias com |diferença| <= 5%: {acc:+.1%} ({acc / yrs_w:+.2%} a.a.)")

# ---- qual fator faz a escolha pelo Yahoo ser melhor? (um fator por vez, retorno sempre B3)
from trader.factors.library import factor_score
print("\nescolha por UM fator, retorno sempre da B3 (95 mapeadas, mesmo filtro):")
b3_frames_t = {s: panel_tr[code2isin[s]] for s in tick}
b3px_frames_t = {s: panel[code2isin[s]] for s in tick}
for fac in FACT5:
    sy = factor_score({s: yahoo_frames[s] for s in tick}, fac, mkt=ibov["close"]).reindex(CAL)
    sb = factor_score(b3_frames_t, fac, mkt=MKT).reindex(CAL)
    sp = factor_score(b3px_frames_t, fac, mkt=MKT).reindex(CAL)
    ry = run(f"{fac} Yahoo", sy.where(elig_y[tick]), FR_TR_T, alive_y[tick])
    rb = run(f"{fac} B3 TR", sb.where(elig_y[tick]), FR_TR_T, alive_y[tick])
    rp = run(f"{fac} B3 preço", sp.where(elig_y[tick]), FR_TR_T, alive_y[tick])
    corr = sy.where(elig_y[tick]).stack().corr(sb.where(elig_y[tick]).stack())
    print(f"  {fac:12} alpha: score Yahoo {_pct(ry['alpha']):>6} | score B3 TR {_pct(rb['alpha']):>6} | "
          f"score B3 só preço {_pct(rp['alpha']):>6} | correlação dos percentis Yahoo x B3 {corr:.2f}")

# ---- robustez: ruído minúsculo nos percentis muda quanto o resultado?
rng = np.random.default_rng(0)
base = SCORE_TR.where(ELIG)
alphas, shs = [], []
for k in range(20):
    noisy = base + rng.normal(0.0, 0.02, base.shape) * base.notna()
    r = run("T com ruído", noisy, FR_TR, ALIVE)
    alphas.append(r["alpha"])
    shs.append(r["sh"])
print(f"\nT com ruído de ±2 p.p. nos percentis (20 rodadas): alpha médio {np.mean(alphas):+.1%}, desvio "
      f"{np.std(alphas):.1%}, faixa [{min(alphas):+.1%}, {max(alphas):+.1%}] | Sh/CDI médio {np.mean(shs):+.2f} "
      f"[{min(shs):+.2f}, {max(shs):+.2f}]")
