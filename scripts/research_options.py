"""Vender opções — o prêmio de volatilidade na B3, com séries reais do COTAHIST.

A volatilidade implícita costuma ficar acima da realizada: quem vende opção é
pago para correr o risco de cauda. Com o CDI alto, put coberta por caixa = CDI +
prêmio; venda coberta = ação + prêmio − a alta acima do strike.

Para PETR4 (calls líquidas desde 2008, puts desde 2014), VALE3 (desde 2017) e
BOVA11 (desde 2020):
  1) vol implícita da call no dinheiro de 1 mês x vol realizada até o vencimento;
  2) venda coberta mensal (strike +5% / +10%) e put coberta por caixa (no dinheiro
     / −5%, com e sem trava comprando put 10 p.p. abaixo);
  3) variante escolhida pelo Sharpe/CDI no TREINO, medida no TESTE; linha otimista
     executando pelo meio do livro.
Vende pelo bid e compra pelo ask, tarifa B3 0,064% do prêmio por ponta, exercício
0,08% (coberta) / 0,05% (put) do notional, CDI no caixa, IR 15%.

Pré-requisitos: pull_cotahist.py, build_options_db.py, pull_basket.py, pull_etfs.py,
pull_cdi.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml

from trader.backtest.metrics import vs_benchmark
from trader.data.store import load
from trader.options import implied_vol, monthly_expiries, pick_series, run_option_writer
from trader.taxes import after_tax_summary

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
DATA = Path(cfg["data"]["dir"])
CDI = pd.read_parquet(DATA / "cdi_d1.parquet")["rate"]
CDI_AA = (1.0 + CDI) ** 252 - 1.0
CDI_AA.index = CDI_AA.index.tz_convert(None)
EQ0 = 1_000_000.0

UNDERLYINGS = {
    "PETR4": dict(root="PETR", codbdi="02", especi="PN", yahoo="petr4_d1.parquet",
                  cc=(("2008-01-01", "2016-12-31"), ("2017-01-01", "2026-12-31")),
                  put=(("2014-01-01", "2019-06-30"), ("2019-07-01", "2026-12-31"))),
    "VALE3": dict(root="VALE", codbdi="02", especi="ON", yahoo="vale3_d1.parquet",
                  cc=(("2017-09-01", "2021-06-30"), ("2021-07-01", "2026-12-31")),
                  put=(("2017-09-01", "2021-06-30"), ("2021-07-01", "2026-12-31"))),
    "BOVA11": dict(root="BOVA", codbdi="14", especi="CI", yahoo="bova11_d1.parquet",
                   cc=(("2020-01-01", "2022-06-30"), ("2022-07-01", "2026-12-31")),
                   put=(("2020-01-01", "2022-06-30"), ("2022-07-01", "2026-12-31"))),
}
CC_GRID = [dict(kind="call", moneyness=m, fee_exercise=0.0008) for m in (0.05, 0.10)]
PUT_GRID = [dict(kind="put", moneyness=m, hedge=h, fee_exercise=0.0005)
            for m in (0.0, 0.05) for h in (None, m + 0.10)]

RAW = pd.concat([pd.read_parquet(p) for p in sorted((DATA / "b3").glob("cotahist_*.parquet"))],
                ignore_index=True)


def _pct(x, fmt="{:+.1%}"):
    return "—" if x is None or x != x else fmt.format(x)


def label(p):
    if p["kind"] == "call":
        return f"venda coberta strike +{p['moneyness']:.0%}"
    return f"put -{p['moneyness']:.0%}" + (f" + trava -{p['hedge']:.0%}" if p.get("hedge") else "")


def bench(tr):
    eq = EQ0 * (1.0 + tr.fillna(0.0)).cumprod()
    eq.index = eq.index.tz_localize("UTC")
    b = vs_benchmark(eq, CDI)
    return {"alpha": b["alpha_aa"], "alpha_pos_ir": after_tax_summary(eq, CDI)["alpha_pos_ir"],
            "sharpe_cdi": b["sharpe_sobre_cdi"], "max_dd": float((eq / eq.cummax() - 1).min())}


for name, u in UNDERLYINGS.items():
    stock = RAW[(RAW["tpmerc"] == "010") & (RAW["codneg"] == name) & (RAW["codbdi"] == u["codbdi"])]
    spot = stock.drop_duplicates("date").set_index("date")["close"].sort_index()
    opts = RAW[RAW["tpmerc"].isin(["070", "080"]) & RAW["codneg"].str.startswith(u["root"])
               & RAW["especi"].str.startswith(u["especi"])]
    y = load(DATA / u["yahoo"])["close"].pct_change()
    y.index = y.index.tz_convert(None).normalize()
    tr = y.reindex(spot.index)

    print("=" * 104)
    print(f"{name}")
    print("=" * 104)
    # ---- vol implícita x realizada
    exps = monthly_expiries(opts)
    by_day = {d: g for d, g in opts.groupby("date")}
    rows = []
    for e0, e1 in zip(exps[:-1], exps[1:]):
        after = spot.index[spot.index > e0]
        if not len(after) or (e1 - after[0]).days < 15 or after[0] not in by_day:
            continue
        d = after[0]
        row = pick_series(by_day[d], kind="call", spot=spot.loc[d], moneyness=0.0, expiry=e1)
        if row is None:
            continue
        iv = implied_vol((row["bid"] + row["ask"]) / 2, spot.loc[d], row["strike"], (e1 - d).days / 365.0,
                         float(np.log1p(CDI_AA.asof(d))), "call")
        life = tr.loc[(tr.index > d) & (tr.index <= e1)].dropna()
        if len(life) >= 10 and iv == iv:
            rows.append((d, iv, life.std() * np.sqrt(252)))
    vol = pd.DataFrame(rows, columns=["d", "iv", "rv"]).set_index("d")
    if len(vol):
        gap = vol["iv"] - vol["rv"]
        print(f"vol implícita (call no dinheiro, 1 mês) x realizada, {vol.index[0]:%Y}–{vol.index[-1]:%Y}, {len(vol)} meses: "
              f"implícita {vol['iv'].mean():.0%} | realizada {vol['rv'].mean():.0%} | implícita maior em "
              f"{(gap > 0).mean():.0%} dos meses | diferença mediana {gap.median():+.1%} | média {gap.mean():+.1%}")

    for family, grid, (train, test) in (("VENDA COBERTA", CC_GRID, u["cc"]), ("PUT COBERTA POR CAIXA", PUT_GRID, u["put"])):
        print(f"\n {family} — treino {train[0][:7]}..{train[1][:7]} | teste {test[0][:7]}..{test[1][:7]}")
        print(f"  {'':34} {'treino':>6} | {'alpha':>6} {'pós-IR':>7} {'Sh/CDI':>6} {'maxDD':>6} {'exercidas':>9} "
              f"{'prêmio':>7} {'pior mês':>16}")
        win = {}
        for lab, (a, b) in (("tr", train), ("te", test)):
            m = (spot.index >= a) & (spot.index <= b)
            win[lab] = (spot[m], tr[m], opts[(opts["date"] >= a) & (opts["date"] <= b)])
        bt, bte = bench(win["tr"][1]), bench(win["te"][1])
        print(f"  {'buy & hold ' + name:34} {bt['sharpe_cdi']:>+6.2f} | {_pct(bte['alpha']):>6} "
              f"{_pct(bte['alpha_pos_ir']):>7} {bte['sharpe_cdi']:>+6.2f} {_pct(bte['max_dd'], '{:.0%}'):>6}")
        results = []
        for p in grid:
            r_tr = run_option_writer(*win["tr"], CDI, **p)
            r_te = run_option_writer(*win["te"], CDI, **p)
            results.append((p, r_tr, r_te))
            st = r_te.stats
            mm = r_te.equity.resample("ME").last().pct_change().dropna()
            worst = f"{mm.min():+.0%} ({mm.idxmin():%Y-%m})" if len(mm) else "—"
            print(f"  {label(p):34} {r_tr.stats['sharpe_cdi'] or 0:>+6.2f} | {_pct(st['alpha']):>6} "
                  f"{_pct(st['alpha_pos_ir']):>7} {st['sharpe_cdi'] or 0:>+6.2f} {_pct(st['max_dd'], '{:.0%}'):>6} "
                  f"{_pct(st['exercidas'], '{:.0%}'):>9} {_pct(st['premio_medio_pct_spot'], '{:.1%}'):>7} {worst:>16}")
        p, r_tr, r_te = max(results, key=lambda x: x[1].stats["sharpe_cdi"] or -9)
        mid = run_option_writer(*win["te"], CDI, execution="mid", **p).stats
        print(f"  -> escolhido no treino: {label(p)} | teste alpha {_pct(r_te.stats['alpha'])}, Sh/CDI "
              f"{r_te.stats['sharpe_cdi']:+.2f} | executando no meio do livro: alpha {_pct(mid['alpha'])}, "
              f"Sh/CDI {mid['sharpe_cdi']:+.2f}")
    print()
