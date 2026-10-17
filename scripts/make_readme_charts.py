"""Gera os gráficos do README em docs/img/ (alocação tática §25 contra CDI e Ibovespa).

Mesma configuração do research_taa.py: R$100 mil, 5 bps + R$4,90/ordem, banda 20%, CDI no
caixa, pré-imposto. Pré-requisitos: pull_taa.py e pull_cdi.py. Rodar da raiz do repo.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from matplotlib.ticker import FuncFormatter

from trader.allocation import faber_sma, run_allocation, splice, static_weights

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
DATA, OUT = Path(cfg["data"]["dir"]), Path("docs/img")
OUT.mkdir(parents=True, exist_ok=True)
CDI = pd.read_parquet(DATA / "cdi_d1.parquet")["rate"]
CDI = CDI.set_axis(CDI.index.tz_convert(None).normalize())
START, END = "2005-10-01", "2026-12-31"

L = lambda n: pd.read_parquet(DATA / f"taa_{n}.parquet")["close"]
usd, bvsp = L("usdbrl"), L("bvsp")
CAL = bvsp.index[bvsp.index >= "2004-11-18"]
in_brl = lambda s: (s * usd.reindex(s.index).ffill()).dropna()
PX = pd.DataFrame({
    "IBOV": splice(L("bova11"), bvsp, CAL, drag_aa=0.0009, etf_start="2009-06-01"),
    "SPX": splice(L("ivvb11"), in_brl(L("sp500tr")), CAL, drag_aa=0.0104, etf_start="2014-06-01"),
    "OURO": splice(L("gold11"), in_brl(L("gld")), CAL, drag_aa=0.0042, etf_start="2021-02-01"),
})


def run(targets: pd.DataFrame) -> pd.Series:
    win = CAL[(CAL >= START) & (CAL <= END)]
    t = targets[(targets.index >= win[0]) & (targets.index <= win[-1])]
    r = run_allocation(PX[list(t.columns)].reindex(win), t, CDI, cost_bps=5.0, order_cost=4.90,
                       capital=100_000.0, band=0.20)
    return r.equity.set_axis(r.equity.index.tz_localize(None) if r.equity.index.tz else r.equity.index)


eq = {
    "Alocação tática (Faber 10m)": run(faber_sma(PX, months=10)),
    "Só Ibovespa (comprar e manter)": run(static_weights(PX, {"IBOV": 1.0})),
    "CDI": run(static_weights(PX, {})),
}
eq = pd.DataFrame(eq).dropna()
eq = eq / eq.iloc[0] * 100

# --- estilo (paleta de referência do skill dataviz, modo claro)
BLUE, ORANGE, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLOR = {"Alocação tática (Faber 10m)": BLUE, "Só Ibovespa (comprar e manter)": ORANGE, "CDI": MUTED}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})


def base(w=9.5, h=4.6):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150, facecolor=SURF)
    ax.set_facecolor(SURF)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(length=0)
    return fig, ax


def finish(fig, ax, title, sub, name):
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=26)
    ax.text(0, 1.04, sub, transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
    fig.text(0.01, 0.01, r"Backtest pré-imposto, R\$100 mil, custos reais (5 bps + R\$4,90/ordem). "
             "Não garante resultado futuro.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / name, facecolor=SURF)
    plt.close(fig)


def label_end(ax, series, fmt):
    for name, s in series.items():
        ax.annotate(f"{name}\n{fmt(s.iloc[-1])}", (s.index[-1], s.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", fontsize=8.5, color=INK, va="center")


# 1) patrimônio (escala log, base 100)
fig, ax = base()
for n in eq:
    ax.plot(eq.index, eq[n], color=COLOR[n], lw=2.0 if n != "CDI" else 1.6, ls="-" if n != "CDI" else "--",
            label=n, solid_capstyle="round")
ax.set_yscale("log")
ax.minorticks_off()
ax.set_yticks([100, 200, 400, 800, 1600])
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", ".")))
ax.set_xlim(eq.index[0], eq.index[-1] + pd.Timedelta(days=1100))
label_end(ax, eq, lambda v: f"{v:,.0f}".replace(",", "."))
ax.legend(loc="upper left", frameon=False, fontsize=9)
finish(fig, ax, r"Quanto R\$ 100 valeria hoje", "Escala logarítmica, a partir de out/2005. "
       "A estratégia segue o Ibovespa, o S&P 500 e o ouro (em reais) só enquanto estão em alta.", "equity.png")

# 2) drawdown
dd = eq / eq.cummax() - 1
fig, ax = base(h=3.8)
for n in ("Só Ibovespa (comprar e manter)", "Alocação tática (Faber 10m)"):
    ax.fill_between(dd.index, dd[n] * 100, 0, color=COLOR[n], alpha=0.18, lw=0)
    ax.plot(dd.index, dd[n] * 100, color=COLOR[n], lw=1.6, label=n)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.legend(loc="lower left", frameon=False, fontsize=9)
finish(fig, ax, "Quedas desde o pico", f"Pior queda: estratégia {dd.iloc[:, 0].min():.0%}, "
       f"Ibovespa {dd['Só Ibovespa (comprar e manter)'].min():.0%}.", "drawdown.png")

# 3) excesso sobre o CDI por ano
r = eq.pct_change().fillna(0.0)
yr = (1 + r).groupby(r.index.year).prod()
exc = (yr["Alocação tática (Faber 10m)"] - yr["CDI"]) * 100
exc = exc[exc.index > eq.index[0].year]
fig, ax = base(h=3.8)
ax.bar(exc.index, exc.values, color=[BLUE if v >= 0 else ORANGE for v in exc.values], width=0.7)
ax.axhline(0, color=MUTED, lw=1)
ax.set_xticks(exc.index[::2])
ax.set_xticklabels([f"{y % 100:02d}" for y in exc.index[::2]])
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}%"))
finish(fig, ax, "Ganho sobre o CDI em cada ano", f"Azul: ganhou do CDI ({(exc > 0).mean():.0%} dos anos). "
       "Laranja: perdeu. 2026 é parcial.", "excess_by_year.png")
print(f"ok: {sorted(p.name for p in OUT.iterdir())}")
print(eq.iloc[-1].round(0).to_dict(), exc.round(1).to_dict())
