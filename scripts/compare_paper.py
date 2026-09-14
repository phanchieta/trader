"""Campeão (paper-trade REAL na demo) vs Blend (simulado, paper_blend_sim.py) vs CDI.
Alinha as duas curvas pelo início comum e mostra retorno acumulado, drawdown e
alpha sobre o CDI. Rodar quando quiser comparar "conservador vs mais risco".
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
DATA = Path(cfg["data"]["dir"])


def _curve(path: Path, col: str = "equity") -> pd.Series:
    df = pd.read_csv(path)
    idx = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(None).dt.normalize()
    return pd.Series(df[col].to_numpy(), index=idx).groupby(level=0).last()


def _stats(eq: pd.Series, cdi: pd.Series) -> dict:
    eq = eq / eq.iloc[0]
    dd = (eq / eq.cummax() - 1.0).min()
    days = max((eq.index[-1] - eq.index[0]).days, 1)
    cagr = eq.iloc[-1] ** (365.0 / days) - 1.0
    cdi_w = (1.0 + cdi.reindex(eq.index).fillna(0.0)).cumprod()
    cdi_cagr = cdi_w.iloc[-1] ** (365.0 / days) - 1.0
    return {"ret": eq.iloc[-1] - 1.0, "cagr": cagr, "dd": dd,
            "alpha": cagr - cdi_cagr, "n": len(eq)}


champ = _curve(DATA / "paper_journal.csv")
cdi = pd.read_parquet(DATA / "cdi_d1.parquet")["rate"]
cdi.index = pd.to_datetime(cdi.index).tz_localize(None).normalize()

rows = [("campeão (real)", champ)]
blend_p = DATA / "blend_sim_equity.csv"
if blend_p.exists():
    rows.append(("blend (sim)", _curve(blend_p)))

MIN_DIAS_CAGR = 60      # anualizar com menos que isso é ruído, não sinal

print(f"{'':16} {'dias':>5} {'ret.acum':>10} {'maxDD':>8} {'CAGR/alpha':>16}")
for name, eq in rows:
    if len(eq) < 2:
        print(f"{name:16} {len(eq):>5}  (poucos pontos — volte em ~2 semanas)")
        continue
    s = _stats(eq, cdi)
    days = (eq.index[-1] - eq.index[0]).days
    ann = f"{s['cagr']:+.1%} / {s['alpha']:+.1%}" if days >= MIN_DIAS_CAGR else "(cedo p/ anualizar)"
    print(f"{name:16} {s['n']:>5} {s['ret']:>+9.2%} {s['dd']:>+7.2%} {ann:>16}")

start = min(eq.index[0] for _, eq in rows)
print(f"\ninício comum: {start.date()} | hoje: {pd.Timestamp.today().date()}")
print(f"CAGR/alpha só aparece com >= {MIN_DIAS_CAGR} dias corridos — antes disso, "
      "anualizar 2 semanas é ruído.")
print("campeão = equity da conta demo (paper_trade.py) · blend = paper_blend_sim.py")
