"""Simulador diário da alocação tática (§25) — roda junto do campeão real e do blend.

Sem MT5 e sem ordem de verdade: marca a carteira no fechamento dos ETFs, credita CDI
no caixa e, no primeiro pregão de cada mês, rebalanceia pelo sinal do mês que fechou
(mesma regra e banda do backtest), cobrando 5 bps + corretagem por ordem. No dia do
rebalanceamento manda as ordens no Telegram — é o lembrete para operar à mão.

Capital em `taa.sim_capital` (o plano do usuário). Estado em data/taa_sim.json; curva
em data/taa_sim_equity.csv. Comparar com: python scripts/compare_paper.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from trader.allocation.live import brl, faber_signal, fetch_etf_closes, last_completed_month_end, plan_orders
from trader.core.dotenv import load_dotenv
from trader.live import alerts, journal

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
T = cfg["taa"]
CORR = cfg["costs"]["corretagem_por_ordem"]
DATA = Path(cfg["data"]["dir"])
STATE, CURVE = DATA / "taa_sim.json", DATA / "taa_sim_equity.csv"

closes = fetch_etf_closes(T["etfs"])
day = closes.index[-1]                                   # último fechamento disponível
prices = {e: float(closes[e].iloc[-1]) for e in T["etfs"]}
cdi = pd.read_parquet(DATA / "cdi_d1.parquet")["rate"]
cdi.index = cdi.index.tz_convert(None).normalize()

st = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {
    "cash": float(T["sim_capital"]), "units": {}, "signal_me": "", "updated": ""}
cash = float(st["cash"])
units = {k: float(v) for k, v in st["units"].items()}
if st["updated"]:
    gap = cdi[(cdi.index > pd.Timestamp(st["updated"])) & (cdi.index <= day)]
    cash *= float((1.0 + gap).prod())

me = last_completed_month_end(closes.index, day)
alert = []
if me is not None and st["signal_me"] != str(me.date()):
    sig = faber_signal(closes, me, months=T["sma_months"])
    target, orders, equity = plan_orders(sig.weights, prices, units, cash, band=T["band"])
    for e, q in orders.items():
        notional = abs(q) * prices[e]
        cash -= q * prices[e] + notional * T["cost_bps"] / 1e4 + CORR
        units[e] = target[e]
    units = {e: u for e, u in units.items() if u}
    st["signal_me"] = str(me.date())
    situ = ", ".join(f"{e} {'acima' if sig.detail.loc[e, 'acima'] else 'abaixo'}" for e in T["etfs"])
    alert.append(f"Alocação tática - rebalanceamento ({me:%m/%Y}): {situ}")
    alert += [f"{'COMPRAR' if q > 0 else 'VENDER'} {abs(q):.0f} {e} (~{brl(abs(q) * prices[e], 0)})"
              for e, q in orders.items()] or ["sem ordens (dentro da banda)"]
    alert.append(f"capital simulado {brl(equity, 0)} - para outro valor: python scripts/taa_signal.py --capital X")

pos = sum(u * prices[e] for e, u in units.items())
equity = cash + pos
st.update({"cash": cash, "units": units, "updated": str(day.date())})
STATE.write_text(json.dumps(st, indent=2), encoding="utf-8")
journal.append(CURVE, {"ts": str(day.date()), "equity": round(equity, 2), "pos_value": round(pos, 2),
                       "n_positions": len(units), "detail": json.dumps(units)})

for line in alert:
    print(line)
print(f"alocação tática sim | {day:%d/%m/%Y} | equity {brl(equity)} | "
      + ", ".join(f"{e} {u:.0f}" for e, u in units.items()) + f" | CDI {brl(cash)}")
if alert:
    load_dotenv()
    print("telegram:", "enviado" if alerts.notify("\n".join(alert)) else "não configurado / falhou")
