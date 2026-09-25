"""Alocação tática — o que ter em carteira neste mês (uso manual, 1x por mês).

Regra (§25): 1/3 em cada ETF que fechou o mês acima da média dos últimos 10 fechamentos
mensais; a fatia de quem está abaixo fica no CDI (Tesouro Selic, CDB de liquidez
diária...). Operar no primeiro pregão depois do fim do mês.

    python scripts/taa_signal.py                         # capital do settings (taa.sim_capital)
    python scripts/taa_signal.py --capital 10000         # quanto comprar de cada, do zero
    python scripts/taa_signal.py --holdings BOVA11=18,IVVB11=7 --cash 350   # ordens a partir da posição
    python scripts/taa_signal.py --telegram              # também manda o resumo no Telegram

Baixa os fechamentos do Yahoo na hora; não precisa do MT5.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from trader.allocation.live import brl, faber_signal, fetch_etf_closes, last_completed_month_end, plan_orders
from trader.core.dotenv import load_dotenv
from trader.live import alerts

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
T = cfg["taa"]
CORR = cfg["costs"]["corretagem_por_ordem"]
MESES = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro",
         "outubro", "novembro", "dezembro"]

ap = argparse.ArgumentParser()
ap.add_argument("--capital", type=float, default=None)
ap.add_argument("--holdings", default="", help="ETF=cotas separados por vírgula")
ap.add_argument("--cash", type=float, default=0.0, help="caixa já disponível para a estratégia")
ap.add_argument("--telegram", action="store_true")
args = ap.parse_args()

closes = fetch_etf_closes(T["etfs"])
today = pd.Timestamp.today().normalize()
me = last_completed_month_end(closes.index, today)
sig = faber_signal(closes, me, months=T["sma_months"])
prev = faber_signal(closes, closes.index[-1], months=T["sma_months"])
last = closes.iloc[-1]

lines = [f"Alocação tática - sinal do fechamento de {MESES[me.month - 1]}/{me.year} ({me:%d/%m/%Y})",
         f"{'ETF':8} {'fech. do mês':>13} {'média 10m':>11} {'situação':>9} {'peso':>6}"]
for e in T["etfs"]:
    d = sig.detail.loc[e]
    lines.append(f"{e:8} {d['fechamento']:>13,.2f} {d['media']:>11,.2f} {'acima' if d['acima'] else 'abaixo':>9} "
                 f"{d['peso']:>6.0%}")
cdi_w = 1.0 - sum(sig.weights.values())
lines.append(f"{'CDI':8} {'':>13} {'':>11} {'':>9} {cdi_w:>6.0%}")

changed = [e for e in T["etfs"] if bool(prev.detail.loc[e, "acima"]) != bool(sig.detail.loc[e, "acima"])]
lines.append(f"\nPrévia se o mês fechasse hoje ({closes.index[-1]:%d/%m}): "
             + ", ".join(f"{e} {'acima' if prev.detail.loc[e, 'acima'] else 'abaixo'}" for e in T["etfs"])
             + (f"  <- mudaria: {', '.join(changed)}" if changed else "  (nenhuma mudança)"))

units = {}
if args.holdings:
    for part in args.holdings.split(","):
        k, v = part.split("=")
        units[k.strip().upper()] = float(v)
capital = args.capital if args.capital is not None else (None if units else float(T["sim_capital"]))
cash = capital if capital is not None else args.cash
prices = {e: float(last[e]) for e in T["etfs"]}
target, orders, equity = plan_orders(sig.weights, prices, units, cash, band=T["band"])

lines.append(f"\nCarteira-alvo para {brl(equity)} (preços de {closes.index[-1]:%d/%m}):")
for e in T["etfs"]:
    lines.append(f"  {e:8} {target[e]:>5.0f} cotas  {brl(target[e] * prices[e]):>14}")
invested = sum(target[e] * prices[e] for e in T["etfs"])
lines.append(f"  {'CDI':8} {'':>5}        {brl(equity - invested):>14}")
if orders:
    lines.append("Ordens:")
    for e, q in orders.items():
        lines.append(f"  {'COMPRAR' if q > 0 else 'VENDER':7} {abs(q):>4.0f} {e} (~{brl(abs(q) * prices[e])})")
    lines.append(f"  corretagem: {len(orders)} ordem(ns) x {brl(CORR)}")
else:
    lines.append("Ordens: nenhuma (dentro da banda de 20% ou sinal igual)")
lines.append(f"\nExecutar no primeiro pregão depois de {me:%d/%m/%Y}; próximo sinal no fim de "
             f"{MESES[today.month - 1]}.")

text = "\n".join(lines)
print(text)
if args.telegram:
    load_dotenv()
    print("telegram:", "enviado" if alerts.notify(text) else "não configurado / falhou")
