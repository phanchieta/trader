"""Estratégia de "nunca realizar perda" (averaging down / grid) — teste honesto.

Compra quando a ação está mais barata que o comum dos últimos dias; se cai mais,
vai comprando e baixando o preço médio; vende tudo num repique acima do preço
médio; nunca vende no vermelho; se o caixa acaba, segura. Caixa parado rende CDI.

Mede o que a regra ESCONDE: utilização de caixa, dias no vermelho, posições que
ficam abertas pra sempre, e o custo de oportunidade contra o CDI e contra o
campeão. Também roda a MESMA regra com um stop de 25% pra ver o que "nunca sair"
custa (ou salva).

Pré-requisitos: python scripts/pull_basket.py ; python scripts/pull_cdi.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml

from trader.backtest.grid_averaging import run_grid_averaging
from trader.backtest.metrics import summarize, vs_benchmark
from trader.costs.b3 import B3Costs
from trader.data.store import load

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
DATA = Path(cfg["data"]["dir"])
EQ0, LOT = 1_000_000.0, cfg["backtest"]["lot_size"]
COSTS = B3Costs.from_config(cfg)

basket = {s: load(DATA / f"{s.lower()}_d1.parquet") for s in cfg["basket"]["symbols"]}
cdi_path = DATA / "cdi_d1.parquet"
if not cdi_path.exists():
    sys.exit("falta data/cdi_d1.parquet — rode: python scripts/pull_cdi.py")
CDI = pd.read_parquet(cdi_path)["rate"]

idx = None
for f in basket.values():
    idx = f.index if idx is None else idx.intersection(f.index)
idx = idx.sort_values()
oos_start = idx[int(len(idx) * 0.40)]                 # back-60% nunca usada pra nada
oos = {s: f.loc[oos_start:] for s, f in basket.items()}
print(f"janela OOS: {oos_start.date()} .. {idx[-1].date()}  ({len(idx[idx >= oos_start])} barras)\n")


def line(name, eq):
    b = vs_benchmark(eq, CDI)
    s = summarize(eq, pd.DataFrame(columns=["price", "shares", "fee"]), EQ0)
    print(f"{name:26} {str(b['cagr_estrategia']):>9} {str(b['cagr_cdi']):>9} "
          f"{str(b['alpha_aa']):>10} {str(b['sharpe_sobre_cdi']):>7} "
          f"{str(s['max_drawdown']):>8}")


# referência: cesta comprada e segurada, equal-weight
norm = pd.DataFrame({s: basket[s]["close"].reindex(oos[s].index) for s in basket})
bh = (norm / norm.iloc[0]).mean(axis=1) * EQ0

print(f"{'config':26} {'CAGR':>9} {'CAGR CDI':>9} {'alpha aa':>10} {'Sh/CDI':>7} {'maxDD':>8}")
print("-" * 74)
line("buy & hold cesta", bh)

variants = [
    dict(entry_z=1.0, add_drop_pct=0.05, take_profit_pct=0.03, tranche_frac=0.10, max_adds=4),
    dict(entry_z=1.5, add_drop_pct=0.07, take_profit_pct=0.04, tranche_frac=0.10, max_adds=4),
    dict(entry_z=1.0, add_drop_pct=0.05, take_profit_pct=0.03, tranche_frac=0.06, max_adds=6),
    dict(entry_z=2.0, add_drop_pct=0.10, take_profit_pct=0.05, tranche_frac=0.08, max_adds=5),
]
results = []
for v in variants:
    r = run_grid_averaging(oos, COSTS, cash_rate=CDI, initial_equity=EQ0,
                           lot_size=LOT, **v)
    tag = f"z{v['entry_z']} drop{int(v['add_drop_pct']*100)} tp{int(v['take_profit_pct']*100)} f{int(v['tranche_frac']*100)}"
    line(tag, r.equity)
    results.append((tag, v, r))

# a mesma regra COM stop de 25% (deixa de ser "nunca perde")
base = variants[0]
r_stop = run_grid_averaging(oos, COSTS, cash_rate=CDI, initial_equity=EQ0,
                            lot_size=LOT, stop_pct=0.25, **base)
line("z1 drop5 tp3 + STOP 25%", r_stop.equity)

print("\n--- diagnóstico da variante base (z1 drop5 tp3 f10) ---")
rb = results[0][2]
for k in ("trades_fechados", "acerto_trades_fechados", "tranches_por_trade_media",
          "espera_max_ate_lucro_dias", "caixa_investido_medio", "caixa_investido_pico",
          "dias_no_vermelho_max", "queda_max_sob_agua", "posicoes_abertas_no_fim",
          "prejuizo_nao_realizado"):
    print(f"  {k:32} {rb.stats[k]}")
if rb.por_ativo["abertas_no_fim"]:
    print("  posições presas no fim da janela:")
    for s, d in rb.por_ativo["abertas_no_fim"].items():
        print(f"    {s:6} {d['tranches']} tranches, {d['dias']} dias, "
              f"pnl aberto R$ {d['pnl_aberto']:,.0f}")

# STRESS: PETR4 sozinha atravessando o crash 2010-2016 (de ~R$30 a ~R$4)
print("\n--- STRESS: PETR4 sozinha, 2010-01 a 2017-12 (o crash de -85%) ---")
petr = load(DATA / "petr4_d1.parquet")
win = {"PETR4": petr.loc["2010-01-01":"2017-12-31"]}
rs = run_grid_averaging(win, COSTS, cash_rate=CDI, initial_equity=EQ0, lot_size=LOT,
                        **base)
line("grid em PETR4 2010-2017", rs.equity)
for k in ("caixa_investido_pico", "dias_no_vermelho_max", "queda_max_sob_agua",
          "posicoes_abertas_no_fim", "prejuizo_nao_realizado", "max_drawdown"):
    print(f"  {k:32} {rs.stats[k]}")
