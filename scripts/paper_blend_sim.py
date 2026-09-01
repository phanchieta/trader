"""Simulador do BLEND — roda em paralelo ao paper_trade.py do campeão SEM MT5 e
SEM enviar ordem. Marca a carteira a fechamento, credita CDI no caixa, aplica
custo de giro. Serve para comparar "mais risco / mais retorno" (blend) com o
"conservador" (campeão real) ao longo das semanas.

Estado em data/blend_sim.json ; curva em data/blend_sim_equity.csv.
Rodar diário (cron), depois de pull_factor_universe.py + pull_cdi.py.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml

from trader.data.store import load
from trader.live import journal
from trader.live.blend_planner import plan_blend

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
DATA = Path(cfg["data"]["dir"])
LOT = cfg["backtest"]["lot_size"]
CB_STOCK = cfg["futures"]["cost_bps_stock"] / 1e4
STATE = DATA / "blend_sim.json"
CURVE = DATA / "blend_sim_equity.csv"

# começa no mesmo capital do paper-trade real do campeão, para a comparação
# "conservador vs mais risco" ficar na mesma escala
_champ_j = DATA / "paper_journal.csv"
if _champ_j.exists():
    EQ0 = float(pd.read_csv(_champ_j)["equity"].iloc[0])
else:
    EQ0 = float(cfg["backtest"]["initial_equity"])

champ = {s: load(DATA / f"{s.lower()}_d1.parquet") for s in cfg["basket"]["symbols"]}
uni = {s: load(DATA / f"{s.lower()}_d1.parquet")
       for s in cfg["factor_universe"]["symbols"]
       if (DATA / f"{s.lower()}_d1.parquet").exists()}
IBOV = load(DATA / "ibov_d1.parquet")
CDI = pd.read_parquet(DATA / "cdi_d1.parquet")["rate"]

px_of = lambda s: float((champ if s in champ else uni)[s]["close"].iloc[-1])
today = date.today()

st = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {
    "cash": EQ0, "positions": {}, "factor": {}, "updated": ""}
cash = float(st["cash"])
pos = {k: float(v) for k, v in st["positions"].items()}

# CDI acumulado desde a última execução
if st["updated"]:
    gap = CDI.loc[pd.Timestamp(st["updated"], tz="UTC"):].iloc[1:]
    cash *= float((1.0 + gap).prod()) if len(gap) else 1.0

equity = cash + sum(sh * px_of(s) for s, sh in pos.items())

plan = plan_blend(champ, uni, IBOV, equity=equity, today=today,
                  prev_quintile=st["factor"].get("quintile", []),
                  last_rebalance=st["factor"].get("last_rebalance"),
                  champ_w=cfg["blend"]["champion_weight"],
                  n_frac=cfg["blend"]["factor_quintile"], lot=LOT,
                  target_vol=cfg["risk_overlay"]["target_ann_vol"],
                  vol_lb=cfg["risk_overlay"]["vol_lookback"],
                  max_lev=cfg["risk_overlay"]["max_leverage"], min_names=20)

# "executa" a fechamento
turn = 0.0
for s in set(plan.target_shares) | set(pos):
    tgt = plan.target_shares.get(s, 0.0)
    cur = pos.get(s, 0.0)
    d = tgt - cur
    if d != 0.0:
        p = px_of(s)
        cash -= d * p + abs(d) * p * CB_STOCK
        turn += abs(d) * p
        if tgt == 0.0:
            pos.pop(s, None)
        else:
            pos[s] = tgt

equity = cash + sum(sh * px_of(s) for s, sh in pos.items())
st = {"cash": cash, "positions": pos,
      "factor": ({"quintile": plan.quintile, "last_rebalance": today.isoformat()}
                 if plan.rebalanced else st["factor"]),
      "updated": today.isoformat()}
STATE.write_text(json.dumps(st, indent=2), encoding="utf-8")
journal.append(CURVE, {"ts": today.isoformat(), "equity": round(equity, 2),
                       "pos_value": round(equity - cash, 2),
                       "n_positions": len(pos), "detail": json.dumps(pos)})

for n in plan.notes:
    print(f"[fatores] {n}")
print(f"blend sim | equity R$ {equity:,.2f} | {len(pos)} posições | "
      f"{(equity - cash) / equity:.1%} investido | giro do dia R$ {turn:,.0f}")
print(f"retorno acumulado: {equity / EQ0 - 1:+.2%}  -> {CURVE}")
