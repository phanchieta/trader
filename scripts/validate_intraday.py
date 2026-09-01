"""Validação out-of-sample no dado intraday (Yahoo 15m backfill).

AVISO: 60 pregões é um regime só de mercado — serve para exercitar a lógica
intraday, NÃO para concluir a comparação de horizontes. Isso pede o histórico
que o coletor MT5 vai acumulando.

Pré-requisito: python scripts/pull_intraday.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from trader.backtest.validation import holdout
from trader.costs.b3 import B3Costs
from trader.data.store import load
from trader.strategy import OpeningRangeBreakout

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
_interval = cfg["data"]["yahoo_intraday_interval"]
df = load(Path(cfg["data"]["dir"]) / f"petr4_yahoo_{_interval}.parquet")
eq0 = cfg["backtest"]["initial_equity"]
lot = cfg["backtest"]["lot_size"]

# estratégia intraday -> custos de day trade
c = cfg["costs"]
costs = B3Costs(emolumentos_pct=c["emolumentos_pct_daytrade"],
                corretagem_por_ordem=c["corretagem_por_ordem"],
                iss_sobre_corretagem=c["iss_sobre_corretagem"],
                slippage_ticks=c["slippage_ticks"], tick_size=c["tick_size"])

grid = {"opening_minutes": [15, 30, 45], "allow_short": [False, True],
        "session_end": ["16:55"], "session_tz": ["America/Sao_Paulo"]}

pregoes = len(set(df.index.tz_convert("America/Sao_Paulo").date))
print(f"{len(df)} barras 15m | {pregoes} pregões | {df.index[0].date()} .. {df.index[-1].date()}")
print("=" * 60)

rep = holdout(OpeningRangeBreakout, grid, df, costs,
              test_frac=0.30, min_trades=5, initial_equity=eq0, lot_size=lot)

print(f"\nparâmetros (escolhidos no treino): {rep.best_params}")
print(f"  {'métrica':18} {'IS (treino)':>13} {'OOS (novo)':>13}")
for label, key in [("Sharpe", "sharpe"), ("retorno_total", "retorno_total"),
                   ("max_drawdown", "max_drawdown"), ("n_trades", "n_trades"),
                   ("custo/lucro_bruto", "custo_sobre_lucro_bruto")]:
    print(f"  {label:18} {str(rep.is_stats.get(key)):>13} {str(rep.oos_stats.get(key)):>13}")
print(f"  degradação OOS-IS: {rep.degradacao()}")
print("\n(amostra minúscula — leia como 'o pipeline roda', não como resultado)")
