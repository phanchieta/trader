"""Comparação de horizontes: as mesmas estratégias em D1 / M15 / M5, na MESMA
janela de tempo, com validação out-of-sample (holdout + walk-forward) e o
modelo de custo do horizonte.

Objetivo: ver qual horizonte sobrevive aos custos OOS — fecha a decisão de
horizonte da Fase 1. NÃO é otimização final; é triagem.

Dados: tudo MT5 (preço bruto B3, sem ajuste de dividendo -> conservador para
posições longas). Pré-requisitos:
    python scripts/pull_mt5_history.py D1 M15 M5
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from trader.backtest.split import WalkForward
from trader.backtest.validation import holdout, walk_forward
from trader.costs.b3 import B3Costs
from trader.data.store import load
from trader.strategy import MeanReversion, Momentum, OpeningRangeBreakout, SmaCrossover

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
DATA = Path(cfg["data"]["dir"])
EQ0, LOT = cfg["backtest"]["initial_equity"], cfg["backtest"]["lot_size"]
c = cfg["costs"]

COSTS_SWING = B3Costs.from_config(cfg)
COSTS_DAYTRADE = B3Costs(emolumentos_pct=c["emolumentos_pct_daytrade"],
                         corretagem_por_ordem=c["corretagem_por_ordem"],
                         iss_sobre_corretagem=c["iss_sobre_corretagem"],
                         slippage_ticks=c["slippage_ticks"], tick_size=c["tick_size"])

CLASSES = {"mean_reversion": MeanReversion, "momentum": Momentum,
           "sma_crossover": SmaCrossover, "opening_range_breakout": OpeningRangeBreakout}
DAYTRADE = {"opening_range_breakout"}

FILES = {"D1": ["petr4_mt5_d1.parquet", "petr4_d1.parquet"],
         "M15": ["petr4_mt5_m15.parquet", "petr4_yahoo_15m.parquet"],
         "M5": ["petr4_mt5_m5.parquet"]}

# grades em nº de BARRAS (escalam com a frequência)
GRIDS = {
    "D1": {
        "mean_reversion": {"lookback": [10, 20, 40], "entry_z": [1.0, 1.5, 2.0],
                           "exit_z": [0.0, 0.5], "allow_short": [False]},
        "momentum": {"lookback": [40, 120, 200], "band": [0.0, 0.02],
                     "trend_filter": [0, 200], "allow_short": [False]},
        "sma_crossover": {"fast": [10, 20, 50], "slow": [50, 100, 200],
                          "allow_short": [False]},
    },
    "M15": {
        "mean_reversion": {"lookback": [28, 56, 112], "entry_z": [1.0, 1.5, 2.0],
                           "exit_z": [0.0, 0.5], "allow_short": [False]},
        "momentum": {"lookback": [56, 168, 336], "band": [0.0, 0.005],
                     "trend_filter": [0, 336], "allow_short": [False]},
        "opening_range_breakout": {"opening_minutes": [15, 30, 45],
                                   "allow_short": [False, True],
                                   "session_end": ["16:55"],
                                   "session_tz": ["America/Sao_Paulo"]},
    },
    "M5": {
        "mean_reversion": {"lookback": [84, 168, 336], "entry_z": [1.0, 1.5, 2.0],
                           "exit_z": [0.0, 0.5], "allow_short": [False]},
        "momentum": {"lookback": [168, 504, 1008], "band": [0.0, 0.003],
                     "trend_filter": [0, 1008], "allow_short": [False]},
        "opening_range_breakout": {"opening_minutes": [15, 25, 35],
                                   "allow_short": [False, True],
                                   "session_end": ["16:55"],
                                   "session_tz": ["America/Sao_Paulo"]},
    },
}


def _load(tf):
    for name in FILES[tf]:
        p = DATA / name
        if p.exists():
            return load(p), name
    return None, None


loaded = {}
for tf in GRIDS:
    df, src = _load(tf)
    if df is None:
        print(f"[{tf}] sem arquivo ({' ou '.join(FILES[tf])}) — pulando")
        continue
    loaded[tf] = (df, src)

if not loaded:
    sys.exit("nenhum arquivo de dados encontrado")

common_start = max(df.index[0] for df, _ in loaded.values())
common_end = min(df.index[-1] for df, _ in loaded.values())
print(f"janela comum: {common_start.date()} .. {common_end.date()}\n")

WF = WalkForward(n_folds=4, test_frac=0.40)
rows = []
for tf, (df, src) in loaded.items():
    df = df.loc[common_start:common_end]
    pregoes = len(set(df.index.tz_convert("America/Sao_Paulo").date))
    print(f"[{tf}] {src}: {len(df)} barras | {pregoes} pregões")
    for strat, grid in GRIDS[tf].items():
        costs = COSTS_DAYTRADE if strat in DAYTRADE else COSTS_SWING
        try:
            h = holdout(CLASSES[strat], grid, df, costs, test_frac=0.30,
                        min_trades=15, initial_equity=EQ0, lot_size=LOT)
            w = walk_forward(CLASSES[strat], grid, df, costs, wf=WF,
                             min_trades=15, initial_equity=EQ0, lot_size=LOT)
        except Exception as e:  # noqa: BLE001
            print(f"    {strat:22} erro: {e}")
            continue
        rows.append({
            "tf": tf, "strat": strat,
            "wf_sharpe": w.oos_stats.get("sharpe"),
            "wf_ret": w.oos_stats.get("retorno_total"),
            "wf_cost": w.oos_stats.get("custo_sobre_lucro_bruto"),
            "wf_trades": w.oos_stats.get("n_trades"),
            "wf_dsharpe": w.degradacao().get("sharpe"),
            "ho_sharpe": h.oos_stats.get("sharpe"),
            "wf_equity": w.oos_equity,
        })

rows.sort(key=lambda r: (r["wf_sharpe"] is None, -(r["wf_sharpe"] or 0)))

print("\n" + "=" * 86)
print(f"{'horiz':6} {'estratégia':22} {'Sharpe WF':>10} {'Sharpe HO':>10} "
      f"{'ret WF':>9} {'custo/gr':>9} {'trades':>7} {'ΔSharpe':>8}")
print("-" * 86)
for r in rows:
    print(f"{r['tf']:6} {r['strat']:22} {str(r['wf_sharpe']):>10} {str(r['ho_sharpe']):>10} "
          f"{str(r['wf_ret']):>9} {str(r['wf_cost']):>9} {str(r['wf_trades']):>7} "
          f"{str(r['wf_dsharpe']):>8}")

print("\nWF = walk-forward 4 folds (mais honesto). HO = holdout 70/30 (cross-check).")
print("ΔSharpe = OOS − IS. custo/gr alto ou ΔSharpe muito longe de 0 = desconfiar.\n")

print("retorno OOS (walk-forward) por ano — top 3 por Sharpe WF:")
for r in [x for x in rows if x["wf_sharpe"] is not None][:3]:
    eq = r["wf_equity"]
    per_year = {int(y): round(float(s.iloc[-1] / s.iloc[0] - 1), 3)
                for y, s in eq.groupby(eq.index.year)}
    print(f"  {r['tf']:4} {r['strat']:22} {per_year}")
