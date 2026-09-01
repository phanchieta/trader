"""Backfill de intraday da PETR4 via Yahoo. Rodar da raiz do repo.

    python scripts/pull_intraday.py            # usa o intervalo do settings.yaml
    python scripts/pull_intraday.py 1m         # sobrescreve o intervalo

Limites do Yahoo: 1m ~7 dias | 5m/15m/30m ~60 dias | 1h ~730 dias.
Para 1m contínuo daqui pra frente, use scripts/collect_intraday.py (MT5).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from trader.data.store import save
from trader.data.yahoo import intraday

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
interval = sys.argv[1] if len(sys.argv) > 1 else cfg["data"]["yahoo_intraday_interval"]
out = Path(cfg["data"]["dir"]) / f"petr4_yahoo_{interval}.parquet"

df = intraday(cfg["yahoo_symbol"], interval=interval)
save(df, out)

print(f"{len(df)} barras {interval} -> {out}")
print(f"período: {df.index[0]} .. {df.index[-1]}  (UTC)")
print(f"pregões cobertos: {df.index.normalize().nunique()}")
