"""Baixa o histórico diário da PETR4 do Yahoo e salva em Parquet. Rodar da raiz do repo."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # roda sem pip install -e .

import yaml

from trader.data.store import save
from trader.data.yahoo import daily

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
out = Path(cfg["data"]["dir"]) / cfg["data"]["daily_file"]

df = daily(cfg["yahoo_symbol"])
save(df, out)
print(f"{len(df)} barras diárias -> {out}")
print(f"período: {df.index[0].date()} .. {df.index[-1].date()}")
