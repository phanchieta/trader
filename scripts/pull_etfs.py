"""Diário de ETFs da B3 no Yahoo (preços negociados de verdade — a abertura do
^BVSP no Yahoo é praticamente o fechamento anterior e não serve para overnight)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from trader.data.store import save
from trader.data.yahoo import daily

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
OUT = Path(cfg["data"]["dir"])
for sym in ("BOVA11",):
    df = daily(f"{sym}.SA")
    save(df, OUT / f"{sym.lower()}_d1.parquet")
    print(f"{sym}: {len(df)} barras | {df.index[0].date()} .. {df.index[-1].date()}")
