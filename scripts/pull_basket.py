"""Baixa o histórico diário da cesta (Yahoo) para Parquet. Rodar da raiz do repo."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from trader.data.store import save
from trader.data.yahoo import daily

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
syms = cfg["basket"]["symbols"]
suffix = cfg["basket"]["yahoo_suffix"]
outdir = Path(cfg["data"]["dir"])

for sym in syms:
    df = daily(f"{sym}{suffix}")
    out = outdir / f"{sym.lower()}_d1.parquet"
    save(df, out)
    print(f"{sym:6} {len(df):>5} barras | {df.index[0].date()} .. {df.index[-1].date()} -> {out}")
