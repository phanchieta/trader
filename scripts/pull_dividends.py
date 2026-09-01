"""Histórico de proventos (Yahoo) do universo de fatores -> data/dividends.parquet
(long-form: symbol, ex_date, amount). Para o fator dividend yield.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml
import yfinance as yf

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
syms = cfg["factor_universe"]["symbols"]
OUT = Path(cfg["data"]["dir"]) / "dividends.parquet"

rows, n_ok = [], 0
for s in syms:
    try:
        d = yf.Ticker(f"{s}.SA").dividends
        if d is None or len(d) == 0:
            continue
        idx = pd.to_datetime(d.index, utc=True)
        for dt, amt in zip(idx, d.to_numpy()):
            rows.append({"symbol": s, "ex_date": dt, "amount": float(amt)})
        n_ok += 1
    except Exception:  # noqa: BLE001
        pass

df = pd.DataFrame(rows).sort_values(["symbol", "ex_date"])
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT, index=False)
print(f"{n_ok} ações com proventos, {len(df)} eventos -> {OUT}")
print(f"período: {df['ex_date'].min().date()} .. {df['ex_date'].max().date()}")
