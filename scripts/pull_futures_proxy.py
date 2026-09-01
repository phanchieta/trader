"""Proxies de longo histórico para WIN (mini-índice) e WDO (mini-dólar):
  ^BVSP    (Ibovespa)  -> data/ibov_d1.parquet
  USDBRL=X (dólar)     -> data/usdbrl_d1.parquet
O sinal de trend só precisa da série de preço; o custo de futuro entra no engine.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml
import yfinance as yf

from trader.data.schema import OHLCV, coerce
from trader.data.store import save

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
OUT = Path(cfg["data"]["dir"])

for ticker, fname in [(cfg["futures"]["ibov_proxy"], "ibov_d1.parquet"),
                      (cfg["futures"]["usdbrl_proxy"], "usdbrl_d1.parquet")]:
    raw = yf.download(ticker, start="2000-01-01", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    df = raw.rename(columns=str.lower)[OHLCV].copy()
    df["volume"] = df["volume"].fillna(1.0)          # índice/câmbio: volume não importa
    df.index = pd.to_datetime(df.index, utc=True)
    df = coerce(df)
    save(df, OUT / fname)
    print(f"{ticker:10} {len(df):>5} barras | {df.index[0].date()} .. {df.index[-1].date()} -> {fname}")
