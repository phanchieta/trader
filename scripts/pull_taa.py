"""Séries para alocação tática entre classes de ativo, todas negociáveis na B3.

ETF (Yahoo, ajustado)       referência para antes do ETF existir
BOVA11  Ibovespa             ^BVSP (índice já reinveste dividendos)
IVVB11  S&P 500 em reais     ^SP500TR (com dividendos) × USDBRL=X
GOLD11  ouro em reais        GLD (ouro físico, 2004+) × USDBRL=X
SMAL11  small caps           — (só desde 2009)

Salva data/taa_<nome>.parquet com o fechamento diário.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml
import yfinance as yf

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
OUT = Path(cfg["data"]["dir"])
TICKERS = {"bova11": "BOVA11.SA", "ivvb11": "IVVB11.SA", "gold11": "GOLD11.SA", "smal11": "SMAL11.SA",
           "bvsp": "^BVSP", "sp500tr": "^SP500TR", "gld": "GLD", "usdbrl": "USDBRL=X"}

for name, t in TICKERS.items():
    d = yf.download(t, start="1993-01-01", auto_adjust=True, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.droplevel(1)
    s = d["Close"].dropna()
    s = s[s > 0]
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s.rename("close").to_frame().to_parquet(OUT / f"taa_{name}.parquet")
    print(f"{name:8} {t:10} {len(s):>6} | {s.index[0].date()} .. {s.index[-1].date()}")
