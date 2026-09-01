"""Fonte diária via Yahoo Finance. Histórico longo (PETR4.SA desde ~2000).

Uso: base para pesquisa de swing e conferência cruzada do D1 do MT5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import OHLCV, coerce

_OHLC = ["open", "high", "low", "close"]

# Teto de histórico que o Yahoo devolve por resolução intraday (dias corridos).
_INTRADAY_MAX_DAYS = {"1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60,
                      "90m": 60, "1h": 730}


def _intraday_days(interval: str, lookback_days: int | None = None) -> int:
    interval = interval.lower().replace("60m", "1h")
    cap = _INTRADAY_MAX_DAYS.get(interval)
    if cap is None:
        raise ValueError(
            f"interval {interval!r} não suportado: {sorted(_INTRADAY_MAX_DAYS)}"
        )
    return min(lookback_days or cap, cap)


def _spike_mask(s: pd.Series, factor: float) -> pd.Series:
    up = (s / s.shift(1) > factor) & (s / s.shift(-1) > factor)
    down = (s.shift(1) / s > factor) & (s.shift(-1) / s > factor)
    return (up | down).fillna(False)


def _drop_single_bar_spikes(df: pd.DataFrame, factor: float = 3.0) -> pd.DataFrame:
    """Remove barras isoladas absurdas (glitch comum do Yahoo, ex.: PETR4 2003-08-05,
    open 10x revertido no dia seguinte). Substitui por interpolação linear."""
    bad = _spike_mask(df["close"], factor) | _spike_mask(df["open"], factor)
    if bad.any():
        for ts in df.index[bad]:
            print(f"[yahoo] spike de 1 barra corrigido em {ts.date()}")
        df.loc[bad, _OHLC] = np.nan
        df[_OHLC] = df[_OHLC].interpolate(limit_direction="both")
    return df


def daily(symbol: str = "PETR4.SA", start: str = "2000-01-01",
          drop_spikes: bool = True) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(symbol, start=start, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"Yahoo não retornou dados para {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):          # versões novas do yfinance
        raw.columns = raw.columns.droplevel(1)
    df = raw.rename(columns=str.lower)[OHLCV].copy()
    df.index = pd.to_datetime(df.index, utc=True)
    if drop_spikes:
        df = _drop_single_bar_spikes(df)
    return coerce(df)


def intraday(symbol: str = "PETR4.SA", interval: str = "15m",
             lookback_days: int | None = None, drop_spikes: bool = True) -> pd.DataFrame:
    """Backfill intraday. Limitado pelo Yahoo: 1m ~7d, 5m/15m/30m ~60d, 1h ~730d.
    Para 1m contínuo daqui pra frente, use o coletor MT5."""
    import yfinance as yf

    interval = interval.lower().replace("60m", "1h")
    days = _intraday_days(interval, lookback_days)
    raw = yf.download(symbol, period=f"{days}d", interval=interval,
                      auto_adjust=True, progress=False, prepost=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"Yahoo não retornou intraday {interval} para {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    df = raw.rename(columns=str.lower)[OHLCV].copy()
    df.index = pd.to_datetime(df.index, utc=True)     # Yahoo devolve tz-aware -> UTC
    if drop_spikes:
        df = _drop_single_bar_spikes(df)
    return coerce(df)
