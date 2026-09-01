"""Contrato de dados OHLCV.

Toda fonte (Yahoo, MT5, ...) devolve um DataFrame:
  - index: DatetimeIndex tz-aware em UTC, nome 'time', ordenado, sem duplicatas
  - colunas exatamente: open, high, low, close, volume  (float64)
"""
from __future__ import annotations

import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]


def validate(df: pd.DataFrame) -> pd.DataFrame:
    if list(df.columns) != OHLCV:
        raise ValueError(f"colunas devem ser {OHLCV}, vieram {list(df.columns)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("index deve ser DatetimeIndex")
    if df.index.tz is None:
        raise ValueError("index deve ser tz-aware (UTC)")
    if df.index.name != "time":
        raise ValueError("index.name deve ser 'time'")
    if not df.index.is_monotonic_increasing:
        raise ValueError("index deve estar ordenado")
    if df.index.has_duplicates:
        raise ValueError("index tem timestamps duplicados")
    if df[OHLCV].isna().to_numpy().any():
        raise ValueError("há NaN em OHLCV")
    return df


def coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Ordena, remove duplicatas (mantém a última), força float, normaliza a
    resolução do índice para nanossegundos, valida.

    A normalização evita que Parquet lido em ms/us quebre cálculos que assumem ns.
    """
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.astype({c: "float64" for c in OHLCV})
    df = df.dropna(subset=OHLCV)                       # linhas incompletas do vendor
    if hasattr(df.index, "as_unit"):
        df.index = df.index.as_unit("ns")
    df.index.name = "time"
    return validate(df)
