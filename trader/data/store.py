"""Persistência em Parquet, sempre passando pelo contrato de `schema.py`."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import coerce


def save(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    coerce(df).to_parquet(path)


def load(path: str | Path) -> pd.DataFrame:
    return coerce(pd.read_parquet(path))


def append_dedup(df_new: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    """Concatena com o que já existe, remove timestamps repetidos, salva."""
    path = Path(path)
    df = pd.concat([load(path), df_new]) if path.exists() else df_new
    df = coerce(df)
    save(df, path)
    return df
