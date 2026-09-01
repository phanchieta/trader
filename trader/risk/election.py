"""Overlay de evento eleitoral: reduz a exposição numa janela pré-comprometida
ao redor das eleições gerais. Simétrico — não aposta no resultado, só corta
risco perto de incerteza alta.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# anos de eleição geral no Brasil + o próximo
ELECTION_YEARS = (2010, 2014, 2018, 2022, 2026)


def haircut_series(index: pd.DatetimeIndex, *, factor: float = 0.5,
                   start_md: tuple[int, int] = (9, 15),
                   end_md: tuple[int, int] = (11, 20),
                   years=ELECTION_YEARS) -> pd.Series:
    """1.0 fora da janela; `factor` (ex.: 0,5) de 15/set a 20/nov dos anos de
    eleição. Multiplique pelo multiplicador de vol-target."""
    local = index.tz_convert("America/Sao_Paulo") if index.tz is not None else index
    y, mo, dy = local.year.to_numpy(), local.month.to_numpy(), local.day.to_numpy()
    ord_ = mo * 100 + dy
    lo, hi = start_md[0] * 100 + start_md[1], end_md[0] * 100 + end_md[1]
    in_win = np.isin(y, years) & (ord_ >= lo) & (ord_ <= hi)
    return pd.Series(np.where(in_win, factor, 1.0), index=index, name="election")
