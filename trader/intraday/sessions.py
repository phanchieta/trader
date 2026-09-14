"""Chaves de sessão em horário de Brasília (sem horário de verão desde 2019)."""
from __future__ import annotations

import numpy as np
import pandas as pd

CLOSE_US_DST, CLOSE_STD = 16 * 60 + 54, 17 * 60 + 54     # última barra M1 do à vista


def brt_keys(index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """(dia YYYYMMDD, minuto do dia) de cada barra, em BRT."""
    loc = index.tz_convert("America/Sao_Paulo")
    day = np.asarray(loc.year * 10000 + loc.month * 100 + loc.day, dtype=np.int64)
    minute = np.asarray(loc.hour * 60 + loc.minute, dtype=np.int64)
    return day, minute


def cash_close_minute(stock_index: pd.DatetimeIndex) -> pd.Series:
    """Minuto da última barra do pregão à vista por dia, encaixado na grade da B3
    (16:54 com horário de verão americano, 17:54 sem)."""
    day, minute = brt_keys(stock_index)
    last = pd.Series(minute).groupby(day).max()
    mid = (CLOSE_US_DST + CLOSE_STD) // 2
    return pd.Series(np.where(last.to_numpy() >= mid, CLOSE_STD, CLOSE_US_DST),
                     index=last.index, name="cash_close")


class BarIndex:
    """Busca vetorizada de barra por (dia, minuto)."""

    def __init__(self, index: pd.DatetimeIndex):
        self.day, self.minute = brt_keys(index)
        self.key = self.day * 10000 + self.minute
        self.n = len(index)

    def at_or_after(self, day, minute) -> np.ndarray:
        day, minute = np.asarray(day), np.asarray(minute)
        i = np.searchsorted(self.key, day * 10000 + minute, side="left")
        j = np.minimum(i, self.n - 1)
        return np.where((i < self.n) & (self.day[j] == day), i, -1)

    def at_or_before(self, day, minute) -> np.ndarray:
        day, minute = np.asarray(day), np.asarray(minute)
        i = np.searchsorted(self.key, day * 10000 + minute, side="right") - 1
        j = np.maximum(i, 0)
        return np.where((i >= 0) & (self.day[j] == day), i, -1)

    def day_bounds(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(dias, início, fim exclusivo) de cada dia."""
        days, start = np.unique(self.day, return_index=True)
        end = np.r_[start[1:], self.n]
        return days, start, end
