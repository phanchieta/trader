"""Rompimento de canal (Donchian / Turtle).

Compra quando o fechamento supera a máxima dos últimos `entry_lookback` dias
(medida até ontem); sai quando perde a mínima dos últimos `exit_lookback` dias.
Espelhado para venda quando `allow_short`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Donchian:
    entry_lookback: int = 20
    exit_lookback: int = 10
    allow_short: bool = False
    name: str = "donchian"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        he = df["high"].rolling(self.entry_lookback).max().shift(1).to_numpy()
        le = df["low"].rolling(self.entry_lookback).min().shift(1).to_numpy()
        hx = df["high"].rolling(self.exit_lookback).max().shift(1).to_numpy()
        lx = df["low"].rolling(self.exit_lookback).min().shift(1).to_numpy()
        c = df["close"].to_numpy(dtype="float64")
        n = len(df)

        pos = np.zeros(n)
        state = 0.0
        for i in range(n):
            if state == 0.0:
                if np.isfinite(he[i]) and c[i] > he[i]:
                    state = 1.0
                elif self.allow_short and np.isfinite(le[i]) and c[i] < le[i]:
                    state = -1.0
            elif state == 1.0:
                if np.isfinite(lx[i]) and c[i] < lx[i]:
                    state = 0.0
            elif state == -1.0:
                if np.isfinite(hx[i]) and c[i] > hx[i]:
                    state = 0.0
            pos[i] = state
        return pd.Series(pos, index=df.index, name="target")
