"""'Oops' / gap reversal (Larry Williams).

O preço abre além do range do dia anterior e volta para dentro dele no mesmo
dia -> fade o gap.
  Long : abre <= mínima de ontem (gap de baixa) e fecha de volta acima dela.
  Short: abre >= máxima de ontem (gap de alta) e fecha de volta abaixo dela.
Segura `hold_days` barras e zera. Filtra gaps pequenos por `min_gap_pct`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class GapReversal:
    hold_days: int = 3
    min_gap_pct: float = 0.005
    allow_short: bool = False
    name: str = "gap_reversal"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        o = df["open"].to_numpy(dtype="float64")
        h = df["high"].to_numpy(dtype="float64")
        l = df["low"].to_numpy(dtype="float64")
        c = df["close"].to_numpy(dtype="float64")
        n = len(df)

        fire = np.zeros(n)
        for i in range(1, n):
            gap_dn = o[i] <= l[i - 1] * (1.0 - self.min_gap_pct)
            gap_up = o[i] >= h[i - 1] * (1.0 + self.min_gap_pct)
            if gap_dn and c[i] > l[i - 1]:
                fire[i] = 1.0
            elif self.allow_short and gap_up and c[i] < h[i - 1]:
                fire[i] = -1.0

        pos = np.zeros(n)
        cur, left = 0.0, 0
        for i in range(n):
            if fire[i] != 0.0:
                cur, left = fire[i], self.hold_days
            if left > 0:
                pos[i] = cur
                left -= 1
            else:
                cur = 0.0
        return pd.Series(pos, index=df.index, name="target")
