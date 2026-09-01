"""Keltner Channel breakout. Canal = EMA ± mult·ATR (base em EMA, não em
máx/mín como o Donchian). Compra no rompimento da banda superior; sai ao
voltar para a EMA (meio). Espelhado para venda quando allow_short.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..indicators import atr


@dataclass
class Keltner:
    ema_period: int = 20
    atr_period: int = 20
    mult: float = 2.0
    allow_short: bool = False
    name: str = "keltner"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        ema = df["close"].ewm(span=self.ema_period, adjust=False).mean()
        a = atr(df, self.atr_period)
        upper = (ema + self.mult * a).shift(1).to_numpy()
        lower = (ema - self.mult * a).shift(1).to_numpy()
        mid = ema.shift(1).to_numpy()
        c = df["close"].to_numpy(dtype="float64")
        n = len(df)

        pos = np.zeros(n)
        state = 0.0
        for i in range(n):
            if not np.isfinite(upper[i]):
                state = 0.0
                continue
            if state == 0.0:
                if c[i] > upper[i]:
                    state = 1.0
                elif self.allow_short and c[i] < lower[i]:
                    state = -1.0
            elif state == 1.0 and c[i] < mid[i]:
                state = 0.0
            elif state == -1.0 and c[i] > mid[i]:
                state = 0.0
            pos[i] = state
        return pd.Series(pos, index=df.index, name="target")
