"""SuperTrend (bandas de ATR com catraca). Trend-following.

lower = (high+low)/2 − mult·ATR  (suporte na alta; só sobe enquanto o preço fica acima)
upper = (high+low)/2 + mult·ATR  (resistência na baixa; só desce enquanto o preço fica abaixo)
A tendência vira para alta quando o fechamento rompe `upper`; para baixa quando perde `lower`.
Sinal = direção da tendência (+1 comprado; −1 se allow_short, senão 0).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..indicators import atr


@dataclass
class SuperTrend:
    period: int = 10
    multiplier: float = 3.0
    allow_short: bool = False
    name: str = "supertrend"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        a = atr(df, self.period).to_numpy(dtype="float64")
        hl2 = ((df["high"] + df["low"]) / 2.0).to_numpy(dtype="float64")
        close = df["close"].to_numpy(dtype="float64")
        n = len(df)

        base_lower = hl2 - self.multiplier * a
        base_upper = hl2 + self.multiplier * a

        trend = np.zeros(n)
        lower = upper = None
        t = 1.0
        for i in range(n):
            if i == 0 or not np.isfinite(a[i]):
                trend[i] = 0.0
                continue
            if lower is None:                       # primeira barra válida: semeia
                lower, upper, t = base_lower[i], base_upper[i], 1.0
                trend[i] = t
                continue
            lower = max(base_lower[i], lower) if close[i - 1] > lower else base_lower[i]
            upper = min(base_upper[i], upper) if close[i - 1] < upper else base_upper[i]
            if t < 0 and close[i] > upper:
                t = 1.0
            elif t > 0 and close[i] < lower:
                t = -1.0
            trend[i] = t

        out = trend if self.allow_short else np.where(trend > 0.0, 1.0, 0.0)
        return pd.Series(out, index=df.index, name="target")
