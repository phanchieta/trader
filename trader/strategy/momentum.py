"""Momentum de série temporal — sinal do retorno passado de `lookback` barras.

Comprado se o retorno dos últimos `lookback` períodos > `band`; vendido se
< -`band` (quando permitido). Filtro de tendência opcional: só fica comprado
acima da SMA(`trend_filter`), só vendido abaixo dela.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Momentum:
    lookback: int = 120
    band: float = 0.0
    trend_filter: int = 0
    allow_short: bool = False
    name: str = "momentum"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        c = df["close"]
        past_ret = c / c.shift(self.lookback) - 1.0

        sig = pd.Series(0.0, index=df.index, name="target")
        sig[past_ret > self.band] = 1.0
        if self.allow_short:
            sig[past_ret < -self.band] = -1.0

        if self.trend_filter > 0:
            sma = c.rolling(self.trend_filter).mean()
            sig[(sig > 0) & (c < sma)] = 0.0
            sig[(sig < 0) & (c > sma)] = 0.0
            sig[sma.isna()] = 0.0

        sig[past_ret.isna()] = 0.0
        return sig
