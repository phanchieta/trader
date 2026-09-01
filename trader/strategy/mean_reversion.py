"""Reversão à média — z-score do fechamento contra a média móvel.

Entra comprado quando o preço está muito abaixo da média (z <= -entry_z) e
sai quando volta para perto dela (z >= -exit_z). Simétrico para vendido.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MeanReversion:
    lookback: int = 20
    entry_z: float = 1.5
    exit_z: float = 0.5
    allow_short: bool = False
    name: str = "mean_reversion"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        c = df["close"]
        ma = c.rolling(self.lookback).mean()
        sd = c.rolling(self.lookback).std(ddof=0)
        z = ((c - ma) / sd.where(sd > 0)).to_numpy()

        pos = np.zeros(len(df))
        state = 0.0
        for i in range(len(df)):
            zi = z[i]
            if not np.isfinite(zi):
                state = 0.0
            elif state == 0.0:
                if zi <= -self.entry_z:
                    state = 1.0
                elif self.allow_short and zi >= self.entry_z:
                    state = -1.0
            elif state == 1.0 and zi >= -self.exit_z:
                state = 0.0
            elif state == -1.0 and zi <= self.exit_z:
                state = 0.0
            pos[i] = state
        return pd.Series(pos, index=df.index, name="target")
