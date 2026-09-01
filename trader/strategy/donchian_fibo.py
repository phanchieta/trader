"""Donchian Fibo + ATR stop (baseado no indicador de Guilherme Santos / Damiani).

Canal: máxima/mínima de `period` barras. Níveis de Fibonacci internos
(0 = fundo, 1 = topo). Ideia de trade:
  - viés de alta quando o topo do canal está subindo;
  - entra comprado no pullback até o nível `entry_fib` (ex. 0,5 / 0,618);
  - sai por stop ATR "chandelier" (máx. fechamento desde a entrada − mult·ATR)
    ou quando perde o fundo do canal.
Espelhado para venda quando `allow_short`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..indicators import atr


@dataclass
class DonchianFibo:
    period: int = 72
    entry_fib: float = 0.5
    trend_lookback: int = 10
    atr_period: int = 14
    atr_mult: float = 3.0
    allow_short: bool = False
    name: str = "donchian_fibo"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        hi = df["high"].rolling(self.period).max().shift(1)
        lo = df["low"].rolling(self.period).min().shift(1)
        diff = (hi - lo).where(lambda x: x > 0)
        lvl_long = (lo + diff * self.entry_fib).to_numpy()
        lvl_short = (lo + diff * (1.0 - self.entry_fib)).to_numpy()
        hv, lv = hi.to_numpy(), lo.to_numpy()
        a = atr(df, self.atr_period).to_numpy(dtype="float64")
        c = df["close"].to_numpy(dtype="float64")
        n = len(df)

        k = self.trend_lookback
        pos = np.zeros(n)
        state, ref = 0.0, 0.0
        for i in range(n):
            if i < k or not (np.isfinite(hv[i]) and np.isfinite(lv[i])
                             and np.isfinite(a[i]) and np.isfinite(hv[i - k])):
                state, ref = 0.0, 0.0
                continue
            if state == 0.0:
                up_bias = hv[i] > hv[i - k]        # canal fez topo mais alto que há k barras
                dn_bias = lv[i] < lv[i - k]
                if up_bias and c[i] <= lvl_long[i]:
                    state, ref = 1.0, c[i]
                elif self.allow_short and dn_bias and c[i] >= lvl_short[i]:
                    state, ref = -1.0, c[i]
            elif state == 1.0:
                ref = max(ref, c[i])
                if c[i] < lv[i] or c[i] <= ref - self.atr_mult * a[i]:
                    state, ref = 0.0, 0.0
            elif state == -1.0:
                ref = min(ref, c[i])
                if c[i] > hv[i] or c[i] >= ref + self.atr_mult * a[i]:
                    state, ref = 0.0, 0.0
            pos[i] = state
        return pd.Series(pos, index=df.index, name="target")
