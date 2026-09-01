"""Breakout do range de abertura (intraday).

Em cada pregão, mede a máxima e a mínima dos primeiros `opening_minutes`.
Depois disso: compra no rompimento da máxima, vende no rompimento da mínima
(se permitido), zera antes de `session_end` (leilão de fechamento).

Precisa de barras INTRADAY. Levanta erro se receber dados diários.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OpeningRangeBreakout:
    opening_minutes: int = 15
    session_tz: str = "America/Sao_Paulo"
    session_end: str = "16:55"
    allow_short: bool = True
    name: str = "opening_range_breakout"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        step = pd.Series(df.index).diff().dropna().median()
        if not isinstance(step, pd.Timedelta) or step >= pd.Timedelta(hours=6):
            raise ValueError(
                "opening_range_breakout precisa de barras intraday "
                f"(espaçamento detectado: {step})"
            )

        local = df.index.tz_convert(self.session_tz)
        mod = (local.hour * 60 + local.minute).to_numpy()        # minuto do dia
        # chave de dia via componentes (não re-localiza -> imune a buracos de DST)
        day_key = (local.year * 10000 + local.month * 100 + local.day).to_numpy()
        hi = df["high"].to_numpy(dtype="float64")
        lo = df["low"].to_numpy(dtype="float64")
        cl = df["close"].to_numpy(dtype="float64")
        end_mod = int(self.session_end[:2]) * 60 + int(self.session_end[3:])

        out = np.zeros(len(df))
        n = len(df)
        start = 0
        for i in range(1, n + 1):
            if i < n and day_key[i] == day_key[start]:
                continue
            sl = slice(start, i)
            m = mod[sl]
            in_or = m < m[0] + self.opening_minutes
            if in_or.any() and not in_or.all():
                or_high = hi[sl][in_or].max()
                or_low = lo[sl][in_or].min()
                c = cl[sl]
                state = 0.0
                seg = np.zeros(i - start)
                for k in range(i - start):
                    if in_or[k]:
                        seg[k] = 0.0
                        continue
                    if m[k] >= end_mod:
                        state = 0.0
                    elif state == 0.0:
                        if c[k] > or_high:
                            state = 1.0
                        elif self.allow_short and c[k] < or_low:
                            state = -1.0
                    seg[k] = state
                out[sl] = seg
            start = i
        return pd.Series(out, index=df.index, name="target")
