"""Canais por quantil empírico (adaptado do artigo MQL5 23719, "distribution-free
price channels"). Bandas robustas: quantil `lower_q`/`upper_q` do fechamento nos
últimos `window` dias, em vez de média ± k·desvio.

  mode="revert"   : compra quando o preço encosta na banda inferior, sai no meio.
  mode="breakout" : compra quando rompe a banda superior, sai ao voltar ao meio.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class QuantileChannel:
    window: int = 60
    lower_q: float = 0.10
    upper_q: float = 0.90
    mode: str = "revert"
    exit_at: float = 0.5          # fração do canal onde a posição zera
    max_hold: int = 20
    allow_short: bool = False
    name: str = "quantile_channel"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        c = df["close"]
        lo = c.rolling(self.window).quantile(self.lower_q).shift(1)
        hi = c.rolling(self.window).quantile(self.upper_q).shift(1)
        mid = lo + (hi - lo) * self.exit_at

        cv, lov, hiv, midv = (c.to_numpy(), lo.to_numpy(), hi.to_numpy(), mid.to_numpy())
        revert = self.mode == "revert"
        n = len(df)
        pos = np.zeros(n)
        state, held = 0.0, 0
        for i in range(n):
            if not (np.isfinite(lov[i]) and np.isfinite(hiv[i])):
                state, held = 0.0, 0
                continue
            if state != 0.0:
                held += 1
                back_long = cv[i] >= midv[i] if revert else cv[i] < midv[i]
                back_short = cv[i] <= midv[i] if revert else cv[i] > midv[i]
                if (held >= self.max_hold
                        or (state > 0 and back_long) or (state < 0 and back_short)):
                    state, held = 0.0, 0
            if state == 0.0:
                if revert:
                    if cv[i] <= lov[i]:
                        state, held = 1.0, 0
                    elif self.allow_short and cv[i] >= hiv[i]:
                        state, held = -1.0, 0
                else:
                    if cv[i] > hiv[i]:
                        state, held = 1.0, 0
                    elif self.allow_short and cv[i] < lov[i]:
                        state, held = -1.0, 0
            pos[i] = state
        return pd.Series(pos, index=df.index, name="target")
