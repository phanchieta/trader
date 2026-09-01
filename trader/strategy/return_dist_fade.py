"""Fade da distribuição de retornos (adaptado do artigo MQL5 19797).

Mede o retorno de `ret_horizon` dias contra a distribuição dos últimos
`dist_window` dias (z-score). Quando o movimento é um extremo estatístico,
faz o contra: compra após queda anômala, vende após alta anômala (se permitido).
Sai quando o retorno normaliza, quando vira a favor, ou após `max_hold` dias.

`scaled=True` dimensiona o alvo pela magnitude do extremo (ideia de sizing
inverso à probabilidade do artigo), em [0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ReturnDistFade:
    dist_window: int = 120
    ret_horizon: int = 1
    entry_z: float = 2.0
    exit_z: float = 0.5
    max_hold: int = 10
    allow_short: bool = False
    scaled: bool = False
    name: str = "return_dist_fade"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        ret = df["close"].pct_change(self.ret_horizon)
        mu = ret.rolling(self.dist_window).mean()
        sd = ret.rolling(self.dist_window).std(ddof=0)
        z = ((ret - mu) / sd.where(sd > 0)).to_numpy()

        n = len(df)
        pos = np.zeros(n)
        state, held = 0.0, 0
        for i in range(n):
            if not np.isfinite(z[i]):
                state, held = 0.0, 0
                continue
            if state != 0.0:
                held += 1
                if (held >= self.max_hold or abs(z[i]) <= self.exit_z
                        or (state > 0 and z[i] > 0.0)
                        or (state < 0 and z[i] < 0.0)):
                    state, held = 0.0, 0
            if state == 0.0:
                if z[i] <= -self.entry_z:
                    state, held = 1.0, 0
                elif self.allow_short and z[i] >= self.entry_z:
                    state, held = -1.0, 0
            if state != 0.0:
                mag = min(1.0, abs(z[i]) / (2.0 * self.entry_z)) if self.scaled else 1.0
                pos[i] = state * mag
        return pd.Series(pos, index=df.index, name="target")
