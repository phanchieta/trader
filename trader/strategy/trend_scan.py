"""Trend-scanning (adaptado do artigo MQL5 23660).

Ajusta uma reta ao log-preço numa janela móvel e usa a ESTATÍSTICA t do
coeficiente angular: |t| grande = tendência estatisticamente significativa.
Comprado quando t > `t_threshold`; vendido quando t < −`t_threshold` (se
permitido); fora entre eles (histerese via `exit_threshold`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def rolling_tvalue(y: np.ndarray, window: int) -> np.ndarray:
    n = len(y)
    x = np.arange(window, dtype="float64")
    xm = x.mean()
    xc = x - xm
    sxx = float((xc ** 2).sum())
    out = np.zeros(n)
    if window <= 2 or sxx == 0.0:
        return out
    for i in range(window - 1, n):
        w = y[i - window + 1: i + 1]
        slope = float((xc * (w - w.mean())).sum()) / sxx
        resid = w - (w.mean() + slope * xc)
        ssr = float((resid ** 2).sum())
        se = np.sqrt(ssr / (window - 2) / sxx) if ssr > 0 else 0.0
        out[i] = slope / se if se > 0 else 0.0
    return out


@dataclass
class TrendScan:
    window: int = 40
    t_threshold: float = 2.0
    exit_threshold: float = 0.5
    allow_short: bool = False
    name: str = "trend_scan"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        t = rolling_tvalue(np.log(df["close"].to_numpy(dtype="float64")), self.window)
        n = len(df)
        pos = np.zeros(n)
        state = 0.0
        for i in range(n):
            if state == 0.0:
                if t[i] > self.t_threshold:
                    state = 1.0
                elif self.allow_short and t[i] < -self.t_threshold:
                    state = -1.0
            elif state > 0.0 and t[i] < self.exit_threshold:
                state = 0.0
            elif state < 0.0 and t[i] > -self.exit_threshold:
                state = 0.0
            pos[i] = state
        return pd.Series(pos, index=df.index, name="target")
