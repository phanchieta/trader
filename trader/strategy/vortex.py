"""Vortex Indicator (Botes & Siepman). Trend-following por cruzamento:
VI+ = Σ|high − low_ontem| / ΣTR ;  VI− = Σ|low − high_ontem| / ΣTR.
Comprado quando VI+ > VI−; vendido no inverso (se allow_short).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..indicators import true_range


@dataclass
class Vortex:
    period: int = 14
    allow_short: bool = False
    name: str = "vortex"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        vm_plus = (df["high"] - df["low"].shift(1)).abs()
        vm_minus = (df["low"] - df["high"].shift(1)).abs()
        tr_sum = true_range(df).rolling(self.period).sum()
        vi_plus = vm_plus.rolling(self.period).sum() / tr_sum
        vi_minus = vm_minus.rolling(self.period).sum() / tr_sum

        sig = (vi_plus > vi_minus).astype("float64")
        if self.allow_short:
            sig = sig - (vi_minus > vi_plus).astype("float64")
        sig[vi_plus.isna() | vi_minus.isna()] = 0.0
        return sig.rename("target")
