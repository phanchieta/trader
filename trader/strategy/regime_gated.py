"""Wrapper de regime: só deixa a estratégia base operar quando o mercado está
DIRECIONAL. Usa o Efficiency Ratio de Kaufman — abaixo de `er_min` (lateral),
zera o sinal. Objetivo: cortar o drawdown de estratégias trend em mercado de chop.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..indicators import efficiency_ratio


@dataclass
class RegimeGated:
    base: object
    er_window: int = 20
    er_min: float = 0.30
    name: str = "regime_gated"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        sig = self.base.generate_signals(df).reindex(df.index).fillna(0.0)
        er = efficiency_ratio(df["close"], self.er_window)
        return sig.where(er >= self.er_min, 0.0).rename("target")
