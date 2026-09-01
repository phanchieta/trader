"""Cruzamento de médias móveis — primeira estratégia de regra.

Referência simples para o engine e as métricas. Não é para operar: serve de
baseline a ser batido pelas estratégias reais (reversão à média, breakout de
abertura, momentum) e depois pelo ML.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class SmaCrossover:
    fast: int = 20
    slow: int = 60
    allow_short: bool = False
    name: str = "sma_crossover"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        fast = df["close"].rolling(self.fast).mean()
        slow = df["close"].rolling(self.slow).mean()
        sig = (fast > slow).astype("float64")                 # +1 comprado / 0 zerado
        if self.allow_short:
            sig = sig - (fast < slow).astype("float64")       # ... / -1 vendido
        sig[slow.isna()] = 0.0                                 # aquecimento: fora
        return sig.rename("target")
