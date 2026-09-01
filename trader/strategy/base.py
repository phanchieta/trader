"""Interface de estratégia. A MESMA função roda no backtest e no live."""
from __future__ import annotations

from typing import Protocol

import pandas as pd


class Strategy(Protocol):
    name: str

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Recebe OHLCV até o presente, devolve posição-alvo por barra em
        {-1, 0, +1} (fração do capital: -1 vendido, 0 zerado, +1 comprado),
        indexada igual a `df`.

        REGRA DE OURO: o valor em t só pode usar informação até t (inclusive).
        O engine aplica shift(1) antes de executar — então nada de .shift(-x),
        janelas centradas, ou `df.iloc[i+1]` aqui dentro.
        """
        ...
