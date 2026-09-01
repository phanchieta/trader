"""Divisões treino/teste e janelas walk-forward.

`embargo` remove barras do fim do treino para criar um intervalo entre treino e
teste — protege contra vazamento quando o alvo/feature olha para frente
(essencial quando o ML entrar; com indicadores só-para-trás é cinto e suspensório).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def train_test_split(df: pd.DataFrame, test_frac: float = 0.3, embargo: int = 0):
    """Primeiros (1-test_frac) para treino, resto para teste. `embargo` encurta
    o treino pelo fim."""
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac deve estar em (0, 1)")
    n = len(df)
    cut = int(n * (1.0 - test_frac))
    train_end = max(1, cut - embargo)
    return df.iloc[:train_end], df.iloc[cut:]


@dataclass
class WalkForward:
    n_folds: int = 4
    test_frac: float = 0.4          # fração final dos dados reservada para OOS
    embargo: int = 0
    anchored: bool = True           # True: treino sempre desde o começo

    def splits(self, n: int) -> list[tuple[slice, slice]]:
        if self.n_folds < 1:
            raise ValueError("n_folds >= 1")
        test_total = int(n * self.test_frac)
        fold = test_total // self.n_folds
        if fold < 1:
            raise ValueError("dados insuficientes ou test_frac pequeno demais")
        train_len = n - test_total
        if train_len < fold:
            raise ValueError("treino menor que um fold de teste")

        out: list[tuple[slice, slice]] = []
        for k in range(self.n_folds):
            ts = train_len + k * fold
            te = ts + fold if k < self.n_folds - 1 else n
            tr_start = 0 if self.anchored else max(0, ts - train_len)
            tr_end = max(tr_start + 1, ts - self.embargo)
            out.append((slice(tr_start, tr_end), slice(ts, te)))
        return out
