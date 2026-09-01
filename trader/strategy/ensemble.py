"""Combinação de estratégias.

  Ensemble          — voto: ±1 quando >= `min_agree` membros concordam.
  WeightedEnsemble  — voto ponderado pela qualidade recente (Sharpe rolante do
                      P&L de cada membro, causal). Alvo contínuo em [-1, 1].
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Ensemble:
    members: list = field(default_factory=list)
    min_agree: int = 2
    name: str = "ensemble"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if not self.members:
            raise ValueError("ensemble sem membros")
        votes = pd.concat(
            [m.generate_signals(df).reindex(df.index).fillna(0.0) for m in self.members],
            axis=1,
        )
        n_long = (votes > 0).sum(axis=1)
        n_short = (votes < 0).sum(axis=1)
        out = pd.Series(0.0, index=df.index, name="target")
        out[n_long >= self.min_agree] = 1.0
        out[n_short >= self.min_agree] = -1.0
        tie = (n_long >= self.min_agree) & (n_short >= self.min_agree)
        out[tie] = np.sign((n_long - n_short)[tie])
        return out


@dataclass
class WeightedEnsemble:
    members: list = field(default_factory=list)
    weight_lookback: int = 252
    name: str = "weighted_ensemble"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if not self.members:
            raise ValueError("ensemble sem membros")
        ret = df["close"].pct_change()
        sigs, weights = [], []
        for m in self.members:
            s = m.generate_signals(df).reindex(df.index).fillna(0.0)
            pnl = s.shift(1) * ret                        # posição de ontem × retorno de hoje
            roll = pnl.rolling(self.weight_lookback)
            sharpe = (roll.mean() / roll.std()).replace([np.inf, -np.inf], np.nan)
            sigs.append(s)
            weights.append(sharpe.shift(1).clip(lower=0.0).fillna(0.0))  # peso causal, só +

        S = pd.concat(sigs, axis=1).to_numpy()
        W = pd.concat(weights, axis=1).to_numpy()
        wsum = W.sum(axis=1)
        combined = (S * W).sum(axis=1)
        target = np.where(wsum > 0.0, combined / np.where(wsum > 0.0, wsum, 1.0), 0.0)
        return pd.Series(np.clip(target, -1.0, 1.0), index=df.index, name="target")
