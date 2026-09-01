"""Wrapper: só confirma ENTRADAS (ou viradas) quando o volume relativo está
acima do limiar. Saídas (ir para zero) passam sempre.

volume relativo = volume / média móvel do volume.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class VolumeConfirmed:
    base: object
    lookback: int = 20
    min_rel_volume: float = 1.0
    name: str = "vol_confirmed"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        raw = self.base.generate_signals(df).reindex(df.index).fillna(0.0).to_numpy()
        rel = (df["volume"] / df["volume"].rolling(self.lookback).mean()).to_numpy()
        out = np.zeros(len(raw))
        cur = 0.0
        for i in range(len(raw)):
            want = raw[i]
            if want == 0.0:
                cur = 0.0
            elif want != cur:                       # quer entrar / virar de lado
                if np.isfinite(rel[i]) and rel[i] >= self.min_rel_volume:
                    cur = want
                # sem volume: mantém a posição atual (ignora o sinal)
            out[i] = cur
        return pd.Series(out, index=df.index, name="target")
