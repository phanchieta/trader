"""Rótulo triple-barrier (López de Prado): a partir da barra t, o preço sobe
`pt_atr`·ATR (alvo) antes de cair `sl_atr`·ATR (stop) dentro de `horizon` dias?
1 = alvo primeiro; 0 = stop primeiro ou timeout. Mais informativo que
'retorno futuro > custo'.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators import atr


def triple_barrier(df: pd.DataFrame, *, horizon: int = 10, pt_atr: float = 2.0,
                   sl_atr: float = 1.5, atr_window: int = 14) -> pd.Series:
    c = df["close"].to_numpy(dtype="float64")
    hi = df["high"].to_numpy(dtype="float64")
    lo = df["low"].to_numpy(dtype="float64")
    a = atr(df, atr_window).to_numpy(dtype="float64")
    n = len(df)
    y = np.full(n, np.nan)
    for i in range(n - 1):
        if not np.isfinite(a[i]) or a[i] <= 0.0:
            continue
        up = c[i] + pt_atr * a[i]
        dn = c[i] - sl_atr * a[i]
        lab = 0.0
        for j in range(i + 1, min(i + 1 + horizon, n)):
            if lo[j] <= dn:
                break
            if hi[j] >= up:
                lab = 1.0
                break
        y[i] = lab
    return pd.Series(y, index=df.index, name="_y")
