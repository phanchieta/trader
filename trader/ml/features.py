"""Features por ação — todas causais (só usam dados até a barra t). Uma linha
por barra. Codificam tudo que a pesquisa de regra construiu.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators import adx, atr, efficiency_ratio, parkinson, yang_zhang
from ..strategy import Donchian, SuperTrend, Vortex


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    f = pd.DataFrame(index=df.index)

    for n in (1, 5, 10, 20, 60, 120):
        f[f"ret_{n}"] = c.pct_change(n)
    f["mom_12_1"] = c.shift(21) / c.shift(252) - 1.0
    f["mom_6_1"] = c.shift(21) / c.shift(126) - 1.0
    f["mom_3_1"] = c.shift(21) / c.shift(63) - 1.0

    f["vol_yz_20"] = yang_zhang(df, 20)
    f["vol_yz_60"] = yang_zhang(df, 60)
    f["vol_pk_20"] = parkinson(df, 20)
    f["vol_cc_20"] = c.pct_change().rolling(20).std() * np.sqrt(252)

    a = atr(df, 14)
    sma50 = c.rolling(50).mean()
    f["sma10_50"] = c.rolling(10).mean() / sma50 - 1.0
    f["dist_sma50_atr"] = (c - sma50) / a.where(a > 0)
    f["adx_14"] = adx(df, 14)
    f["er_20"] = efficiency_ratio(c, 20)

    lo60, hi60 = l.rolling(60).min(), h.rolling(60).max()
    f["chan_pos_60"] = (c - lo60) / (hi60 - lo60).where(lambda x: x > 0)

    f["supertrend_dir"] = SuperTrend(10, 3.0, allow_short=True).generate_signals(df)
    f["donchian_state"] = Donchian(20, 10, allow_short=True).generate_signals(df)
    f["vortex_sig"] = Vortex(14, allow_short=True).generate_signals(df)

    f["vol_rel_20"] = v / v.rolling(20).mean()
    f["gap"] = df["open"] / c.shift(1) - 1.0
    f["dow"] = df.index.dayofweek.astype("float64")
    f["month"] = df.index.month.astype("float64")
    return f


def feature_names(df: pd.DataFrame) -> list[str]:
    return list(build_features(df.iloc[:5]).columns)
