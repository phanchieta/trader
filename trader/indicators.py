"""Indicadores compartilhados (estratégias e overlays de risco).

ATR + estimadores de volatilidade por range — usam o OHLC inteiro, não só o
fechamento, então captam melhor a vol em dias de range largo / gap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_DIAS_ANO = 252.0


def true_range(df: pd.DataFrame) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rename("tr")


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    return true_range(df).rolling(window).mean().rename("atr")


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index (Wilder). Mede a FORÇA da tendência (não a
    direção): ~<15 sem tendência, >25 tendência clara."""
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = up.where((up > dn) & (up > 0.0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0.0), 0.0)
    a = 1.0 / period
    tr_s = true_range(df).ewm(alpha=a, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=a, adjust=False).mean() / tr_s
    minus_di = 100.0 * minus_dm.ewm(alpha=a, adjust=False).mean() / tr_s
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).where(
        lambda x: x > 0)
    return dx.ewm(alpha=a, adjust=False).mean().rename("adx")


def efficiency_ratio(close: pd.Series, window: int = 20) -> pd.Series:
    """Kaufman Efficiency Ratio: |Δpreço na janela| / soma dos |Δ diários|.
    ~1 = movimento direcional (tendência); ~0 = vai-e-volta (lateral)."""
    direction = (close - close.shift(window)).abs()
    path = close.diff().abs().rolling(window).sum()
    return (direction / path.where(path > 0)).rename("er")


def parkinson(df: pd.DataFrame, window: int = 20,
              periods_per_year: float = _DIAS_ANO) -> pd.Series:
    hl2 = np.log(df["high"] / df["low"]) ** 2
    var = hl2.rolling(window).mean() / (4.0 * np.log(2.0))
    return np.sqrt(var * periods_per_year).rename("parkinson_vol")


def garman_klass(df: pd.DataFrame, window: int = 20,
                 periods_per_year: float = _DIAS_ANO) -> pd.Series:
    log_hl2 = np.log(df["high"] / df["low"]) ** 2
    log_co2 = np.log(df["close"] / df["open"]) ** 2
    daily = 0.5 * log_hl2 - (2.0 * np.log(2.0) - 1.0) * log_co2
    var = daily.rolling(window).mean()
    return np.sqrt(var.clip(lower=0) * periods_per_year).rename("gk_vol")


def yang_zhang(df: pd.DataFrame, window: int = 20,
               periods_per_year: float = _DIAS_ANO) -> pd.Series:
    """Yang-Zhang: combina vol overnight, open->close e Rogers-Satchell.
    Robusto a drift e a gaps de abertura."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    pc = c.shift(1)
    log_oc = np.log(o / pc)          # overnight
    log_co = np.log(c / o)           # intradiário
    log_ho, log_lo = np.log(h / o), np.log(l / o)

    n = window
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    sig_o2 = log_oc.rolling(n).var()
    sig_c2 = log_co.rolling(n).var()
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    sig_rs2 = rs.rolling(n).mean()

    var = sig_o2 + k * sig_c2 + (1.0 - k) * sig_rs2
    return np.sqrt(var.clip(lower=0) * periods_per_year).rename("yz_vol")
