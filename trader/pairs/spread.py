"""Spread entre duas classes: log(A) − log(B) com preços ajustados por proventos
(o lado comprado recebe e o vendido paga os dividendos, então o retorno total é o
que importa). Hedge 1:1 em valor — mesma empresa, mesmo fluxo de caixa — sem
estimar beta (nada para sobreajustar).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def log_spread(a: pd.Series, b: pd.Series) -> pd.Series:
    a, b = a.where(a > 0), b.where(b > 0)
    return (np.log(a) - np.log(b)).rename("spread")


def rolling_z(s: pd.Series, lookback: int) -> pd.Series:
    m = s.rolling(lookback, min_periods=lookback).mean()
    sd = s.rolling(lookback, min_periods=lookback).std()
    return ((s - m) / sd.where(sd > 0)).rename("z")


def eligibility(a: pd.DataFrame, b: pd.DataFrame, *, min_traded_value: float,
                window: int = 60, max_jump: float = 0.15, jump_window: int = 5) -> pd.Series:
    """Par operável em t se as DUAS pontas negociaram ≥ `min_traded_value` (mediana de
    `window` pregões) e o spread não saltou mais que `max_jump` nos últimos
    `jump_window` dias (proteção contra erro de ajuste e evento societário). Causal."""
    idx = a.index.intersection(b.index)
    a, b = a.reindex(idx), b.reindex(idx)
    liq = lambda f: (f["close"] * f["volume"]).rolling(window, min_periods=window).median()
    jump = log_spread(a["close"], b["close"]).diff().abs().rolling(jump_window).max()
    ok = (liq(a) >= min_traded_value) & (liq(b) >= min_traded_value) & (jump <= max_jump)
    return ok.fillna(False).rename("elegivel")


def half_life(s: pd.Series) -> float:
    """Meia-vida (pregões) da reversão: Δs_t = a + b·s_{t−1}; hl = −ln2 / ln(1+b)."""
    x = s.dropna().to_numpy()
    if len(x) < 30:
        return float("nan")
    y, lag = np.diff(x), x[:-1]
    b = np.polyfit(lag, y, 1)[0]
    return float(-np.log(2) / np.log1p(b)) if -1 < b < 0 else float("inf")


def adf_tstat(s: pd.Series, lags: int = 1) -> float:
    """t da raiz unitária (Dickey-Fuller aumentado, com constante). Abaixo de ≈ −2,86
    rejeita raiz unitária a 5%."""
    x = s.dropna().to_numpy()
    dx = np.diff(x)
    n = len(dx) - lags
    if n < 30:
        return float("nan")
    cols = [np.ones(n), x[lags:-1]] + [dx[lags - i - 1:len(dx) - i - 1] for i in range(lags)]
    X, y = np.column_stack(cols), dx[lags:]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma2 = resid @ resid / (n - X.shape[1])
    se = np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[1, 1])
    return float(beta[1] / se)
