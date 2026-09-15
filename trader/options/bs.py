"""Black-Scholes europeu (sem dividendos) — só para medir volatilidade implícita."""
from __future__ import annotations

import math

from scipy.optimize import brentq
from scipy.stats import norm


def bs_price(S: float, K: float, T: float, r: float, sigma: float, kind: str = "call") -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol(price: float, S: float, K: float, T: float, r: float, kind: str = "call") -> float:
    lo, hi = 1e-3, 5.0
    intrinsic = max(S - K * math.exp(-r * T), 0.0) if kind == "call" else max(K * math.exp(-r * T) - S, 0.0)
    if T <= 0 or price <= intrinsic + 1e-9:
        return float("nan")
    f = lambda s: bs_price(S, K, T, r, s, kind) - price
    if f(lo) > 0 or f(hi) < 0:
        return float("nan")
    return float(brentq(f, lo, hi, xtol=1e-6))
