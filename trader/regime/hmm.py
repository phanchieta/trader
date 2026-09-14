"""HMM gaussiano (Hamilton 1989; Baum-Welch) — o regime de Markov da literatura:
o estado é LATENTE e sai da distribuição dos retornos (média e volatilidade),
não de um rótulo de retorno passado. Só numpy.

Estados ordenados por variância: 0 = calmo, K−1 = turbulento. O filtro de
Hamilton (forward normalizado) dá P(S_t | retornos até t) — causal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_LOG2PI = np.log(2.0 * np.pi)


@dataclass
class GaussianHMM:
    mu: np.ndarray
    var: np.ndarray
    A: np.ndarray
    pi: np.ndarray
    loglik: float


def _emissions(x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    B = np.exp(-0.5 * ((x[:, None] - mu) ** 2 / var + np.log(var) + _LOG2PI))
    return np.maximum(B, 1e-300)


def _forward(B: np.ndarray, A: np.ndarray, pi: np.ndarray):
    T, K = B.shape
    alpha, c = np.empty((T, K)), np.empty(T)
    a = pi * B[0]
    c[0] = a.sum()
    alpha[0] = a / c[0]
    for t in range(1, T):
        a = (alpha[t - 1] @ A) * B[t]
        c[t] = a.sum()
        alpha[t] = a / c[t]
    return alpha, c


def _backward(B: np.ndarray, A: np.ndarray, c: np.ndarray) -> np.ndarray:
    T, K = B.shape
    beta = np.empty((T, K))
    beta[-1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
    return beta


def fit_hmm(x, *, n_states: int = 2, n_iter: int = 150, tol: float = 1e-7,
            seeds=(0, 1, 2)) -> GaussianHMM:
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    v0, m0 = float(x.var()), float(x.mean())
    best = None
    for seed in seeds:
        rng = np.random.default_rng(seed)
        var = v0 * np.geomspace(0.4, 2.5, n_states) * rng.uniform(0.8, 1.25, n_states)
        mu = m0 + rng.normal(0.0, 0.05 * np.sqrt(v0), n_states)
        A = np.full((n_states, n_states), 0.05 / max(n_states - 1, 1))
        np.fill_diagonal(A, 0.95)
        pi = np.full(n_states, 1.0 / n_states)
        prev = -np.inf
        for _ in range(n_iter):
            B = _emissions(x, mu, var)
            alpha, c = _forward(B, A, pi)
            ll = float(np.log(c).sum())
            if ll - prev < tol * abs(ll):
                break
            prev = ll
            beta = _backward(B, A, c)
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)
            xi = A * (alpha[:-1].T @ (B[1:] * beta[1:] / c[1:, None]))
            A = xi / xi.sum(axis=1, keepdims=True)
            pi = gamma[0]
            w = gamma.sum(axis=0)
            mu = (gamma * x[:, None]).sum(axis=0) / w
            var = np.maximum((gamma * (x[:, None] - mu) ** 2).sum(axis=0) / w, 1e-4 * v0)
        else:                                   # sem convergir: ll dos parâmetros finais
            ll = float(np.log(_forward(_emissions(x, mu, var), A, pi)[1]).sum())
        if best is None or ll > best.loglik:
            best = GaussianHMM(mu, var, A, pi, ll)
    o = np.argsort(best.var)
    return GaussianHMM(best.mu[o], best.var[o], best.A[np.ix_(o, o)], best.pi[o], best.loglik)


def filtered_probs(x, model: GaussianHMM) -> np.ndarray:
    """P(S_t | x_1..x_t). NaN = dia sem informação (emissão neutra)."""
    x = np.asarray(x, dtype="float64")
    bad = ~np.isfinite(x)
    B = _emissions(np.where(bad, 0.0, x), model.mu, model.var)
    B[bad] = 1.0
    return _forward(B, model.A, model.pi)[0]


def calm_probability_next(close: pd.Series, *, train_bars: int = 2520,
                          min_train_bars: int = 500, start_year: int | None = None,
                          n_states: int = 2, seeds=(0, 1, 2)) -> pd.Series:
    """Walk-forward anual: no 1º pregão do ano Y ajusta o HMM só com retornos
    ANTERIORES a Y (até `train_bars`), filtra até o fim de Y com os parâmetros
    congelados e devolve P(calmo em t+1 | retornos até t) para os dias de Y."""
    x = (100.0 * np.log(close).diff()).to_numpy()
    years = close.index.year
    out = np.full(len(close), np.nan)
    for y in sorted(set(years)):
        if start_year is not None and y < start_year:
            continue
        pos = np.nonzero(years == y)[0]
        first, last = int(pos[0]), int(pos[-1])
        if first < min_train_bars:
            continue
        lo = max(1, first - train_bars)
        model = fit_hmm(x[lo:first], n_states=n_states, seeds=seeds)
        pred = filtered_probs(x[lo:last + 1], model) @ model.A
        out[first:last + 1] = pred[first - lo:, 0]
    return pd.Series(out, index=close.index, name="p_calmo")
