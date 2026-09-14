"""Regime de Markov visível — o "hedge fund method" (framework de Roan/@RohOnChain,
github.com/jackson-video-resources/markov-hedge-fund-method).

Estado do dia pelo retorno composto dos últimos `window` pregões: BULL se
> +threshold, BEAR se < −threshold, LATERAL no meio (limites exclusivos, como no
código do autor). Matriz 3×3 por contagem (máxima verossimilhança, sem
suavização), re-estimada todo dia só com transições já observadas.
Sinal em t: P[s_t, BULL] − P[s_t, BEAR] (ou da matriz de h passos, P^h).

Com janela sobreposta, o estado de hoje e o de amanhã dividem window−1 retornos:
a persistência da diagonal é majoritariamente mecânica.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BEAR, SIDE, BULL = 0, 1, 2


def label_states(close: pd.Series, *, window: int = 20, threshold: float = 0.05) -> pd.Series:
    r = close.pct_change(window)
    lab = np.where(r > threshold, BULL, np.where(r < -threshold, BEAR, SIDE)).astype("float64")
    lab[r.isna().to_numpy()] = np.nan
    return pd.Series(lab, index=close.index, name="state")


def _normalize_rows(counts: np.ndarray) -> np.ndarray:
    tot = counts.sum(axis=-1, keepdims=True)
    return np.divide(counts, tot, out=np.zeros(counts.shape), where=tot > 0)


def transition_matrix(states) -> np.ndarray:
    s = np.asarray(states, dtype="float64")
    a, b = s[:-1], s[1:]
    ok = ~(np.isnan(a) | np.isnan(b))
    c = np.zeros(9)
    np.add.at(c, (3 * a[ok] + b[ok]).astype(int), 1.0)
    return _normalize_rows(c.reshape(3, 3))


def expanding_matrices(states: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """(P_t, n_t): matriz T×3×3 com as transições s_{i−1}→s_i de i ≤ t (conhecidas
    no fechamento de t) e quantas transições entraram em cada P_t."""
    s = states.to_numpy(dtype="float64")
    onehot = np.zeros((len(s), 9))
    if len(s) > 1:
        a, b = s[:-1], s[1:]
        ok = ~(np.isnan(a) | np.isnan(b))
        onehot[np.nonzero(ok)[0] + 1, (3 * a[ok] + b[ok]).astype(int)] = 1.0
    cum = np.cumsum(onehot, axis=0).reshape(len(s), 3, 3)
    return _normalize_rows(cum), cum.sum(axis=(1, 2))


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eig(P.T)
    pi = np.real(v[:, int(np.argmin(np.abs(w - 1.0)))])
    return pi / pi.sum()


def markov_signal(close: pd.Series, *, window: int = 20, threshold: float = 0.05,
                  horizon: int = 1, min_train: int = 252, sizing: str = "sign",
                  long_only: bool = False) -> pd.Series:
    """Alvo por dia. `sizing`: 'sign' = sign(P_bull − P_bear), o backtest do autor;
    'prob' = a diferença contínua, como no vídeo (magnitude = tamanho)."""
    states = label_states(close, window=window, threshold=threshold)
    P, n = expanding_matrices(states)
    if horizon > 1:
        P = np.linalg.matrix_power(P, horizon)
    s = states.to_numpy()
    rows = np.nonzero(~np.isnan(s) & (n >= min_train))[0]
    st = s[rows].astype(int)
    sig = np.zeros(len(s))
    sig[rows] = P[rows, st, BULL] - P[rows, st, BEAR]
    if sizing == "sign":
        sig = np.sign(sig)
    if long_only:
        sig = np.clip(sig, 0.0, None)
    return pd.Series(sig, index=close.index, name="markov")
