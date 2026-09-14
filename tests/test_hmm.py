import numpy as np
import pandas as pd

from trader.regime import calm_probability_next, filtered_probs, fit_hmm


def _two_regimes(n=3000, seed=3):
    rng = np.random.default_rng(seed)
    A = np.array([[0.98, 0.02], [0.05, 0.95]])
    s = np.empty(n, dtype=int)
    s[0] = 0
    for t in range(1, n):
        s[t] = rng.choice(2, p=A[s[t - 1]])
    x = np.where(s == 0, rng.normal(0.08, 0.8, n), rng.normal(-0.2, 3.0, n))
    return x, s


def test_recupera_regime_calmo_e_turbulento():
    x, s = _two_regimes()
    m = fit_hmm(x)
    assert m.var[0] < m.var[1]                             # 0 = calmo
    assert np.sqrt(m.var[1] / m.var[0]) > 2.5
    assert np.allclose(m.A.sum(axis=1), 1.0)
    assert m.A[0, 0] > 0.9 and m.A[1, 1] > 0.85
    acc = ((filtered_probs(x, m)[:, 1] > 0.5).astype(int) == s).mean()
    assert acc > 0.85


def test_filtro_e_causal():
    x, _ = _two_regimes(n=800)
    m = fit_hmm(x[:600])
    f1 = filtered_probs(x, m)
    x2 = x.copy()
    x2[500:] = 10.0                                        # muda só o futuro
    f2 = filtered_probs(x2, m)
    assert np.allclose(f1[:500], f2[:500])


def test_probabilidade_calma_walk_forward():
    x, _ = _two_regimes(n=1500, seed=5)
    idx = pd.date_range("2015-01-01", periods=len(x), freq="B")
    close = pd.Series(100 * np.exp(np.cumsum(x / 100)), index=idx)
    p = calm_probability_next(close, min_train_bars=400, seeds=(0,))
    assert p.iloc[:400].isna().all()
    ok = p.dropna()
    assert len(ok) > 900 and ok.between(0, 1).all()
