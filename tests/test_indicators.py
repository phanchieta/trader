import numpy as np
import pandas as pd
import pytest

from trader.indicators import atr, garman_klass, parkinson, true_range, yang_zhang


def _df(n=300, seed=1, flat=False):
    rng = np.random.default_rng(seed)
    if flat:
        c = np.full(n, 40.0)
        o = h = l = c
    else:
        c = 40.0 * np.exp(np.cumsum(rng.normal(0.0, 0.015, n)))
        o = c * (1 + rng.normal(0, 0.003, n))
        h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.004, n)))
        l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.004, n)))
    idx = pd.date_range("2015-01-02", periods=n, freq="B", tz="UTC")
    idx.name = "time"
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                         "volume": 1e6}, index=idx)


@pytest.mark.parametrize("fn", [parkinson, garman_klass, yang_zhang])
def test_vol_positiva_e_finita_apos_warmup(fn):
    v = fn(_df(), window=20)
    assert v.iloc[30:].notna().all()
    assert (v.iloc[30:] > 0).all()


@pytest.mark.parametrize("fn", [parkinson, garman_klass, yang_zhang])
def test_vol_quase_zero_com_preco_constante(fn):
    v = fn(_df(flat=True), window=20)
    assert v.iloc[30:].abs().max() < 1e-6


def test_atr_positivo_e_tamanho():
    a = atr(_df(), window=14)
    assert len(a) == 300
    assert (a.iloc[20:] > 0).all()


def test_true_range_maior_ou_igual_high_menos_low():
    df = _df()
    tr = true_range(df)
    assert (tr.iloc[1:] >= (df["high"] - df["low"]).iloc[1:] - 1e-9).all()


@pytest.mark.parametrize("fn", [parkinson, garman_klass, yang_zhang, atr])
def test_causal(fn):
    df = _df(400)
    full = fn(df, window=20) if fn is not atr else fn(df, window=20)
    cut = 250
    trunc = fn(df.iloc[:cut], window=20)
    pd.testing.assert_series_equal(full.iloc[:cut], trunc, check_names=False)
