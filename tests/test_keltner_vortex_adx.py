import numpy as np
import pandas as pd

from trader.indicators import adx
from trader.risk.overlay import adx_scale_multiplier
from trader.strategy import Keltner, Vortex


def _frame(c):
    n = len(c)
    idx = pd.date_range("2015-01-02", periods=n, freq="B", tz="UTC")
    idx.name = "time"
    c = np.asarray(c, float)
    return pd.DataFrame({"open": c, "high": c * 1.006, "low": c * 0.994,
                         "close": c, "volume": np.full(n, 1e6)}, index=idx)


def _rw(n=500, seed=1):
    rng = np.random.default_rng(seed)
    return _frame(40 * np.exp(np.cumsum(rng.normal(0.0003, 0.018, n))))


# ---------- ADX ----------

def test_adx_alto_em_tendencia_baixo_em_lateral():
    trend = adx(_frame(np.linspace(30, 80, 300)), 14)
    chop = adx(_frame(40 + np.tile([0.0, 0.8, 0.0, -0.8], 80)), 14)
    assert trend.iloc[40:].mean() > 30
    assert chop.iloc[40:].mean() < 25


def test_adx_causal():
    df = _rw(400)
    full = adx(df, 14)
    cut = 250
    pd.testing.assert_series_equal(full.iloc[:cut], adx(df.iloc[:cut], 14),
                                   check_names=False)


def test_adx_scale_multiplier_limitado():
    frames = {"A": _rw(400, 1), "B": _rw(400, 2), "C": _rw(400, 3)}
    m = adx_scale_multiplier(frames, period=14, adx_floor=15.0, adx_cap=40.0)
    assert (m >= 0.0).all() and (m <= 1.0 + 1e-9).all()


# ---------- Keltner ----------

def test_keltner_entra_no_rompimento_da_banda_superior():
    df = _frame(np.concatenate([np.full(40, 40.0), np.linspace(41, 70, 120)]))
    sig = Keltner(ema_period=20, atr_period=20, mult=2.0).generate_signals(df)
    assert set(sig.unique()) <= {0.0, 1.0}
    assert sig.iloc[-1] == 1.0


def test_keltner_causal():
    df = _rw(500)
    k = Keltner(20, 20, 2.0)
    full = k.generate_signals(df)
    cut = 300
    pd.testing.assert_series_equal(full.iloc[:cut],
                                   k.generate_signals(df.iloc[:cut]), check_names=False)


# ---------- Vortex ----------

def test_vortex_long_em_alta():
    df = _frame(np.linspace(30, 70, 300))
    sig = Vortex(period=14).generate_signals(df)
    assert set(sig.unique()) <= {0.0, 1.0}
    assert sig.iloc[-30:].mean() > 0.9


def test_vortex_causal():
    df = _rw(500)
    v = Vortex(21)
    full = v.generate_signals(df)
    cut = 300
    pd.testing.assert_series_equal(full.iloc[:cut],
                                   v.generate_signals(df.iloc[:cut]), check_names=False)
