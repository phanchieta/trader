import numpy as np
import pandas as pd

from trader.indicators import efficiency_ratio
from trader.risk.overlay import vol_target_multiplier
from trader.strategy import Donchian, DonchianFibo, RegimeGated


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


# ---------- efficiency_ratio ----------

def test_efficiency_ratio_alta_em_tendencia_baixa_em_lateral():
    trend = efficiency_ratio(_frame(np.linspace(30, 60, 200))["close"], 20)
    chop = efficiency_ratio(_frame(40 + np.tile([0, 1, 0, -1], 50))["close"], 20)
    assert trend.iloc[30:].mean() > 0.9
    assert chop.iloc[30:].mean() < 0.3


# ---------- DonchianFibo ----------

def test_donchian_fibo_compra_no_pullback_e_para_no_atr():
    # sobe forte (canal sobe), recua até ~meio do canal, depois cai forte -> ATR stop
    up = np.linspace(30, 60, 120)
    pull = np.linspace(60, 52, 10)
    crash = np.linspace(52, 40, 15)
    df = _frame(np.concatenate([up, pull, crash, np.full(40, 40.0)]))
    sig = DonchianFibo(period=72, entry_fib=0.5, atr_mult=3.0).generate_signals(df)
    assert set(sig.unique()) <= {0.0, 1.0}
    assert (sig.iloc[120:135] == 1.0).any()         # entrou no pullback
    assert sig.iloc[-1] == 0.0                       # stop tirou depois da queda


def test_donchian_fibo_causal():
    df = _rw(600)
    s = DonchianFibo(72, 0.5, atr_mult=3.0)
    full = s.generate_signals(df)
    cut = 400
    pd.testing.assert_series_equal(full.iloc[:cut],
                                   s.generate_signals(df.iloc[:cut]), check_names=False)


# ---------- RegimeGated ----------

def test_regime_gate_zera_em_lateral():
    chop = _frame(40 + np.tile([0.0, 0.6, 0.0, -0.6], 120))
    gated = RegimeGated(Donchian(20, 10), er_window=20, er_min=0.4).generate_signals(chop)
    assert (gated == 0.0).mean() > 0.95


def test_regime_gate_deixa_passar_em_tendencia():
    df = _frame(np.linspace(30, 90, 300))
    base = Donchian(20, 10).generate_signals(df)
    gated = RegimeGated(Donchian(20, 10), er_window=20, er_min=0.3).generate_signals(df)
    assert gated.iloc[-30:].equals(base.iloc[-30:])


def test_regime_gate_causal():
    df = _rw(500)
    g = RegimeGated(Donchian(20, 10), er_window=20, er_min=0.3)
    full = g.generate_signals(df)
    cut = 300
    pd.testing.assert_series_equal(full.iloc[:cut],
                                   g.generate_signals(df.iloc[:cut]), check_names=False)


# ---------- overlay estimator ----------

def test_vol_target_yang_zhang_roda_e_limita():
    frames = {"A": _rw(400, 1), "B": _rw(400, 2), "C": _rw(400, 3)}
    m = vol_target_multiplier(frames, target_ann_vol=0.15, lookback=20,
                              max_leverage=1.0, estimator="yang_zhang")
    assert (m <= 1.0 + 1e-9).all() and (m >= 0.0).all()
    assert (m.iloc[30:] > 0).any()
