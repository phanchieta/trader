import numpy as np
import pandas as pd
import pytest

from trader.backtest.cross_sectional import run_cross_sectional
from trader.costs.b3 import B3Costs
from trader.strategy import SuperTrend, TrendScan


def _frame(c, seed=0):
    n = len(c)
    idx = pd.date_range("2015-01-02", periods=n, freq="B", tz="UTC")
    idx.name = "time"
    c = np.asarray(c, float)
    return pd.DataFrame({"open": c, "high": c * 1.004, "low": c * 0.996,
                         "close": c, "volume": np.full(n, 1e6)}, index=idx)


def _rw(n=400, seed=1, drift=0.0003):
    rng = np.random.default_rng(seed)
    return _frame(40 * np.exp(np.cumsum(rng.normal(drift, 0.018, n))))


# ---------- SuperTrend ----------

def test_supertrend_fica_long_em_tendencia_de_alta():
    df = _frame(np.linspace(30, 70, 300))
    sig = SuperTrend(period=10, multiplier=3.0).generate_signals(df)
    assert set(sig.unique()) <= {0.0, 1.0}
    assert sig.iloc[-50:].mean() > 0.9


def test_supertrend_causal():
    df = _rw(500)
    st = SuperTrend(10, 3.0)
    full = st.generate_signals(df)
    cut = 300
    pd.testing.assert_series_equal(full.iloc[:cut],
                                   st.generate_signals(df.iloc[:cut]), check_names=False)


# ---------- TrendScan ----------

def test_trend_scan_reconhece_tendencia_significativa():
    df = _frame(np.linspace(30, 60, 250))          # reta -> t-value alto
    sig = TrendScan(window=40, t_threshold=2.0).generate_signals(df)
    assert sig.iloc[-1] == 1.0


def test_trend_scan_fica_fora_em_lateral():
    rng = np.random.default_rng(3)
    df = _frame(40 + rng.normal(0, 0.3, 300))       # sem tendência
    sig = TrendScan(window=40, t_threshold=2.5).generate_signals(df)
    assert sig.mean() < 0.3


def test_trend_scan_causal():
    df = _rw(500)
    ts = TrendScan(40, 2.0)
    full = ts.generate_signals(df)
    cut = 300
    pd.testing.assert_series_equal(full.iloc[:cut],
                                   ts.generate_signals(df.iloc[:cut]), check_names=False)


# ---------- cross-sectional ----------

def test_cross_sectional_escolhe_o_melhor_nome():
    # A sobe forte, B e C de lado -> a carteira concentra em A
    a = _frame(30 * np.exp(np.cumsum(np.full(400, 0.004))))
    b = _rw(400, seed=10, drift=0.0)
    c = _rw(400, seed=11, drift=0.0)
    frames = {"A": a, "B": b, "C": c}
    res = run_cross_sectional(frames, B3Costs(), lookback=100, skip=5, n_long=1,
                              rebalance=10)
    assert res.equity.iloc[-1] > res.equity.iloc[0]    # ganhou seguindo A
    assert res.stats["n_trades"] > 0


def test_cross_sectional_causal():
    frames = {"A": _rw(500, 1), "B": _rw(500, 2), "C": _rw(500, 3)}
    costs = B3Costs()
    full = run_cross_sectional(frames, costs, lookback=100, skip=5, n_long=2, rebalance=5)
    cut = 300
    sliced = run_cross_sectional({k: v.iloc[:cut] for k, v in frames.items()}, costs,
                                 lookback=100, skip=5, n_long=2, rebalance=5)
    np.testing.assert_allclose(full.equity.iloc[:cut].to_numpy(),
                               sliced.equity.to_numpy(), rtol=1e-9)


def test_cross_sectional_universo_com_historias_diferentes():
    a = _rw(500, 1)
    b = _rw(300, 2)                     # começa mais tarde (menos barras)
    frames = {"A": a, "B": b, "C": _rw(500, 3)}
    res = run_cross_sectional(frames, B3Costs(), lookback=100, skip=5, n_long=1,
                              rebalance=10)
    assert len(res.equity) == 500       # usa a UNIÃO das datas
    assert res.equity.notna().all()


def test_cross_sectional_long_short_abre_vendido():
    up = _frame(30 * np.exp(np.cumsum(np.full(400, 0.004))))
    down = _frame(50 * np.exp(np.cumsum(np.full(400, -0.004))))
    flat = _rw(400, seed=9, drift=0.0)
    frames = {"UP": up, "DOWN": down, "FLAT": flat}
    res = run_cross_sectional(frames, B3Costs(), lookback=100, skip=5, n_long=1,
                              n_short=1, rebalance=10, borrow_annual=0.0)
    assert res.equity.iloc[-1] > res.equity.iloc[0]   # long UP + short DOWN ganha
