import numpy as np
import pandas as pd

from trader.backtest.trend_portfolio import run_trend_portfolio


def _frame(px):
    n = len(px)
    idx = pd.date_range("2016-01-04", periods=n, freq="B", tz="UTC")
    idx.name = "time"
    px = np.asarray(px, float)
    return pd.DataFrame({"open": px, "high": px * 1.01, "low": px * 0.99,
                         "close": px, "volume": 1.0}, index=idx)


def _rw(n=600, seed=1, drift=0.0002):
    rng = np.random.default_rng(seed)
    return _frame(100 * np.exp(np.cumsum(rng.normal(drift, 0.012, n))))


def test_sinal_zero_fica_no_caixa_rendendo_cdi():
    frames = {"A": _rw(400), "B": _rw(400, 2)}
    sig = pd.DataFrame({"A": 0.0, "B": 0.0}, index=frames["A"].index)
    cdi = pd.Series(0.0004, index=frames["A"].index)
    r = run_trend_portfolio(frames, sig, cash_rate=cdi, initial_equity=100_000.0)
    esperado = 100_000.0 * (1 + 0.0004) ** len(frames["A"])
    assert abs(r.equity.iloc[-1] - esperado) / esperado < 1e-6
    assert r.exposure_media == 0.0


def test_long_em_tendencia_de_alta_ganha():
    up = _frame(np.linspace(100, 180, 500))
    frames = {"A": up}
    sig = pd.DataFrame({"A": 1.0}, index=up.index)
    r = run_trend_portfolio(frames, sig, target_vol=0.20, vol_lookback=40,
                            initial_equity=100_000.0)
    assert r.equity.iloc[-1] > 100_000.0


def test_short_ganha_em_tendencia_de_baixa():
    down = _frame(np.linspace(180, 100, 500))
    frames = {"A": down}
    sig = pd.DataFrame({"A": -1.0}, index=down.index)
    r = run_trend_portfolio(frames, sig, target_vol=0.20, initial_equity=100_000.0)
    assert r.equity.iloc[-1] > 100_000.0


def test_causal():
    frames = {"A": _rw(500, 1), "B": _rw(500, 2)}
    from trader.strategy import SuperTrend
    sig = pd.DataFrame({k: SuperTrend(10, 3.0).generate_signals(v)
                        for k, v in frames.items()})
    full = run_trend_portfolio(frames, sig, target_vol=0.15)
    cut = 300
    sl = run_trend_portfolio({k: v.iloc[:cut] for k, v in frames.items()},
                             sig.iloc[:cut], target_vol=0.15)
    np.testing.assert_allclose(full.equity.iloc[:cut].to_numpy(),
                               sl.equity.to_numpy(), rtol=1e-9)
