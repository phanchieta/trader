import numpy as np
import pandas as pd

from trader.backtest.portfolio import run_portfolio
from trader.costs.b3 import B3Costs
from trader.risk.overlay import inv_vol_weights
from trader.strategy import SmaCrossover


def _frames(n=400):
    out = {}
    for k, sig in enumerate((0.010, 0.025, 0.05)):        # vols crescentes
        rng = np.random.default_rng(k + 1)
        px = 40 * np.exp(np.cumsum(rng.normal(0.0003, sig, n)))
        idx = pd.date_range("2016-01-04", periods=n, freq="B", tz="UTC")
        idx.name = "time"
        out[f"S{k}"] = pd.DataFrame({"open": px, "high": px * 1.01, "low": px * 0.99,
                                     "close": px, "volume": 1e6}, index=idx)
    return out


def test_inv_vol_weights_favorecem_o_ativo_calmo():
    frames = _frames()
    w = inv_vol_weights(frames, lookback=20)
    assert np.allclose(w.sum(axis=1).iloc[30:], 1.0)
    # S0 (vol 0,010) deve pesar mais que S2 (vol 0,05)
    assert w["S0"].iloc[50:].mean() > w["S2"].iloc[50:].mean()


def test_name_weights_none_igual_a_equal_weight():
    frames = _frames()
    idx = next(iter(frames.values())).index
    eq_w = pd.DataFrame({s: 1 / 3 for s in frames}, index=idx)
    a = run_portfolio(frames, SmaCrossover(10, 30), B3Costs())
    b = run_portfolio(frames, SmaCrossover(10, 30), B3Costs(), name_weights=eq_w)
    np.testing.assert_allclose(a.equity.to_numpy(), b.equity.to_numpy(), rtol=1e-9)


def test_name_weights_causal():
    frames = _frames(500)
    w = inv_vol_weights(frames, lookback=20)
    full = run_portfolio(frames, SmaCrossover(10, 30), B3Costs(), name_weights=w)
    cut = 300
    sliced = run_portfolio({k: v.iloc[:cut] for k, v in frames.items()},
                           SmaCrossover(10, 30), B3Costs(),
                           name_weights=w.iloc[:cut])
    np.testing.assert_allclose(full.equity.iloc[:cut].to_numpy(),
                               sliced.equity.to_numpy(), rtol=1e-9)
