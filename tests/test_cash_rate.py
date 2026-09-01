import numpy as np
import pandas as pd

from trader.backtest.metrics import vs_benchmark
from trader.backtest.portfolio import run_portfolio
from trader.costs.b3 import B3Costs


class _Flat:
    name = "flat"

    def generate_signals(self, df):
        return pd.Series(0.0, index=df.index)


def _frames(n=300, seed=1):
    out = {}
    for k in range(3):
        rng = np.random.default_rng(seed + k)
        px = 40 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
        idx = pd.date_range("2016-01-04", periods=n, freq="B", tz="UTC")
        idx.name = "time"
        out[f"S{k}"] = pd.DataFrame({"open": px, "high": px * 1.01, "low": px * 0.99,
                                     "close": px, "volume": 1e6}, index=idx)
    return out


def test_caixa_rende_a_taxa_quando_sem_posicao():
    frames = _frames()
    idx = next(iter(frames.values())).index
    rate = pd.Series(0.0004, index=idx)          # 0,04%/dia
    res = run_portfolio(frames, _Flat(), B3Costs(), cash_rate=rate)
    esperado = 100_000.0 * (1.0 + 0.0004) ** len(idx)
    assert abs(res.equity.iloc[-1] - esperado) / esperado < 1e-9
    assert res.stats["exposure_media"] == 0.0


def test_sem_cash_rate_nao_muda_nada():
    frames = _frames()
    a = run_portfolio(frames, _Flat(), B3Costs())
    b = run_portfolio(frames, _Flat(), B3Costs(), cash_rate=None)
    np.testing.assert_allclose(a.equity.to_numpy(), b.equity.to_numpy())


def test_vs_benchmark_alpha():
    idx = pd.date_range("2016-01-04", periods=520, freq="B", tz="UTC")
    idx.name = "time"
    rate = pd.Series(0.0003, index=idx)
    cdi_curve = (1 + rate).cumprod() * 100_000.0
    strat = cdi_curve * 1.5 ** (np.arange(len(idx)) / len(idx))   # cresce mais que o CDI
    b = vs_benchmark(pd.Series(strat, index=idx), rate)
    assert b["alpha_aa"] > 0
    assert b["cagr_estrategia"] > b["cagr_cdi"]
