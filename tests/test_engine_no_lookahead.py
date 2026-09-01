"""Garante que o sinal em t não usa dados de t+1..N e que o backtest é causal."""
import numpy as np
import pandas as pd

from trader.backtest.engine import run
from trader.costs.b3 import B3Costs
from trader.strategy.sma_crossover import SmaCrossover


def _synthetic(n=400, seed=1):
    rng = np.random.default_rng(seed)
    price = 40 + np.cumsum(rng.normal(0, 0.3, n))
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    idx.name = "time"
    return pd.DataFrame({"open": price, "high": price + 0.1, "low": price - 0.1,
                         "close": price, "volume": 1e6}, index=idx)


def test_sinal_e_causal():
    df = _synthetic()
    strat = SmaCrossover(fast=10, slow=30)
    full = strat.generate_signals(df)
    cut = 250
    truncado = strat.generate_signals(df.iloc[:cut])
    pd.testing.assert_series_equal(full.iloc[:cut], truncado, check_names=False)


def test_barras_futuras_nao_mudam_equity_passada():
    df = _synthetic()
    strat = SmaCrossover(fast=10, slow=30)
    costs = B3Costs()
    cut = 250
    r_full = run(df, strat, costs)
    r_cut = run(df.iloc[:cut], strat, costs)
    np.testing.assert_allclose(r_full.equity.iloc[:cut].to_numpy(),
                               r_cut.equity.to_numpy(), rtol=1e-9)


def test_engine_respeita_lote_100():
    df = _synthetic()
    res = run(df, SmaCrossover(fast=10, slow=30), B3Costs(), lot_size=100)
    if len(res.trades):
        posicoes = res.trades["shares"].cumsum()
        assert (posicoes % 100 == 0).all()
