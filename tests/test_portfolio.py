import numpy as np
import pandas as pd

from trader.backtest.portfolio import run_portfolio
from trader.costs.b3 import B3Costs
from trader.strategy import SmaCrossover


def _frame(n, seed, drift=0.0003):
    rng = np.random.default_rng(seed)
    px = 30.0 * np.exp(np.cumsum(rng.normal(drift, 0.02, n)))
    idx = pd.date_range("2015-01-02", periods=n, freq="B", tz="UTC")
    idx.name = "time"
    return pd.DataFrame({"open": px, "high": px * 1.01, "low": px * 0.99,
                         "close": px, "volume": 1e6}, index=idx)


def _basket(n=600):
    return {f"S{k}": _frame(n, seed=k) for k in range(5)}


def test_roda_e_respeita_lote():
    frames = _basket()
    res = run_portfolio(frames, SmaCrossover(20, 60), B3Costs(), lot_size=100)
    assert len(res.equity) == len(next(iter(frames.values())))
    assert set(res.trades_por_ativo) == set(frames)


def test_sem_lookahead_na_carteira():
    frames = _basket(700)
    strat, costs = SmaCrossover(15, 40), B3Costs()
    full = run_portfolio(frames, strat, costs)
    cut = 400
    sliced = run_portfolio({s: f.iloc[:cut] for s, f in frames.items()}, strat, costs)
    np.testing.assert_allclose(full.equity.iloc[:cut].to_numpy(),
                               sliced.equity.to_numpy(), rtol=1e-9)


def test_overlay_zero_zera_a_carteira():
    frames = _basket()
    idx = next(iter(frames.values())).index
    zero = pd.Series(0.0, index=idx)
    res = run_portfolio(frames, SmaCrossover(20, 60), B3Costs(), overlay_mult=zero)
    assert res.stats["n_trades"] == 0
    assert abs(res.equity.iloc[-1] - 100_000.0) < 1e-6


def test_overlay_reduz_exposicao():
    frames = _basket()
    idx = next(iter(frames.values())).index
    full = run_portfolio(frames, SmaCrossover(20, 60), B3Costs(),
                         overlay_mult=pd.Series(1.0, index=idx))
    half = run_portfolio(frames, SmaCrossover(20, 60), B3Costs(),
                         overlay_mult=pd.Series(0.5, index=idx))
    # metade da exposição -> desvio da equity inicial menor em magnitude
    assert (half.equity - 100_000).abs().mean() < (full.equity - 100_000).abs().mean()


def test_trailing_stop_corta_apos_recuo_do_pico():
    # sobe devagar até 60, despenca abrupto para 48 (queda rápida demais para a
    # média reagir) e fica de lado. O trailing de 12% (gatilho 52,8) protege
    # o lucro antes da estratégia sair.
    px = np.concatenate([np.linspace(30, 60, 150),
                         np.linspace(60, 48, 8),
                         np.full(102, 48.0)])
    idx = pd.date_range("2015-01-02", periods=len(px), freq="B", tz="UTC")
    idx.name = "time"
    f = pd.DataFrame({"open": px, "high": px * 1.005, "low": px * 0.995,
                      "close": px, "volume": 1e6}, index=idx)
    frames = {"A": f, "B": f.copy()}
    strat, costs = SmaCrossover(10, 30), B3Costs()
    com = run_portfolio(frames, strat, costs, trailing_stop=0.12)
    sem = run_portfolio(frames, strat, costs)
    assert com.equity.iloc[-1] > sem.equity.iloc[-1]


def test_trailing_stop_causal():
    frames = _basket(700)
    strat, costs = SmaCrossover(15, 40), B3Costs()
    full = run_portfolio(frames, strat, costs, trailing_stop=0.1)
    cut = 400
    sliced = run_portfolio({s: f.iloc[:cut] for s, f in frames.items()},
                           strat, costs, trailing_stop=0.1)
    np.testing.assert_allclose(full.equity.iloc[:cut].to_numpy(),
                               sliced.equity.to_numpy(), rtol=1e-9)
