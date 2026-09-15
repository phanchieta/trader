import numpy as np
import pandas as pd

from trader.backtest.exec_timing import run_exec_timing
from trader.backtest.ml_portfolio import run_target_portfolio
from trader.costs.b3 import B3Costs

FREE = B3Costs(emolumentos_pct=0.0, slippage_ticks=0.0)


def _frame(opens, closes):
    idx = pd.date_range("2024-01-01", periods=len(opens), freq="B", tz="UTC")
    o, c = np.asarray(opens, float), np.asarray(closes, float)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c),
                         "close": c, "volume": 1e6}, index=idx)


def test_abertura_seguinte_igual_ao_motor_padrao():
    rng = np.random.default_rng(0)
    c = 20 * np.exp(np.cumsum(rng.normal(0, 0.02, 120)))
    o = np.r_[c[0], c[:-1]] * np.exp(rng.normal(0, 0.005, 120))
    f = {"X": _frame(o, c), "Y": _frame(o * 1.5, c * 1.5)}
    w = pd.DataFrame({"X": (rng.random(120) > 0.5) * 0.5, "Y": (rng.random(120) > 0.5) * 0.5},
                     index=f["X"].index)
    a = run_exec_timing(f, w, B3Costs(), buy_at="open_next", sell_at="open_next")
    b = run_target_portfolio(f, w, B3Costs(), initial_equity=1_000_000.0, target_is_weight=True)
    assert np.allclose(a.equity.to_numpy(), b.equity.to_numpy())


def test_compra_no_fechamento_ganha_a_noite():
    # sinal de compra no fechamento do dia 1; noite 1->2 sobe 10%, pregão do dia 2 parado
    f = {"X": _frame([10, 10, 11, 11, 11], [10, 10, 11, 11, 11])}
    w = pd.DataFrame({"X": [0.0, 1.0, 1.0, 1.0, 1.0]}, index=f["X"].index)
    kw = dict(initial_equity=100_000.0)
    at_open = run_exec_timing(f, w, FREE, buy_at="open_next", **kw)
    at_close = run_exec_timing(f, w, FREE, buy_at="close_same", **kw)
    late = run_exec_timing(f, w, FREE, buy_at="close_next", **kw)
    assert np.isclose(at_open.equity.iloc[-1], 100_000.0, atol=11.0)     # comprou a 11
    assert at_close.equity.iloc[-1] > 109_000.0                           # comprou a 10
    assert np.isclose(late.equity.iloc[-1], 100_000.0, atol=11.0)        # comprou no fech. do dia 2


def test_venda_na_abertura_e_compra_no_fechamento_ao_mesmo_tempo():
    f = {"X": _frame([10, 10, 12, 12, 12], [10, 10, 12, 9, 9])}
    w = pd.DataFrame({"X": [0.0, 1.0, 0.0, 0.0, 0.0]}, index=f["X"].index)
    r = run_exec_timing(f, w, FREE, buy_at="close_same", sell_at="open_next", initial_equity=100_000.0)
    # compra no fechamento do dia 1 a 10, vende na abertura do dia 2 a 12, escapa da queda do dia 3
    assert np.isclose(r.equity.iloc[-1], 120_000.0)
    assert r.stats["n_ordens"] == 2
