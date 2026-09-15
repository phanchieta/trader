import numpy as np
import pandas as pd

from trader.seasonal import month_offsets, timing_returns, tom_flag


def _days():
    return pd.DatetimeIndex(pd.to_datetime(
        ["2026-01-28", "2026-01-29", "2026-01-30", "2026-02-02", "2026-02-03",
         "2026-02-04", "2026-02-05", "2026-02-26", "2026-02-27", "2026-03-02"]), tz="UTC")


def test_posicao_no_mes_pelo_calendario_de_pregoes():
    off = month_offsets(_days())
    assert list(off["from_start"]) == [1, 2, 3, 1, 2, 3, 4, 5, 6, 1]
    assert list(off["from_end"].iloc[:9]) == [-3, -2, -1, -6, -5, -4, -3, -2, -1]
    assert np.isnan(off["from_end"].iloc[-1])            # último mês pode estar incompleto


def test_janela_da_virada():
    flag = tom_flag(month_offsets(_days()), start=-1, end=3)
    assert list(flag) == [True, True, True, True, True, True, False, False, True, True]
    # jan/28 e jan/29 entram por serem os pregões 1 e 2 da série (mês começa no dado)


def test_execucao_no_fechamento_e_na_abertura():
    idx = pd.date_range("2026-03-02", periods=4, freq="B", tz="UTC")
    closes = pd.Series([100.0, 100.0, 110.0, 110.0], index=idx)
    opens = pd.Series([100.0, 100.0, 105.0, 110.0], index=idx)   # dia 2: gap +5%, pregão +4,76%
    hold = pd.Series([0.0, 0.0, 1.0, 0.0], index=idx)
    cdi = pd.Series(0.0, index=idx)
    at_close = timing_returns(hold, opens, closes, cdi, cost_side=0.0, execution="close")
    at_open = timing_returns(hold, opens, closes, cdi, cost_side=0.0, execution="open")
    assert np.isclose(at_close.returns.iloc[2], 0.10)
    assert np.isclose(at_open.returns.iloc[2], 110 / 105 - 1)     # perde o gap
    cash = timing_returns(pd.Series(0.0, index=idx), opens, closes,
                          pd.Series(0.001, index=idx), cost_side=0.0)
    assert np.isclose(cash.equity.iloc[-1], 1_000_000 * 1.001 ** 4)
    paid = timing_returns(hold, opens, closes, cdi, cost_side=0.001, execution="close")
    assert np.isclose(paid.returns.iloc[2], 0.10 - 0.001) and np.isclose(paid.returns.iloc[3], -0.001)
