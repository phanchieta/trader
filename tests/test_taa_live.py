import numpy as np
import pandas as pd

from trader.allocation import faber_sma
from trader.allocation.live import brl, faber_signal, last_completed_month_end, plan_orders


def test_reais_no_padrao_brasileiro():
    assert brl(1234.5) == "R$ 1.234,50"
    assert brl(3253.5, 0) == "R$ 3.254"


def _closes(n=400):
    cal = pd.bdate_range("2025-01-01", periods=n)
    up = pd.Series(100 * np.exp(np.linspace(0, 0.5, n)), index=cal)
    down = pd.Series(100 * np.exp(np.linspace(0, -0.5, n)), index=cal)
    return pd.DataFrame({"UP": up, "DOWN": down})


def test_ultimo_mes_fechado():
    cal = pd.bdate_range("2026-08-20", "2026-09-10")
    assert last_completed_month_end(cal, pd.Timestamp("2026-09-10")) == pd.Timestamp("2026-08-31")
    assert last_completed_month_end(cal, pd.Timestamp("2026-08-25")) is None


def test_sinal_ao_vivo_igual_ao_backtest():
    c = _closes()
    me = last_completed_month_end(c.index, c.index[-1])
    sig = faber_signal(c, me, months=10)
    bt = faber_sma(c, months=10)
    assert sig.month_end == me
    assert sig.weights == bt.loc[me].to_dict()
    assert sig.weights["UP"] == 0.5 and sig.weights["DOWN"] == 0.0
    assert bool(sig.detail.loc["UP", "acima"]) and not bool(sig.detail.loc["DOWN", "acima"])


def test_ordens_em_cotas_inteiras_com_banda():
    prices = {"A": 100.0, "B": 50.0}
    tgt, orders, eq = plan_orders({"A": 0.5, "B": 0.0}, prices, {}, 10_000.0)
    assert eq == 10_000.0 and tgt == {"A": 50.0, "B": 0.0} and orders == {"A": 50.0}
    # 5% fora do alvo: dentro da banda de 20%, não mexe
    tgt, orders, _ = plan_orders({"A": 0.5, "B": 0.0}, prices, {"A": 45}, 5_500.0)
    assert orders == {}
    # sinal saiu: vende tudo mesmo dentro da banda
    tgt, orders, _ = plan_orders({"A": 0.0, "B": 0.0}, prices, {"A": 50}, 5_000.0)
    assert orders == {"A": -50.0}
