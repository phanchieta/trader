import numpy as np
import pandas as pd

from trader.live.planner import plan_targets
from trader.live.state import PortfolioState


class _Const:
    def __init__(self, v):
        self.value = float(v)
        self.name = "const"

    def generate_signals(self, df):
        return pd.Series(self.value, index=df.index)


def _frame(prices):
    idx = pd.date_range("2020-01-02", periods=len(prices), freq="B", tz="UTC")
    idx.name = "time"
    p = np.asarray(prices, dtype=float)
    return pd.DataFrame({"open": p, "high": p * 1.01, "low": p * 0.99,
                         "close": p, "volume": 1e6}, index=idx)


def test_alvo_respeita_lote_e_capital():
    frames = {"A": _frame([10, 10, 10, 20.0])}   # último fechamento = 20
    plan = plan_targets(frames, _Const(1.0), PortfolioState(),
                        capital_per_name=10_000.0, lot_size=100)
    # 10000 / 20 = 500 ações, já múltiplo de 100
    assert plan.targets["A"] == 500.0


def test_sinal_zero_zera_alvo_e_destrava():
    st = PortfolioState()
    st.get("A").shares = 100.0
    st.get("A").blocked = True
    plan = plan_targets({"A": _frame([10, 11, 12.0])}, _Const(0.0), st,
                        capital_per_name=10_000.0)
    assert plan.targets["A"] == 0.0
    assert st.get("A").blocked is False


def test_trailing_stop_dispara_e_zera():
    st = PortfolioState()
    ss = st.get("A")
    ss.shares, ss.entry, ss.peak = 100.0, 10.0, 20.0     # comprado, no lucro, pico 20
    # fecha a 17 -> recuo de 15% do pico (> 10%)
    plan = plan_targets({"A": _frame([10, 15, 20, 17.0])}, _Const(1.0), st,
                        capital_per_name=10_000.0, trailing_stop=0.10)
    assert st.get("A").blocked is True
    assert plan.targets["A"] == 0.0
    assert any("trailing-stop disparou" in n for n in plan.notes)


def test_trailing_nao_arma_no_prejuizo():
    st = PortfolioState()
    ss = st.get("A")
    ss.shares, ss.entry, ss.peak = 100.0, 20.0, 20.0     # entrou a 20, nunca subiu
    plan = plan_targets({"A": _frame([20, 19, 17.0])}, _Const(1.0), st,
                        capital_per_name=10_000.0, trailing_stop=0.10)
    assert st.get("A").blocked is False                   # sem lucro -> não arma
    assert plan.targets["A"] > 0.0


def test_overlay_mult_escala_o_alvo():
    frames = {"A": _frame([10, 10, 20.0])}
    full = plan_targets(frames, _Const(1.0), PortfolioState(),
                        capital_per_name=10_000.0, overlay_mult=1.0).targets["A"]
    half = plan_targets(frames, _Const(1.0), PortfolioState(),
                        capital_per_name=10_000.0, overlay_mult=0.5).targets["A"]
    assert half == 200.0 and full == 500.0
