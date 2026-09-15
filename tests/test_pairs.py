import numpy as np
import pandas as pd

from trader.pairs import (adf_tstat, build_weights, eligibility, half_life, log_spread,
                          pair_positions, pairs_portfolio, rolling_z)


def _px(values, start="2020-01-01", volume=1e6):
    idx = pd.date_range(start, periods=len(values), freq="B", tz="UTC")
    v = np.asarray(values, float)
    return pd.DataFrame({"open": v, "high": v, "low": v, "close": v, "volume": volume}, index=idx)


def test_estados_entrada_saida_stop_e_bloqueio():
    z = np.array([0, -2.5, -1.0, 0.1, 2.5, 3.0, 4.5, 3.0, 2.5, 1.0, 2.2, 0.0])
    ok = np.ones(len(z), dtype=bool)
    pos = pair_positions(z, ok, entry=2.0, exit_z=0.0, stop_z=4.0, max_hold=None)
    assert list(pos) == [0, 1, 1, 0, -1, -1, 0, 0, 0, 0, -1, 0]
    # bloqueado depois do stop até |z| < entry (t=9), e só então reentra (t=10)


def test_prazo_maximo_e_inelegivel_zera():
    z = np.full(10, -3.0)
    pos = pair_positions(z, np.ones(10, bool), entry=2.0, stop_z=None, max_hold=3)
    assert list(pos[:5]) == [1, 1, 1, 0, 0]
    ok = np.ones(10, bool)
    ok[1] = False
    pos2 = pair_positions(np.full(10, -3.0), ok, entry=2.0, stop_z=None, max_hold=None)
    assert pos2[1] == 0


def test_z_causal_e_estatisticas_de_reversao():
    rng = np.random.default_rng(0)
    ou = np.zeros(3000)
    for t in range(1, 3000):
        ou[t] = 0.95 * ou[t - 1] + rng.normal(0, 0.01)
    s = pd.Series(ou)
    z1 = rolling_z(s, 60)
    s2 = s.copy()
    s2.iloc[2000:] += 5.0
    assert np.allclose(z1.iloc[:2000].dropna(), rolling_z(s2, 60).iloc[:2000].dropna())
    assert 8 < half_life(s) < 20                    # ln2/0.05 ≈ 13,5
    assert adf_tstat(s) < -5
    assert adf_tstat(pd.Series(np.cumsum(rng.normal(0, 0.01, 3000)))) > -2.86


def test_elegibilidade_por_liquidez_e_salto():
    a = _px(np.full(100, 10.0), volume=1e6)          # R$10 mi/dia
    b = _px(np.full(100, 20.0), volume=1e3)          # R$20 mil/dia
    assert not eligibility(a, b, min_traded_value=5e6).any()
    b2 = _px(np.r_[np.full(80, 20.0), np.full(20, 40.0)], volume=1e6)
    ok = eligibility(a, b2, min_traded_value=5e6)
    assert ok.iloc[70] and not ok.iloc[81]


def test_slots_priorizam_maior_z_e_pesos_neutros():
    idx = pd.date_range("2020-01-01", periods=3, freq="B", tz="UTC")
    pos = {"P1": np.array([1, 1, 0]), "P2": np.array([-1, -1, -1]), "P3": np.array([1, 1, 1])}
    zs = {"P1": np.array([-2.1, -2.2, 0]), "P2": np.array([3.0, 2.5, 2.5]),
          "P3": np.array([-2.6, -2.6, -2.6])}
    legs = {"P1": ("A1", "B1"), "P2": ("A2", "B2"), "P3": ("A3", "B3")}
    W, held = build_weights(pos, zs, legs, idx, slots=2, gross=1.0)
    assert list(held.iloc[0]) == [0, -1, 1]          # P1 ficou de fora (menor |z|)
    assert np.allclose(W.sum(axis=1), 0.0)
    assert np.isclose(W.abs().sum(axis=1).iloc[0], 1.0)


def test_carteira_execucao_no_open_seguinte_custo_e_aluguel():
    idx = pd.date_range("2020-01-01", periods=4, freq="B", tz="UTC")
    opens = pd.DataFrame({"A": [10, 10, 11, 11], "B": [10, 10, 10, 10]}, index=idx, dtype=float)
    closes = pd.DataFrame({"A": [10, 10, 11, 11], "B": [10, 10, 10, 10]}, index=idx, dtype=float)
    W = pd.DataFrame({"A": [0.5, 0.5, 0.5, 0.5], "B": [-0.5, -0.5, -0.5, -0.5]}, index=idx)
    held = pd.DataFrame({"P": [1, 1, 1, 1]}, index=idx, dtype=float)
    cdi = pd.Series(0.0, index=idx)
    r = pairs_portfolio(W, held, opens, closes, {"P": ("A", "B")}, cdi, cost_side=0.001,
                        borrow_aa=0.0)
    # peso decidido no fech. do dia 0 entra no open do dia 1; A sobe 10% no gap do dia 2
    assert np.isclose(r.excess.iloc[1], -0.001)                 # giro de 1,0 x 10 bps
    assert np.isclose(r.excess.iloc[2], 0.05)                   # 0,5 x 10%
    paid = pairs_portfolio(W, held, opens, closes, {"P": ("A", "B")}, cdi, cost_side=0.0,
                           borrow_aa=0.10)
    assert paid.excess.iloc[3] < 0
    s = log_spread(closes["A"], closes["B"])
    assert np.isclose(s.iloc[2], np.log(1.1))
