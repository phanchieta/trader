import numpy as np
import pandas as pd

from trader.backtest.ml_portfolio import run_target_portfolio
from trader.costs.b3 import B3Costs
from trader.regime import (BEAR, BULL, SIDE, expanding_matrices, label_states,
                           markov_signal, stationary_distribution, transition_matrix)


def _close(rets, start=100.0):
    idx = pd.date_range("2010-01-04", periods=len(rets) + 1, freq="B")
    return pd.Series(start * np.exp(np.r_[0.0, np.cumsum(rets)]), index=idx)


def test_rotulo_limites_exclusivos():
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    c = pd.Series([100.0, 125.0, 93.75, 60.0, 100.0], index=idx)   # razões exatas em binário
    st = label_states(c, window=1, threshold=0.25)
    assert np.isnan(st.iloc[0])
    assert st.iloc[1] == SIDE                     # exatamente +25% não é bull
    assert st.iloc[2] == SIDE                     # exatamente −25% não é bear
    assert st.iloc[3] == BEAR                     # −36%
    assert st.iloc[4] == BULL                     # +66,7%


def test_matriz_por_contagem():
    P = transition_matrix([0, 0, 1, 2, 2, 2, 1])
    assert np.allclose(P.sum(axis=1), 1.0)
    assert np.isclose(P[BEAR, BEAR], 0.5) and np.isclose(P[BEAR, SIDE], 0.5)
    assert np.isclose(P[BULL, BULL], 2 / 3) and np.isclose(P[BULL, SIDE], 1 / 3)
    assert np.isclose(P[SIDE, BULL], 1.0)


def test_matrizes_expansiveis_sao_causais():
    s = pd.Series([0, 0, 1, 2, 2, 2, 1, 0, 0, 1], dtype=float)
    P, n = expanding_matrices(s)
    s2 = s.copy()
    s2.iloc[6:] = [2, 2, 2, 2]                    # muda só o futuro
    P2, _ = expanding_matrices(s2)
    assert np.allclose(P[:6], P2[:6])
    assert n[0] == 0 and n[5] == 5
    assert np.allclose(P[5], transition_matrix(s.iloc[:6]))


def test_distribuicao_estacionaria():
    P = np.array([[0.8, 0.15, 0.05], [0.1, 0.8, 0.1], [0.05, 0.15, 0.8]])
    pi = stationary_distribution(P)
    assert np.isclose(pi.sum(), 1.0)
    assert np.allclose(pi @ P, pi)


def test_sinal_respeita_treino_minimo_e_formatos():
    rng = np.random.default_rng(1)
    c = _close(rng.normal(0.0005, 0.02, 800))
    sg = markov_signal(c, min_train=252, sizing="sign")
    assert (sg.iloc[:252] == 0).all()
    assert set(np.unique(sg)) <= {-1.0, 0.0, 1.0}
    pr = markov_signal(c, sizing="prob")
    assert pr.between(-1, 1).all()
    lo = markov_signal(c, sizing="prob", long_only=True)
    assert (lo >= 0).all()


def test_persistencia_e_artefato_da_janela_sobreposta():
    # retornos iid (sem regime nenhum): a matriz sobreposta parece "pegajosa" mesmo
    # assim; sem sobreposição a diagonal cai para ~frequência incondicional
    rng = np.random.default_rng(7)
    c = _close(rng.normal(0.0, 0.02, 6000))
    st = label_states(c)
    assert np.diag(transition_matrix(st)).min() > 0.75
    assert np.diag(transition_matrix(st.iloc[::20])).max() < 0.6


def test_banda_de_rebalanceamento_corta_ordens():
    idx = pd.date_range("2020-01-01", periods=40, freq="B")
    rng = np.random.default_rng(3)
    px = 10.0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
    frame = pd.DataFrame({"open": px, "high": px, "low": px, "close": px, "volume": 1e6}, index=idx)
    tgt = pd.DataFrame({"X": 1.0}, index=idx)
    # o vol-target oscilando é o que gera os ajustes de 100 ações todo dia
    mult = pd.Series(0.5 * (1.0 + 0.03 * np.sin(np.arange(len(idx)))), index=idx)
    kw = dict(initial_equity=100_000.0, lot_size=100, target_is_weight=True, overlay_mult=mult)
    tight = run_target_portfolio({"X": frame}, tgt, B3Costs(), **kw)
    banded = run_target_portfolio({"X": frame}, tgt, B3Costs(), rebalance_band=0.10, **kw)
    assert tight.stats["n_trades"] > 20
    assert banded.stats["n_trades"] < tight.stats["n_trades"] / 2
    assert banded.stats["custo_total_R$"] < tight.stats["custo_total_R$"]


def test_aluguel_do_short_e_exposicao():
    idx = pd.date_range("2020-01-01", periods=60, freq="B")
    px = pd.DataFrame({"open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0,
                       "volume": 1e6}, index=idx)
    tgt = pd.DataFrame({"X": -1.0}, index=idx)
    kw = dict(initial_equity=100_000.0, lot_size=100)
    free = run_target_portfolio({"X": px}, tgt, B3Costs(), **kw)
    paid = run_target_portfolio({"X": px}, tgt, B3Costs(), borrow_rate_aa=0.10, **kw)
    assert paid.equity.iloc[-1] < free.equity.iloc[-1]
    assert paid.stats["aluguel_R$"] > 0 and free.stats["aluguel_R$"] == 0
    assert 0.9 < free.stats["exposure_media"] <= 1.0
