import numpy as np
import pandas as pd

from trader.factors.backtest import beta_hedge, build_target


def _score(n=200, k=6, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-02", periods=n, freq="B", tz="UTC")
    idx.name = "time"
    return pd.DataFrame(rng.random((n, k)), index=idx,
                        columns=[f"S{i}" for i in range(k)])


def test_build_target_pesos_somam_1_no_topo():
    sc = _score()
    tgt = build_target(sc, sc.index, n_frac=0.34, rebal=10)   # top ~2 de 6
    row = tgt.iloc[50]
    assert abs(row.sum() - 1.0) < 1e-9
    assert (row > 0).sum() == 2
    assert set(row[row > 0].round(6)) == {round(0.5, 6)}


def test_build_target_congela_entre_rebalances():
    sc = _score()
    tgt = build_target(sc, sc.index, n_frac=0.34, rebal=20)
    # dentro de uma janela de 20 dias sem rebalance, o alvo não muda
    seg = tgt.iloc[21:40]
    assert (seg.nunique() == 1).all()


def test_beta_hedge_reduz_exposicao_ao_indice():
    n = 500
    idx = pd.date_range("2018-01-02", periods=n, freq="B", tz="UTC")
    idx.name = "time"
    rng = np.random.default_rng(3)
    mkt_ret = rng.normal(0.0003, 0.012, n)
    ibov = pd.Series(3000 * np.exp(np.cumsum(mkt_ret)), index=idx)
    book_ret = 1.1 * mkt_ret + rng.normal(0, 0.004, n)        # beta ~1,1
    book = pd.Series(1e5 * np.exp(np.cumsum(book_ret)), index=idx)

    hedged = beta_hedge(book, ibov, window=60)
    rb = book.pct_change().iloc[80:]
    rh = hedged.pct_change().iloc[80:]
    rm = ibov.pct_change().iloc[80:]
    assert abs(rh.corr(rm)) < abs(rb.corr(rm))                # menos ligado ao índice
