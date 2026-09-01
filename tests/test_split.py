import pandas as pd

from trader.backtest.split import WalkForward, train_test_split


def _df(n):
    idx = pd.date_range("2015-01-02", periods=n, freq="B", tz="UTC")
    idx.name = "time"
    return pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                         "volume": 1.0}, index=idx)


def test_train_test_sem_sobreposicao():
    tr, te = train_test_split(_df(1000), test_frac=0.3)
    assert tr.index.max() < te.index.min()
    assert len(te) == 300


def test_embargo_encurta_o_treino_pelo_fim():
    tr0, _ = train_test_split(_df(1000), test_frac=0.3, embargo=0)
    tr5, te5 = train_test_split(_df(1000), test_frac=0.3, embargo=5)
    assert len(tr0) - len(tr5) == 5
    assert tr5.index.max() < te5.index.min()


def test_walk_forward_particiona_a_cauda():
    sp = WalkForward(n_folds=4, test_frac=0.4).splits(1000)
    assert len(sp) == 4

    starts = [s[1].start for s in sp]
    stops = [s[1].stop for s in sp]
    assert starts[0] == 600 and stops[-1] == 1000
    for prev_stop, nxt_start in zip(stops[:-1], starts[1:]):
        assert prev_stop == nxt_start                 # testes contíguos
    for tr, te in sp:
        assert tr.stop <= te.start                    # treino sempre antes do teste


def test_walk_forward_rolante_tem_treino_de_tamanho_fixo():
    sp = WalkForward(n_folds=3, test_frac=0.45, anchored=False).splits(1200)
    tamanhos = {tr.stop - tr.start for tr, _ in sp}
    assert len(tamanhos) == 1
