import pandas as pd

from trader.backtest.split import WalkForward
from trader.backtest.validation import grid_search, holdout, walk_forward
from trader.costs.b3 import B3Costs
from trader.strategy import MeanReversion, Momentum

_MR_GRID = {"lookback": [10, 20], "entry_z": [1.0, 2.0], "exit_z": [0.5],
            "allow_short": [False]}
_MOM_GRID = {"lookback": [20, 40], "band": [0.0], "trend_filter": [0],
             "allow_short": [False]}


def test_grid_search_so_enxerga_o_treino(daily_ohlcv):
    df = daily_ohlcv(900)
    train = df.iloc[:600]
    costs = B3Costs()
    a, _ = grid_search(MeanReversion, _MR_GRID, train, costs, min_trades=1)
    # concatenar dados novos DEPOIS do treino não pode mudar a escolha
    b, _ = grid_search(MeanReversion, _MR_GRID, df.iloc[:600], costs, min_trades=1)
    assert a == b
    assert set(a) == set(_MR_GRID)


def test_holdout_mede_em_periodo_disjunto(daily_ohlcv):
    df = daily_ohlcv(1000)
    rep = holdout(MeanReversion, _MR_GRID, df, B3Costs(),
                  test_frac=0.3, min_trades=1)
    fronteira = df.index[int(len(df) * 0.7) - 1]
    assert rep.oos_equity.index[0] > fronteira
    assert isinstance(rep.degradacao(), dict)
    assert "sharpe" in rep.is_stats and "sharpe" in rep.oos_stats


def test_walk_forward_costura_curva_oos(daily_ohlcv):
    df = daily_ohlcv(1300)
    rep = walk_forward(Momentum, _MOM_GRID, df, B3Costs(),
                       wf=WalkForward(n_folds=3, test_frac=0.45), min_trades=1)
    assert len(rep.folds) == 3
    assert len(rep.best_params) == 3
    # a curva OOS cobre exatamente a soma dos folds de teste, em ordem
    total_oos = sum(sl.stop - sl.start
                    for _, sl in WalkForward(n_folds=3, test_frac=0.45).splits(len(df)))
    assert len(rep.oos_equity) == total_oos
    assert rep.oos_equity.index.is_monotonic_increasing
