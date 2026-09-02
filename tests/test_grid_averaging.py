import numpy as np
import pandas as pd

from trader.backtest.grid_averaging import run_grid_averaging
from trader.costs.b3 import B3Costs

COSTS = B3Costs()


def _frame(px):
    n = len(px)
    idx = pd.date_range("2015-01-02", periods=n, freq="B")
    px = np.asarray(px, float)
    return pd.DataFrame({"open": px, "high": px * 1.005, "low": px * 0.995,
                         "close": px, "volume": 1e6}, index=idx)


def test_compra_no_barato_e_vende_no_repique():
    # cai 20%, range lateral, depois recupera com folga -> fecha um trade no verde
    px = np.r_[np.full(30, 50.0), np.linspace(50, 40, 20), np.full(20, 40.0),
               np.linspace(40, 54, 30)]
    r = run_grid_averaging({"X": _frame(px)}, COSTS, lookback=10, entry_z=1.0,
                           add_drop_pct=0.05, take_profit_pct=0.03, tranche_frac=0.2,
                           max_adds=3, initial_equity=1_000_000.0, lot_size=100)
    assert r.stats["trades_fechados"] >= 1
    assert r.stats["acerto_trades_fechados"] == 1.0          # nunca realiza perda
    assert r.equity.iloc[-1] > 0


def test_sem_stop_segura_posicao_no_vermelho_ate_o_fim():
    # queda contínua sem recuperação: a posição fica presa, nada é "realizado"
    px = np.r_[np.full(20, 100.0), np.linspace(100, 30, 120)]
    r = run_grid_averaging({"X": _frame(px)}, COSTS, lookback=10, entry_z=0.5,
                           add_drop_pct=0.05, take_profit_pct=0.03, tranche_frac=0.15,
                           max_adds=5, initial_equity=1_000_000.0, lot_size=100)
    assert r.stats["trades_fechados"] == 0
    assert r.stats["posicoes_abertas_no_fim"] == 1
    assert r.stats["prejuizo_nao_realizado"] < 0
    assert r.stats["dias_no_vermelho_max"] > 50


def test_stop_realiza_a_perda():
    px = np.r_[np.full(20, 100.0), np.linspace(100, 30, 120)]
    kw = dict(lookback=10, entry_z=0.5, add_drop_pct=0.05, take_profit_pct=0.03,
              tranche_frac=0.15, max_adds=5, initial_equity=1_000_000.0, lot_size=100)
    r = run_grid_averaging({"X": _frame(px)}, COSTS, stop_pct=0.20, **kw)
    assert r.stats["trades_fechados"] >= 1
    assert r.stats["acerto_trades_fechados"] == 0.0


def test_caixa_rende_cdi_sem_posicao():
    px = np.full(120, 100.0)                                  # nunca fica barato
    rate = pd.Series(0.0004, index=_frame(px).index)
    r = run_grid_averaging({"X": _frame(px)}, COSTS, lookback=10, entry_z=1.0,
                           cash_rate=rate, initial_equity=1_000_000.0, lot_size=100)
    assert r.stats["trades_fechados"] == 0
    assert r.equity.iloc[-1] > 1_000_000.0                    # só CDI
