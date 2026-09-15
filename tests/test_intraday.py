import numpy as np
import pandas as pd

from trader.intraday import (WIN, BarIndex, cash_close_minute, intraday_momentum,
                             opening_range_breakout, run_intraday, scalp_m1)
from trader.intraday.costs import Instrument


def _bars(day_prices: dict, first=9 * 60, last=18 * 60 + 24):
    """{'2026-03-02': array de closes (1 por minuto a partir de `first`)} -> M1 OHLC em UTC."""
    frames = []
    for d, closes in day_prices.items():
        closes = np.asarray(closes, float)
        mins = first + np.arange(len(closes))
        assert mins[-1] <= last
        t = (pd.Timestamp(d, tz="America/Sao_Paulo") + pd.to_timedelta(mins, unit="min")).tz_convert("UTC")
        op = np.r_[closes[0], closes[:-1]]
        frames.append(pd.DataFrame({"open": op, "high": np.maximum(op, closes) + 1,
                                    "low": np.minimum(op, closes) - 1, "close": closes,
                                    "volume": 1.0}, index=t))
    return pd.concat(frames)


def _flat_day(level=100_000.0, n=18 * 60 + 25 - 9 * 60):
    return np.full(n, level)


def test_fechamento_do_a_vista_encaixa_na_grade():
    idx = _bars({"2026-03-02": _flat_day()[: (16 * 60 + 51) - 9 * 60],
                 "2026-03-03": _flat_day()[: (17 * 60 + 55) - 9 * 60]}).index
    cm = cash_close_minute(idx)
    assert list(cm.to_numpy()) == [16 * 60 + 54, 17 * 60 + 54]


def test_barindex_busca_no_mesmo_dia():
    b = _bars({"2026-03-02": _flat_day(), "2026-03-03": _flat_day()})
    bi = BarIndex(b.index)
    i = bi.at_or_after(np.array([20260302]), np.array([600]))[0]
    assert bi.minute[i] == 600 and bi.day[i] == 20260302
    assert bi.at_or_after(np.array([20260302]), np.array([19 * 60]))[0] == -1
    assert bi.at_or_before(np.array([20260303]), np.array([8 * 60]))[0] == -1


def test_momentum_intraday_compra_apos_primeira_meia_hora_de_alta():
    n = 18 * 60 + 25 - 9 * 60
    d1 = _flat_day()
    d2 = _flat_day().copy()
    d2[(10 * 60 + 29) - 9 * 60:] += 500.0            # alta até 10:30
    b = _bars({"2026-03-02": d1, "2026-03-03": d2})
    cm = pd.Series([16 * 60 + 54] * 2, index=[20260302, 20260303])
    tr = intraday_momentum(b, cm)
    assert len(tr) == 1
    row = tr.iloc[0]
    bi = BarIndex(b.index)
    assert row.direction == 1
    assert bi.minute[row.entry_i] == 16 * 60 + 25 and bi.minute[row.exit_i] == 16 * 60 + 55
    d2b = d2.copy()
    d2b[(16 * 60) - 9 * 60:] -= 3000.0                # futuro depois das 10:30 não muda o sinal
    tr2 = intraday_momentum(_bars({"2026-03-02": d1, "2026-03-03": d2b}), cm)
    assert tr2.iloc[0].direction == 1
    assert n > 0


def test_orb_rompe_para_cima_e_stopa_no_fundo_do_range():
    p = _flat_day().copy()
    p[20] = 100_300.0                                 # rompe a máxima do range (9:00–9:15)
    p[21:40] = 100_300.0
    p[40:] = 99_000.0                                 # perde a mínima -> stop
    b = _bars({"2026-03-02": p})
    cm = pd.Series([16 * 60 + 54], index=[20260302])
    tr = opening_range_breakout(b, cm, anchor=540, minutes=15, stop=True)
    assert len(tr) == 1 and tr.iloc[0].direction == 1
    assert tr.iloc[0].entry_i == 21 and tr.iloc[0].exit_i == 41


def test_scalp_sem_sobreposicao_e_sai_no_mesmo_dia():
    rng = np.random.default_rng(0)
    days = {f"2026-03-0{k}": 100_000 + np.cumsum(rng.normal(0, 20, 18 * 60 + 25 - 9 * 60))
            for k in (2, 3, 4)}
    b = _bars(days)
    cm = pd.Series([16 * 60 + 54] * 3, index=[20260302, 20260303, 20260304])
    tr = scalp_m1(b, cm, k=1.5, hold=10)
    bi = BarIndex(b.index)
    assert len(tr) > 5
    assert (tr["exit_i"] - tr["entry_i"] == 10).all()
    assert (bi.day[tr["exit_i"]] == bi.day[tr["entry_i"]]).all()
    assert (tr["entry_i"].to_numpy()[1:] >= tr["exit_i"].to_numpy()[:-1]).all()


def test_motor_custos_cdi_e_ir():
    b = _bars({"2026-03-02": _flat_day(), "2026-03-03": _flat_day()})
    b.iloc[100:, b.columns.get_loc("open")] = 100_050.0          # +50 pts no trade
    trades = pd.DataFrame({"day": [20260302], "entry_i": [90], "exit_i": [100], "direction": [1]})
    days = np.array([20260302, 20260303])
    px = pd.Series(100_000.0, index=days)
    idx = pd.to_datetime(["2026-03-02", "2026-03-03"]).tz_localize("UTC")
    cdi = pd.Series(0.0004, index=idx)
    inst = Instrument("X", point_value=0.20, tick=5.0, b3_fee_unit_side=0.25)
    r = run_intraday(trades, b, inst, days=days, notional_px=px, cdi=cdi, slippage_ticks=1.0)
    units = np.floor(1_000_000 / (100_000 * 0.2))                 # 50 contratos
    pnl = units * (50 * 0.2 - 2 * (0.25 + 1.0))
    day1 = 1_000_000 * 1.0004 + pnl
    assert np.isclose(r.equity.iloc[0], day1)
    assert np.isclose(r.equity.iloc[1], day1 * 1.0004)
    none = run_intraday(trades.iloc[:0], b, WIN, days=days, notional_px=px, cdi=cdi)
    assert np.isclose(none.equity.iloc[-1], 1_000_000 * 1.0004 ** 2)
    assert r.equity_after_tax.iloc[-1] < r.equity.iloc[-1]
    over = trades.assign(entry_px=[100_000.0], exit_px=[100_050.0])     # preços explícitos
    r2 = run_intraday(over, b.assign(open=0.0), inst, days=days, notional_px=px, cdi=cdi)
    assert np.isclose(r2.equity.iloc[0], day1)
