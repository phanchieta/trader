import pytest

from trader.data.yahoo import _intraday_days


def test_clamp_ao_teto_do_yahoo():
    assert _intraday_days("1m") == 7
    assert _intraday_days("5m") == 60
    assert _intraday_days("15m") == 60
    assert _intraday_days("1h") == 730
    assert _intraday_days("60m") == 730          # alias


def test_lookback_menor_que_o_teto_e_respeitado():
    assert _intraday_days("5m", 10) == 10
    assert _intraday_days("5m", 999) == 60       # não passa do teto


def test_interval_invalido():
    with pytest.raises(ValueError):
        _intraday_days("3m")
