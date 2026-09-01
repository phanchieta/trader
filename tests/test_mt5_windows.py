from datetime import datetime, timezone

from trader.data.mt5_source import _windows

_S = datetime(2020, 1, 1, tzinfo=timezone.utc)
_E = datetime(2020, 3, 1, tzinfo=timezone.utc)      # 60 dias


def test_janelas_contiguas_e_cobrem_tudo():
    w = _windows(_S, _E, chunk_days=30)
    assert w[0][0] == _S
    assert w[-1][1] == _E
    for (_, a_end), (b_start, _) in zip(w, w[1:]):
        assert a_end == b_start                       # sem buraco, sem sobreposição


def test_ultima_janela_nao_passa_do_fim():
    w = _windows(_S, _E, chunk_days=7)
    assert all(end <= _E for _, end in w)
    assert (w[-1][1] - w[-1][0]).days <= 7


def test_range_menor_que_um_chunk():
    w = _windows(_S, _E, chunk_days=365)
    assert w == [(_S, _E)]
