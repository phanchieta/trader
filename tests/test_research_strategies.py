import numpy as np
import pandas as pd
import pytest

from trader.strategy import (Donchian, GapReversal, MeanReversion, QuantileChannel,
                             ReturnDistFade, SmaCrossover, WeightedEnsemble)


def _frame(o, h, l, c):
    n = len(c)
    idx = pd.date_range("2020-01-02", periods=n, freq="B", tz="UTC")
    idx.name = "time"
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                         "volume": np.full(n, 1e6)}, index=idx)


def _rw(n=400, seed=1):
    rng = np.random.default_rng(seed)
    c = 40.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.02, n)))
    o = c * (1 + rng.normal(0, 0.002, n))
    h = np.maximum(o, c) * 1.01
    l = np.minimum(o, c) * 0.99
    return _frame(o, h, l, c)


# ---------- GapReversal ----------

def test_gap_reversal_dispara_e_segura():
    # dia 3: abre bem abaixo da mínima do dia 2 e fecha de volta acima dela
    o = [40, 40, 40, 36.0, 40, 40, 40, 40]
    h = [41, 41, 41, 40.0, 41, 41, 41, 41]
    l = [39, 39, 39, 35.5, 39, 39, 39, 39]
    c = [40, 40, 40, 39.5, 40, 40, 40, 40]
    sig = GapReversal(hold_days=3, min_gap_pct=0.005).generate_signals(_frame(o, h, l, c))
    assert sig.iloc[3] == 1.0
    assert (sig.iloc[3:6] == 1.0).all()      # segura 3 dias
    assert sig.iloc[6] == 0.0


def test_gap_reversal_sem_gap_fica_zerado():
    df = _rw(200)
    df.iloc[:, :] = df.iloc[:, :].clip(lower=1.0)
    sig = GapReversal(min_gap_pct=0.5).generate_signals(df)   # gap mínimo enorme
    assert (sig == 0.0).all()


# ---------- Donchian ----------

def test_donchian_entra_no_rompimento_e_segura_na_alta():
    c = np.concatenate([np.full(25, 40.0), np.linspace(41, 60, 40)])
    df = _frame(c, c * 1.005, c * 0.995, c)
    sig = Donchian(entry_lookback=20, exit_lookback=10).generate_signals(df)
    assert sig.iloc[-1] == 1.0
    assert set(sig.unique()) <= {0.0, 1.0}


def test_donchian_causal():
    df = _rw(500)
    strat = Donchian(20, 10)
    full = strat.generate_signals(df)
    cut = 300
    pd.testing.assert_series_equal(full.iloc[:cut],
                                   strat.generate_signals(df.iloc[:cut]), check_names=False)


# ---------- WeightedEnsemble ----------

def test_weighted_ensemble_alvo_em_menos_um_a_um():
    df = _rw(600)
    ens = WeightedEnsemble([SmaCrossover(10, 40), MeanReversion(20, 1.5, 0.5),
                            Donchian(20, 10)], weight_lookback=120)
    t = ens.generate_signals(df)
    assert t.between(-1.0, 1.0).all()
    assert t.index.equals(df.index)


def test_weighted_ensemble_favorece_o_membro_bom():
    # série em tendência forte: SMA acerta, um "anti-sinal" erra sempre
    n = 500
    c = 40.0 * np.exp(np.cumsum(np.full(n, 0.004)))
    df = _frame(c, c * 1.001, c * 0.999, c)

    class _Anti:
        name = "anti"
        def generate_signals(self, d):
            return pd.Series(-1.0, index=d.index)

    ens = WeightedEnsemble([SmaCrossover(10, 40), _Anti()], weight_lookback=100)
    t = ens.generate_signals(df).iloc[200:]
    assert t.mean() > 0.3        # o combinado fica comprado (segue o membro bom)


def test_weighted_ensemble_causal():
    df = _rw(500)
    ens = WeightedEnsemble([SmaCrossover(10, 40), Donchian(20, 10)], weight_lookback=120)
    full = ens.generate_signals(df)
    cut = 300
    pd.testing.assert_series_equal(full.iloc[:cut],
                                   ens.generate_signals(df.iloc[:cut]), check_names=False)


def test_weighted_ensemble_sem_membros_erra():
    with pytest.raises(ValueError):
        WeightedEnsemble([]).generate_signals(_rw(50))


# ---------- ReturnDistFade ----------

def test_return_dist_fade_compra_apos_queda_anomala():
    n = 200
    c = np.full(n, 40.0) + np.random.default_rng(0).normal(0, 0.05, n)
    c[150] = 40.0 * 0.85          # -15% num dia: fora da distribuição calma
    df = _frame(c, c * 1.001, c * 0.999, c)
    sig = ReturnDistFade(dist_window=120, entry_z=2.0, max_hold=10).generate_signals(df)
    assert sig.iloc[150] == 1.0


def test_return_dist_fade_valido_e_causal():
    df = _rw(500)
    strat = ReturnDistFade(dist_window=120, entry_z=2.0)
    full = strat.generate_signals(df)
    assert set(full.unique()) <= {-1.0, 0.0, 1.0}
    cut = 300
    pd.testing.assert_series_equal(full.iloc[:cut],
                                   strat.generate_signals(df.iloc[:cut]), check_names=False)


# ---------- QuantileChannel ----------

def test_quantile_channel_revert_compra_no_fundo():
    # dente de serra: repetidas idas ao fundo do canal
    base = np.tile(np.concatenate([np.linspace(40, 44, 15), np.linspace(44, 40, 15)]), 8)
    df = _frame(base, base * 1.002, base * 0.998, base)
    sig = QuantileChannel(window=40, lower_q=0.15, upper_q=0.85,
                          mode="revert").generate_signals(df)
    assert set(sig.unique()) <= {0.0, 1.0}
    assert (sig == 1.0).any()               # entrou comprado em algum fundo


def test_quantile_channel_causal():
    df = _rw(500)
    strat = QuantileChannel(window=60, lower_q=0.1, upper_q=0.9)
    full = strat.generate_signals(df)
    cut = 300
    pd.testing.assert_series_equal(full.iloc[:cut],
                                   strat.generate_signals(df.iloc[:cut]), check_names=False)
