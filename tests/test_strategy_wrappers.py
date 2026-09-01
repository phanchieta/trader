import numpy as np
import pandas as pd
import pytest

from trader.strategy import Ensemble, MeanReversion, Momentum, SmaCrossover, VolumeConfirmed


class _Const:
    """Estratégia de teste: sinal fixo."""
    def __init__(self, value):
        self.value = float(value)
        self.name = f"const{value}"

    def generate_signals(self, df):
        return pd.Series(self.value, index=df.index, name="target")


def _df(n=300, seed=1, vol_scale=1.0):
    rng = np.random.default_rng(seed)
    px = 30.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, n)))
    vol = rng.lognormal(mean=15, sigma=0.4, size=n) * vol_scale
    idx = pd.date_range("2015-01-02", periods=n, freq="B", tz="UTC")
    idx.name = "time"
    return pd.DataFrame({"open": px, "high": px * 1.01, "low": px * 0.99,
                         "close": px, "volume": vol}, index=idx)


# ---------- VolumeConfirmed ----------

def test_volume_baixo_bloqueia_entrada():
    df = _df()
    df.loc[:, "volume"] = 1.0                       # volume mínimo constante -> rel ~1
    df.iloc[50:, df.columns.get_loc("volume")] = 1e-9   # volume despenca
    sig = VolumeConfirmed(_Const(1.0), lookback=20, min_rel_volume=1.5).generate_signals(df)
    assert (sig.iloc[60:] == 0.0).all()             # sem volume, não entra


def test_volume_saida_sempre_passa():
    df = _df()
    base = SmaCrossover(10, 30)
    vc = VolumeConfirmed(base, lookback=20, min_rel_volume=99.0)  # nunca confirma
    sig = vc.generate_signals(df)
    assert set(sig.unique()) <= {-1.0, 0.0, 1.0}
    assert (sig == 0.0).mean() > 0.9               # quase sempre fora (nada confirma)


def test_volume_confirmed_causal():
    df = _df(400)
    vc = VolumeConfirmed(SmaCrossover(10, 30), lookback=20, min_rel_volume=1.1)
    full = vc.generate_signals(df)
    cut = 250
    trunc = vc.generate_signals(df.iloc[:cut])
    pd.testing.assert_series_equal(full.iloc[:cut], trunc, check_names=False)


# ---------- Ensemble ----------

def test_ensemble_and_vs_or():
    df = _df()
    members = [_Const(1.0), _Const(1.0), _Const(0.0)]
    assert (Ensemble(members, min_agree=2).generate_signals(df) == 1.0).all()
    assert (Ensemble(members, min_agree=3).generate_signals(df) == 0.0).all()


def test_ensemble_valores_validos_e_causal():
    df = _df(400)
    ens = Ensemble([SmaCrossover(10, 30), MeanReversion(20, 1.5, 0.5), Momentum(60)],
                   min_agree=2)
    full = ens.generate_signals(df)
    assert set(full.unique()) <= {-1.0, 0.0, 1.0}
    cut = 250
    pd.testing.assert_series_equal(full.iloc[:cut],
                                   ens.generate_signals(df.iloc[:cut]), check_names=False)


def test_ensemble_sem_membros_erra():
    with pytest.raises(ValueError):
        Ensemble([], min_agree=1).generate_signals(_df())
