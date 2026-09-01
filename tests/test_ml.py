import numpy as np
import pandas as pd
import pytest

from trader.ml.dataset import build_panel, feature_columns, purge
from trader.ml.features import build_features
from trader.ml.labels import triple_barrier
from trader.ml.model import LgbmSignal


def _frame(n=900, seed=1):
    rng = np.random.default_rng(seed)
    c = 40 * np.exp(np.cumsum(rng.normal(0.0004, 0.018, n)))
    o = c * (1 + rng.normal(0, 0.003, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.004, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.004, n)))
    idx = pd.date_range("2015-01-02", periods=n, freq="B", tz="UTC")
    idx.name = "time"
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                         "volume": rng.integers(1e6, 9e6, n) * 1.0}, index=idx)


def test_features_causal_e_sem_inf():
    df = _frame(900)
    full = build_features(df)
    cut = 600
    trunc = build_features(df.iloc[:cut])
    pd.testing.assert_frame_equal(full.iloc[:cut], trunc, check_dtype=False)
    assert not np.isinf(full.iloc[300:].to_numpy(dtype="float64")).any()


def test_panel_empilha_e_dropna():
    frames = {"A": _frame(400, 1), "B": _frame(400, 2)}
    panel = build_panel(frames, horizon=10, cost_hurdle=0.004)
    assert set(panel["symbol"].unique()) == {"A", "B"}
    assert set(panel["_y"].unique()) <= {0, 1}
    assert panel[feature_columns(panel)].notna().all().all()


def test_purge_remove_cauda_do_treino():
    frames = {"A": _frame(900, 1)}
    panel = build_panel(frames, horizon=10, cost_hurdle=0.0)
    ts = panel.index[int(len(panel) * 0.6)]
    kept = purge(panel, ts, horizon=10, embargo=5)
    train_dates = sorted(panel.index[panel.index < ts].unique())
    assert kept.index.max() < train_dates[-15]      # corta as últimas 15 datas


def test_triple_barrier_alvo_vs_stop():
    # sobe forte -> deve bater o alvo (1) antes do stop
    up = 40 * np.exp(np.cumsum(np.full(120, 0.01)))
    idx = pd.date_range("2020-01-02", periods=120, freq="B", tz="UTC")
    idx.name = "time"
    df = pd.DataFrame({"open": up, "high": up * 1.01, "low": up * 0.99,
                       "close": up, "volume": 1e6}, index=idx)
    y = triple_barrier(df, horizon=10, pt_atr=2.0, sl_atr=1.5, atr_window=14)
    assert set(y.dropna().unique()) <= {0.0, 1.0}
    assert y.iloc[20:100].mean() > 0.9          # tendência de alta -> quase tudo 1


def test_xs_ranks_no_painel():
    frames = {s: _frame(900, i) for i, s in enumerate("ABC")}
    panel = build_panel(frames, horizon=10, label="triple_barrier", add_xs_ranks=True)
    xs = [c for c in feature_columns(panel) if c.startswith("xs_")]
    assert xs
    assert ((panel[xs] >= 0.0) & (panel[xs] <= 1.0)).all().all()
    assert "month" not in panel.columns          # drop_cols default


def test_lgbm_fit_predict_smoke():
    frames = {s: _frame(900, i) for i, s in enumerate("ABCD")}
    panel = build_panel(frames, horizon=10, cost_hurdle=0.004)
    cols = feature_columns(panel)
    cut = int(len(panel) * 0.7)
    m = LgbmSignal().fit(panel[cols].iloc[:cut], panel["_y"].iloc[:cut])
    p = m.predict(panel[cols].iloc[cut:cut + 200])
    assert p.shape[0] == 200
    assert ((p >= 0.0) & (p <= 1.0)).all()
    assert len(m.importances()) == len(cols)
