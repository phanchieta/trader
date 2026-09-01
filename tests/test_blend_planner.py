from datetime import date

import numpy as np
import pandas as pd

from trader.live.blend_planner import needs_factor_rebalance, plan_blend


def _frame(px):
    n = len(px)
    idx = pd.date_range("2016-01-04", periods=n, freq="B", tz="UTC")
    idx.name = "time"
    px = np.asarray(px, float)
    return pd.DataFrame({"open": px, "high": px * 1.01, "low": px * 0.99,
                         "close": px, "volume": 1e6}, index=idx)


def _up(n=800, start=30, seed=0):
    rng = np.random.default_rng(seed)
    return _frame(start * np.exp(np.cumsum(rng.normal(0.0008, 0.012, n))))


def test_rebalance_por_trimestre():
    assert needs_factor_rebalance(date(2026, 4, 1), None) is True
    assert needs_factor_rebalance(date(2026, 4, 1), "2026-03-31") is True   # Q1 -> Q2
    assert needs_factor_rebalance(date(2026, 5, 20), "2026-04-01") is False  # mesmo Q2
    assert needs_factor_rebalance(date(2026, 7, 1), "2026-05-20") is True    # Q2 -> Q3


def test_plan_blend_combina_as_duas_sleeves():
    champ = {s: _up(seed=i) for i, s in enumerate(["PETR4", "VALE3", "ITUB4"])}
    universe = dict(champ)
    for i, s in enumerate(["A", "B", "C", "D", "E", "F", "G", "H"]):
        universe[s] = _up(seed=10 + i)
    ibov = _up(seed=99)

    plan = plan_blend(champ, universe, ibov, equity=1_000_000.0, today=date(2026, 4, 1),
                      prev_quintile=[], last_rebalance=None, champ_w=0.5, n_frac=0.25,
                      lot=100, target_vol=0.15, vol_lb=20, max_lev=1.0, min_names=5)

    assert plan.rebalanced is True
    assert len(plan.quintile) == round(len(universe) * 0.25)
    # soma dos pesos <= ~1 (as duas sleeves somam <= champ_w + (1-champ_w))
    assert sum(plan.weights.values()) <= 1.0 + 1e-6
    # nome que está nas duas sleeves acumula peso
    both = set(champ) & set(plan.quintile)
    for s in both:
        assert plan.weights[s] > 0.5 * (1 / len(champ)) - 1e-9
    assert all(v % 100 == 0 for v in plan.target_shares.values())


def test_plan_blend_congela_quintil_fora_do_rebalance():
    champ = {s: _up(seed=i) for i, s in enumerate(["PETR4", "VALE3"])}
    universe = dict(champ)
    for i, s in enumerate("ABCDEF"):
        universe[s] = _up(seed=20 + i)
    ibov = _up(seed=7)
    frozen = ["A", "C", "E"]
    plan = plan_blend(champ, universe, ibov, equity=1e6, today=date(2026, 5, 10),
                      prev_quintile=frozen, last_rebalance="2026-04-01",
                      champ_w=0.5, n_frac=0.5, min_names=5)
    assert plan.rebalanced is False
    assert plan.quintile == frozen
