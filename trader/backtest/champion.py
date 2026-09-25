"""Campeão (SuperTrend long-only + vol-target Yang-Zhang da cesta) sobre o painel da B3,
com cesta escolhida só com o que se sabia na data.

`causal_members`: no 1º pregão de cada ano, as `n` empresas mais negociadas (mediana
de R$ em `liq_window` pregões, uma classe por empresa, preço ≥ `min_price`, negociou
nos últimos 5 pregões).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..costs.b3 import B3Costs
from ..indicators import yang_zhang
from ..strategy import SuperTrend
from .ml_portfolio import run_target_portfolio


def causal_members(panel: dict, calendar: pd.DatetimeIndex, *, n: int = 5, liq_window: int = 250,
                   min_price: float = 1.0) -> tuple[pd.DataFrame, list]:
    liq = pd.DataFrame({i: f["volume"].reindex(calendar).fillna(0.0)
                        .rolling(liq_window, min_periods=int(liq_window * 0.8)).median()
                        for i, f in panel.items()})
    ok = pd.DataFrame({i: (f["raw_close"].reindex(calendar) >= min_price)
                       & (f["traded"].reindex(calendar).fillna(False).astype(float)
                          .rolling(5, min_periods=1).max() > 0)
                       for i, f in panel.items()}).fillna(False)
    company = {i: str(f["codneg"].dropna().iloc[-1])[:4] for i, f in panel.items()}
    member = pd.DataFrame(0.0, index=calendar, columns=list(panel))
    history = []
    for y in sorted(set(calendar.year))[1:]:
        days = calendar[calendar.year == y]
        decide = calendar[calendar < days[0]][-1]
        cand = liq.loc[decide].where(ok.loc[decide]).dropna().sort_values(ascending=False)
        picks, seen = [], set()
        for isin in cand.index:
            if company[isin] in seen:
                continue
            picks.append(isin)
            seen.add(company[isin])
            if len(picks) == n:
                break
        member.loc[days, picks] = 1.0
        history.append((y, [panel[i]["codneg"].reindex(calendar).loc[:decide].dropna().iloc[-1] for i in picks]))
    return member, history


def champion_targets(members: pd.DataFrame, frames: dict, calendar: pd.DatetimeIndex, *, n: int,
                     target_vol: float = 0.15, lookback: int = 20, max_leverage: float = 1.0,
                     st_period: int = 10, st_mult: float = 3.0):
    """(colunas usadas, alvo em peso por ação = sinal/n dos membros, multiplicador de vol)."""
    st = SuperTrend(st_period, st_mult)
    cols = [c for c in members.columns if members[c].any()]
    sig = pd.DataFrame({c: st.generate_signals(frames[c].dropna(subset=["close"])).reindex(calendar)
                        for c in cols}).fillna(0.0)
    vol = pd.DataFrame({c: yang_zhang(frames[c].dropna(subset=["close"]), window=lookback).reindex(calendar)
                        for c in cols})
    basket_vol = vol.where(members[cols] > 0).mean(axis=1)
    mult = (target_vol / basket_vol).clip(upper=max_leverage).shift(1).fillna(0.0)
    return cols, sig * members[cols] / n, mult


def run_champion(cols, target, mult, frames: dict, win: pd.DatetimeIndex, costs: B3Costs,
                 cash_rate: pd.Series, *, capital: float = 1_000_000.0, lot: int = 100,
                 band: float = 0.0):
    """Equity (índice UTC) e estatísticas do motor."""
    res = run_target_portfolio({c: frames[c] for c in cols}, target.reindex(win), costs,
                               overlay_mult=mult.reindex(win), cash_rate=cash_rate, target_is_weight=True,
                               initial_equity=capital, lot_size=lot, rebalance_band=band)
    eq = res.equity
    if eq.index.tz is None:
        eq = eq.set_axis(eq.index.tz_localize("UTC"))
    return eq, res.stats
