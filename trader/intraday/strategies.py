"""Estratégias intraday em barras M1. Cada uma devolve a LISTA DE TRADES
(dia, barra de entrada, barra de saída, direção): o sinal fecha na barra t e a
execução é na ABERTURA da barra seguinte; toda posição sai no mesmo dia.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .sessions import BarIndex

TRADE_COLS = ["day", "entry_i", "exit_i", "direction"]


def _frame(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=TRADE_COLS).astype(
        {"day": "int64", "entry_i": "int64", "exit_i": "int64", "direction": "int64"})


def intraday_momentum(bars: pd.DataFrame, cash_close: pd.Series, *, first_end: int = 630,
                      overnight: bool = True, confirm_r12: bool = False) -> pd.DataFrame:
    """Gao, Han, Li & Zhou (2018, JFE): o retorno da primeira meia hora (desde o
    fechamento anterior até 10:30) prevê a última meia hora do pregão à vista.
    Entra na abertura da barra fechamento−29min, sai na abertura da barra seguinte
    ao fechamento. `overnight=False` mede só 10:00→10:30. `confirm_r12`: exige a
    penúltima meia hora no mesmo sentido."""
    bi = BarIndex(bars.index)
    o, c = bars["open"].to_numpy(), bars["close"].to_numpy()
    days = np.array(sorted(set(np.unique(bi.day)) & set(cash_close.index)))
    cm = cash_close.reindex(days).to_numpy()
    i_prev = bi.at_or_before(np.r_[-1, days[:-1]], np.r_[0, cm[:-1]])
    i_open = bi.at_or_after(days, 600)
    i_first = bi.at_or_before(days, first_end - 1)
    i_entry = bi.at_or_after(days, cm - 29)
    i_exit = bi.at_or_after(days, cm + 1)
    i_r12a, i_r12b = bi.at_or_before(days, cm - 60), bi.at_or_before(days, cm - 30)

    base = i_prev if overnight else i_open
    ok = (base >= 0) & (i_first > base) & (i_entry > i_first) & (i_exit > i_entry)
    ref = np.where(base >= 0, c[np.maximum(base, 0)] if overnight else o[np.maximum(base, 0)], np.nan)
    r1 = np.where(ok, c[np.maximum(i_first, 0)] - ref, 0.0)
    d = np.sign(r1)
    if confirm_r12:
        ok12 = (i_r12a >= 0) & (i_r12b > i_r12a)
        r12 = np.where(ok12, c[np.maximum(i_r12b, 0)] - c[np.maximum(i_r12a, 0)], 0.0)
        d = np.where(np.sign(r12) == d, d, 0.0)
    keep = ok & (d != 0)
    return _frame(zip(days[keep], i_entry[keep], i_exit[keep], d[keep].astype(int)))


def opening_range_breakout(bars: pd.DataFrame, cash_close: pd.Series, *, anchor: int = 540,
                           minutes: int = 15, stop: bool = True) -> pd.DataFrame:
    """Primeiro fechamento acima da máxima (compra) ou abaixo da mínima (venda) dos
    `minutes` iniciais a partir de `anchor`. Stop no lado oposto do range (checado
    no fechamento) ou só saída no horário: abertura da barra após o fechamento do
    à vista. Máximo 1 trade por dia."""
    bi = BarIndex(bars.index)
    h, l, c = (bars[k].to_numpy() for k in ("high", "low", "close"))
    rows = []
    days, starts, ends = bi.day_bounds()
    for day, s, e in zip(days, starts, ends):
        if day not in cash_close.index:
            continue
        cm = int(cash_close.loc[day])
        m = bi.minute[s:e]
        rng = (m >= anchor) & (m < anchor + minutes)
        if rng.sum() < max(1, minutes // 2):
            continue
        hi, lo = h[s:e][rng].max(), l[s:e][rng].min()
        exit_rel = np.nonzero(m >= cm + 1)[0]
        if not len(exit_rel):
            continue
        t_exit = s + int(exit_rel[0])
        post = np.nonzero((m >= anchor + minutes) & (m <= cm - 2))[0]
        cc = c[s:e]
        brk = post[(cc[post] > hi) | (cc[post] < lo)]
        if not len(brk):
            continue
        j = s + int(brk[0])
        d = 1 if c[j] > hi else -1
        x = t_exit
        if stop:
            after = np.arange(j + 1, t_exit)
            hit = after[(c[after] < lo) if d == 1 else (c[after] > hi)]
            if len(hit):
                x = int(hit[0]) + 1
        rows.append((day, j + 1, x, d))
    return _frame(rows)


def scalp_m1(bars: pd.DataFrame, cash_close: pd.Series, *, k: float = 3.0, hold: int = 15,
             follow: bool = True, vol_window: int = 60, start: int = 605) -> pd.DataFrame:
    """'Scalping' em barra de 1 minuto: se o retorno do minuto passar `k` desvios da
    vol dos `vol_window` minutos anteriores, entra a favor (`follow`) ou contra na
    abertura seguinte e sai `hold` barras depois. Sem sobreposição; só entre
    `start` e o fechamento do à vista."""
    bi = BarIndex(bars.index)
    c = bars["close"].to_numpy()
    r = np.diff(c, prepend=np.nan)
    first = np.r_[True, bi.day[1:] != bi.day[:-1]]
    r[first] = np.nan
    sd = pd.Series(r).rolling(vol_window, min_periods=vol_window // 2).std().shift(1).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        z = r / sd
    cm = cash_close.reindex(bi.day).to_numpy(dtype="float64")
    ok = (np.abs(z) >= k) & (bi.minute >= start) & (bi.minute <= cm - hold - 2)
    rows, free = [], 0
    for i in np.nonzero(ok)[0]:
        if i < free:
            continue
        ei, xi = i + 1, i + 1 + hold
        if xi >= bi.n or bi.day[xi] != bi.day[i]:
            continue
        rows.append((bi.day[i], ei, xi, int(np.sign(r[i])) * (1 if follow else -1)))
        free = i + hold
    return _frame(rows)
