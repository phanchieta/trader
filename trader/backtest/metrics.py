"""Métricas de desempenho. rf = 0 nesta versão (trocar por CDI na Fase 1)."""
from __future__ import annotations

import numpy as np
import pandas as pd

_PREGAO_SEG = 6.5 * 3600      # ~6h30 de pregão na B3
_DIAS_ANO = 252.0


def _bar_seconds(index: pd.DatetimeIndex) -> float | None:
    """Espaçamento mediano entre barras, em segundos. Independe da resolução
    (ns/us/ms) do índice."""
    if len(index) < 3:
        return None
    deltas = pd.Series(index).diff().dropna()
    if deltas.empty:
        return None
    return float(deltas.median() / pd.Timedelta(seconds=1))


def _periods_per_year(index: pd.DatetimeIndex) -> float:
    s = _bar_seconds(index)
    if not s or s <= 0:
        return _DIAS_ANO
    if s >= 20 * 3600:                                  # barras diárias ou maiores
        return _DIAS_ANO
    return _DIAS_ANO * _PREGAO_SEG / s                  # intraday


def _years(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 0.0
    return float((index[-1] - index[0]) / pd.Timedelta(days=365.25))


def _round(x, nd):
    return round(float(x), nd) if x == x else None      # None em vez de NaN


def summarize(equity: pd.Series, trades: pd.DataFrame, initial: float) -> dict:
    ret = equity.pct_change().dropna()
    ppy = _periods_per_year(equity.index)
    final = float(equity.iloc[-1]) if len(equity) else float(initial)

    total_return = final / initial - 1.0
    years = _years(equity.index)
    try:
        cagr = ((final / initial) ** (1.0 / years) - 1.0
                if (final > 0 and years > 0) else np.nan)
    except (OverflowError, ZeroDivisionError):
        cagr = np.nan
    vol = ret.std() * np.sqrt(ppy) if len(ret) else np.nan
    sharpe = ret.mean() / ret.std() * np.sqrt(ppy) if ret.std() else np.nan

    dd = equity / equity.cummax() - 1.0
    max_dd = dd.min() if len(dd) else np.nan

    fees = float(trades["fee"].sum()) if len(trades) else 0.0
    gross_profit = final - initial + fees
    cost_ratio = fees / gross_profit if gross_profit > 0 else np.nan

    step = equity.diff().dropna()
    wins, losses = step[step > 0].sum(), -step[step < 0].sum()
    profit_factor = wins / losses if losses > 0 else np.nan

    return {
        "retorno_total": _round(total_return, 4),
        "cagr": _round(cagr, 4),
        "vol_anual": _round(vol, 4),
        "sharpe": _round(sharpe, 2),
        "max_drawdown": _round(max_dd, 4),
        "n_trades": int(len(trades)),
        "custo_total_R$": _round(fees, 2),
        "custo_sobre_lucro_bruto": _round(cost_ratio, 3),
        "profit_factor_aprox": _round(profit_factor, 2),
    }


def vs_benchmark(equity: pd.Series, daily_rate: pd.Series) -> dict:
    """Compara a curva de equity com um benchmark de renda fixa (taxa DIÁRIA,
    ex.: CDI). Retorna CAGR de cada um, alpha (excesso a.a.) e o Sharpe medido
    SOBRE o benchmark — o número que importa no Brasil."""
    idx = equity.index
    r = daily_rate.reindex(idx).ffill().fillna(0.0)
    bench = (1.0 + r).cumprod()
    years = _years(idx)
    ppy = _periods_per_year(idx)

    def _cagr(series):
        f = float(series.iloc[-1] / series.iloc[0])
        return f ** (1.0 / years) - 1.0 if (f > 0 and years > 0) else float("nan")

    strat_ret = equity.pct_change().dropna()
    excess = strat_ret - r.reindex(strat_ret.index)
    sh_ex = (excess.mean() / excess.std() * np.sqrt(ppy)) if excess.std() else np.nan
    win = float((excess > 0).mean())

    c_strat, c_bench = _cagr(equity), _cagr(bench)
    return {
        "cagr_estrategia": _round(c_strat, 4),
        "cagr_cdi": _round(c_bench, 4),
        "alpha_aa": _round(c_strat - c_bench, 4),
        "sharpe_sobre_cdi": _round(sh_ex, 2),
        "dias_batendo_cdi": _round(win, 3),
    }
