"""Motor diário para listas de trades intraday.

Por dia: unidades = equity × alavancagem / notional do fechamento anterior
(arredondado ao lote); resultado = unidades × Σ pontos × valor do ponto − custos
(taxa B3, corretagem, slippage por ponta). O caixa inteiro rende CDI — em futuro
só a margem fica bloqueada, e ela também rende. Nunca carrega posição de um dia
para o outro.

IR: day trade 20% sobre o resultado líquido do mês com prejuízo compensável; a
parte de CDI a 15,5% (mesma convenção do resto da pesquisa). O 1% retido na fonte
é antecipação do imposto, não custo.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backtest.metrics import summarize, vs_benchmark
from .costs import Instrument


@dataclass
class IntradayResult:
    equity: pd.Series
    equity_after_tax: pd.Series
    stats: dict


def _day_index(days: np.ndarray) -> pd.DatetimeIndex:
    return pd.to_datetime(pd.Series(days).astype(str), format="%Y%m%d").dt.tz_localize("UTC")


def run_intraday(trades: pd.DataFrame, bars: pd.DataFrame, inst: Instrument, *,
                 days: np.ndarray, notional_px: pd.Series, cdi: pd.Series,
                 equity0: float = 1_000_000.0, leverage: float = 1.0,
                 slippage_ticks: float = 1.0, free: bool = False,
                 tax_daytrade: float = 0.20, tax_cdi: float = 0.155) -> IntradayResult:
    """`days`: todos os pregões da janela (dias sem trade rendem só CDI).
    `notional_px`: preço real por dia (fechamento do dia anterior) para dimensionar.
    `free=True`: sem custo nenhum — mede a vantagem bruta do sinal.
    Se `trades` tiver `entry_px`/`exit_px`, usa esses preços em vez das aberturas das
    barras (ex.: trade overnight que entra no fechamento)."""
    o = bars["open"].to_numpy()
    t = trades[trades["day"].isin(days)]
    entry = t["entry_px"].to_numpy() if "entry_px" in t else o[t["entry_i"].to_numpy()]
    exit_ = t["exit_px"].to_numpy() if "exit_px" in t else o[t["exit_i"].to_numpy()]
    dirn = t["direction"].to_numpy()
    pts = dirn * (exit_ - entry)
    slip = 0.0 if free else slippage_ticks
    unit_cost = (np.zeros(len(t)) if free else
                 np.array([inst.unit_cost_side(a, slip) + inst.unit_cost_side(b, slip)
                           for a, b in zip(entry, exit_)]))
    order_cost = 0.0 if free else 2.0 * inst.brokerage_order_side
    g = pd.DataFrame({"day": t["day"].to_numpy(), "pts": pts, "uc": unit_cost, "n": 1})
    per_day = g.groupby("day").sum().reindex(days).fillna(0.0)

    idx = _day_index(days)
    rate = cdi.reindex(idx).ffill().fillna(0.0).to_numpy()
    px = notional_px.reindex(days).ffill().bfill().to_numpy()
    pts_d, uc_d, n_d = (per_day[k].to_numpy() for k in ("pts", "uc", "n"))
    month = days // 100

    def simulate(taxed: bool) -> np.ndarray:
        eq, carry, m_strat, m_cdi = equity0, 0.0, 0.0, 0.0
        out = np.empty(len(days))
        for i in range(len(days)):
            units = np.floor(eq * leverage / (px[i] * inst.point_value) / inst.lot) * inst.lot
            pnl = units * (pts_d[i] * inst.point_value - uc_d[i]) - n_d[i] * order_cost * (units > 0)
            income = eq * rate[i]
            eq += income + pnl
            m_strat += pnl
            m_cdi += income
            if taxed and (i == len(days) - 1 or month[i + 1] != month[i]):
                net = m_strat + carry
                eq -= tax_cdi * m_cdi + (tax_daytrade * net if net > 0 else 0.0)
                carry = 0.0 if net > 0 else net
                m_strat = m_cdi = 0.0
            out[i] = eq
        return out

    eq = pd.Series(simulate(False), index=idx, name="equity")
    eq_at = pd.Series(simulate(True), index=idx, name="equity_pos_ir")
    b = vs_benchmark(eq, cdi)
    b_at = vs_benchmark(eq_at, cdi)
    n_tr = len(t)
    gross_ticks = pts / inst.tick
    stats = {
        "cagr": b["cagr_estrategia"], "alpha": b["alpha_aa"], "alpha_pos_ir": b_at["alpha_aa"],
        "sharpe_cdi": b["sharpe_sobre_cdi"],
        "max_dd": summarize(eq, pd.DataFrame(columns=["fee"]), equity0)["max_drawdown"],
        "trades": n_tr, "trades_dia": round(n_tr / max(len(days), 1), 2),
        "acerto": round(float((pts > 0).mean()), 3) if n_tr else None,
        "bruto_ticks_trade": round(float(gross_ticks.mean()), 2) if n_tr else None,
        "custo_ticks_trade": round(float(unit_cost.mean() / inst.tick_value), 2) if n_tr else None,
    }
    return IntradayResult(eq, eq_at, stats)
