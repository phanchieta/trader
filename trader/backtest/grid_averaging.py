"""Backtest da estratégia de "nunca realizar perda" (averaging down / grid).

Regra — a MESMA para cada ação da cesta, com CAIXA COMPARTILHADO:

  entrada   compra 1 tranche quando o ativo está "mais barato que o comum dos
            últimos dias": close < média(lookback) - entry_z · desvio(lookback).
  averaging se o close cair `add_drop_pct` abaixo do PREÇO MÉDIO da posição,
            compra outra tranche (até `max_adds`). Isso baixa o preço médio.
  saída     vende TUDO quando close ≥ preço médio · (1 + take_profit_pct).
  sem stop  nunca vende no prejuízo. Se o caixa acaba, não abre nada novo —
            segura o que tem até bater o take-profit.
  caixa     o dinheiro parado rende CDI (`cash_rate`, taxa diária).

Sem look-ahead: decide no fechamento de t, executa na abertura de t+1.
Equity = caixa + posições marcadas a mercado (INCLUI as que estão no vermelho).

`stats` traz, além de retorno/Sharpe/maxDD, o custo escondido da regra:
  utilização de caixa (média e pico), dias com posição no vermelho, posições
  ainda abertas no fim (perdas nunca "realizadas"), maior espera até um
  take-profit, e a taxa de acerto dos trades FECHADos (dá ~100% de propósito).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..costs.b3 import B3Costs
from .metrics import summarize


@dataclass
class GridResult:
    equity: pd.Series
    stats: dict
    por_ativo: dict = field(default_factory=dict)


def _common_index(frames: dict) -> pd.DatetimeIndex:
    idx = None
    for f in frames.values():
        idx = f.index if idx is None else idx.intersection(f.index)
    return idx.sort_values()


def run_grid_averaging(
    frames: dict,
    costs: B3Costs,
    *,
    lookback: int = 20,
    entry_z: float = 1.0,
    add_drop_pct: float = 0.05,
    take_profit_pct: float = 0.03,
    tranche_frac: float = 0.10,
    max_adds: int = 4,
    stop_pct: float | None = None,          # se setado, deixa de ser "nunca perde"
    initial_equity: float = 1_000_000.0,
    lot_size: int = 100,
    cash_rate: pd.Series | None = None,
) -> GridResult:
    syms = list(frames)
    idx = _common_index(frames)
    n = len(idx)
    rate = (cash_rate.reindex(idx).ffill().fillna(0.0).to_numpy("float64")
            if cash_rate is not None else np.zeros(n))

    op = {s: frames[s]["open"].reindex(idx).to_numpy("float64") for s in syms}
    cl = {s: frames[s]["close"].reindex(idx).to_numpy("float64") for s in syms}
    # sinal de "barato" avaliado no fechamento; NaN nos primeiros `lookback`
    cheap = {}
    for s in syms:
        c = pd.Series(cl[s])
        m = c.rolling(lookback).mean()
        sd = c.rolling(lookback).std()
        cheap[s] = (c < m - entry_z * sd).to_numpy()

    tranche_cash = tranche_frac * initial_equity
    cash = float(initial_equity)
    sh = {s: 0.0 for s in syms}
    avg = {s: 0.0 for s in syms}
    adds = {s: 0 for s in syms}
    opened_i = {s: -1 for s in syms}

    equity = np.empty(n)
    invested = np.empty(n)
    underwater_days = {s: 0 for s in syms}
    max_uw_days = {s: 0 for s in syms}
    worst_uw_pct = {s: 0.0 for s in syms}
    closed = []          # (pnl, dias, n_tranches)
    fees_total = 0.0

    def _buy(s, price, i):
        nonlocal cash, fees_total
        qty = np.trunc(tranche_cash / price / lot_size) * lot_size
        if qty <= 0:
            return False
        fee = costs.order_cost(price, qty)
        if cash < qty * price + fee:
            return False
        cash -= qty * price + fee
        fees_total += fee
        new_sh = sh[s] + qty
        avg[s] = (avg[s] * sh[s] + price * qty) / new_sh
        sh[s] = new_sh
        if opened_i[s] < 0:
            opened_i[s] = i
        return True

    def _sell_all(s, price, i):
        nonlocal cash, fees_total
        fee = costs.order_cost(price, sh[s])
        cash += sh[s] * price - fee
        fees_total += fee
        closed.append((sh[s] * (price - avg[s]) - fee, i - opened_i[s], adds[s] + 1))
        sh[s] = 0.0
        avg[s] = 0.0
        adds[s] = 0
        opened_i[s] = -1

    for i in range(n):
        cash *= 1.0 + rate[i]
        for s in syms:
            oi = op[s][i]
            if not np.isfinite(oi) or oi <= 0.0 or i == 0:
                continue
            pc = cl[s][i - 1]                       # decisão tomada no fech. de i-1
            if sh[s] > 0.0:
                if pc >= avg[s] * (1.0 + take_profit_pct):
                    _sell_all(s, oi, i)
                elif stop_pct is not None and pc <= avg[s] * (1.0 - stop_pct):
                    _sell_all(s, oi, i)            # variante COM stop
                elif pc <= avg[s] * (1.0 - add_drop_pct) and adds[s] < max_adds:
                    if _buy(s, oi, i):
                        adds[s] += 1
            elif cheap[s][i - 1]:
                _buy(s, oi, i)

        pos_val = 0.0
        for s in syms:
            pv = sh[s] * cl[s][i] if sh[s] > 0.0 and np.isfinite(cl[s][i]) else 0.0
            pos_val += pv
            if sh[s] > 0.0 and cl[s][i] < avg[s]:
                underwater_days[s] += 1
                max_uw_days[s] = max(max_uw_days[s], underwater_days[s])
                worst_uw_pct[s] = min(worst_uw_pct[s], cl[s][i] / avg[s] - 1.0)
            elif sh[s] == 0.0:
                underwater_days[s] = 0
        equity[i] = cash + pos_val
        invested[i] = pos_val / equity[i] if equity[i] > 0 else 0.0

    eq = pd.Series(equity, index=idx, name="equity")
    last_px = {s: cl[s][-1] for s in syms}
    open_now = {s: {"tranches": adds[s] + 1, "dias": n - 1 - opened_i[s],
                    "pnl_aberto": sh[s] * (last_px[s] - avg[s])}
                for s in syms if sh[s] > 0.0}
    pnls = [c[0] for c in closed]
    fee_frame = pd.DataFrame({"fee": [fees_total]}) if fees_total else \
        pd.DataFrame(columns=["price", "shares", "fee"])
    stats = summarize(eq, fee_frame, initial_equity)
    stats.update({
        "trades_fechados": len(closed),
        "acerto_trades_fechados": _r((np.array(pnls) > 0).mean() if pnls else np.nan),
        "espera_max_ate_lucro_dias": max((c[1] for c in closed), default=0),
        "tranches_por_trade_media": _r(np.mean([c[2] for c in closed]) if closed else np.nan),
        "caixa_investido_medio": _r(float(np.nanmean(invested))),
        "caixa_investido_pico": _r(float(np.nanmax(invested))),
        "dias_no_vermelho_max": max(max_uw_days.values()),
        "queda_max_sob_agua": _r(min(worst_uw_pct.values())),
        "posicoes_abertas_no_fim": len(open_now),
        "prejuizo_nao_realizado": _r(sum(v["pnl_aberto"] for v in open_now.values()
                                         if v["pnl_aberto"] < 0)),
    })
    return GridResult(eq, stats, {"abertas_no_fim": open_now,
                                  "dias_vermelho_por_ativo": max_uw_days})


def _r(x):
    return round(float(x), 4) if x == x else None
