"""Decide a posição-alvo de cada ação para a PRÓXIMA abertura, a partir do
histórico + estado persistido.

Mesma regra do backtest (backtest.portfolio.run_portfolio):
  - sinal calculado no fechamento de hoje vale para a abertura de amanhã;
  - trailing-stop arma só depois que a posição fica no azul; ao disparar,
    zera a ação até o sinal voltar a passar por zero.
Dimensiona pelo último fechamento (aproxima a abertura de amanhã).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .state import PortfolioState


@dataclass
class Plan:
    targets: dict = field(default_factory=dict)   # sym -> nº de ações alvo
    notes: list = field(default_factory=list)


def plan_targets(frames: dict, strategy, state: PortfolioState, *,
                 capital_per_name: float, lot_size: int = 100,
                 trailing_stop: float | None = None,
                 overlay_mult: float = 1.0) -> Plan:
    plan = Plan()
    for sym, df in frames.items():
        ss = state.get(sym)
        sig = float(strategy.generate_signals(df).iloc[-1])
        last_close = float(df["close"].iloc[-1])

        if trailing_stop and ss.shares > 0.0:
            ss.peak = max(ss.peak, last_close)
            if ss.peak > ss.entry and last_close <= ss.peak * (1.0 - trailing_stop):
                if not ss.blocked:
                    plan.notes.append(
                        f"{sym}: trailing-stop disparou (fechou {last_close:.2f}, "
                        f"pico {ss.peak:.2f}, entrada {ss.entry:.2f})")
                ss.blocked = True
        elif trailing_stop and ss.shares < 0.0:
            ss.peak = min(ss.peak, last_close) if ss.peak > 0 else last_close
            if ss.peak < ss.entry and last_close >= ss.peak * (1.0 + trailing_stop):
                if not ss.blocked:
                    plan.notes.append(f"{sym}: trailing-stop disparou (venda)")
                ss.blocked = True

        if sig == 0.0:
            ss.blocked = False

        eff = 0.0 if ss.blocked else sig
        desired = 0.0
        if eff != 0.0 and last_close > 0.0 and overlay_mult > 0.0:
            raw = capital_per_name * eff * overlay_mult / last_close
            desired = float(np.trunc(raw / lot_size) * lot_size)
        plan.targets[sym] = desired
    return plan
