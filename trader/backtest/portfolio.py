"""Backtest multi-ativo. Uma estratégia (o MESMO conjunto de parâmetros) roda
em cada ação da cesta; o capital é dividido igualmente entre as N ações e
remarcado a cada barra. Sem look-ahead: sinal de t executa na abertura de t+1.

Opcionais:
  overlay_mult   série [0, max_lev] que escala a exposição de todas as ações
                 (ver trader.risk.overlay).
  trailing_stop  fração de recuo a partir do pico para proteger lucro. Só arma
                 depois que a posição fica no azul; ao disparar, zera aquela
                 ação e só volta a operá-la num sinal novo (target passa por 0).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..costs.b3 import B3Costs
from .metrics import summarize


@dataclass
class PortfolioResult:
    equity: pd.Series
    trades_por_ativo: dict
    stats: dict


def _common_index(frames: dict) -> pd.DatetimeIndex:
    idx = None
    for f in frames.values():
        idx = f.index if idx is None else idx.intersection(f.index)
    return idx.sort_values()


def run_portfolio(frames: dict, strategy, costs: B3Costs, *,
                  initial_equity: float = 100_000.0, lot_size: int = 100,
                  overlay_mult: pd.Series | None = None,
                  trailing_stop: float | None = None,
                  cash_rate: pd.Series | None = None,
                  name_weights: pd.DataFrame | None = None) -> PortfolioResult:
    syms = list(frames)
    idx = _common_index(frames)
    n = len(syms)
    rate = (cash_rate.reindex(idx).ffill().fillna(0.0).to_numpy(dtype="float64")
            if cash_rate is not None else np.zeros(len(idx)))
    nw = {s: (name_weights[s].reindex(idx).ffill().fillna(1.0 / n).to_numpy(dtype="float64")
              if name_weights is not None else np.full(len(idx), 1.0 / n))
          for s in syms}

    open_ = {s: frames[s]["open"].reindex(idx).to_numpy(dtype="float64") for s in syms}
    close = {s: frames[s]["close"].reindex(idx).to_numpy(dtype="float64") for s in syms}
    tgt = {}
    for s in syms:
        sig = strategy.generate_signals(frames[s].loc[idx]).reindex(idx).fillna(0.0)
        tgt[s] = sig.shift(1).fillna(0.0).to_numpy(dtype="float64")

    mult = (overlay_mult.reindex(idx).fillna(0.0).to_numpy(dtype="float64")
            if overlay_mult is not None else np.ones(len(idx)))

    cash = float(initial_equity)
    shares = {s: 0.0 for s in syms}
    entry = {s: 0.0 for s in syms}     # preço de entrada da posição atual
    peak = {s: 0.0 for s in syms}      # melhor preço a favor desde a entrada
    blocked = {s: False for s in syms}  # travado por trailing stop até sinal novo
    equity = np.empty(len(idx))
    exposure = np.empty(len(idx))
    trades = {s: 0 for s in syms}
    fees_total = 0.0

    for i in range(len(idx)):
        cash *= 1.0 + rate[i]                        # juros do caixa (CDI)
        mtm = cash + sum(shares[s] * open_[s][i] for s in syms)
        for s in syms:
            po = open_[s][i]
            raw_t = tgt[s][i]
            if raw_t == 0.0:
                blocked[s] = False                      # sinal zerou -> destrava
            eff_t = 0.0 if blocked[s] else raw_t

            desired = 0.0
            if eff_t != 0.0 and po > 0.0 and mult[i] > 0.0:
                raw = mtm * nw[s][i] * eff_t * mult[i] / po
                desired = np.trunc(raw / lot_size) * lot_size
            delta = desired - shares[s]
            if delta != 0.0:
                fee = costs.order_cost(po, delta)
                cash -= delta * po + fee
                fees_total += fee
                if shares[s] == 0.0 and desired != 0.0:
                    entry[s] = po
                    peak[s] = po
                shares[s] = desired
                trades[s] += 1
            if shares[s] == 0.0:
                entry[s] = peak[s] = 0.0

            # trailing stop: avalia no fechamento de i -> efeito na abertura de i+1
            pc = close[s][i]
            if trailing_stop and shares[s] > 0.0:
                peak[s] = max(peak[s], pc)
                if peak[s] > entry[s] and pc <= peak[s] * (1.0 - trailing_stop):
                    blocked[s] = True
            elif trailing_stop and shares[s] < 0.0:
                peak[s] = min(peak[s], pc) if peak[s] > 0 else pc
                if peak[s] < entry[s] and pc >= peak[s] * (1.0 + trailing_stop):
                    blocked[s] = True

        pos_val = sum(shares[s] * close[s][i] for s in syms)
        equity[i] = cash + pos_val
        exposure[i] = pos_val / equity[i] if equity[i] > 0 else 0.0

    eq = pd.Series(equity, index=idx, name="equity")
    fee_frame = (pd.DataFrame({"fee": [fees_total]}) if fees_total
                 else pd.DataFrame(columns=["price", "shares", "fee"]))
    stats = summarize(eq, fee_frame, initial_equity)
    stats["n_trades"] = int(sum(trades.values()))
    stats["exposure_media"] = round(float(np.nanmean(exposure)), 3)
    stats["trades_por_ativo"] = dict(trades)
    return PortfolioResult(eq, dict(trades), stats)
