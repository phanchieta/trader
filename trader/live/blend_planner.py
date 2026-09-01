"""Planejador do blend campeão + fatores para execução ao vivo.

  - Sleeve campeão (5 ações): SuperTrend long-only, re-avaliada TODO dia,
    dimensionada por vol-target Yang-Zhang da cesta.
  - Sleeve fatores (~universo B3): top quintil do score combinado de 5 fatores,
    SELEÇÃO trimestral (trimestre-calendário), vol-target Yang-Zhang do Ibov
    aplicado TODO dia. Entre rebalances o quintil fica congelado no estado.

Combina os pesos das duas sleeves num mapa único por ação e converte em nº de
ações alvo pelo último fechamento.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

from ..factors.library import combined_score
from ..indicators import yang_zhang
from ..strategy import SuperTrend

FACTORS5 = ["momentum", "low_vol", "low_beta", "lt_reversal", "st_reversal"]


def _quarter(d: date) -> tuple[int, int]:
    return (d.year, (d.month - 1) // 3)


def needs_factor_rebalance(today: date, last_iso: str | None) -> bool:
    if not last_iso:
        return True
    return _quarter(today) != _quarter(datetime.fromisoformat(last_iso).date())


def _yz_mult(frames: dict, *, lookback: int, target_vol: float, max_lev: float) -> float:
    v = float(pd.DataFrame({s: yang_zhang(frames[s], window=lookback)
                            for s in frames}).mean(axis=1).iloc[-1])
    return min(max_lev, target_vol / v) if v > 0 else 0.0


@dataclass
class BlendPlan:
    target_shares: dict
    weights: dict
    quintile: list
    rebalanced: bool
    champ_mult: float = 0.0
    factor_mult: float = 0.0
    notes: list = field(default_factory=list)


def plan_blend(champ_frames: dict, factor_frames: dict, ibov: pd.DataFrame, *,
               equity: float, today: date, prev_quintile: list,
               last_rebalance: str | None, champ_w: float = 0.5, n_frac: float = 0.20,
               lot: int = 100, target_vol: float = 0.15, vol_lb: int = 20,
               max_lev: float = 1.0, st_period: int = 10, st_mult: float = 3.0,
               min_names: int = 20) -> BlendPlan:
    notes: list[str] = []
    weights: dict[str, float] = {}

    # --- sleeve campeão (diário) ---
    cm = _yz_mult(champ_frames, lookback=vol_lb, target_vol=target_vol, max_lev=max_lev)
    for s, f in champ_frames.items():
        sig = float(SuperTrend(st_period, st_mult).generate_signals(f).iloc[-1])
        weights[s] = champ_w * (1.0 / len(champ_frames)) * sig * cm

    # --- sleeve fatores (seleção trimestral) ---
    rebal = needs_factor_rebalance(today, last_rebalance)
    if rebal or not prev_quintile:
        # ffill: parquets do universo têm datas de fim ligeiramente diferentes;
        # o score é lento, carregar os últimos dias é seguro.
        score = combined_score(factor_frames, FACTORS5, mkt=ibov["close"]).ffill(limit=10)
        row = score.iloc[-1].dropna()
        if len(row) < min_names:
            raise RuntimeError(f"só {len(row)} nomes com score — universo desatualizado?")
        k = max(1, round(len(row) * n_frac))
        quintile = sorted(row.nlargest(k).index)
        notes.append(f"rebalance de fatores ({_quarter(today)}): {len(quintile)} de {len(row)} nomes")
    else:
        quintile = [q for q in prev_quintile if q in factor_frames]

    fm = _yz_mult({"IBOV": ibov}, lookback=vol_lb, target_vol=target_vol, max_lev=max_lev)
    if quintile:
        for s in quintile:
            weights[s] = weights.get(s, 0.0) + (1.0 - champ_w) * (1.0 / len(quintile)) * fm

    # --- pesos -> nº de ações ---
    targets: dict[str, float] = {}
    for s, w in weights.items():
        src = champ_frames if s in champ_frames else factor_frames
        px = float(src[s]["close"].iloc[-1])
        targets[s] = float(np.trunc(w * equity / px / lot) * lot) if (px > 0 and w > 0) else 0.0

    return BlendPlan(targets, weights, quintile, rebal, round(cm, 3), round(fm, 3), notes)
