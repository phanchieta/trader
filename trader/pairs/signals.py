"""Máquina de estados de um par a partir do z-score do spread (decidido no fechamento).

+1 = comprado em A / vendido em B (spread barato, z < −entry)
−1 = vendido em A / comprado em B (spread caro, z > +entry)
Sai quando |z| volta para dentro de `exit_z`, quando diverge além de `stop_z` ou
depois de `max_hold` pregões. Depois de stop ou prazo, só reentra do mesmo lado
quando o z voltar para dentro da banda de entrada.
"""
from __future__ import annotations

import numpy as np


def pair_positions(z: np.ndarray, eligible: np.ndarray, *, entry: float = 2.0,
                   exit_z: float = 0.0, stop_z: float | None = 4.0,
                   max_hold: int | None = 60) -> np.ndarray:
    z = np.asarray(z, dtype="float64")
    ok = np.asarray(eligible, dtype=bool) & np.isfinite(z)
    pos = np.zeros(len(z))
    cur, held, blocked = 0, 0, 0
    for t in range(len(z)):
        zt = z[t]
        if not ok[t]:
            cur, held, blocked = 0, 0, 0
            continue
        if blocked and abs(zt) < entry:
            blocked = 0
        if cur == 1:
            held += 1
            if zt >= -exit_z:
                cur = 0
            elif (stop_z is not None and zt < -stop_z) or (max_hold and held >= max_hold):
                cur, blocked = 0, 1
        elif cur == -1:
            held += 1
            if zt <= exit_z:
                cur = 0
            elif (stop_z is not None and zt > stop_z) or (max_hold and held >= max_hold):
                cur, blocked = 0, -1
        if cur == 0:
            held = 0
            if zt < -entry and blocked != 1 and (stop_z is None or zt >= -stop_z):
                cur = 1
            elif zt > entry and blocked != -1 and (stop_z is None or zt <= stop_z):
                cur = -1
        pos[t] = cur
    return pos
