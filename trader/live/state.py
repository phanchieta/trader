"""Estado persistido do robô: posição, preço de entrada, pico e trava do
trailing por ação. JSON em disco, reconciliado com a corretora no boot.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class SymbolState:
    shares: float = 0.0
    entry: float = 0.0       # preço de entrada da posição atual
    peak: float = 0.0        # melhor preço a favor desde a entrada
    blocked: bool = False    # travado por trailing-stop até um sinal novo


@dataclass
class PortfolioState:
    symbols: dict = field(default_factory=dict)
    updated: str = ""
    factor: dict = field(default_factory=dict)   # {"quintile": [...], "last_rebalance": "iso"}

    def get(self, sym: str) -> SymbolState:
        return self.symbols.setdefault(sym, SymbolState())


def load(path) -> PortfolioState:
    p = Path(path)
    if not p.exists():
        return PortfolioState()
    raw = json.loads(p.read_text(encoding="utf-8"))
    st = PortfolioState(updated=raw.get("updated", ""), factor=raw.get("factor", {}))
    for sym, d in raw.get("symbols", {}).items():
        st.symbols[sym] = SymbolState(**d)
    return st


def save(state: PortfolioState, path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated": state.updated, "factor": state.factor,
               "symbols": {s: asdict(v) for s, v in state.symbols.items()}}
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def reconcile(state: PortfolioState, broker_shares: dict, *, tol: float = 1.0):
    """Corretora é a fonte da verdade. Ajusta `shares` para o que ela reporta
    e devolve a lista de divergências para alerta."""
    disc = []
    for s in set(state.symbols) | set(broker_shares):
        want = state.get(s).shares
        have = float(broker_shares.get(s, 0.0))
        if abs(want - have) > tol:
            disc.append(f"{s}: estado={want:.0f} corretora={have:.0f} -> ajustado")
            ss = state.get(s)
            ss.shares = have
            if have == 0.0:
                ss.entry = ss.peak = 0.0
                ss.blocked = False
    return state, disc
