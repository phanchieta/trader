"""Custo de day trade por ponta na XP + B3 (levantado em 2026-09 — conferir na tabela
vigente antes de operar).

B3 day trade: WIN ≈ R$0,25 e WDO ≈ R$1,20 por contrato por ponta (WDO varia com o
dólar); ação 0,023% do financeiro por ponta (faixa até R$1 mi/dia).
XP corretagem: zero em minicontratos e day trade de ação COM RLP ativado e ordem
enviada pelo próprio cliente em plataforma eletrônica; sem RLP, R$1,00 por
minicontrato e R$4,90 por ordem de ação.
Slippage em ticks por ponta: ordem a mercado cruza o spread + latência do robô.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    name: str
    point_value: float                 # R$ por ponto por unidade (ação: 1)
    tick: float                        # tick em pontos
    lot: int = 1                       # unidade mínima (ação: 100)
    b3_fee_unit_side: float = 0.0      # R$ por contrato por ponta
    b3_fee_pct_side: float = 0.0       # fração do financeiro por ponta
    brokerage_unit_side: float = 0.0   # R$ por contrato por ponta (sem RLP: 1,00)
    brokerage_order_side: float = 0.0  # R$ por ordem (ação sem RLP: 4,90)

    @property
    def tick_value(self) -> float:
        return self.tick * self.point_value

    def unit_cost_side(self, price: float, slippage_ticks: float) -> float:
        """R$ por unidade (contrato ou ação) numa ponta, sem a corretagem por ordem."""
        return (self.b3_fee_unit_side + self.brokerage_unit_side
                + self.b3_fee_pct_side * price * self.point_value
                + slippage_ticks * self.tick_value)


WIN = Instrument("WIN", point_value=0.20, tick=5.0, b3_fee_unit_side=0.25)
WDO = Instrument("WDO", point_value=10.0, tick=0.5, b3_fee_unit_side=1.20)
PETR4 = Instrument("PETR4", point_value=1.0, tick=0.01, lot=100, b3_fee_pct_side=0.00023)
