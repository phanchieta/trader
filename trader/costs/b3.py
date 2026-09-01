"""Custo de fricção da B3 para ação à vista.

Cobre: corretagem, emolumentos + liquidação, ISS, slippage.
NÃO cobre imposto de renda — IR é mensal sobre o resultado líquido e entra
como ajuste no relatório (backtest.metrics), não por trade.

Percentuais: CONFERIR na tabela vigente da B3. Tributário: confirmar com contador.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class B3Costs:
    emolumentos_pct: float = 0.000325     # swing ~0,0325% | daytrade ~0,005%
    corretagem_por_ordem: float = 0.0     # XP ação à vista = 0
    iss_sobre_corretagem: float = 0.0
    slippage_ticks: float = 1.0
    tick_size: float = 0.01

    def order_cost(self, price: float, shares: float) -> float:
        """Custo de UMA ordem (entrada ou saída), em R$. Independe do sinal de `shares`."""
        notional = price * abs(shares)
        emol = notional * self.emolumentos_pct
        corr = self.corretagem_por_ordem
        iss = corr * self.iss_sobre_corretagem
        slippage = self.slippage_ticks * self.tick_size * abs(shares)
        return emol + corr + iss + slippage

    @classmethod
    def from_config(cls, cfg: dict) -> "B3Costs":
        c = cfg["costs"]
        daytrade = c.get("modalidade") == "daytrade"
        return cls(
            emolumentos_pct=(c["emolumentos_pct_daytrade"] if daytrade
                             else c["emolumentos_pct_swing"]),
            corretagem_por_ordem=c["corretagem_por_ordem"],
            iss_sobre_corretagem=c["iss_sobre_corretagem"],
            slippage_ticks=c["slippage_ticks"],
            tick_size=c["tick_size"],
        )
