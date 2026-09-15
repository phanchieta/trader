"""Arbitragem entre classes de ação da mesma empresa (ON x PN, unit x PN)."""
from .portfolio import PairsResult, build_weights, pairs_portfolio
from .signals import pair_positions
from .spread import adf_tstat, eligibility, half_life, log_spread, rolling_z

__all__ = ["PairsResult", "adf_tstat", "build_weights", "eligibility", "half_life",
           "log_spread", "pair_positions", "pairs_portfolio", "rolling_z"]
