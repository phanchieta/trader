"""Alocação tática entre classes de ativo (ETFs da B3) com CDI como porto seguro."""
from .engine import AllocResult, run_allocation
from .series import splice
from .signals import dual_momentum, faber_sma, month_ends, static_weights

__all__ = ["AllocResult", "dual_momentum", "faber_sma", "month_ends", "run_allocation", "splice",
           "static_weights"]
