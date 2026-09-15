"""Opções listadas da B3: cadeia a partir do COTAHIST, Black-Scholes e venda sistemática."""
from .bs import bs_price, implied_vol
from .chain import is_weekly, monthly_expiries, pick_series
from .writer import WriterResult, run_option_writer

__all__ = ["WriterResult", "bs_price", "implied_vol", "is_weekly", "monthly_expiries",
           "pick_series", "run_option_writer"]
