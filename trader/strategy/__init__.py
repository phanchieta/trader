"""Estratégias. Cada uma é uma função pura: OHLCV -> posição-alvo por barra.

Baseline: sma_crossover. Regra: mean_reversion, momentum, gap_reversal,
donchian, opening_range_breakout (intraday). Combinadores: Ensemble (voto),
WeightedEnsemble (voto ponderado por qualidade). Wrapper: VolumeConfirmed.
"""
from .donchian import Donchian
from .donchian_fibo import DonchianFibo
from .ensemble import Ensemble, WeightedEnsemble
from .gap_reversal import GapReversal
from .keltner import Keltner
from .mean_reversion import MeanReversion
from .momentum import Momentum
from .opening_range import OpeningRangeBreakout
from .quantile_channel import QuantileChannel
from .regime_gated import RegimeGated
from .return_dist_fade import ReturnDistFade
from .sma_crossover import SmaCrossover
from .supertrend import SuperTrend
from .trend_scan import TrendScan
from .volume_confirm import VolumeConfirmed
from .vortex import Vortex

__all__ = ["Donchian", "DonchianFibo", "Ensemble", "GapReversal", "Keltner",
           "MeanReversion", "Momentum", "OpeningRangeBreakout", "QuantileChannel",
           "RegimeGated", "ReturnDistFade", "SmaCrossover", "SuperTrend",
           "TrendScan", "VolumeConfirmed", "Vortex", "WeightedEnsemble"]
