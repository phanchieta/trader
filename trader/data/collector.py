"""Acumulador de barras intraday ao vivo.

XP não serve histórico de M1 para ação, então construímos o histórico
intraday coletando ao vivo durante o pregão a partir de agora.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .mt5_source import bars
from .store import append_dedup


def collect_once(mt5, symbol: str, timeframe: str, path: str,
                 lookback_days: int = 3, server_utc_offset_hours: float = -3.0) -> int:
    """Puxa as barras recentes e faz merge no Parquet. Retorna o total acumulado."""
    start = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    df = bars(mt5, symbol, timeframe, start,
              server_utc_offset_hours=server_utc_offset_hours)
    return len(append_dedup(df, path))
