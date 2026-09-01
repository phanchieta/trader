"""Context manager para conexão MT5. Import tardio: só quem usa MT5 precisa do pacote."""
from __future__ import annotations

import contextlib


@contextlib.contextmanager
def mt5_session(path: str | None = None):
    """Abre o terminal, entrega o módulo `mt5` já inicializado, fecha no fim.

        with mt5_session() as mt5:
            ...
    """
    import MetaTrader5 as mt5

    ok = mt5.initialize(path) if path else mt5.initialize()
    if not ok:
        raise RuntimeError(f"mt5.initialize falhou: {mt5.last_error()}")
    try:
        yield mt5
    finally:
        mt5.shutdown()
