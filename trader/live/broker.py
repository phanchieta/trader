"""Wrapper fino de ordens no MT5. Só ordem a mercado, só conta demo.

O construtor chama preflight_trading: aborta fora de conta demo ou com o botão
'Algo Trading' desligado.
"""
from __future__ import annotations

from ..core.safety import preflight_trading

_OK_RETCODES = (10009, 10008)      # DONE, PLACED


class MT5Broker:
    def __init__(self, mt5, *, magic: int = 770001, deviation: int = 30):
        preflight_trading(mt5)
        self.mt5 = mt5
        self.magic = magic
        self.deviation = deviation

    def positions(self) -> dict:
        """{symbol: nº de ações com sinal} das posições abertas."""
        out: dict = {}
        for p in self.mt5.positions_get() or []:
            signed = p.volume if p.type == self.mt5.POSITION_TYPE_BUY else -p.volume
            out[p.symbol] = out.get(p.symbol, 0.0) + signed
        return out

    def _filling(self, symbol: str):
        fm = self.mt5.symbol_info(symbol).filling_mode
        if fm & 1:
            return self.mt5.ORDER_FILLING_FOK
        if fm & 2:
            return self.mt5.ORDER_FILLING_IOC
        return self.mt5.ORDER_FILLING_RETURN

    def market(self, symbol: str, signed_volume: float):
        """Compra (+) ou vende (−) `signed_volume` ações a mercado."""
        if signed_volume == 0.0:
            return None
        mt5 = self.mt5
        mt5.symbol_select(symbol, True)
        tick = mt5.symbol_info_tick(symbol)
        is_buy = signed_volume > 0.0
        req = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol,
            "volume": abs(float(signed_volume)),
            "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
            "price": tick.ask if is_buy else tick.bid,
            "deviation": self.deviation, "magic": self.magic,
            "comment": "paper swing", "type_time": mt5.ORDER_TIME_DAY,
            "type_filling": self._filling(symbol),
        }
        return mt5.order_send(req)

    @staticmethod
    def ok(result) -> bool:
        return result is not None and getattr(result, "retcode", None) in _OK_RETCODES
