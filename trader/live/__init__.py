"""Execução ao vivo (conta DEMO). broker + planner + state.

Todo caminho que envia ordem passa por core.safety.preflight_trading.
Código de pesquisa (data, costs, strategy, backtest) NUNCA importa este pacote.
"""
