"""Travas de segurança. Importado por QUALQUER código que possa enviar ordem.

O módulo recebe `mt5` como argumento (nunca importa MetaTrader5 direto), então
pode ser testado com um fake e não obriga o pacote MT5 em ambiente de pesquisa.
"""
from __future__ import annotations

_MODO = {0: "DEMO", 1: "CONTEST", 2: "REAL"}


def assert_demo(mt5) -> None:
    """Aborta se a conta conectada não for demo."""
    info = mt5.account_info()
    if info is None:
        raise RuntimeError("sem conexão MT5 (account_info() retornou None)")
    if info.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO:
        modo = _MODO.get(info.trade_mode, "?")
        raise RuntimeError(
            f"Conta {info.login} @ {info.server} é {modo}, não DEMO. "
            "Execução bloqueada."
        )


def assert_algo_enabled(mt5) -> None:
    """Aborta se o botão 'Algo Trading' do terminal estiver desligado.

    Foi o que barrou o primeiro teste: retcode 10027 'AutoTrading disabled by
    client' é ESTE botão (Ctrl+E), não bloqueio da corretora.
    """
    term = mt5.terminal_info()
    if term is None or not term.trade_allowed:
        raise RuntimeError(
            "AutoTrading desligado no terminal. Ligue o botão 'Algo Trading' "
            "(Ctrl+E) e marque Ferramentas > Opções > Expert Advisors."
        )


def preflight_trading(mt5) -> None:
    """Chamar no início de todo fluxo que pode enviar ordem."""
    assert_demo(mt5)
    assert_algo_enabled(mt5)
