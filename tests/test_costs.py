from trader.costs.b3 import B3Costs


def test_order_cost_componentes():
    c = B3Costs(emolumentos_pct=0.000325, corretagem_por_ordem=0.0,
               slippage_ticks=1.0, tick_size=0.01)
    # 100 ações a R$40 -> notional 4000; emol = 1.30 ; slippage = 0.01*100 = 1.00
    assert round(c.order_cost(price=40.0, shares=100), 4) == round(1.30 + 1.00, 4)


def test_order_cost_independe_do_sinal():
    c = B3Costs()
    assert c.order_cost(40.0, 100) == c.order_cost(40.0, -100)


def test_from_config_daytrade_usa_emol_menor():
    cfg = {"costs": {"modalidade": "daytrade", "corretagem_por_ordem": 0.0,
                     "emolumentos_pct_swing": 0.000325,
                     "emolumentos_pct_daytrade": 0.00005,
                     "iss_sobre_corretagem": 0.0, "slippage_ticks": 1.0,
                     "tick_size": 0.01}}
    assert B3Costs.from_config(cfg).emolumentos_pct == 0.00005


def test_corretagem_e_iss():
    c = B3Costs(emolumentos_pct=0.0, corretagem_por_ordem=5.0,
                iss_sobre_corretagem=0.05, slippage_ticks=0.0)
    assert c.order_cost(40.0, 100) == 5.0 + 5.0 * 0.05
