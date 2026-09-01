"""Aproximação de imposto de renda sobre a curva de equity (pessoa física).

Modelo simples: IR MENSAL sobre o ganho líquido, com prejuízo compensável
(carry forward), sem a isenção de R$20 mil em vendas/mês. Alíquota `rate` é uma
mistura de swing trade em ação (15%) com renda fixa/CDI (~17,5%) — o default
0,155 assume que a maior parte do resultado vem da parte operada.

Não separa a parte de CDI da parte operada (dado a granularidade da curva); é
uma ESTIMATIVA conservadora do haircut, não cálculo fiscal.
"""
from __future__ import annotations

import pandas as pd


def after_tax_equity(equity: pd.Series, *, rate: float = 0.155) -> pd.Series:
    m = equity.resample("ME").last().dropna()
    if len(m) < 2:
        return equity.copy()
    out = [float(m.iloc[0])]
    carry = 0.0                                   # prejuízo acumulado (<= 0)
    for i in range(1, len(m)):
        g = float(m.iloc[i] - m.iloc[i - 1])      # ganho bruto do mês (em R$ do book bruto)
        net = g + carry
        if net > 0:
            tax = net * rate
            carry = 0.0
        else:
            tax = 0.0
            carry = net
        out.append(out[-1] + g - tax)
    return pd.Series(out, index=m.index, name="equity_pos_ir")


def cagr(series: pd.Series) -> float:
    yrs = (series.index[-1] - series.index[0]) / pd.Timedelta(days=365.25)
    f = float(series.iloc[-1] / series.iloc[0])
    return f ** (1.0 / yrs) - 1.0 if (f > 0 and yrs > 0) else float("nan")


def after_tax_summary(equity: pd.Series, cdi_daily: pd.Series, *,
                      rate: float = 0.155) -> dict:
    at = after_tax_equity(equity, rate=rate)
    cdi_curve = (1.0 + cdi_daily.reindex(equity.index).ffill().fillna(0.0)).cumprod()
    c_cdi = cagr(cdi_curve)
    c_pre, c_pos = cagr(equity), cagr(at)
    return {
        "cagr_pre_ir": round(c_pre, 4),
        "cagr_pos_ir": round(c_pos, 4),
        "cagr_cdi": round(c_cdi, 4),
        "alpha_pre_ir": round(c_pre - c_cdi, 4),
        "alpha_pos_ir": round(c_pos - c_cdi, 4),
    }
