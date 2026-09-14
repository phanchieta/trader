"""Scalping / day trade na XP: custos reais e estratégias intraday testadas OOS.

1) Custo por operação completa (ida e volta) em WIN, WDO e PETR4 — B3 + corretagem
   XP (com e sem RLP) + slippage.
2) Custo contra o movimento típico em 1, 5, 15, 30 e 60 minutos, e a taxa de acerto
   necessária para empatar.
3) Três estratégias em barra M1 (5 anos do MT5), parâmetros escolhidos no TREINO
   (40% iniciais dos dias) e medidos no TESTE (60% finais):
   - momentum intraday (Gao, Han, Li & Zhou 2018): 1ª meia hora prevê a última;
   - rompimento da faixa de abertura (ORB);
   - "scalping" de 1 minuto: segue ou reverte um minuto extremo e sai após N minutos.
   Posição = 1× o patrimônio em notional, CDI no caixa, IR day trade 20%.

Pré-requisitos: pull_mt5_intraday.py, pull_cdi.py.
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml

from trader.backtest.metrics import vs_benchmark
from trader.data.store import load
from trader.intraday import (PETR4, WDO, WIN, BarIndex, cash_close_minute, intraday_momentum,
                             opening_range_breakout, run_intraday, scalp_m1)

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
DATA = Path(cfg["data"]["dir"])
CDI = pd.read_parquet(DATA / "cdi_d1.parquet")["rate"]
EQ0 = 1_000_000.0
MT5_FEE_MONTH = 299.90
XP_NO_RLP_CONTRACT, XP_NO_RLP_ORDER = 1.00, 4.90

petr = load(DATA / "petr4_mt5_m1.parquet")
CASH_CLOSE = cash_close_minute(petr.index)
MARKETS = {
    "WIN": (WIN, load(DATA / "win_d_mt5_m1.parquet"), load(DATA / "win_n_mt5_m1.parquet")),
    "WDO": (WDO, load(DATA / "wdo_d_mt5_m1.parquet"), load(DATA / "wdo_n_mt5_m1.parquet")),
    "PETR4": (PETR4, petr, petr),
}


def _pct(x, fmt="{:+.1%}"):
    return "—" if x is None or x != x else fmt.format(x)


def daily_last_close(bars: pd.DataFrame) -> pd.Series:
    bi = BarIndex(bars.index)
    days, _, ends = bi.day_bounds()
    return pd.Series(bars["close"].to_numpy()[ends - 1], index=days)


# ------------------------------------------------------------------ 1) custos
print("=" * 104)
print("1) CUSTO DE UMA OPERAÇÃO COMPLETA (entrada + saída) POR CONTRATO / LOTE — XP + B3")
print("=" * 104)
print(f"{'ativo':6} {'preço':>10} {'notional':>10} | {'B3':>6} {'slip 1 tick':>11} "
      f"{'total c/RLP':>11} {'em ticks':>8} {'em bps':>7} | {'total s/RLP':>11} {'bps':>6}")
for name, (inst, _, real) in MARKETS.items():
    px = float(real["close"].iloc[-1])
    unit = inst.lot                                    # 1 contrato ou 100 ações
    notional = px * inst.point_value * unit
    b3 = 2 * unit * (inst.b3_fee_unit_side + inst.b3_fee_pct_side * px * inst.point_value)
    slip = 2 * unit * inst.tick_value
    tot = b3 + slip
    no_rlp = tot + (2 * XP_NO_RLP_CONTRACT * unit if inst.lot == 1 else 2 * XP_NO_RLP_ORDER)
    print(f"{name:6} {px:>10,.2f} {notional:>10,.0f} | {b3:>6.2f} {slip:>11.2f} {tot:>11.2f} "
          f"{tot / (unit * inst.tick_value):>8.1f} {tot / notional * 1e4:>7.1f} | "
          f"{no_rlp:>11.2f} {no_rlp / notional * 1e4:>6.1f}")
print(f"  MT5 na XP: R${MT5_FEE_MONTH:.2f}/mês na tabela de plataformas = "
      f"{12 * MT5_FEE_MONTH / EQ0:.2%} a.a. em R$1 mi, {12 * MT5_FEE_MONTH / 1e5:.1%} a.a. em R$100 mil.")
print("  IR day trade: 20% do lucro líquido mensal (1% retido é antecipação).")

# ------------------------------------------------- 2) custo vs movimento típico
print("\n" + "=" * 104)
print("2) CUSTO x MOVIMENTO TÍPICO (últimos 12 meses, 10:05–16:50) — mediana do |movimento| em ticks")
print("=" * 104)
H = (1, 5, 15, 30, 60)
print(f"{'ativo':6} {'custo ida+volta':>15} | " + " ".join(f"{str(h) + ' min':>8}" for h in H)
      + " | acerto p/ empatar (alvo=stop=mediana 15 min)")
for name, (inst, bars, real) in MARKETS.items():
    cut = bars.index[-1] - pd.Timedelta(days=365)
    b = bars[bars.index >= cut]
    bi = BarIndex(b.index)
    c = b["close"].to_numpy()
    px = float(real["close"].iloc[-1])
    cost_ticks = (2 * (inst.b3_fee_unit_side + inst.b3_fee_pct_side * px * inst.point_value)
                  + 2 * inst.tick_value) / inst.tick_value
    med = []
    for h in H:
        j = np.arange(len(c) - h)
        ok = (bi.day[j] == bi.day[j + h]) & (bi.minute[j] >= 605) & (bi.minute[j + h] <= 1010)
        med.append(float(np.median(np.abs(c[j + h][ok] - c[j][ok]))) / inst.tick)
    t15 = med[2]
    need = (t15 + cost_ticks) / (2 * t15)
    print(f"{name:6} {cost_ticks:>11.1f} tk | " + " ".join(f"{m:>8.1f}" for m in med)
          + f" | {need:.0%}  (custo = {cost_ticks / t15:.0%} do movimento de 15 min)")

# ------------------------------------------------------------- 3) estratégias
days_all = {}
for name, (inst, bars, real) in MARKETS.items():
    d = np.array(sorted(set(np.unique(BarIndex(bars.index).day)) & set(CASH_CLOSE.index)))
    days_all[name] = d
cut_idx = int(len(days_all["WIN"]) * 0.40)
CUT_DAY = days_all["WIN"][cut_idx]


def evaluate(name, trades, days, **kw):
    inst, bars, real = MARKETS[name]
    npx = daily_last_close(real).shift(1)
    return run_intraday(trades, bars, inst, days=days, notional_px=npx, cdi=CDI,
                        equity0=EQ0, **kw)


GRIDS = {
    "momentum intraday": (intraday_momentum, [
        dict(overnight=True, confirm_r12=False), dict(overnight=False, confirm_r12=False),
        dict(overnight=True, confirm_r12=True)]),
    "rompimento abertura": (opening_range_breakout, [
        dict(anchor=a, minutes=m, stop=s) for a, m, s in product((540, 600), (5, 15, 30, 60), (True, False))]),
    "scalp 1 minuto": (scalp_m1, [
        dict(k=k, hold=h, follow=f) for k, h, f in product((2.5, 3.0, 4.0), (5, 15, 30), (True, False))]),
}
RUN = [("momentum intraday", "WIN"), ("momentum intraday", "WDO"),
       ("rompimento abertura", "WIN"), ("rompimento abertura", "WDO"),
       ("scalp 1 minuto", "WIN"), ("scalp 1 minuto", "WDO"), ("scalp 1 minuto", "PETR4")]

print("\n" + "=" * 104)
print(f"3) ESTRATÉGIAS — treino {days_all['WIN'][0]}..{CUT_DAY} | teste {CUT_DAY}..{days_all['WIN'][-1]}")
print("   parâmetro escolhido pelo Sharpe/CDI no treino (com custo); números abaixo = TESTE")
print("=" * 104)
print(f"{'estratégia':20} {'ativo':5} {'escolhido no treino':24} {'treino':>6} | {'tr/dia':>6} "
      f"{'acerto':>6} {'bruto':>6} {'custo':>6} | {'s/custo':>7} {'c/custo':>7} {'pós-IR':>7} "
      f"{'Sh/CDI':>6} {'maxDD':>6} | {'slip ½':>7}")
print(f"{'':57} {'':6} | {'':6} {'':6} {'ticks':>6} {'ticks':>6} | {'— alpha sobre o CDI, a.a. —':^23} "
      f"{'':6} {'':6} | {'pós-IR':>7}")
print("-" * 130)
summary = []
for strat, mkt in RUN:
    fn, grid = GRIDS[strat]
    inst, bars, _ = MARKETS[mkt]
    days = days_all[mkt]
    tr_days, te_days = days[days < CUT_DAY], days[days >= CUT_DAY]
    scored = []
    for p in grid:
        trades = fn(bars, CASH_CLOSE, **p)
        sh = evaluate(mkt, trades, tr_days).stats["sharpe_cdi"]
        scored.append((sh if sh is not None else -9.0, p, trades))
    sh_tr, p, trades = max(scored, key=lambda x: x[0])
    te = evaluate(mkt, trades, te_days).stats
    free = evaluate(mkt, trades, te_days, free=True).stats
    half = evaluate(mkt, trades, te_days, slippage_ticks=0.5).stats
    ptxt = ",".join(f"{k}={v}" for k, v in p.items()).replace("True", "S").replace("False", "N")
    print(f"{strat:20} {mkt:5} {ptxt[:24]:24} {sh_tr:>+6.2f} | {te['trades_dia']:>6.2f} "
          f"{_pct(te['acerto'], '{:.0%}'):>6} {te['bruto_ticks_trade'] or 0:>+6.2f} "
          f"{te['custo_ticks_trade'] or 0:>6.2f} | {_pct(free['alpha']):>7} {_pct(te['alpha']):>7} "
          f"{_pct(te['alpha_pos_ir']):>7} {te['sharpe_cdi'] or 0:>+6.2f} {_pct(te['max_dd'], '{:.0%}'):>6} | "
          f"{_pct(half['alpha_pos_ir']):>7}")
    best_hind = max((evaluate(mkt, t, te_days).stats["alpha_pos_ir"] or -9, pp) for _, pp, t in scored
                    ) if strat != "momentum intraday" else None
    summary.append((strat, mkt, best_hind))

print("\nReferência que NÃO vale como resultado (melhor combinação escolhida olhando o teste):")
for strat, mkt, best in summary:
    if best is not None:
        print(f"  {strat:20} {mkt:5} melhor alpha pós-IR no teste = {_pct(best[0])}  ({best[1]})")

# --------------------------------------------- 4) walk-forward rolante (como ao vivo)
TRAIN_D, TEST_D = 252, 126
print("\n" + "=" * 104)
print(f"4) WALK-FORWARD ROLANTE — a cada {TEST_D} pregões re-escolhe o parâmetro com os {TRAIN_D} "
      "anteriores e opera os seguintes")
print("=" * 104)
print(f"{'estratégia':20} {'ativo':5} {'janela OOS':23} | {'s/custo':>7} {'c/custo':>7} {'pós-IR':>7} "
      f"{'Sh/CDI':>6} {'maxDD':>6} | {'trocas de parâmetro':>19}")


def _stitch(parts: list[pd.Series]) -> pd.Series:
    rets = pd.concat([p.pct_change().fillna(p.iloc[0] / EQ0 - 1.0) for p in parts])
    return EQ0 * (1.0 + rets).cumprod()


for strat, mkt in RUN:
    fn, grid = GRIDS[strat]
    inst, bars, _ = MARKETS[mkt]
    days = days_all[mkt]
    cached = [(p, fn(bars, CASH_CLOSE, **p)) for p in grid]
    parts = {"net": [], "free": [], "ir": []}
    picks = []
    for s in range(TRAIN_D, len(days) - 20, TEST_D):
        tr_d, te_d = days[s - TRAIN_D:s], days[s:s + TEST_D]
        best = max(cached, key=lambda pt: evaluate(mkt, pt[1], tr_d).stats["sharpe_cdi"] or -9.0)
        picks.append(str(best[0]))
        net = evaluate(mkt, best[1], te_d)
        parts["net"].append(net.equity)
        parts["ir"].append(net.equity_after_tax)
        parts["free"].append(evaluate(mkt, best[1], te_d, free=True).equity)
    eq = {k: _stitch(v) for k, v in parts.items()}
    b_net, b_free, b_ir = (vs_benchmark(eq[k], CDI) for k in ("net", "free", "ir"))
    dd = float((eq["net"] / eq["net"].cummax() - 1.0).min())
    changes = sum(1 for a, b in zip(picks, picks[1:]) if a != b)
    span = f"{eq['net'].index[0].date()}..{eq['net'].index[-1].date()}"
    print(f"{strat:20} {mkt:5} {span:23} | {_pct(b_free['alpha_aa']):>7} {_pct(b_net['alpha_aa']):>7} "
          f"{_pct(b_ir['alpha_aa']):>7} {b_net['sharpe_sobre_cdi'] or 0:>+6.2f} {dd:>6.0%} | "
          f"{changes:>3} em {len(picks)} janelas")
