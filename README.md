# trader — automação PETR4 (MT5 + Python)

Estratégia baseada em **regras primeiro**, ML depois. O backtest não depende do
MT5; o MT5 entra como fonte read-only e, na Fase 3, como execução (só em conta demo).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[mt5,dev]"     # MT5 só funciona no Windows
```

## Fluxo atual (Fase 0 → 1)

Rode sempre da **raiz do repo**.

```bash
python scripts/pull_daily.py        # histórico diário PETR4 (Yahoo) -> data/petr4_d1.parquet
python scripts/pull_intraday.py     # backfill intraday (Yahoo 15m, ~60 pregões)
python scripts/run_backtest.py      # backtest do baseline (SMA) + métricas
python scripts/validate.py          # validação OOS das estratégias diárias
python scripts/validate_intraday.py # validação OOS do breakout de abertura (15m)
python scripts/compare_horizons.py  # mesmas estratégias em D1/M15/M5, OOS, custo por horizonte
python scripts/deep_dive.py         # walk-forward 8 folds + Sharpe por fold + retorno por ano
python scripts/pull_basket.py       # histórico diário das 5 ações da cesta
python scripts/pull_universe.py     # ~24 ações para o cross-sectional / treino de ML
python scripts/research_strategies.py    # SuperTrend / Keltner / Vortex + vol-target Yang-Zhang
python scripts/stress_champion.py        # stress-test do campeão (SuperTrend + YZ vol-target)
python scripts/research_cross_sectional.py  # momentum cross-sectional (não funcionou)
python scripts/ml_baseline.py       # Fase C: LightGBM walk-forward vs campeão
pytest                              # 112 testes
```

**Horizonte = swing** (decidido): no intraday o custo come 30–82% do lucro bruto e nenhum
resultado sobrevive ao walk-forward.

**Campeão atual:** cesta de 5 (PETR4, VALE3, ITUB4, BBDC4, ABEV3) + **SuperTrend + vol-target
Yang-Zhang** → Sharpe OOS realista ~0,7–0,85, drawdown ~−12%, sem ano de desastre em 16 anos
de walk-forward. O overlay de vol-target com estimador Yang-Zhang (usa o OHLC inteiro)
melhora *todos* os sinais de tendência — é o maior lever isolado.

**Não ajudaram:** reversão à média (B3 segue, não reverte), ensembles (sinais trend
correlacionados), filtros de força de tendência (ER gate, ADX), momentum cross-sectional
(−71% de drawdown). O paper-trade ainda roda o SMA+trailing; migrar para o campeão exige
plugar o overlay no `live/planner.py`.

Yahoo intraday é raso (1m ~7d, 5m/15m/30m ~60d, 1h ~730d). Para 1m com profundidade,
`scripts/collect_intraday.py` (MT5) acumula ao vivo durante o pregão.

Com o MT5 aberto e logado:

```bash
python scripts/check_corretora.py            # diagnóstico read-only da conexão e do símbolo
python scripts/pull_mt5_history.py D1 M15     # histórico do MT5 -> data/petr4_mt5_<tf>.parquet (M15 ~5 anos)
python scripts/collect_intraday.py           # acumula M1 ao vivo (rodar durante o pregão)
python scripts/demo_round_trip.py            # teste de ordem — SÓ conta demo, RUN=False por padrão
python scripts/paper_trade.py                # executor swing da cesta na demo — DRY-RUN por padrão
python scripts/paper_trade.py --send         # ... envia as ordens (aborta se não for demo)
python scripts/paper_report.py               # paper real vs backtest da mesma config -> tracking error
```

`paper_trade.py` roda **uma vez por dia, após o fechamento**: reconcilia posição com a
corretora, calcula o alvo das 5 ações (SMA 10/50 + trailing-stop 10%), mostra os deltas e
(com `--send`) envia. Estado em `data/paper_state.json`, diário em `data/paper_journal.csv`.
Agende com o Agendador de Tarefas do Windows (~17h30 BRT, dias úteis).

Alertas no Telegram (opcional): preencha `TELEGRAM_BOT_TOKEN` e `TELEGRAM_CHAT_ID` no
arquivo `.env` (na raiz, fora do git) e teste com `python scripts/test_telegram.py`.
`paper_trade.py` carrega o `.env` automaticamente.

## Layout

| Pasta | Papel |
|---|---|
| `trader/core/` | `safety` (trava demo + AutoTrading), `mt5_session` |
| `trader/data/` | `schema` (contrato OHLCV), `yahoo`, `mt5_source` (read-only), `store`, `collector` |
| `trader/costs/` | `b3` — emolumentos, corretagem, ISS, slippage |
| `trader/strategy/` | `base` (interface) + `sma_crossover` (baseline), `mean_reversion`, `momentum`, `opening_range` (intraday); a mesma função roda em backtest e live |
| `trader/backtest/` | `engine` (sem look-ahead, executa no open seguinte), `metrics`, `split` (treino/teste, walk-forward), `validation` (grid-search no treino, medição OOS) |
| `scripts/` | pontos de entrada |
| `tests/` | custos + causalidade do sinal |

## Regras de ouro

- Código de pesquisa (`data`, `costs`, `strategy`, `backtest`) **nunca** importa execução.
- Qualquer código que envie ordem chama `core.safety.preflight_trading(mt5)` — aborta
  fora de conta demo ou com AutoTrading desligado.
- Sinal decidido no fechamento de `t` só executa na abertura de `t+1`.
- **Nenhuma estratégia é avaliada pelo desempenho no período em que foi ajustada.**
  Parâmetros escolhidos só no treino; o número que vale é o do teste (dados nunca
  vistos). `scripts/validate.py` mostra IS vs OOS lado a lado.
- Percentuais da B3 em `config/settings.yaml` são aproximados — **conferir** na tabela
  vigente; tributário, confirmar com contador.

## Pendências

- `server_utc_offset_hours` em `settings.yaml`: confirmar comparando a 1ª barra do dia
  com a abertura do pregão (10:00 BRT).
- Histórico intraday: XP não serve M1 de ação; construindo ao vivo via `collect_intraday`.
