# trader — trade sistemático B3 (MT5 + Python)

Programa de trend swing em ações da B3. Ver **`RESEARCH_LOG.md`** para o histórico
completo da pesquisa (o que foi testado e os vereditos).

## Setup

```bash
conda activate trade                       # env com pandas/MetaTrader5/lightgbm/...
pip install -e ".[mt5,ml,dev]"
```

## CAMPEÃO

Cesta de 5 (PETR4, VALE3, ITUB4, BBDC4, ABEV3) + **SuperTrend(10, 3×) long-only +
vol-target Yang-Zhang (alvo 0,15) + equal-weight, sem trailing**.

Walk-forward OOS 2015–2026, CDI creditado no caixa: **CDI + ~4% a.a., max drawdown
~−5%, Sharpe sobre o CDI ~0,6, ~25–33% investido** (resto em caixa rendendo CDI).
Pré-imposto. Rodando no paper-trade da conta demo.

## Scripts (rodar da raiz do repo)

**Dados** — `pull_daily` `pull_basket` `pull_universe` `pull_factor_universe`
`pull_futures_proxy` `pull_cdi` `pull_mt5_history` `pull_intraday`

**Pesquisa** — `research_strategies` (trend family + vol-target) ·
`research_cross_sectional` (momentum cross-sectional — falhou) · `research_futures`
(WIN/WDO — não ajudou) · `factor_model` (fatores B3 no universo grande) ·
`ml_baseline` / `ml_experiment` (LightGBM — AUC 0,54, não bate a regra) ·
`compare_horizons` `deep_dive` `validate` `validate_intraday`

**Campeão** — `stress_champion` (robustez) · `champion_vs_cdi` (alpha sobre o CDI
por nível de risco) · `optimize_champion` (otimização do sizing)

**Ao vivo** — `check_corretora` · `demo_round_trip` · `paper_trade` (campeão;
dry-run, `--send` envia) · `paper_blend` (blend campeão+fatores; estado próprio
em `data/blend_state.json`) · `paper_report` · `collect_intraday`

O `paper_blend` lê os parquets do universo de fatores — mantenha
`pull_factor_universe.py` + `pull_futures_proxy.py` rodando (cron diário). Sleeve
campeão diária; sleeve de fatores rebalanceia por trimestre-calendário.

```bash
pytest        # 128 testes
```

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
