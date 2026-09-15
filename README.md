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
`pull_futures_proxy` `pull_cdi` `pull_mt5_history` `pull_intraday` `pull_mt5_intraday`
(WIN/WDO/PETR4 em M5 e M1, 5 anos, do MT5 da XP) `pull_share_classes` (ON/PN/units)
`pull_etfs` (BOVA11 — abertura real; a do ^BVSP no Yahoo é falsa) `pull_cotahist` +
`build_options_db` (séries históricas da B3 2008–2026: todos os negócios, opções com
bid/ask, inclusive ações deslistadas)

**Pesquisa** — `research_strategies` (trend family + vol-target) ·
`research_cross_sectional` (momentum cross-sectional — falhou) · `research_futures`
(WIN/WDO — não ajudou) · `factor_model` (fatores B3 no universo grande) ·
`ml_baseline` / `ml_experiment` (LightGBM — AUC 0,54, não bate a regra) ·
`research_grid_averaging` (averaging down "nunca realiza perda" — bate o CDI mas
perde pro buy&hold e tem cauda feia) · `research_markov_regime` (regime de Markov
do vídeo + HMM — persistência é artefato da janela; dominado pelo campeão) ·
`research_intraday` (scalping/day trade em WIN, WDO e PETR4 com custos reais da XP
— nada bate o CDI depois de custo e IR) · `research_share_classes` (arbitragem ON x PN
e unit x PN — funcionou até 2014, sumiu depois) · `research_calendar` (virada de mês e
overnight x pregão — o retorno de ação na B3 sai quase todo à noite, mas ida e volta
diária não paga o custo) · `research_exec_timing` (campeão executando no leilão de
fechamento em vez da abertura — +0,1 a +0,2% a.a., ruído) · `research_options`
(venda coberta e put coberta por caixa em PETR4, VALE3 e BOVA11 com séries reais da B3
— sem prêmio de volatilidade na média) ·
`compare_horizons` `deep_dive` `validate` `validate_intraday`

**Campeão** — `stress_champion` (robustez) · `champion_vs_cdi` (alpha sobre o CDI
por nível de risco) · `optimize_champion` (otimização do sizing)

**Ao vivo** — `check_corretora` · `demo_round_trip` · `paper_trade` (campeão;
dry-run, `--send` envia) · `paper_blend` (blend campeão+fatores; estado próprio
em `data/blend_state.json`) · `paper_blend_sim` (blend simulado, sem MT5, roda
junto do campeão real) · `compare_paper` (campeão real vs blend sim vs CDI) ·
`paper_report` · `collect_intraday`

O `paper_blend`/`paper_blend_sim` leem os parquets do universo de fatores —
mantenha `pull_factor_universe.py` + `pull_futures_proxy.py` rodando (cron diário).
Sleeve campeão diária; sleeve de fatores rebalanceia por trimestre-calendário.
**Hoje:** campeão envia de verdade na demo; blend só no simulador (os dois não
podem enviar na mesma conta). Ver "Rodar os dois em paralelo" no RESEARCH_LOG.

```bash
pytest        # 128 testes
```

**Não ajudaram:** reversão à média (B3 segue, não reverte), ensembles (sinais trend
correlacionados), filtros de força de tendência (ER gate, ADX), momentum cross-sectional
(−71% de drawdown), averaging down "nunca realiza perda" (98% de acerto mas perde pro
buy&hold e um nome que não recupera crava rombo permanente — §16 do RESEARCH_LOG),
regime de Markov "hedge fund method" (a persistência da matriz aparece igual com
retornos embaralhados; é momentum de 20d com banda — §17), scalping/day trade
(momentum intraday, rompimento de abertura e scalp de 1 min em WIN/WDO/PETR4 —
negativos no walk-forward rolante depois de custo e IR — §18), arbitragem ON x PN
(4 de 48 combinações positivas no teste 2014–2026; poucos pares líquidos — §19),
virada de mês (forte até 2012, sumiu) e compra no fechamento/venda na abertura
(prêmio noturno real, mas 2 bps de atrito por ponta já zeram — §20), executar o
campeão no leilão de fechamento (ganha uma noite por entrada ≈ 0,1% a.a. — §21),
vender opções (vol implícita ≈ realizada na B3; venda coberta perde para a ação — §22).
O paper-trade ainda roda o SMA+trailing; migrar para o campeão exige plugar o overlay
no `live/planner.py`.

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
| `trader/regime/` | `markov` (cadeia visível do vídeo: rótulo 20d, matriz expansível, sinal), `hmm` (HMM gaussiano / filtro de Hamilton, só numpy) |
| `trader/intraday/` | `costs` (custo por ponta XP + B3), `sessions` (pregão em BRT), `strategies` (momentum intraday, ORB, scalp M1 → lista de trades), `engine` (resultado diário, CDI, IR day trade) |
| `trader/pairs/` | `spread` (log spread, z-score, elegibilidade, meia-vida, ADF), `signals` (estado do par), `portfolio` (slots, execução no open seguinte, aluguel) |
| `trader/seasonal/` | `calendar` (pregão do mês, janela da virada), `timing` (estar ou não posicionado, execução no leilão de fechamento ou na abertura) |
| `trader/options/` | `bs` (Black-Scholes, vol implícita), `chain` (vencimentos mensais, strike com livro são), `writer` (venda coberta, put coberta por caixa, trava) |
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
