# Log de pesquisa — trader automatizado B3

Registro do que foi testado, o que funcionou e o que não funcionou. Ordem
cronológica aproximada. Atualizado em 2026-09-01.

---

## Objetivo e setup

Robô de trade **sistemático** para ações da B3, executado pela conexão MetaTrader 5
da corretora (XP). Python. Ambiente: conda env `trade` (`~/anaconda3/envs/trade`,
Python 3.14; `pandas`, `numpy`, `pyarrow`, `yfinance`, `MetaTrader5`, `lightgbm`,
`scikit-learn`, `requests`).

Rodar scripts sempre da raiz do repo. Ver `README.md` para o fluxo de comandos.

## Decisões travadas

| decisão | escolha | por quê |
|---|---|---|
| Linguagem | Python | dados/ML/backtest; não MQL5 |
| Abordagem | regras primeiro, ML depois | ML só se bater o baseline OOS |
| Instrumento | PETR4 à vista → cesta de 5 | XP roteia ação à vista no MT5 (grupo `BOVESPA\A VISTA`); `trade_mode FULL`, netting, lote 100, tick 0,01 |
| Horizonte | **swing** (dias) | intraday consome 30–82% do lucro bruto em custo e não sobrevive ao walk-forward |
| Benchmark | **CDI** | no Brasil, tem que superar a renda fixa senão não vale o risco |

## Dados

- **Diário longo**: Yahoo Finance (`<TICKER>.SA`), desde ~2000 para a maioria.
  `yahoo.daily()` remove spike de 1 barra (glitch tipo PETR4 2003-08-05) e linhas
  com NaN.
- **MT5 (XP)**: D1 e M15 ~5 anos (desde 2021); M5 ~4,6 anos; M1 ~11 meses.
  `copy_rates_from_pos` dá `-2 Invalid params` para `count` grande demais — usar
  `copy_rates_range` com datas, em janelas (`bars_range_chunked`). Fuso do servidor
  = Brasília (UTC−3), confirmado.
- **CDI**: BACEN SGS série 12, via `requests` (urllib é bloqueado por WAF), em
  janelas de ≤10 anos (SGS rejeita períodos maiores). `data/cdi_d1.parquet`.
  CDI médio 2015–2026 ≈ 9,8% a.a.

## Infra de validação (a régua)

- **Walk-forward** com folds; parâmetros escolhidos SÓ no treino; o número que
  vale é o do teste. Ver `backtest/split.py`, `backtest/validation.py`.
- **Nunca K-fold aleatório** em série temporal. Purga + embargo quando há rótulo
  que olha para frente (ML).
- Sem look-ahead: sinal decidido no fechamento de `t` executa na abertura de `t+1`.
- Custo B3 sempre ligado (emolumentos, corretagem, ISS, slippage). IR (15% swing /
  20% day trade) fora do backtest — entra como ajuste; **confirmar com contador**.
- Métrica-chave: **Sharpe sobre o CDI** e **alpha a.a. sobre o CDI** (`metrics.vs_benchmark`).

---

## O que foi testado

### 1. PETR4 sozinho — SMA crossover (baseline)
Walk-forward 2000–2026: Sharpe OOS 0,43, **max drawdown −43%**. Baseline fraco de
propósito — régua para o resto.

### 2. Diversificação: cesta de 5 (PETR4, VALE3, ITUB4, BBDC4, ABEV3)
**O maior ganho isolado.** SMA na cesta: Sharpe 0,43 → **0,65**, drawdown
−43% → **−22%**. Só diversificar cortou o tombo pela metade.

### 3. Trailing-stop
Varredura 8–20%: **todos batem "sem stop"** no SMA (Sharpe 0,65 → 0,68–0,79), mas
a curva é irregular (perseguir o pico = ajustar a ruído). Travado em 10%.
**Trailing ajuda o SMA e o trend_scan; atrapalha Donchian, SuperTrend, momentum**
— trailing-stop briga com trend-following puro. No campeão final: **sem trailing**.

### 4. Família trend-following (todos na faixa Sharpe OOS 0,5–0,8)
SMA, Donchian (20/10), SuperTrend (10, 3×), Keltner, Vortex, TrendScan,
Donchian-Fibo + ATR stop, momentum. **SuperTrend no topo** (Sharpe ~0,78 na cesta
sem overlay; +219% de retorno; menor custo). Donchian é o mais "liso" (−14% de
drawdown). Donchian-Fibo (canal de Fibonacci + ATR stop, ideia do usuário): fraco
(0,24) — "comprar a queda dentro da tendência" perde para só cabalgar a tendência.

### 5. Reversão à média — **MORTA no swing da B3**
`MeanReversion` (z-score de preço): fraca. `ReturnDistFade` (fade de movimento
anômalo de 1 dia, artigo MQL5 19797): **Sharpe negativo**. `QuantileChannel`
(comprar o fundo do canal por quantil): −41% de drawdown. **As blue chips da B3
seguem, não revertem, no diário.** Confirmado também no M5 (negativo no
walk-forward).

### 6. Ensembles — **não ajudam**
Voto simples (SMA+MR+Mom): Sharpe 0,34. Voto ponderado por qualidade recente
(rolling Sharpe do P&L de cada membro): 0,50–0,55. **Sempre pior que o melhor
membro sozinho.** Os sinais fortes (SMA, Donchian, SuperTrend, Vortex) são todos
trend e correlacionados — juntá-los só dilui e adiciona giro. Falta um sinal
forte E descorrelacionado (que não achamos).

### 7. Filtros de força de tendência — **atrapalham**
`RegimeGated` (Kaufman Efficiency Ratio, gate binário): SuperTrend 0,78 → 0,19,
custo explode. `adx_scale_multiplier` (escala contínua pela força do ADX):
1,02 → 0,89. **Lição: escalar exposição por VOLATILIDADE ajuda; por força de
tendência não.** (O gate binário é pior; o contínuo também não presta.)

### 8. Vol-target Yang-Zhang — **A DESCOBERTA**
Overlay que escala a exposição de todos os nomes por `vol_alvo / vol_realizada`,
com o estimador **Yang-Zhang** (usa o OHLC inteiro — range intradiário + gap —
então detecta pico de vol mais rápido que close-a-close). **Melhora TODO sinal de
tendência**: SuperTrend, Donchian, Keltner, Vortex. É o maior lever depois da
diversificação.
- Vol-target close-a-close (a primeira tentativa) **atrapalhava** — só o estimador
  Yang-Zhang fez a diferença.

### 9. Momentum cross-sectional (19 ações B3) — **MORTO**
Rankear as ações por retorno e ficar long nas melhores. Long-only: Sharpe OOS 0,37,
**maxDD −71%** (2026 sozinho −64%). Long-short: negativo, −89%. O vol-target quase
não ajuda. **Momentum cross-sectional é literatura de universo de 500+ nomes; com
19 é só aposta concentrada que quebra.** (Shortear os "perdedores" da B3 perde —
efeito valor no corte transversal.)

### 10. ML — LightGBM (2 versões)
- **v1** (`ml_baseline.py`): rótulo "retorno futuro 10d > 0,4%", features de
  indicadores. `ml_only` Sharpe 0,30. Como filtro (`ml_and_champ`): **atrapalha**
  (1,09 → 0,47). Feature #1 = `month` — decorando calendário.
- **v2** (`ml_experiment.py`): rótulo **triple-barrier**, features de **rank
  cross-sectional** (`xs_*`), `month` removido, treino+avaliação no universo de 19.
  **AUC OOS = 0,541** (todos os 6 folds > 0,50 — sinal real, mas fraco; para ML em
  ações líquidas, 0,53–0,55 é "modelo que funciona"). Features agora fazem sentido
  (vol Yang-Zhang, momentum, ADX, os `xs_*`). **Mas nenhuma carteira derivada bate
  a regra**: `ml_topk` Sharpe 0,53, `ml_and_st` 0,56, vs `supertrend_ref` 0,92.
- **Veredito**: edge real (AUC 0,54), **insuficiente**. Para competir precisaria de
  universo B3 inteiro (100+ nomes), features fora do preço (fundamentos, fluxo),
  sizing pela magnitude do score. Engavetado.

### 11. Otimização do sizing (`optimize_champion.py`)
Grid walk-forward, CDI no caixa, objetivo = Sharpe sobre o CDI. Knobs: vol-alvo,
janela da vol, alavancagem, peso por ação.
- **Sharpe sobre o CDI trava em ~0,55–0,59** em toda a faixa sensata. Não há
  config mágica.
- **Alavancagem (1,5×): não ajuda** — nos vol-alvos baixos o multiplicador nem
  chega a 1,0.
- **Peso inverso-da-vol: pior** (0,51 vs 0,59 do equal) — com 5 blue chips só
  concentra no nome mais calmo.
- **Equal-weight, sem alavancagem** é o certo.
- O único botão de verdade é o **vol-alvo** (dial de risco): 0,12–0,15 = CDI +
  3–4%, maxDD −4–5%, melhor Sharpe/CDI; 0,25 = CDI + ~6%, maxDD −10%; 0,35 = CDI +
  8%, maxDD −15%.

---

## CAMPEÃO

**Cesta de 5 (PETR4, VALE3, ITUB4, BBDC4, ABEV3) + SuperTrend(period 10,
multiplier 3) long-only + vol-target Yang-Zhang (alvo 0,15, janela 20,
alavancagem 1,0) + equal-weight + sem trailing.**

Walk-forward, CDI creditado no caixa, OOS 2015–2026:

| métrica | valor |
|---|---|
| CAGR estratégia | ~14% a.a. |
| CAGR CDI (mesmo período) | ~9,8% a.a. |
| **alpha sobre o CDI** | **~+4% a.a.** (pré-imposto) |
| **Sharpe sobre o CDI** | **~0,6** |
| max drawdown | ~−5% |
| exposição média | ~25–33% (resto em caixa rendendo CDI) |
| custo / lucro bruto | ~5% |

Stress-test (OOS empurrado até 2010): Sharpe sobre 0 cai de 1,02 → 0,66, sem ano
de desastre (pior −6,6% em 2015). O Sharpe alto nas janelas curtas era inflado por
regime favorável.

**Ressalvas**: janela 2015–2026 foi boa para a bolsa BR (buy&hold também bateu o
CDI, mas com −48% de drawdown). IR não está nas contas — depois de imposto o alpha
real cai para ~+2,5–3,5% a.a. Custo de fundo/Tesouro para guardar o caixa: −0,2 a
−0,5% a.a. Sharpe/CDI ~0,6 é "vale a pena", não "muda a vida".

---

## Estado operacional

- **Paper-trade rodando na conta demo XP** (`XPMT5-DEMO`), config = campeão.
  `scripts/paper_trade.py` (dry-run por padrão; `--send` envia; trava de conta demo
  em `core/safety.py`). Estado em `data/paper_state.json`, diário em
  `data/paper_journal.csv`, log em `data/paper_trade.log`. Alertas no Telegram
  (bot `@tr4d3r_turbo_bot`, credenciais no `.env`).
- **Agendado** no Agendador de Tarefas do Windows: `trader-paper-trade`, dias
  úteis 17h35.
- **Primeira leva de posições: 2026-08-31** — PETR4 2400, VALE3 1400, BBDC4 6500;
  multiplicador de exposição 0,56; ~33% investido. Rodou limpo.
- Plano: acumular ~4 semanas → `scripts/paper_report.py` (tracking error vs
  backtest) → decidir capital real mínimo.

## Lições transferíveis

1. **Diversificar primeiro** — cortou o drawdown pela metade, de graça.
2. **Escalar por volatilidade (Yang-Zhang), não por força de tendência.**
3. **Blue chips da B3 seguem, não revertem** no diário — reversão à média é beco
   sem saída nesse horizonte.
4. **Ensembles precisam de membros descorrelacionados** — juntar sinais trend
   correlacionados não adianta.
5. **ML precisa de universo grande + dados fora do preço** — com poucos nomes e
   só features de preço, ele re-deriva trend-following fraco.
6. **O benchmark é o CDI.** Sharpe sobre 0 engana no Brasil.
7. **Trailing-stop é dependente do sinal** — ajuda MA crossover, atrapalha
   trend-following puro.

## Fios em aberto

- **WIN / WDO** (mini-índice / mini-dólar): trend + vol-target portam direto;
  precisa de fonte de dados de futuro no MT5, modelo de custo de futuro, rolagem.
  Adiciona diversificação real (ações + índice + dólar).
- **Universo B3 inteiro** (~100–150 nomes) + modelo de fatores de verdade
  (momentum + baixa-vol + valor com fundamentos). Cross-sectional falhou com 19;
  com 100+ e fatores é projeto de pesquisa sério.
- **Overlay de evento (eleição)** — reduzir exposição em janela de alta incerteza
  conhecida. Ver discussão à parte; não é backtestável (N pequeno).
- **Portfólio de estratégias** — combinar o trend atual com um sinal de fator
  descorrelacionado.
