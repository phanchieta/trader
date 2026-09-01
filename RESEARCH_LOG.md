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

### 11. WIN / WDO (mini-índice / mini-dólar) — **NÃO AJUDAM**
Engine novo baseado em retornos (`backtest/trend_portfolio.py`) — long/short,
alavancagem, vol-target auto-referente. Proxies do Yahoo: `^BVSP` (Ibov) e
`USDBRL=X` (dólar). Walk-forward, CDI no caixa:
- só ações (ref): Sharpe/CDI 0,44 | ações + USDBRL: 0,39 | ações + IBOV + USDBRL: 0,27.
- **IBOV + USDBRL sozinhos (macro puro): Sharpe/CDI −0,52, −28% de drawdown, CAGR
  2,7% — muito abaixo do CDI.**
- Alavancagem 3×: pior (0,25 / −28%).
- Motivos: (a) Ibov ≈ a cesta (correlação ~0,9, redundante); (b) trend-following
  no USD/BRL perde — moeda de carry, sobe devagar e cai violento, toma chicotada;
  (c) 2–3 instrumentos é pouco — CTA de verdade roda 50–100+ mercados globais.
- Ressalva: proxies do Yahoo, não o futuro real; eficiência de capital / shortear
  fácil não aparece como retorno neste teste. Mas não flipa a conclusão.

### 12. Modelo de fatores cross-sectional (universo B3 grande) — **FUNCIONA**
`factors/library.py` + `factors/backtest.py` + `scripts/factor_model.py` /
`factor_tuning.py`. Universo: 98 ações (`config factor_universe`; 22 de 120
falharam no Yahoo → **viés de sobrevivência** que infla os retornos). Rebalance
periódico, top quintil long-only equal-weight, vol-target + CDI no caixa.

Fatores de preço (walk-forward, janela longa ~2011–2026):
| fator | Sharpe/CDI | alpha | maxDD |
|---|---|---|---|
| **momentum (12-1)** | 0,59 | +8,8% | −24% |
| **combinado 5 fatores** | **0,69** (trimestral) | +8,8% | −15% |
| low_vol | 0,31 | +2,9% | −14% |
| low_beta | 0,33 | +3,3% | −21% |
| lt_reversal (loser 3a = proxy valor) | 0,07 | −0,6% | −50% (value trap) |
| st_reversal (contrário 1 mês) | **−0,29** | −7% | −58% (reversão falha de novo) |

**Momentum cross-sectional funciona COM BREADTH** — falhou com 19 nomes (−71% DD),
funciona com ~98. Antes a conclusão "precisa de 500+" era pessimista demais; ~100
basta na B3.

**div_yield** (DY 12m, dados do Yahoo): Sharpe/CDI 0,36, alpha +2,7% — fator real
mas fraco. **Adicionar ao combinado NÃO muda nada** (COMBINADO 5 = COMBINADO 6):
DY é correlacionado com low-vol/low-beta (pagadoras de dividendo são utilities e
bancos de baixa vol), não traz informação nova.

**Tuning** (`factor_tuning.py`):
- **rebalance trimestral > mensal** (0,69 vs 0,59) — menos giro, menos IR. Mantido.
- **quintil > decil** (0,69 vs 0,23) — decil concentra demais.
- **vol-target pela vol do Ibov > pela vol do book** na janela longa (drawdown menor).
- **hedge de índice (WIN short): atrapalha tudo** (0,66 → 0,26). Shortear índice BR
  sistemático perde. Confirmado.

### 13. Portfólio de 2 sleeves: campeão + fatores (`two_sleeve.py`)
Correlação diária das duas sleeves = **0,41** (finalmente descorrelacionado).
Janela OOS longa e honesta:
| mix | Sharpe/CDI | alpha | maxDD |
|---|---|---|---|
| campeão só | 0,41 | +2,7% | −6,1% |
| fatores só | 0,69 | +8,8% | −15,1% |
| 30% fatores / 70% campeão | 0,64 | +4,7% | −6,7% |
| **50 / 50** | **0,69** | +5,9% | **−9,1%** |
| 70% fatores / 30% campeão | 0,70 | +7,1% | −11,5% |

O blend **não bate os fatores em Sharpe/CDI, mas corta o drawdown pela metade**
(50/50: mesmo 0,69 com −9% em vez de −15%). É o melhor risk-adjusted do projeto.
**Teto realista encontrado: Sharpe sobre o CDI ~0,7.** (Os 0,94/1,01 de um teste
anterior eram janela favorável 2016+.)

### 14. Fundamentos (valor / qualidade)
brapi.dev free dá só fundamento ATUAL (P/L, LPA, market cap), não série histórica.
Caminho usado: **dividend yield** (Yahoo tem histórico de proventos — 96 ações,
4242 eventos, `data/dividends.parquet`, `scripts/pull_dividends.py`). Adicionado
como fator `div_yield` (DY 12m). Ver §12 para o resultado do combinado 6 fatores.

### 14b. Overlay de eleição — **NÃO AJUDA** (`scripts/election_test.py`)
Haircut pré-comprometido: exposição × 0,5 de 15/set a 20/nov nos anos eleitorais
(2010/14/18/22). Walk-forward:
- campeão: Sharpe/CDI 0,49, maxDD −7,0%.
- campeão + haircut: Sharpe/CDI 0,48, maxDD −7,0% (praticamente idêntico).
- 2018 (set–dez): sem haircut +5,7%, com haircut +3,4% — **custou** (Bolsonaro
  eleito, bolsa subiu). 2022: +3,4% vs +3,8% — ajudou de leve.
**O vol-target do próprio campeão já lida com a vol da eleição.** Um calendário
fixo é tão provável de atrapalhar quanto de ajudar. Não implementar.

### 14c. Viés de sobrevivência do universo de fatores
Retry recuperou 4 sucessores por mudança de ticker (DXCO3←DTEX3, AZZA3←ARZZ3/SOMA3,
MBRF3←MRFG3/BRFS3, MOTV3←CCRO3). **~14 nomes vivos que o Yahoo não serve** (ELET3,
EMBR3, JBSS3, RENN3, …) + zero deslistados de verdade (CIEL3, CRFB3 fecharam
capital). Corrigir de vez precisa de base point-in-time paga ou dos arquivos
COTAHIST da B3. **O alpha dos fatores está otimista em ~1–3 pp por causa disso.**

### 15. Otimização do sizing do campeão (`optimize_champion.py`)
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

## Estado da pesquisa (2026-09-01)

**Dois programas validados:**
1. **Campeão** — 5 ações + SuperTrend + vol-target YZ. Sharpe/CDI ~0,4–0,6, drawdown
   ~−6%. Rodando no paper-trade.
2. **Modelo de fatores** — universo B3 (~98 nomes), combinado 5 fatores, quintil,
   rebalance trimestral, vol-target Ibov + CDI. Sharpe/CDI ~0,69, drawdown −15%.

**Depois do IR** (aprox. 15,5% mensal, prejuízo compensável — `scripts/after_tax.py`):

| programa | alpha vs CDI pré-IR | **alpha pós-IR** | maxDD |
|---|---|---|---|
| campeão | +2,7% | **+1,5%** | −6% |
| fatores | +8,8% | **+7,2%** | −15% |
| blend 50/50 | +5,9% | **+4,5%** | −9% |

**Conclusão:** o **campeão sozinho mal supera o CDI depois do imposto** (+1,5% a.a.)
— é redução de risco, não prêmio de retorno. O **modelo de fatores / blend batem o
CDI de forma real** (+4,5 a +7,2% pós-IR), e depois de descontar o viés de
sobrevivência (~1–3 pp) ainda sobra **~CDI + 3–5% a.a. pós tudo**, com drawdown
−9 a −15%. Sharpe sobre o CDI ~0,7 (pré-IR) é o teto do projeto.

## Executor do blend ao vivo

`trader/live/blend_planner.py` (`plan_blend`) + `scripts/paper_blend.py` +
`paper_blend_daily.bat`. Combina as duas sleeves num mapa único de peso por ação:
- campeão: SuperTrend nas 5 ações, re-avaliado diariamente, vol-target YZ da cesta.
- fatores: top quintil do score combinado, SELEÇÃO por trimestre-calendário
  (congelada no `blend_state.json` entre rebalances), vol-target YZ do Ibov diário.
Estado/diário separados do `paper_trade.py` (`blend_state.json` / `blend_journal.csv`).
Dry-run offline (equity R$1M): 21 posições (3 campeão + 20 fatores, 3 sobrepostas),
~59% investido. Lê os parquets do universo — precisa de `pull_factor_universe.py`
rodando diariamente. **Pendente: rodar o dry-run com o MT5 aberto e trocar a
tarefa agendada de `paper_trade_daily.bat` para `paper_blend_daily.bat`.**

## Fios em aberto

- **WIN / WDO**: testado (proxies ^BVSP / USDBRL=X) — **não ajuda** (§11).
- **Fatores fundamentalistas** (valor/qualidade): brapi free não tem histórico;
  fazer via dividend yield (Yahoo).
- **Corrigir o universo de fatores** — 22 tickers faltando no Yahoo (viés de
  sobrevivência). Tentar outra fonte / outro formato de ticker.
- **Overlay de evento (eleição out/2026)** — reduzir exposição na janela; não é
  backtestável (N pequeno). Ver discussão à parte.
- **Paper-trade do blend** — hoje o paper roda só o campeão. Adicionar a sleeve de
  fatores exige um executor cross-sectional ao vivo.
- **IR e viés de sobrevivência**: ajustar os números para depois de imposto e para
  o universo real antes de qualquer capital de verdade.
