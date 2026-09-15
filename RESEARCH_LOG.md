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

### 16. "Nunca realizar perda" — averaging down / grid (`research_grid_averaging.py`)

A estratégia manual do usuário: compra quando a ação está mais barata que o comum
dos últimos dias; se cai mais, vai comprando e baixando o preço médio (até
`max_adds`); vende **tudo** num repique acima do preço médio (`take_profit_pct`);
**nunca vende no vermelho**; se o caixa acaba, segura; caixa parado rende CDI.
`trader/backtest/grid_averaging.py`, caixa compartilhado entre as 5 da cesta.

OOS 2015–2026, cesta de 5, R$1M, CDI no caixa:

| config | CAGR | alpha vs CDI | Sharpe/CDI | maxDD |
|---|---|---|---|---|
| buy & hold da cesta | 17,9% | +8,2% | 0,41 | −48% |
| grid z1 drop5 tp3 f10 | 13,9% | +4,2% | 0,30 | −38% |
| grid z2 drop10 tp5 f8 | 15,1% | +5,3% | 0,47 | −25% |
| grid z1 drop5 tp3 **+ stop 25%** | 10,3% | +0,5% | 0,11 | −33% |

**O que a regra esconde** (variante base, 215 trades fechados):
- **acerto de 98% dos trades fechados** — é o "nunca realizei perda", confirmado.
  Mas é artefato contábil: você só fecha quem voltou pro verde.
- **maior espera até um lucro: 1 946 pregões (~7,7 anos)**. Posição no vermelho por
  1 933 pregões seguidos. Pior momento de uma posição: **−51% "embaixo d'água"**.
- **pico de caixa investido: 99%** — no pior momento, zero pólvora seca pra outra
  coisa (o custo que o usuário já reconhecia: "empata o caixa").
- caixa investido médio 42% → metade do capital parado no CDI é o que puxa o CAGR
  **abaixo do buy & hold da mesma cesta**.

**Stress — PETR4 sozinha no crash 2010–2017 (−85%):** CAGR 4,9% vs CDI 10,6% →
**alpha −5,7%**, Sharpe/CDI −0,44, posição −84% embaixo d'água, 1 lote preso no fim
com **−R$196 mil de prejuízo não realizado** (−19,6% do capital) que nunca virou
"perda" no extrato mas é perda de verdade.

**Veredito:** funciona num sentido estreito — cesta diversificada de sobreviventes,
década majoritariamente de alta → bate o CDI (+4–5% a.a.) com 98% de acerto. Mas
(1) **perde pro buy & hold da mesma cesta** (metade do caixa fica parado); (2) o
risco é **tempo e concentração** — anos embaixo d'água, caixa 100% preso na pior
hora; (3) **um único nome que não se recupera** (PETR4 2014–16, ou qualquer
delisting) crava um rombo permanente e destrói o histórico — e o backtest ainda
**superestima** porque o Yahoo só tem sobreviventes (viés de sobrevivência, §14c).
O stop de 25% **piorou** na cesta de blue chips (corta recuperações), que é
justamente por que a versão manual "deu certo": survivorship + regime bom. Fica
como estudo — não entra no programa. `entry_z 2 / drop 10 / tp 5` é a menos pior
(Sharpe/CDI 0,47, maxDD −25%) se algum dia virar um sleeve pequeno e capado.

### 17. Regimes de Markov — o "hedge fund method" do vídeo (`research_markov_regime.py`)

Fonte: framework de Roan/@RohOnChain, repositório
`github.com/jackson-video-resources/markov-hedge-fund-method` (spec + Pine Script +
backtest do autor). Estado pelo retorno composto de 20 pregões: bull > +5%, bear
< −5%, lateral no meio (limites exclusivos). Matriz 3×3 por contagem, sem
suavização, **re-estimada todo dia** com janela expansível (mín. 252 transições).
Sinal = P(bull amanhã) − P(bear amanhã). O código do autor opera `sign(sinal)`
long/short; o vídeo usa a magnitude como tamanho. `trader/regime/markov.py`.

**Diagnóstico (a parte que importa):**
- **A "persistência" é artefato da janela sobreposta.** Hoje e amanhã dividem 19 de
  20 retornos. Diagonal real da PETR4 0,84/0,78/0,88 (bear/lateral/bull) vs
  **retornos embaralhados — zero regime — 0,86/0,76/0,88**. Igual nas 5 ações. Com
  janelas disjuntas (estado a cada 20 pregões) a diagonal cai para 0,12–0,27 /
  0,39–0,56 / 0,28–0,35 ≈ frequência incondicional: **não há persistência real**.
- **A matriz prevê o estado de amanhã exatamente tão bem quanto "amanhã = hoje"**:
  83,8% vs 83,8% (OOS, 5 ações). Não acrescenta nada.
- **O sinal é função de 3 valores do retorno de 20 dias**: bear ≈ −0,83, lateral ≈
  0,00–0,03, bull ≈ +0,86, em todas as ações. "prob > 0,5" = "r20 > +5%" em
  **100%** dos dias; "sinal long" = "r20 ≥ −5%" em 92,6%. É momentum de 20 dias com
  banda — que já estava testado.
- **O estado bear NÃO prevê queda.** Retorno médio do dia seguinte (OOS, 5 ações
  juntas): bear **+7,0 bps** (t 1,4), lateral +4,6 (t 1,9), bull +12,5 (t 3,8),
  incondicional +7,8. Próximos 20 dias (sem sobreposição): bear +1,1%, lateral
  +1,6%, bull +2,0%. Bull − bear ≈ 5,5 bps/dia, t ≈ 0,9 — não significativo. Leve
  momentum no bull; **vender no bear é vender um ativo com retorno esperado
  positivo** (e pagar aluguel).

**Backtest na cesta**, R$1M, CDI no caixa, custos B3, aluguel 2% a.a. no short.
Escolha de variante em 2010–2015 (bear market BR — tudo negativo, inclusive o
campeão, −0,40), OOS 2015-07..2026-09:

| config | Sh/CDI 10–15 | CAGR | alpha | pós-IR | Sh/CDI | maxDD | expo |
|---|---|---|---|---|---|---|---|
| buy & hold cesta | −0,02 | 18,8% | +9,0% | — | 0,43 | −48% | 1,00 |
| **campeão (vol 0,15)** | −0,40 | 14,1% | +4,3% | +3,0% | **0,59** | −6% | 0,28 |
| campeão vol 0,25 | −0,37 | 16,1% | +6,4% | +5,0% | 0,57 | −10% | 0,44 |
| **campeão vol 0,35** | −0,32 | 16,9% | +7,1% | **+5,7%** | 0,54 | **−15%** | 0,51 |
| markov L/S sinal (autor) | −0,79 | 11,0% | +1,2% | +0,3% | 0,16 | −28% | 0,99 |
| markov L/S prob (vídeo) | −0,83 | 11,0% | +1,3% | +0,2% | 0,15 | −24% | 0,48 |
| markov L/S sinal +YZ | −0,88 | 11,0% | +1,2% | +0,2% | 0,18 | −8% | 0,50 |
| **markov long sinal** (escolhida) | −0,56 | 16,5% | +6,7% | **+5,5%** | 0,43 | **−22%** | 0,69 |
| markov long prob | −0,60 | 13,6% | +3,8% | +2,5% | 0,37 | −11% | 0,29 |
| markov long sinal +YZ | −0,69 | 13,5% | +3,7% | +2,5% | 0,46 | −10% | 0,36 |
| markov long prob +YZ | −0,72 | 11,9% | +2,1% | +0,9% | 0,42 | −5% | 0,15 |
| markov ±2% long sinal +YZ | −0,63 | 13,4% | +3,6% | +2,4% | 0,47 | −10% | 0,32 |
| momentum 20d banda 5% +YZ | −0,70 | 12,2% | +2,4% | +1,2% | 0,41 | −6% | 0,17 |
| HMM P(calmo) | −0,40 | 13,4% | +3,6% | +2,4% | 0,27 | −24% | 0,83 |
| HMM P(calmo) +YZ | −0,55 | 13,0% | +3,2% | +2,0% | 0,37 | −11% | 0,44 |

- **Long/short como no vídeo: praticamente CDI com drawdown de −24 a −28%.** O lado
  vendido é o que mata (coerente com §9 e com o bear que não prevê queda).
- **Long-only funciona como filtro "sai depois de um mês de −5%"**: a variante
  escolhida na janela de escolha dá o mesmo Sharpe do buy & hold (0,43) com metade
  do drawdown. Alpha pós-IR +5,5% parece bom, mas **no mesmo risco o campeão
  domina**: vol-alvo 0,35 = +5,7% pós-IR com −15% (vs −22%) e menos exposição; no
  mesmo drawdown de −10%, campeão +5,0% vs markov +2,5%.
- **HMM gaussiano de 2 estados** (a versão da literatura — Hamilton 1989;
  Kritzman-Page-Turkington 2012; jump models em Shu-Yu-Mulvey 2024) — estado
  latente, walk-forward anual, janela de 10 anos, `trader/regime/hmm.py`. Na PETR4
  acha calmo (+0,21%/dia, vol 1,9%) e turbulento (−0,85%/dia, vol 6,4%): é um
  **regime de volatilidade** — o que o vol-target Yang-Zhang já faz de forma mais
  direta. Sh/CDI 0,27–0,37 e ~12 mil trades (probabilidade contínua gira muito).

**Veredito:** não entra. O "método de hedge fund" é momentum de 20 dias com
banda, apresentado como matriz; a pegajosidade é matemática da janela, não do
mercado; o short perde; o long-only é dominado pelo campeão em qualquer nível de
risco. Lição reaproveitável: **teste nulo com retornos embaralhados** antes de
acreditar em qualquer "persistência" medida com janela móvel.

### 18. Scalping / day trade na XP — custos reais e 3 estratégias intraday (`research_intraday.py`)

**Dados:** o MT5 da XP tem séries contínuas de mini-índice e mini-dólar com **5 anos
de M1** (desde 2021-09, ~690 mil barras cada) — `scripts/pull_mt5_intraday.py`.
`WIN$N`/`WDO$N` = sem ajuste (salta na rolagem); `WIN$D`/`WDO$D` = ajustada por
diferença (movimento em pontos idêntico, sem salto) — resultado sai da `$D`,
notional da `$N`. `WIN$` = ajuste proporcional. Pregão: WIN/WDO 9:00–18:25; à vista
10:00–16:55 com horário de verão americano, 10:00–17:55 sem.

**Custos levantados (2026-09 — conferir na tabela vigente):**
- B3 day trade: WIN ~R$0,25 e WDO ~R$1,20 por contrato por ponta; ação 0,023% por
  ponta (até R$1 mi/dia; swing 0,030%).
- XP: corretagem **zero** em minicontratos e day trade de ação **só com RLP ativado**
  e ordem do próprio cliente em plataforma eletrônica; sem RLP R$1,00 por
  minicontrato. Ação swing no home broker: fontes indicam **R$4,90 por ordem**.
- MetaTrader 5 na XP: **R$299,90/mês** na tabela comparativa de plataformas (há
  menção a isenção por volume de minicontratos — não confirmado).
- IR day trade 20% do lucro líquido mensal (o 1% retido é antecipação).

Por operação completa com 1 tick de slippage por ponta: **WIN R$2,50/contrato =
2,5 ticks = 0,7 bp**; **WDO R$12,40 = 2,5 ticks = 2,4 bps**; **PETR4 R$4,25 por 100
ações = 4,2 ticks = 8,7 bps** (sem RLP: R$14,05 no lote de 100 — a corretagem por
ordem dilui em lote grande).

**Custo x movimento típico (últimos 12 meses, 10:05–16:50, mediana |movimento|):**

| ativo | custo ida+volta | 1 min | 5 min | 15 min | 60 min | acerto p/ empatar (alvo=stop=mediana 15 min) |
|---|---|---|---|---|---|---|
| WIN | 2,5 ticks | 7 | 15 | 26 | 53 | **55%** (custo = 10% do movimento) |
| WDO | 2,5 ticks | 1 | 3 | 5 | 10 | **75%** (50%) |
| PETR4 | 4,2 ticks | 1 | 3 | 4 | 9 | **103% — impossível** (106%) |

Scalping de ação é inviável por construção. WIN é o único onde o custo é pequeno
perto do movimento — por isso é o ativo preferido dos day traders no Brasil.

**Estratégias (M1, 1× o patrimônio em notional, CDI no caixa, IR 20%):**
- **Momentum intraday** (Gao, Han, Li & Zhou 2018, JFE): 1ª meia hora (fechamento
  anterior → 10:30) prevê a última meia hora do à vista.
- **Rompimento da faixa de abertura (ORB)**: âncora 9:00/10:00, faixa 5/15/30/60
  min, com/sem stop no lado oposto, saída no fechamento do à vista.
- **Scalp de 1 minuto**: minuto com |retorno| ≥ k desvios da vol dos 60 min
  anteriores → segue ou reverte, sai após 5/15/30 min.

Corte único (treino 2021-09..2023-09, teste 2023-09..2026-09) — todas negativas
depois do IR no teste (WIN momentum −5,0%, ORB WIN −13,0%, scalp WIN −3,8%, WDO −4,5
a −8,7%, scalp PETR4 −29%). **Walk-forward rolante** (re-escolhe a cada 126 pregões
com os 252 anteriores), OOS 2022-09..2026-09, alpha sobre o CDI a.a.:

| estratégia | ativo | sem custo | com custo | pós-IR | Sh/CDI | maxDD |
|---|---|---|---|---|---|---|
| momentum intraday | WIN | −0,3% | −2,1% | −4,4% | −0,84 | −2% |
| momentum intraday | WDO | +0,6% | −2,8% | −4,9% | −2,20 | 0% |
| rompimento abertura | WIN | **+3,3%** | +0,6% | −4,0% | +0,10 | −15% |
| rompimento abertura | WDO | −0,2% | −6,4% | −9,7% | −0,56 | −11% |
| scalp 1 min | WIN | +0,3% | −4,3% | −7,0% | −0,78 | −9% |
| scalp 1 min | WDO | −2,1% | −7,5% | −9,6% | −2,17 | −3% |
| scalp 1 min | PETR4 | −1,8% | −37,5% | −39,2% | −3,71 | −68% |

- **Momentum intraday de Gao et al. não existe no WIN 2021–2026** (bruto ≈ 0).
- **Scalp de 1 minuto:** vantagem bruta de zero a levemente positiva; o custo é
  várias vezes maior.
- **ORB no WIN é aposta de regime, não estratégia:** perdeu em 14 das 16
  combinações em 2021–23 e ganhou em 13 das 16 em 2023–26. A escolha no treino pegou
  uma das 3 perdedoras do teste (−13%); o walk-forward rolante trocou de parâmetro
  em 6 de 8 janelas e fechou em +0,6% antes do IR com −15% de drawdown.
- Evidência brasileira: Chague, De-Losso & Giovannetti ("Day Trading for a
  Living?", dados da CVM, 19.646 day traders de mini-índice 2013–15): **97% dos que
  persistiram mais de 300 dias perderam dinheiro**; 1,1% ganhou mais que um salário
  mínimo. Sem evidência de aprendizado.
- Motivo estrutural: a vantagem em horizonte de segundos/minuto está na
  microestrutura (fila no livro, latência, co-location) — HFT. Python → MT5 → XP
  paga o spread e chega atrasado.

**Veredito:** não entra. Nosso horizonte continua sendo swing/posição.

### 19. Arbitragem entre classes — ON x PN / unit x PN (`research_share_classes.py`)

Ideia: duas classes da mesma empresa têm o mesmo fluxo de caixa; o spread
log(A) − log(B) (preço ajustado por proventos, hedge 1:1 em valor, sem estimar
beta) deveria oscilar em torno do prêmio de voto/tag along. Referência brasileira:
Caldeira & Moura (2013, Brazilian Review of Finance) — carteira de pares
cointegrados na B3 2005–2012, +16,4% a.a. de excesso, Sharpe 1,34, correlação
< 0,15 com o Ibovespa. `trader/pairs/` (spread, z-score, máquina de estados,
carteira com slots) + `scripts/pull_share_classes.py` + `share_classes.pairs` no
`config/settings.yaml`.

**Dados:** 38 pares candidatos; 13 tickers sem dados no Yahoo — **justamente as
classes extintas** (CESP3/6, CPLE6, ELET3/6, LAME3/4, TIET3/4, TRPL3/4, VIVT4,
RANI4) → viés de sobrevivência de direção desconhecida (unificação às vezes paga a
classe descontada, às vezes pune quem estava vendido nela). Excluídos por dado
podre: OIBR3/4 (recuperação judicial, saltos) e VALE3/5 (VALE5 extinta em 2017,
ajuste inconsistente). Ficaram 28.

**Liquidez derruba o universo.** Com R$5 mi/dia (mediana 60 pregões) nas DUAS
pontas: PETR3/4 100% dos dias desde 2006, BBDC3/4 71%, USIM3/5 35%, ITUB3/4 37%,
SAPR11/4 32%, KLBN11/4 22%, CMIG3/4 11% — o resto ~0 (a ON quase não negocia, ou a
PN da empresa com unit é ilíquida). Com R$1 mi/dia entram mais ~10, caros de operar.

**Diagnóstico dos spreads:**
- ON/PN líquidos revertem **devagar**: meia-vida no treino (2006–2014) PETR 123,
  BBDC 233, ITUB 102, USIM 114 pregões, e o ADF **não** confirma estacionariedade
  (−1,3 a −2,0). O prêmio muda de patamar por anos: PETR3 sobre PETR4 foi de +3%
  (2014) a +31% (2016); USIM3 chegou a valer ~3× a USIM5 na disputa
  Ternium/Nippon (2015–16).
- Unit x PN revertem rápido (SAPR 20, KLBN 9, SANB 6, TAEE 1 pregão; ADF −4,5 a
  −17,7) — é o vai-e-vem do spread de compra e venda de uma unit que é cesta das
  próprias classes, e a PN costuma ser ilíquida.

**Backtest:** z-score móvel, entra além de ±entry, sai dentro de ±exit, stop |z| > 4,
prazo 60 ou 250 pregões (o 250 foi incluído porque a meia-vida no TREINO deu
100–230), até 8 pares abertos, bruto ≤ 100%, neutro em valor (patrimônio no CDI),
execução na abertura seguinte, custo por ponta, aluguel no vendido. Universos
definidos antes: **líquido** (R$5 mi, 8 bps, aluguel 2%) e **amplo** (R$1 mi,
20 bps, aluguel 4%). Grid de 48 (janela 20/60/120/250 × entrada 1,5/2/2,5 × saída
0/0,5 × prazo 60/250), escolha no treino 2006-01..2014-04, teste 2014-04..2026-09:

| universo | escolhido no treino | Sh/CDI treino | alpha teste | pós-IR | Sh/CDI teste | maxDD | expo |
|---|---|---|---|---|---|---|---|
| líquido | janela 60, entra 2,5, sai 0,5, prazo 60 | +0,35 | −0,1% | −1,1% | −0,06 | −1% | 0,10 |
| amplo | janela 60, entra 2,5, sai 0,5, prazo 60 | −0,19 | −1,6% | −2,5% | −0,96 | −6% | 0,19 |

- Grid inteiro no teste: líquido **4 de 48** com Sh/CDI positivo (mediana −0,20);
  amplo **0 de 48** (mediana −0,96). Não é escolha de parâmetro.
- Walk-forward rolante (treino 3 anos, teste 1 ano, 2009–2026): líquido alpha 0,0%,
  Sh/CDI −0,02; amplo −1,7%, −0,90. Excesso anual do líquido oscila entre −1,3% e
  +1,3% — nunca vira nada.
- Custo otimista (4 bps, aluguel 1%): líquido Sh/CDI +0,16, alpha +0,1% — nem de
  graça rende.
- Contribuição por par no teste (antes de custo): BBDC +1,6%, SAPR +1,5%, KLBN
  +0,7%, PETR −0,7%; no amplo, **USIM −8,0%** (a disputa de controle) apaga o resto.
- Correlação diária com o campeão −0,02 e com o Ibovespa +0,04 — é neutra de verdade,
  mas sem retorno não diversifica: campeão + pares por cima Sh/CDI 0,47 vs 0,48
  sozinho (líquido); 0,27 (amplo).

**Veredito:** não entra. O Sharpe/CDI de +0,35 no treino (2006–2014) bate com a era
em que pares funcionavam na B3 (Caldeira & Moura, 2005–2012) — e **sumiu** depois.
Liquidez limita o universo a ~6 pares, os líquidos revertem em meses com mudança
de patamar, e evento societário (controle, conversão) é a cauda que destrói.

### 20. Efeitos de calendário: virada de mês e overnight x pregão (`research_calendar.py`)

`trader/seasonal/` (`month_offsets`, `tom_flag`, `timing_returns` com execução no
leilão de fechamento ou na abertura seguinte), `scripts/pull_etfs.py` (BOVA11).
**A abertura do ^BVSP no Yahoo é falsa** (= fechamento anterior em 21–27% dos dias):
serve para retorno fechamento-a-fechamento, nunca para gap/overnight. BOVA11 e ações
têm abertura real (PETR4 conferida contra o MT5).

#### 20a. Virada de mês (McConnell & Xu 2008 — efeito em 31 de 35 países)

Ibovespa, retorno médio por pregão (bps/dia):

| pregão | 2000–2012 | t | 2013–2026 | t |
|---|---|---|---|---|
| antepenúltimo | +24,0 | 1,7 | +1,0 | 0,1 |
| último | **+31,1** | 2,6 | −9,4 | −0,9 |
| 1º do mês | **+65,9** | 4,1 | +14,2 | 1,2 |
| 2º | +15,3 | 0,9 | +16,2 | 1,4 |
| 3º | +4,3 | 0,3 | +7,0 | 0,7 |
| meio do mês | −1,8 | −0,4 | +3,6 | 1,1 |

- Janela [último, +3] = 19% dos pregões: **+13,1% a.a. em 2000–12 contra −3,2% no
  resto do mês; em 2013–26, +2,9% contra +5,2%.** Muito forte até 2012 e sumiu.
- Janela escolhida no treino: [último, 1º] (Sh/CDI treino +1,13). Teste no BOVA11
  2013–26 (4 bps/ponta, CDI no caixa): alpha −2,4%, Sh/CDI −0,28. Janela da
  literatura [−1,+3]: no leilão de fechamento alpha −0,5% (Sh 0,00); na abertura
  seguinte +2,2% (Sh +0,24) — frágil: a diferença é trocar a noite do último pregão
  pela do 4º. Buy & hold BOVA11 no período: alpha −1,5%, maxDD −47%.
- **Placebo** (os mesmos pregões/mês sorteados, 500 vezes): a virada fica acima de
  89% dos sorteios — não significativo.
- Walk-forward anual (janela re-escolhida com todo o Ibovespa anterior): alpha
  −2,2%, Sh/CDI −0,15. Correlação com o campeão +0,29; 50/50 piora (0,42 → 0,39).

#### 20b. Overnight x pregão (Cliff, Cooper & Gulen 2008; Lou, Polk & Skouras 2019)

- Retorno médio × 252, noite = fechamento → abertura, dia = abertura → fechamento:
  BOVA11 noite **+18,0%** (t 4,7) / dia −5,6%; PETR4 **+35,3%** (t 6,7) / −11,6%; VALE3
  +26,7% / −3,1%; ITUB4 +15,5% / +5,2%; BBDC4 +10,2% / +0,7%; ABEV3 +7,6% / +13,5%.
  PETR4 teve noite positiva em **todos** os anos de 2003 a 2026.
- Universo B3 desde 2013 (89 ações com abertura confiável): noite positiva em 92–97%
  das ações, média +22 a +29% a.a.; pregão médio −7 a −12% a.a.
- Não é artefato de provento: PETR4 2021–26 no MT5 bruto, noite +25,4% a.a. (Yahoo
  ajustado +24,5%); as 22 datas ex somam −0,3% a.a.
- **Mini-índice (5 anos de M1): nenhum trecho do dia é significativo** (noite +2
  pts/dia, t 0,1; 9h–10h −9; pregão +1; pós-fechamento +7). O futuro desconta 1 dia
  de CDI a cada virada de pregão: noite do WIN + CDI = +11,1% a.a. ≈ noite do BOVA11
  no mesmo período. **No índice, a "noite positiva" é o CDI embutido**; nas ações
  individuais sobra mais que isso.
- Estratégia "compra no leilão de fechamento, vende no de abertura", todo pregão,
  peso igual, universo pela liquidez conhecida 2 dias antes, IR swing 15%. Duas
  hipóteses de caixa: **sem CDI** (o dinheiro está na ação à noite — conservador) e
  **com CDI** (liquidação D+2 compensada, patrimônio segue no CDI — otimista).
  Alpha sobre o CDI, teste 2013–2026:

| universo | treino Sh/CDI | sem CDI, só taxa (6 bps/dia) | +2 bps slip/ponta | +corretagem R$4,90 | com CDI, só taxa | com CDI +slip |
|---|---|---|---|---|---|---|
| BOVA11 | −1,22 | −5,4% | −15,2% | −15,4% | +4,8% | −5,9% |
| cesta 5 | −0,62 | −2,6% | −12,8% | −13,9% | +7,8% | −3,3% |
| top 20 | +0,07 | −1,5% | −11,7% | −16,3% | +9,1% | −2,1% |
| top 50 | +0,22 | −2,2% | −12,4% | −23,5% | +8,2% | −2,9% |

- Por ano (top 50, sem CDI, menos o CDI): segue o mercado — +8 a +22% em 2016–21,
  −11 a −22% em 2013–15 e 2022–25. Correlação diária com o Ibovespa +0,64; pior
  noite −14,1% (covid).
- **Espelho — comprado só durante o pregão: alpha −21,7%, Sh/CDI −1,16.**

**Veredito (as duas):** não entram. A virada de mês existiu até 2012 e sumiu. O
overnight é o achado mais didático da pesquisa — na B3 o retorno de ficar comprado
aparece quase todo **fora** do pregão e ficar comprado só durante o pregão é
péssimo —, mas explorar isso exige ida e volta diária: 252 × qualquer atrito. Só
fecha positivo com custo exatamente igual à taxa da B3 **e** caixa rendendo CDI;
2 bps de slippage por ponta já zeram. Derivado útil: testar o campeão entrando no
**leilão de fechamento** (sinal calculado minutos antes) em vez da abertura
seguinte — cada entrada ganharia a noite que hoje perde.

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
- **Checagem em 2026-09-14 (11 pregões de paper real):** equity R$999.998 →
  R$1.022.608 (+2,26%), maxDD −0,45%, 5 posições (~54% investido). `paper_report.py`
  tinha um bug — cortava o histórico exatamente no início da janela, sem
  aquecimento pro SuperTrend/vol-target, e dava backtest 0% flat; corrigido (agora
  aquece 250 pregões antes e reindexa por dia, não por timestamp de execução).
  Depois do fix: backtest da mesma config na mesma janela = **+1,51%**, tracking
  error anualizado **2,2%** — a execução real bate o script quase igual ao
  esperado, levemente à frente. `vs_benchmark` também tinha o mesmo problema de
  timestamp vs CDI diário (`compare_paper.py` já normalizava certo). **Cedo demais
  pra anualizar** (CAGR/alpha só faz sentido com ≥60 dias corridos — `compare_paper.py`
  esconde a anualização até lá).
- **`trader-blend-sim` nunca tinha sido agendado de verdade** (só rodei uma vez na
  hora de construir) — parquets do universo e CDI ficaram parados em 2026-09-01.
  Agora: dados atualizados, sim catch-up rodado (+1,97% até 2026-09-14, não é uma
  simulação dia-a-dia fiel desse intervalo — só o snapshot de chegada), e a tarefa
  `trader-blend-sim` agendada (dias úteis 18h30) pra rodar de verdade daqui pra
  frente. Comparação de fato só fica útil com as duas rodando diariamente por
  algumas semanas.

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
rodando diariamente.

## Rodar os dois em paralelo (conservador vs mais risco)

Pergunta: deixar o **campeão** (conservador) e o **blend** (mais risco / mais
retorno) rodando ao mesmo tempo?

**Não dá para os dois enviarem ordem na mesma conta demo** — os dois miram
PETR4/VALE3/BBDC4 e o `reconcile` de cada um enxergaria a posição do outro como
divergência e tentaria "corrigir". Brigariam pelo sizing.

Solução adotada: **campeão continua enviando de verdade** na demo
(`paper_trade.py --send`, tarefa `trader-paper-trade`); **o blend roda em
simulação** — `scripts/paper_blend_sim.py`: sem MT5, sem ordem, marca a carteira a
fechamento pelos parquets, credita CDI no caixa, aplica 8 bps de custo de giro.
Estado em `data/blend_sim.json`, curva em `data/blend_sim_equity.csv`, começa no
mesmo capital do campeão (lido do `paper_journal.csv`). `scripts/compare_paper.py`
põe as duas curvas lado a lado vs CDI. Bat: `paper_blend_sim_daily.bat`.

Alternativas se quiser o blend **enviando de verdade** também: (a) 2ª conta demo na
XP + `mt5.login()` por script (comparação limpa, com fills reais); (b) trocar de
vez o paper-trade para o blend (`paper_blend_daily.bat`) e abrir mão do campeão.
Por ora fica (sim) — 2–4 semanas de curva comparável custam nada.

## Short — decidido: **fica de fora**

Re-perguntado. Já testado e perdedor em toda configuração: long-short
cross-sectional na B3 = Sharpe −0,20, drawdown −89% (§9); hedge com WIN piora todo
config (Sharpe 0,66 → 0,26, §11). O vol-target YZ já desalavanca em stress — é a
proteção de baixa do projeto. Quem quiser proteção extra: mais caixa (alvo de vol
menor) ou put de índice (custo recorrente, infra de opção), **não** short
sistemático. Design permanece **long-only + CDI no caixa**.

## Programa de eleição — decidido: **não implementar overlay sistemático**

Re-perguntado. `scripts/election_test.py`: o haircut de calendário pré-comprometido
não move nada (Sharpe/CDI 0,49 → 0,48, mesmo drawdown) e custou 2,3 pp no rali de
2018. `trader/risk/election.py::haircut_series` existe para quem quiser ligar na
mão por tranquilidade, mas **não está plugado no executor** e não é recomendado. Um
palpite direcional na eleição, se houver, é um balde tático pequeno e **manual**,
separado do sistema — não uma regra do programa.

## Fios em aberto

- **Fatores fundamentalistas** (valor/qualidade): brapi free não tem histórico;
  fazer via dividend yield (Yahoo).
- **Corrigir o universo de fatores** — 22 tickers faltando no Yahoo (viés de
  sobrevivência). Tentar outra fonte / outro formato de ticker (B3 COTAHIST).
- **IR e viés de sobrevivência**: ajustar os números para depois de imposto e para
  o universo real antes de qualquer capital de verdade.
- **Campeão executando no leilão de fechamento** (§20b): hoje o sinal do fechamento
  executa na abertura seguinte; como a noite rende e o pregão perde na média, entrar
  no leilão de fechamento (sinal com o preço de ~16h50/17h50) deve ganhar uma noite
  por entrada. Barato de testar no backtest; exige rodar o executor antes do
  fechamento, não às 17h35.
- **Custo real do campeão na XP** (§18): o modelo assume corretagem zero em ação
  swing, mas as fontes indicam R$4,90 por ordem no home broker (via MT5: confirmar
  na nota de corretagem) + MT5 a R$299,90/mês. O backtest do campeão faz ~575
  ordens/ano (muitos ajustes de 100 ações pelo vol-target) → ~R$2,8 mil/ano em
  corretagem + ~R$3,6 mil/ano de plataforma ≈ 0,64% a.a. em R$1 mi. Conferir antes
  de capital real; se confirmado, pôr no `config/settings.yaml` e testar uma banda
  de rebalanceamento (só ajustar se o desvio do alvo passar de X%).
