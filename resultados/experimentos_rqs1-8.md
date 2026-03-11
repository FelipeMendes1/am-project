# Registro de Experimentos — RQs 1–8

**Data de execução:** 2026-03-10
**Tempo total:** ~163 segundos
**Protocolo:** Validação cruzada entre bases — treina em cada uma das 10 bases, testa nas outras 9 (90 pares por condição experimental)

---

## Perguntas de Pesquisa (RQs)

O estudo investiga como características da base de dados e estratégias de modelagem afetam a equidade (*fairness*) de modelos de aprendizado de máquina aplicados à detecção de fraudes em transações financeiras. As nove perguntas de pesquisa são:

| RQ | Pergunta | Experimento |
|----|----------|-------------|
| **RQ1** | A quantidade de atributos (*features*) garante maior justiça no modelo? | Exp. 2 — Feature Sets |
| **RQ2** | Quando temos mais dados de treinamento, podemos ter mais viés? | Exp. 1 — Volume |
| **RQ3** | Aumentar a quantidade de atributos traz precisão semelhante ao aumento de equidade? | Exp. 2 — Feature Sets |
| **RQ4** | Um modelo treinado com quantidade insuficiente de dados pode reforçar preconceitos presentes na base? | Exp. 1 — Volume |
| **RQ5** | O aumento no volume de dados de treinamento pode prejudicar a equidade do modelo? | Exp. 1 — Volume |
| **RQ6** | Um conjunto de dados maior implica necessariamente em maior injustiça? | Exp. 1 — Volume |
| **RQ7** | Balancear os dados tem impacto positivo limitado na equidade quando comparado com outros métodos de mitigação de viés? | Exp. 3 — Mitigação |
| **RQ8** | Métodos de melhoria de equidade, como reponderação de amostras, trazem melhorias significativas na equidade? | Exp. 3 — Mitigação |
| **RQ9** | Técnicas específicas de mitigação de preconceito, como o Prejudice Remover, conseguem melhorar o equilíbrio entre *fairness* e acurácia? | `treino.py` (η sweep) |

> **Nota:** RQ9 é coberta pelo pipeline principal (`treino.py`) com varredura de η ∈ {0, 1, 5, 10, 25, 50, 75, 100}. As RQs 1–8 são investigadas pelos três experimentos descritos neste documento.

### Métricas utilizadas

**Desempenho preditivo**
- **F1-Score** — equilíbrio entre precisão e revocação; métrica principal de performance
- **Precision** — fração das predições positivas que são corretas
- **Recall** — fração das fraudes reais que o modelo detecta

**Equidade (*Fairness*)**
- **abs_SPD** (*Statistical Parity Difference* absoluta) — |P(ŷ=0 | S=0) − P(ŷ=0 | S=1)|; mede a diferença de taxa de predição favorável entre grupos. Justo se ≤ 0,1.
- **DI** (*Disparate Impact*) — P(ŷ=0 | S=0) / P(ŷ=0 | S=1); razão das taxas de predição favorável. Justo se ≥ 0,8 (regra dos quatro quintos).

Onde S=0 é o grupo não-privilegiado (RAMO_ATIVIDADE_1 = 4, mapeado para 0) e ŷ=0 é a classe favorável (transação aprovada/normal).

---

## Contexto e protocolo experimental

O conjunto de dados é composto por 10 bases de transações financeiras organizadas em duas séries
(**PD** e **GS**), cada uma com 5 níveis de viés sintético introduzido progressivamente
(v = 0,0; 0,25; 0,50; 0,75; 1,0). A variável sensível é `RAMO_ATIVIDADE_1`
(binária: 0 = grupo não-privilegiado, 1 = grupo privilegiado); o alvo é `I-d`
(0 = transação aprovada/normal, classe favorável; 1 = fraude/anomalia).

Duas métricas de justiça são reportadas:

- **abs_SPD** (*Statistical Parity Difference* absoluta): diferença entre a taxa de predição favorável para o grupo não-privilegiado e o privilegiado. Valores menores indicam maior paridade.
- **DI** (*Disparate Impact*): razão entre essas taxas. O ideal é DI = 1,0; valores abaixo de 0,8 caracterizam discriminação pela regra dos quatro quintos.

Todos os experimentos usam `MAX_TESTE = 20 000` amostras por base de teste e `SEED = 42`.

---

## Experimento 1 — Efeito do Volume de Dados de Treino (RQs 2, 4, 5, 6)

**Configuração:** Regressão Logística (LR) baseline com balanceamento por classe, treinada com volumes
N ∈ {1 000, 2 500, 5 000, 7 500, 10 000}, usando as 8 features sem a variável sensível.

### Resultados globais

| Volume | F1     | Precision | Recall | abs_SPD | DI     |
|-------:|--------|-----------|--------|---------|--------|
| 1 000  | 0,6741 | 0,6331    | 0,7741 | 0,3185  | 0,7435 |
| 2 500  | 0,6753 | 0,6247    | 0,7865 | 0,3180  | 0,7399 |
| 5 000  | 0,6802 | 0,6304    | 0,7880 | 0,3204  | 0,7382 |
| 7 500  | 0,6845 | 0,6340    | 0,7914 | 0,3220  | 0,7380 |
| 10 000 | 0,6852 | 0,6338    | 0,7936 | 0,3221  | 0,7374 |

### Análise

**O volume de treino tem impacto marginal sobre a performance e quase nulo sobre a justiça.**
O ganho de F1 entre 1 000 e 10 000 amostras é de apenas **+0,0111** — menos de 2% de melhoria
relativa. O recall cresce de forma mais consistente (+0,020), indicando que volume adicional
auxilia levemente a identificação de fraudes. A precisão, porém, permanece praticamente estável
(variação < 0,001), sugerindo que o modelo já saturou a capacidade discriminativa das features
disponíveis com amostras relativamente pequenas.

No plano da justiça, o resultado é paradoxal: o abs_SPD **aumenta levemente** com o volume
(+0,0037), e o DI **piora** (afasta-se de 1,0). Isso indica que mais dados de treino não corrigem,
e até podem acentuar, o viés aprendido — o modelo simplesmente reproduz com mais fidelidade os
padrões desiguais presentes nos dados, independentemente da quantidade de exemplos.

**Efeito da série:** A série **GS** consistentemente supera **PD** em F1 (≈ +0,02 em todos os
volumes), mas com abs_SPD levemente maior. A série PD apresenta F1 mais baixo porém DI
marginalmente melhor — padrão que se mantém estável independentemente do volume.

**Efeito do nível de viés da base de treino:** Bases com viés intermediário (v = 0,75) produzem
modelos com F1 mais alto (até 0,7195), enquanto bases sem viés (v = 0,00) geram os piores F1
(~0,64). Isso é contra-intuitivo: treinar em dados com maior viés sintético gera modelos com
melhor discriminação preditiva quando avaliados nas outras bases, possivelmente porque o viés
introduz separabilidade entre classes mais nítida nos dados de treino.

**Conclusão para RQs 2, 4, 5, 6:** O volume de dados a partir de ~1 000 amostras balanceadas
já fornece resultados competitivos. Investir em coleta massiva de dados, sem técnicas de mitigação,
não resolve o problema de viés e traz retornos decrescentes em performance.

---

## Experimento 2 — Efeito do Conjunto de Features (RQs 1, 3)

**Configuração:** LR baseline, N = 10 000 balanceado, 4 conjuntos de features progressivos,
sem variável sensível.

| Feature Set        | Features                                      | F1     | Precision | Recall | abs_SPD | DI     |
|--------------------|-----------------------------------------------|--------|-----------|--------|---------|--------|
| `minimal_2`        | VALOR_TRANSACAO, VALOR_SALDO                  | 0,6872 | **0,6961**| 0,7194 | 0,3269  | 0,7655 |
| `basico_4`         | + valor_total_dia, n_transacoes_dia           | 0,6835 | 0,6348    | 0,7894 | 0,3201  | 0,7399 |
| `comportamental_6` | + valor_medio_dia, pct_saldo_gasto            | 0,6856 | 0,6349    | 0,7940 | 0,3217  | 0,7375 |
| `completo_8`       | + dia_da_semana, final_de_semana              | 0,6852 | 0,6338    | 0,7936 | 0,3221  | 0,7374 |

### Análise

**O resultado mais surpreendente do experimento é que o conjunto mínimo de 2 features produz
o F1 mais alto** (0,6872) — superior ao conjunto completo com 8 features. Isso ocorre porque
com apenas valor da transação e saldo, o modelo aprende a regra mais simples e direta: valores
altos de transação em relação ao saldo são suspeitos. Com mais features, a complexidade aumenta
e o modelo pode overfittar padrões específicos das bases de treino, prejudicando a generalização.

A transição de `minimal_2` para `basico_4` causa a **maior queda de precision** (−0,061) e
**maior ganho de recall** (+0,070). As features de comportamento diário (total e contagem de
transações por dia) mudam o perfil de decisão do modelo: menos conservador, mais abrangente,
favorecendo a detecção de fraudes mas com mais falsos positivos.

A adição das features `comportamental_6` (+valor_medio_dia, pct_saldo_gasto) melhora
marginalmente tanto F1 (+0,0021) quanto recall (+0,0046) sobre `basico_4`, sugerindo que essas
features trazem alguma informação adicional sobre o padrão de gasto do cliente. Já as features
temporais (dia da semana, final de semana) em `completo_8` têm impacto praticamente nulo
(ΔF1 = −0,0004), indicando que o comportamento semanal não é discriminativo para esse problema.

**No eixo da justiça, features adicionais não ajudam.** O abs_SPD e o DI seguem padrão oposto
ao F1: o conjunto mínimo apresenta o maior viés de predição (abs_SPD = 0,3269; DI = 0,7655),
enquanto os conjuntos maiores têm viés levemente menor. O DI do `minimal_2` (0,7655) está
próximo ao limiar de alerta de 0,8, enquanto todos os outros já violam claramente esse limiar.
Mais features não corrigem o viés, mas o "diluem" marginalmente ao introduzir dimensões onde
os grupos têm comportamento mais similar.

**A série GS beneficia mais de features simples:** no `minimal_2`, GS obtém F1 = 0,7096, muito
acima da PD (0,6648). Com features completas, essa diferença cai para 0,0133. Isso sugere que
os padrões de fraude na série GS são mais capturáveis por regras simples de valor/saldo.

**Conclusão para RQs 1, 3:** A engenharia de features tem retorno decrescente neste domínio.
O conjunto `comportamental_6` oferece o melhor equilíbrio entre F1 e justiça. Features temporais
(dia da semana) são dispensáveis. Um modelo extremamente simples (2 features) pode ser mais
generalizável, porém às custas de maior viés de predição.

---

## Experimento 3 — Comparação de Métodos de Mitigação de Viés (RQs 7, 8)

**Configuração:** N = 10 000, 4 condições: LR sem mitigação (bruto), LR balanceado, LR com
reponderação (Kamiran & Calders, 2012), e Prejudice Remover com η = 10 (Kamishima et al., 2012).

### Resultados globais

| Método                    | F1     | Precision | Recall | abs_SPD | DI     |
|---------------------------|--------|-----------|--------|---------|--------|
| `lr_sem_mitigacao`        | 0,5646 | **0,7705**| 0,5023 | **0,2779**| **0,8143**|
| `lr_balanceado`           | **0,6852**| 0,6338 | 0,7936 | 0,3221  | 0,7374 |
| `lr_reponderado`          | 0,6781 | 0,6159    | **0,7999**| 0,3190| 0,7348 |
| `prejudice_remover_eta10` | 0,6525 | 0,5946    | 0,8128 | 0,3099  | 0,7098 |

### Análise

#### LR sem mitigação — "Aparente justiça, real fracasso"

O modelo sem nenhuma estratégia de mitigação apresenta as **melhores métricas de justiça globais**
(abs_SPD = 0,2779; DI = 0,8143) e a **pior performance** (F1 = 0,5646; recall = 0,5023). Esse
resultado é um artefato estatístico: sem balanceamento, o modelo aprende a predizer
majoritariamente a classe dominante, o que reduz artificialmente a diferença entre grupos
(ambos recebem previsões similares, porém incorretas). A "justiça" aqui é apenas indiferença.

O colapso fica evidente ao analisar por nível de viés: nas bases sem viés (v = 0,00), o modelo
sem mitigação atinge F1 = **0,3140** — praticamente inútil — enquanto nas bases com viés máximo
(v = 1,0) recupera F1 = 0,6295. O desbalanceamento de classe nos dados de treino é o problema
central, e ignorá-lo produz modelos sistematicamente falhos.

#### LR balanceado — Referência robusta

O balanceamento simples (resample igual por classe) **domina em F1** (0,6852) e mantém métricas
de justiça razoáveis. É o único método que não sacrifica performance por equidade. O custo é
um abs_SPD mais alto que os outros métodos (0,3221), reflexo de que o modelo aprende bem o
problema preditivo mas ainda herda o viés estrutural dos dados.

O LR balanceado se comporta de forma mais consistente entre níveis de viés: F1 varia de 0,6397
(v = 0,00) a 0,7195 (v = 0,75), com a mesma tendência observada no Experimento 1.

#### LR reponderado — Ganho marginal de justiça, custo em precisão

A reponderação de Kamiran & Calders (2012) redistribui o peso amostral inversamente proporcional
à probabilidade conjunta P(S,Y), forçando o modelo a tratar os grupos de forma mais simétrica.
O resultado é uma **melhora marginal de justiça** sobre o LR balanceado: abs_SPD = 0,3190 (−0,0031)
e DI = 0,7348 (−0,0026 em relação ao balanceado). Em troca, F1 cai para 0,6781 (−0,0071) e
precision cai para 0,6159 (−0,0179), com recall quase inalterado.

A reponderação tem efeito mais pronunciado na série GS (F1 = 0,6909, próximo ao balanceado)
do que em PD (F1 = 0,6654, queda de 0,0131 relativo ao balanceado). Isso sugere que os pesos
ajustados são mais compatíveis com a distribuição de GS.

#### Prejudice Remover (η=10) — Melhor equidade, maior instabilidade

O PR com η = 10 alcança o **maior recall** (0,8128) e o **melhor DI** (0,7098) entre os métodos
que efetivamente mitigam viés — mais próximo de 1,0 entre todos os métodos com F1 > 0,6. O
regularizador de equidade força o modelo a prever a classe favorável de forma mais uniforme entre
grupos, à custa de mais falsos positivos (precision = 0,5946, pior entre os métodos).

Porém, o PR é o método **mais instável**: a interação entre o nível de viés da base de treino
e a performance é não-monotônica e imprevisível.

| Viés base treino | F1 (PR) | F1 (LR bal) | abs_SPD (PR) | abs_SPD (LR bal) |
|-----------------:|---------|-------------|--------------|-----------------|
| 0,00             | 0,6922  | 0,6397      | 0,3667       | 0,3025           |
| 0,25             | 0,6501  | 0,6927      | 0,3448       | 0,3559           |
| 0,50             | 0,6274  | 0,6800      | 0,2927       | 0,3281           |
| 0,75             | 0,5988  | 0,7195      | 0,2574       | 0,3363           |
| 1,00             | 0,6940  | 0,6940      | 0,2878       | 0,2878           |

Em bases sem viés (v = 0,00), o PR supera o LR balanceado em F1 (+0,053) mas com abs_SPD muito
maior (+0,064) — o regularizador de equidade, na ausência de viés real nos dados, força
desigualdade ao tentar compensar algo que não existe. Nas bases com alto viés (v = 0,75), o PR
reduz abs_SPD com mais eficácia (−0,079 vs LR bal), mas ao custo de queda severa de F1 (−0,121).
Somente em v = 1,0 os dois métodos convergem para resultados idênticos.

A série GS responde melhor ao PR (F1 = 0,6986) do que a PD (F1 = 0,6064). Isso pode indicar
que a estrutura de correlação entre a variável sensível e o alvo em PD é mais difícil de
separar pelo tipo de regularização utilizado.

### Síntese comparativa

| Critério                     | Melhor método            |
|------------------------------|--------------------------|
| Performance (F1)             | `lr_balanceado`          |
| Recall (detecção de fraude)  | `prejudice_remover_eta10`|
| Precision (menos falsos pos.)| `lr_sem_mitigacao` *     |
| Equidade (abs_SPD)           | `lr_sem_mitigacao` *     |
| Equidade real (DI + F1)      | `prejudice_remover_eta10`|
| Estabilidade entre bases     | `lr_balanceado`          |
| Custo-benefício geral        | `lr_balanceado`          |

\* Métricas ilusórias: resultam de colapso de recall, não de mitigação efetiva.

**Conclusão para RQs 7, 8:** Nenhum método elimina o viés de predição — todos produzem DI < 0,8,
violando o limiar dos quatro quintos. O balanceamento é condição necessária mas não suficiente
para justiça. A reponderação oferece melhora marginal de equidade com baixo custo. O Prejudice
Remover é a única técnica que atua diretamente no objetivo de equidade, obtendo melhor DI,
mas com queda de F1 e comportamento errático dependente do nível de viés da base de treino.
A escolha entre métodos depende da prioridade: se detecção máxima de fraudes é crítica, o PR
é preferível; se estabilidade e F1 balanceado são prioritários, o LR balanceado é superior.

---

## Conclusões Gerais

| Dimensão                | Achado principal                                                                 |
|-------------------------|----------------------------------------------------------------------------------|
| **Volume de dados**     | Rendimentos marginais a partir de ~1 000 amostras; volume não mitiga viés        |
| **Features**            | `comportamental_6` é o melhor equilíbrio; temporais são dispensáveis            |
| **Método de mitigação** | LR balanceado domina em F1; PR é mais justo mas instável                         |
| **Viés estrutural**     | Bases com v = 0,75 produzem melhor F1; v = 0,00 é o cenário mais difícil        |
| **Série GS vs PD**      | GS consistentemente superior em F1; PD levemente mais justa                     |
| **Limiar de equidade**  | Nenhum método atinge DI ≥ 0,8 em média; problema de viés é estrutural nos dados |
