# Experimento modelos: recuperação

- Gerado em 10/11/2026 09:55 por `python pipeline/avaliacao/executar_experimentos.py modelos`.
- Execução `20261110-095437`, commit `f685fc0`. A ficha completa está em `execucao.json`.
- Experimento 6.2: impacto do modelo de embeddings (MiniLM, E5-small, E5-base, BGE-M3 e o E5-base-v2, só em inglês, como controle), com busca por embeddings nos dois índices e o divisor das perguntas híbridas ligado com o bloqueio de pronomes, só no nível de recuperação.
- Conjunto: `dados/perguntas_validacao.json`, 60 perguntas (20 quantitativas, 20 conceituais, 20 híbridas).
- Os índices têm 13 tabelas e 1283 trechos.
- Métricas calculadas com a biblioteca ranx 0.3.21: Precision@k, Recall@k, nDCG@k e MRR, com relevância binária. Nas tabelas, a tabela `companhias` fica fora do ranking e dos itens esperados (o sistema a acrescenta por regra a todas as listas), então as k tabelas medidas são as k mais parecidas com a pergunta.
- Nas perguntas híbridas, as tabelas são buscadas com a pergunta inteira e os trechos com a parte conceitual quando o divisor está ligado (configuração `divisor.ativo`); as divisões ficam em `metricas/divisoes_hibridas.json`.

## 1. Configurações

| Configuração | Modelo | Tabelas | Trechos | Divisor | Pronomes | Stemmer | Tempo (s) |
|---|---|---|---|---|---|---|---|
| MiniLM | minilm | embeddings | embeddings | ligado | bloqueados | n/a | 39.9 |
| E5-small | e5-small | embeddings | embeddings | ligado | bloqueados | n/a | 4.9 |
| E5-base | e5-base | embeddings | embeddings | ligado | bloqueados | n/a | 6.6 |
| BGE-M3 | bge-m3 | embeddings | embeddings | ligado | bloqueados | n/a | 16.2 |
| E5-base-v2 (só inglês) | e5-base-v2 | embeddings | embeddings | ligado | bloqueados | n/a | 5.7 |

## 2. Perguntas quantitativas (médias)

### Recuperação de tabelas

**Precision@k**

| Configuração | k = 1 | k = 3 | k = 5 |
|---|---|---|---|
| MiniLM | 0,30 | 0,18 | 0,15 |
| E5-small | 0,35 | 0,18 | 0,15 |
| E5-base | 0,40 | 0,27 | 0,19 |
| BGE-M3 | 0,45 | 0,18 | 0,18 |
| E5-base-v2 (só inglês) | 0,40 | 0,22 | 0,15 |

**Recall@k**

| Configuração | k = 1 | k = 3 | k = 5 |
|---|---|---|---|
| MiniLM | 0,30 | 0,55 | 0,75 |
| E5-small | 0,35 | 0,55 | 0,75 |
| E5-base | 0,40 | 0,80 | 0,95 |
| BGE-M3 | 0,45 | 0,55 | 0,90 |
| E5-base-v2 (só inglês) | 0,40 | 0,65 | 0,75 |

**nDCG@k**

| Configuração | k = 1 | k = 3 | k = 5 |
|---|---|---|---|
| MiniLM | 0,30 | 0,44 | 0,53 |
| E5-small | 0,35 | 0,47 | 0,55 |
| E5-base | 0,40 | 0,62 | 0,68 |
| BGE-M3 | 0,45 | 0,51 | 0,66 |
| E5-base-v2 (só inglês) | 0,40 | 0,54 | 0,59 |

**MRR**

| Configuração | MRR |
|---|---|
| MiniLM | 0,49 |
| E5-small | 0,52 |
| E5-base | 0,60 |
| BGE-M3 | 0,60 |
| E5-base-v2 (só inglês) | 0,56 |

## 3. Perguntas conceituais (médias)

### Recuperação de trechos

**Precision@k**

| Configuração | k = 1 | k = 3 | k = 5 | k = 10 | k = 20 |
|---|---|---|---|---|---|
| MiniLM | 0,40 | 0,17 | 0,14 | 0,07 | 0,04 |
| E5-small | 0,55 | 0,23 | 0,15 | 0,08 | 0,05 |
| E5-base | 0,70 | 0,25 | 0,15 | 0,09 | 0,04 |
| BGE-M3 | 0,70 | 0,25 | 0,17 | 0,10 | 0,05 |
| E5-base-v2 (só inglês) | 0,40 | 0,20 | 0,15 | 0,08 | 0,04 |

**Recall@k**

| Configuração | k = 1 | k = 3 | k = 5 | k = 10 | k = 20 |
|---|---|---|---|---|---|
| MiniLM | 0,40 | 0,50 | 0,70 | 0,75 | 0,85 |
| E5-small | 0,55 | 0,70 | 0,75 | 0,80 | 0,95 |
| E5-base | 0,70 | 0,75 | 0,75 | 0,85 | 0,90 |
| BGE-M3 | 0,70 | 0,75 | 0,85 | 0,95 | 0,95 |
| E5-base-v2 (só inglês) | 0,40 | 0,60 | 0,75 | 0,80 | 0,80 |

**nDCG@k**

| Configuração | k = 1 | k = 3 | k = 5 | k = 10 | k = 20 |
|---|---|---|---|---|---|
| MiniLM | 0,40 | 0,46 | 0,54 | 0,55 | 0,58 |
| E5-small | 0,55 | 0,64 | 0,66 | 0,68 | 0,71 |
| E5-base | 0,70 | 0,72 | 0,72 | 0,76 | 0,77 |
| BGE-M3 | 0,70 | 0,72 | 0,76 | 0,79 | 0,79 |
| E5-base-v2 (só inglês) | 0,40 | 0,51 | 0,58 | 0,59 | 0,59 |

**MRR**

| Configuração | MRR |
|---|---|
| MiniLM | 0,50 |
| E5-small | 0,65 |
| E5-base | 0,74 |
| BGE-M3 | 0,75 |
| E5-base-v2 (só inglês) | 0,53 |

## 4. Perguntas híbridas (médias)

### Recuperação de tabelas

**Precision@k**

| Configuração | k = 1 | k = 3 | k = 5 |
|---|---|---|---|
| MiniLM | 0,30 | 0,13 | 0,14 |
| E5-small | 0,25 | 0,17 | 0,14 |
| E5-base | 0,45 | 0,25 | 0,16 |
| BGE-M3 | 0,45 | 0,25 | 0,18 |
| E5-base-v2 (só inglês) | 0,30 | 0,17 | 0,15 |

**Recall@k**

| Configuração | k = 1 | k = 3 | k = 5 |
|---|---|---|---|
| MiniLM | 0,30 | 0,40 | 0,70 |
| E5-small | 0,25 | 0,50 | 0,70 |
| E5-base | 0,45 | 0,75 | 0,80 |
| BGE-M3 | 0,45 | 0,75 | 0,90 |
| E5-base-v2 (só inglês) | 0,30 | 0,50 | 0,75 |

**nDCG@k**

| Configuração | k = 1 | k = 3 | k = 5 |
|---|---|---|---|
| MiniLM | 0,30 | 0,36 | 0,48 |
| E5-small | 0,25 | 0,39 | 0,48 |
| E5-base | 0,45 | 0,61 | 0,63 |
| BGE-M3 | 0,45 | 0,62 | 0,68 |
| E5-base-v2 (só inglês) | 0,30 | 0,41 | 0,52 |

**MRR**

| Configuração | MRR |
|---|---|
| MiniLM | 0,45 |
| E5-small | 0,45 |
| E5-base | 0,61 |
| BGE-M3 | 0,62 |
| E5-base-v2 (só inglês) | 0,48 |

### Recuperação de trechos

**Precision@k**

| Configuração | k = 1 | k = 3 | k = 5 | k = 10 | k = 20 |
|---|---|---|---|---|---|
| MiniLM | 0,40 | 0,17 | 0,10 | 0,07 | 0,04 |
| E5-small | 0,50 | 0,22 | 0,14 | 0,08 | 0,04 |
| E5-base | 0,45 | 0,23 | 0,14 | 0,07 | 0,04 |
| BGE-M3 | 0,20 | 0,18 | 0,13 | 0,09 | 0,05 |
| E5-base-v2 (só inglês) | 0,30 | 0,15 | 0,13 | 0,07 | 0,04 |

**Recall@k**

| Configuração | k = 1 | k = 3 | k = 5 | k = 10 | k = 20 |
|---|---|---|---|---|---|
| MiniLM | 0,40 | 0,47 | 0,47 | 0,62 | 0,78 |
| E5-small | 0,50 | 0,62 | 0,68 | 0,78 | 0,78 |
| E5-base | 0,45 | 0,68 | 0,68 | 0,68 | 0,82 |
| BGE-M3 | 0,20 | 0,53 | 0,62 | 0,82 | 0,93 |
| E5-base-v2 (só inglês) | 0,30 | 0,42 | 0,62 | 0,62 | 0,78 |

**nDCG@k**

| Configuração | k = 1 | k = 3 | k = 5 | k = 10 | k = 20 |
|---|---|---|---|---|---|
| MiniLM | 0,40 | 0,45 | 0,45 | 0,50 | 0,54 |
| E5-small | 0,50 | 0,58 | 0,60 | 0,63 | 0,63 |
| E5-base | 0,45 | 0,57 | 0,57 | 0,57 | 0,61 |
| BGE-M3 | 0,20 | 0,40 | 0,45 | 0,51 | 0,54 |
| E5-base-v2 (só inglês) | 0,30 | 0,37 | 0,45 | 0,45 | 0,49 |

**MRR**

| Configuração | MRR |
|---|---|
| MiniLM | 0,48 |
| E5-small | 0,60 |
| E5-base | 0,55 |
| BGE-M3 | 0,43 |
| E5-base-v2 (só inglês) | 0,42 |

## 5. Custo dos modelos

| Modelo | Idioma | Dimensões | Limite de tokens | Índices (MB) |
|---|---|---|---|---|
| bge-m3 | multilingue | 1024 | 8192 | 44.3 |
| e5-base | multilingue | 768 | 512 | 33.3 |
| e5-base-v2 | ingles | 768 | 512 | 33.3 |
| e5-small | multilingue | 384 | 512 | 17.0 |
| minilm | multilingue | 384 | 128 | 16.8 |

O tempo de indexação é impresso por `rodar_offline.py`; o tempo de cada configuração aparece na tabela de configurações.

## 6. Métricas e escalas

| Métrica | Escala | O que mede |
|---|---|---|
| Precision@k (`tab_prec@k`, `glo_prec@k`) | 0 a 1 | Fração dos k primeiros itens que são relevantes (tabelas ou trechos). Divide sempre por k. Com uma só evidência por pergunta, o máximo nos trechos é 1/k. Maior é melhor. |
| Recall@k (`tab_recall@k`, `glo_recall@k`) | 0 a 1 | Fração dos itens relevantes que aparece entre os k primeiros. Maior é melhor. |
| nDCG@k (`tab_ndcg@k`, `glo_ndcg@k`) | 0 a 1 | Considera a ordem da lista, dando mais peso a um acerto no topo; relevância binária. A ranx usa o desconto 1/log2(i+1), e não o de Järvelin da fundamentação. Maior é melhor. |
| MRR (`tab_rr`, `glo_rr`) | 0 a 1 | Média do inverso da posição do primeiro item relevante. Maior é melhor. |
| Classe certa (`classe_certa`) | 0 ou 1 | O classificador acertou o tipo da pergunta. |
| EX estrito (`ex_estrito`) | 0 ou 1 | O resultado do SQL gerado é igual ao do `sql_referencia` (ordem das colunas e das linhas ignorada, números com 2 casas). Só nas perguntas com SQL. |
| EX adaptado (`ex_adaptado`) | 0 ou 1 | Vale se o EX estrito vale ou se o resultado traz todas as linhas da referência (a coluna `DENOM_CIA` é tolerada) e as linhas a mais são de companhias cujo `DENOM_CIA` contém a palavra que identifica a companhia dita na pergunta. Adaptação do EX de Zhong et al. (2020), sem artigo; mais permissiva que o EX estrito. |
| Tabelas ideais (`ideal_ex_*`) | 0 ou 1 | EX do SQL gerado quando o gerador recebe as tabelas certas (`tabelas_esperadas`) em vez das da busca. É um teto, e a diferença para o EX normal mostra quanto da perda vem da busca. |
| Tempos (`tempo_*`) | segundos | Tempo da pergunta inteira e de cada passo do fluxo. Menor é melhor. |

## 7. Ressalvas

- Amostra pequena: com 20 perguntas por tipo, cada pergunta vale 5 pontos percentuais. Os números comparam as configurações nesta amostra, sem teste estatístico, e não permitem afirmar superioridade estatística de uma sobre outra.
- O critério de relevância dos trechos é a evidência literal; um trecho útil sem essa frase conta como erro.
- Nas tabelas, a regra de pontuação (melhor registro de resumo ou entidades) é a mesma em todos os métodos e a tabela `companhias` fica fora da medição; o RRF nas tabelas é uma extensão exploratória do que o MimirRAG faz com trechos.
- Com uma única evidência por pergunta, a Precision@k dos trechos tem máximo 1/k (por exemplo, 0,2 com k = 5). O nDCG vem da ranx, com o desconto 1/log2(i+1), diferente do de Järvelin da fundamentação.

## 8. Comandos

```
python pipeline/avaliacao/executar_experimentos.py modelos
```
