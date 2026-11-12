# Experimento metodos: recuperação

- Gerado em 09/11/2026 22:21 por `python pipeline/avaliacao/executar_experimentos.py metodos`.
- Execução `20261109-222123`, commit `d6406f7`. A ficha completa está em `execucao.json`.
- Experimento 6.1: impacto do método de busca (embeddings, BM25 e RRF, o BM25 com e sem stemmer), com o divisor das perguntas híbridas ligado e o bloqueio de pronomes, só no nível de recuperação. Modelo de embeddings: e5-base.
- Conjunto: `dados/perguntas_validacao.json`, 60 perguntas (20 quantitativas, 20 conceituais, 20 híbridas).
- Os índices têm 13 tabelas e 1283 trechos.
- Métricas calculadas com a biblioteca ranx 0.3.21: Precision@k, Recall@k, nDCG@k e MRR, com relevância binária. Nas tabelas, a tabela `companhias` fica fora do ranking e dos itens esperados (o sistema a acrescenta por regra a todas as listas), então as k tabelas medidas são as k mais parecidas com a pergunta.
- Nas perguntas híbridas, as tabelas são buscadas com a pergunta inteira e os trechos com a parte conceitual quando o divisor está ligado (configuração `divisor.ativo`); as divisões ficam em `metricas/divisoes_hibridas.json`.

## 1. Configurações

| Configuração | Modelo | Tabelas | Trechos | Divisor | Pronomes | Stemmer | Tempo (s) |
|---|---|---|---|---|---|---|---|
| Embeddings | e5-base | embeddings | embeddings | ligado | bloqueados | n/a | 22.1 |
| BM25 com stemmer | e5-base | bm25 | bm25 | ligado | bloqueados | sim | 1.1 |
| BM25 sem stemmer | e5-base | bm25 | bm25 | ligado | bloqueados | nao | 0.6 |
| RRF com stemmer | e5-base | rrf | rrf | ligado | bloqueados | sim | 3.4 |
| RRF sem stemmer | e5-base | rrf | rrf | ligado | bloqueados | nao | 3.4 |

## 2. Perguntas quantitativas (médias)

### Recuperação de tabelas

**Precision@k**

| Configuração | k = 1 | k = 3 | k = 5 |
|---|---|---|---|
| Embeddings | 0,40 | 0,27 | 0,19 |
| BM25 com stemmer | 0,25 | 0,22 | 0,15 |
| BM25 sem stemmer | 0,15 | 0,15 | 0,15 |
| RRF com stemmer | 0,45 | 0,25 | 0,17 |
| RRF sem stemmer | 0,40 | 0,25 | 0,17 |

**Recall@k**

| Configuração | k = 1 | k = 3 | k = 5 |
|---|---|---|---|
| Embeddings | 0,40 | 0,80 | 0,95 |
| BM25 com stemmer | 0,25 | 0,65 | 0,75 |
| BM25 sem stemmer | 0,15 | 0,45 | 0,75 |
| RRF com stemmer | 0,45 | 0,75 | 0,85 |
| RRF sem stemmer | 0,40 | 0,75 | 0,85 |

**nDCG@k**

| Configuração | k = 1 | k = 3 | k = 5 |
|---|---|---|---|
| Embeddings | 0,40 | 0,62 | 0,68 |
| BM25 com stemmer | 0,25 | 0,49 | 0,53 |
| BM25 sem stemmer | 0,15 | 0,33 | 0,46 |
| RRF com stemmer | 0,45 | 0,61 | 0,65 |
| RRF sem stemmer | 0,40 | 0,59 | 0,64 |

**MRR**

| Configuração | MRR |
|---|---|
| Embeddings | 0,60 |
| BM25 com stemmer | 0,49 |
| BM25 sem stemmer | 0,38 |
| RRF com stemmer | 0,61 |
| RRF sem stemmer | 0,59 |

## 3. Perguntas conceituais (médias)

### Recuperação de trechos

**Precision@k**

| Configuração | k = 1 | k = 3 | k = 5 | k = 10 | k = 20 |
|---|---|---|---|---|---|
| Embeddings | 0,70 | 0,25 | 0,15 | 0,09 | 0,04 |
| BM25 com stemmer | 0,60 | 0,23 | 0,15 | 0,09 | 0,04 |
| BM25 sem stemmer | 0,55 | 0,23 | 0,16 | 0,09 | 0,05 |
| RRF com stemmer | 0,65 | 0,27 | 0,17 | 0,09 | 0,04 |
| RRF sem stemmer | 0,65 | 0,28 | 0,18 | 0,09 | 0,05 |

**Recall@k**

| Configuração | k = 1 | k = 3 | k = 5 | k = 10 | k = 20 |
|---|---|---|---|---|---|
| Embeddings | 0,70 | 0,75 | 0,75 | 0,85 | 0,90 |
| BM25 com stemmer | 0,60 | 0,70 | 0,75 | 0,85 | 0,90 |
| BM25 sem stemmer | 0,55 | 0,70 | 0,80 | 0,85 | 0,95 |
| RRF com stemmer | 0,65 | 0,80 | 0,85 | 0,90 | 0,90 |
| RRF sem stemmer | 0,65 | 0,85 | 0,90 | 0,90 | 0,95 |

**nDCG@k**

| Configuração | k = 1 | k = 3 | k = 5 | k = 10 | k = 20 |
|---|---|---|---|---|---|
| Embeddings | 0,70 | 0,72 | 0,72 | 0,76 | 0,77 |
| BM25 com stemmer | 0,60 | 0,66 | 0,68 | 0,71 | 0,72 |
| BM25 sem stemmer | 0,55 | 0,64 | 0,68 | 0,70 | 0,72 |
| RRF com stemmer | 0,65 | 0,74 | 0,77 | 0,78 | 0,78 |
| RRF sem stemmer | 0,65 | 0,77 | 0,79 | 0,79 | 0,80 |

**MRR**

| Configuração | MRR |
|---|---|
| Embeddings | 0,74 |
| BM25 com stemmer | 0,67 |
| BM25 sem stemmer | 0,65 |
| RRF com stemmer | 0,75 |
| RRF sem stemmer | 0,76 |

## 4. Perguntas híbridas (médias)

### Recuperação de tabelas

**Precision@k**

| Configuração | k = 1 | k = 3 | k = 5 |
|---|---|---|---|
| Embeddings | 0,45 | 0,25 | 0,16 |
| BM25 com stemmer | 0,15 | 0,18 | 0,15 |
| BM25 sem stemmer | 0,05 | 0,18 | 0,13 |
| RRF com stemmer | 0,35 | 0,25 | 0,17 |
| RRF sem stemmer | 0,35 | 0,23 | 0,18 |

**Recall@k**

| Configuração | k = 1 | k = 3 | k = 5 |
|---|---|---|---|
| Embeddings | 0,45 | 0,75 | 0,80 |
| BM25 com stemmer | 0,15 | 0,55 | 0,75 |
| BM25 sem stemmer | 0,05 | 0,55 | 0,65 |
| RRF com stemmer | 0,35 | 0,75 | 0,85 |
| RRF sem stemmer | 0,35 | 0,70 | 0,90 |

**nDCG@k**

| Configuração | k = 1 | k = 3 | k = 5 |
|---|---|---|---|
| Embeddings | 0,45 | 0,61 | 0,63 |
| BM25 com stemmer | 0,15 | 0,38 | 0,47 |
| BM25 sem stemmer | 0,05 | 0,35 | 0,39 |
| RRF com stemmer | 0,35 | 0,57 | 0,61 |
| RRF sem stemmer | 0,35 | 0,54 | 0,63 |

**MRR**

| Configuração | MRR |
|---|---|
| Embeddings | 0,61 |
| BM25 com stemmer | 0,40 |
| BM25 sem stemmer | 0,35 |
| RRF com stemmer | 0,55 |
| RRF sem stemmer | 0,55 |

### Recuperação de trechos

**Precision@k**

| Configuração | k = 1 | k = 3 | k = 5 | k = 10 | k = 20 |
|---|---|---|---|---|---|
| Embeddings | 0,45 | 0,23 | 0,14 | 0,07 | 0,04 |
| BM25 com stemmer | 0,50 | 0,27 | 0,17 | 0,09 | 0,05 |
| BM25 sem stemmer | 0,55 | 0,30 | 0,19 | 0,10 | 0,05 |
| RRF com stemmer | 0,60 | 0,23 | 0,15 | 0,10 | 0,05 |
| RRF sem stemmer | 0,60 | 0,25 | 0,16 | 0,10 | 0,05 |

**Recall@k**

| Configuração | k = 1 | k = 3 | k = 5 | k = 10 | k = 20 |
|---|---|---|---|---|---|
| Embeddings | 0,45 | 0,68 | 0,68 | 0,68 | 0,82 |
| BM25 com stemmer | 0,47 | 0,78 | 0,82 | 0,88 | 0,97 |
| BM25 sem stemmer | 0,55 | 0,88 | 0,90 | 0,95 | 1,00 |
| RRF com stemmer | 0,60 | 0,68 | 0,72 | 0,93 | 0,97 |
| RRF sem stemmer | 0,60 | 0,72 | 0,78 | 0,97 | 0,97 |

**nDCG@k**

| Configuração | k = 1 | k = 3 | k = 5 | k = 10 | k = 20 |
|---|---|---|---|---|---|
| Embeddings | 0,45 | 0,57 | 0,57 | 0,57 | 0,61 |
| BM25 com stemmer | 0,50 | 0,66 | 0,68 | 0,70 | 0,73 |
| BM25 sem stemmer | 0,55 | 0,73 | 0,75 | 0,77 | 0,78 |
| RRF com stemmer | 0,60 | 0,65 | 0,67 | 0,74 | 0,75 |
| RRF sem stemmer | 0,60 | 0,67 | 0,69 | 0,76 | 0,76 |

**MRR**

| Configuração | MRR |
|---|---|
| Embeddings | 0,55 |
| BM25 com stemmer | 0,67 |
| BM25 sem stemmer | 0,70 |
| RRF com stemmer | 0,69 |
| RRF sem stemmer | 0,70 |

## 5. Métricas e escalas

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

## 6. Ressalvas

- Amostra pequena: com 20 perguntas por tipo, cada pergunta vale 5 pontos percentuais. Os números comparam as configurações nesta amostra, sem teste estatístico, e não permitem afirmar superioridade estatística de uma sobre outra.
- O critério de relevância dos trechos é a evidência literal; um trecho útil sem essa frase conta como erro.
- Nas tabelas, a regra de pontuação (melhor registro de resumo ou entidades) é a mesma em todos os métodos e a tabela `companhias` fica fora da medição; o RRF nas tabelas é uma extensão exploratória do que o MimirRAG faz com trechos.
- Com uma única evidência por pergunta, a Precision@k dos trechos tem máximo 1/k (por exemplo, 0,2 com k = 5). O nDCG vem da ranx, com o desconto 1/log2(i+1), diferente do de Järvelin da fundamentação.

## 7. Comandos

```
python pipeline/avaliacao/executar_experimentos.py metodos
```
