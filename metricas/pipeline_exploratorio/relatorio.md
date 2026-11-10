# Experimento pipeline_exploratorio: pipeline completo

- Gerado em 10/11/2026 15:16 por `python pipeline/avaliacao/executar_experimentos.py pipeline_exploratorio`.
- Execução `20261110-141908`, commit `f6f496f`. A ficha completa está em `execucao.json`.
- Experimento 6.3, execução exploratória do pipeline completo: compara dois cenários de busca com o modelo E5-base, o divisor das perguntas híbridas ligado com o bloqueio de pronomes e a passada de tabelas ideais. No cenário 1 os dois índices usam embeddings; no cenário 2 o esquema usa embeddings e o glossário usa RRF sem stemmer. Uma rodada.
- Conjunto: `dados/perguntas_validacao.json`, 60 perguntas; 1 rodada(s).
- O cache de SQL vale entre as configurações da mesma rodada e é refeito a cada rodada. A passada de tabelas ideais gera o SQL só com as tabelas de `tabelas_esperadas`.

## 1. Todas as perguntas

| Configuração | Classe certa | EX adaptado | EX estrito | EX adaptado com tabelas ideais | EX estrito com tabelas ideais | Tempo total por pergunta (s) |
|---|---|---|---|---|---|---|
| Cenário 1: embeddings (esquema e glossário) | 60/60 | 9/40 | 4/40 | 9/40 | 3/40 | 29,39 |
| Cenário 2: embeddings (esquema) + RRF (glossário) | 60/60 | 9/40 | 4/40 | 9/40 | 3/40 | 17,45 |

## 2. Perguntas quantitativas

| Configuração | Classe certa | EX adaptado | EX estrito | EX adaptado com tabelas ideais | EX estrito com tabelas ideais | Tempo total por pergunta (s) |
|---|---|---|---|---|---|---|
| Cenário 1: embeddings (esquema e glossário) | 20/20 | 7/20 | 4/20 | 7/20 | 3/20 | 35,99 |
| Cenário 2: embeddings (esquema) + RRF (glossário) | 20/20 | 7/20 | 4/20 | 7/20 | 3/20 | 14,01 |

## 3. Perguntas conceituais

| Configuração | Classe certa | Tempo total por pergunta (s) |
|---|---|---|
| Cenário 1: embeddings (esquema e glossário) | 20/20 | 16,36 |
| Cenário 2: embeddings (esquema) + RRF (glossário) | 20/20 | 18,26 |

## 4. Perguntas híbridas

| Configuração | Classe certa | EX adaptado | EX estrito | EX adaptado com tabelas ideais | EX estrito com tabelas ideais | Tempo total por pergunta (s) |
|---|---|---|---|---|---|---|
| Cenário 1: embeddings (esquema e glossário) | 20/20 | 2/20 | 0/20 | 2/20 | 0/20 | 35,81 |
| Cenário 2: embeddings (esquema) + RRF (glossário) | 20/20 | 2/20 | 0/20 | 2/20 | 0/20 | 20,08 |

## 5. Matriz de confusão do classificador

Cada linha é o tipo real da pergunta e cada coluna, o tipo que o classificador previu; a diagonal são os acertos. Com mais de uma rodada, as contagens somam as rodadas.

**Cenário 1: embeddings (esquema e glossário)**

| Real / previsto | quantitativa | conceitual | hibrida | Total |
|---|---|---|---|---|
| quantitativa | 20 | 0 | 0 | 20 |
| conceitual | 0 | 20 | 0 | 20 |
| hibrida | 0 | 0 | 20 | 20 |

**Cenário 2: embeddings (esquema) + RRF (glossário)**

| Real / previsto | quantitativa | conceitual | hibrida | Total |
|---|---|---|---|---|
| quantitativa | 20 | 0 | 0 | 20 |
| conceitual | 0 | 20 | 0 | 20 |
| hibrida | 0 | 0 | 20 | 20 |

## 6. Tempo médio por passo (s)

| Configuração | classificador | divisor | sql | sintese | validacao |
|---|---|---|---|---|---|
| Cenário 1: embeddings (esquema e glossário) | 6,20 | 2,28 | 18,35 | 7,41 | 2,76 |
| Cenário 2: embeddings (esquema) + RRF (glossário) | 6,33 | 2,28 | 0,04 | 7,41 | 2,90 |

## 7. Métricas e escalas

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

## 8. Ressalvas

- Amostra pequena: com 20 perguntas por tipo, cada pergunta vale 5 pontos percentuais. Os números comparam as configurações nesta amostra, sem teste estatístico, e não permitem afirmar superioridade estatística de uma sobre outra.
- O critério de relevância dos trechos é a evidência literal; um trecho útil sem essa frase conta como erro.
- Nas tabelas, a regra de pontuação (melhor registro de resumo ou entidades) é a mesma em todos os métodos e a tabela `companhias` fica fora da medição; o RRF nas tabelas é uma extensão exploratória do que o MimirRAG faz com trechos.
- Com uma única evidência por pergunta, a Precision@k dos trechos tem máximo 1/k (por exemplo, 0,2 com k = 5). O nDCG vem da ranx, com o desconto 1/log2(i+1), diferente do de Järvelin da fundamentação.
- O gerador de SQL e a síntese usam o LLM e não dão exatamente o mesmo resultado a cada execução; com uma rodada só, diferenças pequenas não são interpretáveis.
- A fidelidade e a relevância da resposta (RAGAS) não estão neste relatório: saem de `ragas_avaliacao.py`, que lê o `respostas.json` desta pasta e grava `ragas.csv` e `ragas.md`.

## 9. Comandos

```
python pipeline/avaliacao/executar_experimentos.py pipeline_exploratorio
```
