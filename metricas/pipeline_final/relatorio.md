# Experimento pipeline_final: pipeline completo

- Gerado em 23/11/2026 19:36 por `python pipeline/avaliacao/executar_experimentos.py pipeline_final`.
- Execução `20261123-170411`, commit `30c3252`. A ficha completa está em `execucao.json`.
- Execução final do pipeline completo, com a configuração escolhida: modelo E5-base, divisor das perguntas híbridas ligado com o bloqueio de pronomes, embeddings no índice de esquema e RRF sem stemmer no índice do glossário (a parte conceitual das híbridas também usa o RRF). Conjunto de 103 perguntas (as 60 oficiais e 43 novas), sem a passada de tabelas ideais. Três rodadas.
- Conjunto: `dados/perguntas_final.json`, 103 perguntas; 3 rodada(s).
- O cache de SQL vale entre as configurações da mesma rodada e é refeito a cada rodada.

## 1. Todas as perguntas

| Configuração | Classe certa | EX adaptado | EX estrito | Relevância do contexto | Tempo total por pergunta (s) |
|---|---|---|---|---|---|
| Embeddings (esquema) + RRF (glossário) | 103.0/103 [103 a 103] | 42.7/71 [42 a 43] | 22.0/71 [22 a 22] | 0,36 [0,36 a 0,37] | 29,48 [28,50 a 31,01] |

## 2. Perguntas quantitativas

| Configuração | Classe certa | EX adaptado | EX estrito | Tempo total por pergunta (s) |
|---|---|---|---|---|
| Embeddings (esquema) + RRF (glossário) | 35.0/35 [35 a 35] | 22.7/35 [22 a 23] | 14.0/35 [14 a 14] | 34,81 [32,69 a 38,29] |

## 3. Perguntas conceituais

| Configuração | Classe certa | Relevância do contexto | Tempo total por pergunta (s) |
|---|---|---|---|
| Embeddings (esquema) + RRF (glossário) | 32.0/32 [32 a 32] | 0,50 [0,50 a 0,50] | 16,55 [15,79 a 17,77] |

## 4. Perguntas híbridas

| Configuração | Classe certa | EX adaptado | EX estrito | Relevância do contexto | Tempo total por pergunta (s) |
|---|---|---|---|---|---|
| Embeddings (esquema) + RRF (glossário) | 36.0/36 [36 a 36] | 20.0/36 [20 a 20] | 8.0/36 [8 a 8] | 0,24 [0,24 a 0,25] | 35,78 [35,71 a 35,89] |

## 5. Matriz de confusão do classificador

Cada linha é o tipo real da pergunta e cada coluna, o tipo que o classificador previu; a diagonal são os acertos. Com mais de uma rodada, as contagens somam as rodadas.

**Embeddings (esquema) + RRF (glossário)**

| Real / previsto | quantitativa | conceitual | hibrida | Total |
|---|---|---|---|---|
| quantitativa | 105 | 0 | 0 | 105 |
| conceitual | 0 | 96 | 0 | 96 |
| hibrida | 0 | 0 | 108 | 108 |

## 6. Tempo médio por passo (s)

| Configuração | classificador | divisor | sql | sintese | validacao |
|---|---|---|---|---|---|
| Embeddings (esquema) + RRF (glossário) | 5,98 | 2,48 | 17,78 | 7,37 | 3,00 |

Com mais de uma rodada, as células mostram a média das rodadas e, entre colchetes, o mínimo e o máximo.

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
| Relevância do contexto (`relevancia_contexto`) | 0 a 1 | Fração dos trechos do glossário recuperados que o LLM julga úteis para responder a pergunta (validador do sistema, `validador.py`). Só nas conceituais e híbridas com trechos. O juiz é o mesmo modelo que escreve as respostas, e não é a `ContextRelevance` da biblioteca ragas. Maior é melhor. |
| Tempos (`tempo_*`) | segundos | Tempo da pergunta inteira e de cada passo do fluxo. Menor é melhor. |

## 8. Ressalvas

- Amostra pequena: cada pergunta vale, em pontos percentuais, 2,9 nas 35 quantitativas, 3,1 nas 32 conceituais, 2,8 nas 36 híbridas. Os números comparam as configurações nesta amostra, sem teste estatístico, e não permitem afirmar superioridade estatística de uma sobre outra.
- O critério de relevância dos trechos é a evidência literal; um trecho útil sem essa frase conta como erro.
- Nas tabelas, a regra de pontuação (melhor registro de resumo ou entidades) é a mesma em todos os métodos e a tabela `companhias` fica fora da medição; o RRF nas tabelas é uma extensão exploratória do que o MimirRAG faz com trechos.
- Com uma única evidência por pergunta, a Precision@k dos trechos tem máximo 1/k (por exemplo, 0,2 com k = 5). O nDCG vem da ranx, com o desconto 1/log2(i+1), diferente do de Järvelin da fundamentação.
- O gerador de SQL e a síntese usam o LLM e não dão exatamente o mesmo resultado a cada execução; com uma rodada só, diferenças pequenas não são interpretáveis.
- A relevância do contexto vem do mesmo LLM de 7B que escreve as respostas, com o critério "contribui, mesmo que parcialmente"; mede se os trechos recuperados ajudam a responder, e não se a resposta está certa.
- A fidelidade e a relevância da resposta (RAGAS) não estão neste relatório: saem de `ragas_avaliacao.py`, que lê o `respostas.json` desta pasta e grava `ragas.csv` e `ragas.md`.

## 9. Comandos

```
python pipeline/avaliacao/executar_experimentos.py pipeline_final
```
