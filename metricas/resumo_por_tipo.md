# Resumo das métricas de recuperação por tipo de pergunta

- Gerado em 06/11/2026 19:13 por `python pipeline/online/resumir_por_tipo.py`, a partir dos CSVs de `metricas/`.
- Contagens "x/n": x perguntas com acerto entre n. Val. é o conjunto de validação (`perguntas_validacao.json`), Teste é `perguntas_dfp.json`, Total soma os dois. Os dois conjuntos são pequenos e não totalmente independentes do ajuste do E5-base (o teste o favorece mais).
- Todas as tabelas no top-k: todas as tabelas necessárias estão entre as k escolhidas (mais as referenciadas por chave estrangeira). Hit@k: pelo menos um trecho relevante entre os k primeiros. Ambos: tabelas completas e trecho relevante ao mesmo tempo. MRR: média de 1/posição do primeiro item certo.
- Métodos: BM25 sem stemmer com k1 = 2 e b = 0,5; RRF com k = 60; Por índice é a política padrão de `busca_roteada.py` (tabelas com E5 e trechos com BM25, para todos os tipos; escolhida depois de ver estes mesmos dados, então não é confirmação independente) e Política anterior é a que usava RRF nas híbridas, ambas com o tipo correto de cada pergunta. No piso aleatório, o número é o valor esperado de um sorteio.

## Perguntas quantitativas (SQL puro): tabelas

| Método | Todas as tabelas no top-3 (Val.) | Todas as tabelas no top-3 (Teste) | Todas as tabelas no top-3 (Total) | Todas as tabelas no top-5 (Val.) | Todas as tabelas no top-5 (Teste) | Todas as tabelas no top-5 (Total) | MRR (média) |
|---|---|---|---|---|---|---|---|
| E5-base | 5/7 | 11/14 | 16/21 | 6/7 | 13/14 | 19/21 | 0,59 |
| BM25 | 4/7 | 5/14 | 9/21 | 6/7 | 12/14 | 18/21 | 0,44 |
| RRF | 4/7 | 10/14 | 14/21 | 7/7 | 14/14 | 21/21 | 0,61 |
| Por índice (E5 tabelas, BM25 trechos) | 5/7 | 11/14 | 16/21 | 6/7 | 13/14 | 19/21 | 0,59 |
| Política anterior (híbrida com RRF) | 5/7 | 11/14 | 16/21 | 6/7 | 13/14 | 19/21 | 0,59 |
| Piso aleatório | 1,62/7 | 4,00/14 | 5,62/21 | 2,69/7 | 6,00/14 | 8,69/21 | 0,35 |

## Perguntas conceituais (glossário puro): trechos

| Método | Hit@3 (Val.) | Hit@3 (Teste) | Hit@3 (Total) | Hit@10 (Val.) | Hit@10 (Teste) | Hit@10 (Total) | MRR (média) |
|---|---|---|---|---|---|---|---|
| E5-base | 1/3 | 4/7 | 5/10 | 1/3 | 5/7 | 6/10 | 0,46 |
| BM25 | 1/3 | 5/7 | 6/10 | 1/3 | 5/7 | 6/10 | 0,50 |
| RRF | 1/3 | 4/7 | 5/10 | 1/3 | 6/7 | 7/10 | 0,51 |
| Por índice (E5 tabelas, BM25 trechos) | 1/3 | 5/7 | 6/10 | 1/3 | 5/7 | 6/10 | 0,50 |
| Política anterior (híbrida com RRF) | 1/3 | 5/7 | 6/10 | 1/3 | 5/7 | 6/10 | 0,50 |
| Piso aleatório | 0,01/3 | 0,02/7 | 0,03/10 | 0,02/3 | 0,07/7 | 0,09/10 | 0,01 |

## Perguntas híbridas: tabelas

| Método | Todas as tabelas no top-3 (Val.) | Todas as tabelas no top-3 (Teste) | Todas as tabelas no top-3 (Total) | Todas as tabelas no top-5 (Val.) | Todas as tabelas no top-5 (Teste) | Todas as tabelas no top-5 (Total) | MRR (média) |
|---|---|---|---|---|---|---|---|
| E5-base | 2/3 | 5/7 | 7/10 | 2/3 | 5/7 | 7/10 | 0,54 |
| BM25 | 3/3 | 4/7 | 7/10 | 3/3 | 4/7 | 7/10 | 0,66 |
| RRF | 2/3 | 3/7 | 5/10 | 3/3 | 4/7 | 7/10 | 0,52 |
| Por índice (E5 tabelas, BM25 trechos) | 2/3 | 5/7 | 7/10 | 2/3 | 5/7 | 7/10 | 0,54 |
| Política anterior (híbrida com RRF) | 2/3 | 3/7 | 5/10 | 3/3 | 4/7 | 7/10 | 0,52 |
| Piso aleatório | 0,69/3 | 1,62/7 | 2,31/10 | 1,15/3 | 2,69/7 | 3,85/10 | 0,35 |

## Perguntas híbridas: trechos

| Método | Hit@3 (Val.) | Hit@3 (Teste) | Hit@3 (Total) | Hit@10 (Val.) | Hit@10 (Teste) | Hit@10 (Total) | MRR (média) |
|---|---|---|---|---|---|---|---|
| E5-base | 0/3 | 2/7 | 2/10 | 2/3 | 2/7 | 4/10 | 0,18 |
| BM25 | 0/3 | 2/7 | 2/10 | 2/3 | 5/7 | 7/10 | 0,25 |
| RRF | 0/3 | 2/7 | 2/10 | 2/3 | 4/7 | 6/10 | 0,22 |
| Por índice (E5 tabelas, BM25 trechos) | 0/3 | 2/7 | 2/10 | 2/3 | 5/7 | 7/10 | 0,25 |
| Política anterior (híbrida com RRF) | 0/3 | 2/7 | 2/10 | 2/3 | 4/7 | 6/10 | 0,22 |
| Piso aleatório | 0,01/3 | 0,02/7 | 0,03/10 | 0,02/3 | 0,07/7 | 0,09/10 | 0,01 |

## Perguntas híbridas: tabelas e trecho certos juntos

| Método | Ambos no top-3 (Val.) | Ambos no top-3 (Teste) | Ambos no top-3 (Total) | Ambos no top-5 (Val.) | Ambos no top-5 (Teste) | Ambos no top-5 (Total) |
|---|---|---|---|---|---|---|
| E5-base | 0/3 | 2/7 | 2/10 | 0/3 | 2/7 | 2/10 |
| BM25 | 0/3 | 2/7 | 2/10 | 1/3 | 3/7 | 4/10 |
| RRF | 0/3 | 2/7 | 2/10 | 2/3 | 2/7 | 4/10 |
| Por índice (E5 tabelas, BM25 trechos) | 0/3 | 2/7 | 2/10 | 1/3 | 4/7 | 5/10 |
| Política anterior (híbrida com RRF) | 0/3 | 2/7 | 2/10 | 2/3 | 2/7 | 4/10 |
| Piso aleatório | 0,00/3 | 0,01/7 | 0,01/10 | 0,00/3 | 0,01/7 | 0,02/10 |
