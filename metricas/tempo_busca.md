# Tempo da busca: partida e por pergunta

- Gerado em 08/11/2026 21:43 por `python pipeline/online/medir_tempo_busca.py`.
- Máquina: Windows-11-10.0.26200-SP0, Python 3.13.7; os embeddings rodam na CPU. Uma máquina e uma execução, então os valores são só uma ordem de grandeza.
- Por pergunta: 41 perguntas dos dois conjuntos, 3 rodadas cada, depois de uma chamada de aquecimento. O LLM (classificador, SQL, síntese, validação) não entra; o fluxo completo leva cerca de 29 s por pergunta.

## 1. Custos de partida (uma vez, quando o programa inicia)

| Etapa | Tempo (s) |
|---|---|
| Importar rank-bm25 (o nltk só é importado se houver radicalização) | 0,12 |
| Montar o índice BM25 do esquema (26 registros, lê o índice de palavras) | 0,00 |
| Montar o índice BM25 do glossário (1.283 trechos, lê o índice de palavras) | 0,03 |
| Para comparar: ler o JSON do índice vetorial do glossário (29.4 MB) | 0,43 |
| Para comparar: importar as bibliotecas de embeddings | 6,35 |
| Para comparar: carregar o modelo E5-base | 1,80 |
| Para comparar: ler o índice vetorial do glossário | 0,41 |

## 2. Custo por pergunta (busca de 3 trechos e de tabelas)

| Busca | Média (ms) | Mediana (ms) | Máximo (ms) |
|---|---|---|---|
| E5-base, trechos (inclui vetorizar a pergunta) | 31,17 | 30,53 | 38,73 |
| BM25, trechos | 1,14 | 0,97 | 2,97 |
| RRF, trechos (E5 e BM25 e a fusão) | 36,01 | 35,40 | 48,12 |
| E5-base, tabelas (inclui vetorizar a pergunta) | 31,65 | 30,96 | 39,81 |
| BM25, tabelas | 0,10 | 0,10 | 0,21 |
| RRF, tabelas (E5 e BM25 e a fusão) | 32,22 | 31,84 | 40,96 |

## 3. Ressalvas

- O BM25 lê um índice de palavras próprio (sem vetores), gerado no estágio offline, por isso a partida é curta; com radicalização, o nltk é importado na primeira busca (cerca de 1,7 s a mais, medido à parte).
- A primeira chamada ao E5-base depois de carregar o modelo é mais lenta; a medida por pergunta vem depois do aquecimento.
- Em uso real, o custo de partida é pago uma vez por processo; `main.py` hoje roda uma pergunta por processo, então pagaria o custo de partida a cada pergunta (ponto já registrado como melhoria possível).
