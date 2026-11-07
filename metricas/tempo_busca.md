# Tempo da busca: partida e por pergunta

- Gerado em 06/11/2026 19:12 por `python pipeline/online/medir_tempo_busca.py`.
- Máquina: Windows-11-10.0.26200-SP0, Python 3.13.7; os embeddings rodam na CPU. Uma máquina e uma execução, então os valores são só uma ordem de grandeza.
- Por pergunta: 41 perguntas dos dois conjuntos, 3 rodadas cada, depois de uma chamada de aquecimento. O LLM (classificador, SQL, síntese, validação) não entra; o fluxo completo leva cerca de 29 s por pergunta.

## 1. Custos de partida (uma vez, quando o programa inicia)

| Etapa | Tempo (s) |
|---|---|
| Importar nltk e rank-bm25 | 1,94 |
| Ler a lista de stopwords do NLTK | 0,01 |
| Ler o JSON do índice do glossário (37 MB, usado pelo BM25 só pelo texto) | 0,50 |
| Montar o índice BM25 do esquema (26 registros) | 0,04 |
| Montar o índice BM25 do glossário (1.634 trechos, inclui ler o JSON) | 0,62 |
| Para comparar: importar as bibliotecas de embeddings | 5,18 |
| Para comparar: carregar o modelo E5-base | 1,73 |
| Para comparar: ler o índice vetorial do glossário | 0,56 |

## 2. Custo por pergunta (busca de 3 trechos e de tabelas)

| Busca | Média (ms) | Mediana (ms) | Máximo (ms) |
|---|---|---|---|
| E5-base, trechos (inclui vetorizar a pergunta) | 31,70 | 31,28 | 40,75 |
| BM25, trechos | 1,37 | 1,23 | 3,42 |
| RRF, trechos (E5 e BM25 e a fusão) | 36,85 | 36,44 | 46,62 |
| E5-base, tabelas (inclui vetorizar a pergunta) | 31,34 | 30,84 | 39,00 |
| BM25, tabelas | 0,13 | 0,13 | 0,27 |
| RRF, tabelas (E5 e BM25 e a fusão) | 31,45 | 31,11 | 38,75 |

## 3. Ressalvas

- O BM25 lê o mesmo JSON dos vetores (37 MB) só para usar o texto; guardar um arquivo só com o texto dos trechos reduziria esse custo, se ele pesar.
- A primeira chamada ao E5-base depois de carregar o modelo é mais lenta; a medida por pergunta vem depois do aquecimento.
- Em uso real, o custo de partida é pago uma vez por processo; `main.py` hoje roda uma pergunta por processo, então pagaria o custo de partida a cada pergunta (ponto já registrado como melhoria possível).
