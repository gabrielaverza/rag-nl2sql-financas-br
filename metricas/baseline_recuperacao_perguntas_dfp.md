# Baseline BM25 e fusão RRF contra E5-base: recuperação (perguntas_dfp.json)

- Gerado em 06/11/2026 19:13 por `python pipeline/online/avaliar_baseline.py dados/perguntas_dfp.json`.
- Perguntas: 28 no arquivo; 21 com tabelas esperadas, 14 com evidência, 7 híbridas com os dois.
- Bibliotecas: rank-bm25 0.2.2, nltk 3.10.4, sentence-transformers 5.6.0, numpy 2.4.6.
- BM25: Okapi BM25 da rank_bm25, sem ajuste nos dados. Parâmetros: BM25 sem stemmer: k1 = 2,00, b = 0,50; BM25 com stemmer (Snowball): k1 = 2,00, b = 0,50; BM25 sem stemmer (parâmetros da biblioteca): k1 = 1,50, b = 0,75; epsilon = 0,25 (padrão da biblioteca). Os valores k1 = 2 e b = 0,5 são a combinação comum citada por Robertson e Zaragoza (2009, p. 31); os da biblioteca (k1 = 1,5 e b = 0,75) ficam dentro das faixas 0,5 < b < 0,8 e 1,2 < k1 < 2 (p. 30) e entram só como verificação de sensibilidade.
- RRF: fusão por posição dos rankings do E5-base e do BM25 sem stemmer; a nota de cada item é a soma de 1 / (k + posição), com k = 60, o valor do artigo original (Cormack, Clarke e Büttcher, 2009), sem ajuste. No glossário reproduz a busca híbrida do MimirRAG; no esquema é uma extensão exploratória (as tabelas fundidas recebem depois a expansão por chave estrangeira).
- Por índice: a política padrão de busca_roteada.py, com as tabelas buscadas pelo E5-base e os trechos pelo BM25 (BM25 sem stemmer, k1 = 2 e b = 0,5), para todos os tipos de pergunta, aplicada com o tipo correto de cada pergunta (sem erro do classificador). Foi escolhida pela autora depois de ver os resultados dos métodos nestes mesmos conjuntos, então esta linha NÃO é uma confirmação independente: só a avaliação em perguntas novas pode confirmá-la.
- Por tipo (híbrida com RRF): a política anterior de busca_roteada.py (quantitativa: tabelas com E5; conceitual: trechos com BM25; híbrida: tabelas e trechos com RRF), mantida só para comparação.
- Pré-processamento do BM25: minúsculas, palavras (letras e dígitos), stopwords do português do NLTK, stemmer Snowball do português só na variante com stemmer, acentos removidos no fim. O E5-base recebe a pergunta com o prefixo `query: `.
- Premissas iguais em todos os métodos: mesmas perguntas, mesmos registros indexados, pontuação da tabela pelo melhor registro de resumo ou entidades, expansão por chave estrangeira, mesmos k.
- Índice de esquema no BM25 (só resumo e entidades): 26 registros, 103 palavras, 4 em mais da metade dos registros (IDF negativo trocado por epsilon vezes o IDF médio = 0,420; IDF médio = 1,678), na variante sem stemmer.
- Índice do glossário no BM25: 1634 trechos, 5418 palavras, 0 em mais da metade dos trechos, na variante sem stemmer.

## 1. Tabelas (esquema, 21 perguntas)

### 1.1 Com a tabela companhias entre as esperadas (como no pipeline)

| Método | Recall@1 | Recall@3 | Recall@5 | Todas@1 | Todas@3 | Todas@5 | MRR |
|---|---|---|---|---|---|---|---|
| E5-base (embeddings) | 0,60 | 0,88 | 0,93 | 0,29 | 0,76 | 0,86 | 0,60 |
| BM25 sem stemmer | 0,57 | 0,67 | 0,83 | 0,24 | 0,43 | 0,76 | 0,51 |
| BM25 com stemmer (Snowball) | 0,55 | 0,79 | 0,81 | 0,19 | 0,67 | 0,71 | 0,47 |
| BM25 sem stemmer (parâmetros da biblioteca) | 0,60 | 0,67 | 0,83 | 0,29 | 0,43 | 0,76 | 0,51 |
| RRF (E5 + BM25) | 0,60 | 0,76 | 0,90 | 0,29 | 0,62 | 0,86 | 0,57 |
| Por índice (tabelas E5, trechos BM25) | 0,60 | 0,88 | 0,93 | 0,29 | 0,76 | 0,86 | 0,60 |
| Por tipo (híbrida com RRF) | 0,60 | 0,81 | 0,88 | 0,29 | 0,67 | 0,81 | 0,58 |
| Piso aleatório | 0,52 | 0,60 | 0,68 | 0,12 | 0,27 | 0,41 | 0,35 |

Tamanho médio da lista (as k escolhidas mais as referenciadas por chave estrangeira): E5-base (embeddings): 3,8 (k = 3) e 5,8 (k = 5), BM25 sem stemmer: 3,8 (k = 3) e 5,8 (k = 5), BM25 com stemmer (Snowball): 3,9 (k = 3) e 5,8 (k = 5), BM25 sem stemmer (parâmetros da biblioteca): 3,8 (k = 3) e 5,8 (k = 5), RRF (E5 + BM25): 3,8 (k = 3) e 5,8 (k = 5), Por índice (tabelas E5, trechos BM25): 3,8 (k = 3) e 5,8 (k = 5), Por tipo (híbrida com RRF): 3,8 (k = 3) e 5,8 (k = 5).

### 1.2 Sem companhias (só o acerto da demonstração, 20 perguntas)

| Método | Recall@1 | Recall@3 | Recall@5 | Todas@1 | Todas@3 | Todas@5 | MRR |
|---|---|---|---|---|---|---|---|
| E5-base (embeddings) | 0,25 | 0,75 | 0,85 | 0,25 | 0,75 | 0,85 | 0,53 |
| BM25 sem stemmer | 0,20 | 0,40 | 0,75 | 0,20 | 0,40 | 0,75 | 0,41 |
| BM25 com stemmer (Snowball) | 0,15 | 0,65 | 0,70 | 0,15 | 0,65 | 0,70 | 0,44 |
| BM25 sem stemmer (parâmetros da biblioteca) | 0,25 | 0,40 | 0,75 | 0,25 | 0,40 | 0,75 | 0,43 |
| RRF (E5 + BM25) | 0,25 | 0,60 | 0,85 | 0,25 | 0,60 | 0,85 | 0,49 |
| Por índice (tabelas E5, trechos BM25) | 0,25 | 0,75 | 0,85 | 0,25 | 0,75 | 0,85 | 0,53 |
| Por tipo (híbrida com RRF) | 0,25 | 0,65 | 0,80 | 0,25 | 0,65 | 0,80 | 0,50 |
| Piso aleatório | 0,08 | 0,23 | 0,38 | 0,08 | 0,23 | 0,38 | 0,24 |

Perguntas sem nenhuma palavra em comum com os registros de esquema (BM25, todas as tabelas empatadas em zero): BM25 sem stemmer: 0 de 21; BM25 com stemmer (Snowball): 0 de 21; BM25 sem stemmer (parâmetros da biblioteca): 0 de 21.

## 2. Trechos dos normativos (glossário, 14 perguntas)

| Método | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | MRR |
|---|---|---|---|---|---|---|
| E5-base (embeddings) | 0,36 | 0,43 | 0,43 | 0,50 | 0,64 | 0,41 |
| BM25 sem stemmer | 0,36 | 0,50 | 0,71 | 0,71 | 0,86 | 0,48 |
| BM25 com stemmer (Snowball) | 0,29 | 0,36 | 0,43 | 0,57 | 0,71 | 0,37 |
| BM25 sem stemmer (parâmetros da biblioteca) | 0,29 | 0,50 | 0,57 | 0,71 | 0,86 | 0,43 |
| RRF (E5 + BM25) | 0,36 | 0,43 | 0,50 | 0,71 | 0,71 | 0,44 |
| Por índice (tabelas E5, trechos BM25) | 0,36 | 0,50 | 0,71 | 0,71 | 0,86 | 0,48 |
| Por tipo (híbrida com RRF) | 0,36 | 0,50 | 0,50 | 0,64 | 0,71 | 0,45 |
| Piso aleatório | 0,00 | 0,00 | 0,01 | 0,01 | 0,02 | 0,01 |

| Método | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Recall@20 |
|---|---|---|---|---|---|
| E5-base (embeddings) | 0,30 | 0,41 | 0,43 | 0,50 | 0,60 |
| BM25 sem stemmer | 0,30 | 0,48 | 0,66 | 0,66 | 0,80 |
| BM25 com stemmer (Snowball) | 0,23 | 0,34 | 0,43 | 0,57 | 0,71 |
| BM25 sem stemmer (parâmetros da biblioteca) | 0,23 | 0,48 | 0,52 | 0,66 | 0,80 |
| RRF (E5 + BM25) | 0,30 | 0,41 | 0,50 | 0,66 | 0,66 |
| Por índice (tabelas E5, trechos BM25) | 0,30 | 0,48 | 0,66 | 0,66 | 0,80 |
| Por tipo (híbrida com RRF) | 0,30 | 0,48 | 0,50 | 0,59 | 0,66 |
| Piso aleatório | 0,00 | 0,00 | 0,00 | 0,01 | 0,01 |

| Método | nDCG@3 | nDCG@5 | nDCG@10 | Precision@1 | Precision@3 | Precision@5 |
|---|---|---|---|---|---|---|
| E5-base (embeddings) | 0,40 | 0,40 | 0,42 | 0,36 | 0,19 | 0,13 |
| BM25 sem stemmer | 0,47 | 0,55 | 0,55 | 0,36 | 0,21 | 0,19 |
| BM25 com stemmer (Snowball) | 0,33 | 0,37 | 0,42 | 0,29 | 0,17 | 0,13 |
| BM25 sem stemmer (parâmetros da biblioteca) | 0,47 | 0,49 | 0,54 | 0,29 | 0,21 | 0,16 |
| RRF (E5 + BM25) | 0,40 | 0,44 | 0,50 | 0,36 | 0,19 | 0,14 |
| Por índice (tabelas E5, trechos BM25) | 0,47 | 0,55 | 0,55 | 0,36 | 0,21 | 0,19 |
| Por tipo (híbrida com RRF) | 0,47 | 0,47 | 0,51 | 0,36 | 0,21 | 0,14 |
| Piso aleatório | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 |

Perguntas sem nenhuma palavra em comum com os trechos (BM25): BM25 sem stemmer: 0 de 14; BM25 com stemmer (Snowball): 0 de 14; BM25 sem stemmer (parâmetros da biblioteca): 0 de 14.

## 3. Perguntas híbridas (7 perguntas): contagem de acertos

| Método | Tabelas completas@3 | Trecho@3 | Ambos@3 | Ambos (sem companhias)@3 | Tabelas completas@5 | Trecho@5 | Ambos@5 | Ambos (sem companhias)@5 |
|---|---|---|---|---|---|---|---|---|
| E5-base (embeddings) | 5/7 | 2/7 | 2/7 | 2/7 | 5/7 | 2/7 | 2/7 | 2/7 |
| BM25 sem stemmer | 4/7 | 2/7 | 2/7 | 2/7 | 4/7 | 5/7 | 3/7 | 3/7 |
| BM25 com stemmer (Snowball) | 4/7 | 2/7 | 2/7 | 2/7 | 4/7 | 2/7 | 2/7 | 2/7 |
| BM25 sem stemmer (parâmetros da biblioteca) | 4/7 | 2/7 | 2/7 | 2/7 | 4/7 | 3/7 | 2/7 | 2/7 |
| RRF (E5 + BM25) | 3/7 | 2/7 | 2/7 | 2/7 | 4/7 | 2/7 | 2/7 | 2/7 |
| Por índice (tabelas E5, trechos BM25) | 5/7 | 2/7 | 2/7 | 2/7 | 5/7 | 5/7 | 4/7 | 4/7 |
| Por tipo (híbrida com RRF) | 3/7 | 2/7 | 2/7 | 2/7 | 4/7 | 2/7 | 2/7 | 2/7 |
| Piso aleatório | 1,62/7 | 0,02/7 | 0,01/7 | 0,01/7 | 2,69/7 | 0,04/7 | 0,01/7 | 0,01/7 |

No piso aleatório, o valor é o número esperado de perguntas (produto das probabilidades dos dois índices, sorteados de forma independente).

## 4. Comparação pareada por pergunta

Diferença média (primeiro método menos o segundo), intervalo de 95% por bootstrap sobre as perguntas e vitórias, derrotas e empates do primeiro método.

| Comparação | Métrica | n | Diferença média | IC 95% | Vitórias/derrotas/empates |
|---|---|---|---|---|---|
| E5-base (embeddings) contra BM25 sem stemmer | Tabelas: todas no top-3 | 21 | 0,33 | [0,05; 0,62] | 9/2/10 |
| E5-base (embeddings) contra BM25 sem stemmer | Demonstração no top-3 (sem companhias) | 20 | 0,35 | [0,05; 0,65] | 9/2/9 |
| E5-base (embeddings) contra BM25 sem stemmer | Tabelas: MRR | 21 | 0,09 | [-0,07; 0,25] | 10/4/7 |
| E5-base (embeddings) contra BM25 sem stemmer | Trechos: Hit@3 | 14 | -0,07 | [-0,21; 0,00] | 0/1/13 |
| E5-base (embeddings) contra BM25 sem stemmer | Trechos: Hit@5 | 14 | -0,29 | [-0,50; -0,07] | 0/4/10 |
| E5-base (embeddings) contra BM25 sem stemmer | Trechos: MRR | 14 | -0,07 | [-0,24; 0,07] | 3/6/5 |
| E5-base (embeddings) contra BM25 sem stemmer | Trechos: nDCG@10 | 14 | -0,12 | [-0,30; 0,02] | 1/5/8 |
| E5-base (embeddings) contra BM25 sem stemmer | Híbridas: tabelas e trecho no top-3 | 7 | 0,00 | [0,00; 0,00] | 0/0/7 |
| E5-base (embeddings) contra BM25 sem stemmer | Híbridas: demonstração e trecho no top-3 | 7 | 0,00 | [0,00; 0,00] | 0/0/7 |
| RRF (E5 + BM25) contra E5-base (embeddings) | Tabelas: todas no top-3 | 21 | -0,14 | [-0,33; 0,05] | 1/4/16 |
| RRF (E5 + BM25) contra E5-base (embeddings) | Demonstração no top-3 (sem companhias) | 20 | -0,15 | [-0,35; 0,05] | 1/4/15 |
| RRF (E5 + BM25) contra E5-base (embeddings) | Tabelas: MRR | 21 | -0,03 | [-0,12; 0,06] | 3/6/12 |
| RRF (E5 + BM25) contra E5-base (embeddings) | Trechos: Hit@3 | 14 | 0,00 | [0,00; 0,00] | 0/0/14 |
| RRF (E5 + BM25) contra E5-base (embeddings) | Trechos: Hit@5 | 14 | 0,07 | [0,00; 0,21] | 1/0/13 |
| RRF (E5 + BM25) contra E5-base (embeddings) | Trechos: MRR | 14 | 0,03 | [0,01; 0,06] | 7/1/6 |
| RRF (E5 + BM25) contra E5-base (embeddings) | Trechos: nDCG@10 | 14 | 0,07 | [0,01; 0,16] | 5/0/9 |
| RRF (E5 + BM25) contra E5-base (embeddings) | Híbridas: tabelas e trecho no top-3 | 7 | 0,00 | [0,00; 0,00] | 0/0/7 |
| RRF (E5 + BM25) contra E5-base (embeddings) | Híbridas: demonstração e trecho no top-3 | 7 | 0,00 | [0,00; 0,00] | 0/0/7 |
| RRF (E5 + BM25) contra BM25 sem stemmer | Tabelas: todas no top-3 | 21 | 0,19 | [0,00; 0,38] | 5/1/15 |
| RRF (E5 + BM25) contra BM25 sem stemmer | Demonstração no top-3 (sem companhias) | 20 | 0,20 | [0,00; 0,45] | 5/1/14 |
| RRF (E5 + BM25) contra BM25 sem stemmer | Tabelas: MRR | 21 | 0,06 | [-0,08; 0,20] | 9/4/8 |
| RRF (E5 + BM25) contra BM25 sem stemmer | Trechos: Hit@3 | 14 | -0,07 | [-0,21; 0,00] | 0/1/13 |
| RRF (E5 + BM25) contra BM25 sem stemmer | Trechos: Hit@5 | 14 | -0,21 | [-0,43; 0,00] | 0/3/11 |
| RRF (E5 + BM25) contra BM25 sem stemmer | Trechos: MRR | 14 | -0,04 | [-0,18; 0,08] | 3/6/5 |
| RRF (E5 + BM25) contra BM25 sem stemmer | Trechos: nDCG@10 | 14 | -0,05 | [-0,16; 0,05] | 1/4/9 |
| RRF (E5 + BM25) contra BM25 sem stemmer | Híbridas: tabelas e trecho no top-3 | 7 | 0,00 | [0,00; 0,00] | 0/0/7 |
| RRF (E5 + BM25) contra BM25 sem stemmer | Híbridas: demonstração e trecho no top-3 | 7 | 0,00 | [0,00; 0,00] | 0/0/7 |
| BM25 com stemmer (Snowball) contra BM25 sem stemmer | Tabelas: todas no top-3 | 21 | 0,24 | [0,10; 0,43] | 5/0/16 |
| BM25 com stemmer (Snowball) contra BM25 sem stemmer | Demonstração no top-3 (sem companhias) | 20 | 0,25 | [0,10; 0,45] | 5/0/15 |
| BM25 com stemmer (Snowball) contra BM25 sem stemmer | Tabelas: MRR | 21 | -0,04 | [-0,16; 0,05] | 5/6/10 |
| BM25 com stemmer (Snowball) contra BM25 sem stemmer | Trechos: Hit@3 | 14 | -0,14 | [-0,36; 0,00] | 0/2/12 |
| BM25 com stemmer (Snowball) contra BM25 sem stemmer | Trechos: Hit@5 | 14 | -0,29 | [-0,50; -0,07] | 0/4/10 |
| BM25 com stemmer (Snowball) contra BM25 sem stemmer | Trechos: MRR | 14 | -0,11 | [-0,25; -0,02] | 1/8/5 |
| BM25 com stemmer (Snowball) contra BM25 sem stemmer | Trechos: nDCG@10 | 14 | -0,13 | [-0,25; -0,03] | 0/5/9 |
| BM25 com stemmer (Snowball) contra BM25 sem stemmer | Híbridas: tabelas e trecho no top-3 | 7 | 0,00 | [0,00; 0,00] | 0/0/7 |
| BM25 com stemmer (Snowball) contra BM25 sem stemmer | Híbridas: demonstração e trecho no top-3 | 7 | 0,00 | [0,00; 0,00] | 0/0/7 |
| BM25 sem stemmer (parâmetros da biblioteca) contra BM25 sem stemmer | Tabelas: todas no top-3 | 21 | 0,00 | [0,00; 0,00] | 0/0/21 |
| BM25 sem stemmer (parâmetros da biblioteca) contra BM25 sem stemmer | Demonstração no top-3 (sem companhias) | 20 | 0,00 | [0,00; 0,00] | 0/0/20 |
| BM25 sem stemmer (parâmetros da biblioteca) contra BM25 sem stemmer | Tabelas: MRR | 21 | 0,00 | [-0,07; 0,07] | 1/1/19 |
| BM25 sem stemmer (parâmetros da biblioteca) contra BM25 sem stemmer | Trechos: Hit@3 | 14 | 0,00 | [0,00; 0,00] | 0/0/14 |
| BM25 sem stemmer (parâmetros da biblioteca) contra BM25 sem stemmer | Trechos: Hit@5 | 14 | -0,14 | [-0,36; 0,00] | 0/2/12 |
| BM25 sem stemmer (parâmetros da biblioteca) contra BM25 sem stemmer | Trechos: MRR | 14 | -0,05 | [-0,13; 0,02] | 1/8/5 |
| BM25 sem stemmer (parâmetros da biblioteca) contra BM25 sem stemmer | Trechos: nDCG@10 | 14 | -0,01 | [-0,09; 0,07] | 1/3/10 |
| BM25 sem stemmer (parâmetros da biblioteca) contra BM25 sem stemmer | Híbridas: tabelas e trecho no top-3 | 7 | 0,00 | [0,00; 0,00] | 0/0/7 |
| BM25 sem stemmer (parâmetros da biblioteca) contra BM25 sem stemmer | Híbridas: demonstração e trecho no top-3 | 7 | 0,00 | [0,00; 0,00] | 0/0/7 |
| Por índice (tabelas E5, trechos BM25) contra E5-base (embeddings) | Tabelas: todas no top-3 | 21 | 0,00 | [0,00; 0,00] | 0/0/21 |
| Por índice (tabelas E5, trechos BM25) contra E5-base (embeddings) | Demonstração no top-3 (sem companhias) | 20 | 0,00 | [0,00; 0,00] | 0/0/20 |
| Por índice (tabelas E5, trechos BM25) contra E5-base (embeddings) | Tabelas: MRR | 21 | 0,00 | [0,00; 0,00] | 0/0/21 |
| Por índice (tabelas E5, trechos BM25) contra E5-base (embeddings) | Trechos: Hit@3 | 14 | 0,07 | [0,00; 0,21] | 1/0/13 |
| Por índice (tabelas E5, trechos BM25) contra E5-base (embeddings) | Trechos: Hit@5 | 14 | 0,29 | [0,07; 0,50] | 4/0/10 |
| Por índice (tabelas E5, trechos BM25) contra E5-base (embeddings) | Trechos: MRR | 14 | 0,07 | [-0,07; 0,25] | 6/3/5 |
| Por índice (tabelas E5, trechos BM25) contra E5-base (embeddings) | Trechos: nDCG@10 | 14 | 0,12 | [-0,02; 0,29] | 5/1/8 |
| Por índice (tabelas E5, trechos BM25) contra E5-base (embeddings) | Híbridas: tabelas e trecho no top-3 | 7 | 0,00 | [0,00; 0,00] | 0/0/7 |
| Por índice (tabelas E5, trechos BM25) contra E5-base (embeddings) | Híbridas: demonstração e trecho no top-3 | 7 | 0,00 | [0,00; 0,00] | 0/0/7 |
| Por índice (tabelas E5, trechos BM25) contra Por tipo (híbrida com RRF) | Tabelas: todas no top-3 | 21 | 0,10 | [0,00; 0,24] | 2/0/19 |
| Por índice (tabelas E5, trechos BM25) contra Por tipo (híbrida com RRF) | Demonstração no top-3 (sem companhias) | 20 | 0,10 | [0,00; 0,25] | 2/0/18 |
| Por índice (tabelas E5, trechos BM25) contra Por tipo (híbrida com RRF) | Tabelas: MRR | 21 | 0,02 | [-0,01; 0,06] | 3/1/17 |
| Por índice (tabelas E5, trechos BM25) contra Por tipo (híbrida com RRF) | Trechos: Hit@3 | 14 | 0,00 | [0,00; 0,00] | 0/0/14 |
| Por índice (tabelas E5, trechos BM25) contra Por tipo (híbrida com RRF) | Trechos: Hit@5 | 14 | 0,21 | [0,00; 0,43] | 3/0/11 |
| Por índice (tabelas E5, trechos BM25) contra Por tipo (híbrida com RRF) | Trechos: MRR | 14 | 0,03 | [0,00; 0,06] | 5/0/9 |
| Por índice (tabelas E5, trechos BM25) contra Por tipo (híbrida com RRF) | Trechos: nDCG@10 | 14 | 0,04 | [0,00; 0,10] | 3/0/11 |
| Por índice (tabelas E5, trechos BM25) contra Por tipo (híbrida com RRF) | Híbridas: tabelas e trecho no top-3 | 7 | 0,00 | [0,00; 0,00] | 0/0/7 |
| Por índice (tabelas E5, trechos BM25) contra Por tipo (híbrida com RRF) | Híbridas: demonstração e trecho no top-3 | 7 | 0,00 | [0,00; 0,00] | 0/0/7 |

## 5. Ressalvas

- Amostras pequenas (21 perguntas de tabelas, 14 de trechos, 7 híbridas): uma pergunta vale de 5 a 10 pontos percentuais, e diferenças de 1 a 3 perguntas são ruído. Os intervalos de bootstrap refletem só a variação entre perguntas deste conjunto.
- O BM25 roda sem ajuste nos dados (parâmetros tirados da referência, ver acima). A regra de pontuação por tabela foi escolhida olhando o conjunto de teste (`perguntas_dfp.json`), e a escolha do E5-base entre quatro modelos usou as perguntas de teste e de validação. Logo, nenhum dos dois conjuntos é totalmente independente para o E5-base, e o teste o favorece mais; o BM25 não foi ajustado em nenhum deles.
- Cada pergunta tem uma só evidência literal, e um trecho só conta como relevante se a contém. Trechos úteis sem essa frase contam como erro, e a Precision@k fica limitada (com um relevante e k = 5, o máximo é 0,2).
- As perguntas foram escritas pela autora; as do teste, com a evidência à vista, o que tende a favorecer a busca por palavras.
- A tabela companhias está entre as esperadas em quase todas as perguntas e entra na lista pela regra de chave estrangeira em todos os métodos; a seção 1.2 mostra o acerto sem ela.
- Nenhuma referência do trabalho define um valor mínimo aceitável de Recall. O piso aleatório e o BM25 são referências relativas.
- Empates de pontuação no BM25 ficam na ordem do índice (alfabética por tabela no esquema).
- O RRF usa k = 60 como no artigo original, sem ajuste. A escolha do BM25 sem stemmer para a fusão veio de resultados anteriores (o stemmer piorou a recuperação dos trechos), então é uma escolha informada pelos próprios dados.
- No esquema, a fusão é uma extensão exploratória do que o MimirRAG faz, pois ele aplica a fusão só a trechos de documentos.
