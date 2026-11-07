# Divisão das perguntas híbridas: dados nas tabelas e conceito nos trechos

- Gerado em 06/11/2026 19:15 por `python pipeline/online/avaliar_divisao_hibridas.py`.
- 10 perguntas híbridas (7 do teste e 3 da validação). Divisões em `dados/perguntas_hibridas_divididas.json`.
- As divisões foram escritas à mão, só a partir do texto das perguntas e antes de ver qualquer resultado: separação na conjunção, anáfora trocada pelo substantivo, palavras do conceito intactas. Ainda não foram revisadas pela autora. Como são manuais, o teste mede o teto da ideia; uma divisão automática por LLM erraria mais.
- Métodos e premissas: os de `avaliar_baseline.py` (E5-base; BM25 sem stemmer com k1 = 2 e b = 0,5; RRF com k = 60). Na condição dividida, a parte de dados vai às tabelas e a parte conceitual aos trechos; na inteira, a pergunta completa vai às duas; na mista, as tabelas usam a pergunta inteira e os trechos usam a parte conceitual.
- Na condição mista com o divisor do LLM, a divisão vem do `divisor.py` (LLM local), que é o que rodaria em tempo de execução, e não das divisões manuais.

## 1. Contagem de perguntas com acerto

| Condição | Tabelas completas@3 | Tabelas completas@5 | Hit@3 | Hit@5 | Hit@10 | Ambos@3 | Ambos@5 |
|---|---|---|---|---|---|---|---|
| E5-base (embeddings), pergunta inteira | 7/10 | 7/10 | 2/10 | 3/10 | 4/10 | 2/10 | 2/10 |
| E5-base (embeddings), pergunta dividida | 4/10 | 7/10 | 4/10 | 6/10 | 8/10 | 3/10 | 4/10 |
| E5-base (embeddings), pergunta mista | 7/10 | 7/10 | 4/10 | 6/10 | 8/10 | 3/10 | 5/10 |
| E5-base (embeddings), pergunta mista (divisor do LLM) | 7/10 | 7/10 | 3/10 | 5/10 | 8/10 | 2/10 | 4/10 |
| BM25 sem stemmer, pergunta inteira | 7/10 | 7/10 | 2/10 | 6/10 | 7/10 | 2/10 | 4/10 |
| BM25 sem stemmer, pergunta dividida | 5/10 | 7/10 | 5/10 | 6/10 | 7/10 | 3/10 | 6/10 |
| BM25 sem stemmer, pergunta mista | 7/10 | 7/10 | 5/10 | 6/10 | 7/10 | 4/10 | 5/10 |
| BM25 sem stemmer, pergunta mista (divisor do LLM) | 7/10 | 7/10 | 3/10 | 4/10 | 6/10 | 2/10 | 3/10 |
| RRF (E5 + BM25), pergunta inteira | 5/10 | 7/10 | 2/10 | 4/10 | 6/10 | 2/10 | 4/10 |
| RRF (E5 + BM25), pergunta dividida | 4/10 | 8/10 | 5/10 | 7/10 | 9/10 | 4/10 | 6/10 |
| RRF (E5 + BM25), pergunta mista | 5/10 | 7/10 | 5/10 | 7/10 | 9/10 | 4/10 | 7/10 |
| RRF (E5 + BM25), pergunta mista (divisor do LLM) | 5/10 | 7/10 | 3/10 | 6/10 | 9/10 | 3/10 | 6/10 |
| Piso aleatório | 2,31/10 | 3,85/10 | 0,03/10 | 0,05/10 | 0,09/10 | 0,01/10 | 0,02/10 |

## 2. Médias

| Condição | MRR tabelas | MRR trechos | nDCG@10 trechos |
|---|---|---|---|
| E5-base (embeddings), pergunta inteira | 0,54 | 0,18 | 0,24 |
| E5-base (embeddings), pergunta dividida | 0,50 | 0,31 | 0,49 |
| E5-base (embeddings), pergunta mista | 0,54 | 0,31 | 0,49 |
| E5-base (embeddings), pergunta mista (divisor do LLM) | 0,54 | 0,27 | 0,42 |
| BM25 sem stemmer, pergunta inteira | 0,66 | 0,25 | 0,35 |
| BM25 sem stemmer, pergunta dividida | 0,36 | 0,36 | 0,46 |
| BM25 sem stemmer, pergunta mista | 0,66 | 0,36 | 0,46 |
| BM25 sem stemmer, pergunta mista (divisor do LLM) | 0,66 | 0,24 | 0,33 |
| RRF (E5 + BM25), pergunta inteira | 0,52 | 0,22 | 0,31 |
| RRF (E5 + BM25), pergunta dividida | 0,41 | 0,41 | 0,61 |
| RRF (E5 + BM25), pergunta mista | 0,52 | 0,41 | 0,61 |
| RRF (E5 + BM25), pergunta mista (divisor do LLM) | 0,52 | 0,36 | 0,50 |
| Piso aleatório | 0,35 | 0,01 | 0,00 |

## 3. Comparação pareada (condição menos pergunta inteira)

Diferença média, intervalo de 95% por bootstrap sobre as perguntas e vitórias, derrotas e empates da condição comparada (dividida ou mista).

| Método e condição | Métrica | n | Diferença média | IC 95% | Vitórias/derrotas/empates |
|---|---|---|---|---|---|
| E5-base (embeddings), dividida | Tabelas: todas no top-3 | 10 | -0,30 | [-0,60; 0,00] | 0/3/7 |
| E5-base (embeddings), dividida | Tabelas: todas no top-5 | 10 | 0,00 | [-0,30; 0,30] | 1/1/8 |
| E5-base (embeddings), dividida | Trechos: Hit@3 | 10 | 0,20 | [0,00; 0,50] | 2/0/8 |
| E5-base (embeddings), dividida | Trechos: Hit@5 | 10 | 0,30 | [0,00; 0,60] | 3/0/7 |
| E5-base (embeddings), dividida | Trechos: Hit@10 | 10 | 0,40 | [0,10; 0,70] | 4/0/6 |
| E5-base (embeddings), dividida | Trechos: MRR | 10 | 0,13 | [0,06; 0,20] | 8/1/1 |
| E5-base (embeddings), dividida | Trechos: nDCG@10 | 10 | 0,25 | [0,11; 0,40] | 7/0/3 |
| E5-base (embeddings), dividida | Tabelas e trecho no top-3 | 10 | 0,10 | [0,00; 0,30] | 1/0/9 |
| E5-base (embeddings), dividida | Tabelas e trecho no top-5 | 10 | 0,20 | [0,00; 0,50] | 2/0/8 |
| E5-base (embeddings), mista | Tabelas: todas no top-3 | 10 | 0,00 | [0,00; 0,00] | 0/0/10 |
| E5-base (embeddings), mista | Tabelas: todas no top-5 | 10 | 0,00 | [0,00; 0,00] | 0/0/10 |
| E5-base (embeddings), mista | Trechos: Hit@3 | 10 | 0,20 | [0,00; 0,50] | 2/0/8 |
| E5-base (embeddings), mista | Trechos: Hit@5 | 10 | 0,30 | [0,00; 0,60] | 3/0/7 |
| E5-base (embeddings), mista | Trechos: Hit@10 | 10 | 0,40 | [0,10; 0,70] | 4/0/6 |
| E5-base (embeddings), mista | Trechos: MRR | 10 | 0,13 | [0,06; 0,20] | 8/1/1 |
| E5-base (embeddings), mista | Trechos: nDCG@10 | 10 | 0,25 | [0,12; 0,40] | 7/0/3 |
| E5-base (embeddings), mista | Tabelas e trecho no top-3 | 10 | 0,10 | [0,00; 0,30] | 1/0/9 |
| E5-base (embeddings), mista | Tabelas e trecho no top-5 | 10 | 0,30 | [0,00; 0,60] | 3/0/7 |
| E5-base (embeddings), mista (divisor do LLM) | Tabelas: todas no top-3 | 10 | 0,00 | [0,00; 0,00] | 0/0/10 |
| E5-base (embeddings), mista (divisor do LLM) | Tabelas: todas no top-5 | 10 | 0,00 | [0,00; 0,00] | 0/0/10 |
| E5-base (embeddings), mista (divisor do LLM) | Trechos: Hit@3 | 10 | 0,10 | [-0,20; 0,40] | 2/1/7 |
| E5-base (embeddings), mista (divisor do LLM) | Trechos: Hit@5 | 10 | 0,20 | [-0,20; 0,60] | 3/1/6 |
| E5-base (embeddings), mista (divisor do LLM) | Trechos: Hit@10 | 10 | 0,40 | [0,10; 0,70] | 4/0/6 |
| E5-base (embeddings), mista (divisor do LLM) | Trechos: MRR | 10 | 0,09 | [-0,01; 0,18] | 7/2/1 |
| E5-base (embeddings), mista (divisor do LLM) | Trechos: nDCG@10 | 10 | 0,18 | [0,01; 0,36] | 6/1/3 |
| E5-base (embeddings), mista (divisor do LLM) | Tabelas e trecho no top-3 | 10 | 0,00 | [-0,30; 0,30] | 1/1/8 |
| E5-base (embeddings), mista (divisor do LLM) | Tabelas e trecho no top-5 | 10 | 0,20 | [-0,20; 0,60] | 3/1/6 |
| BM25 sem stemmer, dividida | Tabelas: todas no top-3 | 10 | -0,20 | [-0,50; 0,00] | 0/2/8 |
| BM25 sem stemmer, dividida | Tabelas: todas no top-5 | 10 | 0,00 | [-0,30; 0,30] | 1/1/8 |
| BM25 sem stemmer, dividida | Trechos: Hit@3 | 10 | 0,30 | [0,00; 0,60] | 3/0/7 |
| BM25 sem stemmer, dividida | Trechos: Hit@5 | 10 | 0,00 | [-0,30; 0,30] | 1/1/8 |
| BM25 sem stemmer, dividida | Trechos: Hit@10 | 10 | 0,00 | [-0,30; 0,30] | 1/1/8 |
| BM25 sem stemmer, dividida | Trechos: MRR | 10 | 0,11 | [-0,02; 0,30] | 5/1/4 |
| BM25 sem stemmer, dividida | Trechos: nDCG@10 | 10 | 0,12 | [-0,00; 0,27] | 4/1/5 |
| BM25 sem stemmer, dividida | Tabelas e trecho no top-3 | 10 | 0,10 | [-0,20; 0,40] | 2/1/7 |
| BM25 sem stemmer, dividida | Tabelas e trecho no top-5 | 10 | 0,20 | [0,00; 0,50] | 2/0/8 |
| BM25 sem stemmer, mista | Tabelas: todas no top-3 | 10 | 0,00 | [0,00; 0,00] | 0/0/10 |
| BM25 sem stemmer, mista | Tabelas: todas no top-5 | 10 | 0,00 | [0,00; 0,00] | 0/0/10 |
| BM25 sem stemmer, mista | Trechos: Hit@3 | 10 | 0,30 | [0,00; 0,60] | 3/0/7 |
| BM25 sem stemmer, mista | Trechos: Hit@5 | 10 | 0,00 | [-0,30; 0,30] | 1/1/8 |
| BM25 sem stemmer, mista | Trechos: Hit@10 | 10 | 0,00 | [-0,30; 0,30] | 1/1/8 |
| BM25 sem stemmer, mista | Trechos: MRR | 10 | 0,11 | [-0,02; 0,30] | 5/1/4 |
| BM25 sem stemmer, mista | Trechos: nDCG@10 | 10 | 0,12 | [-0,00; 0,27] | 4/1/5 |
| BM25 sem stemmer, mista | Tabelas e trecho no top-3 | 10 | 0,20 | [0,00; 0,50] | 2/0/8 |
| BM25 sem stemmer, mista | Tabelas e trecho no top-5 | 10 | 0,10 | [0,00; 0,30] | 1/0/9 |
| BM25 sem stemmer, mista (divisor do LLM) | Tabelas: todas no top-3 | 10 | 0,00 | [0,00; 0,00] | 0/0/10 |
| BM25 sem stemmer, mista (divisor do LLM) | Tabelas: todas no top-5 | 10 | 0,00 | [0,00; 0,00] | 0/0/10 |
| BM25 sem stemmer, mista (divisor do LLM) | Trechos: Hit@3 | 10 | 0,10 | [-0,20; 0,40] | 2/1/7 |
| BM25 sem stemmer, mista (divisor do LLM) | Trechos: Hit@5 | 10 | -0,20 | [-0,50; 0,00] | 0/2/8 |
| BM25 sem stemmer, mista (divisor do LLM) | Trechos: Hit@10 | 10 | -0,10 | [-0,40; 0,20] | 1/2/7 |
| BM25 sem stemmer, mista (divisor do LLM) | Trechos: MRR | 10 | -0,01 | [-0,08; 0,06] | 4/3/3 |
| BM25 sem stemmer, mista (divisor do LLM) | Trechos: nDCG@10 | 10 | -0,02 | [-0,13; 0,08] | 3/3/4 |
| BM25 sem stemmer, mista (divisor do LLM) | Tabelas e trecho no top-3 | 10 | 0,00 | [-0,30; 0,30] | 1/1/8 |
| BM25 sem stemmer, mista (divisor do LLM) | Tabelas e trecho no top-5 | 10 | -0,10 | [-0,30; 0,00] | 0/1/9 |
| RRF (E5 + BM25), dividida | Tabelas: todas no top-3 | 10 | -0,10 | [-0,30; 0,00] | 0/1/9 |
| RRF (E5 + BM25), dividida | Tabelas: todas no top-5 | 10 | 0,10 | [-0,20; 0,40] | 2/1/7 |
| RRF (E5 + BM25), dividida | Trechos: Hit@3 | 10 | 0,30 | [0,00; 0,60] | 3/0/7 |
| RRF (E5 + BM25), dividida | Trechos: Hit@5 | 10 | 0,30 | [0,00; 0,60] | 3/0/7 |
| RRF (E5 + BM25), dividida | Trechos: Hit@10 | 10 | 0,30 | [-0,10; 0,70] | 4/1/5 |
| RRF (E5 + BM25), dividida | Trechos: MRR | 10 | 0,19 | [0,06; 0,39] | 7/2/1 |
| RRF (E5 + BM25), dividida | Trechos: nDCG@10 | 10 | 0,30 | [0,12; 0,50] | 7/2/1 |
| RRF (E5 + BM25), dividida | Tabelas e trecho no top-3 | 10 | 0,20 | [0,00; 0,50] | 2/0/8 |
| RRF (E5 + BM25), dividida | Tabelas e trecho no top-5 | 10 | 0,20 | [0,00; 0,50] | 2/0/8 |
| RRF (E5 + BM25), mista | Tabelas: todas no top-3 | 10 | 0,00 | [0,00; 0,00] | 0/0/10 |
| RRF (E5 + BM25), mista | Tabelas: todas no top-5 | 10 | 0,00 | [0,00; 0,00] | 0/0/10 |
| RRF (E5 + BM25), mista | Trechos: Hit@3 | 10 | 0,30 | [0,00; 0,60] | 3/0/7 |
| RRF (E5 + BM25), mista | Trechos: Hit@5 | 10 | 0,30 | [0,00; 0,60] | 3/0/7 |
| RRF (E5 + BM25), mista | Trechos: Hit@10 | 10 | 0,30 | [-0,10; 0,70] | 4/1/5 |
| RRF (E5 + BM25), mista | Trechos: MRR | 10 | 0,19 | [0,06; 0,39] | 7/2/1 |
| RRF (E5 + BM25), mista | Trechos: nDCG@10 | 10 | 0,30 | [0,12; 0,51] | 7/2/1 |
| RRF (E5 + BM25), mista | Tabelas e trecho no top-3 | 10 | 0,20 | [0,00; 0,50] | 2/0/8 |
| RRF (E5 + BM25), mista | Tabelas e trecho no top-5 | 10 | 0,30 | [0,00; 0,60] | 3/0/7 |
| RRF (E5 + BM25), mista (divisor do LLM) | Tabelas: todas no top-3 | 10 | 0,00 | [0,00; 0,00] | 0/0/10 |
| RRF (E5 + BM25), mista (divisor do LLM) | Tabelas: todas no top-5 | 10 | 0,00 | [0,00; 0,00] | 0/0/10 |
| RRF (E5 + BM25), mista (divisor do LLM) | Trechos: Hit@3 | 10 | 0,10 | [-0,20; 0,40] | 2/1/7 |
| RRF (E5 + BM25), mista (divisor do LLM) | Trechos: Hit@5 | 10 | 0,20 | [-0,20; 0,60] | 3/1/6 |
| RRF (E5 + BM25), mista (divisor do LLM) | Trechos: Hit@10 | 10 | 0,30 | [-0,10; 0,70] | 4/1/5 |
| RRF (E5 + BM25), mista (divisor do LLM) | Trechos: MRR | 10 | 0,14 | [-0,01; 0,34] | 5/4/1 |
| RRF (E5 + BM25), mista (divisor do LLM) | Trechos: nDCG@10 | 10 | 0,19 | [-0,00; 0,42] | 5/4/1 |
| RRF (E5 + BM25), mista (divisor do LLM) | Tabelas e trecho no top-3 | 10 | 0,10 | [-0,20; 0,40] | 2/1/7 |
| RRF (E5 + BM25), mista (divisor do LLM) | Tabelas e trecho no top-5 | 10 | 0,20 | [-0,20; 0,60] | 3/1/6 |

## 4. Posição do primeiro trecho relevante, por pergunta (entre 1.634 trechos)

| Pergunta | E5-base, inteira | E5-base, dividida | E5-base, divisor do LLM | BM25 sem stemmer, inteira | BM25 sem stemmer, dividida | BM25 sem stemmer, divisor do LLM | RRF, inteira | RRF, dividida | RRF, divisor do LLM |
|---|---|---|---|---|---|---|---|---|---|
| h01 | 3 | 2 | 9 | 3 | 3 | 6 | 3 | 2 | 4 |
| h02 | 304 | 3 | 3 | 5 | 3 | 3 | 38 | 1 | 1 |
| h03 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| h04 | 38 | 29 | 29 | 4 | 17 | 17 | 9 | 16 | 16 |
| h05 | 97 | 10 | 10 | 12 | 12 | 12 | 29 | 6 | 6 |
| h06 | 11 | 13 | 13 | 5 | 3 | 3 | 7 | 8 | 8 |
| h07 | 185 | 8 | 10 | 31 | 9 | 10 | 47 | 5 | 4 |
| v11 | 130 | 5 | 5 | 24 | 14 | 14 | 28 | 4 | 4 |
| v12 | 5 | 2 | 2 | 9 | 1 | 65 | 4 | 2 | 7 |
| v13 | 8 | 5 | 5 | 4 | 4 | 4 | 4 | 3 | 3 |

## 5. Divisões feitas pelo LLM e as manuais

- O divisor (`divisor.py`, `qwen2.5-coder:7b`, uma execução) dividiu 10 de 10 perguntas; nas demais, a divisão foi descartada e valeu a pergunta inteira. Tempo por pergunta: média de 2,54 s, máximo de 10,33 s (inclui o carregamento do modelo na primeira chamada).
- A parte conceitual saiu igual à manual em 6 de 10 perguntas (ignorando maiúsculas). As divisões geradas estão em `metricas\divisoes_geradas_pelo_llm.json`.

| Pergunta | Parte conceitual (manual) | Parte conceitual (LLM) | Parte de dados (manual) | Parte de dados (LLM) |
|---|---|---|---|---|
| h01 | Como o método indireto chega ao caixa líquido das atividades operacionais? | Como o método indireto chega a esse valor? | Qual foi o caixa líquido das atividades operacionais da Ambev em 2025? | Qual foi o caixa líquido das atividades operacionais da Ambev em 2025? |
| h02 | O que representa o valor adicionado que está sendo distribuído? | O que representa o valor adicionado que está sendo distribuído? | Quanto a Ambev distribuiu para pessoal na DVA de 2025? | Quanto a Ambev distribuiu para pessoal na DVA de 2025? |
| h03 | O balanço patrimonial é um dos documentos do conjunto completo de demonstrações contábeis? | O balanço patrimonial é um dos quais documentos do conjunto completo de demonstrações contábeis? | Qual foi o ativo total consolidado da Alpargatas em 2025? | Qual foi o ativo total consolidado da Alpargatas em 2025? |
| h04 | Que informações periódicas uma companhia aberta deve divulgar segundo a Resolução CVM 80? | Que informações periódicas uma companhia aberta deve divulgar segundo a Resolução CVM 80? | Quantas companhias reportaram prejuízo líquido em 2025? | Quantas companhias reportaram prejuízo líquido em 2025? |
| h05 | O que são equivalentes de caixa segundo o CPC 03? | O que são equivalentes de caixa segundo o CPC 03? | Qual foi o caixa e equivalentes de caixa da Ambev no balanço de 2025? | Qual foi o caixa e equivalentes de caixa da Ambev no balanço de 2025? |
| h06 | A demonstração das mutações do patrimônio líquido faz parte de qual conjunto de demonstrações? | A demonstração das mutações do patrimônio líquido faz parte de qual conjunto de demonstrações? | Qual foi o patrimônio líquido consolidado da Ambev em 2025? | Qual foi o patrimônio líquido consolidado da Ambev em 2025? |
| h07 | Quais grupos compõem a distribuição do valor adicionado? | Quais grupos compõem a distribuição do valor adicionado que essas companhias também divulgam? | Entre Ambev e JBS, qual teve maior receita líquida em 2025? | Entre Ambev e JBS, qual teve maior receita líquida em 2025? |
| v11 | O que é o valor adicionado que está sendo distribuído? | O que é o valor adicionado que está sendo distribuído? | Quanto a WEG distribuiu para pessoal em 2025? | Quanto a WEG distribuiu para pessoal em 2025? |
| v12 | O que entra nas atividades de investimento? | O que entra nessa atividade? | Qual foi o caixa líquido das atividades de investimento da Suzano em 2025? | Qual foi o caixa líquido das atividades de investimento da Suzano em 2025? |
| v13 | O que faz um passivo ser circulante? | O que faz um passivo ser circulante? | Qual o passivo circulante da Natura? | Qual o passivo circulante da Natura? |

## 6. Ressalvas

- O divisor do LLM foi avaliado uma vez, sem ajustar o prompt depois de ver os resultados. O prompt foi escrito depois de conhecer a forma das perguntas híbridas (não os resultados da busca) e seus exemplos não são perguntas avaliadas.
- Só 10 perguntas: uma vale 10 pontos percentuais, e os intervalos de bootstrap refletem só a variação entre elas.
- As divisões são do Claude, a partir do texto das perguntas, e ainda aguardam a revisão da autora; trocar uma divisão pode mudar o resultado.
- A condição mista foi acrescentada depois de ver que usar só a parte de dados piora as tabelas; é uma combinação escolhida após ver os resultados das outras duas condições, e precisa ser confirmada em perguntas novas.
- A divisão manual é um teto: no pipeline, separar a pergunta exigiria um passo novo (por exemplo, um planejador com LLM, como no MimirRAG e no kartheesan2026), que a Metodologia não prevê e que o modelo de 7B pode errar.
- O critério de relevância é a evidência literal do trecho; trechos úteis sem essa frase contam como erro.
- O pipeline oficial não foi alterado.
