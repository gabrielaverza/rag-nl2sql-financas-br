"""
Etapa: Online (baseline de recuperação).

Função: busca por palavras (BM25) nos mesmos registros dos índices vetoriais, usada como linha de
base para comparar com a busca por embeddings de busca.py.

Descrição geral do funcionamento:
- Os registros vêm dos índices de palavras embeddings/index_bm25_esquema.json e
  embeddings/index_bm25_glossario.json, gerados pelo estágio offline
  (pipeline/offline/gerar_indice_bm25.py) a partir dos mesmos registros dos índices vetoriais.
  Cada registro traz o texto ("conteudo") e as palavras já limpas, com e sem radicalização; os
  vetores não são usados.
- O pré-processamento é o mesmo para os registros (feito no offline) e para a pergunta (feito
  aqui): minúsculas, separação em palavras, remoção de stopwords do português (lista do NLTK,
  guardada no próprio índice) e, se pedido, radicalização com o stemmer Snowball do português.
  No fim, os acentos são removidos (o stemmer recebe a palavra ainda acentuada, porque as regras
  dele usam os acentos).
- O NLTK só é importado quando é preciso: para gerar o índice (offline) ou para radicalizar a
  pergunta. Sem radicalização, a busca não carrega o NLTK.
- A pontuação é o Okapi BM25 da biblioteca rank_bm25, sem ajuste nos dados. Por padrão usa k1 = 2 e
  b = 0,5, a combinação comum citada por Robertson e Zaragoza (2009, p. 31); os valores da biblioteca
  (k1 = 1,5 e b = 0,75) podem ser passados no construtor. O epsilon fica no padrão da biblioteca (0,25).
- A interface é a mesma de Indice (buscar e buscar_por_fonte), então buscar_tabelas de busca.py
  funciona sem mudança. A consulta é o texto da pergunta, no lugar do vetor.
- No índice de esquema, só os registros de resumo e de entidades participam da pontuação e do
  cálculo do IDF, como em busca.py. Os de colunas, contas e relações ficam de fora.

Dependências:
- os índices de palavras em embeddings/ (gerados por pipeline/offline/gerar_indice_bm25.py);
- biblioteca rank-bm25; a nltk só para radicalizar (e, para gerar o índice, o corpus de
  stopwords do NLTK: python -m nltk.downloader stopwords).

Parâmetros: nenhum argumento. É um módulo importado pelos demais módulos de busca e pelo executor dos experimentos.

Resultado: fornece preprocessar(), tirar_acentos(), os caminhos dos índices de palavras e a classe
IndiceBM25 (buscar, buscar_por_fonte, diagnostico).
"""

import json
import re
import unicodedata
from collections import Counter

import numpy as np
from rank_bm25 import BM25Okapi

CAMINHO_BM25_ESQUEMA = "embeddings/index_bm25_esquema.json"
CAMINHO_BM25_GLOSSARIO = "embeddings/index_bm25_glossario.json"

# Combinação comum de k1 e b citada por Robertson e Zaragoza (2009, p. 31).
K1_REFERENCIA = 2.0
B_REFERENCIA = 0.5
# Padrões da biblioteca rank_bm25, dentro das faixas 1,2 < k1 < 2 e 0,5 < b < 0,8 (mesma obra, p. 30).
K1_BIBLIOTECA = 1.5
B_BIBLIOTECA = 0.75

_stopwords = None
_stemmer = None


# Lê do corpus do NLTK a lista de stopwords do português, uma única vez (usada para gerar o índice).
def _carregar_stopwords() -> set[str]:
    global _stopwords
    if _stopwords is None:
        from nltk.corpus import stopwords

        try:
            _stopwords = set(stopwords.words("portuguese"))
        except LookupError:
            raise RuntimeError("Falta o corpus de stopwords do NLTK. Baixe com: python -m nltk.downloader stopwords")
    return _stopwords


# Cria o radicalizador Snowball do português na primeira vez que ele é preciso.
def _obter_stemmer():
    global _stemmer
    if _stemmer is None:
        from nltk.stem.snowball import SnowballStemmer

        _stemmer = SnowballStemmer("portuguese")
    return _stemmer


# Tira os acentos de uma palavra.
def tirar_acentos(palavra: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", palavra) if not unicodedata.combining(c))


# Transforma um texto em uma lista de palavras para o BM25; sem a lista de stopwords, usa a do NLTK.
def preprocessar(texto: str, radicalizar: bool, stopwords: set[str] | None = None) -> list[str]:
    palavras = re.findall(r"[^\W_]+", texto.lower())
    stop = stopwords if stopwords is not None else _carregar_stopwords()
    palavras = [p for p in palavras if p not in stop]
    if radicalizar:
        stemmer = _obter_stemmer()
        palavras = [stemmer.stem(p) for p in palavras]
    return [tirar_acentos(p) for p in palavras]


class IndiceBM25:
    # Lê um índice de palavras, descarta as seções fora da pontuação e monta o BM25.
    def __init__(
        self, caminho: str, radicalizar: bool, secoes_fora: tuple[str, ...] = (),
        k1: float = K1_REFERENCIA, b: float = B_REFERENCIA,
    ):
        with open(caminho, encoding="utf-8") as f:
            dados = json.load(f)
        self.stopwords = set(dados["stopwords"])
        campo = "tokens_radicalizados" if radicalizar else "tokens"
        selecionados = [
            r for r in dados["registros"]
            if not any(r["conteudo"].startswith(f"Tabela {r['fonte']}. {secao}:") for secao in secoes_fora)
        ]
        self.registros = [{"id": r["id"], "fonte": r["fonte"], "conteudo": r["conteudo"]} for r in selecionados]
        self.radicalizar = radicalizar
        self.bm25 = BM25Okapi([r[campo] for r in selecionados], k1=k1, b=b)

    # Pontua todos os registros para uma pergunta.
    def _pontuar(self, consulta: str) -> np.ndarray:
        return self.bm25.get_scores(preprocessar(consulta, self.radicalizar, self.stopwords))

    # Devolve os k registros de maior pontuação (empates ficam na ordem do índice).
    def buscar(self, consulta: str, k: int) -> list[dict]:
        pontuacoes = self._pontuar(consulta)
        ordem = np.argsort(-pontuacoes, kind="stable")[:k]
        return [{**self.registros[i], "pontuacao": float(pontuacoes[i])} for i in ordem]

    # Devolve as k fontes (tabelas) de maior pontuação, pelo melhor registro de cada uma.
    def buscar_por_fonte(self, consulta: str, k: int) -> list[dict]:
        pontuacoes = self._pontuar(consulta)
        por_fonte = {}
        for registro, pontuacao in zip(self.registros, pontuacoes):
            fonte = registro["fonte"]
            por_fonte[fonte] = max(por_fonte.get(fonte, float("-inf")), float(pontuacao))
        ranking = [{"fonte": fonte, "pontuacao": p} for fonte, p in por_fonte.items()]
        return sorted(ranking, key=lambda x: -x["pontuacao"])[:k]

    # Resume o vocabulário e o piso de IDF do índice (palavras em mais da metade dos registros).
    def diagnostico(self) -> dict:
        total = self.bm25.corpus_size
        presenca = Counter(palavra for doc in self.bm25.doc_freqs for palavra in doc)
        return {
            "registros": total,
            "palavras": len(presenca),
            "idf_negativo": sum(1 for n in presenca.values() if n > total / 2),
            "idf_medio": self.bm25.average_idf,
            "piso_idf": self.bm25.epsilon * self.bm25.average_idf,
        }
