"""
Etapa: Online (baseline de recuperação).

Função: busca por palavras (BM25) nos mesmos registros dos índices vetoriais, usada como linha de
base para comparar com a busca por embeddings de busca.py.

Descrição geral do funcionamento:
- Os registros são os de embeddings/index_esquema.json e embeddings/index_glossario.json, e o texto
  indexado é o campo "conteudo". Os vetores não são usados.
- O pré-processamento é o mesmo para os registros e para a pergunta: minúsculas, separação em
  palavras, remoção de stopwords do português (lista do NLTK) e, se pedido, radicalização com o
  stemmer Snowball do português. No fim, os acentos são removidos (o stemmer recebe a palavra
  ainda acentuada, porque as regras dele usam os acentos).
- A pontuação é o Okapi BM25 da biblioteca rank_bm25, sem ajuste nos dados. Por padrão usa k1 = 2 e
  b = 0,5, a combinação comum citada por Robertson e Zaragoza (2009, p. 31); os valores da biblioteca
  (k1 = 1,5 e b = 0,75) podem ser passados no construtor. O epsilon fica no padrão da biblioteca (0,25).
- A interface é a mesma de Indice (buscar e buscar_por_fonte), então buscar_tabelas de busca.py
  funciona sem mudança. A consulta é o texto da pergunta, no lugar do vetor.
- No índice de esquema, só os registros de resumo e de entidades participam da pontuação e do
  cálculo do IDF, como em busca.py. Os de colunas, contas e relações ficam de fora.

Dependências:
- pipeline/online/busca.py e os índices em embeddings/;
- bibliotecas rank-bm25 e nltk, e o corpus de stopwords do NLTK
  (python -m nltk.downloader stopwords).

Parâmetros: nenhum argumento. É um módulo importado por avaliar_baseline.py.

Resultado: fornece preprocessar() e a classe IndiceBM25 (buscar, buscar_por_fonte, diagnostico).
"""

import json
import re
import unicodedata
from collections import Counter

import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from rank_bm25 import BM25Okapi

# Combinação comum de k1 e b citada por Robertson e Zaragoza (2009, p. 31).
K1_REFERENCIA = 2.0
B_REFERENCIA = 0.5
# Padrões da biblioteca rank_bm25, dentro das faixas 1,2 < k1 < 2 e 0,5 < b < 0,8 (mesma obra, p. 30).
K1_BIBLIOTECA = 1.5
B_BIBLIOTECA = 0.75

_stopwords = None
_stemmer = SnowballStemmer("portuguese")


# Lê a lista de stopwords do português uma única vez.
def _carregar_stopwords() -> set[str]:
    global _stopwords
    if _stopwords is None:
        try:
            _stopwords = set(stopwords.words("portuguese"))
        except LookupError:
            raise RuntimeError("Falta o corpus de stopwords do NLTK. Baixe com: python -m nltk.downloader stopwords")
    return _stopwords


# Tira os acentos de uma palavra.
def tirar_acentos(palavra: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", palavra) if not unicodedata.combining(c))


# Transforma um texto em uma lista de palavras para o BM25.
def preprocessar(texto: str, radicalizar: bool) -> list[str]:
    palavras = re.findall(r"[^\W_]+", texto.lower())
    stop = _carregar_stopwords()
    palavras = [p for p in palavras if p not in stop]
    if radicalizar:
        palavras = [_stemmer.stem(p) for p in palavras]
    return [tirar_acentos(p) for p in palavras]


class IndiceBM25:
    # Lê os registros de um índice, descarta as seções fora da pontuação e monta o BM25.
    def __init__(
        self, caminho: str, radicalizar: bool, secoes_fora: tuple[str, ...] = (),
        k1: float = K1_REFERENCIA, b: float = B_REFERENCIA,
    ):
        with open(caminho, encoding="utf-8") as f:
            todos = json.load(f)
        self.registros = [
            {"id": r["id"], "fonte": r["fonte"], "conteudo": r["conteudo"]}
            for r in todos
            if not any(r["conteudo"].startswith(f"Tabela {r['fonte']}. {secao}:") for secao in secoes_fora)
        ]
        self.radicalizar = radicalizar
        self.bm25 = BM25Okapi([preprocessar(r["conteudo"], radicalizar) for r in self.registros], k1=k1, b=b)

    # Pontua todos os registros para uma pergunta.
    def _pontuar(self, consulta: str) -> np.ndarray:
        return self.bm25.get_scores(preprocessar(consulta, self.radicalizar))

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
