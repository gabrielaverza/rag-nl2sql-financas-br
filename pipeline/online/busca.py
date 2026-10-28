"""
Etapa: Online.

Função: busca por similaridade nos índices vetoriais de esquema e de glossário.

Descrição geral do funcionamento:
- O MiniLM gera o vetor da pergunta.
- O produto interno com os vetores do índice indica os registros mais parecidos.
  Como todos os vetores são normalizados, ele equivale à similaridade de cosseno.
- No índice de esquema, as tabelas são ordenadas pela média das pontuações de
  todos os seus registros.

Dependências:
- embeddings/index_esquema.json e embeddings/index_glossario.json (gerados por
  pipeline/gerar_indice_esquema.py e pipeline/gerar_indice_glossario.py);
- bibliotecas numpy e sentence-transformers;
- modelo MiniLM em cache local (se não estiver, é baixado).

Parâmetros: nenhum argumento. É um módulo importado pelos demais passos do estágio online.

Resultado: fornece vetorizar(), a classe Indice (buscar e buscar_por_fonte) e os
caminhos dos dois índices.
"""

import json

import numpy as np
from sentence_transformers import SentenceTransformer

MODELO_EMBEDDING = "paraphrase-multilingual-MiniLM-L12-v2"
CAMINHO_INDICE_ESQUEMA = "embeddings/index_esquema.json"
CAMINHO_INDICE_GLOSSARIO = "embeddings/index_glossario.json"

_modelo = None


# Carrega o MiniLM uma única vez e o reaproveita nas buscas seguintes.
def carregar_modelo() -> SentenceTransformer:
    global _modelo
    if _modelo is None:
        # Usa o cache local para não depender da rede; só baixa se não houver cache.
        try:
            _modelo = SentenceTransformer(MODELO_EMBEDDING, local_files_only=True)
        except OSError:
            _modelo = SentenceTransformer(MODELO_EMBEDDING)
    return _modelo


# Transforma o texto em um vetor normalizado (tamanho 1).
def vetorizar(texto: str) -> np.ndarray:
    return carregar_modelo().encode(texto, normalize_embeddings=True)


class Indice:
    # Carrega os registros de um índice e normaliza seus vetores.
    def __init__(self, caminho: str):
        with open(caminho, encoding="utf-8") as f:
            self.registros = json.load(f)
        matriz = np.array([r["embedding"] for r in self.registros], dtype=np.float32)
        # Com os vetores normalizados, o produto interno vira o cosseno.
        self.matriz = matriz / np.linalg.norm(matriz, axis=1, keepdims=True)

    # Devolve os k registros mais parecidos com a pergunta, sem o embedding.
    def buscar(self, vetor: np.ndarray, k: int) -> list[dict]:
        pontuacoes = self.matriz @ vetor
        ordem = np.argsort(-pontuacoes)[:k]
        return [
            {
                "id": self.registros[i]["id"],
                "fonte": self.registros[i]["fonte"],
                "conteudo": self.registros[i]["conteudo"],
                "pontuacao": float(pontuacoes[i]),
            }
            for i in ordem
        ]

    # Devolve as k fontes (tabelas) mais parecidas, pela média das pontuações de seus registros.
    def buscar_por_fonte(self, vetor: np.ndarray, k: int) -> list[dict]:
        # Cada tabela tem vários registros (um por aspecto). A média separou
        # melhor as tabelas do que o registro mais parecido, porque registros
        # curtos e genéricos de outras tabelas chegavam a ganhar do registro
        # descritivo da tabela certa.
        pontuacoes = self.matriz @ vetor
        por_fonte = {}
        for i, pontuacao in enumerate(pontuacoes):
            por_fonte.setdefault(self.registros[i]["fonte"], []).append(float(pontuacao))
        ranking = [{"fonte": fonte, "pontuacao": float(np.mean(lista))} for fonte, lista in por_fonte.items()]
        return sorted(ranking, key=lambda x: -x["pontuacao"])[:k]
