"""
Etapa: Online.

Função: busca por similaridade nos índices vetoriais de esquema e de glossário.

Descrição geral do funcionamento:
- O MiniLM gera o vetor da pergunta.
- O produto interno com os vetores do índice indica os registros mais parecidos.
  Como todos os vetores são normalizados, ele equivale à similaridade de cosseno.
- No índice de esquema, cada tabela é pontuada pelo melhor registro de resumo ou
  de entidades. Os registros de colunas, contas e relações não entram na pontuação.
- As tabelas escolhidas são acompanhadas das tabelas que elas referenciam por
  chave estrangeira (por exemplo, companhias).

Dependências:
- embeddings/index_esquema.json, embeddings/index_glossario.json e
  embeddings/esquema.json (gerados por pipeline/offline/);
- bibliotecas numpy e sentence-transformers;
- modelo MiniLM em cache local (se não estiver, é baixado).

Parâmetros: nenhum argumento. É um módulo importado pelos demais passos do estágio online.

Resultado: fornece vetorizar(), a classe Indice (buscar e buscar_por_fonte),
buscar_tabelas() e os caminhos dos índices.
"""

import json

import numpy as np
from sentence_transformers import SentenceTransformer

MODELO_EMBEDDING = "paraphrase-multilingual-MiniLM-L12-v2"
CAMINHO_INDICE_ESQUEMA = "embeddings/index_esquema.json"
CAMINHO_INDICE_GLOSSARIO = "embeddings/index_glossario.json"
CAMINHO_ESQUEMA = "embeddings/esquema.json"

# Colunas, contas e relações são parecidas entre as demonstrações e puxavam a
# busca para tabelas erradas, por isso ficam fora da pontuação da tabela.
SECOES_FORA_DA_PONTUACAO = ("Relações", "Colunas", "Contas")

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


# Lê do esquema.json, para cada tabela, as tabelas que ela referencia por chave estrangeira.
def carregar_tabelas_referenciadas(caminho: str = CAMINHO_ESQUEMA) -> dict[str, list[str]]:
    with open(caminho, encoding="utf-8") as f:
        esquema = json.load(f)
    return {
        entrada["tabela"]: [
            r["tabela_relacionada"] for r in entrada["chaves_de_juncao"] if r["sentido"] == "referencia"
        ]
        for entrada in esquema
    }


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

    # Devolve as k fontes (tabelas) mais parecidas, pelo melhor registro de resumo ou entidades.
    def buscar_por_fonte(self, vetor: np.ndarray, k: int) -> list[dict]:
        pontuacoes = self.matriz @ vetor
        por_fonte = {}
        for i, pontuacao in enumerate(pontuacoes):
            fonte = self.registros[i]["fonte"]
            conteudo = self.registros[i]["conteudo"]
            if any(conteudo.startswith(f"Tabela {fonte}. {secao}:") for secao in SECOES_FORA_DA_PONTUACAO):
                continue
            por_fonte.setdefault(fonte, []).append(float(pontuacao))
        ranking = [{"fonte": fonte, "pontuacao": max(lista)} for fonte, lista in por_fonte.items()]
        return sorted(ranking, key=lambda x: -x["pontuacao"])[:k]


# Devolve as k tabelas mais parecidas com a pergunta, seguidas das tabelas que elas referenciam.
def buscar_tabelas(indice: Indice, vetor: np.ndarray, k: int, referenciadas: dict[str, list[str]]) -> list[str]:
    # As tabelas referenciadas entram além das k escolhidas, pois o SQL precisa
    # delas no JOIN (por exemplo, companhias, que traz o nome da empresa).
    escolhidas = [r["fonte"] for r in indice.buscar_por_fonte(vetor, k)]
    ligadas = [t for fonte in escolhidas for t in referenciadas[fonte] if t not in escolhidas]
    return escolhidas + list(dict.fromkeys(ligadas))
