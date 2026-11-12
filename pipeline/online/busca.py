"""
Etapa: Online.

Função: busca por similaridade nos índices vetoriais de esquema e de glossário.

Descrição geral do funcionamento:
- O modelo de embeddings vem de configuracao.json (padrão e5-base, que usa o prefixo "query: "
  que o E5 exige). A ficha do modelo (pipeline/modelos_embedding.py) dá o identificador e o
  prefixo, os mesmos usados no offline para gerar os índices.
- O índice do modelo padrão fica em embeddings/ (versionado); o dos outros modelos, em
  embeddings/modelos/<chave>/ (gerado por pipeline/offline/rodar_offline.py, ignorado pelo git).
- O produto interno com os vetores do índice indica os registros mais parecidos.
  Como todos os vetores são normalizados, ele equivale à similaridade de cosseno.
- No índice de esquema, cada tabela é pontuada pelo melhor registro de resumo ou
  de entidades. Os registros de colunas, contas e relações não entram na pontuação.
- As tabelas escolhidas são acompanhadas das tabelas que elas referenciam por
  chave estrangeira (por exemplo, companhias).

Dependências:
- embeddings/index_esquema.json, embeddings/index_glossario.json e
  embeddings/esquema.json (gerados por pipeline/offline/);
- configuracao.json (lido por configuracao.py) e pipeline/modelos_embedding.py;
- bibliotecas numpy e sentence-transformers;
- o modelo escolhido em cache local. Carregar o modelo nunca baixa nada: se faltar, o erro
  mostra o comando (o download é feito pelos scripts offline, com --baixar-modelos).

Parâmetros: nenhum argumento. É um módulo importado pelos demais passos do estágio online.

Resultado: fornece vetorizar(), a classe Indice (buscar e buscar_por_fonte),
buscar_tabelas(), caminho_indice() e os caminhos dos índices do modelo padrão.
"""

import json
import os
import sys

import numpy as np

# Os módulos compartilhados ficam em pipeline/, fora da pasta deste script.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modelos_embedding  # noqa: E402

import configuracao  # noqa: E402

CAMINHO_INDICE_ESQUEMA = "embeddings/index_esquema.json"
CAMINHO_INDICE_GLOSSARIO = "embeddings/index_glossario.json"
CAMINHO_ESQUEMA = "embeddings/esquema.json"
PASTA_MODELOS = "embeddings/modelos"

# Colunas, contas e relações são parecidas entre as demonstrações e puxavam a
# busca para tabelas erradas, por isso ficam fora da pontuação da tabela.
SECOES_FORA_DA_PONTUACAO = ("Relações", "Colunas", "Contas")

_chave_modelo = None


# Devolve a chave do modelo de embeddings em uso: a de configuracao.json, ou a definida por definir_modelo().
def modelo_configurado() -> str:
    global _chave_modelo
    if _chave_modelo is None:
        _chave_modelo = configuracao.carregar()["modelo_embedding"]
    return _chave_modelo


# Troca o modelo de embeddings em uso (os experimentos e a busca configurável chamam isto).
def definir_modelo(chave: str):
    global _chave_modelo
    modelos_embedding.obter_ficha(chave)
    _chave_modelo = chave


# Devolve onde está o índice vetorial ("esquema" ou "glossario") do modelo: o caminho versionado
# para o padrão, e a pasta ignorada pelo git para os outros (a mesma regra dos scripts offline).
def caminho_indice(indice: str, chave_modelo: str | None = None) -> str:
    chave_modelo = chave_modelo or modelo_configurado()
    if chave_modelo == modelos_embedding.MODELO_PADRAO:
        return CAMINHO_INDICE_ESQUEMA if indice == "esquema" else CAMINHO_INDICE_GLOSSARIO
    return os.path.join(PASTA_MODELOS, chave_modelo, f"index_{indice}.json")


# Carrega o modelo de embeddings em uso uma única vez (o registro guarda o modelo carregado).
def carregar_modelo(chave_modelo: str | None = None):
    return modelos_embedding.carregar_modelo(chave_modelo or modelo_configurado())


# Transforma a pergunta em um vetor normalizado (tamanho 1), com o prefixo de pergunta do modelo.
def vetorizar(texto: str, chave_modelo: str | None = None) -> np.ndarray:
    return modelos_embedding.vetorizar_pergunta(texto, chave_modelo or modelo_configurado())


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
