"""
Etapa: Online (preparação, ainda fora do pipeline).

Função: escolhe o método de busca (embeddings, BM25 ou fusão RRF) conforme o tipo da pergunta, para
quando a autora decidir adotar a busca por tipo. Nada do pipeline oficial chama este módulo.

Descrição geral do funcionamento:
- A política padrão (POLITICA_PADRAO, decisão da autora em 06/11/2026) escolhe o método por índice,
  igual para todos os tipos: as tabelas usam embeddings e os trechos usam BM25. Uma pergunta híbrida
  faz as duas buscas, cada uma com o seu método. A política anterior, com a fusão RRF nas perguntas
  híbridas, fica em POLITICA_HIBRIDA_RRF, só para comparação.
- A política é um dicionário (tipo, e dentro dele "tabelas" e "trechos", com o nome do método), então
  trocar o comportamento é trocar um valor, sem mexer no resto do código.
- Os métodos são "embeddings" (E5-base, busca.py), "bm25" (BM25 sem stemmer, k1 = 2 e b = 0,5,
  busca_bm25.py) e "rrf" (fusão dos dois, fusao_rrf.py).
- O tipo da pergunta vem do classificador (classificador.py), que este módulo não chama. Pedir um
  tipo ou um índice que a política não prevê levanta ValueError.
- Os índices são montados na primeira vez em que são usados.
- avaliar_baseline.py mede a política com o tipo correto de cada pergunta, sem erro de classificação.

Dependências:
- pipeline/online/busca.py, busca_bm25.py e fusao_rrf.py, e os índices e o esquema em embeddings/;
- bibliotecas numpy, sentence-transformers, rank-bm25 e nltk (com o corpus de stopwords).

Parâmetros: nenhum argumento. É um módulo importado.

Resultado: fornece POLITICA_PADRAO, POLITICA_HIBRIDA_RRF e a classe BuscaRoteada (tabelas e trechos).
"""

from busca import (
    CAMINHO_INDICE_ESQUEMA,
    CAMINHO_INDICE_GLOSSARIO,
    SECOES_FORA_DA_PONTUACAO,
    Indice,
    buscar_tabelas,
    carregar_tabelas_referenciadas,
    vetorizar,
)
from busca_bm25 import IndiceBM25
from fusao_rrf import IndiceHibrido

METODOS = ("embeddings", "bm25", "rrf")
# Método por índice: tabelas com embeddings e trechos com BM25, para todos os tipos.
POLITICA_PADRAO = {
    "quantitativa": {"tabelas": "embeddings"},
    "conceitual": {"trechos": "bm25"},
    "hibrida": {"tabelas": "embeddings", "trechos": "bm25"},
}
# Política anterior, só para comparação: perguntas híbridas com a fusão RRF nas duas buscas.
POLITICA_HIBRIDA_RRF = {
    "quantitativa": {"tabelas": "embeddings"},
    "conceitual": {"trechos": "bm25"},
    "hibrida": {"tabelas": "rrf", "trechos": "rrf"},
}


# Prepara a pergunta para o método: vetor, texto ou o par dos dois (fusão).
def _consulta(metodo: str, pergunta: str):
    if metodo == "embeddings":
        return vetorizar(pergunta)
    if metodo == "bm25":
        return pergunta
    return (vetorizar(pergunta), pergunta)


class BuscaRoteada:
    # Guarda a política e carrega as relações de chave estrangeira; os índices vêm sob demanda.
    def __init__(self, politica: dict = POLITICA_PADRAO):
        self.politica = politica
        self.referenciadas = carregar_tabelas_referenciadas()
        self._indices = {}

    # Devolve o método que a política indica para o tipo e o índice pedidos.
    def _metodo(self, tipo: str, onde: str) -> str:
        try:
            metodo = self.politica[tipo][onde]
        except KeyError:
            raise ValueError(f"a política não prevê busca de {onde} para o tipo {tipo}")
        if metodo not in METODOS:
            raise ValueError(f"método desconhecido: {metodo}")
        return metodo

    # Monta (uma vez) o índice de "tabelas" ou de "trechos" do método pedido.
    def _indice(self, onde: str, metodo: str):
        chave = (onde, metodo)
        if chave not in self._indices:
            esquema = onde == "tabelas"
            caminho = CAMINHO_INDICE_ESQUEMA if esquema else CAMINHO_INDICE_GLOSSARIO
            if metodo == "embeddings":
                indice = Indice(caminho)
            elif metodo == "bm25":
                indice = IndiceBM25(caminho, False, SECOES_FORA_DA_PONTUACAO if esquema else ())
            else:
                indice = IndiceHibrido(self._indice(onde, "embeddings"), self._indice(onde, "bm25"))
            self._indices[chave] = indice
        return self._indices[chave]

    # Devolve as k tabelas escolhidas mais as referenciadas por chave estrangeira.
    def tabelas(self, pergunta: str, tipo: str, k: int = 3) -> list[str]:
        metodo = self._metodo(tipo, "tabelas")
        return buscar_tabelas(self._indice("tabelas", metodo), _consulta(metodo, pergunta), k, self.referenciadas)

    # Devolve os k trechos do glossário (id, fonte, conteudo e pontuacao).
    def trechos(self, pergunta: str, tipo: str, k: int = 3) -> list[dict]:
        metodo = self._metodo(tipo, "trechos")
        return self._indice("trechos", metodo).buscar(_consulta(metodo, pergunta), k)
