"""
Etapa: Online.

Função: ponto único de busca do estágio online. Escolhe o método (embeddings, BM25 ou fusão RRF)
e o modelo de embeddings conforme configuracao.json, para o gerador de SQL e o sintetizador
buscarem sem saber qual método está em uso.

Descrição geral do funcionamento:
- A configuração (configuracao.py) diz qual modelo de embeddings usar, qual método busca em cada
  índice (esquema e glossário), se um tipo de pergunta troca o método, quantas tabelas (k_tabelas)
  e quantos trechos (k_trechos) entram no contexto e se o BM25 usa o radicalizador. Não há política
  escrita no código: mudar o comportamento é mudar o arquivo ou passar --set chave=valor.
- Os métodos são "embeddings" (busca.py, com o modelo escolhido), "bm25" (busca_bm25.py, k1 = 2 e
  b = 0,5) e "rrf" (fusão dos dois, fusao_rrf.py).
- O tipo da pergunta (quantitativa, conceitual ou hibrida) vem do classificador, que este módulo
  não chama. O tipo só serve para o método por tipo da configuração; quem decide se busca tabelas,
  trechos ou os dois é o orquestrador.
- Os índices são montados na primeira vez em que são usados, ou todos de uma vez em preaquecer(),
  para a primeira pergunta não pagar o custo de partida.
- obter_busca() devolve a busca única do processo, criada com a configuração do arquivo; o
  orquestrador pode trocá-la com definir_configuracao() (por exemplo, com sobrescritas de --set).
- politica() converte uma configuração em um quadro "tipo -> método das tabelas e dos trechos",
  usado por avaliar_baseline.py para montar as linhas "por índice" e "por tipo".

Dependências:
- pipeline/online/busca.py, busca_bm25.py, fusao_rrf.py e configuracao.py, e os índices e o esquema
  em embeddings/ (os índices de outros modelos ficam em embeddings/modelos/<chave>/);
- bibliotecas numpy, sentence-transformers e rank-bm25 (e nltk, com o corpus de stopwords, só se
  o BM25 usar o radicalizador).

Parâmetros: nenhum argumento. É um módulo importado.

Resultado: fornece BuscaConfiguravel (tabelas, trechos e preaquecer), obter_busca(),
definir_configuracao() e politica().
"""

import os
import time

from busca import (
    SECOES_FORA_DA_PONTUACAO,
    Indice,
    buscar_tabelas,
    caminho_indice,
    carregar_tabelas_referenciadas,
    definir_modelo,
    vetorizar,
)
from busca_bm25 import CAMINHO_BM25_ESQUEMA, CAMINHO_BM25_GLOSSARIO, IndiceBM25
from configuracao import INDICES, TIPOS, carregar, metodo_de_busca
from fusao_rrf import IndiceHibrido


# Prepara a pergunta para o método: vetor, texto ou o par dos dois (fusão).
def _consulta(metodo: str, pergunta: str):
    if metodo == "embeddings":
        return vetorizar(pergunta)
    if metodo == "bm25":
        return pergunta
    return (vetorizar(pergunta), pergunta)


class BuscaConfiguravel:
    # Guarda a configuração, fixa o modelo de embeddings e carrega as relações de chave estrangeira;
    # os índices vêm sob demanda.
    def __init__(self, config: dict | None = None):
        self.config = config or carregar()
        self.chave_modelo = self.config["modelo_embedding"]
        definir_modelo(self.chave_modelo)
        self.referenciadas = carregar_tabelas_referenciadas()
        self._indices = {}

    # Monta (uma vez) o índice "esquema" ou "glossario" do método pedido.
    def _indice(self, onde: str, metodo: str):
        chave = (onde, metodo)
        if chave not in self._indices:
            esquema = onde == "esquema"
            if metodo == "embeddings":
                caminho = caminho_indice(onde, self.chave_modelo)
                if not os.path.exists(caminho):
                    raise FileNotFoundError(
                        f"Índice {caminho} não encontrado. Gere-o com: "
                        f"python pipeline/offline/rodar_offline.py --modelos {self.chave_modelo}"
                    )
                indice = Indice(caminho)
            elif metodo == "bm25":
                caminho_bm25 = CAMINHO_BM25_ESQUEMA if esquema else CAMINHO_BM25_GLOSSARIO
                indice = IndiceBM25(
                    caminho_bm25, self.config["bm25"]["radicalizar"], SECOES_FORA_DA_PONTUACAO if esquema else ()
                )
            else:
                indice = IndiceHibrido(self._indice(onde, "embeddings"), self._indice(onde, "bm25"))
            self._indices[chave] = indice
        return self._indices[chave]

    # Devolve as k tabelas escolhidas (padrão: k_tabelas da configuração) mais as referenciadas por chave estrangeira.
    def tabelas(self, pergunta: str, tipo: str | None = None, k: int | None = None) -> list[str]:
        metodo = metodo_de_busca(self.config, "esquema", tipo)
        k = k or self.config["k_tabelas"]
        return buscar_tabelas(self._indice("esquema", metodo), _consulta(metodo, pergunta), k, self.referenciadas)

    # Devolve os k trechos do glossário (padrão: k_trechos da configuração), com id, fonte, conteudo e pontuacao.
    def trechos(self, pergunta: str, tipo: str | None = None, k: int | None = None) -> list[dict]:
        metodo = metodo_de_busca(self.config, "glossario", tipo)
        k = k or self.config["k_trechos"]
        return self._indice("glossario", metodo).buscar(_consulta(metodo, pergunta), k)

    # Carrega o modelo e monta todos os índices que a configuração pode usar, e devolve o tempo gasto.
    def preaquecer(self) -> float:
        inicio = time.perf_counter()
        usa_vetor = False
        for onde in INDICES:
            bloco = self.config["busca"][onde]
            for metodo in {bloco["metodo"], *bloco["por_tipo"].values()}:
                self._indice(onde, metodo)
                usa_vetor = usa_vetor or metodo in ("embeddings", "rrf")
        if usa_vetor:
            # A primeira vetorização é mais lenta; fazê-la aqui tira esse custo da primeira pergunta.
            vetorizar("aquecimento")
        return time.perf_counter() - inicio


_busca = None


# Devolve a busca única do processo, criada na primeira chamada com a configuração do arquivo.
def obter_busca() -> BuscaConfiguravel:
    global _busca
    if _busca is None:
        _busca = BuscaConfiguravel()
    return _busca


# Troca a busca única por uma criada com a configuração dada (os índices são montados de novo).
def definir_configuracao(config: dict) -> BuscaConfiguravel:
    global _busca
    _busca = BuscaConfiguravel(config)
    return _busca


# Converte uma configuração em "tipo -> {tabelas: método, trechos: método}", para a avaliação.
def politica(config: dict) -> dict:
    return {
        tipo: {
            "tabelas": metodo_de_busca(config, "esquema", tipo),
            "trechos": metodo_de_busca(config, "glossario", tipo),
        }
        for tipo in TIPOS
    }
