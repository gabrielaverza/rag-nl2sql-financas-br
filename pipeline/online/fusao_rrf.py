"""
Etapa: Online (baseline de recuperação).

Função: fusão por posição (Reciprocal Rank Fusion, RRF) dos rankings da busca por embeddings e da
busca por palavras (BM25), para avaliar a busca híbrida.

Descrição geral do funcionamento:
- Cada ranking dá a cada item a nota 1 / (k + posição), com a posição contada a partir de 1, e a
  nota final é a soma das notas dos rankings (Cormack, Clarke e Büttcher, 2009, p. 1). A fusão usa só
  as posições, e não as pontuações, que têm escalas diferentes (cosseno e BM25).
- A constante k vale 60, o valor fixado no estudo piloto do artigo, sem ajuste nos dados deste
  trabalho.
- Empates de nota ficam na ordem em que o item apareceu primeiro (primeiro ranking, o denso).
- IndiceHibrido tem a mesma interface de Indice e de IndiceBM25 (buscar e buscar_por_fonte), então
  buscar_tabelas de busca.py funciona sem mudança. A consulta é um par (vetor da pergunta, texto da
  pergunta), o primeiro para a busca densa e o segundo para o BM25.
- No glossário, a fusão é feita sobre os trechos, como na busca híbrida do MimirRAG (Samuelsen et
  al., 2026, seção 4.4.2). No esquema, é feita sobre as tabelas, antes da expansão por chave
  estrangeira; isso é uma extensão exploratória, pois o MimirRAG não a aplica a tabelas.

Dependências: nenhuma além dos índices que recebe.

Parâmetros: nenhum argumento. É um módulo importado por avaliar_baseline.py.

Resultado: fornece CONSTANTE_RRF, fundir() e a classe IndiceHibrido.
"""

CONSTANTE_RRF = 60


# Funde rankings (listas de itens, do melhor ao pior) e devolve (item, nota) em ordem decrescente de nota.
def fundir(rankings: list[list], constante: int = CONSTANTE_RRF) -> list[tuple]:
    notas = {}
    for ranking in rankings:
        for posicao, item in enumerate(ranking, start=1):
            notas[item] = notas.get(item, 0.0) + 1 / (constante + posicao)
    return sorted(notas.items(), key=lambda x: -x[1])


class IndiceHibrido:
    # Junta um índice denso (embeddings) e um esparso (BM25) sobre os mesmos registros.
    def __init__(self, denso, esparso, constante: int = CONSTANTE_RRF):
        self.denso = denso
        self.esparso = esparso
        self.constante = constante
        self.registros = denso.registros

    # Devolve os k registros de maior nota fundida.
    def buscar(self, consulta: tuple, k: int) -> list[dict]:
        vetor, texto = consulta
        n = len(self.registros)
        densos = self.denso.buscar(vetor, n)
        esparsos = self.esparso.buscar(texto, n)
        por_id = {r["id"]: r for r in densos}
        fundidos = fundir([[r["id"] for r in densos], [r["id"] for r in esparsos]], self.constante)
        return [{**por_id[i], "pontuacao": nota} for i, nota in fundidos[:k]]

    # Devolve as k fontes (tabelas) de maior nota fundida.
    def buscar_por_fonte(self, consulta: tuple, k: int) -> list[dict]:
        vetor, texto = consulta
        n = len({r["fonte"] for r in self.registros})
        listas = [
            [r["fonte"] for r in self.denso.buscar_por_fonte(vetor, n)],
            [r["fonte"] for r in self.esparso.buscar_por_fonte(texto, n)],
        ]
        return [{"fonte": fonte, "pontuacao": nota} for fonte, nota in fundir(listas, self.constante)[:k]]
