"""
Etapa: Online (medição de tempo da busca).

Função: mede quanto tempo custa montar e consultar cada busca (embeddings, BM25 e fusão RRF), para
saber se adotar o BM25 afeta o desempenho do programa.

Descrição geral do funcionamento:
- Custos de partida, pagos uma vez quando o programa inicia: importar as bibliotecas do BM25
  (nltk e rank-bm25), ler a lista de stopwords, montar o índice do BM25 do esquema e do glossário
  (que inclui ler o JSON do índice), e, para comparar, importar as bibliotecas de embeddings,
  carregar o modelo E5-base e ler os índices vetoriais.
- Custo por pergunta: tempo de uma busca de 3 trechos no glossário e de uma busca de tabelas, com
  cada método, nas perguntas do conjunto oficial (depois de uma rodada de aquecimento). O tempo do
  E5-base inclui transformar a pergunta em vetor.
- Mede só a busca; o LLM (classificador, SQL, síntese e validação) fica de fora.

Dependências: os índices e o esquema em embeddings/, dados/perguntas_validacao.json; bibliotecas do pipeline mais rank-bm25 e nltk.

Parâmetros: nenhum argumento.

Resultado: imprime o relatório e grava metricas/tempo_busca.md.

Executar de dentro de rag-nl2sql-financas-br/:
    python pipeline/online/medir_tempo_busca.py
"""

import json
import os
import platform
import sys
import time
from datetime import datetime
from statistics import mean, median

CAMINHOS_PERGUNTAS = ("dados/perguntas_validacao.json",)
PASTA_SAIDA = "metricas"
RODADAS = 3


# Executa uma função e devolve o resultado e o tempo gasto, em segundos.
def cronometrar(funcao):
    inicio = time.perf_counter()
    resultado = funcao()
    return resultado, time.perf_counter() - inicio


# Roda a busca para todas as perguntas, RODADAS vezes, e devolve o tempo de cada chamada em milissegundos.
def tempos_por_pergunta(funcao, perguntas: list[str]) -> list[float]:
    tempos = []
    for _ in range(RODADAS):
        for pergunta in perguntas:
            _, t = cronometrar(lambda: funcao(pergunta))
            tempos.append(t * 1000)
    return tempos


# Escreve um número com vírgula decimal.
def fmt(valor: float, casas: int = 2) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    perguntas = []
    for caminho in CAMINHOS_PERGUNTAS:
        with open(caminho, encoding="utf-8") as f:
            perguntas += [p["pergunta"] for p in json.load(f)]

    # Custos de partida do BM25 (medidos antes de importar as bibliotecas de embeddings).
    busca_bm25, t_import_bm25 = cronometrar(lambda: __import__("busca_bm25"))

    # As constantes dos caminhos vêm de busca.py, que importa as bibliotecas de embeddings.
    busca, t_import_busca = cronometrar(lambda: __import__("busca"))
    fusao_rrf = __import__("fusao_rrf")

    _, t_json = cronometrar(lambda: json.load(open(busca.CAMINHO_INDICE_GLOSSARIO, encoding="utf-8")))
    bm25_esquema, t_bm25_esquema = cronometrar(
        lambda: busca_bm25.IndiceBM25(busca_bm25.CAMINHO_BM25_ESQUEMA, False, busca.SECOES_FORA_DA_PONTUACAO))
    bm25_glossario, t_bm25_glossario = cronometrar(
        lambda: busca_bm25.IndiceBM25(busca_bm25.CAMINHO_BM25_GLOSSARIO, False))
    _, t_modelo = cronometrar(busca.carregar_modelo)
    e5_esquema, t_e5_esquema = cronometrar(lambda: busca.Indice(busca.CAMINHO_INDICE_ESQUEMA))
    e5_glossario, t_e5_glossario = cronometrar(lambda: busca.Indice(busca.CAMINHO_INDICE_GLOSSARIO))
    referenciadas = busca.carregar_tabelas_referenciadas()
    rrf_glossario = fusao_rrf.IndiceHibrido(e5_glossario, bm25_glossario)
    rrf_esquema = fusao_rrf.IndiceHibrido(e5_esquema, bm25_esquema)

    buscas = {
        "E5-base, trechos (inclui vetorizar a pergunta)": lambda q: e5_glossario.buscar(busca.vetorizar(q), 3),
        "BM25, trechos": lambda q: bm25_glossario.buscar(q, 3),
        "RRF, trechos (E5 e BM25 e a fusão)": lambda q: rrf_glossario.buscar((busca.vetorizar(q), q), 3),
        "E5-base, tabelas (inclui vetorizar a pergunta)": lambda q: busca.buscar_tabelas(e5_esquema, busca.vetorizar(q), 3, referenciadas),
        "BM25, tabelas": lambda q: busca.buscar_tabelas(bm25_esquema, q, 3, referenciadas),
        "RRF, tabelas (E5 e BM25 e a fusão)": lambda q: busca.buscar_tabelas(rrf_esquema, (busca.vetorizar(q), q), 3, referenciadas),
    }
    for funcao in buscas.values():
        funcao(perguntas[0])
    medidas = {nome: tempos_por_pergunta(funcao, perguntas) for nome, funcao in buscas.items()}

    t = ["# Tempo da busca: partida e por pergunta", ""]
    t += [
        f"- Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')} por `python pipeline/online/medir_tempo_busca.py`.",
        f"- Máquina: {platform.platform()}, Python {platform.python_version()}; os embeddings rodam na CPU. Uma máquina e uma execução, então os valores são só uma ordem de grandeza.",
        f"- Por pergunta: {len(perguntas)} perguntas do conjunto oficial, {RODADAS} rodadas cada, depois de uma chamada de aquecimento. O LLM (classificador, SQL, síntese, validação) não entra; o fluxo completo leva cerca de 29 s por pergunta.",
        "",
        "## 1. Custos de partida (uma vez, quando o programa inicia)",
        "",
        "| Etapa | Tempo (s) |",
        "|---|---|",
        f"| Importar rank-bm25 (o nltk só é importado se houver radicalização) | {fmt(t_import_bm25)} |",
        f"| Montar o índice BM25 do esquema ({len(bm25_esquema.registros)} registros, lê o índice de palavras) | {fmt(t_bm25_esquema)} |",
        f"| Montar o índice BM25 do glossário ({f'{len(bm25_glossario.registros):,}'.replace(',', '.')} trechos, lê o índice de palavras) | {fmt(t_bm25_glossario)} |",
        f"| Para comparar: ler o JSON do índice vetorial do glossário ({os.path.getsize(busca.CAMINHO_INDICE_GLOSSARIO) / 1e6:.1f} MB) | {fmt(t_json)} |",
        f"| Para comparar: importar as bibliotecas de embeddings | {fmt(t_import_busca)} |",
        f"| Para comparar: carregar o modelo E5-base | {fmt(t_modelo)} |",
        f"| Para comparar: ler o índice vetorial do glossário | {fmt(t_e5_glossario)} |",
        "",
        "## 2. Custo por pergunta (busca de 3 trechos e de tabelas)",
        "",
        "| Busca | Média (ms) | Mediana (ms) | Máximo (ms) |",
        "|---|---|---|---|",
    ]
    for nome, tempos in medidas.items():
        t.append(f"| {nome} | {fmt(mean(tempos))} | {fmt(median(tempos))} | {fmt(max(tempos))} |")
    t += [
        "",
        "## 3. Ressalvas",
        "",
        "- O BM25 lê um índice de palavras próprio (sem vetores), gerado no estágio offline, por isso a partida é curta; com radicalização, o nltk é importado na primeira busca (cerca de 1,7 s a mais, medido à parte).",
        "- A primeira chamada ao E5-base depois de carregar o modelo é mais lenta; a medida por pergunta vem depois do aquecimento.",
        "- Em uso real, o custo de partida é pago uma vez por processo; `main.py` hoje roda uma pergunta por processo, então pagaria o custo de partida a cada pergunta (ponto já registrado como melhoria possível).",
        "",
    ]
    texto = "\n".join(t)
    print(texto)
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    with open(os.path.join(PASTA_SAIDA, "tempo_busca.md"), "w", encoding="utf-8") as f:
        f.write(texto)


if __name__ == "__main__":
    main()
