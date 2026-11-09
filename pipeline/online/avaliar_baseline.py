"""
Etapa: Online (avaliação do baseline de recuperação).

Função: compara a busca por embeddings (E5-base) com o baseline BM25 (sem e com stemmer) e com a
fusão dos dois por RRF, nas mesmas perguntas.

Descrição geral do funcionamento:
- As premissas são as mesmas em todos os métodos: mesmas perguntas, mesmos registros indexados,
  mesma pontuação por tabela (melhor registro de resumo ou entidades) e mesma expansão por chave
  estrangeira. Mudam só o método de pontuação e, no BM25, o pré-processamento do texto.
- Tabelas (quantitativas e híbridas): Recall@k, cobertura total@k (todas as tabelas esperadas na
  lista) e MRR, com e sem a tabela companhias entre as esperadas. Sem ela, mede-se só o acerto da
  demonstração, pois companhias entra na lista pela regra de chave estrangeira.
- Trechos dos normativos (conceituais e híbridas): Hit@k, Recall@k, Precision@k, nDCG@k e MRR.
  Um trecho é relevante se contém o texto da evidência (como em avaliar_recuperacao.py). O nDCG
  usa a fórmula do monografia.tex (DCG com o desconto 1/log2(i) a partir da posição 2).
- Híbridas: contagem das perguntas com as tabelas completas e o trecho relevante ao mesmo tempo
  no top-k.
- Por índice e por tipo: duas políticas, escritas como configurações do configuracao.json (tabelas E5 e
  trechos BM25; e a anterior, com RRF nas híbridas) e aplicadas com o tipo correto de cada pergunta.
- Piso aleatório: valor esperado de uma lista sorteada ao acaso, por fórmula exata (sem sorteio).
- Comparação pareada por pergunta: diferença média, intervalo de 95% por bootstrap (10.000
  reamostragens, semente 42) e vitórias, derrotas e empates.

Dependências:
- pipeline/online/busca.py, busca_bm25.py, fusao_rrf.py, busca_configuravel.py, configuracao.py e avaliar_recuperacao.py, e os índices e o esquema em
  embeddings/;
- bibliotecas numpy, sentence-transformers, rank-bm25 e nltk (com o corpus de stopwords);
- arquivo de perguntas em dados/ (por padrão, dados/perguntas_validacao.json).

Parâmetros: caminho opcional do arquivo de perguntas, como primeiro argumento.

Resultado: imprime o relatório e grava, em metricas/, um arquivo .md com as tabelas e um .csv com
as métricas por pergunta e por método.

Executar de dentro de rag-nl2sql-financas-br/:
    python pipeline/online/avaliar_baseline.py
    python pipeline/online/avaliar_baseline.py dados/perguntas_dfp.json
"""

import csv
import json
import math
import os
import sys
from datetime import datetime
from importlib.metadata import version
from math import comb

import numpy as np

from avaliar_recuperacao import normalizar
from busca import (
    CAMINHO_INDICE_ESQUEMA,
    CAMINHO_INDICE_GLOSSARIO,
    SECOES_FORA_DA_PONTUACAO,
    Indice,
    buscar_tabelas,
    carregar_tabelas_referenciadas,
    vetorizar,
)
from busca_bm25 import (
    B_BIBLIOTECA,
    B_REFERENCIA,
    CAMINHO_BM25_ESQUEMA,
    CAMINHO_BM25_GLOSSARIO,
    K1_BIBLIOTECA,
    K1_REFERENCIA,
    IndiceBM25,
)
from busca_configuravel import politica
from configuracao import carregar as carregar_configuracao
from fusao_rrf import CONSTANTE_RRF, IndiceHibrido

CAMINHO_PERGUNTAS = "dados/perguntas_validacao.json"
PASTA_SAIDA = "metricas"
KS_TABELAS = (1, 3, 5)
KS_TRECHOS = (1, 3, 5, 10, 20)
KS_CONJUNTA = (3, 5)
TABELA_COMUM = "companhias"
AMOSTRAS_BOOTSTRAP = 10000
SEMENTE = 42
NOME_PISO = "Piso aleatório"
NOME_E5 = "E5-base (embeddings)"
NOME_BM25 = "BM25 sem stemmer"
NOME_BM25_STEMMER = "BM25 com stemmer (Snowball)"
NOME_BM25_BIBLIOTECA = "BM25 sem stemmer (parâmetros da biblioteca)"
NOME_RRF = "RRF (E5 + BM25)"
NOME_ROTEADO = "Por índice (tabelas E5, trechos BM25)"
NOME_ROTEADO_RRF = "Por tipo (híbrida com RRF)"
NOME_DO_METODO = {"embeddings": NOME_E5, "bm25": NOME_BM25, "rrf": NOME_RRF}

# As duas linhas "por índice" e "por tipo" são configurações do configuracao.json, escritas por inteiro
# aqui para não dependerem do que estiver no arquivo: método por índice (tabelas com embeddings e trechos
# com BM25, igual para todos os tipos) e a política anterior, com RRF nas duas buscas das híbridas.
SOBRESCRITAS_POR_INDICE = [
    "busca.esquema.metodo=embeddings", "busca.esquema.por_tipo={}",
    "busca.glossario.metodo=bm25", "busca.glossario.por_tipo={}",
]
SOBRESCRITAS_POR_TIPO_RRF = SOBRESCRITAS_POR_INDICE + [
    "busca.esquema.por_tipo.hibrida=rrf", "busca.glossario.por_tipo.hibrida=rrf",
]

# Pares comparados: (chave da métrica, descrição).
METRICAS_COMPARADAS = [
    ("tab_todos@3", "Tabelas: todas no top-3"),
    ("tabsem_todos@3", "Demonstração no top-3 (sem companhias)"),
    ("tab_rr", "Tabelas: MRR"),
    ("glo_hit@3", "Trechos: Hit@3"),
    ("glo_hit@5", "Trechos: Hit@5"),
    ("glo_rr", "Trechos: MRR"),
    ("glo_ndcg@10", "Trechos: nDCG@10"),
    ("hib_conj@3", "Híbridas: tabelas e trecho no top-3"),
    ("hib_conjsem@3", "Híbridas: demonstração e trecho no top-3"),
]


class Metodo:
    # Reúne os dois índices de um método e a função que prepara a pergunta (vetor ou texto).
    def __init__(self, nome: str, esquema, glossario, preparar, lexical: bool):
        self.nome = nome
        self.esquema = esquema
        self.glossario = glossario
        self.preparar = preparar
        self.lexical = lexical


# Cria o método de embeddings, as variantes do BM25 e a fusão dos dois por RRF.
def criar_metodos() -> list[Metodo]:
    e5 = Metodo(NOME_E5, Indice(CAMINHO_INDICE_ESQUEMA), Indice(CAMINHO_INDICE_GLOSSARIO), vetorizar, False)
    metodos = [e5]
    variantes = (
        (NOME_BM25, False, K1_REFERENCIA, B_REFERENCIA),
        (NOME_BM25_STEMMER, True, K1_REFERENCIA, B_REFERENCIA),
        (NOME_BM25_BIBLIOTECA, False, K1_BIBLIOTECA, B_BIBLIOTECA),
    )
    for nome, radicalizar, k1, b in variantes:
        esquema = IndiceBM25(CAMINHO_BM25_ESQUEMA, radicalizar, SECOES_FORA_DA_PONTUACAO, k1, b)
        glossario = IndiceBM25(CAMINHO_BM25_GLOSSARIO, radicalizar, k1=k1, b=b)
        metodos.append(Metodo(nome, esquema, glossario, lambda texto: texto, True))
    # A fusão usa o BM25 sem stemmer, com os parâmetros da referência.
    bm25 = metodos[1]
    rrf = Metodo(
        NOME_RRF, IndiceHibrido(e5.esquema, bm25.esquema), IndiceHibrido(e5.glossario, bm25.glossario),
        lambda texto: (vetorizar(texto), texto), False,
    )
    return metodos + [rrf]


# ---------------------------------------------------------------------------
# 1) Métricas de uma pergunta
# ---------------------------------------------------------------------------

# Desconto do nDCG do monografia.tex: 1 na primeira posição e 1/log2(i) depois.
def desconto(posicao: int) -> float:
    return 1.0 if posicao == 1 else 1 / math.log2(posicao)


# Soma o ganho descontado de uma lista de relevâncias (0 ou 1).
def dcg(relevancias: list) -> float:
    return sum(r * desconto(i) for i, r in enumerate(relevancias, start=1))


# Calcula Hit@k, Recall@k, Precision@k, nDCG@k e o inverso da posição do primeiro relevante.
def metricas_trechos(relevancias: list[bool]) -> dict:
    total = sum(relevancias)
    ideal = sorted(relevancias, reverse=True)
    resultado = {}
    for k in KS_TRECHOS:
        achados = sum(relevancias[:k])
        resultado[f"glo_hit@{k}"] = float(achados > 0)
        resultado[f"glo_recall@{k}"] = achados / total
        resultado[f"glo_prec@{k}"] = achados / k
        resultado[f"glo_ndcg@{k}"] = dcg(relevancias[:k]) / dcg(ideal[:k])
    resultado["glo_rr"] = 1 / next(i for i, r in enumerate(relevancias, start=1) if r)
    return resultado


# Calcula Recall@k, cobertura total@k e o inverso da posição da primeira tabela esperada.
def metricas_tabelas(prefixo: str, ranking: list[str], listas: dict, alvo: list[str]) -> dict:
    resultado = {}
    for k in KS_TABELAS:
        achadas = len(set(alvo) & set(listas[k]))
        resultado[f"{prefixo}_recall@{k}"] = achadas / len(alvo)
        resultado[f"{prefixo}_todos@{k}"] = float(achadas == len(alvo))
    resultado[f"{prefixo}_rr"] = 1 / (min(ranking.index(t) for t in alvo) + 1)
    return resultado


# Avalia uma pergunta em um método e devolve as métricas e as posições dos itens esperados.
def avaliar_pergunta(metodo: Metodo, p: dict, referenciadas: dict, n_tabelas: int, n_trechos: int, ids_relevantes: set) -> dict:
    consulta = metodo.preparar(p["pergunta"])
    m, posicoes = {}, {}
    if p["tabelas_esperadas"]:
        completo = metodo.esquema.buscar_por_fonte(consulta, n_tabelas)
        ranking = [r["fonte"] for r in completo]
        listas = {k: buscar_tabelas(metodo.esquema, consulta, k, referenciadas) for k in KS_TABELAS}
        esperadas = p["tabelas_esperadas"]
        sem_comum = [t for t in esperadas if t != TABELA_COMUM]
        m.update(metricas_tabelas("tab", ranking, listas, esperadas))
        if sem_comum:
            m.update(metricas_tabelas("tabsem", ranking, listas, sem_comum))
        for k in KS_TABELAS:
            m[f"tab_tam@{k}"] = float(len(listas[k]))
        if metodo.lexical:
            m["tab_zero"] = float(completo[0]["pontuacao"] <= 0)
        posicoes["tabelas"] = {t: ranking.index(t) + 1 for t in esperadas}
    if p["evidencia"]:
        ranking_trechos = metodo.glossario.buscar(consulta, n_trechos)
        relevancias = [r["id"] in ids_relevantes for r in ranking_trechos]
        m.update(metricas_trechos(relevancias))
        if metodo.lexical:
            m["glo_zero"] = float(ranking_trechos[0]["pontuacao"] <= 0)
        posicoes["trechos"] = [i for i, r in enumerate(relevancias, start=1) if r][:5]
    if p["tipo"] == "hibrida" and p["tabelas_esperadas"] and p["evidencia"]:
        juntar_hibrida(m)
    return {"id": p["id"], "tipo": p["tipo"], "m": m, "posicoes": posicoes}


# Acrescenta as métricas das híbridas: tabelas completas, trecho relevante e os dois juntos.
def juntar_hibrida(m: dict):
    for k in KS_CONJUNTA:
        m[f"hib_tab@{k}"] = m[f"tab_todos@{k}"]
        m[f"hib_glo@{k}"] = m[f"glo_hit@{k}"]
        m[f"hib_conj@{k}"] = m[f"tab_todos@{k}"] * m[f"glo_hit@{k}"]
        if f"tabsem_todos@{k}" in m:
            m[f"hib_conjsem@{k}"] = m[f"tabsem_todos@{k}"] * m[f"glo_hit@{k}"]


# ---------------------------------------------------------------------------
# 2) Piso aleatório (fórmulas exatas)
# ---------------------------------------------------------------------------

# Valor esperado de 1/posição do primeiro relevante, com m relevantes entre n itens em ordem aleatória.
def piso_rr(n: int, m: int) -> float:
    return sum(comb(n - j, m - 1) / comb(n, m) / j for j in range(1, n - m + 2))


# Probabilidade de uma lista de k itens sorteados conter os r itens necessários, entre n.
def piso_cobertura(n: int, k: int, r: int) -> float:
    return comb(n - r, k - r) / comb(n, k) if k >= r else 0.0


# Probabilidade de uma lista de k itens sorteados conter ao menos um dos m relevantes, entre n.
def piso_hit(n: int, m: int, k: int) -> float:
    return 1 - comb(n - m, k) / comb(n, k)


# Valor esperado do nDCG@k de uma ordem aleatória, com m relevantes entre n itens.
def piso_ndcg(n: int, m: int, k: int) -> float:
    esperado = (m / n) * dcg([1] * k)
    return esperado / dcg([1] * min(m, k))


# Calcula o piso aleatório de uma pergunta, com as mesmas chaves das métricas dos métodos.
def avaliar_piso(p: dict, n_tabelas: int, n_trechos: int, n_relevantes: int) -> dict:
    m = {}
    if p["tabelas_esperadas"]:
        esperadas = p["tabelas_esperadas"]
        comum = int(TABELA_COMUM in esperadas)
        outras = len(esperadas) - comum
        for k in KS_TABELAS:
            # A tabela companhias entra sempre na lista (regra de chave estrangeira).
            m[f"tab_recall@{k}"] = (comum + outras * k / n_tabelas) / len(esperadas)
            m[f"tab_todos@{k}"] = piso_cobertura(n_tabelas, k, outras)
            if outras:
                m[f"tabsem_recall@{k}"] = k / n_tabelas
                m[f"tabsem_todos@{k}"] = piso_cobertura(n_tabelas, k, outras)
        m["tab_rr"] = piso_rr(n_tabelas, len(esperadas))
        if outras:
            m["tabsem_rr"] = piso_rr(n_tabelas, outras)
    if p["evidencia"]:
        for k in KS_TRECHOS:
            m[f"glo_hit@{k}"] = piso_hit(n_trechos, n_relevantes, k)
            m[f"glo_recall@{k}"] = k / n_trechos
            m[f"glo_prec@{k}"] = n_relevantes / n_trechos
            m[f"glo_ndcg@{k}"] = piso_ndcg(n_trechos, n_relevantes, k)
        m["glo_rr"] = piso_rr(n_trechos, n_relevantes)
    if p["tipo"] == "hibrida" and p["tabelas_esperadas"] and p["evidencia"]:
        juntar_hibrida(m)
    return {"id": p["id"], "tipo": p["tipo"], "m": m, "posicoes": {}}


# ---------------------------------------------------------------------------
# 3) Agregação e comparação pareada
# ---------------------------------------------------------------------------

# Monta o resultado da busca por tipo: tabelas e trechos de cada pergunta vêm do método que a política indica.
def montar_roteado(perguntas: list[dict], resultados: dict, politica: dict) -> list[dict]:
    roteado = []
    for i, p in enumerate(perguntas):
        passos = politica.get(p["tipo"], {})
        m, posicoes = {}, {}
        for onde, prefixos in (("tabelas", ("tab_", "tabsem_")), ("trechos", ("glo_",))):
            metodo = passos.get(onde)
            if metodo is None:
                continue
            origem = resultados[NOME_DO_METODO[metodo]][i]
            m.update({c: v for c, v in origem["m"].items() if c.startswith(prefixos)})
            if onde in origem["posicoes"]:
                posicoes[onde] = origem["posicoes"][onde]
        if p["tipo"] == "hibrida" and p["tabelas_esperadas"] and p["evidencia"]:
            juntar_hibrida(m)
        roteado.append({"id": p["id"], "tipo": p["tipo"], "m": m, "posicoes": posicoes})
    return roteado


# Média de uma métrica sobre as perguntas que a têm, e quantas perguntas entraram.
def media(resultados: list[dict], chave: str):
    valores = [r["m"][chave] for r in resultados if r["m"].get(chave) is not None]
    return (sum(valores) / len(valores), len(valores)) if valores else (None, 0)


# Soma de uma métrica de 0 ou 1, e quantas perguntas entraram.
def contagem(resultados: list[dict], chave: str):
    valores = [r["m"][chave] for r in resultados if r["m"].get(chave) is not None]
    return sum(valores), len(valores)


# Compara dois métodos pergunta a pergunta: diferença média, IC de 95% por bootstrap e vitórias.
def comparar(a: list[dict], b: list[dict], chave: str, rng) -> dict | None:
    dif = np.array([x["m"][chave] - y["m"][chave] for x, y in zip(a, b) if chave in x["m"] and chave in y["m"]])
    if len(dif) == 0:
        return None
    amostras = rng.choice(dif, size=(AMOSTRAS_BOOTSTRAP, len(dif)), replace=True).mean(axis=1)
    inferior, superior = np.percentile(amostras, [2.5, 97.5])
    return {
        "n": len(dif), "dif": float(dif.mean()), "ic": (float(inferior), float(superior)),
        "vitorias": int((dif > 0).sum()), "derrotas": int((dif < 0).sum()), "empates": int((dif == 0).sum()),
    }


# ---------------------------------------------------------------------------
# 4) Relatório
# ---------------------------------------------------------------------------

# Escreve um número com vírgula decimal, ou "n/d" se não houver.
def fmt(valor, casas: int = 2) -> str:
    return "n/d" if valor is None else f"{valor:.{casas}f}".replace(".", ",")


# Monta uma tabela em Markdown.
def tabela_md(cabecalho: list[str], linhas: list[list[str]]) -> list[str]:
    saida = ["| " + " | ".join(cabecalho) + " |", "|" + "|".join("---" for _ in cabecalho) + "|"]
    saida += ["| " + " | ".join(linha) + " |" for linha in linhas]
    return saida + [""]


# Uma linha por método com as médias das chaves pedidas.
def linhas_medias(resultados: dict, chaves: list[str]) -> list[list[str]]:
    return [[nome] + [fmt(media(lista, c)[0]) for c in chaves] for nome, lista in resultados.items()]


# Uma linha por método com as contagens "x/n" das chaves pedidas (no piso, o valor esperado).
def linhas_contagens(resultados: dict, chaves: list[str]) -> list[list[str]]:
    linhas = []
    for nome, lista in resultados.items():
        linha = [nome]
        for c in chaves:
            total, n = contagem(lista, c)
            linha.append(f"{fmt(total) if nome == NOME_PISO else int(total)}/{n}")
        linhas.append(linha)
    return linhas


# Monta o texto do relatório em Markdown.
def montar_relatorio(caminho_perguntas: str, perguntas: list[dict], resultados: dict, metodos: list[Metodo], comparacoes: list[str]) -> str:
    n_tab = max(media(l, "tab_rr")[1] for l in resultados.values())
    n_sem = max(media(l, "tabsem_rr")[1] for l in resultados.values())
    n_glo = max(media(l, "glo_rr")[1] for l in resultados.values())
    n_hib = max(contagem(l, "hib_conj@3")[1] for l in resultados.values())
    bm25 = metodos[1]
    diag_e = bm25.esquema.diagnostico()
    diag_g = bm25.glossario.diagnostico()
    eps = bm25.esquema.bm25.epsilon
    params = "; ".join(f"{m.nome}: k1 = {fmt(m.esquema.bm25.k1)}, b = {fmt(m.esquema.bm25.b)}" for m in metodos if m.lexical)

    t = [f"# Baseline BM25 e fusão RRF contra E5-base: recuperação ({os.path.basename(caminho_perguntas)})", ""]
    t += [
        f"- Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')} por `python pipeline/online/avaliar_baseline.py {caminho_perguntas}`.",
        f"- Perguntas: {len(perguntas)} no arquivo; {n_tab} com tabelas esperadas, {n_glo} com evidência, {n_hib} híbridas com os dois.",
        f"- Bibliotecas: rank-bm25 {version('rank-bm25')}, nltk {version('nltk')}, sentence-transformers {version('sentence-transformers')}, numpy {version('numpy')}.",
        f"- BM25: Okapi BM25 da rank_bm25, sem ajuste nos dados. Parâmetros: {params}; epsilon = {fmt(eps)} (padrão da biblioteca). Os valores k1 = 2 e b = 0,5 são a combinação comum citada por Robertson e Zaragoza (2009, p. 31); os da biblioteca (k1 = 1,5 e b = 0,75) ficam dentro das faixas 0,5 < b < 0,8 e 1,2 < k1 < 2 (p. 30) e entram só como verificação de sensibilidade.",
        f"- RRF: fusão por posição dos rankings do E5-base e do BM25 sem stemmer; a nota de cada item é a soma de 1 / (k + posição), com k = {CONSTANTE_RRF}, o valor do artigo original (Cormack, Clarke e Büttcher, 2009), sem ajuste. No glossário reproduz a busca híbrida do MimirRAG; no esquema é uma extensão exploratória (as tabelas fundidas recebem depois a expansão por chave estrangeira).",
        "- Por índice: a política com as tabelas buscadas pelo E5-base e os trechos pelo BM25 (BM25 sem stemmer, k1 = 2 e b = 0,5), para todos os tipos de pergunta, aplicada com o tipo correto de cada pergunta (sem erro do classificador). Foi escolhida pela autora depois de ver os resultados dos métodos nestes mesmos conjuntos, então esta linha NÃO é uma confirmação independente: só a avaliação em perguntas novas pode confirmá-la.",
        "- Por tipo (híbrida com RRF): a política anterior (quantitativa: tabelas com E5; conceitual: trechos com BM25; híbrida: tabelas e trechos com RRF), mantida só para comparação.",
        "- Pré-processamento do BM25: minúsculas, palavras (letras e dígitos), stopwords do português do NLTK, stemmer Snowball do português só na variante com stemmer, acentos removidos no fim. O E5-base recebe a pergunta com o prefixo `query: `.",
        "- Premissas iguais em todos os métodos: mesmas perguntas, mesmos registros indexados, pontuação da tabela pelo melhor registro de resumo ou entidades, expansão por chave estrangeira, mesmos k.",
        f"- Índice de esquema no BM25 (só resumo e entidades): {diag_e['registros']} registros, {diag_e['palavras']} palavras, {diag_e['idf_negativo']} em mais da metade dos registros (IDF negativo trocado por epsilon vezes o IDF médio = {fmt(diag_e['piso_idf'], 3)}; IDF médio = {fmt(diag_e['idf_medio'], 3)}), na variante sem stemmer.",
        f"- Índice do glossário no BM25: {diag_g['registros']} trechos, {diag_g['palavras']} palavras, {diag_g['idf_negativo']} em mais da metade dos trechos, na variante sem stemmer.",
        "",
    ]

    chaves_com = [f"tab_recall@{k}" for k in KS_TABELAS] + [f"tab_todos@{k}" for k in KS_TABELAS] + ["tab_rr"]
    chaves_sem = [c.replace("tab_", "tabsem_") for c in chaves_com]
    cab_tab = ["Método"] + [f"Recall@{k}" for k in KS_TABELAS] + [f"Todas@{k}" for k in KS_TABELAS] + ["MRR"]
    t += [f"## 1. Tabelas (esquema, {n_tab} perguntas)", ""]
    t += ["### 1.1 Com a tabela companhias entre as esperadas (como no pipeline)", ""]
    t += tabela_md(cab_tab, linhas_medias(resultados, chaves_com))
    tamanhos = ", ".join(f"{nome}: {fmt(media(lista, 'tab_tam@3')[0], 1)} (k = 3) e {fmt(media(lista, 'tab_tam@5')[0], 1)} (k = 5)" for nome, lista in resultados.items() if nome != NOME_PISO)
    t += [f"Tamanho médio da lista (as k escolhidas mais as referenciadas por chave estrangeira): {tamanhos}.", ""]
    t += [f"### 1.2 Sem companhias (só o acerto da demonstração, {n_sem} perguntas)", ""]
    t += tabela_md(cab_tab, linhas_medias(resultados, chaves_sem))
    zeros = "; ".join(f"{nome}: {contagem(lista, 'tab_zero')[0]:.0f} de {contagem(lista, 'tab_zero')[1]}" for nome, lista in resultados.items() if nome.startswith("BM25"))
    t += [f"Perguntas sem nenhuma palavra em comum com os registros de esquema (BM25, todas as tabelas empatadas em zero): {zeros}.", ""]

    ks_t = KS_TRECHOS
    t += [f"## 2. Trechos dos normativos (glossário, {n_glo} perguntas)", ""]
    t += tabela_md(["Método"] + [f"Hit@{k}" for k in ks_t] + ["MRR"], linhas_medias(resultados, [f"glo_hit@{k}" for k in ks_t] + ["glo_rr"]))
    t += tabela_md(["Método"] + [f"Recall@{k}" for k in ks_t], linhas_medias(resultados, [f"glo_recall@{k}" for k in ks_t]))
    t += tabela_md(["Método"] + [f"nDCG@{k}" for k in (3, 5, 10)] + [f"Precision@{k}" for k in (1, 3, 5)], linhas_medias(resultados, [f"glo_ndcg@{k}" for k in (3, 5, 10)] + [f"glo_prec@{k}" for k in (1, 3, 5)]))
    zeros = "; ".join(f"{nome}: {contagem(lista, 'glo_zero')[0]:.0f} de {contagem(lista, 'glo_zero')[1]}" for nome, lista in resultados.items() if nome.startswith("BM25"))
    t += [f"Perguntas sem nenhuma palavra em comum com os trechos (BM25): {zeros}.", ""]

    t += [f"## 3. Perguntas híbridas ({n_hib} perguntas): contagem de acertos", ""]
    chaves_hib = [f"hib_{n}@{k}" for k in KS_CONJUNTA for n in ("tab", "glo", "conj", "conjsem")]
    cab_hib = ["Método"] + [f"{rot}@{k}" for k in KS_CONJUNTA for rot in ("Tabelas completas", "Trecho", "Ambos", "Ambos (sem companhias)")]
    t += tabela_md(cab_hib, linhas_contagens(resultados, chaves_hib))
    t += ["No piso aleatório, o valor é o número esperado de perguntas (produto das probabilidades dos dois índices, sorteados de forma independente).", ""]

    t += ["## 4. Comparação pareada por pergunta", ""]
    t += ["Diferença média (primeiro método menos o segundo), intervalo de 95% por bootstrap sobre as perguntas e vitórias, derrotas e empates do primeiro método.", ""]
    t += comparacoes

    t += [
        "## 5. Ressalvas",
        "",
        f"- Amostras pequenas ({n_tab} perguntas de tabelas, {n_glo} de trechos, {n_hib} híbridas): uma pergunta vale de 5 a 10 pontos percentuais, e diferenças de 1 a 3 perguntas são ruído. Os intervalos de bootstrap refletem só a variação entre perguntas deste conjunto.",
        "- O BM25 roda sem ajuste nos dados (parâmetros tirados da referência, ver acima). A regra de pontuação por tabela foi escolhida olhando o conjunto de teste (`perguntas_dfp.json`), e a escolha do E5-base entre quatro modelos usou as perguntas de teste e de validação. Logo, nenhum dos dois conjuntos é totalmente independente para o E5-base, e o teste o favorece mais; o BM25 não foi ajustado em nenhum deles.",
        "- Cada pergunta tem uma só evidência literal, e um trecho só conta como relevante se a contém. Trechos úteis sem essa frase contam como erro, e a Precision@k fica limitada (com um relevante e k = 5, o máximo é 0,2).",
        "- As perguntas foram escritas pela autora; as do teste, com a evidência à vista, o que tende a favorecer a busca por palavras.",
        "- A tabela companhias está entre as esperadas em quase todas as perguntas e entra na lista pela regra de chave estrangeira em todos os métodos; a seção 1.2 mostra o acerto sem ela.",
        "- Nenhuma referência do trabalho define um valor mínimo aceitável de Recall. O piso aleatório e o BM25 são referências relativas.",
        "- Empates de pontuação no BM25 ficam na ordem do índice (alfabética por tabela no esquema).",
        "- O RRF usa k = 60 como no artigo original, sem ajuste. A escolha do BM25 sem stemmer para a fusão veio de resultados anteriores (o stemmer piorou a recuperação dos trechos), então é uma escolha informada pelos próprios dados.",
        "- No esquema, a fusão é uma extensão exploratória do que o MimirRAG faz, pois ele aplica a fusão só a trechos de documentos.",
        "",
    ]
    return "\n".join(t)


# Grava as métricas por pergunta e por método em CSV.
def gravar_csv(caminho: str, conjunto: str, resultados: dict):
    chaves = sorted({c for lista in resultados.values() for r in lista for c in r["m"]})
    with open(caminho, "w", encoding="utf-8", newline="") as f:
        escritor = csv.writer(f)
        escritor.writerow(["conjunto", "id", "tipo", "metodo"] + chaves + ["posicoes_tabelas", "posicoes_trechos"])
        for nome, lista in resultados.items():
            for r in lista:
                escritor.writerow(
                    [conjunto, r["id"], r["tipo"], nome]
                    + [f"{r['m'][c]:.6f}" if c in r["m"] else "" for c in chaves]
                    + [json.dumps(r["posicoes"].get("tabelas", ""), ensure_ascii=False), json.dumps(r["posicoes"].get("trechos", ""))]
                )


# Roda os três métodos e o piso nas perguntas, imprime o relatório e grava os arquivos.
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    caminho_perguntas = sys.argv[1] if len(sys.argv) > 1 else CAMINHO_PERGUNTAS
    with open(caminho_perguntas, encoding="utf-8") as f:
        perguntas = json.load(f)

    metodos = criar_metodos()
    e5 = metodos[0]
    referenciadas = carregar_tabelas_referenciadas()
    n_tabelas = len({r["fonte"] for r in e5.esquema.registros})
    n_trechos = len(e5.glossario.registros)

    # Os trechos relevantes de cada pergunta não dependem do método.
    relevantes = {}
    for p in perguntas:
        if p["evidencia"]:
            relevantes[p["id"]] = {
                r["id"] for r in e5.glossario.registros
                if any(normalizar(e) in normalizar(r["conteudo"]) for e in p["evidencia"])
            }
            if not relevantes[p["id"]]:
                raise ValueError(f"{p['id']}: nenhum trecho contém a evidência")

    resultados = {}
    for metodo in metodos:
        print(f"Avaliando {metodo.nome}...", file=sys.stderr)
        resultados[metodo.nome] = [
            avaliar_pergunta(metodo, p, referenciadas, n_tabelas, n_trechos, relevantes.get(p["id"], set()))
            for p in perguntas
        ]
    resultados[NOME_ROTEADO] = montar_roteado(perguntas, resultados, politica(carregar_configuracao(sobrescritas=SOBRESCRITAS_POR_INDICE)))
    resultados[NOME_ROTEADO_RRF] = montar_roteado(perguntas, resultados, politica(carregar_configuracao(sobrescritas=SOBRESCRITAS_POR_TIPO_RRF)))
    resultados[NOME_PISO] = [
        avaliar_piso(p, n_tabelas, n_trechos, len(relevantes.get(p["id"], ())))
        for p in perguntas
    ]

    rng = np.random.default_rng(SEMENTE)
    pares = [
        (NOME_E5, NOME_BM25), (NOME_RRF, NOME_E5), (NOME_RRF, NOME_BM25),
        (NOME_BM25_STEMMER, NOME_BM25), (NOME_BM25_BIBLIOTECA, NOME_BM25),
        (NOME_ROTEADO, NOME_E5), (NOME_ROTEADO, NOME_ROTEADO_RRF),
    ]
    linhas = []
    for primeiro, segundo in pares:
        for chave, descricao in METRICAS_COMPARADAS:
            c = comparar(resultados[primeiro], resultados[segundo], chave, rng)
            if c:
                linhas.append([f"{primeiro} contra {segundo}", descricao, str(c["n"]), fmt(c["dif"], 2),
                               f"[{fmt(c['ic'][0])}; {fmt(c['ic'][1])}]", f"{c['vitorias']}/{c['derrotas']}/{c['empates']}"])
    comparacoes = tabela_md(["Comparação", "Métrica", "n", "Diferença média", "IC 95%", "Vitórias/derrotas/empates"], linhas)

    relatorio = montar_relatorio(caminho_perguntas, perguntas, resultados, metodos, comparacoes)
    print(relatorio)

    os.makedirs(PASTA_SAIDA, exist_ok=True)
    base = os.path.join(PASTA_SAIDA, "baseline_recuperacao_" + os.path.splitext(os.path.basename(caminho_perguntas))[0])
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(relatorio)
    gravar_csv(base + ".csv", os.path.basename(caminho_perguntas), resultados)
    print(f"Arquivos gravados: {base}.md e {base}.csv", file=sys.stderr)


if __name__ == "__main__":
    main()
