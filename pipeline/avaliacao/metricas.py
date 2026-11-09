"""
Etapa: Avaliação (funções de métrica e de comparação).

Função: reúne as funções de métrica usadas pelo executor dos experimentos, para a avaliação ter um
só lugar com as fórmulas.

Descrição geral do funcionamento:
- Recuperação de tabelas (quantitativas e híbridas): Recall@k, cobertura total@k (todas as tabelas
  esperadas na lista) e MRR, com e sem a tabela companhias entre as esperadas.
- Recuperação de trechos dos normativos (conceituais e híbridas): Hit@k, Recall@k, Precision@k, nDCG@k
  e MRR. O nDCG usa a fórmula do monografia.tex (DCG com o desconto 1/log2(i) a partir da posição 2).
- Híbridas: contagem das perguntas com as tabelas completas e o trecho relevante ao mesmo tempo no top-k.
- Piso aleatório: valor esperado de uma lista sorteada ao acaso, por fórmula exata (sem sorteio).
- Execução das consultas (EX): compara o resultado do SQL gerado com o da referência. Como em Zhong et
  al. (2020), resultados que só diferem pela ordem das colunas são equivalentes; a ordem das linhas é
  ignorada e os números são arredondados em 2 casas. Há também a versão que aceita a coluna DENOM_CIA e
  as linhas de outras companhias que combinam com o nome do LIKE.
- Formatação: números com vírgula decimal e tabelas em Markdown, para os relatórios.

Dependências: só a biblioteca padrão do Python. Não lê arquivos nem chama o LLM.

Parâmetros: nenhum. É um módulo importado por executar_experimentos.py.

Resultado: nenhum arquivo; só as funções e as constantes.

Executar: não é executado diretamente.
"""

import itertools
import math
import re
from collections import Counter
from math import comb

KS_TABELAS = (1, 3, 5)
KS_TRECHOS = (1, 3, 5, 10, 20)
KS_CONJUNTA = (3, 5)
TABELA_COMUM = "companhias"
NOME_PISO = "Piso aleatório"


# ---------------------------------------------------------------------------
# 1) Métricas de recuperação de uma pergunta
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


# Acrescenta as métricas das híbridas: tabelas completas, trecho relevante e os dois juntos.
def juntar_hibrida(m: dict):
    for k in KS_CONJUNTA:
        m[f"hib_tab@{k}"] = m[f"tab_todos@{k}"]
        m[f"hib_glo@{k}"] = m[f"glo_hit@{k}"]
        m[f"hib_conj@{k}"] = m[f"tab_todos@{k}"] * m[f"glo_hit@{k}"]
        if f"tabsem_todos@{k}" in m:
            m[f"hib_conjsem@{k}"] = m[f"tabsem_todos@{k}"] * m[f"glo_hit@{k}"]


# Troca qualquer sequência de espaços e quebras de linha por um espaço (para achar a evidência no trecho).
def normalizar_texto(texto: str) -> str:
    return re.sub(r"\s+", " ", texto)


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
# 3) Execução das consultas (EX)
# ---------------------------------------------------------------------------

# Arredonda os números e ordena as linhas, para comparar resultados sem depender da ordem.
def normalizar_linhas(linhas: list[tuple]) -> list[tuple]:
    def valor(v):
        return round(v, 2) if isinstance(v, float) else v
    return sorted((tuple(valor(v) for v in linha) for linha in linhas), key=repr)


# Diz se existe alguma ordem das colunas do resultado gerado que o deixe idêntico ao da referência.
def resultados_equivalentes(gerado: list[tuple], referencia: list[tuple]) -> bool:
    if len(gerado) != len(referencia):
        return False
    if not gerado:
        return True
    if len(gerado[0]) != len(referencia[0]):
        return False
    alvo = normalizar_linhas(referencia)
    for ordem in itertools.permutations(range(len(gerado[0]))):
        if normalizar_linhas([tuple(linha[i] for i in ordem) for linha in gerado]) == alvo:
            return True
    return False


# Diz se o resultado gerado traz todas as linhas da referência e, além delas, só outras companhias
# que também combinam com o nome do LIKE. A coluna DENOM_CIA é a única coluna extra aceita.
def resultado_contem_referencia(colunas: list[str], gerado: list[tuple], referencia: list[tuple]) -> bool:
    if "DENOM_CIA" not in colunas or not gerado or not referencia:
        return False
    posicao = colunas.index("DENOM_CIA")
    sem_nome = [tuple(v for i, v in enumerate(linha) if i != posicao) for linha in gerado]
    for candidato in (gerado, sem_nome):
        if len(candidato[0]) != len(referencia[0]):
            continue
        alvo = Counter(repr(linha) for linha in normalizar_linhas(referencia))
        for ordem in itertools.permutations(range(len(candidato[0]))):
            obtido = Counter(repr(linha) for linha in normalizar_linhas([tuple(l[i] for i in ordem) for l in candidato]))
            if not alvo - obtido:
                return True
    return False


# ---------------------------------------------------------------------------
# 4) Formatação dos relatórios
# ---------------------------------------------------------------------------

# Escreve um número com vírgula decimal, ou "n/d" se não houver.
def fmt(valor, casas: int = 2) -> str:
    return "n/d" if valor is None else f"{valor:.{casas}f}".replace(".", ",")


# Monta uma tabela em Markdown.
def tabela_md(cabecalho: list[str], linhas: list[list[str]]) -> list[str]:
    saida = ["| " + " | ".join(cabecalho) + " |", "|" + "|".join("---" for _ in cabecalho) + "|"]
    saida += ["| " + " | ".join(linha) + " |" for linha in linhas]
    return saida + [""]
