"""
Etapa: Avaliação (funções de métrica).

Função: reúne as funções de métrica usadas pelo executor dos experimentos, para a avaliação ter um
só lugar com as fórmulas.

Descrição geral do funcionamento:
- Recuperação (tabelas do esquema e trechos dos normativos): Precision@k, Recall@k, nDCG@k e MRR,
  calculadas pela biblioteca ranx a partir do ranking devolvido pela busca e do conjunto de itens
  relevantes de cada pergunta (relevância binária). A ranx calcula o nDCG com o desconto 1/log2(i+1) em
  todas as posições (a forma do TREC Eval), e não com o de Järvelin usado na equação da fundamentação
  (rel1 + soma de rel_i/log2 i, a partir de i = 2); a divergência deve ser declarada.
  Nas tabelas, a tabela companhias fica fora do ranking e dos itens relevantes: o sistema a acrescenta
  por regra a todas as listas (chave estrangeira), então ela não mede a qualidade da busca.
- Execução das consultas (EX): compara o resultado do SQL gerado com o da referência. Como em Zhong et
  al. (2020), resultados que só diferem pela ordem das colunas são equivalentes; a ordem das linhas é
  ignorada e os números são arredondados em 2 casas. Há também o EX adaptado (sem artigo, declarado
  como adaptação): o resultado gerado deve conter todas as linhas da referência; a coluna DENOM_CIA é
  tolerada; uma linha a mais só vale se o DENOM_CIA dela tiver uma palavra que começa com um termo
  que identifica a companhia dita na pergunta (como o LIKE do gerador de SQL). O termo é a palavra mais
  rara, entre as da pergunta que são o início de uma palavra do nome da companhia de referência,
  ignorando as palavras comuns nos nomes das companhias (as que aparecem em mais de 1% dos nomes). A
  frequência vem dos nomes do banco.
- Conferência dos números (sem LLM): quantos valores numéricos do resultado do SQL aparecem no texto
  da resposta, comparando só os dígitos. Serve às perguntas com dados; não é métrica de artigo.
- Formatação: números com vírgula decimal e tabelas em Markdown, para os relatórios.

Dependências: só a biblioteca padrão do Python. Não lê arquivos nem chama o LLM.

Parâmetros: nenhum. É um módulo importado por executar_experimentos.py.

Resultado: nenhum arquivo; só as funções e as constantes.

Executar: não é executado diretamente.
"""

import itertools
import re
import unicodedata
import warnings
from collections import Counter

from ranx import Qrels, Run, evaluate

# A ranx (numba) emite um aviso de conversão de tipos que não afeta o resultado.
warnings.filterwarnings("ignore", module="numba")
warnings.filterwarnings("ignore", module="ranx")

KS_TABELAS = (1, 3, 5)
KS_TRECHOS = (1, 3, 5, 10, 20)
TABELA_COMUM = "companhias"
# Palavra que aparece em mais que esta fração dos nomes de companhias é comum demais para identificar uma.
FRACAO_PALAVRA_COMUM = 0.01


# ---------------------------------------------------------------------------
# 1) Métricas de recuperação de uma pergunta (biblioteca ranx)
# ---------------------------------------------------------------------------

# Calcula, com a ranx, Precision@k, Recall@k e nDCG@k para cada k e o inverso da posição do primeiro
# item relevante (o MRR é a média desse valor sobre as perguntas). O ranking é a lista de ids na ordem
# da busca; relevantes é o conjunto de ids relevantes da pergunta. As chaves levam o prefixo do índice
# ("tab" para tabelas e "glo" para trechos).
def metricas_recuperacao(prefixo: str, ranking: list[str], relevantes, ks: tuple) -> dict:
    qrels = Qrels({"pergunta": {item: 1 for item in relevantes}})
    # Notas decrescentes e distintas, para a ranx respeitar a ordem da busca sem empates.
    run = Run({"pergunta": {item: float(len(ranking) - i) for i, item in enumerate(ranking)}})
    nomes = {}
    for k in ks:
        nomes[f"{prefixo}_prec@{k}"] = f"precision@{k}"
        nomes[f"{prefixo}_recall@{k}"] = f"recall@{k}"
        nomes[f"{prefixo}_ndcg@{k}"] = f"ndcg@{k}"
    nomes[f"{prefixo}_rr"] = "mrr"
    notas = evaluate(qrels, run, list(nomes.values()))
    return {chave: float(notas[metrica]) for chave, metrica in nomes.items()}


# Troca qualquer sequência de espaços e quebras de linha por um espaço (para achar a evidência no trecho).
def normalizar_texto(texto: str) -> str:
    return re.sub(r"\s+", " ", texto)


# ---------------------------------------------------------------------------
# 2) Execução das consultas (EX)
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


# Devolve as palavras de um texto (nome de companhia ou pergunta), em maiúsculas, sem acento e com pelo
# menos 3 letras ou dígitos.
def palavras_do_nome(texto: str) -> set[str]:
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().upper()
    return {p for p in re.findall(r"[A-Z0-9]+", sem_acento) if len(p) >= 3}


# Diz se alguma palavra do texto começa com algum dos termos (o mesmo efeito do LIKE '%termo%' sobre o nome).
def casa_termos(texto: str, termos: set[str]) -> bool:
    return any(palavra.startswith(termo) for palavra in palavras_do_nome(texto) for termo in termos)


# Palavras dos nomes das companhias do banco, usadas para achar o termo que identifica a companhia dita
# na pergunta.
class NomesDeCompanhias:
    def __init__(self, nomes: list[str], fracao_comum: float = FRACAO_PALAVRA_COMUM):
        self.palavras = [palavras_do_nome(nome) for nome in nomes]
        self.limite = fracao_comum * len(nomes)

    # Em quantos nomes há uma palavra que começa com o termo.
    def frequencia(self, termo: str) -> int:
        return sum(1 for palavras in self.palavras if any(p.startswith(termo) for p in palavras))

    # Devolve os termos que identificam as companhias da pergunta: para cada companhia de referência,
    # a palavra mais rara entre as da pergunta que são o início de uma palavra do nome dela, sem as
    # palavras comuns.
    def termos(self, pergunta: str, companhias: list[str]) -> set[str]:
        na_pergunta = palavras_do_nome(pergunta)
        termos = set()
        for nome in companhias:
            do_nome = palavras_do_nome(nome)
            candidatas = {p for p in na_pergunta if any(v.startswith(p) for v in do_nome)}
            frequencias = {p: self.frequencia(p) for p in candidatas}
            candidatas = {p for p, f in frequencias.items() if f <= self.limite}
            if candidatas:
                menor = min(frequencias[p] for p in candidatas)
                termos |= {p for p in candidatas if frequencias[p] == menor}
        return termos


# EX adaptado. Vale se o resultado gerado é equivalente ao da referência (EX estrito) ou se traz todas as
# linhas da referência, com a coluna DENOM_CIA tolerada, e as linhas a mais são de companhias cujo
# DENOM_CIA tem uma palavra que começa com um dos termos que identificam a companhia da pergunta.
def resultado_adaptado(colunas: list[str], gerado: list[tuple], referencia: list[tuple], termos: set[str]) -> bool:
    if resultados_equivalentes(gerado, referencia):
        return True
    if "DENOM_CIA" not in colunas or not gerado or not referencia:
        return False
    posicao = colunas.index("DENOM_CIA")
    alvo = Counter(repr(linha) for linha in normalizar_linhas(referencia))
    for com_nome in (True, False):
        linhas = gerado if com_nome else [tuple(v for i, v in enumerate(linha) if i != posicao) for linha in gerado]
        if len(linhas[0]) != len(referencia[0]):
            continue
        for ordem in itertools.permutations(range(len(linhas[0]))):
            faltam = Counter(alvo)
            extras_validas = True
            for original, linha in zip(gerado, linhas):
                chave = repr(normalizar_linhas([tuple(linha[i] for i in ordem)])[0])
                if faltam[chave] > 0:
                    faltam[chave] -= 1
                elif not casa_termos(str(original[posicao]), termos):
                    extras_validas = False
                    break
            if extras_validas and not +faltam:
                return True
    return False


# ---------------------------------------------------------------------------
# 3) Conferência dos números da resposta
# ---------------------------------------------------------------------------

MAX_LINHAS_CONFERIDAS = 30


def so_digitos(texto: str) -> str:
    return re.sub(r"[^0-9]", "", texto)


# Conta os valores numéricos do resultado que aparecem na resposta. Compara só os dígitos, para
# "88.242.467.000,00" casar com 88242467000.0. Valores com parte decimal usam as 2 casas.
def numeros_presentes(resposta: str, linhas: list) -> tuple[int, int]:
    alvo = so_digitos(resposta)
    numeros = [v for linha in linhas[:MAX_LINHAS_CONFERIDAS] for v in linha if isinstance(v, (int, float))]
    achados = 0
    for v in numeros:
        texto = f"{v:.0f}" if float(v) == round(v) else f"{v:.2f}"
        achados += so_digitos(texto) in alvo
    return achados, len(numeros)


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
