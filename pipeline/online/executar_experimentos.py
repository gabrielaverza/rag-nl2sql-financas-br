"""
Etapa: Online (execução dos experimentos).

Função: executor único dos experimentos de avaliação, com dois níveis: "recuperacao" (mede se a busca
traz as tabelas e os trechos certos, sem LLM, exceto o divisor das perguntas híbridas) e "pipeline"
(roda o fluxo completo de main.py para cada pergunta, com LLM, e mede as demais métricas).

Descrição geral do funcionamento:
- Um experimento é um grupo de configurações comparadas entre si e está descrito em experimentos.json
  (na raiz do repositório): nível, configurações nomeadas, pares a comparar e, no nível pipeline,
  rodadas e a passada de tabelas ideais. Cada configuração é uma lista de trocas "chave=valor" sobre o
  configuracao.json, no formato do --set (configuracao.py), então qualquer chave pode variar: modelo de
  embeddings, método de busca por índice e por tipo, k, divisor, bloqueio de pronomes, radicalizador.
- O conjunto de perguntas é sempre dados/perguntas_validacao.json (60 perguntas, o conjunto oficial).
- Nível recuperação: para cada configuração e pergunta, busca as tabelas (quantitativas e híbridas) e
  os trechos dos normativos (conceituais e híbridas) com o método da configuração e calcula as métricas
  de avaliar_baseline.py (todas as tabelas no top-k, Recall, MRR, Hit@k, Precision@k, nDCG@k e o
  acerto conjunto das híbridas). Com o divisor ligado, as híbridas buscam as tabelas com a pergunta
  inteira e os trechos com a parte conceitual (condição "mista"). As divisões são feitas uma vez e
  guardadas em metricas/divisoes_hibridas.json, para todas as configurações usarem as mesmas.
- Nível pipeline: para cada rodada, configuração e pergunta, chama executar() de main.py e mede a
  classe prevista, o EX (estrito e aceitando outras companhias que combinam com o nome), a
  fidelidade, a relevância do contexto, a aprovação do validador e o tempo de cada passo. O SQL gerado
  fica em um cache por (pergunta, tabelas do contexto) que vale entre as configurações da mesma
  rodada e é refeito a cada rodada, para a variação do LLM aparecer nas rodadas. Com
  "tabelas_ideais": true, uma passada extra gera o SQL só com as tabelas de tabelas_esperadas.
- Comparação pareada: diferença média pergunta a pergunta (sempre o primeiro menos o segundo), vitórias,
  derrotas e empates, e intervalo de 95% por bootstrap (comparar() de avaliar_baseline.py: 10.000
  reamostragens, semente 42). O relatório diz onde o intervalo fica em relação ao zero.

Dependências:
- pipeline/online/avaliar_baseline.py, avaliar_recuperacao.py, avaliar_sql.py, busca*.py, divisor.py e
  main.py, os índices e o esquema em embeddings/ (os de outros modelos, em embeddings/modelos/<chave>/),
  dados/perguntas_validacao.json, dados/dfp.db e experimentos.json;
- servidor Ollama com o modelo qwen2.5-coder:7b (divisor e nível pipeline).

Parâmetros: o nome do experimento, e opcionalmente --rodadas N (nível pipeline), --ids q01,c01,...
(roda só essas perguntas; os arquivos ganham o sufixo _parcial), --refazer-divisoes, --listar e
--experimentos CAMINHO (outro arquivo de experimentos).

Resultado: em metricas/, com nomes fixos que são sobrescritos a cada execução: experimento_<nome>.csv
(uma linha por rodada, configuração, pergunta e métrica), experimento_<nome>_resumo.csv (uma linha por
configuração, tipo e métrica) e experimento_<nome>.md (tabelas, comparações, métricas e escalas,
ressalvas e comandos); no nível pipeline, também experimento_<nome>_respostas.json.

Executar de dentro de rag-nl2sql-financas-br/:
    python pipeline/online/executar_experimentos.py --listar
    python pipeline/online/executar_experimentos.py metodos
    python pipeline/online/executar_experimentos.py teste_pipeline --ids q01,c01,h01
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from statistics import mean

import numpy as np

from avaliar_baseline import (
    KS_TABELAS,
    NOME_PISO,
    SEMENTE,
    TABELA_COMUM,
    avaliar_piso,
    comparar,
    fmt,
    juntar_hibrida,
    metricas_tabelas,
    metricas_trechos,
    tabela_md,
)
from avaliar_recuperacao import normalizar
from avaliar_sql import resultado_contem_referencia, resultados_equivalentes
from busca import (
    SECOES_FORA_DA_PONTUACAO,
    Indice,
    buscar_tabelas,
    caminho_indice,
    carregar_tabelas_referenciadas,
    vetorizar,
)
from busca_bm25 import CAMINHO_BM25_ESQUEMA, CAMINHO_BM25_GLOSSARIO, IndiceBM25
from busca_configuravel import definir_configuracao
from configuracao import carregar, metodo_de_busca
from fusao_rrf import IndiceHibrido
from modelos_embedding import MODELO_PADRAO

CAMINHO_EXPERIMENTOS = "experimentos.json"
CAMINHO_PERGUNTAS = "dados/perguntas_validacao.json"
PASTA_SAIDA = "metricas"
CAMINHO_DIVISOES = os.path.join(PASTA_SAIDA, "divisoes_hibridas.json")
TIPOS = ("quantitativa", "conceitual", "hibrida")
PLURAL = {"quantitativa": "quantitativas", "conceitual": "conceituais", "hibrida": "híbridas"}

# Métricas mostradas por tipo no relatório: (chave, rótulo, "contagem" ou "media").
METRICAS_POR_TIPO = {
    "quantitativa": [
        ("tab_todos@3", "Todas as tabelas no top-3", "contagem"),
        ("tab_todos@5", "Todas as tabelas no top-5", "contagem"),
        ("tab_rr", "MRR das tabelas", "media"),
    ],
    "conceitual": [
        ("glo_hit@3", "Hit@3 dos trechos", "contagem"),
        ("glo_hit@5", "Hit@5 dos trechos", "contagem"),
        ("glo_hit@10", "Hit@10 dos trechos", "contagem"),
        ("glo_rr", "MRR dos trechos", "media"),
        ("glo_ndcg@10", "nDCG@10 dos trechos", "media"),
    ],
    "hibrida": [
        ("tab_todos@3", "Todas as tabelas no top-3", "contagem"),
        ("tab_todos@5", "Todas as tabelas no top-5", "contagem"),
        ("glo_hit@3", "Hit@3 dos trechos", "contagem"),
        ("glo_hit@10", "Hit@10 dos trechos", "contagem"),
        ("glo_ndcg@10", "nDCG@10 dos trechos", "media"),
        ("hib_conj@3", "Tabelas e trecho no top-3", "contagem"),
        ("hib_conj@5", "Tabelas e trecho no top-5", "contagem"),
    ],
}
METRICAS_PIPELINE = [
    ("classe_certa", "Classe certa", "contagem"),
    ("ex_estrito", "EX estrito", "contagem"),
    ("ex_companhias", "EX aceitando outras companhias", "contagem"),
    ("sql_ok", "SQL executou com linhas", "contagem"),
    ("ideal_ex_estrito", "EX estrito com tabelas ideais", "contagem"),
    ("ideal_ex_companhias", "EX aceitando outras companhias, tabelas ideais", "contagem"),
    ("fidelidade", "Fidelidade média", "media"),
    ("relevancia", "Relevância média do contexto", "media"),
    ("aprovado", "Respostas aprovadas", "contagem"),
    ("tempo_total", "Tempo total por pergunta (s)", "media"),
]
PASSOS = ("classificador", "divisor", "sql", "sintese", "validacao")


# ---------------------------------------------------------------------------
# 1) Descrição das métricas (escala e sentido)
# ---------------------------------------------------------------------------

# Devolve (escala, sentido) de uma métrica a partir do nome da chave.
def escala_e_sentido(chave: str) -> tuple[str, str]:
    if chave.startswith("tempo_"):
        return "segundos", "menor é melhor"
    if chave.startswith("tab_tam@"):
        return "número de tabelas na lista", "neutro"
    if chave in ("tab_zero", "glo_zero"):
        return "0 ou 1", "menor é melhor"
    binarias = ("tab_todos@", "tabsem_todos@", "glo_hit@", "hib_", "classe_certa", "ex_", "ideal_ex_", "sql_ok", "aprovado")
    if chave.startswith(binarias):
        return "0 ou 1", "maior é melhor"
    return "0 a 1", "maior é melhor"


DESCRICAO_METRICAS = [
    ("Todas as tabelas no top-k (`tab_todos@k`)", "0 ou 1 por pergunta", "1 se todas as tabelas necessárias estão entre as k mais parecidas (mais as que elas referenciam por chave estrangeira). É a que limita o SQL. Maior é melhor."),
    ("Recall das tabelas (`tab_recall@k`)", "0 a 1", "Fração das tabelas necessárias que aparece na lista. Maior é melhor."),
    ("MRR das tabelas (`tab_rr`)", "0 a 1", "1 dividido pela posição da primeira tabela necessária no ranking de tabelas. Maior é melhor."),
    ("Variantes `tabsem_*`", "igual às de tabelas", "As mesmas medidas sem contar a tabela `companhias`, que entra sempre pela chave estrangeira; mede só o acerto da demonstração."),
    ("Hit@k dos trechos (`glo_hit@k`)", "0 ou 1 por pergunta", "1 se há ao menos um trecho relevante entre os k primeiros. É a base do \"X de 20\". Maior é melhor."),
    ("Recall e Precision dos trechos (`glo_recall@k`, `glo_prec@k`)", "0 a 1", "Fração dos trechos relevantes achados, e fração dos k trazidos que são relevantes."),
    ("MRR dos trechos (`glo_rr`)", "0 a 1", "1 dividido pela posição do primeiro trecho relevante. Maior é melhor."),
    ("nDCG@k dos trechos (`glo_ndcg@k`)", "0 a 1", "Nota a ordem da lista dando mais peso a um acerto no topo (desconto 1/log2 da posição a partir da segunda). Maior é melhor; a principal na comparação de modelos."),
    ("Acerto conjunto das híbridas (`hib_conj@k`)", "0 ou 1 por pergunta", "1 se as tabelas completas e um trecho relevante aparecem ao mesmo tempo no top-k."),
    ("`tab_zero`, `glo_zero`", "0 ou 1", "Só no BM25: 1 se o primeiro resultado tem nota zero, isto é, nenhuma palavra da pergunta aparece no índice. Menor é melhor."),
    ("`tab_tam@k`", "número", "Tamanho da lista de tabelas (as k escolhidas mais as referenciadas)."),
    ("Piso aleatório", "mesma escala", "Valor esperado de uma lista sorteada ao acaso, por fórmula exata. Mostra se o método está acima do acaso."),
    ("Classe certa (`classe_certa`)", "0 ou 1", "O classificador acertou o tipo da pergunta."),
    ("EX estrito (`ex_estrito`)", "0 ou 1", "O resultado do SQL gerado é igual ao do `sql_referencia` (ordem das colunas e das linhas ignorada, números com 2 casas). Só nas perguntas com SQL."),
    ("EX aceitando outras companhias (`ex_companhias`)", "0 ou 1", "Aceita também a coluna `DENOM_CIA` e linhas de outras companhias cujo nome combina com o `LIKE` (extensão do EX de Zhong et al., 2020)."),
    ("`sql_ok`", "0 ou 1", "O SQL executou e devolveu linhas."),
    ("Tabelas ideais (`ideal_ex_*`)", "0 ou 1", "EX do SQL gerado quando o gerador recebe as tabelas certas (`tabelas_esperadas`) em vez das da busca. É um teto, e a diferença para o EX normal mostra quanto da perda vem da busca."),
    ("Fidelidade e relevância (`fidelidade`, `relevancia`)", "0 a 1", "Do validador: fração das afirmações da resposta sustentadas pelo contexto, e fração dos trechos que contribuem para a pergunta. Medem sustentação pelo contexto, não a correção do dado."),
    ("Aprovado (`aprovado`)", "0 ou 1", "A fidelidade chegou ao limiar do validador (0,8)."),
    ("Tempos (`tempo_*`)", "segundos", "Tempo da pergunta inteira e de cada passo do fluxo. Menor é melhor."),
]


# ---------------------------------------------------------------------------
# 2) Perguntas, experimentos e divisões
# ---------------------------------------------------------------------------

# Lê o conjunto oficial e, se pedido, só as perguntas de uma lista de ids.
def carregar_perguntas(ids: list[str] | None) -> list[dict]:
    with open(CAMINHO_PERGUNTAS, encoding="utf-8") as f:
        perguntas = json.load(f)
    if ids:
        desconhecidos = [i for i in ids if i not in {p["id"] for p in perguntas}]
        if desconhecidos:
            raise ValueError(f"ids desconhecidos: {desconhecidos}")
        perguntas = [p for p in perguntas if p["id"] in ids]
    return perguntas


# Lê um experimento do arquivo de experimentos e monta a configuração efetiva de cada configuração.
def carregar_experimento(caminho: str, nome: str) -> dict:
    with open(caminho, encoding="utf-8") as f:
        todos = json.load(f)["experimentos"]
    if nome not in todos:
        raise ValueError(f"experimento {nome!r} não existe em {caminho}; disponíveis: {', '.join(todos)}")
    exp = dict(todos[nome])
    exp["nome"] = nome
    if exp.get("nivel") not in ("recuperacao", "pipeline"):
        raise ValueError(f"{nome}: nivel deve ser 'recuperacao' ou 'pipeline'")
    nomes = [c["nome"] for c in exp["configuracoes"]]
    if len(set(nomes)) != len(nomes):
        raise ValueError(f"{nome}: nomes de configuração repetidos")
    exp["configs"] = {c["nome"]: carregar(sobrescritas=c.get("sobrescritas", [])) for c in exp["configuracoes"]}
    for a, b in exp.get("comparacoes", []):
        if a not in exp["configs"] or b not in exp["configs"]:
            raise ValueError(f"{nome}: comparação com configuração inexistente: {a!r}, {b!r}")
    return exp


# Guarda e reaproveita as divisões das perguntas híbridas, para todas as configurações usarem as mesmas.
class Divisoes:
    def __init__(self, refazer: bool):
        self.refazer = refazer
        self.dados = {}
        self.alterado = False
        if os.path.exists(CAMINHO_DIVISOES):
            with open(CAMINHO_DIVISOES, encoding="utf-8") as f:
                self.dados = json.load(f)

    # Devolve a divisão da pergunta (a guardada, se a pergunta e o bloqueio de pronomes forem os mesmos).
    def obter(self, p: dict, bloquear_pronomes: bool) -> dict:
        chave = f"{p['id']}|bloquear_pronomes={bloquear_pronomes}"
        guardada = self.dados.get(chave)
        if guardada and guardada["pergunta"] == p["pergunta"] and not self.refazer:
            return guardada
        from divisor import dividir

        inicio = time.perf_counter()
        divisao = dividir(p["pergunta"], bloquear_pronomes)
        self.dados[chave] = {"pergunta": p["pergunta"], **divisao, "segundos": round(time.perf_counter() - inicio, 2)}
        self.alterado = True
        return self.dados[chave]

    def salvar(self):
        if self.alterado:
            os.makedirs(PASTA_SAIDA, exist_ok=True)
            with open(CAMINHO_DIVISOES, "w", encoding="utf-8") as f:
                json.dump(self.dados, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 3) Nível recuperação
# ---------------------------------------------------------------------------

class Recuperador:
    # Prepara o que não depende da configuração: relações de chave estrangeira, tamanhos e trechos relevantes.
    def __init__(self, perguntas: list[dict]):
        self.referenciadas = carregar_tabelas_referenciadas()
        self._indices = {}
        padrao = self.indice("glossario", "embeddings", MODELO_PADRAO, False)
        self.n_trechos = len(padrao.registros)
        self.n_tabelas = len({r["fonte"] for r in self.indice("esquema", "embeddings", MODELO_PADRAO, False).registros})
        # Os trechos relevantes de cada pergunta não dependem do método nem do modelo (o texto dos
        # registros é o mesmo em todos os índices).
        self.relevantes = {}
        for p in perguntas:
            if p["evidencia"]:
                ids = {r["id"] for r in padrao.registros if any(normalizar(e) in normalizar(r["conteudo"]) for e in p["evidencia"])}
                if not ids:
                    raise ValueError(f"{p['id']}: nenhum trecho contém a evidência")
                self.relevantes[p["id"]] = ids

    # Monta (uma vez) o índice "esquema" ou "glossario" do método, do modelo e do radicalizador pedidos.
    def indice(self, onde: str, metodo: str, modelo: str | None, radicalizar: bool):
        chave = (onde, metodo, modelo if metodo != "bm25" else None, radicalizar if metodo != "embeddings" else None)
        if chave not in self._indices:
            esquema = onde == "esquema"
            if metodo == "embeddings":
                caminho = caminho_indice(onde, modelo)
                if not os.path.exists(caminho):
                    raise FileNotFoundError(
                        f"Índice {caminho} não encontrado. Gere-o com: python pipeline/offline/rodar_offline.py --modelos {modelo}"
                    )
                indice = Indice(caminho)
            elif metodo == "bm25":
                caminho_bm25 = CAMINHO_BM25_ESQUEMA if esquema else CAMINHO_BM25_GLOSSARIO
                indice = IndiceBM25(caminho_bm25, radicalizar, SECOES_FORA_DA_PONTUACAO if esquema else ())
            else:
                indice = IndiceHibrido(self.indice(onde, "embeddings", modelo, radicalizar), self.indice(onde, "bm25", modelo, radicalizar))
            self._indices[chave] = indice
        return self._indices[chave]

    # Prepara a pergunta para o método: vetor, texto ou o par dos dois (fusão).
    @staticmethod
    def consulta(metodo: str, texto: str, modelo: str):
        if metodo == "embeddings":
            return vetorizar(texto, modelo)
        if metodo == "bm25":
            return texto
        return (vetorizar(texto, modelo), texto)

    # Avalia uma pergunta com a configuração: tabelas pelo método do índice de esquema e trechos pelo do glossário.
    def avaliar(self, config: dict, p: dict, divisao: dict | None) -> dict:
        tipo, modelo, radicalizar = p["tipo"], config["modelo_embedding"], config["bm25"]["radicalizar"]
        m, posicoes = {}, {}
        if p["tabelas_esperadas"]:
            metodo = metodo_de_busca(config, "esquema", tipo)
            indice = self.indice("esquema", metodo, modelo, radicalizar)
            consulta = self.consulta(metodo, p["pergunta"], modelo)
            completo = indice.buscar_por_fonte(consulta, self.n_tabelas)
            ranking = [r["fonte"] for r in completo]
            listas = {k: buscar_tabelas(indice, consulta, k, self.referenciadas) for k in KS_TABELAS}
            esperadas = p["tabelas_esperadas"]
            sem_comum = [t for t in esperadas if t != TABELA_COMUM]
            m.update(metricas_tabelas("tab", ranking, listas, esperadas))
            if sem_comum:
                m.update(metricas_tabelas("tabsem", ranking, listas, sem_comum))
            for k in KS_TABELAS:
                m[f"tab_tam@{k}"] = float(len(listas[k]))
            if metodo == "bm25":
                m["tab_zero"] = float(completo[0]["pontuacao"] <= 0)
            posicoes["tabelas"] = {t: ranking.index(t) + 1 for t in esperadas}
        if p["evidencia"]:
            metodo = metodo_de_busca(config, "glossario", tipo)
            texto = p["pergunta"]
            if divisao and divisao["dividida"]:
                texto = divisao["parte_conceitual"]
            indice = self.indice("glossario", metodo, modelo, radicalizar)
            consulta = self.consulta(metodo, texto, modelo)
            ranking_trechos = indice.buscar(consulta, self.n_trechos)
            relevancias = [r["id"] in self.relevantes[p["id"]] for r in ranking_trechos]
            m.update(metricas_trechos(relevancias))
            if metodo == "bm25":
                m["glo_zero"] = float(ranking_trechos[0]["pontuacao"] <= 0)
            posicoes["trechos"] = [i for i, r in enumerate(relevancias, start=1) if r][:5]
        if tipo == "hibrida" and p["tabelas_esperadas"] and p["evidencia"]:
            juntar_hibrida(m)
        return {"id": p["id"], "tipo": tipo, "m": m, "posicoes": posicoes}


# Roda o nível recuperação e devolve (resultados por configuração, tempos, divisões usadas).
def rodar_recuperacao(exp: dict, perguntas: list[dict], refazer_divisoes: bool) -> dict:
    recuperador = Recuperador(perguntas)
    divisoes = Divisoes(refazer_divisoes)
    resultados, tempos, usadas = {}, {}, {}
    for nome, config in exp["configs"].items():
        print(f"Avaliando {nome}...", file=sys.stderr)
        inicio = time.perf_counter()
        lista = []
        for p in perguntas:
            divisao = None
            if p["tipo"] == "hibrida" and config["divisor"]["ativo"]:
                divisao = divisoes.obter(p, config["divisor"]["bloquear_pronomes"])
                usadas[p["id"]] = divisao
            lista.append(recuperador.avaliar(config, p, divisao))
        resultados[nome] = lista
        tempos[nome] = time.perf_counter() - inicio
    divisoes.salvar()
    resultados[NOME_PISO] = [
        avaliar_piso(p, recuperador.n_tabelas, recuperador.n_trechos, len(recuperador.relevantes.get(p["id"], ())))
        for p in perguntas
    ]
    return {"resultados": resultados, "tempos": tempos, "divisoes": usadas, "n_tabelas": recuperador.n_tabelas, "n_trechos": recuperador.n_trechos}


# ---------------------------------------------------------------------------
# 4) Nível pipeline
# ---------------------------------------------------------------------------

# Compara o SQL gerado com o de referência e devolve (EX estrito, EX aceitando outras companhias).
def calcular_ex(r_sql: dict | None, referencia_sql: str) -> tuple[bool, bool]:
    from gerador_sql import executar_sql

    if not r_sql or r_sql["status"] != "ok":
        return False, False
    _, referencia = executar_sql(referencia_sql, limite_linhas=None)
    estrito = resultados_equivalentes(r_sql["linhas"], referencia)
    return estrito, estrito or resultado_contem_referencia(r_sql["colunas"], r_sql["linhas"], referencia)


# Roda o nível pipeline: rodadas, configurações e perguntas. Devolve linhas de métricas e as respostas.
def rodar_pipeline(exp: dict, perguntas: list[dict], rodadas: int) -> dict:
    from gerador_sql import gerar_sql
    from main import executar

    resultados = {nome: {r: [] for r in range(1, rodadas + 1)} for nome in exp["configs"]}
    respostas = []
    for rodada in range(1, rodadas + 1):
        cache = {}  # o SQL guardado vale só dentro desta rodada
        for nome, config in exp["configs"].items():
            print(f"Rodada {rodada}, configuração {nome}...", file=sys.stderr)
            definir_configuracao(config)
            for p in perguntas:
                inicio = time.time()
                r = executar(p["pergunta"], cache_sql=cache)
                total = time.time() - inicio
                r_sql = r["resultado_sql"]
                v = r["validacao"]
                m = {
                    "classe_certa": float(r["classe"] == p["tipo"]),
                    "fidelidade": v["fidelidade"]["score"],
                    "aprovado": float(v["aprovado"]),
                    "tempo_total": total,
                }
                if v["relevancia_contexto"]["score"] is not None:
                    m["relevancia"] = v["relevancia_contexto"]["score"]
                for passo in PASSOS:
                    if passo in r["tempos"]:
                        m[f"tempo_{passo}"] = r["tempos"][passo]
                if p["sql_referencia"]:
                    estrito, companhias = calcular_ex(r_sql, p["sql_referencia"])
                    m.update(ex_estrito=float(estrito), ex_companhias=float(companhias), sql_ok=float(bool(r_sql and r_sql["status"] == "ok")))
                    if exp.get("tabelas_ideais"):
                        ideal = gerar_sql(p["pergunta"], tabelas=p["tabelas_esperadas"], cache=cache)
                        e_i, c_i = calcular_ex(ideal, p["sql_referencia"])
                        m.update(ideal_ex_estrito=float(e_i), ideal_ex_companhias=float(c_i))
                resultados[nome][rodada].append({"id": p["id"], "tipo": p["tipo"], "m": m, "posicoes": {}})
                respostas.append({
                    "rodada": rodada, "configuracao": nome, "id": p["id"], "tipo": p["tipo"], "classe": r["classe"],
                    "pergunta": p["pergunta"], "divisao": r["divisao"], "sql": r_sql["sql"] if r_sql else None,
                    "tabelas_contexto": r_sql["tabelas_contexto"] if r_sql else None, "resposta": r["resposta"],
                })
    return {"resultados": resultados, "respostas": respostas}


# ---------------------------------------------------------------------------
# 5) Saídas
# ---------------------------------------------------------------------------

# Rótulos de cada configuração, para as colunas de contexto do CSV longo.
def descrever_config(config: dict, tipo: str) -> dict:
    metodos = (metodo_de_busca(config, "esquema", tipo), metodo_de_busca(config, "glossario", tipo))
    usa_bm25 = any(m in ("bm25", "rrf") for m in metodos)
    return {
        "modelo": config["modelo_embedding"],
        "metodo_tabelas": metodo_de_busca(config, "esquema", tipo),
        "metodo_trechos": metodo_de_busca(config, "glossario", tipo),
        "divisor": "ligado" if config["divisor"]["ativo"] else "desligado",
        "pronomes": "bloqueados" if config["divisor"]["bloquear_pronomes"] else "permitidos",
        # O radicalizador só vale quando algum índice usa o BM25 (sozinho ou na fusão).
        "stemmer": ("sim" if config["bm25"]["radicalizar"] else "nao") if usa_bm25 else "n/a",
    }


# Junta os resultados de todas as rodadas em linhas (rodada, configuração, pergunta, métrica, valor).
def linhas_longas(exp: dict, por_config: dict) -> list[dict]:
    linhas = []
    for nome, rodadas in por_config.items():
        config = exp["configs"].get(nome)
        for rodada, lista in rodadas.items():
            for r in lista:
                contexto = descrever_config(config, r["tipo"]) if config else {k: "" for k in ("modelo", "metodo_tabelas", "metodo_trechos", "divisor", "pronomes", "stemmer")}
                for chave in sorted(r["m"]):
                    if r["m"][chave] is None:
                        continue
                    escala, _ = escala_e_sentido(chave)
                    linhas.append({
                        "experimento": exp["nome"], "configuracao": nome, **contexto,
                        "conjunto": os.path.splitext(os.path.basename(CAMINHO_PERGUNTAS))[0],
                        "id_pergunta": r["id"], "tipo": r["tipo"], "metrica": chave, "escala": escala,
                        "valor": r["m"][chave], "rodada": rodada,
                    })
    return linhas


COLUNAS_LONGO = ["experimento", "configuracao", "modelo", "metodo_tabelas", "metodo_trechos", "divisor", "pronomes",
                 "stemmer", "conjunto", "id_pergunta", "tipo", "metrica", "escala", "valor", "rodada"]


def gravar_csv_longo(caminho: str, linhas: list[dict]):
    with open(caminho, "w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=COLUNAS_LONGO)
        escritor.writeheader()
        for l in linhas:
            escritor.writerow({**l, "valor": f"{l['valor']:.6f}"})


# Resume por configuração, tipo e métrica: soma e média por rodada, e a média e a variação entre rodadas.
def resumir(linhas: list[dict]) -> list[dict]:
    grupos = {}
    for l in linhas:
        for tipo in (l["tipo"], "todos"):
            grupos.setdefault((l["configuracao"], tipo, l["metrica"], l["escala"], l["rodada"]), []).append(l["valor"])
    por_alvo = {}
    for (config, tipo, metrica, escala, rodada), valores in grupos.items():
        por_alvo.setdefault((config, tipo, metrica, escala), []).append((rodada, len(valores), sum(valores)))
    saida = []
    for (config, tipo, metrica, escala), rodadas in por_alvo.items():
        medias = [soma / n for _, n, soma in rodadas]
        saida.append({
            "experimento": linhas[0]["experimento"], "configuracao": config, "tipo": tipo, "metrica": metrica, "escala": escala,
            "n_perguntas": rodadas[0][1], "rodadas": len(rodadas),
            "soma_media": f"{mean(s for _, _, s in rodadas):.4f}", "media": f"{mean(medias):.6f}",
            "minimo": f"{min(medias):.6f}", "maximo": f"{max(medias):.6f}",
        })
    return saida


def gravar_csv_resumo(caminho: str, linhas: list[dict]):
    campos = ["experimento", "configuracao", "tipo", "metrica", "escala", "n_perguntas", "rodadas", "soma_media", "media", "minimo", "maximo"]
    with open(caminho, "w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(linhas)


# Soma, quantidade e média de uma métrica em uma lista de resultados.
def agregar(lista: list[dict], chave: str, tipo: str | None = None):
    valores = [r["m"][chave] for r in lista if (tipo is None or r["tipo"] == tipo) and r["m"].get(chave) is not None]
    return (sum(valores), len(valores), sum(valores) / len(valores)) if valores else (None, 0, None)


# Escreve uma célula "x/n" (contagem) ou a média, no formato do relatório.
def celula(lista: list[dict], chave: str, forma: str, tipo: str | None, piso: bool = False) -> str:
    soma, n, media = agregar(lista, chave, tipo)
    if n == 0:
        return "n/d"
    if forma == "contagem":
        return f"{fmt(soma) if piso else int(round(soma))}/{n}"
    return fmt(media)


# Diz onde o intervalo de 95% fica em relação ao zero (a leitura da diferença do primeiro menos o segundo).
def posicao_intervalo(ic: tuple) -> str:
    if ic[0] > 0:
        return "acima de zero"
    if ic[1] < 0:
        return "abaixo de zero"
    return "inclui zero"


COMANDO_PADRAO = "python pipeline/online/executar_experimentos.py {nome}"


# Monta o relatório do nível recuperação.
def relatorio_recuperacao(exp: dict, perguntas: list[dict], saida: dict, comando: str, parcial: bool) -> str:
    resultados = saida["resultados"]
    t = [f"# Experimento {exp['nome']}: recuperação", ""]
    t += [
        f"- Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')} por `{comando}`.",
        f"- {exp.get('descricao', '')}",
        f"- Conjunto: `{CAMINHO_PERGUNTAS}`, {len(perguntas)} perguntas ({', '.join(f'{sum(1 for p in perguntas if p['tipo'] == tp)} {PLURAL[tp]}' for tp in TIPOS)})." + (" **Execução parcial (só alguns ids): não use como resultado.**" if parcial else ""),
        f"- Os índices têm {saida['n_tabelas']} tabelas e {saida['n_trechos']} trechos.",
        "- Nas perguntas híbridas, as tabelas são buscadas com a pergunta inteira e os trechos com a parte conceitual quando o divisor está ligado (configuração `divisor.ativo`); as divisões ficam em `metricas/divisoes_hibridas.json`.",
        "",
        "## 1. Configurações",
        "",
    ]
    linhas = []
    for nome, config in exp["configs"].items():
        d = descrever_config(config, "quantitativa")
        glossario = metodo_de_busca(config, "glossario", "conceitual")
        linhas.append([nome, d["modelo"], d["metodo_tabelas"], glossario, d["divisor"], d["pronomes"], d["stemmer"], f"{saida['tempos'][nome]:.1f}"])
    t += tabela_md(["Configuração", "Modelo", "Tabelas", "Trechos", "Divisor", "Pronomes", "Stemmer", "Tempo (s)"], linhas)
    n_secao = 2
    for tipo in TIPOS:
        if not any(p["tipo"] == tipo for p in perguntas):
            continue
        medidas = METRICAS_POR_TIPO[tipo]
        t += [f"## {n_secao}. Perguntas {PLURAL[tipo]} (contagens \"x/n\" e médias)", ""]
        corpo = [[nome] + [celula(lista, c, f, tipo, nome == NOME_PISO) for c, _, f in medidas] for nome, lista in resultados.items()]
        t += tabela_md(["Configuração"] + [rotulo for _, rotulo, _ in medidas], corpo)
        n_secao += 1
    if exp.get("comparacoes"):
        t += [f"## {n_secao}. Comparação pareada (primeira configuração menos a segunda)", ""]
        t += ["Diferença média pergunta a pergunta (0,10 equivale a 2 perguntas de 20), intervalo de 95% por bootstrap (10.000 reamostragens, semente 42) e vitórias, derrotas e empates da primeira. A última coluna só diz onde o intervalo fica em relação ao zero.", ""]
        rng = np.random.default_rng(SEMENTE)
        corpo = []
        for a, b in exp["comparacoes"]:
            for tipo in TIPOS:
                for chave, rotulo, _ in METRICAS_POR_TIPO[tipo]:
                    la = [r for r in resultados[a] if r["tipo"] == tipo]
                    lb = [r for r in resultados[b] if r["tipo"] == tipo]
                    c = comparar(la, lb, chave, rng) if la else None
                    if c:
                        corpo.append([f"{a} contra {b}", tipo, rotulo, str(c["n"]), fmt(c["dif"]), f"[{fmt(c['ic'][0])}; {fmt(c['ic'][1])}]",
                                      f"{c['vitorias']}/{c['derrotas']}/{c['empates']}", posicao_intervalo(c["ic"])])
        t += tabela_md(["Comparação", "Tipo", "Métrica", "n", "Diferença média", "IC 95%", "V/D/E", "Intervalo"], corpo)
        n_secao += 1
    modelos = {c["modelo_embedding"] for c in exp["configs"].values()}
    if len(modelos) > 1:
        from modelos_embedding import MODELOS
        t += [f"## {n_secao}. Custo dos modelos", ""]
        corpo = []
        for modelo in sorted(modelos):
            ficha = MODELOS[modelo]
            tamanho = sum(os.path.getsize(caminho_indice(o, modelo)) for o in ("esquema", "glossario") if os.path.exists(caminho_indice(o, modelo))) / 1e6
            corpo.append([modelo, ficha["idioma"], str(ficha["dimensoes"]), str(ficha["limite_tokens"]), f"{tamanho:.1f}"])
        t += tabela_md(["Modelo", "Idioma", "Dimensões", "Limite de tokens", "Índices (MB)"], corpo)
        t += ["O tempo de indexação é impresso por `rodar_offline.py`; o tempo de cada configuração aparece na tabela de configurações.", ""]
        n_secao += 1
    return "\n".join(t + secao_metricas_e_ressalvas(exp, n_secao, comando, "recuperacao"))


# Monta o relatório do nível pipeline.
def relatorio_pipeline(exp: dict, perguntas: list[dict], saida: dict, rodadas: int, comando: str, parcial: bool) -> str:
    por_config = saida["resultados"]
    t = [f"# Experimento {exp['nome']}: pipeline completo", ""]
    t += [
        f"- Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')} por `{comando}`.",
        f"- {exp.get('descricao', '')}",
        f"- Conjunto: `{CAMINHO_PERGUNTAS}`, {len(perguntas)} perguntas; {rodadas} rodada(s)." + (" **Execução parcial (só alguns ids): não use como resultado.**" if parcial else ""),
        "- O cache de SQL vale entre as configurações da mesma rodada e é refeito a cada rodada." + (" A passada de tabelas ideais gera o SQL só com as tabelas de `tabelas_esperadas`." if exp.get("tabelas_ideais") else ""),
        "",
    ]
    n_secao = 1
    for tipo in ("todos",) + TIPOS:
        if tipo != "todos" and not any(p["tipo"] == tipo for p in perguntas):
            continue
        t += [f"## {n_secao}. {'Todas as perguntas' if tipo == 'todos' else 'Perguntas ' + PLURAL[tipo]}", ""]
        medidas = [mt for mt in METRICAS_PIPELINE if any(agregar(r, mt[0], None if tipo == 'todos' else tipo)[1] for rs in por_config.values() for r in rs.values())]
        corpo = []
        for nome, rs in por_config.items():
            linha = [nome]
            for chave, _, forma in medidas:
                por_rodada = [agregar(lista, chave, None if tipo == "todos" else tipo) for lista in rs.values()]
                por_rodada = [x for x in por_rodada if x[1]]
                if not por_rodada:
                    linha.append("n/d")
                elif forma == "contagem":
                    somas = [s for s, _, _ in por_rodada]
                    texto = f"{mean(somas):.1f}/{por_rodada[0][1]}" if len(somas) > 1 else f"{int(somas[0])}/{por_rodada[0][1]}"
                    if len(somas) > 1:
                        texto += f" [{int(min(somas))} a {int(max(somas))}]"
                    linha.append(texto)
                else:
                    medias = [m for _, _, m in por_rodada]
                    texto = fmt(mean(medias))
                    if len(medias) > 1:
                        texto += f" [{fmt(min(medias))} a {fmt(max(medias))}]"
                    linha.append(texto)
            corpo.append(linha)
        t += tabela_md(["Configuração"] + [rotulo for _, rotulo, _ in medidas], corpo)
        n_secao += 1
    t += [f"## {n_secao}. Tempo médio por passo (s)", ""]
    corpo = []
    for nome, rs in por_config.items():
        todos = [r for lista in rs.values() for r in lista]
        corpo.append([nome] + [fmt(agregar(todos, f"tempo_{passo}")[2]) if agregar(todos, f"tempo_{passo}")[1] else "n/d" for passo in PASSOS])
    t += tabela_md(["Configuração"] + list(PASSOS), corpo)
    n_secao += 1
    if rodadas > 1:
        t += ["Com mais de uma rodada, as células mostram a média das rodadas e, entre colchetes, o mínimo e o máximo.", ""]
    return "\n".join(t + secao_metricas_e_ressalvas(exp, n_secao, comando, "pipeline"))


# Seção fixa de métricas e escalas, ressalvas e comandos.
def secao_metricas_e_ressalvas(exp: dict, n: int, comando: str, nivel: str) -> list[str]:
    t = [f"## {n}. Métricas e escalas", "", "| Métrica | Escala | O que mede |", "|---|---|---|"]
    t += [f"| {nome} | {escala} | {texto} |" for nome, escala, texto in DESCRICAO_METRICAS]
    t += ["", f"## {n + 1}. Ressalvas", ""]
    t += [
        "- Amostra pequena: com 20 perguntas por tipo, diferenças de uma ou duas perguntas são ruído. Atravessar o zero não prova que os métodos são iguais.",
        "- O critério de relevância dos trechos é a evidência literal; um trecho útil sem essa frase conta como erro.",
        "- Nas tabelas, a regra de pontuação (melhor registro de resumo ou entidades) e a expansão por chave estrangeira são as mesmas em todos os métodos; o RRF nas tabelas é uma extensão exploratória do que o MimirRAG faz com trechos.",
    ]
    if nivel == "pipeline":
        t += [
            "- O gerador de SQL e a síntese usam o LLM e não dão exatamente o mesmo resultado a cada execução; com uma rodada só, diferenças pequenas não são interpretáveis.",
            "- A fidelidade e a relevância medem sustentação pelo contexto, não a correção do dado; o validador aprova respostas com SQL errado.",
        ]
    t += ["", f"## {n + 2}. Comandos", "", "```", comando, "```", ""]
    return t


# ---------------------------------------------------------------------------
# 6) Principal
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Executa um experimento de avaliação.")
    parser.add_argument("experimento", nargs="?", help="nome do experimento em experimentos.json")
    parser.add_argument("--experimentos", default=CAMINHO_EXPERIMENTOS, help="arquivo de experimentos")
    parser.add_argument("--rodadas", type=int, help="número de rodadas (nível pipeline)")
    parser.add_argument("--ids", help="ids separados por vírgula; roda só essas perguntas (saída com sufixo _parcial)")
    parser.add_argument("--refazer-divisoes", action="store_true", help="refaz as divisões das híbridas com o LLM")
    parser.add_argument("--listar", action="store_true", help="lista os experimentos e sai")
    args = parser.parse_args()

    if args.listar or not args.experimento:
        with open(args.experimentos, encoding="utf-8") as f:
            for nome, exp in json.load(f)["experimentos"].items():
                print(f"{nome} [{exp.get('nivel')}]: {exp.get('descricao', '')}")
                print("   configurações:", "; ".join(c["nome"] for c in exp["configuracoes"]))
        return

    exp = carregar_experimento(args.experimentos, args.experimento)
    ids = [i.strip() for i in args.ids.split(",")] if args.ids else None
    perguntas = carregar_perguntas(ids)
    parcial = bool(ids)
    comando = COMANDO_PADRAO.format(nome=exp["nome"]) + (f" --ids {args.ids}" if ids else "") + (f" --rodadas {args.rodadas}" if args.rodadas else "")
    sufixo = "_parcial" if parcial else ""
    base = os.path.join(PASTA_SAIDA, f"experimento_{exp['nome']}{sufixo}")
    os.makedirs(PASTA_SAIDA, exist_ok=True)

    if exp["nivel"] == "recuperacao":
        saida = rodar_recuperacao(exp, perguntas, args.refazer_divisoes)
        por_config = {nome: {1: lista} for nome, lista in saida["resultados"].items()}
        relatorio = relatorio_recuperacao(exp, perguntas, saida, comando, parcial)
    else:
        rodadas = args.rodadas or exp.get("rodadas", 1)
        saida = rodar_pipeline(exp, perguntas, rodadas)
        por_config = saida["resultados"]
        relatorio = relatorio_pipeline(exp, perguntas, saida, rodadas, comando, parcial)
        with open(base + "_respostas.json", "w", encoding="utf-8") as f:
            json.dump(saida["respostas"], f, ensure_ascii=False, indent=2)

    linhas = linhas_longas(exp, por_config)
    gravar_csv_longo(base + ".csv", linhas)
    gravar_csv_resumo(base + "_resumo.csv", resumir(linhas))
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(relatorio)
    print(relatorio)
    print(f"Arquivos gravados: {base}.md, {base}.csv e {base}_resumo.csv", file=sys.stderr)


if __name__ == "__main__":
    main()
