"""
Etapa: Online (execução dos experimentos).

Função: executor único dos experimentos de avaliação, com dois níveis: "recuperacao" (mede se a busca
traz as tabelas e os trechos certos, sem LLM, exceto o divisor das perguntas híbridas) e "pipeline"
(roda o fluxo completo de main.py para cada pergunta, com LLM, e mede as demais métricas).

Descrição geral do funcionamento:
- Um experimento é um grupo de configurações comparadas entre si e está descrito em experimentos.json
  (na raiz do repositório): nível, configurações nomeadas e, no nível pipeline,
  rodadas e a passada de tabelas ideais. Cada configuração é uma lista de trocas "chave=valor" sobre o
  configuracao.json, no formato do --set (configuracao.py), então qualquer chave pode variar: modelo de
  embeddings, método de busca por índice e por tipo, k, divisor, bloqueio de pronomes, radicalizador.
- O conjunto de perguntas é sempre dados/perguntas_validacao.json (60 perguntas, o conjunto oficial).
- Nível recuperação: para cada configuração e pergunta, busca as tabelas (quantitativas e híbridas) e
  os trechos dos normativos (conceituais e híbridas) com o método da configuração e calcula as métricas
  de metricas.py (todas as tabelas no top-k, Recall, MRR, Hit@k, Precision@k, nDCG@k e o
  acerto conjunto das híbridas). Com o divisor ligado, as híbridas buscam as tabelas com a pergunta
  inteira e os trechos com a parte conceitual (condição "mista"). As divisões são feitas uma vez e
  guardadas em metricas/divisoes_hibridas.json, para todas as configurações usarem as mesmas.
- Nível pipeline: para cada rodada, configuração e pergunta, chama executar() de main.py e mede a
  classe prevista, o EX (estrito e adaptado, que aceita a coluna DENOM_CIA e linhas de companhias com o nome da pergunta), a
  o tempo de cada passo; guarda no respostas.json o texto da resposta, o contexto e o resultado do SQL,
  que o ragas_avaliacao.py usa para a RAGAS (fidelidade e relevância da resposta). O SQL gerado
  fica em um cache por (pergunta, tabelas do contexto) que vale entre as configurações da mesma
  rodada e é refeito a cada rodada, para a variação do LLM aparecer nas rodadas. Com
  "tabelas_ideais": true, uma passada extra gera o SQL só com as tabelas de tabelas_esperadas.
- Sem teste estatístico: o relatório mostra as métricas de cada configuração, por tipo de pergunta
  (contagens "x/n" e médias), e quem compara é o leitor, como em BEIR, E5 e BGE-M3. Com 20 perguntas por
  tipo, cada pergunta vale 5 pontos percentuais; os números comparam as configurações nesta amostra.

Dependências:
- pipeline/avaliacao/metricas.py; pipeline/online/busca*.py, divisor.py e main.py; os índices e o
  esquema em embeddings/ (os de outros modelos, em embeddings/modelos/<chave>/),
  dados/perguntas_validacao.json, dados/dfp.db e experimentos.json;
- servidor Ollama com o modelo qwen2.5-coder:7b (divisor e nível pipeline).

Parâmetros: o nome do experimento, e opcionalmente --rodadas N (nível pipeline), --ids q01,c01,...
(roda só essas perguntas, como rascunho), --refazer-divisoes, --listar e --experimentos CAMINHO (outro
arquivo de experimentos).

Resultado: uma pasta por experimento, metricas/<nome>/, com nomes fixos que são sobrescritos a cada
execução: resultados.csv (uma linha por rodada, configuração, pergunta e métrica), resumo.csv (uma
linha por configuração, tipo e métrica), relatorio.md (tabelas, comparações, métricas e escalas,
ressalvas e comandos) e execucao.json, a ficha que identifica a execução; no nível pipeline, também
respostas.json. Execuções de teste (--ids, ou "rascunho": true no experimento) vão para
metricas/rascunhos/<nome>/, pasta ignorada pelo git, para não se misturarem com as oficiais.

Identificação e versão. Cada execução recebe um identificador (data e hora, AAAAMMDD-HHMMSS) que aparece
no relatorio.md e na coluna "execucao" dos CSV. A ficha execucao.json guarda o comando, o commit do
código (com aviso de alterações não commitadas), a impressão digital (SHA-256) da configuração, dos
experimentos, das perguntas e dos índices, os modelos, as versões das bibliotecas, as configurações
efetivas, o número de rodadas e a duração. A versão de uma execução oficial é o commit que guarda a
pasta, com o identificador na mensagem, e uma tag (<nome>-v1, <nome>-v2...); o executor imprime os
comandos sugeridos no fim, mas não faz commit nem cria tag.

Executar de dentro de rag-nl2sql-financas-br/:
    python pipeline/avaliacao/executar_experimentos.py --listar
    python pipeline/avaliacao/executar_experimentos.py metodos
    python pipeline/avaliacao/executar_experimentos.py teste_pipeline --ids q01,c01,h01
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from statistics import mean

# Os módulos do fluxo ficam em pipeline/online/ e os compartilhados em pipeline/.
PASTA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PASTA, "..", "online"))
sys.path.insert(0, os.path.join(PASTA, ".."))

from metricas import (  # noqa: E402
    KS_TABELAS,
    KS_TRECHOS,
    TABELA_COMUM,
    fmt,
    NomesDeCompanhias,
    metricas_recuperacao,
    normalizar_texto,
    resultado_adaptado,
    resultados_equivalentes,
    tabela_md,
)
from busca import (
    SECOES_FORA_DA_PONTUACAO,
    Indice,
    caminho_indice,
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
PASTA_RASCUNHOS = os.path.join(PASTA_SAIDA, "rascunhos")
CAMINHO_DIVISOES = "/".join((PASTA_SAIDA, "divisoes_hibridas.json"))
TIPOS = ("quantitativa", "conceitual", "hibrida")
PLURAL = {"quantitativa": "quantitativas", "conceitual": "conceituais", "hibrida": "híbridas"}

# Índices avaliados em cada tipo de pergunta: (prefixo das chaves, rótulo, cortes de k).
INDICES_POR_TIPO = {
    "quantitativa": (("tab", "tabelas", KS_TABELAS),),
    "conceitual": (("glo", "trechos", KS_TRECHOS),),
    "hibrida": (("tab", "tabelas", KS_TABELAS), ("glo", "trechos", KS_TRECHOS)),
}
FAMILIAS = (("prec", "Precision@k"), ("recall", "Recall@k"), ("ndcg", "nDCG@k"))
METRICAS_PIPELINE = [
    ("classe_certa", "Classe certa", "contagem"),
    ("ex_adaptado", "EX adaptado", "contagem"),
    ("ex_estrito", "EX estrito", "contagem"),
    ("ideal_ex_adaptado", "EX adaptado com tabelas ideais", "contagem"),
    ("ideal_ex_estrito", "EX estrito com tabelas ideais", "contagem"),
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
    binarias = ("classe_certa", "ex_", "ideal_ex_")
    if chave.startswith(binarias):
        return "0 ou 1", "maior é melhor"
    return "0 a 1", "maior é melhor"


DESCRICAO_METRICAS = [
    ("Precision@k (`tab_prec@k`, `glo_prec@k`)", "0 a 1", "Fração dos k primeiros itens que são relevantes (tabelas ou trechos). Divide sempre por k. Com uma só evidência por pergunta, o máximo nos trechos é 1/k. Maior é melhor."),
    ("Recall@k (`tab_recall@k`, `glo_recall@k`)", "0 a 1", "Fração dos itens relevantes que aparece entre os k primeiros. Maior é melhor."),
    ("nDCG@k (`tab_ndcg@k`, `glo_ndcg@k`)", "0 a 1", "Considera a ordem da lista, dando mais peso a um acerto no topo; relevância binária. A ranx usa o desconto 1/log2(i+1), e não o de Järvelin da fundamentação. Maior é melhor."),
    ("MRR (`tab_rr`, `glo_rr`)", "0 a 1", "Média do inverso da posição do primeiro item relevante. Maior é melhor."),
    ("Classe certa (`classe_certa`)", "0 ou 1", "O classificador acertou o tipo da pergunta."),
    ("EX estrito (`ex_estrito`)", "0 ou 1", "O resultado do SQL gerado é igual ao do `sql_referencia` (ordem das colunas e das linhas ignorada, números com 2 casas). Só nas perguntas com SQL."),
    ("EX adaptado (`ex_adaptado`)", "0 ou 1", "Vale se o EX estrito vale ou se o resultado traz todas as linhas da referência (a coluna `DENOM_CIA` é tolerada) e as linhas a mais são de companhias cujo `DENOM_CIA` contém a palavra que identifica a companhia dita na pergunta. Adaptação do EX de Zhong et al. (2020), sem artigo; mais permissiva que o EX estrito."),
    ("Tabelas ideais (`ideal_ex_*`)", "0 ou 1", "EX do SQL gerado quando o gerador recebe as tabelas certas (`tabelas_esperadas`) em vez das da busca. É um teto, e a diferença para o EX normal mostra quanto da perda vem da busca."),
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
    # Prepara o que não depende da configuração: tamanhos dos índices e trechos relevantes.
    def __init__(self, perguntas: list[dict]):
        self._indices = {}
        padrao = self.indice("glossario", "embeddings", MODELO_PADRAO, False)
        self.n_trechos = len(padrao.registros)
        self.n_tabelas = len({r["fonte"] for r in self.indice("esquema", "embeddings", MODELO_PADRAO, False).registros})
        # Os trechos relevantes de cada pergunta não dependem do método nem do modelo (o texto dos
        # registros é o mesmo em todos os índices).
        self.relevantes = {}
        for p in perguntas:
            if p["evidencia"]:
                ids = {r["id"] for r in padrao.registros if any(normalizar_texto(e) in normalizar_texto(r["conteudo"]) for e in p["evidencia"])}
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
            # A tabela companhias fica fora: o sistema a acrescenta por regra a todas as listas.
            ranking = [r["fonte"] for r in completo if r["fonte"] != TABELA_COMUM]
            esperadas = [t for t in p["tabelas_esperadas"] if t != TABELA_COMUM]
            m.update(metricas_recuperacao("tab", ranking, esperadas, KS_TABELAS))
            posicoes["tabelas"] = {t: ranking.index(t) + 1 for t in esperadas}
        if p["evidencia"]:
            metodo = metodo_de_busca(config, "glossario", tipo)
            texto = p["pergunta"]
            if divisao and divisao["dividida"]:
                texto = divisao["parte_conceitual"]
            indice = self.indice("glossario", metodo, modelo, radicalizar)
            consulta = self.consulta(metodo, texto, modelo)
            ranking_trechos = [r["id"] for r in indice.buscar(consulta, self.n_trechos)]
            m.update(metricas_recuperacao("glo", ranking_trechos, self.relevantes[p["id"]], KS_TRECHOS))
            posicoes["trechos"] = [i for i, r in enumerate(ranking_trechos, start=1) if r in self.relevantes[p["id"]]][:5]
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
    return {"resultados": resultados, "tempos": tempos, "divisoes": usadas, "n_tabelas": recuperador.n_tabelas, "n_trechos": recuperador.n_trechos}


# ---------------------------------------------------------------------------
# 4) Nível pipeline
# ---------------------------------------------------------------------------

# Compara o SQL gerado com o de referência e devolve (EX estrito, EX aceitando outras companhias).
def calcular_ex(r_sql: dict | None, p: dict, nomes: NomesDeCompanhias) -> tuple[bool, bool]:
    from gerador_sql import executar_sql

    if not r_sql or r_sql["status"] != "ok":
        return False, False
    _, referencia = executar_sql(p["sql_referencia"], limite_linhas=None)
    estrito = resultados_equivalentes(r_sql["linhas"], referencia)
    termos = nomes.termos(p["pergunta"], [b["companhia"] for b in p.get("base_dados") or []])
    return estrito, resultado_adaptado(r_sql["colunas"], r_sql["linhas"], referencia, termos)


# Lê do banco os nomes das companhias, para achar a palavra que identifica a companhia de cada pergunta.
def carregar_nomes_companhias() -> NomesDeCompanhias:
    conn = sqlite3.connect("file:dados/dfp.db?mode=ro", uri=True)
    try:
        return NomesDeCompanhias([linha[0] for linha in conn.execute("SELECT DENOM_CIA FROM companhias")])
    finally:
        conn.close()


# Roda o nível pipeline: rodadas, configurações e perguntas. Devolve linhas de métricas e as respostas.
def rodar_pipeline(exp: dict, perguntas: list[dict], rodadas: int) -> dict:
    from gerador_sql import gerar_sql
    from main import executar

    nomes = carregar_nomes_companhias()
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
                m = {
                    "classe_certa": float(r["classe"] == p["tipo"]),
                    "tempo_total": total,
                }
                for passo in PASSOS:
                    if passo in r["tempos"]:
                        m[f"tempo_{passo}"] = r["tempos"][passo]
                if p["sql_referencia"]:
                    estrito, adaptado = calcular_ex(r_sql, p, nomes)
                    m.update(ex_estrito=float(estrito), ex_adaptado=float(adaptado))
                    if exp.get("tabelas_ideais"):
                        ideal = gerar_sql(p["pergunta"], tabelas=p["tabelas_esperadas"], cache=cache)
                        e_i, a_i = calcular_ex(ideal, p, nomes)
                        m.update(ideal_ex_estrito=float(e_i), ideal_ex_adaptado=float(a_i))
                resultados[nome][rodada].append({"id": p["id"], "tipo": p["tipo"], "m": m, "posicoes": {}})
                respostas.append({
                    "rodada": rodada, "configuracao": nome, "id": p["id"], "tipo": p["tipo"], "classe": r["classe"],
                    "pergunta": p["pergunta"], "divisao": r["divisao"], "sql": r_sql["sql"] if r_sql else None,
                    "tabelas_contexto": r_sql["tabelas_contexto"] if r_sql else None, "resposta": r["resposta"],
                    # Para a pontuação da RAGAS (ragas_avaliacao.py): o texto que o LLM escreveu, o contexto que ele
                    # recebeu e o resultado do SQL (até 30 linhas).
                    "texto_resposta": r["texto_resposta"], "contexto": r["contexto"], "trechos_ids": [t["id"] for t in r["trechos"]],
                    "resultado_sql": {"colunas": r_sql["colunas"], "linhas": [list(l) for l in r_sql["linhas"][:30]]} if r_sql and r_sql["status"] == "ok" else None,
                })
    return {"resultados": resultados, "respostas": respostas}


# ---------------------------------------------------------------------------
# 5) Identificação da execução (ficha)
# ---------------------------------------------------------------------------

# Impressão digital (SHA-256) de um arquivo, lida em blocos.
def sha256_arquivo(caminho: str) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


# Executa um comando do git e devolve a saída, ou None se o git não estiver disponível.
def _git(*args) -> str | None:
    try:
        r = subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8", timeout=30)
        return r.stdout.rstrip("\n") if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


# Commit do código e se há alterações não commitadas (os resultados em metricas/ não contam).
def info_git() -> dict:
    commit = _git("rev-parse", "HEAD")
    if not commit:
        return {"commit": None}
    status = _git("status", "--porcelain", "--", ".", ":(exclude)metricas") or ""
    alterados = [linha[3:] for linha in status.splitlines()]
    return {
        "commit": commit, "commit_curto": commit[:7], "ramo": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "alteracoes_nao_commitadas": bool(alterados), "arquivos_alterados": alterados[:20],
    }


# Versões do Python e das bibliotecas que influenciam os resultados.
def versoes() -> dict:
    saida = {"python": platform.python_version(), "sistema": platform.platform()}
    for pacote in ("numpy", "sentence-transformers", "transformers", "torch", "rank-bm25", "nltk", "ollama"):
        try:
            saida[pacote] = version(pacote)
        except PackageNotFoundError:
            saida[pacote] = None
    return saida


# Identifica o modelo de linguagem no Ollama (nome e digest), se o servidor responder.
def info_llm() -> dict:
    from gerador_sql import MODELO_LLM

    try:
        import ollama

        for m in ollama.list().models:
            if m.model == MODELO_LLM:
                return {"modelo": MODELO_LLM, "digest": m.digest, "tamanho_bytes": m.size}
    except Exception:
        pass
    return {"modelo": MODELO_LLM, "digest": None}


# Monta a ficha da execução com tudo o que identifica o teste, antes de rodá-lo.
def iniciar_ficha(exp: dict, perguntas: list[dict], comando: str, ids: list[str] | None, rascunho: bool, rodadas: int) -> dict:
    from modelos_embedding import MODELOS

    modelos = sorted({c["modelo_embedding"] for c in exp["configs"].values()})
    arquivos = ["configuracao.json", "experimentos.json", CAMINHO_PERGUNTAS, "embeddings/esquema.json", CAMINHO_DIVISOES,
                "embeddings/index_esquema.json", "embeddings/index_glossario.json",
                "embeddings/index_bm25_esquema.json", "embeddings/index_bm25_glossario.json"]
    for modelo in modelos:
        if modelo != MODELO_PADRAO:
            arquivos += [caminho_indice("esquema", modelo), caminho_indice("glossario", modelo)]
    usa_llm = exp["nivel"] == "pipeline" or any(c["divisor"]["ativo"] for c in exp["configs"].values())
    agora = datetime.now()
    return {
        "execucao": agora.strftime("%Y%m%d-%H%M%S"),
        "experimento": exp["nome"], "nivel": exp["nivel"], "descricao": exp.get("descricao", ""),
        "comando": comando, "inicio": agora.isoformat(timespec="seconds"),
        "rascunho": rascunho, "ids": ids, "n_perguntas": len(perguntas), "rodadas": rodadas,
        "tabelas_ideais": bool(exp.get("tabelas_ideais")),
        "git": info_git(),
        "arquivos_sha256": {a: sha256_arquivo(a) for a in dict.fromkeys(arquivos) if os.path.exists(a)},
        "modelos_embeddings": {m: MODELOS[m]["id"] for m in modelos},
        "llm": info_llm() if usa_llm else None,
        "bibliotecas": versoes(),
        "configuracoes": exp["configs"],
    }


# Linha do relatório que identifica a execução.
def linha_execucao(ficha: dict) -> str:
    g = ficha["git"]
    commit = f"commit `{g['commit_curto']}`" + (" **com alterações não commitadas**" if g["alteracoes_nao_commitadas"] else "") if g.get("commit") else "commit n/d"
    return f"- Execução `{ficha['execucao']}`, {commit}. A ficha completa está em `execucao.json`."


# Pasta de saída: metricas/<nome>/, ou metricas/rascunhos/<nome>/ para testes e execuções parciais.
def pasta_de_saida(nome: str, rascunho: bool) -> str:
    if nome == "rascunhos":
        raise ValueError("'rascunhos' é nome reservado")
    return os.path.join(PASTA_RASCUNHOS if rascunho else PASTA_SAIDA, nome)


# ---------------------------------------------------------------------------
# 6) Saídas
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
def linhas_longas(exp: dict, por_config: dict, execucao: str) -> list[dict]:
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
                        "experimento": exp["nome"], "execucao": execucao, "configuracao": nome, **contexto,
                        "conjunto": os.path.splitext(os.path.basename(CAMINHO_PERGUNTAS))[0],
                        "id_pergunta": r["id"], "tipo": r["tipo"], "metrica": chave, "escala": escala,
                        "valor": r["m"][chave], "rodada": rodada,
                    })
    return linhas


COLUNAS_LONGO = ["experimento", "execucao", "configuracao", "modelo", "metodo_tabelas", "metodo_trechos", "divisor", "pronomes",
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
            "experimento": linhas[0]["experimento"], "execucao": linhas[0]["execucao"], "configuracao": config, "tipo": tipo, "metrica": metrica, "escala": escala,
            "n_perguntas": rodadas[0][1], "rodadas": len(rodadas),
            "soma_media": f"{mean(s for _, _, s in rodadas):.4f}", "media": f"{mean(medias):.6f}",
            "minimo": f"{min(medias):.6f}", "maximo": f"{max(medias):.6f}",
        })
    return saida


def gravar_csv_resumo(caminho: str, linhas: list[dict]):
    campos = ["experimento", "execucao", "configuracao", "tipo", "metrica", "escala", "n_perguntas", "rodadas", "soma_media", "media", "minimo", "maximo"]
    with open(caminho, "w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(linhas)


# Soma, quantidade e média de uma métrica em uma lista de resultados.
def agregar(lista: list[dict], chave: str, tipo: str | None = None):
    valores = [r["m"][chave] for r in lista if (tipo is None or r["tipo"] == tipo) and r["m"].get(chave) is not None]
    return (sum(valores), len(valores), sum(valores) / len(valores)) if valores else (None, 0, None)


# Escreve uma célula "x/n" (contagem) ou a média, no formato do relatório.
def celula(lista: list[dict], chave: str, forma: str, tipo: str | None) -> str:
    soma, n, media = agregar(lista, chave, tipo)
    if n == 0:
        return "n/d"
    if forma == "contagem":
        return f"{int(round(soma))}/{n}"
    return fmt(media)


COMANDO_PADRAO = "python pipeline/avaliacao/executar_experimentos.py {nome}"


# Monta o relatório do nível recuperação.
def relatorio_recuperacao(exp: dict, perguntas: list[dict], saida: dict, comando: str, parcial: bool, ficha: dict) -> str:
    resultados = saida["resultados"]
    t = [f"# Experimento {exp['nome']}: recuperação", ""]
    t += [
        f"- Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')} por `{comando}`.",
        linha_execucao(ficha),
        f"- {exp.get('descricao', '')}",
        f"- Conjunto: `{CAMINHO_PERGUNTAS}`, {len(perguntas)} perguntas ({', '.join(f'{sum(1 for p in perguntas if p['tipo'] == tp)} {PLURAL[tp]}' for tp in TIPOS)})." + (" **Rascunho (teste ou só alguns ids): não use como resultado.**" if parcial else ""),
        f"- Os índices têm {saida['n_tabelas']} tabelas e {saida['n_trechos']} trechos.",
        f"- Métricas calculadas com a biblioteca ranx {version('ranx')}: Precision@k, Recall@k, nDCG@k e MRR, com relevância binária. Nas tabelas, a tabela `companhias` fica fora do ranking e dos itens esperados (o sistema a acrescenta por regra a todas as listas), então as k tabelas medidas são as k mais parecidas com a pergunta.",
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
        t += [f"## {n_secao}. Perguntas {PLURAL[tipo]} (médias)", ""]
        for prefixo, rotulo, ks in INDICES_POR_TIPO[tipo]:
            t += [f"### Recuperação de {rotulo}", ""]
            for familia, titulo in FAMILIAS:
                corpo = [[nome] + [celula(lista, f"{prefixo}_{familia}@{k}", "media", tipo) for k in ks] for nome, lista in resultados.items()]
                t += [f"**{titulo}**", ""] + tabela_md(["Configuração"] + [f"k = {k}" for k in ks], corpo)
            corpo = [[nome, celula(lista, f"{prefixo}_rr", "media", tipo)] for nome, lista in resultados.items()]
            t += ["**MRR**", ""] + tabela_md(["Configuração", "MRR"], corpo)
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
# Monta a matriz de confusão do classificador de uma configuração: linhas = tipo real, colunas = tipo previsto.
# Soma as rodadas; uma classe fora de TIPOS (se o classificador devolver algo inesperado) ganha coluna própria.
def matriz_confusao(respostas: list[dict], configuracao: str) -> list[str]:
    contagem = {}
    for r in respostas:
        if r["configuracao"] == configuracao:
            contagem[(r["tipo"], r["classe"])] = contagem.get((r["tipo"], r["classe"]), 0) + 1
    previstas = list(TIPOS) + sorted({c for _, c in contagem if c not in TIPOS})
    linhas = []
    for real in TIPOS:
        total = sum(contagem.get((real, c), 0) for c in previstas)
        if total:
            linhas.append([real] + [str(contagem.get((real, c), 0)) for c in previstas] + [str(total)])
    return tabela_md(["Real / previsto"] + previstas + ["Total"], linhas)


def relatorio_pipeline(exp: dict, perguntas: list[dict], saida: dict, rodadas: int, comando: str, parcial: bool, ficha: dict) -> str:
    por_config = saida["resultados"]
    t = [f"# Experimento {exp['nome']}: pipeline completo", ""]
    t += [
        f"- Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')} por `{comando}`.",
        linha_execucao(ficha),
        f"- {exp.get('descricao', '')}",
        f"- Conjunto: `{CAMINHO_PERGUNTAS}`, {len(perguntas)} perguntas; {rodadas} rodada(s)." + (" **Rascunho (teste ou só alguns ids): não use como resultado.**" if parcial else ""),
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
    t += [f"## {n_secao}. Matriz de confusão do classificador", ""]
    t += ["Cada linha é o tipo real da pergunta e cada coluna, o tipo que o classificador previu; a diagonal são os acertos. Com mais de uma rodada, as contagens somam as rodadas.", ""]
    for nome in por_config:
        t += [f"**{nome}**", ""] + matriz_confusao(saida["respostas"], nome)
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
        "- Amostra pequena: com 20 perguntas por tipo, cada pergunta vale 5 pontos percentuais. Os números comparam as configurações nesta amostra, sem teste estatístico, e não permitem afirmar superioridade estatística de uma sobre outra.",
        "- O critério de relevância dos trechos é a evidência literal; um trecho útil sem essa frase conta como erro.",
        "- Nas tabelas, a regra de pontuação (melhor registro de resumo ou entidades) é a mesma em todos os métodos e a tabela `companhias` fica fora da medição; o RRF nas tabelas é uma extensão exploratória do que o MimirRAG faz com trechos.",
        "- Com uma única evidência por pergunta, a Precision@k dos trechos tem máximo 1/k (por exemplo, 0,2 com k = 5). O nDCG vem da ranx, com o desconto 1/log2(i+1), diferente do de Järvelin da fundamentação.",
    ]
    if nivel == "pipeline":
        t += [
            "- O gerador de SQL e a síntese usam o LLM e não dão exatamente o mesmo resultado a cada execução; com uma rodada só, diferenças pequenas não são interpretáveis.",
            "- A fidelidade e a relevância da resposta (RAGAS) não estão neste relatório: saem de `ragas_avaliacao.py`, que lê o `respostas.json` desta pasta e grava `ragas.csv` e `ragas.md`.",
        ]
    t += ["", f"## {n + 2}. Comandos", "", "```", comando, "```", ""]
    return t


# ---------------------------------------------------------------------------
# 7) Principal
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
    rascunho = bool(ids) or bool(exp.get("rascunho"))
    rodadas = (args.rodadas or exp.get("rodadas", 1)) if exp["nivel"] == "pipeline" else 1
    comando = COMANDO_PADRAO.format(nome=exp["nome"]) + (f" --ids {args.ids}" if ids else "") + (f" --rodadas {args.rodadas}" if args.rodadas else "")
    pasta = pasta_de_saida(exp["nome"], rascunho)
    os.makedirs(pasta, exist_ok=True)
    ficha = iniciar_ficha(exp, perguntas, comando, ids, rascunho, rodadas)
    inicio = time.perf_counter()

    if exp["nivel"] == "recuperacao":
        saida = rodar_recuperacao(exp, perguntas, args.refazer_divisoes)
        por_config = {nome: {1: lista} for nome, lista in saida["resultados"].items()}
        relatorio = relatorio_recuperacao(exp, perguntas, saida, comando, rascunho, ficha)
    else:
        saida = rodar_pipeline(exp, perguntas, rodadas)
        por_config = saida["resultados"]
        relatorio = relatorio_pipeline(exp, perguntas, saida, rodadas, comando, rascunho, ficha)
        with open(os.path.join(pasta, "respostas.json"), "w", encoding="utf-8") as f:
            json.dump(saida["respostas"], f, ensure_ascii=False, indent=2)

    linhas = linhas_longas(exp, por_config, ficha["execucao"])
    gravar_csv_longo(os.path.join(pasta, "resultados.csv"), linhas)
    gravar_csv_resumo(os.path.join(pasta, "resumo.csv"), resumir(linhas))
    with open(os.path.join(pasta, "relatorio.md"), "w", encoding="utf-8") as f:
        f.write(relatorio)
    # O cache das divisões pode ter sido criado ou atualizado durante a execução.
    if os.path.exists(CAMINHO_DIVISOES):
        ficha["arquivos_sha256"][CAMINHO_DIVISOES] = sha256_arquivo(CAMINHO_DIVISOES)
    ficha["fim"] = datetime.now().isoformat(timespec="seconds")
    ficha["duracao_s"] = round(time.perf_counter() - inicio, 1)
    with open(os.path.join(pasta, "execucao.json"), "w", encoding="utf-8") as f:
        json.dump(ficha, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(relatorio)
    print(f"Execução {ficha['execucao']} gravada em {pasta}/ (relatorio.md, resultados.csv, resumo.csv, execucao.json).", file=sys.stderr)
    if not rascunho:
        existentes = (_git("tag", "-l", f"{exp['nome']}-v*") or "").split()
        proxima = f"{exp['nome']}-v{len(existentes) + 1}"
        print("Para registrar esta execução como versão oficial:\n"
              f"  git add {pasta}\n"
              f"  git commit -m \"data: resultados do experimento {exp['nome']} (execução {ficha['execucao']})\"\n"
              f"  git tag {proxima}", file=sys.stderr)

if __name__ == "__main__":
    main()
