"""
Etapa: Online.

Função: caminho quantitativo, geração e execução do SQL sobre as DFPs.

Descrição geral do funcionamento:
- A busca escolhe as k tabelas mais parecidas com a pergunta no índice de esquema, mais as
  tabelas que elas referenciam por chave estrangeira (companhias). Também é possível
  passar a lista de tabelas, sem usar a busca (usado para medir o gerador com tabelas ideais).
- O contexto traz só essas tabelas (colunas, tipos, valores possíveis, plano de contas e
  junções), lido do esquema.json. O esquema completo não vai ao LLM.
- Cada conta do plano de contas aparece com o filtro CD_CONTA pronto e com os termos de
  analista (sinônimos) de dados/sinonimos_contas.json, como "receita líquida" para a conta 3.01.
- O LLM gera o SQL. O código só aceita um único SELECT e executa em modo somente leitura.
- Se der erro ou o resultado vier vazio, o LLM recebe o problema e gera outro SQL, até o
  limite de tentativas.

Dependências:
- pipeline/online/busca.py, os índices e o esquema.json em embeddings/, dados/dfp.db e
  dados/sinonimos_contas.json;
- servidor Ollama em execução, com o modelo qwen2.5-coder:7b baixado.

Parâmetros: nenhum argumento. É um módulo importado pelos demais passos do estágio online.

Resultado: fornece gerar_sql(), executar_sql() e validar_sql().
"""

import json
import re
import sqlite3
import time

import ollama

from busca import (
    CAMINHO_ESQUEMA,
    CAMINHO_INDICE_ESQUEMA,
    Indice,
    buscar_tabelas,
    carregar_tabelas_referenciadas,
    vetorizar,
)

CAMINHO_DB = "dados/dfp.db"
CAMINHO_SINONIMOS = "dados/sinonimos_contas.json"
MODELO_LLM = "qwen2.5-coder:7b"
# Número de tabelas escolhidas pela busca (as referenciadas por chave estrangeira vêm além delas).
K_TABELAS = 3
# O contexto com o plano de contas passa do tamanho padrão do Ollama, que cortaria o início do prompt.
NUM_CTX = 16384
MAX_TENTATIVAS = 3
LIMITE_LINHAS = 200
TEMPO_MAXIMO_SEGUNDOS = 30
# Palavras que nunca podem aparecer em um SQL gerado (a conexão já é somente
# leitura, isto é uma segunda camada de proteção).
PALAVRAS_PROIBIDAS = {
    "insert", "update", "delete", "drop", "alter", "create", "attach",
    "detach", "pragma", "vacuum", "reindex", "truncate",
}

_indice = None
_esquema = None
_referenciadas = None
_sinonimos = None


# Lê o índice, o esquema e os sinônimos das contas uma vez só e os reaproveita nas consultas seguintes.
def _carregar():
    global _indice, _esquema, _referenciadas, _sinonimos
    if _indice is None:
        _indice = Indice(CAMINHO_INDICE_ESQUEMA)
        with open(CAMINHO_ESQUEMA, encoding="utf-8") as f:
            _esquema = {e["tabela"]: e for e in json.load(f)}
        _referenciadas = carregar_tabelas_referenciadas()
        with open(CAMINHO_SINONIMOS, encoding="utf-8") as f:
            _sinonimos = json.load(f)
    return _indice, _esquema, _referenciadas


# Devolve as entradas do esquema.json das tabelas da busca, ou das tabelas passadas.
def selecionar_contexto(pergunta: str, k: int = K_TABELAS, tabelas: list[str] | None = None) -> list[dict]:
    indice, esquema, referenciadas = _carregar()
    if tabelas is None:
        tabelas = buscar_tabelas(indice, vetorizar(pergunta), k, referenciadas)
    return [esquema[t] for t in tabelas]


# Monta o texto das tabelas escolhidas.
def montar_contexto(entradas: list[dict]) -> str:
    # Só entram junções entre tabelas que estão no contexto, para o LLM não usar
    # uma tabela que não foi mostrada a ele.
    nomes = {e["tabela"] for e in entradas}
    blocos, juncoes = [], set()
    for e in entradas:
        linhas = [f'Tabela "{e["tabela"]}": {e["resumo"]}', "Colunas:"]
        for c in e["colunas"]:
            chave = ", chave primária" if c["chave_primaria"] else ""
            valores = ""
            if c["valores_possiveis"]:
                valores = " Valores possíveis: " + ", ".join(f"'{v}'" for v in c["valores_possiveis"]) + "."
            linhas.append(f'  - {c["nome"]} ({c["tipo"]}{chave}): {c["descricao"]}.{valores}')
        # O significado das contas está no valor de CD_CONTA, não no nome da coluna. Cada linha
        # traz o filtro pronto e os termos de analista (sinônimos) da conta, quando existem.
        if e["contas"]:
            linhas.append("Plano de contas (filtre com CD_CONTA = código):")
            # Os sinônimos valem para a demonstração toda (BPA_con e BPA_ind usam os de BPA).
            da_demonstracao = _sinonimos.get(e["tabela"].rsplit("_", 1)[0], {})
            for c in e["contas"]:
                termos = da_demonstracao.get(c["codigo"])
                chamada = f' (também chamada de: {", ".join(termos)})' if termos else ""
                linhas.append(f"  - {c['descricao']}{chamada} -> CD_CONTA = '{c['codigo']}'")
        blocos.append("\n".join(linhas))
        for r in e["chaves_de_juncao"]:
            if r["tabela_relacionada"] not in nomes:
                continue
            if r["sentido"] == "referencia":
                juncoes.add(f'{e["tabela"]}.{r["coluna"]} = {r["tabela_relacionada"]}.{r["coluna_relacionada"]}')
            else:
                juncoes.add(f'{r["tabela_relacionada"]}.{r["coluna_relacionada"]} = {e["tabela"]}.{r["coluna"]}')
    texto = "\n\n".join(blocos)
    if juncoes:
        texto += "\n\nJunções possíveis entre estas tabelas:\n" + "\n".join(f"  - {j}" for j in sorted(juncoes))
    return texto


# Monta as instruções do LLM, com o contexto e as regras do banco.
def montar_instrucoes(contexto: str) -> str:
    return f"""Você escreve consultas SQL para um banco SQLite com as demonstrações financeiras (DFP) de companhias abertas brasileiras. Use somente as tabelas e colunas descritas abaixo.

{contexto}

Regras:
- Escreva um único comando SELECT, sem comentários.
- Use os nomes exatos das tabelas e colunas.
- Para juntar tabelas, use as junções listadas.
- O valor de uma conta está em VL_CONTA. A conta é escolhida pelo CD_CONTA, com o código exato do plano de contas da tabela.
- Tabelas terminadas em _con são consolidadas e terminadas em _ind são individuais (só da controladora). Se a pergunta não disser, use a consolidada.
- Todos os dados são do exercício de 2025, então não filtre por ano. Use ORDEM_EXERC = 'ÚLTIMO' para 2025 e 'PENÚLTIMO' para 2024.
- O nome da companhia fica em companhias.DENOM_CIA, em letras maiúsculas. Filtre com LIKE e uma parte do nome, por exemplo DENOM_CIA LIKE '%AMBEV%'. Mais de uma companhia pode combinar com o nome. Quando a pergunta citar companhias, não use SUM, GROUP BY nem LIMIT para reuni-las em uma só: devolva uma linha para cada companhia e coloque companhias.DENOM_CIA como primeira coluna.
- VL_CONTA está na escala da coluna ESCALA_MOEDA: 'MIL' quer dizer milhares de reais e 'UNIDADE' quer dizer reais. Devolva e compare sempre os valores em reais, trocando VL_CONTA por CASE WHEN ESCALA_MOEDA = 'MIL' THEN VL_CONTA * 1000 ELSE VL_CONTA END, inclusive ao ordenar, somar ou comparar companhias. Não use SUM para o valor de uma única conta.
- Quando combinar condições com OR e AND, use parênteses.
- Devolva só as colunas que a pergunta pede, mais DENOM_CIA quando ela citar companhias.
- Se a pergunta também pedir uma explicação ou definição, ignore essa parte e escreva o SQL só para a parte que pede dados.
- Responda somente com um JSON no formato {{"sql": "<consulta>"}}."""


# Tira o que está entre aspas (valores e nomes), para não confundir uma palavra
# dentro de um valor, como 'Delete', com um comando.
def _sem_textos(sql: str) -> str:
    return re.sub(r"'[^']*'|\"[^\"]*\"", " ", sql)


# Só aceita um único SELECT (ou WITH ... SELECT). Levanta ValueError com o motivo
# se o SQL não for permitido.
def validar_sql(sql: str) -> str:
    limpo = sql.strip().rstrip(";").strip()
    aberto = _sem_textos(limpo)
    if ";" in aberto:
        raise ValueError("a consulta tem mais de um comando")
    if not re.match(r"(?is)^(select|with)\b", limpo):
        raise ValueError("só é permitido um comando SELECT")
    for palavra in re.findall(r"[a-z_]+", aberto.lower()):
        if palavra in PALAVRAS_PROIBIDAS:
            raise ValueError(f"palavra não permitida: {palavra}")
    return limpo


# Executa em uma conexão somente leitura, com limite de tempo, e devolve os nomes
# das colunas e as linhas.
def executar_sql(sql: str, limite_linhas: int | None = LIMITE_LINHAS) -> tuple[list[str], list[tuple]]:
    conn = sqlite3.connect(f"file:{CAMINHO_DB}?mode=ro", uri=True)
    try:
        prazo = time.time() + TEMPO_MAXIMO_SEGUNDOS
        conn.set_progress_handler(lambda: 1 if time.time() > prazo else 0, 100000)
        cursor = conn.execute(sql)
        colunas = [d[0] for d in cursor.description]
        linhas = cursor.fetchall() if limite_linhas is None else cursor.fetchmany(limite_linhas)
        return colunas, linhas
    finally:
        conn.close()


# Gera o SQL da pergunta e o executa, devolvendo o problema ao LLM para corrigir
# a consulta até o limite de tentativas.
def gerar_sql(pergunta: str, k: int = K_TABELAS, tabelas: list[str] | None = None) -> dict:
    entradas = selecionar_contexto(pergunta, k, tabelas)
    resultado = {
        "pergunta": pergunta,
        "tabelas_contexto": [e["tabela"] for e in entradas],
        "sql": None, "colunas": [], "linhas": [], "status": "erro", "tentativas": [],
    }
    mensagens = [
        {"role": "system", "content": montar_instrucoes(montar_contexto(entradas))},
        {"role": "user", "content": pergunta},
    ]

    for numero in range(1, MAX_TENTATIVAS + 1):
        resposta = ollama.chat(
            model=MODELO_LLM,
            messages=mensagens,
            format="json",
            options={"temperature": 0 if numero == 1 else 0.3, "num_ctx": NUM_CTX},
        )
        conteudo = resposta["message"]["content"]
        tentativa = {"sql": None, "problema": None}
        try:
            sql = json.loads(conteudo)["sql"]
            tentativa["sql"] = sql
            colunas, linhas = executar_sql(validar_sql(sql))
            if linhas:
                resultado.update(sql=sql, colunas=colunas, linhas=linhas, status="ok")
                resultado["tentativas"].append(tentativa)
                return resultado
            tentativa["problema"] = "a consulta não retornou nenhuma linha"
            resultado.update(sql=sql, colunas=colunas, linhas=[], status="vazio")
        except (json.JSONDecodeError, KeyError, TypeError):
            tentativa["problema"] = "a resposta não era um JSON com o campo sql"
            resultado["status"] = "erro"
        except (ValueError, sqlite3.Error) as erro:
            tentativa["problema"] = str(erro)
            resultado.update(sql=tentativa["sql"], status="erro")
        resultado["tentativas"].append(tentativa)

        if tentativa["problema"] == "a consulta não retornou nenhuma linha":
            pedido = ("A consulta anterior executou, mas não retornou nenhuma linha. "
                      "Revise as condições e as junções e responda de novo só com o JSON.")
        else:
            pedido = (f"A consulta anterior falhou: {tentativa['problema']}. "
                      "Corrija e responda de novo só com o JSON.")
        mensagens.append({"role": "assistant", "content": conteudo})
        mensagens.append({"role": "user", "content": pedido})

    return resultado
