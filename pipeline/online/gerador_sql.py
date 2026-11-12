"""
Etapa: Online.

Função: caminho quantitativo, geração e execução do SQL sobre as DFPs.

Descrição geral do funcionamento:
- A busca configurável (busca_configuravel.py) escolhe as k tabelas mais parecidas com a
  pergunta (k_tabelas e o método vêm de configuracao.json), mais as tabelas que elas referenciam
  por chave estrangeira (companhias). Também é possível passar a lista de tabelas, sem usar a
  busca (usado para medir o gerador com tabelas ideais).
- Nas perguntas híbridas divididas pelo divisor, o LLM recebe só a parte de dados (parâmetro pergunta_sql):
  a parte conceitual o levava a responder a explicação com colunas a mais e a escolher a conta errada. As
  tabelas continuam sendo buscadas com a pergunta inteira.
- O contexto traz só essas tabelas (colunas, tipos, valores possíveis, plano de contas e
  junções), lido do esquema.json. O esquema completo não vai ao LLM.
- Cada conta do plano de contas aparece com a descrição oficial e o filtro CD_CONTA pronto.
- O LLM gera o SQL. O código só aceita um único SELECT e executa em modo somente leitura.
- Antes de executar, o código confere o SQL: recusa CD_CVM escrito direto (o LLM inventa esses
  códigos) e confere os nomes de companhia do LIKE contra a tabela companhias. Se o LIKE não casa
  com nenhum nome mas o termo, sem acento, pontuação e espaço, casa com um ou mais, a condição é
  trocada por DENOM_CIA IN (nomes exatos) e o SQL original fica em sql_original; se nada casa, o
  LLM recebe o problema com os nomes parecidos.
- Também antes de executar, o código confere se cada CD_CONTA existe no plano da tabela em que é
  usado (se não existe nela mas existe em outra tabela do contexto, o LLM recebe o aviso com a
  tabela certa) e recusa SQL que usa VL_CONTA sem ESCALA_MOEDA, pedindo a conversão para reais.
- Se der erro, o resultado vier vazio ou só com valores nulos (NULL, como o de SUM sobre nenhuma
  linha), o LLM recebe o problema e gera outro SQL, até o limite de tentativas.
- Opcionalmente, um dicionário de cache (parâmetro cache) guarda o SQL por (pergunta, tabelas do
  contexto) e o reaproveita em chamadas seguintes; sem ele, nada é guardado.

Dependências:
- pipeline/online/busca_configuravel.py e busca.py, os índices e o esquema.json em embeddings/,
  dados/dfp.db;
- servidor Ollama em execução, com o modelo qwen2.5-coder:7b baixado.

Parâmetros: nenhum argumento. É um módulo importado pelos demais passos do estágio online.

Resultado: fornece gerar_sql(), executar_sql(), validar_sql(), conferir_companhias(), conferir_contas() e
conferir_escala().
"""

import copy
import difflib
import json
import re
import sqlite3
import time
import unicodedata

import ollama

from busca import CAMINHO_ESQUEMA
from busca_configuravel import obter_busca

CAMINHO_DB = "dados/dfp.db"
MODELO_LLM = "qwen2.5-coder:7b"
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
PROBLEMA_VAZIO = "a consulta não retornou nenhuma linha"
PROBLEMA_NULO = "a consulta só retornou valores nulos (NULL)"
# CD_CVM comparado com um valor escrito no SQL, como CD_CVM = '1133' ou CD_CVM IN ('1133').
CD_CVM_LITERAL = re.compile(r"\bCD_CVM\s*(?:=|<>|!=)\s*'?\d+'?|\bCD_CVM\s+IN\s*\(\s*'?\d", re.IGNORECASE)
# Condição sobre o nome da companhia, com ou sem alias da tabela (companhias.DENOM_CIA LIKE '%X%').
LIKE_COMPANHIA = re.compile(r"((?:\w+\.)?DENOM_CIA)\s+LIKE\s+'((?:[^']|'')*)'", re.IGNORECASE)
# Tabela depois de FROM ou JOIN, com o alias opcional (FROM DRE_con d, JOIN DRE_con AS d).
TABELA_E_ALIAS = re.compile(
    r"\b(?:FROM|JOIN)\s+(\w+)(?:\s+(?:AS\s+)?(?!ON\b|WHERE\b|JOIN\b|LEFT\b|INNER\b|GROUP\b|ORDER\b|LIMIT\b|USING\b|UNION\b|CROSS\b)(\w+))?",
    re.IGNORECASE)
# Condição sobre o código da conta, com ou sem o nome ou alias da tabela: CD_CONTA = '3.01' ou CD_CONTA IN ('3.01', '3.02').
CONDICAO_CONTA = re.compile(r"(?:(\w+)\.)?CD_CONTA\s*(?:=\s*'([^']+)'|IN\s*\(([^)]*)\))", re.IGNORECASE)
# Acima disso o termo é genérico demais para trocar o LIKE pelos nomes exatos.
MAX_NOMES_NA_REESCRITA = 5
# Semelhança mínima (difflib) entre uma palavra do termo e uma palavra de nome, para sugerir o nome.
LIMIAR_NOME_PARECIDO = 0.8

_esquema = None
_nomes = None


# Lê o esquema uma vez só e o reaproveita nas consultas seguintes.
def _carregar():
    global _esquema
    if _esquema is None:
        with open(CAMINHO_ESQUEMA, encoding="utf-8") as f:
            _esquema = {e["tabela"]: e for e in json.load(f)}
    return _esquema


# Devolve as entradas do esquema.json das tabelas da busca configurável (k padrão: k_tabelas da
# configuração), ou das tabelas passadas.
def selecionar_contexto(pergunta: str, k: int | None = None, tabelas: list[str] | None = None, tipo: str | None = None) -> list[dict]:
    esquema = _carregar()
    if tabelas is None:
        tabelas = obter_busca().tabelas(pergunta, tipo, k)
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
        # traz a descrição oficial da conta e o filtro pronto.
        if e["contas"]:
            linhas.append("Plano de contas (filtre com CD_CONTA = código):")
            for c in e["contas"]:
                linhas.append(f"  - {c['descricao']} -> CD_CONTA = '{c['codigo']}'")
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


# Devolve o texto só com letras e números maiúsculos, sem acento, para comparar nomes que diferem em
# pontuação, espaço ou acento (por exemplo, "M. Dias Branco" e "M.DIAS BRANCO").
def _normalizar_nome(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().upper()
    return re.sub(r"[^A-Z0-9]", "", sem_acento)


# Escreve um texto como literal do SQL, dobrando as aspas simples.
def _literal_sql(texto: str) -> str:
    return "'" + texto.replace("'", "''") + "'"


# Lê uma vez só os nomes das companhias do banco, cada um com a forma normalizada e as palavras normalizadas.
def _carregar_nomes() -> list[tuple[str, str, list[str]]]:
    global _nomes
    if _nomes is None:
        conn = sqlite3.connect(f"file:{CAMINHO_DB}?mode=ro", uri=True)
        try:
            nomes = [linha[0] for linha in conn.execute("SELECT DENOM_CIA FROM companhias")]
        finally:
            conn.close()
        _nomes = [(n, _normalizar_nome(n), [p for p in map(_normalizar_nome, n.split()) if p]) for n in nomes]
    return _nomes


# Sugere até 5 nomes de companhia que têm uma palavra parecida com o termo inteiro ou com alguma palavra dele.
def _nomes_parecidos(padrao: str, termo: str) -> list[str]:
    candidatas = {termo} | {p for p in map(_normalizar_nome, re.split(r"\W+", padrao)) if len(p) >= 4}
    pontuados = []
    for nome, _, palavras in _carregar_nomes():
        melhor = max((difflib.SequenceMatcher(None, c, p).ratio()
                      for c in candidatas if len(c) >= 4 for p in palavras if len(p) >= 4), default=0)
        if melhor >= LIMIAR_NOME_PARECIDO:
            pontuados.append((melhor, nome))
    return [nome for _, nome in sorted(pontuados, key=lambda x: -x[0])[:5]]


# Confere o SQL antes de executar. Recusa CD_CVM escrito direto e confere os nomes do LIKE em
# DENOM_CIA: se o LIKE não casa com nenhuma companhia mas o termo normalizado casa, troca a condição
# por DENOM_CIA IN (nomes exatos). Devolve (sql, problema, reescritas): o problema é o texto que vai ao
# LLM (ou None) e as reescritas são os termos trocados, cada um com os nomes encontrados.
def conferir_companhias(sql: str) -> tuple[str, str | None, list[dict]]:
    if CD_CVM_LITERAL.search(sql):
        return sql, ("não escreva valores de CD_CVM diretamente, pois esses códigos não são confiáveis. "
                     "Para filtrar a companhia, use companhias.DENOM_CIA LIKE com uma parte do nome "
                     "que aparece na pergunta"), []
    reescritas, problemas = [], []
    conn = sqlite3.connect(f"file:{CAMINHO_DB}?mode=ro", uri=True)

    def conferir(achado):
        coluna, padrao = achado.group(1), achado.group(2).replace("''", "'")
        # O LIKE do SQLite não ignora pontuação nem acento: só confere se ele casa com algum nome.
        if conn.execute("SELECT COUNT(*) FROM companhias WHERE DENOM_CIA LIKE ?", (padrao,)).fetchone()[0]:
            return achado.group(0)
        termo = _normalizar_nome(padrao.replace("%", "").replace("_", ""))
        if not termo:
            return achado.group(0)
        exibido = padrao.strip("%")
        encontrados = [nome for nome, normal, _ in _carregar_nomes() if termo in normal]
        if 1 <= len(encontrados) <= MAX_NOMES_NA_REESCRITA:
            reescritas.append({"padrao": padrao, "nomes": encontrados})
            return f"{coluna} IN ({', '.join(_literal_sql(n) for n in encontrados)})"
        if encontrados:
            problemas.append(f"o nome '{exibido}' casa com {len(encontrados)} companhias (por exemplo "
                             f"{', '.join(encontrados[:3])}). Use uma parte mais específica do nome")
            return achado.group(0)
        parecidos = _nomes_parecidos(padrao, termo)
        dica = f" (nomes parecidos: {', '.join(parecidos)})" if parecidos else ""
        problemas.append(f"nenhuma companhia tem '{exibido}' no nome{dica}. "
                         "Use uma parte do nome como ela aparece no banco")
        return achado.group(0)

    try:
        novo = LIKE_COMPANHIA.sub(conferir, sql)
    finally:
        conn.close()
    return novo, ("; ".join(problemas) if problemas else None), reescritas


# Confere se cada CD_CONTA do SQL existe no plano da tabela em que é usado. O aviso só sai quando o código não
# está no plano dessa tabela e está no de outra tabela do contexto: o plano guarda só as contas fixas, então
# um código fora dele não é necessariamente errado (companhias têm contas próprias). Devolve o problema ou None.
def conferir_contas(sql: str, plano: dict[str, set[str]]) -> str | None:
    por_nome, usadas = {}, set()
    for tabela, alias in TABELA_E_ALIAS.findall(sql):
        if tabela in plano:
            usadas.add(tabela)
            por_nome[tabela] = tabela
            if alias:
                por_nome[alias] = tabela
    problemas = []
    for qualificador, unico, lista in CONDICAO_CONTA.findall(sql):
        # Sem qualificador, só dá para saber a tabela se o SQL usa uma única tabela com plano de contas.
        tabela = por_nome.get(qualificador) if qualificador else (next(iter(usadas)) if len(usadas) == 1 else None)
        if not tabela or not plano[tabela]:
            continue
        codigos = [unico] if unico else [c.strip().strip("'") for c in lista.split(",")]
        for codigo in codigos:
            outras = [t for t, contas in plano.items() if t != tabela and codigo in contas]
            if codigo and codigo not in plano[tabela] and outras:
                problemas.append(f"a conta '{codigo}' não existe em {tabela}, mas existe em {', '.join(outras)}. "
                                 f"Use uma conta que exista em {tabela} ou faça a consulta na tabela que tem essa conta")
    return "; ".join(dict.fromkeys(problemas)) if problemas else None


# Recusa o SQL que usa VL_CONTA sem ESCALA_MOEDA: as companhias reportam em milhares ou em reais, e a regra
# do prompt é devolver sempre em reais. Devolve o problema ou None.
def conferir_escala(sql: str) -> str | None:
    if re.search(r"\bVL_CONTA\b", sql, re.IGNORECASE) and not re.search(r"\bESCALA_MOEDA\b", sql, re.IGNORECASE):
        return "a consulta usa VL_CONTA sem converter a escala. Converta o valor para reais usando ESCALA_MOEDA, como nas regras"
    return None


# Gera o SQL da pergunta e o executa. As tabelas do contexto são sempre buscadas com a pergunta; o texto que o
# LLM recebe para escrever o SQL é pergunta_sql, quando dado (a parte de dados de uma pergunta híbrida dividida),
# ou a própria pergunta. Com um dicionário em cache, o SQL de um par (texto, lista de tabelas do contexto) já
# gerado nele é reaproveitado, sem nova chamada ao LLM; o executor de experimentos usa isso para comparar
# configurações que recuperam as mesmas tabelas.
def gerar_sql(pergunta: str, k: int | None = None, tabelas: list[str] | None = None, tipo: str | None = None,
              cache: dict | None = None, pergunta_sql: str | None = None) -> dict:
    entradas = selecionar_contexto(pergunta, k, tabelas, tipo)
    texto = pergunta_sql or pergunta
    chave = (texto, tuple(e["tabela"] for e in entradas))
    if cache is not None and chave in cache:
        return copy.deepcopy(cache[chave])
    resultado = _gerar_e_executar(texto, entradas)
    if cache is not None:
        cache[chave] = copy.deepcopy(resultado)
    return resultado


# Pede o SQL ao LLM e o executa, devolvendo o problema ao LLM para corrigir a consulta até o limite
# de tentativas.
def _gerar_e_executar(pergunta: str, entradas: list[dict]) -> dict:
    resultado = {
        "pergunta": pergunta,
        "tabelas_contexto": [e["tabela"] for e in entradas],
        "sql": None, "sql_original": None, "colunas": [], "linhas": [], "status": "erro", "tentativas": [],
    }
    plano = {e["tabela"]: {c["codigo"] for c in e["contas"]} for e in entradas}
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
            sql_executado, problema, reescritas = conferir_companhias(sql)
            problemas = [p for p in (problema, conferir_contas(sql_executado, plano), conferir_escala(sql_executado)) if p]
            if problemas:
                raise ValueError("; ".join(problemas))
            if reescritas:
                tentativa["reescrita"] = reescritas
                resultado["sql_original"] = sql
            colunas, linhas = executar_sql(validar_sql(sql_executado))
            # SUM sobre nenhuma linha devolve uma linha com NULL: também conta como resultado vazio.
            so_nulos = bool(linhas) and all(valor is None for linha in linhas for valor in linha)
            if linhas and not so_nulos:
                resultado.update(sql=sql_executado, colunas=colunas, linhas=linhas, status="ok")
                resultado["tentativas"].append(tentativa)
                return resultado
            tentativa["problema"] = PROBLEMA_NULO if so_nulos else PROBLEMA_VAZIO
            resultado.update(sql=sql_executado, colunas=colunas, linhas=[], status="vazio")
        except (json.JSONDecodeError, KeyError, TypeError):
            tentativa["problema"] = "a resposta não era um JSON com o campo sql"
            resultado["status"] = "erro"
        except (ValueError, sqlite3.Error) as erro:
            tentativa["problema"] = str(erro)
            resultado.update(sql=tentativa["sql"], status="erro")
        resultado["tentativas"].append(tentativa)

        if tentativa["problema"] == PROBLEMA_VAZIO:
            pedido = ("A consulta anterior executou, mas não retornou nenhuma linha. "
                      "Revise as condições e as junções e responda de novo só com o JSON.")
        elif tentativa["problema"] == PROBLEMA_NULO:
            pedido = ("A consulta anterior executou, mas só retornou valores nulos (NULL), o que indica que "
                      "nenhuma linha atendeu às condições. Revise as condições, os nomes e as junções e "
                      "responda de novo só com o JSON.")
        else:
            pedido = (f"A consulta anterior falhou: {tentativa['problema']}. "
                      "Corrija e responda de novo só com o JSON.")
        mensagens.append({"role": "assistant", "content": conteudo})
        mensagens.append({"role": "user", "content": pedido})

    return resultado
