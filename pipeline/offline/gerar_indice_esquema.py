"""
Etapa: Offline.

Função: constrói o índice de esquema usado pelo estágio online.

Descrição geral do funcionamento:
- Documenta o esquema do banco dfp.db e gera um índice vetorial com um
  registro por aspecto de cada tabela (resumo e propósito, entidades,
  relações, colunas e contas).
- A estrutura (colunas, tipos, chaves e relações) vem direto do SQLite.
- O plano de contas e os valores possíveis de colunas de poucos valores
  distintos também vêm direto do banco, sem LLM.
- O qwen2.5-coder:7b escreve só os textos em português (resumo, propósito,
  entidades e descrição das colunas).
- O paraphrase-multilingual-MiniLM-L12-v2 gera os embeddings.

Dependências:
- dados/dfp.db (gerado por pipeline/offline/carregar_dfp.py);
- servidor Ollama em execução com o modelo qwen2.5-coder:7b baixado;
- bibliotecas ollama e sentence-transformers;
- modelo MiniLM já em cache local (carregado com local_files_only).

Parâmetros: argumentos opcionais de linha de comando com os nomes das tabelas
a processar. Sem argumentos, processa todas.

Resultado: gera embeddings/esquema.json, com uma entrada por tabela, e
embeddings/index_esquema.json, com os registros vetoriais no formato
id/fonte/conteudo/embedding, com a fonte igual ao nome da tabela. No modo de
teste, os dois arquivos ganham o sufixo _teste.

Executar de dentro de rag-nl2sql-financas-br/:
    python pipeline/offline/gerar_indice_esquema.py         (todas as tabelas)
    python pipeline/offline/gerar_indice_esquema.py DRE_con (teste: só as tabelas indicadas)
"""

import json
import sqlite3
import sys

import ollama
from sentence_transformers import SentenceTransformer

CAMINHO_DB = "dados/dfp.db"
CAMINHO_ESQUEMA = "embeddings/esquema.json"
CAMINHO_INDICE = "embeddings/index_esquema.json"
MODELO_LLM = "qwen2.5-coder:7b"
MODELO_EMBEDDING = "paraphrase-multilingual-MiniLM-L12-v2"

# O MiniLM lê no máximo 128 tokens por texto, e 2 são reservados pelo modelo.
# Usamos 120 para ter uma folga.
LIMITE_TOKENS = 120

TENTATIVAS_LLM = 3

# Colunas com até esse número de valores distintos têm os valores listados na
# descrição (o LLM não tem como adivinhar códigos como "MIL"/"UNIDADE").
LIMITE_CARDINALIDADE_VALORES = 5

# Nome por extenso de cada demonstração, para o LLM não depender só da sigla.
NOMES_DEMONSTRACOES = {
    "BPA": "Balanço Patrimonial Ativo",
    "BPP": "Balanço Patrimonial Passivo",
    "DRE": "Demonstração de Resultado (Demonstração do Resultado do Exercício)",
    "DFC_MI": "Demonstração de Fluxo de Caixa, método indireto",
    "DVA": "Demonstração de Valor Adicionado",
    "DMPL": "Demonstração das Mutações do Patrimônio Líquido",
}


# ---------------------------------------------------------------------------
# 1) Estrutura do banco (fatos, sem LLM)
# ---------------------------------------------------------------------------

# Lê do SQLite o DDL, as colunas e as chaves estrangeiras de cada tabela.
def ler_estrutura_banco(conn: sqlite3.Connection) -> dict:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name, sql FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    estrutura = {}
    for nome, ddl in cursor.fetchall():
        colunas = [
            {"nome": c[1], "tipo": c[2], "chave_primaria": c[5] > 0}
            for c in conn.execute(f'PRAGMA table_info("{nome}")')
        ]
        estrutura[nome] = {"ddl": ddl, "colunas": colunas, "fks": []}

    # As chaves estrangeiras são lidas depois, porque, quando a chave não diz
    # a coluna de destino, ela aponta para a chave primária da outra tabela.
    for nome in estrutura:
        for f in conn.execute(f'PRAGMA foreign_key_list("{nome}")').fetchall():
            tabela_destino, coluna_origem, coluna_destino = f[2], f[3], f[4]
            if coluna_destino is None:
                pks = [c["nome"] for c in estrutura[tabela_destino]["colunas"] if c["chave_primaria"]]
                coluna_destino = pks[0] if pks else None
            estrutura[nome]["fks"].append((coluna_origem, tabela_destino, coluna_destino))
    return estrutura


# Extrai do banco os códigos e as descrições das contas fixas de uma demonstração.
def ler_plano_contas(conn: sqlite3.Connection, tabela: str, colunas: list[dict]) -> list[dict]:
    # Nas DFPs, o significado está nos valores de CD_CONTA/DS_CONTA, não nos
    # nomes de coluna. A tabela companhias não tem essas colunas.
    nomes_colunas = {c["nome"] for c in colunas}
    if "CD_CONTA" not in nomes_colunas or "DS_CONTA" not in nomes_colunas:
        return []
    # Só contas fixas (ST_CONTA_FIXA='S'), padronizadas pela CVM.
    tem_st_fixa = "ST_CONTA_FIXA" in nomes_colunas
    filtro = "WHERE ST_CONTA_FIXA = 'S'" if tem_st_fixa else ""
    linhas = conn.execute(f"""
        SELECT CD_CONTA, DS_CONTA, COUNT(*) freq FROM "{tabela}" {filtro}
        GROUP BY CD_CONTA, DS_CONTA
        ORDER BY CD_CONTA
    """).fetchall()
    # A mesma conta pode ter descrições diferentes por setor (ex.: "3.01" é
    # "Receita de Venda de Bens e/ou Serviços" na maioria das empresas, mas
    # "Receitas da Intermediação Financeira" nos bancos). Fica a mais frequente,
    # então setores minoritários (bancos, seguradoras) não são cobertos.
    melhor_por_conta: dict[str, tuple[str, int]] = {}
    for codigo, descricao, freq in linhas:
        atual = melhor_por_conta.get(codigo)
        if atual is None or freq > atual[1]:
            melhor_por_conta[codigo] = (descricao, freq)
    return [
        {"codigo": codigo, "descricao": descricao}
        for codigo, (descricao, _freq) in sorted(melhor_por_conta.items())
    ]


# Lista os valores reais de colunas com poucos valores distintos (ex.: ESCALA_MOEDA).
def ler_valores_baixa_cardinalidade(conn: sqlite3.Connection, tabela: str, colunas: list[dict]) -> dict[str, list[str]]:
    valores: dict[str, list[str]] = {}
    for c in colunas:
        # Colunas de data ficam de fora: têm poucos valores só pelo recorte
        # atual, e a lista ficaria desatualizada a cada nova carga.
        if c["chave_primaria"] or c["nome"].startswith("DT_") or c["nome"] in ("CD_CONTA", "DS_CONTA", "VL_CONTA"):
            continue
        distintos = conn.execute(
            f'SELECT DISTINCT "{c["nome"]}" FROM "{tabela}" LIMIT {LIMITE_CARDINALIDADE_VALORES + 1}'
        ).fetchall()
        if 1 < len(distintos) <= LIMITE_CARDINALIDADE_VALORES:
            valores[c["nome"]] = sorted(str(v[0]) for v in distintos)
    return valores


# Devolve o nome por extenso da demonstração e se é consolidada ou individual, ou None para outras tabelas.
def nome_demonstracao(tabela: str) -> str | None:
    for prefixo, nome_extenso in NOMES_DEMONSTRACOES.items():
        if tabela == f"{prefixo}_con":
            return f"{nome_extenso}, versão consolidada (todas as empresas do grupo)"
        if tabela == f"{prefixo}_ind":
            return f"{nome_extenso}, versão individual (só a controladora)"
    return None


# Lista as ligações da tabela com as outras, nos dois sentidos.
def calcular_relacoes(estrutura: dict, tabela: str) -> list[dict]:
    # O DDL de uma tabela só mostra as chaves que ela mesma possui
    # ("referencia"); as que apontam para ela ("referenciada_por") são
    # calculadas a partir das demais tabelas.
    relacoes = []
    for coluna, destino, coluna_destino in estrutura[tabela]["fks"]:
        relacoes.append({
            "coluna": coluna,
            "tabela_relacionada": destino,
            "coluna_relacionada": coluna_destino,
            "sentido": "referencia",
        })
    for outra, dados in estrutura.items():
        if outra == tabela:
            continue
        for coluna, destino, coluna_destino in dados["fks"]:
            if destino == tabela:
                relacoes.append({
                    "coluna": coluna_destino,
                    "tabela_relacionada": outra,
                    "coluna_relacionada": coluna,
                    "sentido": "referenciada_por",
                })
    return relacoes


# ---------------------------------------------------------------------------
# 2) Textos em português (LLM) e validação
# ---------------------------------------------------------------------------

# Monta o prompt que pede ao LLM a descrição em português de uma tabela.
def montar_prompt(tabela: str, ddl: str, relacoes: list[dict], problemas_anteriores: list[str]) -> str:
    # As relações vão calculadas, para o LLM não precisar adivinhá-las.
    linhas = []
    for r in relacoes:
        if r["sentido"] == "referencia":
            linhas.append(f"- {tabela}.{r['coluna']} referencia {r['tabela_relacionada']}.{r['coluna_relacionada']}")
        else:
            linhas.append(f"- {r['tabela_relacionada']}.{r['coluna_relacionada']} referencia {tabela}.{r['coluna']}")
    texto_relacoes = "\n".join(linhas) or "- (nenhuma)"

    nome_extenso = nome_demonstracao(tabela)
    contexto_dominio = (
        f'A tabela "{tabela}" contém dados da demonstração financeira "{nome_extenso}", '
        "publicada pela CVM (Comissão de Valores Mobiliários) no Brasil, dentro do "
        "Formulário de Demonstrações Financeiras Padronizadas (DFP).\n\n"
        if nome_extenso else ""
    )

    prompt = f"""Você documenta bancos de dados. Descreva a tabela "{tabela}" em português.

{contexto_dominio}DDL da tabela:
{ddl}

Relações da tabela (calculadas a partir do banco, estão corretas):
{texto_relacoes}

Responda somente com um JSON com estes campos:
- "resumo": 1 ou 2 frases curtas dizendo o que a tabela representa
- "proposito": 1 frase curta dizendo para que a tabela é usada
- "entidades": lista de 3 a 6 conceitos de negócio ligados à tabela, em português
- "colunas": objeto em que cada chave é o nome de uma coluna e o valor é uma frase curta em português descrevendo a coluna

Use exatamente os nomes de coluna do DDL. Não invente colunas nem tabelas."""

    if problemas_anteriores:
        prompt += "\n\nA resposta anterior teve estes problemas. Corrija: " + "; ".join(problemas_anteriores)
    return prompt


# Devolve o valor sem espaços nas pontas se for texto, ou string vazia caso contrário.
def _texto(valor) -> str:
    return valor.strip() if isinstance(valor, str) else ""


# Coloca a resposta do LLM sempre no mesmo formato e lista os problemas encontrados.
def normalizar_descricao(bruto: dict, nomes_colunas: list[str]) -> tuple[dict, list[str]]:
    # O LLM às vezes devolve listas de objetos no lugar de textos, ou esquece colunas.
    problemas = []

    resumo = _texto(bruto.get("resumo"))
    if not resumo:
        problemas.append("resumo vazio")
    proposito = _texto(bruto.get("proposito"))
    if not proposito:
        problemas.append("propósito vazio")

    entidades = []
    for item in bruto.get("entidades") or []:
        if isinstance(item, dict):
            item = item.get("nome") or item.get("entidade") or ""
        if _texto(item):
            entidades.append(_texto(item))
    if not entidades:
        problemas.append("lista de entidades vazia")

    descricoes = bruto.get("colunas") or {}
    # Formato alternativo: [{"nome": ..., "descricao": ...}]
    if isinstance(descricoes, list):
        descricoes = {
            _texto(i.get("nome") or i.get("coluna")): i.get("descricao")
            for i in descricoes if isinstance(i, dict)
        }
    colunas = {}
    for nome in nomes_colunas:
        colunas[nome] = _texto(descricoes.get(nome))
        if not colunas[nome]:
            problemas.append(f"sem descrição para a coluna {nome}")
    inventadas = [n for n in descricoes if n not in nomes_colunas]
    if inventadas:
        problemas.append(f"colunas que não existem: {inventadas}")

    return {"resumo": resumo, "proposito": proposito, "entidades": entidades, "colunas": colunas}, problemas


# Pede ao LLM a descrição da tabela e devolve a melhor tentativa com os problemas que sobraram.
def gerar_descricao(tabela: str, dados: dict, relacoes: list[dict]) -> tuple[dict, list[str]]:
    nomes_colunas = [c["nome"] for c in dados["colunas"]]
    melhor, menos_problemas = None, None
    problemas = []
    for tentativa in range(1, TENTATIVAS_LLM + 1):
        prompt = montar_prompt(tabela, dados["ddl"], relacoes, problemas)
        resposta = ollama.chat(
            model=MODELO_LLM,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            # Na primeira tentativa a resposta é a mais previsível; nas
            # seguintes há um pouco de variação, para não repetir o mesmo erro.
            options={"temperature": 0 if tentativa == 1 else 0.4},
        )
        try:
            bruto = json.loads(resposta["message"]["content"])
        except json.JSONDecodeError:
            problemas = ["a resposta não era um JSON válido"]
            continue
        descricao, problemas = normalizar_descricao(bruto, nomes_colunas)
        if not problemas:
            return descricao, []
        if menos_problemas is None or len(problemas) < len(menos_problemas):
            melhor, menos_problemas = descricao, problemas
        print(f"    tentativa {tentativa}: {len(problemas)} problema(s)")
    # Nenhuma tentativa devolveu JSON válido.
    if melhor is None:
        melhor = {"resumo": "", "proposito": "", "entidades": [], "colunas": {n: "" for n in nomes_colunas}}
        menos_problemas = problemas
    return melhor, menos_problemas


# ---------------------------------------------------------------------------
# 3) Documento do esquema (esquema.json)
# ---------------------------------------------------------------------------

# Junta os fatos do banco com os textos do LLM em uma entrada por tabela.
def montar_entrada_esquema(
    tabela: str, dados: dict, relacoes: list[dict], descricao: dict,
    contas: list[dict], valores: dict[str, list[str]],
) -> dict:
    return {
        "tabela": tabela,
        "resumo": descricao["resumo"],
        "proposito": descricao["proposito"],
        "entidades": descricao["entidades"],
        "colunas": [
            {
                "nome": c["nome"],
                "tipo": c["tipo"],
                "chave_primaria": c["chave_primaria"],
                "descricao": descricao["colunas"][c["nome"]],
                "valores_possiveis": valores.get(c["nome"], []),
            }
            for c in dados["colunas"]
        ],
        "chaves_de_juncao": relacoes,
        "tabelas_conectadas": sorted({r["tabela_relacionada"] for r in relacoes}),
        "contas": contas,
    }


# Confere o esquema contra o banco e devolve os pontos de atenção.
def validar_esquema(esquema: list[dict], estrutura: dict) -> list[str]:
    problemas = []
    for entrada in esquema:
        tabela = entrada["tabela"]
        reais = [c["nome"] for c in estrutura[tabela]["colunas"]]
        if [c["nome"] for c in entrada["colunas"]] != reais:
            problemas.append(f"{tabela}: colunas diferentes do DDL")
        for r in entrada["chaves_de_juncao"]:
            outra = r["tabela_relacionada"]
            if r["coluna"] not in reais:
                problemas.append(f"{tabela}: coluna {r['coluna']} não existe")
            if outra not in estrutura:
                problemas.append(f"{tabela}: tabela {outra} não existe")
            elif r["coluna_relacionada"] not in [c["nome"] for c in estrutura[outra]["colunas"]]:
                problemas.append(f"{tabela}: coluna {outra}.{r['coluna_relacionada']} não existe")
        if not entrada["resumo"] or not entrada["proposito"] or not entrada["entidades"]:
            problemas.append(f"{tabela}: resumo, propósito ou entidades vazios")
        vazias = [c["nome"] for c in entrada["colunas"] if not c["descricao"]]
        if vazias:
            problemas.append(f"{tabela}: colunas sem descrição: {vazias}")
        if "CD_CONTA" in reais and not entrada["contas"]:
            problemas.append(f"{tabela}: tem CD_CONTA mas nenhuma conta foi extraída")
    return problemas


# ---------------------------------------------------------------------------
# 4) Índice vetorial (registros de até 120 tokens)
# ---------------------------------------------------------------------------

# Junta os itens em textos com o mesmo prefixo que cabem no limite de tokens.
def empacotar(prefixo: str, itens: list[str], separador: str, contar) -> list[str]:
    textos, atual = [], []
    for item in itens:
        candidato = prefixo + separador.join(atual + [item])
        if atual and contar(candidato) > LIMITE_TOKENS:
            textos.append(prefixo + separador.join(atual))
            atual = [item]
        else:
            atual.append(item)
    if atual:
        textos.append(prefixo + separador.join(atual))
    return textos


# Tira palavras do fim do texto até ele caber no limite de tokens.
def cortar_no_limite(texto: str, contar) -> str:
    palavras = texto.split(" ")
    while contar(" ".join(palavras)) > LIMITE_TOKENS and len(palavras) > 1:
        palavras.pop()
    return " ".join(palavras)


# Gera os textos do índice de uma tabela, um para cada aspecto da descrição.
def textos_do_indice(entrada: dict, contar) -> list[str]:
    t = entrada["tabela"]
    textos = []

    itens = [x for x in (entrada["resumo"], entrada["proposito"]) if x]
    textos += empacotar(f"Tabela {t}. ", itens, " ", contar) if itens else []

    if entrada["entidades"]:
        textos += empacotar(f"Tabela {t}. Entidades: ", entrada["entidades"], ", ", contar)

    frases = []
    for r in entrada["chaves_de_juncao"]:
        if r["sentido"] == "referencia":
            frases.append(f"{t}.{r['coluna']} referencia {r['tabela_relacionada']}.{r['coluna_relacionada']}")
        else:
            frases.append(f"{r['tabela_relacionada']}.{r['coluna_relacionada']} referencia {t}.{r['coluna']}")
    if frases:
        textos += empacotar(f"Tabela {t}. Relações: ", frases, "; ", contar)

    itens = []
    for c in entrada["colunas"]:
        item = f"{c['nome']} ({c['descricao']})" if c["descricao"] else c["nome"]
        if c["valores_possiveis"]:
            item += f" [valores: {', '.join(c['valores_possiveis'])}]"
        itens.append(item)
    textos += empacotar(f"Tabela {t}. Colunas: ", itens, "; ", contar)

    if entrada["contas"]:
        itens = [f"{c['codigo']} {c['descricao']}" for c in entrada["contas"]]
        textos += empacotar(f"Tabela {t}. Contas: ", itens, "; ", contar)

    return [cortar_no_limite(texto, contar) for texto in textos]


# Gera os embeddings dos textos de uma tabela e monta os registros do índice.
def montar_registros(entrada: dict, modelo_embedding: SentenceTransformer) -> list[dict]:
    tokenizador = modelo_embedding.tokenizer
    contar = lambda texto: len(tokenizador.tokenize(texto))
    textos = textos_do_indice(entrada, contar)
    vetores = modelo_embedding.encode(textos)
    base = entrada["tabela"].lower().replace(" ", "_")
    return [
        {
            "id": f"esquema_{base}_{numero:03d}",
            "fonte": entrada["tabela"],
            "conteudo": texto,
            "embedding": vetor.tolist(),
        }
        for numero, (texto, vetor) in enumerate(zip(textos, vetores), start=1)
    ]


# ---------------------------------------------------------------------------

# Gera o esquema e o índice para todas as tabelas, ou só para as pedidas na linha de comando.
def main():
    tabelas_pedidas = sys.argv[1:]

    conn = sqlite3.connect(CAMINHO_DB)
    estrutura = ler_estrutura_banco(conn)

    desconhecidas = [t for t in tabelas_pedidas if t not in estrutura]
    if desconhecidas:
        raise ValueError(f"Tabelas não encontradas: {desconhecidas}")
    tabelas = tabelas_pedidas or list(estrutura)

    # No modo de teste os arquivos ganham o sufixo _teste, para não apagar os completos.
    sufixo = "_teste" if tabelas_pedidas else ""
    caminho_esquema = CAMINHO_ESQUEMA.replace(".json", f"{sufixo}.json")
    caminho_indice = CAMINHO_INDICE.replace(".json", f"{sufixo}.json")

    # local_files_only evita depender da rede a cada carga do modelo.
    modelo_embedding = SentenceTransformer(MODELO_EMBEDDING, local_files_only=True)
    esquema, registros = [], []

    for tabela in tabelas:
        print(f"Tabela {tabela}")
        relacoes = calcular_relacoes(estrutura, tabela)
        colunas = estrutura[tabela]["colunas"]
        contas = ler_plano_contas(conn, tabela, colunas)
        valores = ler_valores_baixa_cardinalidade(conn, tabela, colunas)
        descricao, problemas = gerar_descricao(tabela, estrutura[tabela], relacoes)
        if problemas:
            print(f"    ATENÇÃO, problemas que ficaram: {problemas}")
        entrada = montar_entrada_esquema(tabela, estrutura[tabela], relacoes, descricao, contas, valores)
        esquema.append(entrada)
        novos = montar_registros(entrada, modelo_embedding)
        registros.extend(novos)
        print(f"    {len(relacoes)} relações, {len(contas)} contas, {len(novos)} registros no índice")

    conn.close()

    problemas = validar_esquema(esquema, estrutura)
    if problemas:
        print("\nValidação do esquema.json, pontos de atenção:")
        for p in problemas:
            print(f"  - {p}")
    else:
        print("\nValidação do esquema.json: sem problemas")

    with open(caminho_esquema, "w", encoding="utf-8") as f:
        json.dump(esquema, f, ensure_ascii=False, indent=2)
    with open(caminho_indice, "w", encoding="utf-8") as f:
        json.dump(registros, f, ensure_ascii=False, indent=2)

    print(f"Esquema salvo em {caminho_esquema} ({len(esquema)} tabelas)")
    print(f"Índice salvo em {caminho_indice} ({len(registros)} registros)")


if __name__ == "__main__":
    main()
