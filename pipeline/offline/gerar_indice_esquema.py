"""
Etapa: Offline.

Função: documenta o esquema do banco e constrói o índice vetorial de esquema usado pelo estágio online.

Descrição geral do funcionamento:
- Documenta o esquema do banco dfp.db (embeddings/esquema.json) e, a partir dele, gera o corpus de
  esquema (com gerar_corpus.py) e um índice vetorial com um registro por aspecto de cada tabela
  (resumo e propósito, entidades, relações, colunas e contas).
- A estrutura (colunas, tipos, chaves e relações) vem direto do SQLite.
- O plano de contas e os valores possíveis de colunas de poucos valores
  distintos também vêm direto do banco, sem LLM.
- O qwen2.5-coder:7b escreve só os textos em português (resumo, propósito,
  entidades e descrição das colunas).
- O modelo de embeddings escolhido (opção --modelo, padrão e5-base) gera os vetores, com o
  prefixo de trecho da ficha dele. Os registros vêm do corpus (embeddings/corpus_esquema.json),
  que é o mesmo qualquer que seja o modelo.
- Antes de tudo, confere se os modelos necessários estão no cache local; só baixa os que
  faltam, e só com --baixar-modelos.

Dependências:
- dados/dfp.db (gerado por pipeline/offline/carregar_dfp.py), exceto com --so-indice;
- servidor Ollama em execução com o modelo qwen2.5-coder:7b baixado, exceto com --so-indice;
- com --so-indice, embeddings/corpus_esquema.json (gerado por pipeline/offline/gerar_corpus.py);
- bibliotecas ollama, transformers e sentence-transformers;
- pipeline/modelos_embedding.py e os modelos em cache local.

Parâmetros: nomes de tabelas (opcionais) para processar só elas; sem nomes, processa todas.
--so-indice reaproveita o corpus existente e só regera o índice vetorial, sem banco nem LLM. --modelo CHAVE escolhe o modelo de embeddings (padrão e5-base; as chaves estão em
pipeline/modelos_embedding.py); outro modelo exige --so-indice, porque o esquema.json é um
só, comum a todos. --baixar-modelos autoriza o download dos modelos que faltarem.

Resultado: gera embeddings/esquema.json, com uma entrada por tabela, o corpus de esquema e o
índice vetorial no formato id/fonte/conteudo/embedding, com a fonte igual ao nome da tabela. Com o modelo
padrão, o índice é embeddings/index_esquema.json (versionado); com outro modelo, é
embeddings/modelos/<chave>/index_esquema.json (pasta ignorada pelo git). No modo de teste
(tabelas indicadas), os três arquivos ganham o sufixo _teste.

Executar de dentro de rag-nl2sql-financas-br/:
    python pipeline/offline/gerar_indice_esquema.py               (todas as tabelas)
    python pipeline/offline/gerar_indice_esquema.py DRE_con       (teste: só as tabelas indicadas)
    python pipeline/offline/gerar_indice_esquema.py --so-indice   (só o índice, a partir do corpus)
    python pipeline/offline/gerar_indice_esquema.py --so-indice --modelo minilm
"""

import argparse
import json
import os
import sqlite3
import sys

import ollama

# Os módulos compartilhados ficam em pipeline/, fora da pasta deste script.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modelos_embedding  # noqa: E402

import gerar_corpus  # noqa: E402

CAMINHO_DB = "dados/dfp.db"
CAMINHO_ESQUEMA = gerar_corpus.CAMINHO_ESQUEMA
CAMINHO_CORPUS = gerar_corpus.CAMINHO_CORPUS_ESQUEMA
CAMINHO_INDICE_PADRAO = "embeddings/index_esquema.json"
PASTA_MODELOS = "embeddings/modelos"
MODELO_LLM = "qwen2.5-coder:7b"

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
# 4) Índice vetorial (a partir do corpus)
# ---------------------------------------------------------------------------

# Devolve onde o índice do modelo é gravado: o caminho versionado para o padrão, uma pasta ignorada para os outros.
def caminho_do_indice(chave_modelo: str) -> str:
    if chave_modelo == modelos_embedding.MODELO_PADRAO:
        return CAMINHO_INDICE_PADRAO
    return os.path.join(PASTA_MODELOS, chave_modelo, "index_esquema.json")


# Junta o vetor de cada registro do corpus de esquema e grava o índice.
def gravar_indice(corpus: list[dict], chave_modelo: str, caminho_saida: str) -> str:
    registros = modelos_embedding.registros_com_vetores(corpus, chave_modelo)
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    with open(caminho_saida, "w", encoding="utf-8") as f:
        json.dump(registros, f, ensure_ascii=False, indent=2)
    print(f"Índice salvo em {caminho_saida} ({len(registros)} registros, modelo {chave_modelo})")
    return caminho_saida


# Regera só o índice vetorial a partir do corpus existente, sem banco nem LLM.
def gerar_so_indice(chave_modelo: str, baixar_modelos: bool, caminho_saida: str | None = None) -> str:
    # O início de toda execução confere o cache; sem --baixar-modelos, só avisa o que falta.
    modelos_embedding.garantir_modelos([chave_modelo, modelos_embedding.MODELO_FRAGMENTACAO], baixar_modelos)
    with open(CAMINHO_CORPUS, encoding="utf-8") as f:
        corpus = json.load(f)
    return gravar_indice(corpus, chave_modelo, caminho_saida or caminho_do_indice(chave_modelo))


# Gera o esquema e o índice para todas as tabelas, ou só para as pedidas na linha de comando.
def main():
    parser = argparse.ArgumentParser(description="Gera o esquema.json e o índice de esquema.")
    parser.add_argument("tabelas", nargs="*", help="nomes de tabelas, para processar só elas (modo de teste)")
    parser.add_argument("--so-indice", action="store_true", help="regera só o índice, a partir do esquema.json")
    parser.add_argument("--modelo", default=modelos_embedding.MODELO_PADRAO, choices=list(modelos_embedding.MODELOS))
    parser.add_argument("--baixar-modelos", action="store_true", help="autoriza baixar os modelos que faltarem")
    args = parser.parse_args()

    if args.so_indice:
        gerar_so_indice(args.modelo, args.baixar_modelos)
        return
    if args.modelo != modelos_embedding.MODELO_PADRAO:
        parser.error("outro modelo exige --so-indice: o esquema.json é um só, comum a todos os modelos")
    tabelas_pedidas = args.tabelas
    # Antes de abrir o banco e chamar o LLM, confere os modelos de embeddings.
    modelos_embedding.garantir_modelos([args.modelo, modelos_embedding.MODELO_FRAGMENTACAO], args.baixar_modelos)

    conn = sqlite3.connect(CAMINHO_DB)
    estrutura = ler_estrutura_banco(conn)

    desconhecidas = [t for t in tabelas_pedidas if t not in estrutura]
    if desconhecidas:
        raise ValueError(f"Tabelas não encontradas: {desconhecidas}")
    tabelas = tabelas_pedidas or list(estrutura)

    # No modo de teste os arquivos ganham o sufixo _teste, para não apagar os completos.
    sufixo = "_teste" if tabelas_pedidas else ""
    caminho_esquema = CAMINHO_ESQUEMA.replace(".json", f"{sufixo}.json")
    caminho_corpus = CAMINHO_CORPUS.replace(".json", f"{sufixo}.json")
    caminho_indice = CAMINHO_INDICE_PADRAO.replace(".json", f"{sufixo}.json")

    tokenizador = modelos_embedding.carregar_tokenizador(modelos_embedding.MODELO_FRAGMENTACAO)
    esquema = []

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
        novos = gerar_corpus.gerar_corpus_esquema([entrada], tokenizador)
        print(f"    {len(relacoes)} relações, {len(contas)} contas, {len(novos)} registros no corpus")

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
    print(f"Esquema salvo em {caminho_esquema} ({len(esquema)} tabelas)")

    corpus = gerar_corpus.gerar_corpus_esquema(esquema, tokenizador)
    gerar_corpus.gravar_corpus(corpus, caminho_corpus)
    gravar_indice(corpus, args.modelo, caminho_indice)


if __name__ == "__main__":
    main()
