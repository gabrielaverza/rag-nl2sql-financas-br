"""
Etapa: Offline.

Função: carrega os dados DFP/CVM de 2025 em um banco SQLite.

Descrição geral do funcionamento:
- Usa só DT_REFER = 2025-12-31 (exercício social padrão dezembro).
- Mantém só a VERSAO mais alta por CNPJ_CIA (descarta reapresentações antigas).
- Carrega 6 demonstrações, consolidadas e individuais (12 tabelas): BPA, BPP,
  DRE, DFC_MI, DVA e DMPL. DRA e DFC_MD ficam de fora.

Dependências:
- apenas a biblioteca padrão do Python;
- CSVs da CVM em dados/dfp_cia_aberta_2025/.

Parâmetros: nenhum argumento. O recorte é definido pelas constantes do arquivo.

Resultado: gera dados/dfp.db, com a tabela companhias e as 12 tabelas de
demonstração, ligadas a companhias por CD_CVM. O arquivo anterior é apagado.

Executar de dentro de rag-nl2sql-financas-br/:
    python pipeline/carregar_dfp.py
"""

import csv
import os
import sqlite3

CAMINHO_DADOS_CVM = "dados/dfp_cia_aberta_2025"
CAMINHO_DB = "dados/dfp.db"
DT_REFER_ALVO = "2025-12-31"

# (sufixo do nome do arquivo/tabela, tem DT_INI_EXERC, tem COLUNA_DF)
DEMONSTRACOES = [
    ("BPA", False, False),
    ("BPP", False, False),
    ("DRE", True, False),
    ("DFC_MI", True, False),
    ("DVA", True, False),
    ("DMPL", True, True),
]
CONSOLIDACOES = ["con", "ind"]


# Abre um CSV da CVM e devolve o leitor e o arquivo aberto.
def ler_csv(caminho: str) -> csv.DictReader:
    # Os CSVs da CVM são latin-1, separados por ';'.
    arquivo = open(caminho, encoding="latin-1", newline="")
    return csv.DictReader(arquivo, delimiter=";"), arquivo


# ---------------------------------------------------------------------------
# 1) Índice mestre
# ---------------------------------------------------------------------------

# Escolhe, por CNPJ, a linha de maior VERSAO do índice mestre.
def selecionar_companhias(caminho_master: str) -> dict[str, dict]:
    # Uma companhia pode ter até 3 linhas para o mesmo DT_REFER (reapresentações).
    leitor, arquivo = ler_csv(caminho_master)
    escolhidas: dict[str, dict] = {}
    try:
        for linha in leitor:
            if linha["DT_REFER"] != DT_REFER_ALVO:
                continue
            cnpj = linha["CNPJ_CIA"]
            versao = int(linha["VERSAO"])
            atual = escolhidas.get(cnpj)
            if atual is None or versao > atual["VERSAO"]:
                escolhidas[cnpj] = {
                    "CNPJ_CIA": cnpj,
                    "DENOM_CIA": linha["DENOM_CIA"],
                    "CD_CVM": linha["CD_CVM"],
                    "VERSAO": versao,
                }
    finally:
        arquivo.close()
    return escolhidas


# ---------------------------------------------------------------------------
# 2) Esquema do banco
# ---------------------------------------------------------------------------

# Monta o nome da tabela, por exemplo "BPA_con".
def nome_tabela(demonstracao: str, consolidacao: str) -> str:
    return f"{demonstracao}_{consolidacao}"


# Cria a tabela companhias e as 12 tabelas de demonstração.
def criar_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE companhias (
            CD_CVM TEXT PRIMARY KEY,
            CNPJ_CIA TEXT NOT NULL,
            DENOM_CIA TEXT NOT NULL
        )
    """)
    for demonstracao, tem_dt_ini, tem_coluna_df in DEMONSTRACOES:
        for consolidacao in CONSOLIDACOES:
            colunas = ["CD_CVM TEXT NOT NULL REFERENCES companhias(CD_CVM)"]
            if tem_dt_ini:
                colunas.append("DT_INI_EXERC TEXT NOT NULL")
            colunas.append("DT_FIM_EXERC TEXT NOT NULL")
            colunas.append("ORDEM_EXERC TEXT NOT NULL")
            colunas.append("MOEDA TEXT NOT NULL")
            colunas.append("ESCALA_MOEDA TEXT NOT NULL")
            if tem_coluna_df:
                colunas.append("COLUNA_DF TEXT NOT NULL")
            colunas.append("CD_CONTA TEXT NOT NULL")
            colunas.append("DS_CONTA TEXT NOT NULL")
            colunas.append("VL_CONTA REAL NOT NULL")
            colunas.append("ST_CONTA_FIXA TEXT NOT NULL")
            tabela = nome_tabela(demonstracao, consolidacao)
            conn.execute(f"CREATE TABLE {tabela} ({', '.join(colunas)})")


# Insere as companhias selecionadas na tabela companhias.
def carregar_companhias(conn: sqlite3.Connection, companhias: dict[str, dict]) -> None:
    conn.executemany(
        "INSERT INTO companhias (CD_CVM, CNPJ_CIA, DENOM_CIA) VALUES (?, ?, ?)",
        [(c["CD_CVM"], c["CNPJ_CIA"], c["DENOM_CIA"]) for c in companhias.values()],
    )


# ---------------------------------------------------------------------------
# 3) Carga de cada demonstração
# ---------------------------------------------------------------------------

# Carrega o CSV de uma demonstração na tabela correspondente e devolve quantas linhas inseriu.
def carregar_demonstracao(
    conn: sqlite3.Connection,
    caminho_csv: str,
    tabela: str,
    companhias: dict[str, dict],
    tem_dt_ini: bool,
    tem_coluna_df: bool,
) -> int:
    if not os.path.exists(caminho_csv):
        print(f"  aviso: arquivo nao encontrado, pulando: {caminho_csv}")
        return 0

    colunas = ["CD_CVM"]
    if tem_dt_ini:
        colunas.append("DT_INI_EXERC")
    colunas += ["DT_FIM_EXERC", "ORDEM_EXERC", "MOEDA", "ESCALA_MOEDA"]
    if tem_coluna_df:
        colunas.append("COLUNA_DF")
    colunas += ["CD_CONTA", "DS_CONTA", "VL_CONTA", "ST_CONTA_FIXA"]

    placeholders = ", ".join("?" for _ in colunas)
    sql_insert = f"INSERT INTO {tabela} ({', '.join(colunas)}) VALUES ({placeholders})"

    leitor, arquivo = ler_csv(caminho_csv)
    linhas_vistas = set()
    linhas_para_inserir = []
    linhas_duplicadas = 0
    try:
        for linha in leitor:
            if linha["DT_REFER"] != DT_REFER_ALVO:
                continue
            cnpj = linha["CNPJ_CIA"]
            companhia = companhias.get(cnpj)
            # Reapresentações antigas de uma demonstração ficam de fora.
            if companhia is None or int(linha["VERSAO"]) != companhia["VERSAO"]:
                continue
            # A CVM às vezes repete a mesma linha dentro do mesmo arquivo (em
            # 2025: FGR Incorporações e VLI Multimodal). Descarta só repetições
            # exatas: no DMPL, o mesmo CD_CONTA aparece legitimamente em
            # COLUNA_DF diferentes.
            chave_linha = tuple(linha.values())
            if chave_linha in linhas_vistas:
                linhas_duplicadas += 1
                continue
            linhas_vistas.add(chave_linha)
            valores = [companhia["CD_CVM"]]
            if tem_dt_ini:
                valores.append(linha["DT_INI_EXERC"])
            valores += [linha["DT_FIM_EXERC"], linha["ORDEM_EXERC"], linha["MOEDA"], linha["ESCALA_MOEDA"]]
            if tem_coluna_df:
                valores.append(linha["COLUNA_DF"])
            valores += [
                linha["CD_CONTA"],
                linha["DS_CONTA"],
                float(linha["VL_CONTA"]),
                linha["ST_CONTA_FIXA"],
            ]
            linhas_para_inserir.append(tuple(valores))
    finally:
        arquivo.close()

    conn.executemany(sql_insert, linhas_para_inserir)
    if linhas_duplicadas:
        print(f"    ({linhas_duplicadas} linha(s) duplicada(s) na fonte, descartadas)")
    return len(linhas_para_inserir)


# ---------------------------------------------------------------------------

# Recria o banco do zero e carrega as companhias e as 12 demonstrações.
def main():
    if os.path.exists(CAMINHO_DB):
        os.remove(CAMINHO_DB)
    os.makedirs(os.path.dirname(CAMINHO_DB), exist_ok=True)

    caminho_master = os.path.join(CAMINHO_DADOS_CVM, "dfp_cia_aberta_2025.csv")
    print(f"Lendo indice mestre: {caminho_master}")
    companhias = selecionar_companhias(caminho_master)
    print(f"  {len(companhias)} companhias com DT_REFER={DT_REFER_ALVO}")

    conn = sqlite3.connect(CAMINHO_DB)
    conn.execute("PRAGMA foreign_keys = ON")
    criar_schema(conn)
    carregar_companhias(conn, companhias)

    total_linhas = 0
    for demonstracao, tem_dt_ini, tem_coluna_df in DEMONSTRACOES:
        for consolidacao in CONSOLIDACOES:
            tabela = nome_tabela(demonstracao, consolidacao)
            nome_arquivo = f"dfp_cia_aberta_{tabela}_2025.csv"
            caminho_csv = os.path.join(CAMINHO_DADOS_CVM, nome_arquivo)
            n = carregar_demonstracao(conn, caminho_csv, tabela, companhias, tem_dt_ini, tem_coluna_df)
            print(f"  {tabela}: {n} linhas")
            total_linhas += n

    conn.commit()
    conn.close()
    print(f"\nBanco salvo em {CAMINHO_DB}: {len(companhias)} companhias, {total_linhas} linhas nas demonstracoes")


if __name__ == "__main__":
    main()
