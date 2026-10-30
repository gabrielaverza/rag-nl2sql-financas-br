"""
Etapa: Online.

Função: formatação da resposta final com proveniência.

Descrição geral do funcionamento:
- A Metodologia prevê que a resposta seja entregue com proveniência: empresa consultada,
  demonstração financeira, exercício social e código CVM (kartheesan2026). Esses campos são
  obtidos do SQL gerado e do banco, sem LLM:
  - demonstração: as tabelas de demonstração citadas no SQL (ex.: DRE_con);
  - empresa e código CVM: as companhias do resultado (coluna DENOM_CIA) ou, se ele não as
    trouxer, as que o SQL filtra (por nome ou por CD_CVM), buscadas em companhias;
  - exercício social: DT_INI_EXERC e DT_FIM_EXERC das linhas da demonstração, para a
    ordem de exercício que o SQL usa (ÚLTIMO ou PENÚLTIMO).
- Também entram a consulta SQL, a tabela de dados do resultado e os documentos do glossário
  usados. A tabela completa vem do banco, e o LLM só a comenta (a resposta de texto pode ser
  um resumo).
- Se a fidelidade ficar abaixo do limiar, a resposta recebe um aviso.
- Se a resposta for uma recusa, nenhum documento é listado.

Dependências:
- pipeline/online/gerador_sql.py (caminho do banco) e sintetizador.py;
- dados/dfp.db.

Parâmetros: nenhum argumento. É um módulo importado pelos demais passos do estágio online.

Resultado: fornece formatar(pergunta, tipo, resposta, resultado_sql, validacao), que devolve o
texto final, e obter_proveniencia(resultado_sql).
"""

import re
import sqlite3

from gerador_sql import CAMINHO_DB
from sintetizador import SEM_INFORMACAO, formatar_valor, rotular_colunas

MAX_LINHAS_NA_TABELA = 20
NOMES_DEMONSTRACAO = {
    "BPA": "Balanço Patrimonial Ativo",
    "BPP": "Balanço Patrimonial Passivo",
    "DRE": "Demonstração do Resultado do Exercício",
    "DFC_MI": "Demonstração do Fluxo de Caixa (método indireto)",
    "DVA": "Demonstração do Valor Adicionado",
    "DMPL": "Demonstração das Mutações do Patrimônio Líquido",
}
PADRAO_DEMONSTRACAO = re.compile(r"\b(BPA|BPP|DRE|DFC_MI|DVA|DMPL)_(con|ind)\b")


def _consultar(sql: str, parametros: tuple = ()) -> list[tuple]:
    conn = sqlite3.connect(f"file:{CAMINHO_DB}?mode=ro", uri=True)
    try:
        return conn.execute(sql, parametros).fetchall()
    finally:
        conn.close()


# Demonstrações citadas no SQL, na ordem em que aparecem.
def _demonstracoes(sql: str) -> list[str]:
    return list(dict.fromkeys(m.group(0) for m in PADRAO_DEMONSTRACAO.finditer(sql)))


# Companhias da resposta: as do resultado (DENOM_CIA) ou, se ele não as trouxer, as que o SQL filtra.
def _empresas(resultado_sql: dict) -> list[tuple[str, str]]:
    colunas, linhas, sql = resultado_sql["colunas"], resultado_sql["linhas"], resultado_sql["sql"]
    if "DENOM_CIA" in colunas:
        nomes = list(dict.fromkeys(l[colunas.index("DENOM_CIA")] for l in linhas))
        achadas = [r for n in nomes for r in _consultar("SELECT DENOM_CIA, CD_CVM FROM companhias WHERE DENOM_CIA = ?", (n,))]
        return achadas
    achadas = []
    for operador, texto in re.findall(r"DENOM_CIA\s+(LIKE|=)\s+'([^']+)'", sql, flags=re.IGNORECASE):
        comparacao = "LIKE" if operador.upper() == "LIKE" else "="
        achadas += _consultar(f"SELECT DENOM_CIA, CD_CVM FROM companhias WHERE DENOM_CIA {comparacao} ?", (texto,))
    for codigo in re.findall(r"CD_CVM\s*=\s*'?(\d+)'?", sql):
        achadas += _consultar("SELECT DENOM_CIA, CD_CVM FROM companhias WHERE CD_CVM = ?", (codigo,))
    return list(dict.fromkeys(achadas))


# Períodos (DT_INI_EXERC a DT_FIM_EXERC) das linhas da demonstração, para a ordem de exercício do SQL.
def _exercicios(demonstracoes: list[str], empresas: list[tuple[str, str]], sql: str) -> list[str]:
    ordem = "PENÚLTIMO" if "PENÚLTIMO" in sql.upper() else "ÚLTIMO"
    periodos = []
    for tabela in demonstracoes:
        # O nome da tabela vem de PADRAO_DEMONSTRACAO, então é seguro usá-lo no SQL. Os balanços
        # (BPA e BPP) não têm DT_INI_EXERC: valem para uma data, o fim do exercício.
        colunas = [c[1] for c in _consultar(f"PRAGMA table_info({tabela})")]
        inicio = "DT_INI_EXERC" if "DT_INI_EXERC" in colunas else "NULL"
        consulta = f"SELECT DISTINCT {inicio}, DT_FIM_EXERC FROM {tabela} WHERE ORDEM_EXERC = ?"
        parametros = [ordem]
        if empresas:
            consulta += " AND CD_CVM IN (" + ",".join("?" * len(empresas)) + ")"
            parametros += [c for _, c in empresas]
        periodos += _consultar(consulta, tuple(parametros))
    return [f"{_data(i)} a {_data(f)} ({ordem})" if i else f"em {_data(f)} ({ordem})" for i, f in dict.fromkeys(periodos)]


def _data(texto: str) -> str:
    ano, mes, dia = texto.split("-")
    return f"{dia}/{mes}/{ano}"


# Devolve os quatro campos de proveniência da Metodologia, obtidos do SQL e do banco.
def obter_proveniencia(resultado_sql: dict) -> dict:
    sql = resultado_sql["sql"] or ""
    demonstracoes = _demonstracoes(sql)
    empresas = _empresas(resultado_sql) if resultado_sql["linhas"] or "DENOM_CIA" in sql else []
    return {
        "empresas": empresas,
        "demonstracoes": demonstracoes,
        "exercicios": _exercicios(demonstracoes, empresas, sql),
    }


def _nome_demonstracao(tabela: str) -> str:
    base, versao = tabela.rsplit("_", 1)
    return f"{NOMES_DEMONSTRACAO[base]}, {'consolidada' if versao == 'con' else 'individual'} ({tabela})"


def _tabela_de_dados(resultado_sql: dict) -> str:
    linhas = resultado_sql["linhas"]
    texto = " | ".join(rotular_colunas(resultado_sql["colunas"]))
    for linha in linhas[:MAX_LINHAS_NA_TABELA]:
        texto += "\n" + " | ".join(formatar_valor(v) for v in linha)
    if len(linhas) > MAX_LINHAS_NA_TABELA:
        texto += f"\n(mostrando as primeiras {MAX_LINHAS_NA_TABELA} de {len(linhas)} linhas)"
    return texto


# resposta: retorno de sintetizador.responder (tem "resposta" e "trechos"). resultado_sql: retorno
# de gerador_sql.gerar_sql, ou None no tipo conceitual. validacao: retorno de validador.validar.
def formatar(pergunta: str, tipo: str, resposta: dict, resultado_sql: dict | None, validacao: dict) -> str:
    partes = [resposta["resposta"]]

    if resultado_sql and resultado_sql["linhas"]:
        partes.append("Dados:\n" + _tabela_de_dados(resultado_sql))

    proveniencia = []
    if resultado_sql and resultado_sql["sql"]:
        p = obter_proveniencia(resultado_sql)
        if p["empresas"]:
            proveniencia.append("Empresa: " + "; ".join(f"{n} (código CVM {c})" for n, c in p["empresas"]))
        else:
            proveniencia.append("Empresa: não identificada no SQL (consulta sobre várias companhias)")
        if p["demonstracoes"]:
            proveniencia.append("Demonstração financeira: " + "; ".join(_nome_demonstracao(t) for t in p["demonstracoes"]))
        if p["exercicios"]:
            proveniencia.append("Exercício social: " + "; ".join(p["exercicios"]))
        proveniencia.append(f"Consulta SQL: {resultado_sql['sql']}")
    if resposta["trechos"]:
        if SEM_INFORMACAO.lower()[:30] in resposta["resposta"].lower():
            proveniencia.append("Documentos consultados: nenhum documento relevante encontrado")
        else:
            fontes = sorted({t["fonte"] for t in resposta["trechos"]})
            proveniencia.append(f"Documentos consultados: {', '.join(fontes)}")
    if proveniencia:
        partes.append("Proveniência:\n" + "\n".join(f"- {linha}" for linha in proveniencia))

    if not validacao["aprovado"]:
        partes.append(
            f"Aviso: a fidelidade da resposta ao contexto ficou em "
            f"{validacao['fidelidade']['score']:.2f} (abaixo do esperado). Confira os dados antes de usar."
        )

    return "\n\n".join(partes)
