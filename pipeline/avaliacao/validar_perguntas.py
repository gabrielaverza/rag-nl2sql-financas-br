"""
Etapa: Online (preparação da avaliação em perguntas novas).

Função: confere um arquivo de perguntas novas antes de ele ser usado nas avaliações de recuperação.

Descrição geral do funcionamento:
- Cada pergunta tem os campos id, pergunta, tipo (quantitativa, conceitual ou hibrida),
  tabelas_esperadas (lista de nomes de tabela) e evidencia (lista de textos copiados dos normativos).
  O campo sql_referencia é opcional (só serve à avaliação do SQL).
- Erros: campo faltando, tipo desconhecido, id ou pergunta repetidos no arquivo, tabela que não existe em embeddings/esquema.json, evidência que não
  aparece em nenhum trecho de embeddings/index_glossario.json, e combinação errada com o tipo
  (quantitativa: só tabelas; conceitual: só evidência; híbrida: as duas).
- Avisos: evidência que aparece em mais de 5 trechos (genérica demais, o Recall perde sentido) e
  conjunto muito desequilibrado entre os tipos.
- A evidência é comparada com os trechos como no executor dos experimentos: espaços normalizados, sem
  ignorar maiúsculas.
- Nenhuma busca por embeddings ou BM25 é feita aqui, então a conferência não depende dos métodos
  avaliados (a evidência deve ser escolhida lendo os normativos, e não olhando o que a busca devolve).
- Com a opção --base, confere também os campos de base do conjunto final: base_dados (companhia, cd_cvm,
  tabela e cd_conta existem em dados/dfp.db) e sql_referencia (executa e devolve valor), nas perguntas
  quantitativas e híbridas; base_normativo (arquivo, pagina, trecho_id, dispositivo, e os trechos
  listados são exatamente os que contêm a evidência, ou qualquer pedaço dela quando a frase cruza o limite entre dois
  trechos), nas conceituais e híbridas.

Dependências: embeddings/esquema.json e embeddings/index_glossario.json; dados/dfp.db (só com --base).
Só biblioteca padrão.

Parâmetros: caminho do arquivo de perguntas a conferir (o conjunto oficial é dados/perguntas_validacao.json),
como primeiro argumento; --base é opcional.

Resultado: imprime os erros, os avisos e a contagem por tipo; termina com código 1 se houver erro.

Executar de dentro de rag-nl2sql-financas-br/:
    python pipeline/avaliacao/validar_perguntas.py dados/perguntas_validacao.json --base
"""

import json
import re
import sqlite3
import sys

CAMINHO_ESQUEMA = "embeddings/esquema.json"
CAMINHO_GLOSSARIO = "embeddings/index_glossario.json"
CAMINHO_BANCO = "dados/dfp.db"
TIPOS = ("quantitativa", "conceitual", "hibrida")
CAMPOS = ("id", "pergunta", "tipo", "tabelas_esperadas", "evidencia")
CAMPOS_BASE_DADOS = ("companhia", "cd_cvm", "tabela", "cd_conta")
CAMPOS_BASE_NORMATIVO = ("arquivo", "pagina", "trecho_id", "dispositivo")
MAX_TRECHOS_POR_EVIDENCIA = 5


# Troca qualquer sequência de espaços e quebras de linha por um espaço.
def normalizar(texto: str) -> str:
    return re.sub(r"\s+", " ", texto)


# Lê um arquivo JSON.
def ler(caminho: str):
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


# Confere as perguntas e devolve (erros, avisos).
def conferir(perguntas: list[dict], caminho: str) -> tuple[list[str], list[str]]:
    erros, avisos = [], []
    tabelas = {e["tabela"] for e in ler(CAMINHO_ESQUEMA)}
    trechos = [normalizar(r["conteudo"]) for r in ler(CAMINHO_GLOSSARIO)]

    vistos, perguntas_vistas = set(), set()
    for i, p in enumerate(perguntas, start=1):
        nome = p.get("id", f"#{i}")
        faltando = [c for c in CAMPOS if c not in p]
        if faltando:
            erros.append(f"{nome}: campos faltando: {faltando}")
            continue
        if p["tipo"] not in TIPOS:
            erros.append(f"{nome}: tipo desconhecido: {p['tipo']}")
            continue
        if p["id"] in vistos:
            erros.append(f"{nome}: id repetido neste arquivo")
        vistos.add(p["id"])
        if normalizar(p["pergunta"]).lower() in perguntas_vistas:
            erros.append(f"{nome}: pergunta repetida neste arquivo")
        perguntas_vistas.add(normalizar(p["pergunta"]).lower())

        tem_tabelas, tem_evidencia = bool(p["tabelas_esperadas"]), bool(p["evidencia"])
        if p["tipo"] == "quantitativa" and (not tem_tabelas or tem_evidencia):
            erros.append(f"{nome}: quantitativa precisa de tabelas_esperadas e de evidencia vazia")
        if p["tipo"] == "conceitual" and (tem_tabelas or not tem_evidencia):
            erros.append(f"{nome}: conceitual precisa de evidencia e de tabelas_esperadas vazia")
        if p["tipo"] == "hibrida" and not (tem_tabelas and tem_evidencia):
            erros.append(f"{nome}: híbrida precisa de tabelas_esperadas e de evidencia")

        for tabela in p["tabelas_esperadas"]:
            if tabela not in tabelas:
                erros.append(f"{nome}: tabela inexistente: {tabela}")
        for evidencia in p["evidencia"]:
            achados = sum(1 for t in trechos if normalizar(evidencia) in t)
            if achados == 0:
                erros.append(f"{nome}: evidência não aparece em nenhum trecho: {evidencia[:60]!r}")
            elif achados > MAX_TRECHOS_POR_EVIDENCIA:
                avisos.append(f"{nome}: evidência em {achados} trechos (genérica demais): {evidencia[:60]!r}")

    por_tipo = {t: sum(1 for p in perguntas if p.get("tipo") == t) for t in TIPOS}
    if perguntas and min(por_tipo.values()) < len(perguntas) / 6:
        avisos.append(f"conjunto desequilibrado entre os tipos: {por_tipo}")
    return erros, avisos


# Confere os campos de base do conjunto final e devolve a lista de erros.
def conferir_base(perguntas: list[dict]) -> list[str]:
    erros = []
    tabelas = {e["tabela"] for e in ler(CAMINHO_ESQUEMA)}
    registros = {r["id"]: r for r in ler(CAMINHO_GLOSSARIO)}
    trechos = {rid: normalizar(r["conteudo"]) for rid, r in registros.items()}
    # Modo somente leitura: a conferência nunca altera o banco.
    banco = sqlite3.connect(f"file:{CAMINHO_BANCO}?mode=ro", uri=True)

    for p in perguntas:
        nome = p.get("id", "?")
        tipo = p.get("tipo")
        precisa_dados = tipo in ("quantitativa", "hibrida")
        precisa_norma = tipo in ("conceitual", "hibrida")
        dados, norma = p.get("base_dados"), p.get("base_normativo")

        if precisa_dados:
            if not dados:
                erros.append(f"{nome}: base_dados faltando")
            for b in dados or []:
                faltando = [c for c in CAMPOS_BASE_DADOS if c not in b]
                if faltando:
                    erros.append(f"{nome}: base_dados sem {faltando}")
                    continue
                if b["tabela"] not in tabelas:
                    erros.append(f"{nome}: base_dados com tabela inexistente: {b['tabela']}")
                    continue
                if b["tabela"] not in p["tabelas_esperadas"]:
                    erros.append(f"{nome}: tabela da base ({b['tabela']}) fora de tabelas_esperadas")
                linha = banco.execute("SELECT DENOM_CIA FROM companhias WHERE CD_CVM = ?", (b["cd_cvm"],)).fetchone()
                if linha is None or linha[0] != b["companhia"]:
                    erros.append(f"{nome}: companhia e cd_cvm não conferem com o banco: {b['companhia']!r}")
                existe = banco.execute(
                    f"SELECT 1 FROM {b['tabela']} WHERE CD_CVM = ? AND CD_CONTA = ? AND ORDEM_EXERC = 'ÚLTIMO'",
                    (b["cd_cvm"], b["cd_conta"])).fetchone()
                if existe is None:
                    erros.append(f"{nome}: conta {b['cd_conta']} não existe em {b['tabela']} para {b['companhia']}")
            sql = p.get("sql_referencia")
            if not sql:
                erros.append(f"{nome}: sql_referencia faltando")
            else:
                try:
                    linhas = banco.execute(sql).fetchall()
                    if not linhas or any(v is None for linha in linhas for v in linha):
                        erros.append(f"{nome}: sql_referencia sem resultado ou com valor nulo")
                except sqlite3.Error as e:
                    erros.append(f"{nome}: sql_referencia não executa: {e}")
        elif dados:
            erros.append(f"{nome}: conceitual não deve ter base_dados")

        if precisa_norma:
            if not norma:
                erros.append(f"{nome}: base_normativo faltando")
                continue
            faltando = [c for c in CAMPOS_BASE_NORMATIVO if c not in norma]
            if faltando:
                erros.append(f"{nome}: base_normativo sem {faltando}")
                continue
            if not isinstance(norma["pagina"], int) or norma["pagina"] < 1:
                erros.append(f"{nome}: pagina inválida: {norma['pagina']!r}")
            if not str(norma["dispositivo"]).strip():
                erros.append(f"{nome}: dispositivo vazio")
            ids = norma["trecho_id"]
            if not ids or any(i not in registros for i in ids):
                erros.append(f"{nome}: trecho_id inexistente no índice: {ids}")
                continue
            if any(registros[i]["fonte"] != norma["arquivo"] for i in ids):
                erros.append(f"{nome}: arquivo da base não é a fonte dos trechos listados")
            # Os trechos listados têm de ser exatamente os que contêm a evidência (ou qualquer um dos
            # pedaços dela, quando a frase cruza o limite entre dois trechos).
            evidencias = [normalizar(e) for e in p["evidencia"]]
            com_evidencia = {rid for rid, t in trechos.items() if any(e in t for e in evidencias)}
            if set(ids) != com_evidencia:
                erros.append(f"{nome}: trecho_id difere dos trechos que contêm a evidência: {sorted(com_evidencia)}")
        elif norma:
            erros.append(f"{nome}: quantitativa não deve ter base_normativo")
    return erros


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    argumentos = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not argumentos:
        print("Uso: python pipeline/avaliacao/validar_perguntas.py arquivo_de_perguntas.json [--base]")
        sys.exit(1)
    caminho = argumentos[0]
    perguntas = ler(caminho)
    erros, avisos = conferir(perguntas, caminho)
    if "--base" in sys.argv:
        erros += conferir_base(perguntas)
    por_tipo = {t: sum(1 for p in perguntas if p.get("tipo") == t) for t in TIPOS}
    print(f"{caminho}: {len(perguntas)} perguntas {por_tipo}")
    for e in erros:
        print(f"  ERRO: {e}")
    for a in avisos:
        print(f"  aviso: {a}")
    print(f"{len(erros)} erro(s), {len(avisos)} aviso(s)")
    sys.exit(1 if erros else 0)


if __name__ == "__main__":
    main()
