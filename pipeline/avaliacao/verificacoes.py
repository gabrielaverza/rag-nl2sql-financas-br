"""
Etapa: Avaliação (verificações de sanidade da síntese, da validação e da formatação).

Função: testes rápidos, com casos conhecidos, para conferir que a síntese, o validador e o formatador
se comportam como esperado. Não são métricas dos experimentos: servem para perceber, antes de uma
rodada longa, se um desses passos quebrou ou passou a se comportar de outro jeito.

Descrição geral do funcionamento:
- Para isolar cada passo dos erros do gerador de SQL, as perguntas quantitativas e híbridas usam o
  resultado do SQL de referência do arquivo de perguntas.
- sintese: imprime as respostas para leitura e faz duas checagens automáticas simples: quantos
  valores numéricos do resultado aparecem no texto (comparando só os dígitos) e se perguntas fora do
  escopo dos documentos (EBITDA, capital da França, preço da ação) são recusadas.
- validador: aplica a validação a respostas geradas e a casos conhecidos: um número trocado e um fato
  inventado devem ser reprovados, e uma resposta correta deve ser aprovada.
- formatador: mostra a resposta formatada de uma pergunta de cada tipo e de casos em que a empresa não
  vem no resultado (filtro por nome no SQL), de fidelidade baixa (o aviso deve aparecer) e de recusa
  (nenhum documento deve ser listado).

Dependências:
- pipeline/online/sintetizador.py, validador.py, formatador.py e gerador_sql.py (servidor Ollama em
  execução, com o modelo qwen2.5-coder:7b);
- dados/dfp.db e dados/perguntas_validacao.json.

Parâmetros: o nome da verificação (sintese, validador, formatador ou todas); em sintese, --todas usa
todas as quantitativas (por padrão, só as 5 primeiras e as híbridas).

Resultado: imprime o resultado de cada caso. Não grava arquivos.

Executar de dentro de rag-nl2sql-financas-br/:
    python pipeline/avaliacao/verificacoes.py sintese
    python pipeline/avaliacao/verificacoes.py validador
    python pipeline/avaliacao/verificacoes.py formatador
"""

import argparse
import json
import os
import re
import sys

# Os módulos do fluxo ficam em pipeline/online/ e os compartilhados em pipeline/.
PASTA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PASTA, "..", "online"))
sys.path.insert(0, os.path.join(PASTA, ".."))

CAMINHO_PERGUNTAS = "dados/perguntas_validacao.json"

# Síntese. Perguntas conceituais cuja resposta NÃO está nos normativos indexados (conferido: EBITDA
# tem zero ocorrências no índice do glossário).
FORA_DO_ESCOPO = [
    "O que é EBITDA?",
    "Qual a capital da França?",
    "Qual é o preço da ação da Petrobras hoje?",
]
MAX_LINHAS_CONFERIDAS = 30
QUANTITATIVAS_POR_PADRAO = 5

# Validador. Quantitativas conferidas (as demais são puladas para o teste não demorar).
QUANTITATIVAS_CONFERIDAS = ("q01", "q02", "q03")
CONTEXTO_CONHECIDO = (
    "DADOS DO BANCO\n"
    "Consulta executada: SELECT CASE WHEN ESCALA_MOEDA = 'MIL' THEN VL_CONTA * 1000 ELSE VL_CONTA END "
    "FROM DRE_con WHERE CD_CONTA = '3.01'\n"
    "Resultado (valores monetários em reais):\n"
    "- DENOM_CIA: AMBEV S.A.; receita: 88.242.467.000,00\n"
)
PERGUNTA_CONHECIDA = "Qual foi a receita líquida consolidada da Ambev em 2025?"
CASOS_COM_ERRO = [
    ("número trocado", "A receita líquida da AMBEV S.A. foi de R$ 98.242.467.000,00."),
    ("fato inventado", "A receita líquida da AMBEV S.A. foi de R$ 88.242.467.000,00, a maior da América Latina."),
]
CASO_CORRETO = ("resposta correta", "A receita líquida da AMBEV S.A. foi de R$ 88.242.467.000,00.")

# Formatador. Uma pergunta conceitual, uma quantitativa (empresa só no filtro do SQL), uma que compara
# duas companhias, uma no balanço individual e uma híbrida.
PERGUNTAS_FORMATADOR = ("c01", "q01", "q09", "q05", "h01")


# Lê o arquivo de perguntas.
def ler_perguntas(caminho: str = CAMINHO_PERGUNTAS) -> list[dict]:
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


# Executa o SQL de referência da pergunta e devolve o resultado no formato do gerador de SQL.
def resultado_de_referencia(p: dict, com_tabelas: bool = False):
    from gerador_sql import executar_sql
    if not p["sql_referencia"]:
        return None
    colunas, linhas = executar_sql(p["sql_referencia"])
    resultado = {"sql": p["sql_referencia"], "colunas": colunas, "linhas": linhas, "status": "ok"}
    if com_tabelas:
        resultado["tabelas_contexto"] = p["tabelas_esperadas"]
    return resultado


# ---------------------------------------------------------------------------
# 1) Síntese
# ---------------------------------------------------------------------------

def so_digitos(texto: str) -> str:
    return re.sub(r"[^0-9]", "", texto)


# Conta os valores numéricos do resultado que aparecem na resposta. Compara só os dígitos, para
# "88.242.467.000,00" casar com 88242467000.0. Valores com parte decimal usam as 2 casas.
def numeros_presentes(resposta: str, linhas: list[tuple]) -> tuple[int, int]:
    alvo = so_digitos(resposta)
    numeros = [v for linha in linhas[:MAX_LINHAS_CONFERIDAS] for v in linha if isinstance(v, (int, float))]
    achados = 0
    for v in numeros:
        texto = f"{v:.0f}" if float(v) == round(v) else f"{v:.2f}"
        achados += so_digitos(texto) in alvo
    return achados, len(numeros)


# Imprime as respostas da síntese com o SQL de referência e confere as recusas fora do escopo.
def verificar_sintese(todas: bool):
    from sintetizador import SEM_INFORMACAO, responder
    perguntas = ler_perguntas()

    print(f"=== {CAMINHO_PERGUNTAS}: respostas com o resultado do SQL de referência ===")
    quantitativas = 0
    achados_total, numeros_total = 0, 0
    for p in perguntas:
        if p["tipo"] == "quantitativa":
            quantitativas += 1
            if not todas and quantitativas > QUANTITATIVAS_POR_PADRAO:
                continue
        resultado = resultado_de_referencia(p)
        r = responder(p["pergunta"], p["tipo"], resultado)
        print(f"\n{p['id']} [{p['tipo']}] {p['pergunta']}")
        print("  ->", r["resposta"].replace("\n", "\n     "))
        if resultado:
            achados, total = numeros_presentes(r["resposta"], resultado["linhas"])
            achados_total, numeros_total = achados_total + achados, numeros_total + total
            print(f"  [números do resultado presentes na resposta: {achados}/{total}]")
        if r["trechos"]:
            print("  [trechos usados:", ", ".join(t["id"].split("_")[-1] for t in r["trechos"]), "]")
    if numeros_total:
        print(f"\nNúmeros do resultado presentes nas respostas: {achados_total}/{numeros_total}")

    print("\n=== Fora do escopo dos documentos (deve dizer que não encontrou) ===")
    corretas = 0
    for pergunta in FORA_DO_ESCOPO:
        r = responder(pergunta, "conceitual")
        recusou = SEM_INFORMACAO.lower()[:30] in r["resposta"].lower()
        corretas += recusou
        print(f"\n{'ok  ' if recusou else 'FALHA'} {pergunta}\n  -> {r['resposta']}")
    print(f"\nRecusas corretas: {corretas}/{len(FORA_DO_ESCOPO)}")


# ---------------------------------------------------------------------------
# 2) Validador
# ---------------------------------------------------------------------------

# Aplica a validação a respostas geradas e a casos conhecidos (erros propositais e resposta correta).
def verificar_validador():
    from sintetizador import responder
    from validador import validar
    perguntas = ler_perguntas()

    print(f"=== {CAMINHO_PERGUNTAS}: validação das respostas geradas ===")
    fidelidades, relevancias = [], []
    for p in perguntas:
        if p["tipo"] == "quantitativa" and p["id"] not in QUANTITATIVAS_CONFERIDAS:
            continue
        resultado_sql = resultado_de_referencia(p)
        r = responder(p["pergunta"], p["tipo"], resultado_sql)
        v = validar(p["pergunta"], r["resposta"], r["contexto"], r["trechos"])

        fidelidades.append(v["fidelidade"]["score"])
        print(f"\n{p['id']} [{p['tipo']}] {p['pergunta']}")
        print("  resposta:", r["resposta"].replace("\n", " ")[:200])
        print(f"  fidelidade: {v['fidelidade']['score']:.2f} "
              f"({sum(a['sustentada'] for a in v['fidelidade']['afirmacoes'])}/{len(v['fidelidade']['afirmacoes'])} afirmações) "
              f"| aprovado={v['aprovado']}")
        for a in v["fidelidade"]["afirmacoes"]:
            if not a["sustentada"]:
                print(f"    NÃO SUSTENTADA: {a['afirmacao']}")
        if v["relevancia_contexto"]["score"] is not None:
            relevancias.append(v["relevancia_contexto"]["score"])
            irrelevantes = [t["id"] for t in v["relevancia_contexto"]["por_trecho"] if not t["relevante"]]
            print(f"  relevância do contexto: {v['relevancia_contexto']['score']:.2f}"
                  + (f" (irrelevantes: {irrelevantes})" if irrelevantes else ""))

    print("\n=== Casos conhecidos (os com erro devem ser reprovados e o correto aprovado) ===")
    reprovados = 0
    for nome, resposta in CASOS_COM_ERRO + [CASO_CORRETO]:
        v = validar(PERGUNTA_CONHECIDA, resposta, CONTEXTO_CONHECIDO, [])
        print(f"\n{nome}: \"{resposta}\"")
        print(f"  fidelidade: {v['fidelidade']['score']:.2f} | aprovado={v['aprovado']}")
        for a in v["fidelidade"]["afirmacoes"]:
            print(f"    {'ok  ' if a['sustentada'] else 'FALHA'} {a['afirmacao']}")
        if nome != CASO_CORRETO[0]:
            reprovados += not v["aprovado"]
    print(f"\nErros propositais reprovados: {reprovados}/{len(CASOS_COM_ERRO)}")

    if fidelidades:
        print(f"Fidelidade média (respostas geradas): {sum(fidelidades) / len(fidelidades):.2f}")
    if relevancias:
        print(f"Relevância média do contexto: {sum(relevancias) / len(relevancias):.2f}")


# ---------------------------------------------------------------------------
# 3) Formatador
# ---------------------------------------------------------------------------

# Mostra a resposta formatada de casos de cada tipo, de fidelidade baixa e de recusa.
def verificar_formatador():
    from formatador import formatar
    from gerador_sql import executar_sql
    from sintetizador import responder
    from validador import validar
    perguntas = {p["id"]: p for p in ler_perguntas()}

    for id_ in PERGUNTAS_FORMATADOR:
        p = perguntas[id_]
        resultado_sql = resultado_de_referencia(p, com_tabelas=True)
        r = responder(p["pergunta"], p["tipo"], resultado_sql)
        v = validar(p["pergunta"], r["resposta"], r["contexto"], r["trechos"])
        print(f"=== {id_} [{p['tipo']}] {p['pergunta']} ===\n{formatar(p['pergunta'], p['tipo'], r, resultado_sql, v)}\n")

    print("=== Caso com fidelidade baixa (o aviso deve aparecer) ===")
    sql = ("SELECT companhias.DENOM_CIA, CASE WHEN ESCALA_MOEDA = 'MIL' THEN VL_CONTA * 1000 ELSE VL_CONTA END "
           "FROM DRE_con JOIN companhias ON companhias.CD_CVM = DRE_con.CD_CVM "
           "WHERE companhias.DENOM_CIA LIKE '%AMBEV%' AND CD_CONTA = '3.01' AND ORDEM_EXERC = 'ÚLTIMO'")
    colunas, linhas = executar_sql(sql)
    resultado_sql = {"sql": sql, "colunas": colunas, "linhas": linhas, "status": "ok", "tabelas_contexto": ["DRE_con"]}
    resposta_falsa = {"resposta": "A receita líquida da AMBEV S.A. foi de R$ 1.000,00.", "trechos": []}
    contexto = "DADOS DO BANCO\nResultado (valores monetários em reais):\n- DENOM_CIA: AMBEV S.A.; receita: 88.242.467.000,00\n"
    v = validar("Qual foi a receita líquida da Ambev?", resposta_falsa["resposta"], contexto, [])
    print(formatar("Qual foi a receita líquida da Ambev?", "quantitativa", resposta_falsa, resultado_sql, v))

    print("\n=== Recusa (nenhum documento deve ser listado) ===")
    r = responder("O que é EBITDA?", "conceitual")
    v = validar("O que é EBITDA?", r["resposta"], r["contexto"], r["trechos"])
    print(formatar("O que é EBITDA?", "conceitual", r, None, v))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Verificações de sanidade da síntese, do validador e do formatador.")
    parser.add_argument("verificacao", choices=("sintese", "validador", "formatador", "todas"))
    parser.add_argument("--todas", action="store_true", help="em sintese, usa todas as quantitativas")
    args = parser.parse_args()
    if args.verificacao in ("sintese", "todas"):
        verificar_sintese(args.todas)
    if args.verificacao in ("validador", "todas"):
        verificar_validador()
    if args.verificacao in ("formatador", "todas"):
        verificar_formatador()


if __name__ == "__main__":
    main()
