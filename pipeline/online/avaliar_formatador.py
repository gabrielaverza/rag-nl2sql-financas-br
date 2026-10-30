"""
Etapa: Online (avaliação do passo de formatação).

Função: inspeciona o formatador (proveniência com os quatro campos e aviso de baixa fidelidade).

Descrição geral do funcionamento:
- Usa o resultado do SQL de referência (como em avaliar_sintese.py), para não depender do
  gerador de SQL, e valida a resposta de verdade com o validador.
- Confere os quatro campos de proveniência (empresa, demonstração, exercício social, código CVM)
  em uma pergunta de cada tipo e em casos em que a empresa não vem no resultado (filtro por
  nome no SQL), vem de duas companhias (nome ambíguo) ou não existe (consulta sobre todas).
- Confere o aviso quando a fidelidade fica abaixo do limiar e a omissão dos documentos numa recusa.

Dependências:
- pipeline/online/formatador.py, sintetizador.py, validador.py e gerador_sql.py (servidor Ollama
  em execução, com o modelo qwen2.5-coder:7b);
- dados/dfp.db e dados/perguntas_dfp.json.

Parâmetros: nenhum argumento.

Resultado: imprime a resposta formatada de cada caso. Não grava arquivos.

Executar de dentro de rag-nl2sql-financas-br/:
    python pipeline/online/avaliar_formatador.py
"""

import json
import sys

from formatador import formatar
from gerador_sql import executar_sql
from sintetizador import responder
from validador import validar

CAMINHO_PERGUNTAS = "dados/perguntas_dfp.json"
# Uma pergunta conceitual, uma quantitativa (empresa fora do resultado), duas com nome ambíguo
# ou várias companhias (JBS) e uma híbrida.
PERGUNTAS = ("c01", "q01", "q08", "q13", "h01")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    with open(CAMINHO_PERGUNTAS, encoding="utf-8") as f:
        perguntas = {p["id"]: p for p in json.load(f)}

    for id_ in PERGUNTAS:
        p = perguntas[id_]
        resultado_sql = None
        if p["sql_referencia"]:
            colunas, linhas = executar_sql(p["sql_referencia"])
            resultado_sql = {
                "sql": p["sql_referencia"], "colunas": colunas, "linhas": linhas, "status": "ok",
                "tabelas_contexto": p["tabelas_esperadas"],
            }
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


if __name__ == "__main__":
    main()
