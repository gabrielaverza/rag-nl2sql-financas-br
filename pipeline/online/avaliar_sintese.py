"""
Etapa: Online (avaliação do passo de síntese).

Função: inspeciona a síntese da resposta (conceitual, quantitativa e híbrida).

Descrição geral do funcionamento:
- Para isolar a síntese dos erros do gerador de SQL, os tipos quantitativa e híbrida usam o
  resultado do SQL de referência do arquivo de perguntas. As conceituais usam a busca no glossário.
- O script imprime as respostas para leitura e faz duas checagens automáticas simples:
  - números do resultado: quantos valores numéricos do resultado aparecem na resposta
    (comparando só os dígitos, sem separadores de milhar e decimal);
  - perguntas fora do escopo dos documentos: a resposta deve dizer que não encontrou.

Dependências:
- pipeline/online/sintetizador.py e gerador_sql.py (servidor Ollama em execução, com o modelo
  qwen2.5-coder:7b);
- dados/dfp.db e o arquivo de perguntas em dados/ (por padrão, dados/perguntas_dfp.json).

Parâmetros: caminho opcional do arquivo de perguntas, como primeiro argumento; --todas para
usar todas as quantitativas (por padrão, só as 5 primeiras e as híbridas).

Resultado: imprime a resposta de cada pergunta, a contagem de números presentes e as recusas
corretas. Não grava arquivos.

Executar de dentro de rag-nl2sql-financas-br/:
    python pipeline/online/avaliar_sintese.py
    python pipeline/online/avaliar_sintese.py dados/perguntas_validacao.json
"""

import argparse
import json
import re
import sys

from gerador_sql import executar_sql
from sintetizador import SEM_INFORMACAO, responder

CAMINHO_PERGUNTAS = "dados/perguntas_dfp.json"
# Perguntas conceituais cuja resposta NÃO está nos normativos indexados (conferido: EBITDA
# tem zero ocorrências no índice do glossário).
FORA_DO_ESCOPO = [
    "O que é EBITDA?",
    "Qual a capital da França?",
    "Qual é o preço da ação da Petrobras hoje?",
]
MAX_LINHAS_CONFERIDAS = 30
QUANTITATIVAS_POR_PADRAO = 5


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


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("perguntas", nargs="?", default=CAMINHO_PERGUNTAS)
    parser.add_argument("--todas", action="store_true")
    args = parser.parse_args()
    with open(args.perguntas, encoding="utf-8") as f:
        perguntas = json.load(f)

    print(f"=== {args.perguntas}: respostas com o resultado do SQL de referência ===")
    quantitativas = 0
    achados_total, numeros_total = 0, 0
    for p in perguntas:
        if p["tipo"] == "quantitativa":
            quantitativas += 1
            if not args.todas and quantitativas > QUANTITATIVAS_POR_PADRAO:
                continue
        resultado = None
        if p["sql_referencia"]:
            colunas, linhas = executar_sql(p["sql_referencia"])
            resultado = {"sql": p["sql_referencia"], "colunas": colunas, "linhas": linhas, "status": "ok"}
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


if __name__ == "__main__":
    main()
