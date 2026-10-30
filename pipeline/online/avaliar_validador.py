"""
Etapa: Online (avaliação do passo de validação).

Função: inspeciona a validação (relevância do contexto e fidelidade da resposta).

Descrição geral do funcionamento:
- Como em avaliar_sintese.py, os tipos quantitativa e híbrida usam o resultado do SQL de
  referência, para isolar a validação dos erros do gerador de SQL.
- Inclui também respostas propositalmente erradas (um número trocado e um fato inventado),
  para conferir se a fidelidade detecta o problema, e uma resposta correta conhecida, para
  conferir que ela não é reprovada sem motivo.
- Imprime a fidelidade e a relevância de cada resposta e as médias.

Dependências:
- pipeline/online/sintetizador.py, validador.py e gerador_sql.py (servidor Ollama em execução,
  com o modelo qwen2.5-coder:7b);
- dados/dfp.db e o arquivo de perguntas em dados/ (por padrão, dados/perguntas_dfp.json).

Parâmetros: caminho opcional do arquivo de perguntas, como primeiro argumento.

Resultado: imprime a fidelidade e a relevância por pergunta, os casos com erro proposital e as
médias. Não grava arquivos.

Executar de dentro de rag-nl2sql-financas-br/:
    python pipeline/online/avaliar_validador.py
"""

import json
import sys

from gerador_sql import executar_sql
from sintetizador import responder
from validador import validar

CAMINHO_PERGUNTAS = "dados/perguntas_dfp.json"
# Quantitativas conferidas (as demais são puladas para o script não demorar).
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


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    caminho = sys.argv[1] if len(sys.argv) > 1 else CAMINHO_PERGUNTAS
    with open(caminho, encoding="utf-8") as f:
        perguntas = json.load(f)

    print(f"=== {caminho}: validação das respostas geradas ===")
    fidelidades, relevancias = [], []
    for p in perguntas:
        if p["tipo"] == "quantitativa" and p["id"] not in QUANTITATIVAS_CONFERIDAS:
            continue
        resultado_sql = None
        if p["sql_referencia"]:
            colunas, linhas = executar_sql(p["sql_referencia"])
            resultado_sql = {"sql": p["sql_referencia"], "colunas": colunas, "linhas": linhas, "status": "ok"}
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


if __name__ == "__main__":
    main()
