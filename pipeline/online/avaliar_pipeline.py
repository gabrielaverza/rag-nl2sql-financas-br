"""
Etapa: Online (avaliação do fluxo completo).

Função: roda o orquestrador em cada pergunta do arquivo e resume o desempenho de ponta a ponta.

Descrição geral do funcionamento:
- Chama executar() de main.py (classificador, busca, SQL, síntese, validação e formatação).
- Mede, por pergunta e no conjunto: acerto da classe, EX do SQL (estrito e aceitando outras
  companhias que combinam com o nome, como em avaliar_sql.py), fidelidade e relevância médias,
  proporção de respostas aprovadas pelo validador e tempo.
- Não mede ainda a qualidade da resposta em texto (faithfulness e answer relevance do RAGAS
  completos); para isso, a saída opcional em JSON permite ler as respostas.

Dependências:
- pipeline/online/main.py e avaliar_sql.py (servidor Ollama em execução, com o modelo
  qwen2.5-coder:7b);
- dados/dfp.db e o arquivo de perguntas em dados/ (por padrão, dados/perguntas_dfp.json).

Parâmetros: caminho opcional do arquivo de perguntas, como primeiro argumento; --saida CAMINHO
para gravar as respostas completas em JSON; --limite N para rodar só as N primeiras perguntas.

Resultado: imprime o resultado de cada pergunta e o resumo. Só grava arquivo com --saida.

Executar de dentro de rag-nl2sql-financas-br/:
    python pipeline/online/avaliar_pipeline.py
    python pipeline/online/avaliar_pipeline.py dados/perguntas_validacao.json --saida resultados.json
"""

import argparse
import json
import sys
import time

from avaliar_sql import resultado_contem_referencia, resultados_equivalentes
from gerador_sql import executar_sql
from main import executar

CAMINHO_PERGUNTAS = "dados/perguntas_dfp.json"


def media(valores: list[float]) -> str:
    return f"{sum(valores) / len(valores):.2f}" if valores else "-"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("perguntas", nargs="?", default=CAMINHO_PERGUNTAS)
    parser.add_argument("--saida")
    parser.add_argument("--limite", type=int)
    args = parser.parse_args()
    with open(args.perguntas, encoding="utf-8") as f:
        perguntas = json.load(f)
    if args.limite:
        perguntas = perguntas[: args.limite]

    print(f"=== {args.perguntas} ===")
    registros = []
    for p in perguntas:
        inicio = time.time()
        r = executar(p["pergunta"])
        tempo = time.time() - inicio
        classe_certa = r["classe"] == p["tipo"]

        ex_estrito = ex_companhias = None
        r_sql = r["resultado_sql"]
        if p["sql_referencia"]:
            ex_estrito = ex_companhias = False
            if r_sql and r_sql["status"] == "ok":
                _, referencia = executar_sql(p["sql_referencia"], limite_linhas=None)
                ex_estrito = resultados_equivalentes(r_sql["linhas"], referencia)
                ex_companhias = ex_estrito or resultado_contem_referencia(r_sql["colunas"], r_sql["linhas"], referencia)

        v = r["validacao"]
        registros.append({
            "id": p["id"], "tipo": p["tipo"], "classe": r["classe"], "classe_certa": classe_certa,
            "ex_estrito": ex_estrito, "ex_companhias": ex_companhias,
            "fidelidade": v["fidelidade"]["score"], "relevancia": v["relevancia_contexto"]["score"],
            "aprovado": v["aprovado"], "tempo": tempo, "pergunta": p["pergunta"], "resposta": r["resposta"],
            "sql": r_sql["sql"] if r_sql else None,
        })
        ex = "-" if ex_estrito is None else ("ok" if ex_estrito else ("ok+" if ex_companhias else "ERRO"))
        print(f"  {p['id']} [{p['tipo']:12s}] classe={'ok  ' if classe_certa else 'ERRO'} ex={ex:4s} "
              f"fidelidade={v['fidelidade']['score']:.2f} aprovado={'sim' if v['aprovado'] else 'NAO'} ({tempo:.0f}s)")

    print("\n=== Resumo ===")
    print(f"Perguntas: {len(registros)}")
    print(f"Classe correta: {sum(x['classe_certa'] for x in registros)}/{len(registros)}")
    com_sql = [x for x in registros if x["ex_estrito"] is not None]
    if com_sql:
        print(f"EX estrito (quantitativas e híbridas): {sum(x['ex_estrito'] for x in com_sql)}/{len(com_sql)}")
        print(f"EX aceitando outras companhias que combinam com o nome: {sum(x['ex_companhias'] for x in com_sql)}/{len(com_sql)}")
    print(f"Fidelidade média: {media([x['fidelidade'] for x in registros])}")
    print(f"Relevância média do contexto: {media([x['relevancia'] for x in registros if x['relevancia'] is not None])}")
    print(f"Respostas aprovadas pelo validador: {sum(x['aprovado'] for x in registros)}/{len(registros)}")
    print(f"Tempo médio por pergunta: {media([x['tempo'] for x in registros])} s")

    if args.saida:
        with open(args.saida, "w", encoding="utf-8") as f:
            json.dump(registros, f, ensure_ascii=False, indent=2)
        print(f"Respostas gravadas em {args.saida}")


if __name__ == "__main__":
    main()
