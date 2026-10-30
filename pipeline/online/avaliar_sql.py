"""
Etapa: Online (avaliação do passo de geração de SQL).

Função: avalia o gerador de SQL com um arquivo de perguntas de teste.

Descrição geral do funcionamento:
- Usa as perguntas quantitativas e híbridas, que têm SQL de referência.
- Métrica: EX (acurácia de execução). Uma consulta acerta se o resultado dela é igual ao
  resultado do SQL de referência. Como na avaliação de Zhong et al. (2020), dois resultados
  que só diferem pela ordem das colunas são equivalentes. Além disso, a comparação ignora a
  ordem das linhas (as perguntas não pedem uma ordem) e arredonda números para 2 casas
  decimais. O nome das colunas não conta, mas a quantidade delas sim.
- O gerador devolve uma linha para cada companhia que combina com o nome do LIKE (por exemplo,
  "JBS" traz JBS S.A. e JBS N.V.), com DENOM_CIA na primeira coluna. Por isso há uma segunda
  contagem, que também aceita o resultado que traz as linhas da referência mais as de outras
  companhias que combinam com o nome (marcado como ok+). A coluna DENOM_CIA é a única extra aceita.
- As tabelas do contexto vêm da busca (com k tabelas) ou, com --tabelas-ideais, de
  tabelas_esperadas. A segunda configuração mostra quanto a busca ainda custa.

Dependências:
- pipeline/online/gerador_sql.py (servidor Ollama em execução, com o modelo qwen2.5-coder:7b);
- dados/dfp.db e o arquivo de perguntas em dados/ (por padrão, dados/perguntas_dfp.json).

Parâmetros: caminho opcional do arquivo de perguntas, como primeiro argumento; --k N para o
número de tabelas da busca; --tabelas-ideais para usar as tabelas esperadas.

Resultado: imprime o resultado de cada pergunta, o EX resumido por tipo e por completude do
contexto, e os erros com o SQL gerado e o de referência. Não grava arquivos.

Executar de dentro de rag-nl2sql-financas-br/:
    python pipeline/online/avaliar_sql.py
    python pipeline/online/avaliar_sql.py dados/perguntas_validacao.json --k 5
    python pipeline/online/avaliar_sql.py --tabelas-ideais
"""

import argparse
import itertools
import json
import sys
import time
from collections import Counter

from gerador_sql import K_TABELAS, executar_sql, gerar_sql

CAMINHO_PERGUNTAS = "dados/perguntas_dfp.json"


# Arredonda os números e ordena as linhas, para comparar resultados sem depender da ordem.
def normalizar(linhas: list[tuple]) -> list[tuple]:
    def valor(v):
        return round(v, 2) if isinstance(v, float) else v
    return sorted((tuple(valor(v) for v in linha) for linha in linhas), key=repr)


# Diz se existe alguma ordem das colunas do resultado gerado que o deixe idêntico ao da referência.
def resultados_equivalentes(gerado: list[tuple], referencia: list[tuple]) -> bool:
    if len(gerado) != len(referencia):
        return False
    if not gerado:
        return True
    if len(gerado[0]) != len(referencia[0]):
        return False
    alvo = normalizar(referencia)
    for ordem in itertools.permutations(range(len(gerado[0]))):
        if normalizar([tuple(linha[i] for i in ordem) for linha in gerado]) == alvo:
            return True
    return False


# Diz se o resultado gerado traz todas as linhas da referência e, além delas, só outras companhias
# que também combinam com o nome do LIKE. A coluna DENOM_CIA é a única coluna extra aceita.
def resultado_contem_referencia(colunas: list[str], gerado: list[tuple], referencia: list[tuple]) -> bool:
    if "DENOM_CIA" not in colunas or not gerado or not referencia:
        return False
    posicao = colunas.index("DENOM_CIA")
    sem_nome = [tuple(v for i, v in enumerate(linha) if i != posicao) for linha in gerado]
    for candidato in (gerado, sem_nome):
        if len(candidato[0]) != len(referencia[0]):
            continue
        alvo = Counter(repr(linha) for linha in normalizar(referencia))
        for ordem in itertools.permutations(range(len(candidato[0]))):
            obtido = Counter(repr(linha) for linha in normalizar([tuple(l[i] for i in ordem) for l in candidato]))
            if not alvo - obtido:
                return True
    return False


# Formata o EX de um grupo de perguntas como "acertos/total".
def ex(lista: list[dict], chave: str = "ex") -> str:
    return f"{sum(x[chave] for x in lista)}/{len(lista)}" if lista else "-"


# Gera o SQL de cada pergunta, compara com a referência e imprime o EX e os erros.
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("perguntas", nargs="?", default=CAMINHO_PERGUNTAS)
    parser.add_argument("--k", type=int, default=K_TABELAS)
    parser.add_argument("--tabelas-ideais", action="store_true")
    args = parser.parse_args()
    with open(args.perguntas, encoding="utf-8") as f:
        perguntas = [p for p in json.load(f) if p["sql_referencia"]]

    configuracao = "tabelas ideais" if args.tabelas_ideais else f"busca com k={args.k}"
    print(f"=== {args.perguntas} ({configuracao}) ===")
    resultados = []
    for p in perguntas:
        inicio = time.time()
        tabelas = p["tabelas_esperadas"] if args.tabelas_ideais else None
        r = gerar_sql(p["pergunta"], args.k, tabelas)
        tempo = time.time() - inicio
        _, referencia = executar_sql(p["sql_referencia"], limite_linhas=None)
        acertou = r["status"] == "ok" and resultados_equivalentes(r["linhas"], referencia)
        aceito = acertou or (r["status"] == "ok" and resultado_contem_referencia(r["colunas"], r["linhas"], referencia))
        contexto_ok = set(p["tabelas_esperadas"]) <= set(r["tabelas_contexto"])
        resultados.append({"p": p, "r": r, "referencia": referencia, "ex": acertou, "ex_companhias": aceito,
                           "contexto_ok": contexto_ok})
        marca = "ok  " if acertou else ("ok+ " if aceito else "ERRO")
        print(f"  {marca} {p['id']} [{p['tipo']:12s}] status={r['status']:6s} "
              f"tentativas={len(r['tentativas'])} contexto_completo={'sim' if contexto_ok else 'NAO'} ({tempo:.0f}s)")

    print("\n=== Resumo ===")
    print(f"EX geral: {ex(resultados)} = {sum(x['ex'] for x in resultados) / len(resultados):.2f}")
    print(f"EX aceitando outras companhias que combinam com o nome (marcadas ok+): "
          f"{ex(resultados, 'ex_companhias')} = {sum(x['ex_companhias'] for x in resultados) / len(resultados):.2f}")
    print(f"  quantitativas: {ex([x for x in resultados if x['p']['tipo'] == 'quantitativa'])}")
    print(f"  híbridas:      {ex([x for x in resultados if x['p']['tipo'] == 'hibrida'])}")
    print(f"  com todas as tabelas no contexto: {ex([x for x in resultados if x['contexto_ok']])}")
    print(f"  com alguma tabela faltando no contexto: {ex([x for x in resultados if not x['contexto_ok']])}")
    tamanhos = [len(x["r"]["tabelas_contexto"]) for x in resultados]
    print(f"Tamanho médio do contexto: {sum(tamanhos) / len(tamanhos):.1f} tabelas")
    corrigidas = [x for x in resultados if len(x["r"]["tentativas"]) > 1]
    print(f"Perguntas que precisaram de nova tentativa: {len(corrigidas)}; "
          f"das quais acertaram: {sum(x['ex'] for x in corrigidas)}")

    erros = [x for x in resultados if not x["ex_companhias"]]
    if erros:
        print("\n=== Erros ===")
        for x in erros:
            p, r = x["p"], x["r"]
            print(f"\n{p['id']} {p['pergunta']}")
            print(f"  contexto: {r['tabelas_contexto']} (esperadas: {p['tabelas_esperadas']})")
            for i, t in enumerate(r["tentativas"], start=1):
                print(f"  tentativa {i}: {t['problema'] or 'executou'}\n    SQL: {t['sql']}")
            print(f"  referência: {p['sql_referencia']}")
            print(f"  resultado da referência: {x['referencia'][:3]}")
            print(f"  resultado gerado:        {r['linhas'][:3]}")


if __name__ == "__main__":
    main()
