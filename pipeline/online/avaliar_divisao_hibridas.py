"""
Etapa: Online (avaliação da divisão das perguntas híbridas).

Função: testa a hipótese de que, nas perguntas híbridas, separar a parte de dados da parte conceitual
melhora a recuperação: tabelas buscadas só com a parte de dados e trechos buscados só com a parte
conceitual, em comparação com a pergunta inteira nas duas buscas.

Descrição geral do funcionamento:
- Usa as 10 híbridas dos dois conjuntos (7 do teste e 3 da validação). As divisões estão em
  dados/perguntas_hibridas_divididas.json (id, parte_dados, parte_conceitual), escritas à mão, sem LLM:
  separação na conjunção da pergunta, com a anáfora trocada pelo substantivo e sem alterar as palavras
  do conceito. Foram escritas só a partir do texto das perguntas, antes de ver qualquer resultado.
  Como a divisão é manual, o teste mede o teto da ideia: uma divisão automática por LLM erraria mais.
- Métodos: E5-base, BM25 sem stemmer e RRF, com as mesmas premissas de avaliar_baseline.py.
- Para cada método, quatro condições: pergunta inteira nas duas buscas; pergunta dividida (parte de
  dados nas tabelas, parte conceitual nos trechos); mista (pergunta inteira nas tabelas, parte
  conceitual nos trechos), que junta o melhor de cada busca; e mista com o divisor automático
  (divisor.py, o LLM local), que é o que rodaria em tempo de execução, no lugar das divisões manuais.
- Métricas: tabelas completas no top-k, Hit@k dos trechos, MRR, nDCG@10 e acerto conjunto, em contagem;
  comparação pareada (dividida menos inteira) com bootstrap; posição do primeiro trecho relevante por
  pergunta.

Dependências:
- pipeline/online/avaliar_baseline.py (e os módulos que ele usa), os índices e o esquema em embeddings/;
- dados/perguntas_dfp.json, dados/perguntas_validacao.json e dados/perguntas_hibridas_divididas.json.

Parâmetros: nenhum argumento.

Resultado: imprime o relatório e grava metricas/divisao_hibridas.md e metricas/divisao_hibridas.csv.

Executar de dentro de rag-nl2sql-financas-br/:
    python pipeline/online/avaliar_divisao_hibridas.py
"""

import json
import os
import sys
import time
from datetime import datetime
from statistics import mean

import numpy as np

from avaliar_baseline import (
    NOME_BM25,
    NOME_E5,
    NOME_PISO,
    NOME_RRF,
    PASTA_SAIDA,
    SEMENTE,
    avaliar_pergunta,
    avaliar_piso,
    comparar,
    criar_metodos,
    fmt,
    gravar_csv,
    juntar_hibrida,
    linhas_contagens,
    linhas_medias,
    tabela_md,
)
from avaliar_recuperacao import normalizar
from busca import carregar_tabelas_referenciadas
from divisor import dividir

CAMINHOS_PERGUNTAS = ("dados/perguntas_dfp.json", "dados/perguntas_validacao.json")
CAMINHO_DIVISOES = "dados/perguntas_hibridas_divididas.json"
CAMINHO_DIVISOES_LLM = os.path.join(PASTA_SAIDA, "divisoes_geradas_pelo_llm.json")
CONDICAO_LLM = "mista (divisor do LLM)"
CONDICOES_COMPARADAS = ("dividida", "mista", CONDICAO_LLM)
METRICAS_COMPARADAS = [
    ("tab_todos@3", "Tabelas: todas no top-3"),
    ("tab_todos@5", "Tabelas: todas no top-5"),
    ("glo_hit@3", "Trechos: Hit@3"),
    ("glo_hit@5", "Trechos: Hit@5"),
    ("glo_hit@10", "Trechos: Hit@10"),
    ("glo_rr", "Trechos: MRR"),
    ("glo_ndcg@10", "Trechos: nDCG@10"),
    ("hib_conj@3", "Tabelas e trecho no top-3"),
    ("hib_conj@5", "Tabelas e trecho no top-5"),
]


# Avalia uma híbrida com a parte de dados nas tabelas e a parte conceitual nos trechos.
def avaliar_dividida(metodo, p: dict, divisao: dict, referenciadas: dict, n_tabelas: int, n_trechos: int, ids: set) -> dict:
    so_dados = {**p, "pergunta": divisao["parte_dados"], "evidencia": []}
    so_conceito = {**p, "pergunta": divisao["parte_conceitual"], "tabelas_esperadas": []}
    a = avaliar_pergunta(metodo, so_dados, referenciadas, n_tabelas, n_trechos, ids)
    b = avaliar_pergunta(metodo, so_conceito, referenciadas, n_tabelas, n_trechos, ids)
    m = {**a["m"], **b["m"]}
    juntar_hibrida(m)
    return {"id": p["id"], "tipo": p["tipo"], "m": m, "posicoes": {**a["posicoes"], **b["posicoes"]}}


# Combina as tabelas da pergunta inteira com os trechos da parte conceitual.
def montar_mista(inteira: list[dict], dividida: list[dict]) -> list[dict]:
    mista = []
    for a, b in zip(inteira, dividida):
        m = {c: v for c, v in a["m"].items() if c.startswith(("tab_", "tabsem_"))}
        m.update({c: v for c, v in b["m"].items() if c.startswith("glo_")})
        juntar_hibrida(m)
        posicoes = {"tabelas": a["posicoes"].get("tabelas"), "trechos": b["posicoes"].get("trechos")}
        mista.append({"id": a["id"], "tipo": a["tipo"], "m": m, "posicoes": posicoes})
    return mista


# Posição do primeiro trecho relevante de um resultado.
def primeira_posicao(resultado: dict) -> str:
    return str(min(resultado["posicoes"]["trechos"]))


# Roda o teste, imprime o relatório e grava os arquivos.
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    hibridas = []
    for caminho in CAMINHOS_PERGUNTAS:
        with open(caminho, encoding="utf-8") as f:
            hibridas += [p for p in json.load(f) if p["tipo"] == "hibrida"]
    with open(CAMINHO_DIVISOES, encoding="utf-8") as f:
        divisoes = {d["id"]: d for d in json.load(f)}
    faltando = [p["id"] for p in hibridas if p["id"] not in divisoes]
    if faltando:
        raise ValueError(f"sem divisão para: {faltando}")

    # O divisor automático roda uma vez por pergunta, sem ajuste do prompt depois de ver os resultados.
    print("Dividindo as perguntas com o LLM...", file=sys.stderr)
    divisoes_llm, tempos_llm = {}, []
    for p in hibridas:
        inicio = time.perf_counter()
        divisoes_llm[p["id"]] = dividir(p["pergunta"])
        tempos_llm.append(time.perf_counter() - inicio)
    with open(CAMINHO_DIVISOES_LLM, "w", encoding="utf-8") as f:
        json.dump(
            [{"id": p["id"], "pergunta": p["pergunta"], **divisoes_llm[p["id"]], "segundos": round(t, 2)}
             for p, t in zip(hibridas, tempos_llm)],
            f, ensure_ascii=False, indent=2,
        )

    metodos = {m.nome: m for m in criar_metodos()}
    escolhidos = [metodos[n] for n in (NOME_E5, NOME_BM25, NOME_RRF)]
    e5 = escolhidos[0]
    referenciadas = carregar_tabelas_referenciadas()
    n_tabelas = len({r["fonte"] for r in e5.esquema.registros})
    n_trechos = len(e5.glossario.registros)
    relevantes = {
        p["id"]: {
            r["id"] for r in e5.glossario.registros
            if any(normalizar(e) in normalizar(r["conteudo"]) for e in p["evidencia"])
        }
        for p in hibridas
    }

    resultados = {}
    for metodo in escolhidos:
        print(f"Avaliando {metodo.nome}...", file=sys.stderr)
        inteira = [
            avaliar_pergunta(metodo, p, referenciadas, n_tabelas, n_trechos, relevantes[p["id"]]) for p in hibridas
        ]
        dividida = [
            avaliar_dividida(metodo, p, divisoes[p["id"]], referenciadas, n_tabelas, n_trechos, relevantes[p["id"]])
            for p in hibridas
        ]
        resultados[f"{metodo.nome}, pergunta inteira"] = inteira
        resultados[f"{metodo.nome}, pergunta dividida"] = dividida
        resultados[f"{metodo.nome}, pergunta mista"] = montar_mista(inteira, dividida)
        dividida_llm = [
            avaliar_dividida(metodo, p, divisoes_llm[p["id"]], referenciadas, n_tabelas, n_trechos, relevantes[p["id"]])
            for p in hibridas
        ]
        resultados[f"{metodo.nome}, pergunta {CONDICAO_LLM}"] = montar_mista(inteira, dividida_llm)
    resultados[NOME_PISO] = [avaliar_piso(p, n_tabelas, n_trechos, len(relevantes[p["id"]])) for p in hibridas]

    n = len(hibridas)
    t = ["# Divisão das perguntas híbridas: dados nas tabelas e conceito nos trechos", ""]
    t += [
        f"- Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')} por `python pipeline/online/avaliar_divisao_hibridas.py`.",
        f"- {n} perguntas híbridas (7 do teste e 3 da validação). Divisões em `{CAMINHO_DIVISOES}`.",
        "- As divisões foram escritas à mão, só a partir do texto das perguntas e antes de ver qualquer resultado: separação na conjunção, anáfora trocada pelo substantivo, palavras do conceito intactas. Ainda não foram revisadas pela autora. Como são manuais, o teste mede o teto da ideia; uma divisão automática por LLM erraria mais.",
        "- Métodos e premissas: os de `avaliar_baseline.py` (E5-base; BM25 sem stemmer com k1 = 2 e b = 0,5; RRF com k = 60). Na condição dividida, a parte de dados vai às tabelas e a parte conceitual aos trechos; na inteira, a pergunta completa vai às duas; na mista, as tabelas usam a pergunta inteira e os trechos usam a parte conceitual.",
        "- Na condição mista com o divisor do LLM, a divisão vem do `divisor.py` (LLM local), que é o que rodaria em tempo de execução, e não das divisões manuais.",
        "",
    ]
    t += ["## 1. Contagem de perguntas com acerto", ""]
    chaves = ["tab_todos@3", "tab_todos@5", "glo_hit@3", "glo_hit@5", "glo_hit@10", "hib_conj@3", "hib_conj@5"]
    t += tabela_md(
        ["Condição", "Tabelas completas@3", "Tabelas completas@5", "Hit@3", "Hit@5", "Hit@10", "Ambos@3", "Ambos@5"],
        linhas_contagens(resultados, chaves),
    )
    t += ["## 2. Médias", ""]
    t += tabela_md(["Condição", "MRR tabelas", "MRR trechos", "nDCG@10 trechos"], linhas_medias(resultados, ["tab_rr", "glo_rr", "glo_ndcg@10"]))

    rng = np.random.default_rng(SEMENTE)
    linhas = []
    for metodo in escolhidos:
        inteira = resultados[f"{metodo.nome}, pergunta inteira"]
        for condicao in CONDICOES_COMPARADAS:
            comparada = resultados[f"{metodo.nome}, pergunta {condicao}"]
            for chave, descricao in METRICAS_COMPARADAS:
                c = comparar(comparada, inteira, chave, rng)
                if c:
                    linhas.append([f"{metodo.nome}, {condicao}", descricao, str(c["n"]), fmt(c["dif"]),
                                   f"[{fmt(c['ic'][0])}; {fmt(c['ic'][1])}]", f"{c['vitorias']}/{c['derrotas']}/{c['empates']}"])
    t += ["## 3. Comparação pareada (condição menos pergunta inteira)", ""]
    t += ["Diferença média, intervalo de 95% por bootstrap sobre as perguntas e vitórias, derrotas e empates da condição comparada (dividida ou mista).", ""]
    t += tabela_md(["Método e condição", "Métrica", "n", "Diferença média", "IC 95%", "Vitórias/derrotas/empates"], linhas)

    t += ["## 4. Posição do primeiro trecho relevante, por pergunta (entre 1.634 trechos)", ""]
    condicoes_posicao = ("inteira", "dividida", CONDICAO_LLM)
    cabecalho = ["Pergunta"] + [
        f"{m.nome.split(' (')[0]}, {'divisor do LLM' if cond == CONDICAO_LLM else cond}" for m in escolhidos for cond in condicoes_posicao
    ]
    corpo = []
    for i, p in enumerate(hibridas):
        linha = [p["id"]]
        for metodo in escolhidos:
            for cond in condicoes_posicao:
                linha.append(primeira_posicao(resultados[f"{metodo.nome}, pergunta {cond}"][i]))
        corpo.append(linha)
    t += tabela_md(cabecalho, corpo)

    usou = sum(1 for d in divisoes_llm.values() if d["dividida"])
    iguais = sum(1 for p in hibridas if normalizar(divisoes_llm[p["id"]]["parte_conceitual"]).lower() == normalizar(divisoes[p["id"]]["parte_conceitual"]).lower())
    t += ["## 5. Divisões feitas pelo LLM e as manuais", ""]
    t += [
        f"- O divisor (`divisor.py`, `qwen2.5-coder:7b`, uma execução) dividiu {usou} de {n} perguntas; nas demais, a divisão foi descartada e valeu a pergunta inteira. Tempo por pergunta: média de {fmt(mean(tempos_llm))} s, máximo de {fmt(max(tempos_llm))} s (inclui o carregamento do modelo na primeira chamada).",
        f"- A parte conceitual saiu igual à manual em {iguais} de {n} perguntas (ignorando maiúsculas). As divisões geradas estão em `{CAMINHO_DIVISOES_LLM}`.",
        "",
    ]
    corpo = []
    for p in hibridas:
        d = divisoes_llm[p["id"]]
        corpo.append([p["id"], divisoes[p["id"]]["parte_conceitual"], d["parte_conceitual"] if d["dividida"] else "(descartada: pergunta inteira)",
                      divisoes[p["id"]]["parte_dados"], d["parte_dados"] if d["dividida"] else "(descartada)"])
    t += tabela_md(["Pergunta", "Parte conceitual (manual)", "Parte conceitual (LLM)", "Parte de dados (manual)", "Parte de dados (LLM)"], corpo)

    t += [
        "## 6. Ressalvas",
        "",
        "- O divisor do LLM foi avaliado uma vez, sem ajustar o prompt depois de ver os resultados. O prompt foi escrito depois de conhecer a forma das perguntas híbridas (não os resultados da busca) e seus exemplos não são perguntas avaliadas.",
        f"- Só {n} perguntas: uma vale 10 pontos percentuais, e os intervalos de bootstrap refletem só a variação entre elas.",
        "- As divisões são do Claude, a partir do texto das perguntas, e ainda aguardam a revisão da autora; trocar uma divisão pode mudar o resultado.",
        "- A condição mista foi acrescentada depois de ver que usar só a parte de dados piora as tabelas; é uma combinação escolhida após ver os resultados das outras duas condições, e precisa ser confirmada em perguntas novas.",
        "- A divisão manual é um teto: no pipeline, separar a pergunta exigiria um passo novo (por exemplo, um planejador com LLM, como no MimirRAG e no kartheesan2026), que a Metodologia não prevê e que o modelo de 7B pode errar.",
        "- O critério de relevância é a evidência literal do trecho; trechos úteis sem essa frase contam como erro.",
        "- O pipeline oficial não foi alterado.",
        "",
    ]
    relatorio = "\n".join(t)
    print(relatorio)

    os.makedirs(PASTA_SAIDA, exist_ok=True)
    base = os.path.join(PASTA_SAIDA, "divisao_hibridas")
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(relatorio)
    gravar_csv(base + ".csv", "hibridas (teste e validação)", resultados)
    print(f"Arquivos gravados: {base}.md e {base}.csv", file=sys.stderr)


if __name__ == "__main__":
    main()
