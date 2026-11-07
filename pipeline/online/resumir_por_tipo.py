"""
Etapa: Online (resumo das métricas de recuperação).

Função: resume, por tipo de pergunta, as métricas de recuperação que avaliar_baseline.py já gravou em
CSV, em contagens "x de n", para comparar os métodos sem reexecutar nenhuma busca.

Descrição geral do funcionamento:
- Lê os CSVs de metricas/ (validação e teste) e agrupa as perguntas por tipo (quantitativa, conceitual
  e híbrida).
- Para cada tipo, mostra os métodos lado a lado (E5-base, BM25, RRF, a política por tipo e o piso
  aleatório) nas métricas que fazem sentido para ele: tabelas nas quantitativas, trechos nas
  conceituais, e tabelas, trechos e os dois juntos nas híbridas.
- Cada medida de 0 ou 1 aparece como contagem de perguntas na validação, no teste e nos dois juntos;
  o MRR aparece como média dos dois conjuntos. No piso aleatório, a contagem é o valor esperado.

Dependências: metricas/baseline_recuperacao_perguntas_validacao.csv e
metricas/baseline_recuperacao_perguntas_dfp.csv (gerados por avaliar_baseline.py). Só biblioteca
padrão.

Parâmetros: nenhum argumento.

Resultado: grava metricas/resumo_por_tipo.md e imprime o mesmo texto.

Executar de dentro de rag-nl2sql-financas-br/:
    python pipeline/online/resumir_por_tipo.py
"""

import csv
import os
import sys
from datetime import datetime

PASTA = "metricas"
CSVS = (
    ("Val.", os.path.join(PASTA, "baseline_recuperacao_perguntas_validacao.csv")),
    ("Teste", os.path.join(PASTA, "baseline_recuperacao_perguntas_dfp.csv")),
)
METODOS = [
    ("E5-base (embeddings)", "E5-base"),
    ("BM25 sem stemmer", "BM25"),
    ("RRF (E5 + BM25)", "RRF"),
    ("Por índice (tabelas E5, trechos BM25)", "Por índice (E5 tabelas, BM25 trechos)"),
    ("Por tipo (híbrida com RRF)", "Política anterior (híbrida com RRF)"),
    ("Piso aleatório", "Piso aleatório"),
]
PISO = "Piso aleatório"
# (título, tipo, [(chave no CSV, rótulo, "contagem" ou "media")])
SECOES = [
    ("Perguntas quantitativas (SQL puro): tabelas", "quantitativa",
     [("tab_todos@3", "Todas as tabelas no top-3", "contagem"), ("tab_todos@5", "Todas as tabelas no top-5", "contagem"), ("tab_rr", "MRR", "media")]),
    ("Perguntas conceituais (glossário puro): trechos", "conceitual",
     [("glo_hit@3", "Hit@3", "contagem"), ("glo_hit@10", "Hit@10", "contagem"), ("glo_rr", "MRR", "media")]),
    ("Perguntas híbridas: tabelas", "hibrida",
     [("tab_todos@3", "Todas as tabelas no top-3", "contagem"), ("tab_todos@5", "Todas as tabelas no top-5", "contagem"), ("tab_rr", "MRR", "media")]),
    ("Perguntas híbridas: trechos", "hibrida",
     [("glo_hit@3", "Hit@3", "contagem"), ("glo_hit@10", "Hit@10", "contagem"), ("glo_rr", "MRR", "media")]),
    ("Perguntas híbridas: tabelas e trecho certos juntos", "hibrida",
     [("hib_conj@3", "Ambos no top-3", "contagem"), ("hib_conj@5", "Ambos no top-5", "contagem")]),
]


# Escreve um número com vírgula decimal.
def fmt(valor: float, casas: int = 2) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


# Lê os CSVs e devolve {rótulo do conjunto: linhas}.
def ler_csvs() -> dict:
    dados = {}
    for rotulo, caminho in CSVS:
        with open(caminho, encoding="utf-8", newline="") as f:
            dados[rotulo] = list(csv.DictReader(f))
    return dados


# Valores de uma métrica para um método e um tipo, em um conjunto.
def valores(linhas: list[dict], metodo: str, tipo: str, chave: str) -> list[float]:
    return [float(l[chave]) for l in linhas if l["metodo"] == metodo and l["tipo"] == tipo and l.get(chave) not in (None, "")]


# Texto "x/n" de uma contagem (no piso, o valor esperado com duas casas).
def contagem(vals: list[float], piso: bool) -> str:
    total = sum(vals)
    return f"{fmt(total) if piso else int(round(total))}/{len(vals)}"


# Monta a tabela de uma seção.
def tabela_secao(dados: dict, tipo: str, medidas: list) -> list[str]:
    cabecalho = ["Método"]
    for _, rotulo, forma in medidas:
        cabecalho += [f"{rotulo} ({c})" for c in (*dados, "Total")] if forma == "contagem" else [f"{rotulo} (média)"]
    linhas = ["| " + " | ".join(cabecalho) + " |", "|" + "|".join("---" for _ in cabecalho) + "|"]
    for metodo, curto in METODOS:
        celulas = [curto]
        for chave, _, forma in medidas:
            por_conjunto = [valores(l, metodo, tipo, chave) for l in dados.values()]
            todos = [v for vs in por_conjunto for v in vs]
            if forma == "contagem":
                celulas += [contagem(vs, metodo == PISO) for vs in por_conjunto] + [contagem(todos, metodo == PISO)]
            else:
                celulas.append(fmt(sum(todos) / len(todos)) if todos else "n/d")
        linhas.append("| " + " | ".join(celulas) + " |")
    return linhas + [""]


# Monta o relatório, imprime e grava.
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    dados = ler_csvs()
    t = ["# Resumo das métricas de recuperação por tipo de pergunta", ""]
    t += [
        f"- Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')} por `python pipeline/online/resumir_por_tipo.py`, a partir dos CSVs de `metricas/`.",
        "- Contagens \"x/n\": x perguntas com acerto entre n. Val. é o conjunto de validação (`perguntas_validacao.json`), Teste é `perguntas_dfp.json`, Total soma os dois. Os dois conjuntos são pequenos e não totalmente independentes do ajuste do E5-base (o teste o favorece mais).",
        "- Todas as tabelas no top-k: todas as tabelas necessárias estão entre as k escolhidas (mais as referenciadas por chave estrangeira). Hit@k: pelo menos um trecho relevante entre os k primeiros. Ambos: tabelas completas e trecho relevante ao mesmo tempo. MRR: média de 1/posição do primeiro item certo.",
        "- Métodos: BM25 sem stemmer com k1 = 2 e b = 0,5; RRF com k = 60; Por índice é a política padrão de `busca_roteada.py` (tabelas com E5 e trechos com BM25, para todos os tipos; escolhida depois de ver estes mesmos dados, então não é confirmação independente) e Política anterior é a que usava RRF nas híbridas, ambas com o tipo correto de cada pergunta. No piso aleatório, o número é o valor esperado de um sorteio.",
        "",
    ]
    for titulo, tipo, medidas in SECOES:
        t += [f"## {titulo}", ""] + tabela_secao(dados, tipo, medidas)
    texto = "\n".join(t)
    print(texto)
    with open(os.path.join(PASTA, "resumo_por_tipo.md"), "w", encoding="utf-8") as f:
        f.write(texto)


if __name__ == "__main__":
    main()
