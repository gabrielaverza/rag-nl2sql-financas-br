"""
Etapa: Online (avaliação do passo de busca).

Função: avalia a recuperação por embeddings com as perguntas de dados/perguntas_dfp.json.

Descrição geral do funcionamento:
- Usa o tipo correto de cada pergunta, sem passar pelo classificador de intenção.
- Índice de esquema (quantitativas e híbridas): itens = tabelas; relevantes = tabelas_esperadas.
- Índice de glossário (conceituais e híbridas): itens = trechos; relevante = trecho
  que contém a evidência da pergunta.
- Métricas: Recall@k (fração dos relevantes entre os k primeiros) e MRR (média do
  inverso da posição do primeiro relevante).

Dependências:
- pipeline/online/busca.py e os dois índices em embeddings/;
- dados/perguntas_dfp.json.

Parâmetros: nenhum argumento.

Resultado: imprime a posição dos itens relevantes de cada pergunta e as métricas
resumidas por tipo de pergunta. Não grava arquivos.

Executar de dentro de rag-nl2sql-financas-br/:
    python pipeline/online/avaliar_recuperacao.py
"""

import json
import re
import sys

from busca import CAMINHO_INDICE_ESQUEMA, CAMINHO_INDICE_GLOSSARIO, Indice, vetorizar

KS = (1, 3, 5)


# Troca qualquer sequência de espaços e quebras de linha por um espaço.
def normalizar(texto: str) -> str:
    return re.sub(r"\s+", " ", texto)


# Calcula Recall@k, reciprocal rank e posições a partir de uma lista de True/False por posição.
def metricas(ranking: list[bool], total_relevantes: int) -> dict:
    resultado = {f"recall@{k}": sum(ranking[:k]) / total_relevantes for k in KS}
    resultado["todos@3"] = sum(ranking[:3]) == total_relevantes
    resultado["todos@5"] = sum(ranking[:5]) == total_relevantes
    primeira = next((i for i, relevante in enumerate(ranking, start=1) if relevante), None)
    resultado["rr"] = 1 / primeira if primeira else 0.0
    resultado["posicoes"] = [i for i, relevante in enumerate(ranking, start=1) if relevante]
    return resultado


# Calcula a média de uma métrica sobre as perguntas.
def media(lista: list[dict], chave: str) -> float:
    return sum(x[chave] for x in lista) / len(lista)


# Imprime as métricas médias de um grupo de perguntas.
def resumo(nome: str, lista: list[dict], com_todos: bool):
    if not lista:
        return
    linha = " | ".join(f"Recall@{k}={media(lista, f'recall@{k}'):.2f}" for k in KS)
    linha += f" | MRR={media(lista, 'rr'):.2f}"
    if com_todos:
        linha += f" | todas as tabelas no top-3={media(lista, 'todos@3'):.2f}, top-5={media(lista, 'todos@5'):.2f}"
    print(f"  {nome} (n={len(lista)}): {linha}")


# Roda a busca para cada pergunta e imprime as posições e as métricas.
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    with open("dados/perguntas_dfp.json", encoding="utf-8") as f:
        perguntas = json.load(f)

    esquema = Indice(CAMINHO_INDICE_ESQUEMA)
    glossario = Indice(CAMINHO_INDICE_GLOSSARIO)
    total_tabelas = len({r["fonte"] for r in esquema.registros})

    tabelas, trechos = [], []
    print("=== Índice de esquema: tabelas esperadas (posição na busca) ===")
    for p in perguntas:
        vetor = vetorizar(p["pergunta"])
        if p["tabelas_esperadas"]:
            ranking = esquema.buscar_por_fonte(vetor, total_tabelas)
            nomes = [r["fonte"] for r in ranking]
            m = metricas([n in p["tabelas_esperadas"] for n in nomes], len(p["tabelas_esperadas"]))
            m.update(id=p["id"], tipo=p["tipo"])
            tabelas.append(m)
            posicoes = {t: nomes.index(t) + 1 for t in p["tabelas_esperadas"]}
            print(f"  {p['id']} {posicoes}   top-3: {nomes[:3]}")
        if p["evidencia"]:
            ranking = glossario.buscar(vetor, len(glossario.registros))
            relevantes = [any(normalizar(e) in normalizar(r["conteudo"]) for e in p["evidencia"]) for r in ranking]
            m = metricas(relevantes, sum(relevantes))
            m.update(id=p["id"], tipo=p["tipo"])
            trechos.append(m)

    print("\n=== Glossário: posição do trecho relevante ===")
    for m in trechos:
        print(f"  {m['id']} posição {m['posicoes']}")

    print("\n=== Resumo ===")
    print("Índice de esquema")
    resumo("todas", tabelas, True)
    resumo("quantitativas", [m for m in tabelas if m["tipo"] == "quantitativa"], True)
    resumo("híbridas", [m for m in tabelas if m["tipo"] == "hibrida"], True)
    print(f"  (o banco tem {total_tabelas} tabelas)")
    print("Índice do glossário")
    resumo("todas", trechos, False)
    resumo("conceituais", [m for m in trechos if m["tipo"] == "conceitual"], False)
    resumo("híbridas", [m for m in trechos if m["tipo"] == "hibrida"], False)
    print(f"  (o índice tem {len(glossario.registros)} trechos)")


if __name__ == "__main__":
    main()
