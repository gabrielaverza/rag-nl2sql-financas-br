"""
Etapa: Online (avaliação do passo de busca).

Função: avalia a recuperação por embeddings com um arquivo de perguntas de teste.

Descrição geral do funcionamento:
- Usa o tipo correto de cada pergunta, sem passar pelo classificador de intenção.
- Índice de esquema (quantitativas e híbridas): itens = tabelas; relevantes = tabelas_esperadas.
  Para cada k, a lista avaliada tem as k tabelas mais parecidas mais as tabelas que elas
  referenciam por chave estrangeira.
- Índice de glossário (conceituais e híbridas): itens = trechos; relevante = trecho
  que contém a evidência da pergunta.
- Métricas: Recall@k (fração dos relevantes entre os k primeiros) e MRR (média do
  inverso da posição do primeiro relevante).

Dependências:
- pipeline/online/busca.py e os índices e o esquema em embeddings/;
- arquivo de perguntas em dados/ (por padrão, dados/perguntas_dfp.json).

Parâmetros: caminho opcional do arquivo de perguntas, como primeiro argumento.

Resultado: imprime a posição dos itens relevantes de cada pergunta e as métricas
resumidas por tipo de pergunta. Não grava arquivos.

Executar de dentro de rag-nl2sql-financas-br/:
    python pipeline/online/avaliar_recuperacao.py
    python pipeline/online/avaliar_recuperacao.py dados/perguntas_validacao.json
"""

import json
import re
import sys

from busca import (
    CAMINHO_INDICE_ESQUEMA,
    CAMINHO_INDICE_GLOSSARIO,
    Indice,
    buscar_tabelas,
    carregar_tabelas_referenciadas,
    vetorizar,
)

KS = (1, 3, 5)
CAMINHO_PERGUNTAS = "dados/perguntas_dfp.json"


# Troca qualquer sequência de espaços e quebras de linha por um espaço.
def normalizar(texto: str) -> str:
    return re.sub(r"\s+", " ", texto)


# Calcula Recall@k, reciprocal rank e posições a partir de uma lista de True/False por posição.
def metricas(ranking: list[bool], total_relevantes: int) -> dict:
    resultado = {f"recall@{k}": sum(ranking[:k]) / total_relevantes for k in KS}
    resultado["rr"] = 1 / next((i for i, relevante in enumerate(ranking, start=1) if relevante), float("inf"))
    resultado["posicoes"] = [i for i, relevante in enumerate(ranking, start=1) if relevante]
    return resultado


# Calcula as métricas das tabelas, contando as tabelas referenciadas que acompanham as k escolhidas.
def metricas_tabelas(esquema: Indice, vetor, esperadas: list[str], referenciadas: dict[str, list[str]]) -> dict:
    resultado = {}
    for k in KS:
        lista = buscar_tabelas(esquema, vetor, k, referenciadas)
        achadas = len(set(esperadas) & set(lista))
        resultado[f"recall@{k}"] = achadas / len(esperadas)
        resultado[f"todos@{k}"] = achadas == len(esperadas)
        resultado[f"tamanho@{k}"] = len(lista)
    nomes = [r["fonte"] for r in esquema.buscar_por_fonte(vetor, len(esquema.registros))]
    primeira = min(nomes.index(t) for t in esperadas) + 1
    resultado["rr"] = 1 / primeira
    resultado["posicoes"] = {t: nomes.index(t) + 1 for t in esperadas}
    resultado["top3"] = nomes[:3]
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
        linha += " | todas as tabelas em " + ", ".join(f"top-{k}={media(lista, f'todos@{k}'):.2f}" for k in KS)
        linha += " | tamanho médio da lista " + "/".join(f"{media(lista, f'tamanho@{k}'):.1f}" for k in KS)
    print(f"  {nome} (n={len(lista)}): {linha}")


# Roda a busca para cada pergunta e imprime as posições e as métricas.
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    caminho_perguntas = sys.argv[1] if len(sys.argv) > 1 else CAMINHO_PERGUNTAS
    with open(caminho_perguntas, encoding="utf-8") as f:
        perguntas = json.load(f)

    esquema = Indice(CAMINHO_INDICE_ESQUEMA)
    glossario = Indice(CAMINHO_INDICE_GLOSSARIO)
    referenciadas = carregar_tabelas_referenciadas()
    total_tabelas = len({r["fonte"] for r in esquema.registros})

    tabelas, trechos = [], []
    print(f"=== {caminho_perguntas} ===")
    print("=== Índice de esquema: posição das tabelas esperadas na busca ===")
    for p in perguntas:
        vetor = vetorizar(p["pergunta"])
        if p["tabelas_esperadas"]:
            m = metricas_tabelas(esquema, vetor, p["tabelas_esperadas"], referenciadas)
            m.update(id=p["id"], tipo=p["tipo"])
            tabelas.append(m)
            print(f"  {p['id']} {m['posicoes']}   top-3: {m['top3']}")
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
