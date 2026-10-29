"""
Etapa: Online (avaliação do passo de classificação de intenção).

Função: avalia o classificador de intenção com um arquivo de perguntas de teste.

Descrição geral do funcionamento:
- Compara a classe prevista por classificar() com o campo "tipo" de cada pergunta.
- Métricas: acurácia geral, acertos por classe e matriz de confusão.

Dependências:
- pipeline/online/classificador.py (servidor Ollama em execução, com o modelo qwen2.5-coder:7b);
- arquivo de perguntas em dados/ (por padrão, dados/perguntas_dfp.json).

Parâmetros: caminho opcional do arquivo de perguntas, como primeiro argumento.

Resultado: imprime a classe prevista de cada pergunta, as métricas, os erros e o
tempo médio por pergunta. Não grava arquivos.

Executar de dentro de rag-nl2sql-financas-br/:
    python pipeline/online/avaliar_classificador.py
    python pipeline/online/avaliar_classificador.py dados/perguntas_validacao.json
"""

import json
import sys
import time

from classificador import CLASSES, classificar

CAMINHO_PERGUNTAS = "dados/perguntas_dfp.json"


# Classifica cada pergunta e imprime a acurácia, a matriz de confusão e os erros.
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    caminho_perguntas = sys.argv[1] if len(sys.argv) > 1 else CAMINHO_PERGUNTAS
    with open(caminho_perguntas, encoding="utf-8") as f:
        perguntas = json.load(f)

    confusao = {real: {pred: 0 for pred in CLASSES} for real in CLASSES}
    erros, tempos = [], []
    print(f"=== {caminho_perguntas} ===")
    for p in perguntas:
        inicio = time.time()
        previsto = classificar(p["pergunta"])
        tempos.append(time.time() - inicio)
        confusao[p["tipo"]][previsto] += 1
        marca = "ok  " if previsto == p["tipo"] else "ERRO"
        print(f"  {marca} {p['id']} real={p['tipo']:12s} previsto={previsto:12s} ({tempos[-1]:.1f}s)")
        if previsto != p["tipo"]:
            erros.append((p["id"], p["pergunta"], p["tipo"], previsto))

    total = len(perguntas)
    acertos = sum(confusao[c][c] for c in CLASSES)
    print(f"\nAcurácia: {acertos}/{total} = {acertos / total:.2f}")
    for classe in CLASSES:
        n = sum(confusao[classe].values())
        print(f"  {classe:12s}: {confusao[classe][classe]}/{n}")

    print("\nMatriz de confusão (linha = real, coluna = previsto)")
    print("  " + " " * 12 + "".join(f"{c:>14s}" for c in CLASSES))
    for real in CLASSES:
        print("  " + f"{real:12s}" + "".join(f"{confusao[real][c]:14d}" for c in CLASSES))

    if erros:
        print("\nErros:")
        for id_, pergunta, real, previsto in erros:
            print(f"  {id_} (real {real}, previsto {previsto}): {pergunta}")
    print(f"\nTempo médio por pergunta: {sum(tempos) / len(tempos):.1f}s")


if __name__ == "__main__":
    main()
