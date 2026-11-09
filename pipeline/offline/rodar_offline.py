"""
Etapa: Offline.

Função: roda de uma vez os scripts que preparam o corpus e geram os índices de busca (esquema,
glossário e BM25), para um ou mais modelos de embeddings, na ordem certa.

Descrição geral do funcionamento:
- Confere, antes de começar, se todos os modelos necessários estão no cache local. Se faltar algum,
  o script para e mostra nome, fonte e tamanho; só baixa com --baixar-modelos.
- Primeiro prepara o corpus (os registros de esquema e de glossário, com id, fonte e conteudo), uma
  vez só: o esquema vem do esquema.json que já existe (sem banco nem LLM) e o glossário, dos PDFs
  (extrai, remove o texto riscado e fragmenta). O corpus é o mesmo para todos os modelos e para o BM25.
- Depois, para cada modelo pedido, gera os vetores do corpus de esquema e do de glossário. O modelo
  padrão grava nos arquivos versionados de embeddings/; os outros gravam em
  embeddings/modelos/<chave>/, pasta ignorada pelo git.
- No fim, gera os índices de palavras do BM25, que leem o corpus (não dependem dos vetores nem de
  nenhum modelo de embeddings).
- Não regera o esquema.json (isso usa o banco e o LLM): para isso, use
  pipeline/offline/gerar_indice_esquema.py sem --so-indice, que regera também o corpus e o índice
  do esquema.
- Cada etapa chama as mesmas funções dos scripts individuais, que continuam funcionando sozinhos.

Dependências:
- as dos quatro scripts chamados (gerar_corpus.py, gerar_indice_esquema.py,
  gerar_indice_glossario.py e gerar_indice_bm25.py), o esquema.json, os PDFs de dados/normativos/ e
  os modelos em cache local.

Parâmetros:
- --modelos CHAVE [CHAVE ...]: modelos de embeddings (padrão: só o e5-base). As chaves estão em
  pipeline/modelos_embedding.py.
- --todos-os-modelos: usa todos os modelos do registro.
- --etapas corpus esquema glossario bm25: roda só as etapas pedidas (padrão: as quatro). As etapas
  esquema e glossario leem o corpus, que precisa existir (rode a etapa corpus antes).
- --baixar-modelos: autoriza baixar os modelos que faltarem.

Resultado: o corpus e os índices gravados, e um resumo com o tempo de cada etapa.

Executar de dentro de rag-nl2sql-financas-br/:
    python pipeline/offline/rodar_offline.py
    python pipeline/offline/rodar_offline.py --modelos e5-base minilm
    python pipeline/offline/rodar_offline.py --todos-os-modelos --baixar-modelos
    python pipeline/offline/rodar_offline.py --etapas glossario
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modelos_embedding  # noqa: E402

import gerar_corpus  # noqa: E402
import gerar_indice_bm25  # noqa: E402
import gerar_indice_esquema  # noqa: E402
import gerar_indice_glossario  # noqa: E402

ETAPAS = ("corpus", "esquema", "glossario", "bm25")
RAIZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


# Roda uma etapa, imprime um cabeçalho e devolve o tempo gasto em segundos.
def rodar_etapa(nome: str, funcao) -> float:
    print(f"\n=== {nome} ===")
    inicio = time.perf_counter()
    funcao()
    return time.perf_counter() - inicio


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Prepara o corpus e gera os índices de esquema, glossário e BM25.")
    parser.add_argument("--modelos", nargs="+", choices=list(modelos_embedding.MODELOS), help="modelos de embeddings (padrão: e5-base)")
    parser.add_argument("--todos-os-modelos", action="store_true", help="usa todos os modelos do registro")
    parser.add_argument("--etapas", nargs="+", choices=ETAPAS, default=list(ETAPAS), help="etapas a rodar (padrão: todas)")
    parser.add_argument("--baixar-modelos", action="store_true", help="autoriza baixar os modelos que faltarem")
    args = parser.parse_args()

    if args.todos_os_modelos:
        modelos = list(modelos_embedding.MODELOS)
    else:
        modelos = args.modelos or [modelos_embedding.MODELO_PADRAO]
    modelos = list(dict.fromkeys(modelos))

    # Os caminhos dos scripts são relativos à raiz do repositório.
    os.chdir(RAIZ)

    # Uma única conferência no início, para não descobrir um modelo faltando depois de horas de trabalho.
    # O corpus e os vetores usam o tokenizador do modelo de fragmentação; os vetores, também o modelo pedido.
    precisam_vetores = [m for m in modelos if "esquema" in args.etapas or "glossario" in args.etapas]
    usa_tokenizador = any(e in args.etapas for e in ("corpus", "esquema", "glossario"))
    if precisam_vetores or usa_tokenizador:
        modelos_embedding.garantir_modelos(precisam_vetores + [modelos_embedding.MODELO_FRAGMENTACAO], args.baixar_modelos)

    tempos = []
    if "corpus" in args.etapas:
        tempos.append(("corpus", rodar_etapa("Corpus do esquema e do glossário", lambda: gerar_corpus.gerar_corpus(["esquema", "glossario"], args.baixar_modelos))))
    for chave in modelos:
        if "esquema" in args.etapas:
            tempos.append((f"esquema ({chave})", rodar_etapa(
                f"Índice de esquema, modelo {chave}",
                lambda c=chave: gerar_indice_esquema.gerar_so_indice(c, args.baixar_modelos))))
        if "glossario" in args.etapas:
            tempos.append((f"glossário ({chave})", rodar_etapa(
                f"Índice do glossário, modelo {chave}",
                lambda c=chave: gerar_indice_glossario.gerar_indice(c, args.baixar_modelos))))

    if "bm25" in args.etapas:
        tempos.append(("BM25", rodar_etapa("Índices de palavras do BM25", gerar_indice_bm25.main)))

    print("\n=== Resumo ===")
    for nome, segundos in tempos:
        print(f"  {nome}: {segundos:.1f} s")
    print(f"  total: {sum(s for _, s in tempos):.1f} s")


if __name__ == "__main__":
    main()
