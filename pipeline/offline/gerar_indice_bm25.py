"""
Etapa: Offline.

Função: constrói os índices de palavras (BM25) do esquema e do glossário, usados pela busca por
palavras.

Descrição geral do funcionamento:
- Lê o texto dos registros dos índices vetoriais do modelo padrão (embeddings/index_esquema.json
  e embeddings/index_glossario.json). Os registros são os mesmos para todos os modelos de
  embeddings, então um só índice de palavras serve a todos.
- Limpa o texto com as mesmas regras da busca (preprocessar, de busca_bm25.py): minúsculas,
  separação em palavras, sem stopwords do português, sem acentos, com e sem radicalização pelo
  stemmer Snowball do português.
- Grava, para cada índice, a lista de stopwords usada e, por registro, o id, a fonte, o texto e as
  palavras limpas (com e sem radicalização). Não há vetores nem modelo de embeddings. Assim, a
  busca não precisa montar o índice a cada partida e, sem radicalização, nem carrega o NLTK.
- Rode depois dos índices vetoriais: se eles mudarem (por exemplo, ao regerar o glossário),
  este índice também precisa ser regerado.

Dependências:
- embeddings/index_esquema.json e embeddings/index_glossario.json (gerados por pipeline/offline/);
- bibliotecas nltk e rank-bm25 (esta, porque busca_bm25.py a importa) e o corpus de stopwords do
  NLTK (python -m nltk.downloader stopwords);
- pipeline/online/busca_bm25.py (as regras de limpeza ficam lá, para o offline e a busca nunca
  divergirem).

Parâmetros: nenhum argumento.

Resultado: grava embeddings/index_bm25_esquema.json e embeddings/index_bm25_glossario.json, cada
um com "origem", "stemmer", "stopwords" e "registros" (id, fonte, conteudo, tokens e
tokens_radicalizados), um registro por linha.

Executar de dentro de rag-nl2sql-financas-br/:
    python pipeline/offline/gerar_indice_bm25.py
"""

import json
import os
import sys

# As regras de limpeza do texto ficam na busca; o offline usa as mesmas.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "online"))
import busca_bm25  # noqa: E402

# (índice vetorial de onde vem o texto, índice de palavras que será gravado)
INDICES = (
    ("embeddings/index_esquema.json", busca_bm25.CAMINHO_BM25_ESQUEMA),
    ("embeddings/index_glossario.json", busca_bm25.CAMINHO_BM25_GLOSSARIO),
)


# Monta o índice de palavras a partir do texto dos registros de um índice vetorial.
def montar_indice_de_palavras(caminho_vetorial: str, stopwords: set[str]) -> dict:
    with open(caminho_vetorial, encoding="utf-8") as f:
        registros = json.load(f)
    return {
        "origem": os.path.basename(caminho_vetorial),
        "stemmer": "Snowball do português (NLTK)",
        "stopwords": sorted(stopwords),
        "registros": [
            {
                "id": r["id"],
                "fonte": r["fonte"],
                "conteudo": r["conteudo"],
                "tokens": busca_bm25.preprocessar(r["conteudo"], False, stopwords),
                "tokens_radicalizados": busca_bm25.preprocessar(r["conteudo"], True, stopwords),
            }
            for r in registros
        ],
    }


# Grava o índice de palavras com um registro por linha, para o diff do git ficar legível.
def gravar_indice(indice: dict, caminho: str):
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    cabecalho = {k: indice[k] for k in ("origem", "stemmer", "stopwords")}
    linhas = ["    " + json.dumps(r, ensure_ascii=False) for r in indice["registros"]]
    with open(caminho, "w", encoding="utf-8") as f:
        f.write("{\n")
        for chave, valor in cabecalho.items():
            f.write(f'  "{chave}": {json.dumps(valor, ensure_ascii=False)},\n')
        f.write('  "registros": [\n' + ",\n".join(linhas) + "\n  ]\n}\n")


def main():
    try:
        stopwords = busca_bm25._carregar_stopwords()
    except RuntimeError as erro:
        print(erro)
        sys.exit(1)
    for caminho_vetorial, caminho_saida in INDICES:
        indice = montar_indice_de_palavras(caminho_vetorial, stopwords)
        gravar_indice(indice, caminho_saida)
        tamanho = os.path.getsize(caminho_saida) / 1e6
        print(f"{caminho_saida}: {len(indice['registros'])} registros, {len(stopwords)} stopwords, {tamanho:.1f} MB")


if __name__ == "__main__":
    main()
