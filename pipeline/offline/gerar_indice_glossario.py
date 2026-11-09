"""
Etapa: Offline.

Função: constrói o índice vetorial conceitual usado pelo estágio online.

Descrição geral do funcionamento:
- Lê o corpus do glossário (embeddings/corpus_glossario.json, gerado por gerar_corpus.py), que tem
  os trechos dos normativos já extraídos dos PDFs e divididos em até 120 tokens.
- O modelo de embeddings escolhido (opção --modelo, padrão e5-base) gera um vetor por trecho,
  com o prefixo de trecho da ficha dele. Os trechos são os mesmos qualquer que seja o modelo.
- Antes de tudo, confere se os modelos necessários estão no cache local; só baixa os que
  faltam, e só com --baixar-modelos.

Dependências:
- embeddings/corpus_glossario.json (gerado por pipeline/offline/gerar_corpus.py);
- bibliotecas transformers e sentence-transformers;
- pipeline/modelos_embedding.py e os modelos em cache local.

Parâmetros: --modelo CHAVE (padrão e5-base; as chaves estão em pipeline/modelos_embedding.py) e
--baixar-modelos (opcional, autoriza o download dos modelos que faltarem).

Resultado: um índice com um registro por trecho, no formato id/fonte/conteudo/embedding, com a
fonte igual ao nome do PDF de origem. Com o modelo padrão, grava embeddings/index_glossario.json
(versionado); com outro modelo, grava embeddings/modelos/<chave>/index_glossario.json (pasta
ignorada pelo git).

Executar de dentro de rag-nl2sql-financas-br/:
    python pipeline/offline/gerar_indice_glossario.py
    python pipeline/offline/gerar_indice_glossario.py --modelo minilm
"""

import argparse
import json
import os
import sys

# Os módulos compartilhados ficam em pipeline/, fora da pasta deste script.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modelos_embedding  # noqa: E402

CAMINHO_CORPUS = "embeddings/corpus_glossario.json"
CAMINHO_SAIDA_PADRAO = "embeddings/index_glossario.json"
PASTA_MODELOS = "embeddings/modelos"


# Devolve onde o índice do modelo é gravado: o caminho versionado para o padrão, uma pasta ignorada para os outros.
def caminho_de_saida(chave_modelo: str) -> str:
    if chave_modelo == modelos_embedding.MODELO_PADRAO:
        return CAMINHO_SAIDA_PADRAO
    return os.path.join(PASTA_MODELOS, chave_modelo, "index_glossario.json")


# Gera o índice vetorial a partir do corpus do glossário.
def gerar_indice(chave_modelo: str, baixar_modelos: bool, caminho_saida: str | None = None) -> str:
    # O início de toda execução confere o cache; sem --baixar-modelos, só avisa o que falta.
    modelos_embedding.garantir_modelos([chave_modelo, modelos_embedding.MODELO_FRAGMENTACAO], baixar_modelos)
    with open(CAMINHO_CORPUS, encoding="utf-8") as f:
        corpus = json.load(f)
    registros = modelos_embedding.registros_com_vetores(corpus, chave_modelo)

    caminho_saida = caminho_saida or caminho_de_saida(chave_modelo)
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    with open(caminho_saida, "w", encoding="utf-8") as f:
        json.dump(registros, f, ensure_ascii=False, indent=2)

    print(f"Índice salvo em {caminho_saida} ({len(registros)} registros, a partir de {CAMINHO_CORPUS}, modelo {chave_modelo})")
    return caminho_saida


def main():
    parser = argparse.ArgumentParser(description="Gera o índice do glossário (normativos CVM e CPC).")
    parser.add_argument("--modelo", default=modelos_embedding.MODELO_PADRAO, choices=list(modelos_embedding.MODELOS))
    parser.add_argument("--baixar-modelos", action="store_true", help="autoriza baixar os modelos que faltarem")
    args = parser.parse_args()
    gerar_indice(args.modelo, args.baixar_modelos)


if __name__ == "__main__":
    main()
