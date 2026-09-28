"""
Pipeline offline: gera o índice conceitual (index_glossario.json)
a partir dos PDFs de data/normativos_ficticios/. Extrai o texto de cada PDF,
divide em trechos (chunks) e gera um embedding por trecho com
paraphrase-multilingual-MiniLM-L12-v2.

Executar de dentro de tcc-prototype/: python pipeline/gerar_indice_glossario.py
"""

import glob
import json
import os
import re
import unicodedata

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

PASTA_DOCUMENTOS = "data/normativos_ficticios"
CAMINHO_SAIDA = "embeddings/index_glossario.json"
MODELO_EMBEDDING = "paraphrase-multilingual-MiniLM-L12-v2"

# O MiniLM só lê até 128 tokens por texto (o resto é ignorado). Dois desses
# tokens são reservados pelo modelo (início e fim), então o trecho deve ter
# no máximo 126. Usamos 120 para ter uma folga.
TAMANHO_CHUNK_TOKENS = 120
# Sobreposição de cerca de 15% entre trechos vizinhos, para não cortar
# uma definição exatamente no limite entre dois trechos.
SOBREPOSICAO_TOKENS = 18


def _continua_paragrafo(anterior: str, atual: str) -> bool:
    # Quando uma quebra de página corta um parágrafo ao meio, o bloco anterior
    # termina sem pontuação final e o bloco seguinte começa em letra minúscula
    # ou parêntese. Títulos e parágrafos novos começam em maiúscula ou número,
    # então não são confundidos com uma continuação.
    terminou_frase = anterior.endswith((".", "!", "?", ":", ";"))
    return not terminou_frase and (atual[0].islower() or atual[0] in "(,")


def extrair_texto_pdf(caminho_pdf: str) -> str:
    # Lê o PDF por blocos. Cada bloco é um parágrafo, e os parágrafos são
    # unidos com uma linha em branco. O pymupdf, por padrão, não devolve
    # essa linha em branco, e o divisor de texto precisa dela para respeitar
    # os limites de parágrafo.
    documento = pymupdf.open(caminho_pdf)
    paragrafos = []
    for pagina in documento:
        primeiro_bloco_da_pagina = True
        for bloco in pagina.get_text("blocks", sort=True):
            texto, tipo_bloco = bloco[4], bloco[6]
            if tipo_bloco != 0:  # 0 = texto; outros tipos são imagens
                continue
            # NFKC desfaz ligaturas (ex.: "ﬁ" vira "fi"), que atrapalhariam a busca.
            texto = unicodedata.normalize("NFKC", texto)
            # Junta as linhas quebradas do parágrafo em uma linha só.
            texto = re.sub(r"\s+", " ", texto).strip()
            if not texto:
                continue
            # Se o parágrafo foi cortado pela quebra de página, recoloca as duas
            # metades juntas, para que a definição não seja dividida no meio.
            if primeiro_bloco_da_pagina and paragrafos and _continua_paragrafo(paragrafos[-1], texto):
                paragrafos[-1] += " " + texto
            else:
                paragrafos.append(texto)
            primeiro_bloco_da_pagina = False
    documento.close()
    return "\n\n".join(paragrafos)


def criar_divisor(modelo_embedding: SentenceTransformer) -> RecursiveCharacterTextSplitter:
    # O tamanho do trecho é medido em tokens do próprio MiniLM (e não em
    # caracteres), para garantir que o modelo leia o trecho inteiro.
    tokenizador = modelo_embedding.tokenizer
    return RecursiveCharacterTextSplitter(
        chunk_size=TAMANHO_CHUNK_TOKENS,
        chunk_overlap=SOBREPOSICAO_TOKENS,
        length_function=lambda texto: len(tokenizador.tokenize(texto)),
        # Ordem de preferência para quebrar: parágrafo, linha, frase, palavra.
        # Só quebra em um nível menor se o trecho ainda estiver grande demais.
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def montar_registros(nome_arquivo: str, trechos: list[str], modelo_embedding: SentenceTransformer) -> list[dict]:
    # Gera um embedding para cada trecho e monta o registro no formato do índice.
    nome_base = os.path.splitext(nome_arquivo)[0]
    vetores = modelo_embedding.encode(trechos)
    registros = []
    for numero, (trecho, vetor) in enumerate(zip(trechos, vetores), start=1):
        registros.append({
            "id": f"glossario_{nome_base}_{numero:03d}",
            "fonte": nome_arquivo,
            "conteudo": trecho,
            "embedding": vetor.tolist(),
        })
    return registros


def main():
    modelo_embedding = SentenceTransformer(MODELO_EMBEDDING)
    divisor = criar_divisor(modelo_embedding)
    limite_modelo = modelo_embedding.max_seq_length - 2

    caminhos_pdf = sorted(glob.glob(os.path.join(PASTA_DOCUMENTOS, "*.pdf")))
    if not caminhos_pdf:
        raise FileNotFoundError(f"Nenhum PDF encontrado em {PASTA_DOCUMENTOS}")

    todos_registros = []
    for caminho_pdf in caminhos_pdf:
        nome_arquivo = os.path.basename(caminho_pdf)
        texto = extrair_texto_pdf(caminho_pdf)
        trechos = divisor.split_text(texto)

        # Confere se algum trecho passou do limite de leitura do modelo.
        tamanhos = [len(modelo_embedding.tokenizer.tokenize(t)) for t in trechos]
        if max(tamanhos) > limite_modelo:
            print(f"ATENÇÃO: trecho com {max(tamanhos)} tokens em {nome_arquivo} (limite {limite_modelo})")

        registros = montar_registros(nome_arquivo, trechos, modelo_embedding)
        todos_registros.extend(registros)
        print(f"{nome_arquivo}: {len(registros)} trechos (maior trecho: {max(tamanhos)} tokens)")

    os.makedirs(os.path.dirname(CAMINHO_SAIDA), exist_ok=True)
    with open(CAMINHO_SAIDA, "w", encoding="utf-8") as f:
        json.dump(todos_registros, f, ensure_ascii=False, indent=2)

    print(f"Índice salvo em {CAMINHO_SAIDA} ({len(todos_registros)} registros no total)")


if __name__ == "__main__":
    main()
