"""
Etapa: Offline.

Função: constrói o índice conceitual usado pelo estágio online.

Descrição geral do funcionamento:
- Extrai o texto dos PDFs normativos (Resolução CVM 80 e CPCs), removendo
  cabeçalhos e rodapés repetidos.
- Divide o texto em trechos de até 120 tokens, com sobreposição de cerca de 15%.
- O paraphrase-multilingual-MiniLM-L12-v2 gera um embedding por trecho.

Dependências:
- PDFs em dados/normativos/ (inclui as subpastas cvm/ e cpc/);
- bibliotecas pymupdf, langchain-text-splitters e sentence-transformers;
- modelo MiniLM já em cache local (carregado com local_files_only).

Parâmetros: nenhum argumento. O comportamento é definido pelas constantes do arquivo.

Resultado: gera embeddings/index_glossario.json, com um registro por trecho no
formato id/fonte/conteudo/embedding, com a fonte igual ao nome do PDF de origem.

Executar de dentro de rag-nl2sql-financas-br/:
    python pipeline/offline/gerar_indice_glossario.py
"""

import glob
import json
import os
import re
import unicodedata
from collections import Counter

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

PASTA_DOCUMENTOS = "dados/normativos"
CAMINHO_SAIDA = "embeddings/index_glossario.json"
MODELO_EMBEDDING = "paraphrase-multilingual-MiniLM-L12-v2"

# Um bloco que aparece em mais dessa fração das páginas é cabeçalho ou rodapé.
LIMIAR_REPETICAO = 0.5

# Blocos até esse tamanho também são agrupados por assinatura (sem pontuação e
# espaços), para pegar variantes do mesmo rodapé, como "CPC_09R1" e "CPC 09(R1)".
TAMANHO_MAX_BLOCO_CURTO = 30

# O MiniLM lê no máximo 128 tokens por texto, e 2 são reservados pelo modelo.
# Usamos 120 para ter uma folga.
TAMANHO_CHUNK_TOKENS = 120

# Sobreposição de cerca de 15% entre trechos vizinhos, para não cortar uma
# definição exatamente no limite entre dois trechos.
SOBREPOSICAO_TOKENS = 18


# Diz se o bloco atual continua o parágrafo anterior, cortado por uma quebra de página.
def _continua_paragrafo(anterior: str, atual: str) -> bool:
    # Continuação: o anterior termina sem pontuação final e o atual começa em
    # minúscula ou parêntese. Títulos e parágrafos novos começam em maiúscula ou número.
    terminou_frase = anterior.endswith((".", "!", "?", ":", ";"))
    return not terminou_frase and (atual[0].islower() or atual[0] in "(,")


# Devolve o texto do bloco normalizado, ou vazio se não for conteúdo.
def _texto_normalizado(bloco) -> str:
    texto, tipo_bloco = bloco[4], bloco[6]
    # Tipo 0 é texto; os outros são imagens.
    if tipo_bloco != 0:
        return ""
    # NFKC desfaz ligaturas (ex.: "ﬁ" vira "fi"), que atrapalhariam a busca.
    texto = unicodedata.normalize("NFKC", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    # Número de página isolado não é conteúdo.
    if texto.isdigit():
        return ""
    return texto


# Devolve só as letras e dígitos do texto, em maiúsculas.
def _assinatura(texto: str) -> str:
    return re.sub(r"[^0-9A-Za-zÀ-ÿ]", "", texto).upper()


# Devolve os blocos que se repetem em muitas páginas (cabeçalhos e rodapés).
def detectar_blocos_repetidos(documento: pymupdf.Document) -> set[str]:
    # A capa costuma ter uma variante do cabeçalho com quebras de linha
    # diferentes, que não é pega aqui; é aceitável, pois é uma página só.
    contagem = Counter()
    for pagina in documento:
        blocos_da_pagina = {_texto_normalizado(b) for b in pagina.get_text("blocks", sort=True)}
        contagem.update(b for b in blocos_da_pagina if b)
    minimo = max(2, int(len(documento) * LIMIAR_REPETICAO))

    repetidos = {texto for texto, n in contagem.items() if n >= minimo}

    # Para blocos curtos, soma as variantes de mesma assinatura: se o total
    # bater o limiar, todas viram rodapé, mesmo que nenhuma sozinha bastasse.
    contagem_por_assinatura = Counter()
    variantes_por_assinatura: dict[str, set[str]] = {}
    for texto, n in contagem.items():
        if len(texto) > TAMANHO_MAX_BLOCO_CURTO:
            continue
        assinatura = _assinatura(texto)
        contagem_por_assinatura[assinatura] += n
        variantes_por_assinatura.setdefault(assinatura, set()).add(texto)
    for assinatura, total in contagem_por_assinatura.items():
        if total >= minimo:
            repetidos |= variantes_por_assinatura[assinatura]

    return repetidos


# Extrai o texto do PDF como parágrafos separados por linha em branco.
def extrair_texto_pdf(caminho_pdf: str) -> str:
    # O divisor de texto usa a linha em branco para respeitar os limites de
    # parágrafo, e o pymupdf não a devolve por padrão.
    documento = pymupdf.open(caminho_pdf)
    blocos_repetidos = detectar_blocos_repetidos(documento)
    paragrafos = []
    for pagina in documento:
        primeiro_bloco_da_pagina = True
        for bloco in pagina.get_text("blocks", sort=True):
            texto = _texto_normalizado(bloco)
            if not texto or texto in blocos_repetidos:
                continue
            # Junta as duas metades de um parágrafo cortado pela quebra de página.
            if primeiro_bloco_da_pagina and paragrafos and _continua_paragrafo(paragrafos[-1], texto):
                paragrafos[-1] += " " + texto
            else:
                paragrafos.append(texto)
            primeiro_bloco_da_pagina = False
    documento.close()
    return "\n\n".join(paragrafos)


# Cria o divisor de texto, que mede o tamanho dos trechos em tokens do MiniLM.
def criar_divisor(modelo_embedding: SentenceTransformer) -> RecursiveCharacterTextSplitter:
    tokenizador = modelo_embedding.tokenizer
    return RecursiveCharacterTextSplitter(
        chunk_size=TAMANHO_CHUNK_TOKENS,
        chunk_overlap=SOBREPOSICAO_TOKENS,
        length_function=lambda texto: len(tokenizador.tokenize(texto)),
        # Só quebra em um nível menor se o trecho ainda estiver grande demais.
        separators=["\n\n", "\n", ". ", " ", ""],
    )


# Diz se o trecho é só um número solto de célula de tabela, sem valor para a busca.
def eh_fragmento_degenerado(trecho: str) -> bool:
    nucleo = trecho.strip(" .\n")
    return nucleo.isdigit() and len(nucleo) <= 4


# Gera um embedding por trecho e monta os registros do índice.
def montar_registros(nome_arquivo: str, trechos: list[str], modelo_embedding: SentenceTransformer) -> list[dict]:
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


# Gera o índice conceitual a partir de todos os PDFs de dados/normativos/.
def main():
    # local_files_only evita depender da rede a cada carga do modelo.
    modelo_embedding = SentenceTransformer(MODELO_EMBEDDING, local_files_only=True)
    divisor = criar_divisor(modelo_embedding)
    limite_modelo = modelo_embedding.max_seq_length - 2

    caminhos_pdf = sorted(glob.glob(os.path.join(PASTA_DOCUMENTOS, "**", "*.pdf"), recursive=True))
    if not caminhos_pdf:
        raise FileNotFoundError(f"Nenhum PDF encontrado em {PASTA_DOCUMENTOS}")

    todos_registros = []
    for caminho_pdf in caminhos_pdf:
        nome_arquivo = os.path.basename(caminho_pdf)
        texto = extrair_texto_pdf(caminho_pdf)
        trechos = divisor.split_text(texto)

        antes = len(trechos)
        trechos = [t for t in trechos if not eh_fragmento_degenerado(t)]
        if len(trechos) < antes:
            print(f"    ({antes - len(trechos)} fragmento(s) degenerado(s) descartado(s))")

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
