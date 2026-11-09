"""
Etapa: Offline.

Função: prepara o corpus de busca, isto é, os registros (texto e origem) que os dois métodos de busca
indexam: os vetores de embeddings e o BM25. Assim os dois métodos partem sempre das mesmas unidades de
busca, e o BM25 deixa de depender dos arquivos de vetores.

Descrição geral do funcionamento:
- Corpus do glossário: extrai o texto dos PDFs normativos (Resolução CVM 80 e CPCs), removendo
  cabeçalhos e rodapés repetidos e as linhas riscadas. Nos PDFs consolidados, o texto riscado é a
  redação antiga, já substituída por uma redação nova; ele sai do corpus para não ser recuperado como
  norma vigente. Divide o texto em trechos de até 120 tokens, com sobreposição de cerca de 15%.
- Corpus do esquema: lê embeddings/esquema.json e escreve, para cada tabela, os textos de cada
  aspecto (resumo e propósito, entidades, relações, colunas e contas), cada um com até 120 tokens.
- O tamanho é sempre medido com o tokenizador do e5-base (MODELO_FRAGMENTACAO, em
  modelos_embedding.py), para o corpus ser o mesmo qualquer que seja o modelo de embeddings.
- Cada registro tem só id, fonte e conteudo (a fonte é o nome da tabela ou do PDF). Não há vetores:
  eles são gerados depois, por modelo, por gerar_indice_esquema.py e gerar_indice_glossario.py, e o
  índice de palavras do BM25 por gerar_indice_bm25.py.

Dependências:
- PDFs em dados/normativos/ (inclui as subpastas cvm/ e cpc/) e embeddings/esquema.json;
- bibliotecas pymupdf, langchain-text-splitters e transformers (só o tokenizador);
- pipeline/modelos_embedding.py e o tokenizador do e5-base no cache local (só baixa com
  --baixar-modelos).

Parâmetros: --partes esquema glossario (padrão: as duas) e --baixar-modelos (opcional, autoriza o
download do que faltar).

Resultado: embeddings/corpus_esquema.json e embeddings/corpus_glossario.json (versionados), listas de
registros no formato id/fonte/conteudo.

Executar de dentro de rag-nl2sql-financas-br/:
    python pipeline/offline/gerar_corpus.py
    python pipeline/offline/gerar_corpus.py --partes glossario
"""

import argparse
import glob
import json
import os
import re
import sys
import unicodedata
from collections import Counter

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Os módulos compartilhados ficam em pipeline/, fora da pasta deste script.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import modelos_embedding  # noqa: E402

PASTA_DOCUMENTOS = "dados/normativos"
CAMINHO_ESQUEMA = "embeddings/esquema.json"
CAMINHO_CORPUS_ESQUEMA = "embeddings/corpus_esquema.json"
CAMINHO_CORPUS_GLOSSARIO = "embeddings/corpus_glossario.json"

# Tamanho máximo, em tokens do tokenizador do e5-base, de cada registro do esquema.
LIMITE_TOKENS = 120

# Um bloco que aparece em mais dessa fração das páginas é cabeçalho ou rodapé.
LIMIAR_REPETICAO = 0.5

# Blocos até esse tamanho também são agrupados por assinatura (sem pontuação e
# espaços), para pegar variantes do mesmo rodapé, como "CPC_09R1" e "CPC 09(R1)".
TAMANHO_MAX_BLOCO_CURTO = 30

# O modelo aceita até 512 tokens, mas trechos maiores recuperaram pior nos
# testes (250 e 400 tokens contra 120), então o tamanho fica em 120.
TAMANHO_CHUNK_TOKENS = 120

# Sobreposição de cerca de 15% entre trechos vizinhos, para não cortar uma
# definição exatamente no limite entre dois trechos.
SOBREPOSICAO_TOKENS = 18

# Uma linha é considerada riscada quando traços finos, desenhados na altura do meio do texto,
# cobrem pelo menos essa fração da largura dela. Traços na base da linha são sublinhados, e
# traços grossos são bordas de tabela; nenhum dos dois conta.
COBERTURA_MINIMA_RISCO = 0.6
ALTURA_MAXIMA_TRACO = 1.6
LARGURA_MINIMA_TRACO = 8
FAIXA_DO_MEIO = (0.35, 0.65)


# ---------------------------------------------------------------------------
# 1) Corpus do esquema
# ---------------------------------------------------------------------------

# Junta itens em textos de até LIMITE_TOKENS tokens, cada um começando pelo prefixo.
def empacotar(prefixo: str, itens: list[str], separador: str, contar) -> list[str]:
    textos, atual = [], []
    for item in itens:
        candidato = prefixo + separador.join(atual + [item])
        if atual and contar(candidato) > LIMITE_TOKENS:
            textos.append(prefixo + separador.join(atual))
            atual = [item]
        else:
            atual.append(item)
    if atual:
        textos.append(prefixo + separador.join(atual))
    return textos


# Tira palavras do fim do texto até ele caber no limite de tokens.
def cortar_no_limite(texto: str, contar) -> str:
    palavras = texto.split(" ")
    while contar(" ".join(palavras)) > LIMITE_TOKENS and len(palavras) > 1:
        palavras.pop()
    return " ".join(palavras)


# Gera os textos de uma tabela, um para cada aspecto da descrição.
def textos_da_tabela(entrada: dict, contar) -> list[str]:
    t = entrada["tabela"]
    textos = []

    itens = [x for x in (entrada["resumo"], entrada["proposito"]) if x]
    textos += empacotar(f"Tabela {t}. ", itens, " ", contar) if itens else []

    if entrada["entidades"]:
        textos += empacotar(f"Tabela {t}. Entidades: ", entrada["entidades"], ", ", contar)

    frases = []
    for r in entrada["chaves_de_juncao"]:
        if r["sentido"] == "referencia":
            frases.append(f"{t}.{r['coluna']} referencia {r['tabela_relacionada']}.{r['coluna_relacionada']}")
        else:
            frases.append(f"{r['tabela_relacionada']}.{r['coluna_relacionada']} referencia {t}.{r['coluna']}")
    if frases:
        textos += empacotar(f"Tabela {t}. Relações: ", frases, "; ", contar)

    itens = []
    for c in entrada["colunas"]:
        item = f"{c['nome']} ({c['descricao']})" if c["descricao"] else c["nome"]
        if c["valores_possiveis"]:
            item += f" [valores: {', '.join(c['valores_possiveis'])}]"
        itens.append(item)
    textos += empacotar(f"Tabela {t}. Colunas: ", itens, "; ", contar)

    if entrada["contas"]:
        itens = [f"{c['codigo']} {c['descricao']}" for c in entrada["contas"]]
        textos += empacotar(f"Tabela {t}. Contas: ", itens, "; ", contar)

    return [cortar_no_limite(texto, contar) for texto in textos]


# Monta o corpus do esquema (id, fonte e conteudo) a partir das entradas do esquema.json.
def gerar_corpus_esquema(esquema: list[dict], tokenizador) -> list[dict]:
    contar = lambda texto: len(tokenizador.tokenize(texto))
    corpus = []
    for entrada in esquema:
        base = entrada["tabela"].lower().replace(" ", "_")
        for numero, texto in enumerate(textos_da_tabela(entrada, contar), start=1):
            corpus.append({"id": f"esquema_{base}_{numero:03d}", "fonte": entrada["tabela"], "conteudo": texto})
    return corpus


# ---------------------------------------------------------------------------
# 2) Corpus do glossário (PDFs normativos)
# ---------------------------------------------------------------------------

# Troca qualquer sequência de espaços e quebras de linha por um espaço e aplica o NFKC.
def _normalizar(texto: str) -> str:
    # NFKC desfaz ligaturas (ex.: "ﬁ" vira "fi"), que atrapalhariam a busca.
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", texto)).strip()


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
    texto = _normalizar(texto)
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


# Devolve as linhas riscadas da página, como (posição da linha, texto normalizado).
def detectar_linhas_riscadas(pagina) -> list[tuple[tuple, str]]:
    # O risco é um retângulo fino e comprido desenhado sobre o texto.
    tracos = [
        d["rect"] for d in pagina.get_drawings()
        if d["rect"].height < ALTURA_MAXIMA_TRACO and d["rect"].width > LARGURA_MINIMA_TRACO and d.get("fill") is not None
    ]
    if not tracos:
        return []
    riscadas = []
    for bloco in pagina.get_text("dict")["blocks"]:
        for linha in bloco.get("lines", []):
            x0, y0, x1, y1 = linha["bbox"]
            altura, largura = y1 - y0, x1 - x0
            texto = _normalizar("".join(s["text"] for s in linha["spans"]))
            if not texto or largura <= 0:
                continue
            # Trechos horizontais da linha cobertos por traços que cruzam o meio dela.
            cobertos = sorted(
                (max(t.x0, x0), min(t.x1, x1)) for t in tracos
                if FAIXA_DO_MEIO[0] * altura < t.y0 - y0 < FAIXA_DO_MEIO[1] * altura and t.x0 < x1 and t.x1 > x0
            )
            # União dos trechos, para dois traços sobrepostos não contarem duas vezes.
            largura_coberta, fim_atual = 0.0, None
            for inicio, fim in cobertos:
                if fim_atual is None or inicio > fim_atual:
                    largura_coberta += fim - inicio
                    fim_atual = fim
                elif fim > fim_atual:
                    largura_coberta += fim - fim_atual
                    fim_atual = fim
            if largura_coberta / largura >= COBERTURA_MINIMA_RISCO:
                riscadas.append(((x0, y0, x1, y1), texto))
    return riscadas


# Tira do texto de um bloco as linhas riscadas que estão dentro dele.
def remover_linhas_riscadas(texto: str, bloco, riscadas: list[tuple[tuple, str]]) -> str:
    x0, y0, x1, y1 = bloco[:4]
    for (lx0, ly0, lx1, ly1), linha in riscadas:
        if lx0 >= x0 - 1.5 and lx1 <= x1 + 1.5 and ly0 >= y0 - 1.5 and ly1 <= y1 + 1.5:
            posicao = texto.find(linha)
            if posicao >= 0:
                texto = texto[:posicao] + " " + texto[posicao + len(linha):]
    return _normalizar(texto)


# Extrai o texto do PDF como parágrafos separados por linha em branco, sem o texto riscado.
def extrair_texto_pdf(caminho_pdf: str) -> tuple[str, dict]:
    # O divisor de texto usa a linha em branco para respeitar os limites de
    # parágrafo, e o pymupdf não a devolve por padrão.
    documento = pymupdf.open(caminho_pdf)
    blocos_repetidos = detectar_blocos_repetidos(documento)
    paragrafos = []
    caracteres_total, caracteres_riscados = 0, 0
    for pagina in documento:
        riscadas = detectar_linhas_riscadas(pagina)
        primeiro_bloco_da_pagina = True
        for bloco in pagina.get_text("blocks", sort=True):
            texto = _texto_normalizado(bloco)
            if not texto or texto in blocos_repetidos:
                continue
            caracteres_total += len(texto)
            sem_risco = remover_linhas_riscadas(texto, bloco, riscadas)
            caracteres_riscados += len(texto) - len(sem_risco)
            if not sem_risco:
                continue
            texto = sem_risco
            # Junta as duas metades de um parágrafo cortado pela quebra de página.
            if primeiro_bloco_da_pagina and paragrafos and _continua_paragrafo(paragrafos[-1], texto):
                paragrafos[-1] += " " + texto
            else:
                paragrafos.append(texto)
            primeiro_bloco_da_pagina = False
    documento.close()
    estatisticas = {"caracteres_total": caracteres_total, "caracteres_riscados": caracteres_riscados}
    return "\n\n".join(paragrafos), estatisticas


# Cria o divisor de texto, que mede o tamanho dos trechos em tokens do modelo de fragmentação.
def criar_divisor(tokenizador) -> RecursiveCharacterTextSplitter:
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


# Monta o corpus do glossário (id, fonte e conteudo) a partir de todos os PDFs de dados/normativos/.
def gerar_corpus_glossario(tokenizador) -> list[dict]:
    divisor = criar_divisor(tokenizador)
    contar = lambda texto: len(tokenizador.tokenize(texto))

    caminhos_pdf = sorted(glob.glob(os.path.join(PASTA_DOCUMENTOS, "**", "*.pdf"), recursive=True))
    if not caminhos_pdf:
        raise FileNotFoundError(f"Nenhum PDF encontrado em {PASTA_DOCUMENTOS}")

    corpus = []
    for caminho_pdf in caminhos_pdf:
        nome_arquivo = os.path.basename(caminho_pdf)
        nome_base = os.path.splitext(nome_arquivo)[0]
        texto, estatisticas = extrair_texto_pdf(caminho_pdf)
        trechos = divisor.split_text(texto)

        antes = len(trechos)
        trechos = [t for t in trechos if not eh_fragmento_degenerado(t)]
        if len(trechos) < antes:
            print(f"    ({antes - len(trechos)} fragmento(s) degenerado(s) descartado(s))")

        for numero, trecho in enumerate(trechos, start=1):
            corpus.append({"id": f"glossario_{nome_base}_{numero:03d}", "fonte": nome_arquivo, "conteudo": trecho})
        percentual = 100 * estatisticas["caracteres_riscados"] / max(1, estatisticas["caracteres_total"])
        print(
            f"{nome_arquivo}: {len(trechos)} trechos (maior trecho: {max(contar(t) for t in trechos)} tokens; "
            f"texto riscado removido: {percentual:.1f}%)"
        )
    return corpus


# ---------------------------------------------------------------------------

# Grava o corpus como lista de registros, no mesmo formato JSON dos índices.
def gravar_corpus(corpus: list[dict], caminho: str):
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    print(f"Corpus salvo em {caminho} ({len(corpus)} registros)")


# Gera o corpus do esquema e/ou do glossário.
def gerar_corpus(partes: list[str], baixar_modelos: bool = False):
    # O início de toda execução confere o cache; sem --baixar-modelos, só avisa o que falta.
    modelos_embedding.garantir_modelos([modelos_embedding.MODELO_FRAGMENTACAO], baixar_modelos)
    tokenizador = modelos_embedding.carregar_tokenizador(modelos_embedding.MODELO_FRAGMENTACAO)
    if "esquema" in partes:
        with open(CAMINHO_ESQUEMA, encoding="utf-8") as f:
            esquema = json.load(f)
        gravar_corpus(gerar_corpus_esquema(esquema, tokenizador), CAMINHO_CORPUS_ESQUEMA)
    if "glossario" in partes:
        gravar_corpus(gerar_corpus_glossario(tokenizador), CAMINHO_CORPUS_GLOSSARIO)


def main():
    parser = argparse.ArgumentParser(description="Gera o corpus de busca do esquema e do glossário.")
    parser.add_argument("--partes", nargs="+", choices=("esquema", "glossario"), default=["esquema", "glossario"])
    parser.add_argument("--baixar-modelos", action="store_true", help="autoriza baixar os modelos que faltarem")
    args = parser.parse_args()
    gerar_corpus(args.partes, args.baixar_modelos)


if __name__ == "__main__":
    main()
