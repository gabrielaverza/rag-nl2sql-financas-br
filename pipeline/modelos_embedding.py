"""
Etapa: Offline e Online (módulo compartilhado).

Função: registro dos modelos de embeddings que o projeto sabe usar.

Descrição geral do funcionamento:
- Cada modelo tem uma chave curta (por exemplo, "e5-base") e uma ficha com o identificador
  no Hugging Face, os prefixos que o modelo exige, as dimensões do vetor, o limite de tokens
  do modelo e o idioma.
- Os scripts offline (que geram os índices) e a busca online (que vetoriza a pergunta) leem
  a mesma ficha, para o modelo e os prefixos nunca ficarem diferentes entre os dois lados.
- Os quatro modelos multilíngues (minilm, e5-small, e5-base e bge-m3) são os comparados nos
  experimentos. O e5-base-v2 é um controle só em inglês, para mostrar o efeito do
  multilinguismo.
- Ao carregar, cada modelo usa o seu próprio limite de tokens (campo limite_tokens), com teto
  de 512: o BGE-M3 aceita 8192, mas nenhum texto do projeto passa de cerca de 125 tokens.
- Ler o registro é leve: o sentence-transformers só é importado quando um modelo é carregado.

Dependências:
- bibliotecas numpy e sentence-transformers (esta só ao carregar um modelo);
- modelos já em cache local. Carregar um modelo nunca baixa nada: se faltar, o erro mostra o
  comando. O download só acontece em garantir_modelos(..., baixar=True), que os scripts
  offline chamam no início quando a autora passa --baixar-modelos.

Parâmetros: nenhum argumento para importar. Executado como script, imprime a tabela dos
modelos e diz quais já estão no cache.

Resultado: fornece MODELOS, MODELO_PADRAO, MODELO_FRAGMENTACAO, obter_ficha(), limite_aplicado(),
esta_em_cache(), garantir_modelos(), carregar_modelo(), carregar_tokenizador(),
vetorizar_pergunta(), vetorizar_trechos() e registros_com_vetores().

Executar de dentro de rag-nl2sql-financas-br/:
    python pipeline/modelos_embedding.py
"""

import numpy as np

MODELO_PADRAO = "e5-base"

# O tamanho dos registros do corpus (trechos dos normativos e textos do esquema) é medido sempre
# com o tokenizador deste modelo, para o corpus ser o mesmo qualquer que seja o modelo de embeddings.
MODELO_FRAGMENTACAO = "e5-base"

# Teto do tamanho de entrada aplicado ao carregar. Cada modelo usa o seu limite nativo
# (campo limite_tokens), mas nunca mais que este teto. Com os textos de hoje (no máximo 124
# tokens, medidos com os tokenizadores do MiniLM e do E5) nenhum modelo corta nada.
TETO_TOKENS = 512

# Os vetores são sempre normalizados na busca (a pergunta e o índice), então o produto
# interno vale o cosseno; por isso "normalizar" é True em todos.
MODELOS = {
    "minilm": {
        "id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "prefixo_pergunta": "",
        "prefixo_trecho": "",
        "dimensoes": 384,
        "limite_tokens": 128,
        "normalizar": True,
        "idioma": "multilingue",
        "tamanho_mb": 480,
    },
    "e5-small": {
        "id": "intfloat/multilingual-e5-small",
        "prefixo_pergunta": "query: ",
        "prefixo_trecho": "passage: ",
        "dimensoes": 384,
        "limite_tokens": 512,
        "normalizar": True,
        "idioma": "multilingue",
        "tamanho_mb": 493,
    },
    "e5-base": {
        "id": "intfloat/multilingual-e5-base",
        "prefixo_pergunta": "query: ",
        "prefixo_trecho": "passage: ",
        "dimensoes": 768,
        "limite_tokens": 512,
        "normalizar": True,
        "idioma": "multilingue",
        "tamanho_mb": 1135,
    },
    "bge-m3": {
        "id": "BAAI/bge-m3",
        "prefixo_pergunta": "",
        "prefixo_trecho": "",
        "dimensoes": 1024,
        "limite_tokens": 8192,
        "normalizar": True,
        "idioma": "multilingue",
        "tamanho_mb": 2296,
    },
    "e5-base-v2": {
        "id": "intfloat/e5-base-v2",
        "prefixo_pergunta": "query: ",
        "prefixo_trecho": "passage: ",
        "dimensoes": 768,
        "limite_tokens": 512,
        "normalizar": True,
        "idioma": "ingles",
        "tamanho_mb": 439,
    },
}

_modelos_carregados = {}


# Devolve a ficha do modelo ou explica quais chaves existem.
def obter_ficha(chave: str) -> dict:
    if chave not in MODELOS:
        raise ValueError(f"Modelo de embeddings desconhecido: {chave!r}. Chaves válidas: {', '.join(MODELOS)}")
    return MODELOS[chave]


# Devolve o tamanho de entrada aplicado ao modelo: o limite dele, com teto de TETO_TOKENS.
def limite_aplicado(chave: str) -> int:
    return min(obter_ficha(chave)["limite_tokens"], TETO_TOKENS)


# Diz se os arquivos do modelo já estão no cache local do Hugging Face, sem baixar nada.
def esta_em_cache(chave: str) -> bool:
    from huggingface_hub import try_to_load_from_cache

    caminho = try_to_load_from_cache(obter_ficha(chave)["id"], "config.json")
    return isinstance(caminho, str)


# Baixa um modelo para o cache local. Só é chamada quando a autora pede o download.
def baixar_modelo(chave: str):
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(obter_ficha(chave)["id"])


# Confere no início de uma execução se os modelos estão no cache e baixa só os que faltam, e só se pedido.
def garantir_modelos(chaves: list[str], baixar: bool = False):
    faltando = [c for c in dict.fromkeys(chaves) if not esta_em_cache(c)]
    if not faltando:
        return
    linhas = [
        f"  - {c}: {obter_ficha(c)['id']} (https://huggingface.co/{obter_ficha(c)['id']}), "
        f"cerca de {obter_ficha(c)['tamanho_mb']} MB"
        for c in faltando
    ]
    total = sum(obter_ficha(c)["tamanho_mb"] for c in faltando)
    if not baixar:
        # Baixar é uma decisão da autora: sem a opção explícita, o script para e diz o que falta.
        raise SystemExit(
            "Faltam modelos no cache local:\n" + "\n".join(linhas) + f"\nTotal: cerca de {total} MB. "
            "Para baixá-los, rode de novo com --baixar-modelos (a opção é a autorização do download)."
        )
    print("Baixando os modelos que faltam (autorizado por --baixar-modelos):\n" + "\n".join(linhas))
    for c in faltando:
        baixar_modelo(c)


# Carrega só o tokenizador do modelo (sem os pesos), a partir do cache local.
def carregar_tokenizador(chave: str = MODELO_PADRAO):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(obter_ficha(chave)["id"], local_files_only=True)


# Carrega o modelo uma única vez e o reaproveita; nunca baixa, só lê do cache local.
def carregar_modelo(chave: str = MODELO_PADRAO):
    if chave not in _modelos_carregados:
        from sentence_transformers import SentenceTransformer

        ficha = obter_ficha(chave)
        try:
            modelo = SentenceTransformer(ficha["id"], local_files_only=True)
        except OSError as erro:
            # Baixar é uma decisão da autora (pede autorização, informa nome e tamanho).
            raise OSError(
                f"O modelo {ficha['id']} não está no cache local. Para baixá-lo, rode: "
                f"python -c \"from sentence_transformers import SentenceTransformer; "
                f"SentenceTransformer('{ficha['id']}')\""
            ) from erro
        modelo.max_seq_length = limite_aplicado(chave)
        _modelos_carregados[chave] = modelo
    return _modelos_carregados[chave]


# Transforma a pergunta em um vetor normalizado, com o prefixo de pergunta do modelo.
def vetorizar_pergunta(texto: str, chave: str = MODELO_PADRAO) -> np.ndarray:
    ficha = obter_ficha(chave)
    return carregar_modelo(chave).encode(ficha["prefixo_pergunta"] + texto, normalize_embeddings=True)


# Transforma os textos que vão para o índice em vetores, com o prefixo de trecho do modelo.
def vetorizar_trechos(textos: list[str], chave: str = MODELO_PADRAO) -> np.ndarray:
    ficha = obter_ficha(chave)
    return carregar_modelo(chave).encode([ficha["prefixo_trecho"] + t for t in textos])


# Junta ao corpus (registros com id, fonte e conteudo) o vetor de cada registro, no formato do índice.
# A codificação é feita por fonte (uma tabela ou um PDF de cada vez), como sempre foi, para os vetores
# saírem idênticos aos já versionados. Avisa se algum registro passa do limite de entrada do modelo.
def registros_com_vetores(corpus: list[dict], chave: str = MODELO_PADRAO) -> list[dict]:
    from itertools import groupby

    tokenizador = carregar_tokenizador(MODELO_FRAGMENTACAO)
    limite = limite_aplicado(chave) - 2
    maior = max(len(tokenizador.tokenize(r["conteudo"])) for r in corpus)
    if maior > limite:
        print(f"ATENÇÃO: registro com {maior} tokens (limite do modelo {chave}: {limite})")
    registros = []
    for _, grupo in groupby(corpus, key=lambda r: r["fonte"]):
        grupo = list(grupo)
        vetores = vetorizar_trechos([r["conteudo"] for r in grupo], chave)
        registros += [
            {"id": r["id"], "fonte": r["fonte"], "conteudo": r["conteudo"], "embedding": v.tolist()}
            for r, v in zip(grupo, vetores)
        ]
    return registros


# Imprime a tabela dos modelos registrados e se cada um está no cache.
def listar():
    print(f"{'chave':<11}{'idioma':<13}{'dim':>5}{'limite':>8}{'usado':>7}{'MB':>6}  {'pergunta':<11}{'trecho':<13}{'cache':<6}id")
    for chave, ficha in MODELOS.items():
        print(
            f"{chave:<11}{ficha['idioma']:<13}{ficha['dimensoes']:>5}{ficha['limite_tokens']:>8}{limite_aplicado(chave):>7}{ficha['tamanho_mb']:>6}  "
            f"{ficha['prefixo_pergunta']!r:<11}{ficha['prefixo_trecho']!r:<13}"
            f"{'sim' if esta_em_cache(chave) else 'não':<6}{ficha['id']}"
        )
    print(f"\nPadrão: {MODELO_PADRAO}")


if __name__ == "__main__":
    listar()
