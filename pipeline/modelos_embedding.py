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
- Ler o registro é leve: o sentence-transformers só é importado quando um modelo é carregado.

Dependências:
- bibliotecas numpy e sentence-transformers (esta só ao carregar um modelo);
- modelos já em cache local. Este módulo não baixa nada: se faltar um modelo, o erro
  mostra o comando para baixá-lo.

Parâmetros: nenhum argumento para importar. Executado como script, imprime a tabela dos
modelos e diz quais já estão no cache.

Resultado: fornece MODELOS, MODELO_PADRAO, obter_ficha(), esta_em_cache(), carregar_modelo(),
vetorizar_pergunta() e vetorizar_trechos().

Executar de dentro de rag-nl2sql-financas-br/:
    python pipeline/modelos_embedding.py
"""

import numpy as np

MODELO_PADRAO = "e5-base"

# Tamanho de entrada aplicado a todos os modelos ao carregar, como no teste de modelos de
# 28/10/2026. O limite do próprio modelo (campo limite_tokens) fica só como informação.
MAX_SEQ_LENGTH_APLICADO = 512

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
    },
    "e5-small": {
        "id": "intfloat/multilingual-e5-small",
        "prefixo_pergunta": "query: ",
        "prefixo_trecho": "passage: ",
        "dimensoes": 384,
        "limite_tokens": 512,
        "normalizar": True,
        "idioma": "multilingue",
    },
    "e5-base": {
        "id": "intfloat/multilingual-e5-base",
        "prefixo_pergunta": "query: ",
        "prefixo_trecho": "passage: ",
        "dimensoes": 768,
        "limite_tokens": 512,
        "normalizar": True,
        "idioma": "multilingue",
    },
    "bge-m3": {
        "id": "BAAI/bge-m3",
        "prefixo_pergunta": "",
        "prefixo_trecho": "",
        "dimensoes": 1024,
        "limite_tokens": 8192,
        "normalizar": True,
        "idioma": "multilingue",
    },
    "e5-base-v2": {
        "id": "intfloat/e5-base-v2",
        "prefixo_pergunta": "query: ",
        "prefixo_trecho": "passage: ",
        "dimensoes": 768,
        "limite_tokens": 512,
        "normalizar": True,
        "idioma": "ingles",
    },
}

_modelos_carregados = {}


# Devolve a ficha do modelo ou explica quais chaves existem.
def obter_ficha(chave: str) -> dict:
    if chave not in MODELOS:
        raise ValueError(f"Modelo de embeddings desconhecido: {chave!r}. Chaves válidas: {', '.join(MODELOS)}")
    return MODELOS[chave]


# Diz se os arquivos do modelo já estão no cache local do Hugging Face, sem baixar nada.
def esta_em_cache(chave: str) -> bool:
    from huggingface_hub import try_to_load_from_cache

    caminho = try_to_load_from_cache(obter_ficha(chave)["id"], "config.json")
    return isinstance(caminho, str)


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
        modelo.max_seq_length = MAX_SEQ_LENGTH_APLICADO
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


# Imprime a tabela dos modelos registrados e se cada um está no cache.
def listar():
    print(f"{'chave':<11}{'idioma':<13}{'dim':>5}{'limite':>8}  {'pergunta':<11}{'trecho':<13}{'cache':<6}id")
    for chave, ficha in MODELOS.items():
        print(
            f"{chave:<11}{ficha['idioma']:<13}{ficha['dimensoes']:>5}{ficha['limite_tokens']:>8}  "
            f"{ficha['prefixo_pergunta']!r:<11}{ficha['prefixo_trecho']!r:<13}"
            f"{'sim' if esta_em_cache(chave) else 'não':<6}{ficha['id']}"
        )
    print(f"\nPadrão: {MODELO_PADRAO}")


if __name__ == "__main__":
    listar()
