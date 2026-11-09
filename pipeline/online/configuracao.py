"""
Etapa: Online (também lido pelo estágio offline).

Função: lê a configuração única do projeto, confere os valores e permite sobrescrevê-los
por argumento, sem editar o arquivo.

Descrição geral do funcionamento:
- configuracao.json fica na raiz do repositório e reúne as escolhas que mudam entre os
  experimentos. Chaves:
  - modelo_embedding: chave de um modelo de pipeline/modelos_embedding.py.
  - busca.esquema.metodo e busca.glossario.metodo: método que busca em cada índice
    (embeddings, bm25 ou rrf). O campo por_tipo troca o método para um tipo de pergunta
    (quantitativa, conceitual ou hibrida); vazio, vale o método do índice para todos.
  - k_tabelas: quantas tabelas mais parecidas entram no contexto do gerador de SQL (as
    tabelas referenciadas por chave estrangeira entram além dessas).
  - k_trechos: quantos trechos do glossário entram na síntese.
  - divisor.ativo: se a pergunta híbrida é dividida em parte de dados e parte conceitual.
  - divisor.bloquear_pronomes: se a parte conceitual com pronome é recusada.
  - bm25.radicalizar: se o BM25 usa o radicalizador (stemmer) nas palavras.
- Os valores do arquivo são o padrão do pipeline: E5-base nos dois índices, divisor das
  perguntas híbridas ligado, com recusa de pronomes, e BM25 com radicalizador. O divisor e o
  radicalizador são decisões de arquitetura da autora (09/11/2026); as chaves continuam
  existindo para os experimentos. O comportamento anterior (divisor e radicalizador desligados)
  se obtém com --set divisor.ativo=false --set bm25.radicalizar=false.
- Uma sobrescrita é um texto "chave=valor", com a chave separada por pontos (por exemplo,
  busca.glossario.metodo=bm25). O valor é lido como JSON quando possível (true, 3) e, se
  não for, vale como texto. Só chaves que já existem podem ser sobrescritas (a exceção é
  um tipo dentro de por_tipo).
- Chave desconhecida, tipo errado ou opção fora da lista geram um erro que diz o problema.

Dependências:
- apenas a biblioteca padrão do Python;
- pipeline/modelos_embedding.py (só para conferir a chave do modelo).

Parâmetros (ao executar como script): --set chave=valor (pode repetir) e --arquivo com o
caminho de outro JSON de configuração.

Resultado: fornece carregar() e metodo_de_busca(). Executado como script, imprime a
configuração efetiva depois das sobrescritas.

Executar de dentro de rag-nl2sql-financas-br/:
    python pipeline/online/configuracao.py
    python pipeline/online/configuracao.py --set busca.glossario.metodo=bm25 --set divisor.ativo=true
"""

import argparse
import copy
import json
import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Os módulos compartilhados ficam em pipeline/, fora da pasta deste script.
sys.path.insert(0, os.path.join(RAIZ, "pipeline"))
import modelos_embedding  # noqa: E402

CAMINHO_PADRAO = os.path.join(RAIZ, "configuracao.json")
METODOS = ("embeddings", "bm25", "rrf")
INDICES = ("esquema", "glossario")
TIPOS = ("quantitativa", "conceitual", "hibrida")


# Aponta chaves que faltam ou que não existem em um bloco da configuração.
def _conferir_chaves(nome: str, bloco, esperadas: tuple, erros: list[str]) -> bool:
    if not isinstance(bloco, dict):
        erros.append(f"{nome}: deveria ser um bloco com as chaves {list(esperadas)}")
        return False
    for chave in bloco:
        if chave not in esperadas:
            erros.append(f"{nome}: chave desconhecida {chave!r} (válidas: {list(esperadas)})")
    for chave in esperadas:
        if chave not in bloco:
            erros.append(f"{nome}: falta a chave {chave!r}")
    return all(chave in bloco for chave in esperadas)


# Confere se um valor é um método de busca permitido.
def _conferir_metodo(nome: str, valor, erros: list[str]):
    if valor not in METODOS:
        erros.append(f"{nome}: método inválido {valor!r} (válidos: {list(METODOS)})")


# Confere a configuração inteira e levanta ValueError com todos os problemas encontrados.
def validar(config: dict) -> dict:
    erros = []
    esperadas = ("modelo_embedding", "busca", "k_tabelas", "k_trechos", "divisor", "bm25")
    if _conferir_chaves("configuração", config, esperadas, erros):
        if config["modelo_embedding"] not in modelos_embedding.MODELOS:
            erros.append(
                f"modelo_embedding: modelo desconhecido {config['modelo_embedding']!r} "
                f"(válidos: {list(modelos_embedding.MODELOS)})"
            )
        if _conferir_chaves("busca", config["busca"], INDICES, erros):
            for indice in INDICES:
                bloco = config["busca"][indice]
                if _conferir_chaves(f"busca.{indice}", bloco, ("metodo", "por_tipo"), erros):
                    _conferir_metodo(f"busca.{indice}.metodo", bloco["metodo"], erros)
                    if not isinstance(bloco["por_tipo"], dict):
                        erros.append(f"busca.{indice}.por_tipo: deveria ser um bloco (pode ser vazio)")
                    else:
                        for tipo, metodo in bloco["por_tipo"].items():
                            if tipo not in TIPOS:
                                erros.append(f"busca.{indice}.por_tipo: tipo desconhecido {tipo!r} (válidos: {list(TIPOS)})")
                            _conferir_metodo(f"busca.{indice}.por_tipo.{tipo}", metodo, erros)
        for chave in ("k_tabelas", "k_trechos"):
            valor = config[chave]
            # bool é subclasse de int em Python, então precisa ser excluído à parte.
            if isinstance(valor, bool) or not isinstance(valor, int) or valor < 1:
                erros.append(f"{chave}: deveria ser um número inteiro maior ou igual a 1, veio {valor!r}")
        if _conferir_chaves("divisor", config["divisor"], ("ativo", "bloquear_pronomes"), erros):
            for chave in ("ativo", "bloquear_pronomes"):
                if not isinstance(config["divisor"][chave], bool):
                    erros.append(f"divisor.{chave}: deveria ser true ou false, veio {config['divisor'][chave]!r}")
        if _conferir_chaves("bm25", config["bm25"], ("radicalizar",), erros):
            if not isinstance(config["bm25"]["radicalizar"], bool):
                erros.append(f"bm25.radicalizar: deveria ser true ou false, veio {config['bm25']['radicalizar']!r}")
    if erros:
        raise ValueError("Configuração inválida:\n  - " + "\n  - ".join(erros))
    return config


# Lê o valor de uma sobrescrita como JSON (true, 3) e, se não for JSON, como texto.
def _ler_valor(texto: str):
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        return texto


# Aplica uma sobrescrita "chave.pontuada=valor" sobre a configuração.
def _sobrescrever(config: dict, sobrescrita: str):
    if "=" not in sobrescrita:
        raise ValueError(f"Sobrescrita inválida {sobrescrita!r}: use o formato chave=valor")
    caminho, texto = sobrescrita.split("=", 1)
    partes = caminho.strip().split(".")
    bloco = config
    for i, parte in enumerate(partes[:-1]):
        if not isinstance(bloco, dict) or parte not in bloco:
            raise ValueError(f"Sobrescrita {sobrescrita!r}: a chave {'.'.join(partes[:i + 1])!r} não existe")
        bloco = bloco[parte]
    ultima = partes[-1]
    # Um tipo novo pode ser criado dentro de por_tipo; em qualquer outro lugar, só chaves existentes.
    pode_criar = len(partes) >= 2 and partes[-2] == "por_tipo"
    if not isinstance(bloco, dict) or (ultima not in bloco and not pode_criar):
        raise ValueError(f"Sobrescrita {sobrescrita!r}: a chave {caminho.strip()!r} não existe")
    bloco[ultima] = _ler_valor(texto.strip())


# Lê a configuração (do arquivo padrão ou de outro), aplica as sobrescritas e confere o resultado.
def carregar(caminho: str | None = None, sobrescritas: list[str] | None = None) -> dict:
    with open(caminho or CAMINHO_PADRAO, encoding="utf-8") as f:
        config = copy.deepcopy(json.load(f))
    for sobrescrita in sobrescritas or []:
        _sobrescrever(config, sobrescrita)
    return validar(config)


# Devolve o método que busca no índice (esquema ou glossario) para o tipo de pergunta dado.
def metodo_de_busca(config: dict, indice: str, tipo: str | None = None) -> str:
    if indice not in INDICES:
        raise ValueError(f"Índice desconhecido {indice!r} (válidos: {list(INDICES)})")
    if tipo is not None and tipo not in TIPOS:
        raise ValueError(f"Tipo de pergunta desconhecido {tipo!r} (válidos: {list(TIPOS)})")
    bloco = config["busca"][indice]
    return bloco["por_tipo"].get(tipo, bloco["metodo"])


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Mostra a configuração efetiva do projeto.")
    parser.add_argument("--arquivo", default=None, help="caminho de outro JSON de configuração")
    parser.add_argument("--set", dest="sobrescritas", action="append", default=[], metavar="chave=valor")
    args = parser.parse_args()
    try:
        config = carregar(args.arquivo, args.sobrescritas)
    except ValueError as erro:
        print(erro)
        sys.exit(2)
    print(json.dumps(config, ensure_ascii=False, indent=2))
    for indice in INDICES:
        print(f"Método no índice de {indice}:", {tipo: metodo_de_busca(config, indice, tipo) for tipo in TIPOS})


if __name__ == "__main__":
    main()
