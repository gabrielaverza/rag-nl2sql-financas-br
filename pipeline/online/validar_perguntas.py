"""
Etapa: Online (preparação da avaliação em perguntas novas).

Função: confere um arquivo de perguntas novas antes de ele ser usado nas avaliações de recuperação.

Descrição geral do funcionamento:
- Cada pergunta tem os campos id, pergunta, tipo (quantitativa, conceitual ou hibrida),
  tabelas_esperadas (lista de nomes de tabela) e evidencia (lista de textos copiados dos normativos).
  O campo sql_referencia é opcional (só serve à avaliação do SQL).
- Erros: campo faltando, tipo desconhecido, id repetido ou já usado nos outros conjuntos, pergunta igual
  a uma dos outros conjuntos, tabela que não existe em embeddings/esquema.json, evidência que não
  aparece em nenhum trecho de embeddings/index_glossario.json, e combinação errada com o tipo
  (quantitativa: só tabelas; conceitual: só evidência; híbrida: as duas).
- Avisos: evidência que aparece em mais de 5 trechos (genérica demais, o Recall perde sentido),
  pergunta que cita companhia já usada nos conjuntos existentes (para manter a independência, prefira
  companhias novas) e conjunto muito desequilibrado entre os tipos.
- A evidência é comparada com os trechos como em avaliar_recuperacao.py: espaços normalizados, sem
  ignorar maiúsculas.
- Nenhuma busca por embeddings ou BM25 é feita aqui, então a conferência não depende dos métodos
  avaliados (a evidência deve ser escolhida lendo os normativos, e não olhando o que a busca devolve).

Dependências: embeddings/esquema.json e embeddings/index_glossario.json; dados/perguntas_dfp.json e
dados/perguntas_validacao.json (para as comparações). Só biblioteca padrão.

Parâmetros: caminho do arquivo de perguntas a conferir, como primeiro argumento.

Resultado: imprime os erros, os avisos e a contagem por tipo; termina com código 1 se houver erro.

Executar de dentro de rag-nl2sql-financas-br/:
    python pipeline/online/validar_perguntas.py dados/perguntas_novas.json
"""

import json
import os
import re
import sys

CAMINHOS_EXISTENTES = ("dados/perguntas_dfp.json", "dados/perguntas_validacao.json")
CAMINHO_ESQUEMA = "embeddings/esquema.json"
CAMINHO_GLOSSARIO = "embeddings/index_glossario.json"
TIPOS = ("quantitativa", "conceitual", "hibrida")
CAMPOS = ("id", "pergunta", "tipo", "tabelas_esperadas", "evidencia")
MAX_TRECHOS_POR_EVIDENCIA = 5
EMPRESAS_JA_USADAS = (
    "Ambev", "Alpargatas", "JBS", "Petrobras", "Vale", "WEG", "Natura", "Suzano",
    "Raia Drogasil", "Klabin", "Lojas Renner", "Gerdau",
)


# Troca qualquer sequência de espaços e quebras de linha por um espaço.
def normalizar(texto: str) -> str:
    return re.sub(r"\s+", " ", texto)


# Lê um arquivo JSON.
def ler(caminho: str):
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


# Confere as perguntas e devolve (erros, avisos).
def conferir(perguntas: list[dict], caminho: str) -> tuple[list[str], list[str]]:
    erros, avisos = [], []
    tabelas = {e["tabela"] for e in ler(CAMINHO_ESQUEMA)}
    trechos = [normalizar(r["conteudo"]) for r in ler(CAMINHO_GLOSSARIO)]

    # Os outros conjuntos servem de comparação; o próprio arquivo, se for um deles, fica de fora.
    outros = [c for c in CAMINHOS_EXISTENTES if os.path.abspath(c) != os.path.abspath(caminho)]
    ids_usados, perguntas_usadas = set(), set()
    for outro in outros:
        for p in ler(outro):
            ids_usados.add(p["id"])
            perguntas_usadas.add(normalizar(p["pergunta"]).lower())

    vistos = set()
    for i, p in enumerate(perguntas, start=1):
        nome = p.get("id", f"#{i}")
        faltando = [c for c in CAMPOS if c not in p]
        if faltando:
            erros.append(f"{nome}: campos faltando: {faltando}")
            continue
        if p["tipo"] not in TIPOS:
            erros.append(f"{nome}: tipo desconhecido: {p['tipo']}")
            continue
        if p["id"] in vistos:
            erros.append(f"{nome}: id repetido neste arquivo")
        vistos.add(p["id"])
        if p["id"] in ids_usados:
            erros.append(f"{nome}: id já usado em outro conjunto")
        if normalizar(p["pergunta"]).lower() in perguntas_usadas:
            erros.append(f"{nome}: pergunta igual a uma de outro conjunto")

        tem_tabelas, tem_evidencia = bool(p["tabelas_esperadas"]), bool(p["evidencia"])
        if p["tipo"] == "quantitativa" and (not tem_tabelas or tem_evidencia):
            erros.append(f"{nome}: quantitativa precisa de tabelas_esperadas e de evidencia vazia")
        if p["tipo"] == "conceitual" and (tem_tabelas or not tem_evidencia):
            erros.append(f"{nome}: conceitual precisa de evidencia e de tabelas_esperadas vazia")
        if p["tipo"] == "hibrida" and not (tem_tabelas and tem_evidencia):
            erros.append(f"{nome}: híbrida precisa de tabelas_esperadas e de evidencia")

        for tabela in p["tabelas_esperadas"]:
            if tabela not in tabelas:
                erros.append(f"{nome}: tabela inexistente: {tabela}")
        for evidencia in p["evidencia"]:
            achados = sum(1 for t in trechos if normalizar(evidencia) in t)
            if achados == 0:
                erros.append(f"{nome}: evidência não aparece em nenhum trecho: {evidencia[:60]!r}")
            elif achados > MAX_TRECHOS_POR_EVIDENCIA:
                avisos.append(f"{nome}: evidência em {achados} trechos (genérica demais): {evidencia[:60]!r}")

        citadas = [e for e in EMPRESAS_JA_USADAS if re.search(rf"\b{re.escape(e)}\b", p["pergunta"])]
        if citadas and outros:
            avisos.append(f"{nome}: cita companhia já usada nos outros conjuntos: {citadas}")

    por_tipo = {t: sum(1 for p in perguntas if p.get("tipo") == t) for t in TIPOS}
    if perguntas and min(por_tipo.values()) < len(perguntas) / 6:
        avisos.append(f"conjunto desequilibrado entre os tipos: {por_tipo}")
    return erros, avisos


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        print("Uso: python pipeline/online/validar_perguntas.py arquivo_de_perguntas.json")
        sys.exit(1)
    caminho = sys.argv[1]
    perguntas = ler(caminho)
    erros, avisos = conferir(perguntas, caminho)
    por_tipo = {t: sum(1 for p in perguntas if p.get("tipo") == t) for t in TIPOS}
    print(f"{caminho}: {len(perguntas)} perguntas {por_tipo}")
    for e in erros:
        print(f"  ERRO: {e}")
    for a in avisos:
        print(f"  aviso: {a}")
    print(f"{len(erros)} erro(s), {len(avisos)} aviso(s)")
    sys.exit(1 if erros else 0)


if __name__ == "__main__":
    main()
