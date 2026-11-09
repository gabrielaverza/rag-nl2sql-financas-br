"""
Etapa: Online.

Função: orquestrador do estágio online, que liga todos os passos em um fluxo único por pergunta.

Descrição geral do funcionamento:
1. Classificador de intenção (classificador.py).
2. Divisor da pergunta híbrida (divisor.py), só se divisor.ativo for true em configuracao.json:
   separa a parte de dados e a parte conceitual. As tabelas continuam sendo buscadas com a
   pergunta inteira; os trechos dos normativos são buscados com a parte conceitual.
3. Caminho quantitativo ou híbrido: seleção de contexto e geração de SQL, com autocorreção
   (gerador_sql.py).
4. Síntese da resposta em linguagem natural (sintetizador.py).
5. Validação: relevância do contexto e fidelidade da resposta (validador.py).
6. Formatação final com proveniência (formatador.py).
- A busca (modelo de embeddings, método por índice e por tipo, k_tabelas e k_trechos) segue
  configuracao.json, que pode ser sobrescrito por --set chave=valor ou por outro arquivo
  (--configuracao), sem editar nada.
- Se o SQL falhar (nenhuma tentativa produziu uma consulta válida), a pergunta quantitativa
  recebe uma mensagem de falha. A híbrida cai para o caminho conceitual, usando só os trechos
  do glossário, e a resposta avisa que os dados não foram obtidos.
- Se o SQL executar mas devolver vazio nas três tentativas, a resposta informa que a consulta
  não retornou linhas.
- Modo vivo: com --interativo (várias perguntas digitadas) ou --arquivo (várias perguntas de um
  arquivo), o processo carrega o modelo de embeddings, os índices e o LLM uma vez (preaquecer())
  e responde as perguntas seguintes sem pagar o custo de partida de novo.

Dependências:
- todos os módulos de pipeline/online/, configuracao.json, os índices e o esquema.json em
  embeddings/ e dados/dfp.db;
- servidor Ollama em execução, com o modelo qwen2.5-coder:7b baixado.

Parâmetros:
- a pergunta, em linguagem natural, como argumento de linha de comando (ou nenhuma, com os modos
  abaixo);
- --interativo: lê uma pergunta por vez do teclado até uma linha vazia ou "sair";
- --arquivo CAMINHO: responde as perguntas de um arquivo (texto, uma por linha, ou JSON com uma
  lista de textos ou de objetos com o campo "pergunta");
- --configuracao CAMINHO: outro arquivo de configuração no lugar de configuracao.json;
- --set chave=valor: sobrescreve um valor da configuração (pode repetir).

Resultado: fornece executar(pergunta, cache_sql), que devolve a classe, a divisão da pergunta (ou None), o
resultado do SQL, a resposta final formatada, a validação e o tempo de cada passo, e preaquecer().
Pela linha de comando, imprime a resposta.

Executar de dentro de rag-nl2sql-financas-br/:
    python pipeline/online/main.py "Qual foi a receita líquida consolidada da Ambev em 2025?"
    python pipeline/online/main.py --interativo
    python pipeline/online/main.py --arquivo dados/perguntas_exemplo.txt --set divisor.ativo=true
"""

import argparse
import json
import sys
import time

import ollama

from busca_configuravel import definir_configuracao, obter_busca
from classificador import MODELO_LLM, classificar
from configuracao import carregar
from divisor import dividir
from formatador import formatar
from gerador_sql import gerar_sql
from sintetizador import responder
from validador import validar

FALHA_NO_SQL = "Não consegui gerar uma consulta SQL válida para responder a essa pergunta com os dados disponíveis."
AVISO_HIBRIDA = ("Aviso: não foi possível obter os dados do banco para esta pergunta, então a resposta "
                 "usa apenas os documentos normativos.")
# Validação neutra, para respostas sem conteúdo gerado pelo LLM (nada a conferir).
VALIDACAO_NEUTRA = {
    "relevancia_contexto": {"score": None, "por_trecho": []},
    "fidelidade": {"score": 1.0, "afirmacoes": []},
    "aprovado": True,
}


# cache_sql (opcional) é um dicionário que o executor de experimentos passa para reaproveitar o SQL já
# gerado para a mesma pergunta e as mesmas tabelas; no uso normal, fica vazio.
def executar(pergunta: str, cache_sql: dict | None = None) -> dict:
    config = obter_busca().config
    tempos = {}
    inicio = time.time()
    classe = classificar(pergunta)
    tempos["classificador"] = time.time() - inicio

    # O divisor só age nas híbridas e só se estiver ligado. Se não conseguir dividir, a busca de
    # trechos continua com a pergunta inteira.
    divisao = None
    consulta_trechos = None
    if classe == "hibrida" and config["divisor"]["ativo"]:
        inicio = time.time()
        divisao = dividir(pergunta, config["divisor"]["bloquear_pronomes"])
        tempos["divisor"] = time.time() - inicio
        if divisao["dividida"]:
            consulta_trechos = divisao["parte_conceitual"]

    resultado_sql = None
    if classe in ("quantitativa", "hibrida"):
        inicio = time.time()
        resultado_sql = gerar_sql(pergunta, tipo=classe, cache=cache_sql)
        tempos["sql"] = time.time() - inicio

    inicio = time.time()
    aviso = None
    if resultado_sql and resultado_sql["status"] == "erro":
        if classe == "quantitativa":
            resposta = {"resposta": FALHA_NO_SQL, "trechos": [], "contexto": ""}
        else:
            resposta = responder(pergunta, "conceitual", consulta_trechos=consulta_trechos)
            aviso = AVISO_HIBRIDA
        resultado_para_resposta = None
    else:
        resposta = responder(pergunta, classe, resultado_sql, consulta_trechos)
        resultado_para_resposta = resultado_sql
    tempos["sintese"] = time.time() - inicio

    inicio = time.time()
    if resposta["resposta"] == FALHA_NO_SQL:
        validacao = VALIDACAO_NEUTRA
    else:
        validacao = validar(pergunta, resposta["resposta"], resposta["contexto"], resposta["trechos"])
    tempos["validacao"] = time.time() - inicio

    resposta_final = formatar(pergunta, classe, resposta, resultado_para_resposta, validacao)
    if aviso:
        resposta_final += "\n\n" + aviso

    return {
        "pergunta": pergunta,
        "classe": classe,
        "divisao": divisao,
        "resultado_sql": resultado_sql,
        "resposta": resposta_final,
        "validacao": validacao,
        "tempos": tempos,
    }


# Carrega o que a primeira pergunta teria de carregar (modelo de embeddings, índices e LLM) e devolve o tempo de cada parte.
def preaquecer() -> dict:
    tempos = {"busca": obter_busca().preaquecer()}
    inicio = time.perf_counter()
    # Uma chamada curta faz o Ollama carregar o modelo para a memória.
    ollama.chat(model=MODELO_LLM, messages=[{"role": "user", "content": "ok"}], options={"num_predict": 1})
    tempos["llm"] = time.perf_counter() - inicio
    return tempos


# Imprime o resultado de uma pergunta.
def imprimir_resultado(r: dict, tempo: float):
    print(f"Pergunta: {r['pergunta']}")
    print(f"Classificada como: {r['classe']}")
    if r["divisao"]:
        if r["divisao"]["dividida"]:
            print(f"Divisão: dados = {r['divisao']['parte_dados']} | conceito = {r['divisao']['parte_conceitual']}")
        else:
            print("Divisão: não foi possível dividir; os trechos foram buscados com a pergunta inteira")
    if r["resultado_sql"]:
        print(f"Tabelas do contexto: {r['resultado_sql']['tabelas_contexto']}")
        print(f"Status do SQL: {r['resultado_sql']['status']} ({len(r['resultado_sql']['tentativas'])} tentativa(s))")
    print(f"Fidelidade: {r['validacao']['fidelidade']['score']:.2f} | aprovado: {r['validacao']['aprovado']}")
    print(f"\n{r['resposta']}")
    print(f"\n({tempo:.1f} s)")


# Lê as perguntas de um arquivo de texto (uma por linha) ou JSON (lista de textos ou de objetos com "pergunta").
def ler_perguntas(caminho: str) -> list[str]:
    with open(caminho, encoding="utf-8") as f:
        if caminho.lower().endswith(".json"):
            itens = json.load(f)
            return [i if isinstance(i, str) else i["pergunta"] for i in itens]
        return [linha.strip() for linha in f if linha.strip()]


# Responde e imprime uma pergunta, medindo o tempo total.
def responder_e_imprimir(pergunta: str):
    inicio = time.time()
    r = executar(pergunta)
    imprimir_resultado(r, time.time() - inicio)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Responde perguntas sobre as DFPs e os normativos.")
    parser.add_argument("pergunta", nargs="*", help="a pergunta em linguagem natural")
    parser.add_argument("--interativo", action="store_true", help="lê uma pergunta por vez do teclado")
    parser.add_argument("--arquivo", help="arquivo com várias perguntas (texto, uma por linha, ou JSON)")
    parser.add_argument("--configuracao", help="outro arquivo de configuração no lugar de configuracao.json")
    parser.add_argument("--set", dest="sobrescritas", action="append", default=[], metavar="chave=valor",
                        help="sobrescreve um valor da configuração (pode repetir)")
    args = parser.parse_args()
    if not (args.pergunta or args.interativo or args.arquivo):
        parser.print_usage()
        print('Informe uma pergunta, --interativo ou --arquivo. Exemplo: python pipeline/online/main.py "pergunta"')
        sys.exit(1)

    try:
        definir_configuracao(carregar(args.configuracao, args.sobrescritas))
    except ValueError as erro:
        print(erro)
        sys.exit(2)

    if args.pergunta:
        responder_e_imprimir(" ".join(args.pergunta))
        return

    perguntas = ler_perguntas(args.arquivo) if args.arquivo else None
    print("Carregando o modelo de embeddings, os índices e o LLM (uma vez só)...")
    tempos = preaquecer()
    print(f"Pronto: busca {tempos['busca']:.1f} s, LLM {tempos['llm']:.1f} s.\n")

    if perguntas is not None:
        for numero, pergunta in enumerate(perguntas, start=1):
            print(f"===== Pergunta {numero} de {len(perguntas)} =====")
            responder_e_imprimir(pergunta)
            print()
        return

    while True:
        try:
            pergunta = input('Pergunta (linha vazia ou "sair" encerra): ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not pergunta or pergunta.lower() == "sair":
            break
        print()
        responder_e_imprimir(pergunta)
        print()


if __name__ == "__main__":
    main()
