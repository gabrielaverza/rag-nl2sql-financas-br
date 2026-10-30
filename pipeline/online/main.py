"""
Etapa: Online.

Função: orquestrador do estágio online, que liga todos os passos em um fluxo único por pergunta.

Descrição geral do funcionamento:
1. Classificador de intenção (classificador.py).
2. Caminho quantitativo ou híbrido: seleção de contexto e geração de SQL, com autocorreção
   (gerador_sql.py).
3. Síntese da resposta em linguagem natural (sintetizador.py).
4. Validação: relevância do contexto e fidelidade da resposta (validador.py).
5. Formatação final com proveniência (formatador.py).
- Se o SQL falhar (nenhuma tentativa produziu uma consulta válida), a pergunta quantitativa
  recebe uma mensagem de falha. A híbrida cai para o caminho conceitual, usando só os trechos
  do glossário, e a resposta avisa que os dados não foram obtidos.
- Se o SQL executar mas devolver vazio nas três tentativas, a resposta informa que a consulta
  não retornou linhas.

Dependências:
- todos os módulos de pipeline/online/, os índices e o esquema.json em embeddings/ e dados/dfp.db;
- servidor Ollama em execução, com o modelo qwen2.5-coder:7b baixado.

Parâmetros: a pergunta, em linguagem natural, como argumento de linha de comando.

Resultado: fornece executar(pergunta), que devolve a classe, o resultado do SQL, a resposta final
formatada, a validação e o tempo de cada passo. Pela linha de comando, imprime a resposta.

Executar de dentro de rag-nl2sql-financas-br/:
    python pipeline/online/main.py "Qual foi a receita líquida consolidada da Ambev em 2025?"
"""

import sys
import time

from classificador import classificar
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


def executar(pergunta: str) -> dict:
    tempos = {}
    inicio = time.time()
    classe = classificar(pergunta)
    tempos["classificador"] = time.time() - inicio

    resultado_sql = None
    if classe in ("quantitativa", "hibrida"):
        inicio = time.time()
        resultado_sql = gerar_sql(pergunta)
        tempos["sql"] = time.time() - inicio

    inicio = time.time()
    aviso = None
    if resultado_sql and resultado_sql["status"] == "erro":
        if classe == "quantitativa":
            resposta = {"resposta": FALHA_NO_SQL, "trechos": [], "contexto": ""}
        else:
            resposta = responder(pergunta, "conceitual")
            aviso = AVISO_HIBRIDA
        resultado_para_resposta = None
    else:
        resposta = responder(pergunta, classe, resultado_sql)
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
        "resultado_sql": resultado_sql,
        "resposta": resposta_final,
        "validacao": validacao,
        "tempos": tempos,
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        print('Uso: python pipeline/online/main.py "pergunta em linguagem natural"')
        sys.exit(1)
    pergunta = " ".join(sys.argv[1:])

    inicio = time.time()
    r = executar(pergunta)
    tempo = time.time() - inicio

    print(f"Pergunta: {r['pergunta']}")
    print(f"Classificada como: {r['classe']}")
    if r["resultado_sql"]:
        print(f"Tabelas do contexto: {r['resultado_sql']['tabelas_contexto']}")
        print(f"Status do SQL: {r['resultado_sql']['status']} ({len(r['resultado_sql']['tentativas'])} tentativa(s))")
    print(f"Fidelidade: {r['validacao']['fidelidade']['score']:.2f} | aprovado: {r['validacao']['aprovado']}")
    print(f"\n{r['resposta']}")
    print(f"\n({tempo:.1f} s)")


if __name__ == "__main__":
    main()
