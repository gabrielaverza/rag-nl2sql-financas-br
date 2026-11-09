"""
Etapa: Online.

Função: síntese da resposta em linguagem natural.

Descrição geral do funcionamento:
- Conceitual: recupera os trechos mais parecidos no índice do glossário (normativos), pela busca
  configurável (método e k_trechos de configuracao.json), e o LLM responde só com eles, sem SQL.
- Quantitativa: o LLM escreve a resposta a partir do resultado do SQL.
- Híbrida: o LLM combina o resultado do SQL com os trechos do glossário.
- O resultado do SQL vai ao LLM com cada valor rotulado pelo nome da coluna, e os valores
  monetários em pt-BR (já estão em reais, pela regra do gerador de SQL).
- Em todos os casos o LLM só pode usar o que recebeu no prompt. Se os trechos não trouxerem
  a informação pedida, ele deve dizer que não encontrou.

Dependências:
- pipeline/online/busca_configuravel.py e os índices do glossário em embeddings/;
- servidor Ollama em execução, com o modelo qwen2.5-coder:7b baixado.

Parâmetros: nenhum argumento. É um módulo importado pelos demais passos do estágio online.

Resultado: fornece responder(pergunta, tipo, resultado_sql, consulta_trechos), que devolve a resposta, os trechos
usados e o contexto mostrado ao LLM (a validação confere a fidelidade contra ele).
"""

import re

import ollama

from busca_configuravel import obter_busca

MODELO_LLM = "qwen2.5-coder:7b"
MAX_LINHAS_NO_PROMPT = 30
# Sem limite de tokens, o 7B pode repetir texto sem parar (já aconteceu, passou de 19 mil tokens);
# a resposta pedida tem até 5 frases. O contexto passa do tamanho padrão do Ollama (4096) com
# resultados longos.
MAX_TOKENS_RESPOSTA = 400
NUM_CTX = 8192
SEM_INFORMACAO = "Não encontrei essa informação nos documentos disponíveis."

# Os k trechos do glossário mais parecidos com o texto da busca, pela busca configurável
# (k padrão: k_trechos de configuracao.json).
def recuperar_trechos(pergunta: str, k: int | None = None, tipo: str | None = None) -> list[dict]:
    return obter_busca().trechos(pergunta, tipo, k)


def _formatar_trechos(trechos: list[dict]) -> str:
    return "\n\n".join(f"[{i}] (fonte: {t['fonte']})\n{t['conteudo']}" for i, t in enumerate(trechos, start=1))


# Escreve o número no padrão brasileiro: decimais (valores monetários) com 2 casas, inteiros sem casas.
def formatar_valor(valor) -> str:
    if isinstance(valor, float):
        return f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return str(valor)


# Colunas sem alias levam a própria expressão como nome (ex.: CASE WHEN ...); nesse caso o
# rótulo vira "valor".
def rotular_colunas(colunas: list[str]) -> list[str]:
    return [c if re.fullmatch(r"\w+", c) else "valor" for c in colunas]


# Mostra a consulta e o resultado com cada valor rotulado pelo nome da coluna, para o LLM
# não precisar adivinhar o que cada número significa.
def _formatar_resultado(resultado_sql: dict) -> str:
    linhas = resultado_sql["linhas"]
    texto = f"Consulta executada: {resultado_sql['sql']}\n"
    if not linhas:
        return texto + "Resultado: a consulta não retornou nenhuma linha."
    texto += "Resultado (valores monetários em reais):\n"
    rotulos = rotular_colunas(resultado_sql["colunas"])
    for linha in linhas[:MAX_LINHAS_NO_PROMPT]:
        texto += "- " + "; ".join(f"{c}: {formatar_valor(v)}" for c, v in zip(rotulos, linha)) + "\n"
    if len(linhas) > MAX_LINHAS_NO_PROMPT:
        texto += f"(mostrando as primeiras {MAX_LINHAS_NO_PROMPT} de {len(linhas)} linhas)\n"
    return texto


def _instrucoes(tem_dados: bool, tem_trechos: bool) -> str:
    regras = ["Responda em português, de forma direta, em até 5 frases, em texto simples e sem títulos.",
              "Use somente as informações fornecidas abaixo. Não use conhecimento externo.",
              "Não acrescente unidades, moedas, interpretações ou informações que não estejam escritas nas informações fornecidas.",
              "Não traduza nomes de tabelas, colunas, categorias ou outros nomes próprios: escreva-os exatamente como aparecem."]
    if tem_dados:
        regras.append("Para os números, copie os valores exatamente como aparecem no resultado da consulta. "
                      "Os valores monetários estão em reais: escreva R$ antes deles.")
        regras.append("Se o resultado tiver mais de uma companhia, cite cada uma com o seu valor.")
    if tem_dados and tem_trechos:
        regras.append("A pergunta tem duas partes: responda a parte de dados com o resultado da consulta "
                      "e a parte de explicação com os trechos.")
        regras.append(f'Se os trechos não tiverem a informação da parte de explicação, responda mesmo assim a parte de '
                      f'dados e, para a explicação, diga: "{SEM_INFORMACAO}"')
    elif tem_trechos:
        regras.append(f'Se os trechos não tiverem a informação pedida, diga: "{SEM_INFORMACAO}"')
    return "Você responde perguntas sobre demonstrações financeiras de companhias abertas brasileiras.\n" \
        + "\n".join(f"- {r}" for r in regras)


# tipo: "conceitual", "quantitativa" ou "hibrida". resultado_sql é o dicionário devolvido por
# gerador_sql.gerar_sql (necessário nos tipos quantitativa e híbrida). consulta_trechos é o texto
# que busca os trechos; sem ele, a busca usa a pergunta inteira (com o divisor ligado, o
# orquestrador passa a parte conceitual da pergunta híbrida).
def responder(pergunta: str, tipo: str, resultado_sql: dict | None = None, consulta_trechos: str | None = None) -> dict:
    trechos = recuperar_trechos(consulta_trechos or pergunta, tipo=tipo) if tipo in ("conceitual", "hibrida") else []
    tem_dados = tipo in ("quantitativa", "hibrida")
    if tem_dados and resultado_sql is None:
        raise ValueError("resultado_sql é obrigatório nos tipos quantitativa e híbrida")

    partes_contexto = []
    if tem_dados:
        partes_contexto.append("DADOS DO BANCO\n" + _formatar_resultado(resultado_sql))
    if trechos:
        partes_contexto.append("TRECHOS DE DOCUMENTOS\n" + _formatar_trechos(trechos))
    contexto = "\n\n".join(partes_contexto)

    resposta = ollama.chat(
        model=MODELO_LLM,
        messages=[
            {"role": "system", "content": _instrucoes(tem_dados, bool(trechos))},
            {"role": "user", "content": contexto + f"\n\nPERGUNTA: {pergunta}"},
        ],
        options={"temperature": 0, "num_ctx": NUM_CTX, "num_predict": MAX_TOKENS_RESPOSTA},
    )
    # O contexto é devolvido junto, para o validador conferir a fidelidade da resposta
    # exatamente contra o que foi mostrado ao LLM.
    return {"resposta": resposta["message"]["content"].strip(), "trechos": trechos, "contexto": contexto}
