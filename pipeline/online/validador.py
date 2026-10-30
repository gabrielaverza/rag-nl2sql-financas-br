"""
Etapa: Online.

Função: validação da resposta antes da entrega.

Descrição geral do funcionamento:
- Duas verificações, aplicadas juntas:
  - Relevância do contexto: os trechos recuperados do glossário realmente ajudam a responder a
    pergunta (inspirado no Validator Agent do samuelsen2026, que descarta trechos que não
    contribuem para a resposta).
  - Fidelidade da resposta: as afirmações da resposta são sustentadas pelo contexto (resultado
    do SQL e/ou trechos), seguindo a métrica de faithfulness do RAGAS (es2025): extrai as
    afirmações da resposta e verifica cada uma contra o contexto. É uma adaptação de uma
    métrica de avaliação para uma etapa de validação em tempo de resposta, e não uma
    reprodução exata do RAGAS.
- A resposta é aprovada se a fidelidade ficar no limiar ou acima dele.

Dependências:
- servidor Ollama em execução, com o modelo qwen2.5-coder:7b baixado;
- biblioteca ollama.

Parâmetros: nenhum argumento. É um módulo importado pelos demais passos do estágio online.

Resultado: fornece validar(pergunta, resposta, contexto, trechos), que devolve a relevância do
contexto, a fidelidade e a marca de aprovação.
"""

import json

import ollama

MODELO_LLM = "qwen2.5-coder:7b"
# Abaixo deste valor a resposta é sinalizada como não confiável. É uma escolha da implementação.
LIMIAR_FIDELIDADE = 0.8
# Limites das chamadas ao LLM: evitam geração sem fim e o corte do contexto no tamanho padrão do Ollama.
OPCOES_LLM = {"temperature": 0, "num_ctx": 8192, "num_predict": 600}


# Para cada trecho, pergunta ao LLM se ele contribui para responder à pergunta. Os trechos são
# numerados e julgados em uma única chamada.
def validar_relevancia_contexto(pergunta: str, trechos: list[dict]) -> dict:
    if not trechos:
        return {"score": None, "por_trecho": []}

    listagem = "\n\n".join(f"[{i}] {t['conteudo']}" for i, t in enumerate(trechos, start=1))
    prompt = f"""Pergunta: {pergunta}

Trechos recuperados:
{listagem}

Para cada trecho numerado, diga se ele contribui para responder à pergunta
(mesmo que parcialmente) ou se é irrelevante para ela.

Responda somente com um JSON no formato {{"relevantes": [1, 3]}}, com os
números dos trechos que contribuem. Se nenhum contribuir, responda {{"relevantes": []}}."""

    resposta = ollama.chat(
        model=MODELO_LLM,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options=OPCOES_LLM,
    )
    try:
        relevantes = set(json.loads(resposta["message"]["content"]).get("relevantes", []))
    except json.JSONDecodeError:
        relevantes = set()

    por_trecho = [{"numero": i, "id": t["id"], "relevante": i in relevantes} for i, t in enumerate(trechos, start=1)]
    return {"score": len(relevantes) / len(trechos), "por_trecho": por_trecho}


# Decompõe a resposta em afirmações curtas e verificáveis, como no passo de extração de
# statements do RAGAS.
def _extrair_afirmacoes(resposta: str) -> list[str]:
    prompt = f"""Divida o texto abaixo em afirmações curtas e independentes, uma para
cada informação factual que ele apresenta. Ignore frases que são só
recusa ou cortesia (por exemplo, "não encontrei essa informação").

Texto:
{resposta}

Responda somente com um JSON no formato {{"afirmacoes": ["...", "..."]}}.
Se o texto não tiver nenhuma afirmação factual, responda {{"afirmacoes": []}}."""

    resultado = ollama.chat(
        model=MODELO_LLM,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options=OPCOES_LLM,
    )
    try:
        afirmacoes = json.loads(resultado["message"]["content"]).get("afirmacoes", [])
    except json.JSONDecodeError:
        afirmacoes = []
    return [a for a in afirmacoes if isinstance(a, str) and a.strip()]


# Verifica, em uma única chamada, se cada afirmação pode ser inferida do contexto (resultado do
# SQL e/ou trechos do glossário).
def _verificar_afirmacoes(afirmacoes: list[str], contexto: str) -> list[bool]:
    listagem = "\n".join(f"{i}. {a}" for i, a in enumerate(afirmacoes, start=1))
    prompt = f"""Contexto:
{contexto}

Afirmações a verificar:
{listagem}

Para cada afirmação numerada, diga se ela pode ser sustentada pelo contexto
acima. Uma afirmação com um número, nome ou palavra que não está no contexto
não é sustentada, mesmo que pareça razoável.

Responda somente com um JSON no formato {{"sustentadas": [1, 2]}}, com os
números das afirmações sustentadas pelo contexto."""

    resposta = ollama.chat(
        model=MODELO_LLM,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options=OPCOES_LLM,
    )
    try:
        sustentadas = set(json.loads(resposta["message"]["content"]).get("sustentadas", []))
    except json.JSONDecodeError:
        sustentadas = set()
    return [i in sustentadas for i in range(1, len(afirmacoes) + 1)]


def validar_fidelidade(resposta: str, contexto: str) -> dict:
    afirmacoes = _extrair_afirmacoes(resposta)
    if not afirmacoes:
        # Nada a verificar (por exemplo, uma recusa): não há afirmação para ser infiel, então a
        # fidelidade é máxima, como na definição do RAGAS.
        return {"score": 1.0, "afirmacoes": []}
    sustentadas = _verificar_afirmacoes(afirmacoes, contexto)
    detalhe = [{"afirmacao": a, "sustentada": s} for a, s in zip(afirmacoes, sustentadas)]
    return {"score": sum(sustentadas) / len(afirmacoes), "afirmacoes": detalhe}


# Junta as duas verificações e uma marca de aprovação, usada como sinal para o formatador
# (por exemplo, avisar o usuário quando a fidelidade ficar abaixo do limiar).
def validar(pergunta: str, resposta: str, contexto: str, trechos: list[dict]) -> dict:
    relevancia = validar_relevancia_contexto(pergunta, trechos)
    fidelidade = validar_fidelidade(resposta, contexto)
    return {
        "relevancia_contexto": relevancia,
        "fidelidade": fidelidade,
        "aprovado": fidelidade["score"] >= LIMIAR_FIDELIDADE,
    }
