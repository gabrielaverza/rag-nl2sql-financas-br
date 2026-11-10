"""
Etapa: Online.

Função: divide uma pergunta híbrida em duas partes, a de dados (que vai à busca de tabelas e ao SQL) e a
conceitual (que vai à busca de trechos dos normativos), usando o LLM local.

Descrição geral do funcionamento:
- Um LLM local recebe as regras de divisão e alguns exemplos (few-shot) e responde com um JSON
  {"parte_dados": ..., "parte_conceitual": ...}, como o classificador faz com a classe.
- Os exemplos do prompt NÃO são perguntas do conjunto oficial (dados/perguntas_validacao.json) e usam
  outras companhias e outros conceitos, para a avaliação não ser contaminada. Os do classificador
  (classificador.py) seguem a mesma regra.
- Cada parte é conferida: não pode estar vazia nem trazer muitas palavras que não estejam na pergunta
  (o LLM não deve inventar conceitos). A parte conceitual também não pode ter pronomes
  demonstrativos ou pessoais ("esse", "nessa", "ele"), que perdem o sentido longe da parte de dados e
  tiram da busca as palavras do conceito. Se houver pronome, a nova tentativa recebe o problema de
  volta (como o gerador de SQL faz com o erro) e pede a troca do pronome pelo substantivo.
- Se a resposta continuar inválida em TENTATIVAS chamadas, a divisão é descartada e as duas partes
  viram a pergunta inteira (dividida = False), como o classificador que cai na classe padrão.
- O orquestrador (main.py) chama este módulo depois do classificador, só nas perguntas híbridas e
  só se divisor.ativo for true em configuracao.json. A parte de dados não muda a busca das tabelas
  (que usa a pergunta inteira); a parte conceitual busca os trechos. O parâmetro bloquear_pronomes
  (divisor.bloquear_pronomes) liga ou desliga a recusa de pronomes, para medir a influência dela.

Dependências:
- servidor Ollama em execução, com o modelo qwen2.5-coder:7b baixado;
- biblioteca ollama; pipeline/online/busca_bm25.py (só a função tirar_acentos).

Parâmetros: nenhum argumento. É um módulo importado.

Resultado: fornece dividir(pergunta, bloquear_pronomes), que devolve parte_dados, parte_conceitual e dividida.
"""

import json
import re

import ollama

from busca_bm25 import tirar_acentos

MODELO_LLM = "qwen2.5-coder:7b"
TENTATIVAS = 2
# Fração máxima de palavras de uma parte que podem não estar na pergunta original (a troca de um
# pronome pelo substantivo introduz poucas palavras novas).
LIMITE_PALAVRAS_NOVAS = 0.25
# Pronomes que não podem aparecer na parte conceitual (comparados em minúsculas, com acento, para
# não confundir "esta" com "está").
PRONOMES = {
    "esse", "essa", "esses", "essas", "nesse", "nessa", "nesses", "nessas", "desse", "dessa", "desses", "dessas",
    "este", "esta", "estes", "estas", "neste", "nesta", "nestes", "nestas", "deste", "desta", "destes", "destas",
    "isso", "disso", "nisso", "isto", "disto", "nisto", "ele", "ela", "eles", "elas", "dele", "dela", "deles", "delas",
}

INSTRUCOES = """Você separa perguntas feitas a um sistema que responde sobre as demonstrações financeiras (DFP) de companhias abertas brasileiras e sobre os normativos contábeis da CVM e do CPC.

Cada pergunta que você recebe tem duas partes, ligadas por "e" ou por vírgula:
- a parte de dados: pede um valor ou uma contagem tirada das demonstrações (cita a companhia, o ano e a conta);
- a parte conceitual: pede um significado, uma definição, uma função ou uma regra, que se responde com os normativos.

Separe a pergunta nessas duas partes seguindo estas regras:
- Use as palavras da pergunta original. Não acrescente conceitos, termos, companhias ou números que não estejam nela.
- A parte conceitual não leva nomes de companhias, anos nem valores.
- Se uma parte tiver um pronome como "esse", "essa", "ele", "ela" ou "essas companhias", troque-o pelo substantivo a que se refere, ou tire a referência.
- Cada parte deve ser uma pergunta completa e curta, terminando em "?".

Responda somente com um JSON no formato {"parte_dados": "<pergunta>", "parte_conceitual": "<pergunta>"}.

Exemplos:
Pergunta: Qual foi o resultado operacional da Embraer em 2025 e o que são partes relacionadas?
{"parte_dados": "Qual foi o resultado operacional da Embraer em 2025?", "parte_conceitual": "O que são partes relacionadas?"}

Pergunta: Quanto a Magazine Luiza tinha em estoques ao fim de 2025, e o que as políticas contábeis significativas devem divulgar segundo o CPC 26?
{"parte_dados": "Quanto a Magazine Luiza tinha em estoques ao fim de 2025?", "parte_conceitual": "O que as políticas contábeis significativas devem divulgar segundo o CPC 26?"}

Pergunta: Qual o saldo de reservas de lucros da Localiza e como é apresentado o resultado por ação?
{"parte_dados": "Qual o saldo de reservas de lucros da Localiza?", "parte_conceitual": "Como é apresentado o resultado por ação?"}

Pergunta: Entre Braskem e Marfrig, qual teve o maior ativo intangível em 2025, e o que diferencia uma demonstração individual de uma consolidada?
{"parte_dados": "Entre Braskem e Marfrig, qual teve o maior ativo intangível em 2025?", "parte_conceitual": "O que diferencia uma demonstração individual de uma consolidada?"}

Pergunta: Qual foi o saldo de ajustes de avaliação patrimonial da Eletrobras em 2025 e o que ele representa?
{"parte_dados": "Qual foi o saldo de ajustes de avaliação patrimonial da Eletrobras em 2025?", "parte_conceitual": "O que representam os ajustes de avaliação patrimonial?"}"""


# Devolve o texto sem espaços nas pontas se for texto, ou string vazia caso contrário.
def _texto(valor) -> str:
    return valor.strip() if isinstance(valor, str) else ""


# Palavras (minúsculas, sem acento, com mais de 2 letras) de um texto.
def _palavras(texto: str) -> list[str]:
    return [p for p in re.findall(r"[^\W_]+", tirar_acentos(texto.lower())) if len(p) > 2]


# Fração das palavras da parte que não aparecem na pergunta original.
def _fracao_nova(pergunta: str, parte: str) -> float:
    originais = set(_palavras(pergunta))
    palavras = _palavras(parte)
    return sum(1 for p in palavras if p not in originais) / len(palavras) if palavras else 1.0


# Pronomes da lista que aparecem no texto.
def _pronomes(texto: str) -> list[str]:
    return sorted({p for p in re.findall(r"[^\W_]+", texto.lower()) if p in PRONOMES})


# Divide a pergunta em parte de dados e parte conceitual; se não conseguir, devolve a pergunta inteira nas duas.
# Com bloquear_pronomes (padrão, divisor.bloquear_pronomes de configuracao.json), uma parte conceitual
# com pronome é recusada e o LLM é chamado de novo; sem ele, a parte é aceita com o pronome.
def dividir(pergunta: str, bloquear_pronomes: bool = True) -> dict:
    mensagens = [{"role": "system", "content": INSTRUCOES}, {"role": "user", "content": f"Pergunta: {pergunta}"}]
    for tentativa in range(1, TENTATIVAS + 1):
        resposta = ollama.chat(
            model=MODELO_LLM,
            messages=mensagens,
            format="json",
            options={"temperature": 0 if tentativa == 1 else 0.4, "num_predict": 300},
        )
        conteudo = resposta["message"]["content"]
        try:
            bruto = json.loads(conteudo)
            dados, conceito = _texto(bruto.get("parte_dados")), _texto(bruto.get("parte_conceitual"))
        except (json.JSONDecodeError, AttributeError):
            continue
        pronomes = _pronomes(conceito) if bloquear_pronomes else []
        if dados and conceito and not pronomes and max(_fracao_nova(pergunta, dados), _fracao_nova(pergunta, conceito)) <= LIMITE_PALAVRAS_NOVAS:
            return {"parte_dados": dados, "parte_conceitual": conceito, "dividida": True}
        if pronomes:
            # Devolve o problema ao LLM, como o gerador de SQL faz com o erro do SQL.
            mensagens += [
                {"role": "assistant", "content": conteudo},
                {"role": "user", "content": (
                    f"A parte conceitual ainda tem o pronome {', '.join(pronomes)}. Reescreva as duas partes trocando "
                    "o pronome pelo substantivo a que ele se refere, ou tirando a referência, e responda de novo só com o JSON."
                )},
            ]
    return {"parte_dados": pergunta, "parte_conceitual": pergunta, "dividida": False}
