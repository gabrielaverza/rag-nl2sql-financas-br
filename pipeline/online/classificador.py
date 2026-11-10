"""
Etapa: Online.

Função: classifica a intenção da pergunta em conceitual, quantitativa ou híbrida.

Descrição geral do funcionamento:
- Um LLM local recebe a definição de cada classe e alguns exemplos (few-shot) e
  responde com um JSON {"tipo": ...}.
- O tipo decide o caminho: conceitual usa só o índice do glossário (normativos),
  quantitativa usa o índice de esquema e o SQL sobre as DFPs, e híbrida usa os dois.
- Os exemplos NÃO fazem parte de dados/perguntas_validacao.json, para a avaliação não ser contaminada
  (nenhum é uma pergunta do conjunto e nenhum usa companhia ou conceito normativo de alguma delas).
- As híbridas dos exemplos têm estruturas variadas (dados e depois explicação, explicação e depois
  dados, duas frases, comparação entre companhias, pergunta de regra e citação de norma). Há também
  quantitativas e conceituais de forma parecida, mas sem as duas partes, para o modelo não tratar o
  "e" ou o nome de uma companhia como sinal de pergunta híbrida.
- Se o LLM não devolver uma classe válida em TENTATIVAS chamadas, a pergunta é
  tratada como híbrida.

Dependências:
- servidor Ollama em execução, com o modelo qwen2.5-coder:7b baixado;
- biblioteca ollama.

Parâmetros: nenhum argumento. É um módulo importado pelos demais passos do estágio online.

Resultado: fornece classificar(pergunta), que devolve "conceitual", "quantitativa" ou "hibrida".
"""

import json
import unicodedata

import ollama

MODELO_LLM = "qwen2.5-coder:7b"
CLASSES = ("conceitual", "quantitativa", "hibrida")
# Se o LLM não devolver uma classe válida, usa a híbrida: ela aciona os dois
# caminhos, então um erro aqui custa mais tempo, mas não deixa a resposta sem contexto.
CLASSE_PADRAO = "hibrida"
TENTATIVAS = 2

INSTRUCOES = """Você classifica perguntas feitas a um sistema que responde sobre as demonstrações financeiras (DFP) de companhias abertas brasileiras, guardadas em um banco de dados, e sobre os normativos contábeis da CVM e do CPC que explicam essas demonstrações.

Classifique a pergunta em uma destas classes:
- conceitual: pergunta sobre o significado, a definição, a função ou o funcionamento de algo, que se responde com os normativos. Não pede valores nem listas calculados a partir dos dados das companhias.
- quantitativa: pergunta que pede valores das demonstrações (receita, lucro, ativo, caixa, dívida e semelhantes), contagens, somas, médias, rankings ou comparações entre companhias, calculados a partir dos dados do banco.
- hibrida: pergunta que pede os dois ao mesmo tempo: um valor calculado a partir dos dados e também uma explicação ou definição que está nos normativos.

Responda somente com um JSON, por exemplo {"tipo": "conceitual"}. Os valores possíveis são "conceitual", "quantitativa" e "hibrida"."""

# (pergunta, classe). As classes estão intercaladas de propósito, para o modelo
# não tender a repetir a última classe que viu. As híbridas variam de estrutura e a
# parte conceitual nem sempre trata da conta pedida; as quantitativas e conceituais
# com "e" ou com nome de companhia mostram que isso sozinho não torna a pergunta híbrida.
EXEMPLOS = [
    ("Para que serve a demonstração das mutações do patrimônio líquido?", "conceitual"),
    ("Qual foi o resultado operacional da Embraer em 2025?", "quantitativa"),
    ("Explique o que são reservas de lucros e mostre o saldo delas na Braskem em 2025.", "hibrida"),
    ("Quanto a Eneva tinha de caixa e equivalentes de caixa ao fim de 2025? Explique também o que é uma mudança de política contábil.", "hibrida"),
    ("Qual a diferença entre demonstrações consolidadas e individuais?", "conceitual"),
    ("Qual companhia tinha o maior ativo intangível consolidado ao fim de 2025?", "quantitativa"),
    ("Qual o capital social da Localiza em 2025 e o que ele representa no balanço?", "hibrida"),
    ("Entre a CSN e a Light, qual teve o maior ativo total consolidado em 2025, e o que são instrumentos financeiros derivativos?", "hibrida"),
    ("O que são ajustes de avaliação patrimonial?", "conceitual"),
    ("Qual o saldo de provisões da Sabesp no balanço individual de 2025?", "quantitativa"),
    ("Compare o estoque consolidado da Cemig e da Sabesp em 2025 e explique como o estoque é apresentado no balanço.", "hibrida"),
    ("Como uma entidade deve tratar os eventos ocorridos após o período de reporte, e qual foi o lucro líquido consolidado da Light em 2025?", "hibrida"),
    ("O que é o ajuste a valor presente e quando ele deve ser aplicado?", "conceitual"),
    ("Qual foi a receita líquida e o lucro líquido consolidados da CSN em 2025?", "quantitativa"),
    ("Qual foi o resultado de equivalência patrimonial da Raízen em 2025, e o que a entidade deve divulgar sobre ações em tesouraria?", "hibrida"),
    ("Qual o saldo de arrendamentos a pagar da CSN no fim de 2025, e a entidade pode reconhecer uma provisão sem obrigação presente?", "hibrida"),
    ("Segundo a CVM, uma companhia aberta como a Light precisa ter um comitê de auditoria?", "conceitual"),
    ("Quanto a Eneva pagou de tributos sobre o lucro em 2025, no consolidado, e qual foi a variação em relação a 2024?", "quantitativa"),
    ("Quanto a Magazine Luiza pagou de juros sobre o capital próprio em 2025, e o que o conselho fiscal deve examinar nas demonstrações?", "hibrida"),
    ("Segundo a Resolução CVM 80, em que casos o emissor deve divulgar fato relevante, e qual foi a receita líquida consolidada da Eneva em 2025?", "hibrida"),
    ("O que caracteriza um segmento operacional?", "conceitual"),
    ("Entre a CSN e a Light, qual teve a maior dívida bruta consolidada em 2025?", "quantitativa"),
    ("Qual era a dívida bruta consolidada da Energisa ao fim de 2025, e em que prazo a companhia deve divulgar o aviso aos acionistas sobre a disponibilização das demonstrações?", "hibrida"),
]


# Aceita variações como "Híbrida", "hibrido" ou "Quantitativa" e devolve a classe
# no formato padrão (ou None se não reconhecer).
def _normalizar_rotulo(valor) -> str | None:
    if not isinstance(valor, str):
        return None
    texto = unicodedata.normalize("NFKD", valor.strip().lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    for classe in CLASSES:
        if texto == classe or texto.startswith(classe[:5]):
            return classe
    return None


# Monta a conversa: as instruções, cada exemplo como uma pergunta do usuário
# seguida da resposta certa, e por fim a pergunta a classificar.
def _montar_mensagens(pergunta: str) -> list[dict]:
    mensagens = [{"role": "system", "content": INSTRUCOES}]
    for exemplo, classe in EXEMPLOS:
        mensagens.append({"role": "user", "content": exemplo})
        mensagens.append({"role": "assistant", "content": json.dumps({"tipo": classe})})
    mensagens.append({"role": "user", "content": pergunta})
    return mensagens


# Devolve a classe da pergunta, ou a classe padrão se o LLM não responder uma classe válida.
def classificar(pergunta: str) -> str:
    mensagens = _montar_mensagens(pergunta)
    for tentativa in range(1, TENTATIVAS + 1):
        resposta = ollama.chat(
            model=MODELO_LLM,
            messages=mensagens,
            format="json",
            # Temperatura 0 deixa a resposta previsível; na nova tentativa há um
            # pouco de variação, para não repetir a mesma resposta inválida.
            options={"temperature": 0 if tentativa == 1 else 0.4},
        )
        try:
            classe = _normalizar_rotulo(json.loads(resposta["message"]["content"]).get("tipo"))
        except (json.JSONDecodeError, AttributeError):
            classe = None
        if classe:
            return classe
    return CLASSE_PADRAO
