"""
Etapa: Avaliação (RAGAS adaptado e conferência dos números).

Função: pontua as respostas de um experimento do nível pipeline. Dá fidelidade e relevância da resposta
(RAGAS, biblioteca ragas) às perguntas conceituais e híbridas, e confere por código, sem LLM, se os
números do resultado do SQL aparecem no texto da resposta (perguntas com dados).

Descrição geral do funcionamento:
- Lê o respostas.json que o executor grava no nível pipeline (texto da resposta, contexto que o LLM
  recebeu e resultado do SQL) e grava ragas.csv e ragas.md na mesma pasta do experimento.
- Fidelidade (faithfulness): fração das afirmações da resposta que podem ser deduzidas do contexto.
  Relevância da resposta (answer relevancy): o LLM gera perguntas a partir da resposta, e a nota é a
  similaridade média delas com a pergunta original, pelo cosseno dos embeddings. As duas seguem as
  definições do artigo da RAGAS (es2025). A relevância do contexto não é usada: a recuperação já é
  medida com gabarito pelas quatro métricas de recuperação.
- O juiz é o Qwen local (o mesmo modelo que escreve as respostas) pelo endereço do Ollama compatível com
  a OpenAI; os embeddings são os do E5 local, com o prefixo de pergunta nos dois lados da comparação.
- Adaptações e limites a declarar: o artigo usa o gpt-3.5-turbo e a biblioteca usa o juiz que se
  entrega a ela; a ragas 0.4.3 só importa com um contorno (ver contornar_vertexai); o juiz é um modelo
  pequeno que julga texto que ele mesmo escreveu, então a nota mede sustentação pelo contexto, e não
  a correção do dado; uma resposta de recusa recebe nota própria da biblioteca; falhas do juiz ficam
  sem nota, e o relatório as conta.
- Conferência dos números: conta quantos valores numéricos do resultado do SQL (até 30 linhas)
  aparecem no texto da resposta, comparando só os dígitos. Não é métrica de artigo.

Dependências:
- ragas 0.4.3, openai (cliente) e sentence-transformers; servidor Ollama com o modelo qwen2.5-coder:7b;
- metricas/<experimento>/respostas.json, gerado por executar_experimentos.py (nível pipeline).

Parâmetros: o nome do experimento; --rascunho (lê de metricas/rascunhos/); --ids q01,c01,... (só
essas perguntas).

Resultado: ragas.csv (uma linha por resposta e métrica) e ragas.md (médias por configuração e tipo,
número de respostas sem nota e as declarações acima), na pasta do experimento.

Executar de dentro de rag-nl2sql-financas-br/:
    python pipeline/avaliacao/ragas_avaliacao.py teste_pipeline --rascunho
    python pipeline/avaliacao/ragas_avaliacao.py pipeline_gargalos
"""

import argparse
import asyncio
import csv
import json
import math
import os
import sys
import time
import types
import warnings
from datetime import datetime
from importlib.metadata import version
from statistics import mean

# Os módulos do fluxo ficam em pipeline/online/ e os compartilhados em pipeline/.
PASTA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PASTA, "..", "online"))
sys.path.insert(0, os.path.join(PASTA, ".."))

from metricas import fmt, numeros_presentes, tabela_md  # noqa: E402

MODELO_JUIZ = "qwen2.5-coder:7b"
ENDERECO_OLLAMA = "http://localhost:11434/v1"
PASTA_SAIDA = "metricas"
PASTA_RASCUNHOS = os.path.join(PASTA_SAIDA, "rascunhos")
TIPOS_COM_TEXTO = ("conceitual", "hibrida")
TIPOS_COM_DADOS = ("quantitativa", "hibrida")
PLURAL = {"quantitativa": "quantitativas", "conceitual": "conceituais", "hibrida": "híbridas"}
METRICAS = (
    ("ragas_fidelidade", "Fidelidade (RAGAS)"),
    ("ragas_relevancia_resposta", "Relevância da resposta (RAGAS)"),
    ("numeros_no_texto", "Números do resultado no texto (por código)"),
)


# A ragas 0.4.3 importa classes do Vertex AI que o langchain-community atual removeu. Este contorno cria,
# antes da importação, um módulo falso com uma classe vazia no lugar delas; só serve a uma checagem interna
# da ragas e não afeta o uso aqui. É frágil: precisa constar na descrição das versões usadas.
def contornar_vertexai():
    modulo = types.ModuleType("langchain_community.chat_models.vertexai")
    modulo.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = modulo


# Monta as duas métricas da ragas com o Qwen local como juiz e o E5 local como modelo de embeddings.
def criar_metricas():
    contornar_vertexai()
    warnings.filterwarnings("ignore")
    from openai import AsyncOpenAI
    from ragas.embeddings.base import BaseRagasEmbedding
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, Faithfulness

    import modelos_embedding

    class EmbeddingsLocais(BaseRagasEmbedding):
        # Usa o prefixo de pergunta do E5 nos dois lados (pergunta original e perguntas geradas).
        def embed_text(self, text: str, **kwargs) -> list[float]:
            return modelos_embedding.vetorizar_pergunta(text).tolist()

        async def aembed_text(self, text: str, **kwargs) -> list[float]:
            return await asyncio.to_thread(self.embed_text, text)

    cliente = AsyncOpenAI(base_url=ENDERECO_OLLAMA, api_key="ollama")
    llm = llm_factory(MODELO_JUIZ, provider="openai", client=cliente)
    return Faithfulness(llm=llm), AnswerRelevancy(llm=llm, embeddings=EmbeddingsLocais())


# Devolve a nota de uma métrica da ragas, ou (None, mensagem) se o juiz falhar ou devolver valor inválido.
async def nota(chamada) -> tuple[float | None, str]:
    try:
        resultado = await chamada
        valor = float(resultado.value)
    except Exception as erro:  # a biblioteca levanta tipos variados quando o juiz não responde no formato
        return None, f"{type(erro).__name__}: {str(erro)[:120]}"
    if math.isnan(valor):
        return None, "nota vazia (NaN)"
    return valor, ""


# Pontua todas as respostas e devolve as linhas (uma por resposta e métrica).
async def pontuar(respostas: list[dict], ids: set | None, execucao: str, experimento: str) -> list[dict]:
    fidelidade, relevancia = criar_metricas()
    linhas = []
    selecionadas = [r for r in respostas if not ids or r["id"] in ids]
    for i, r in enumerate(selecionadas, start=1):
        base = {"experimento": experimento, "execucao": execucao, "rodada": r["rodada"], "configuracao": r["configuracao"], "id_pergunta": r["id"], "tipo": r["tipo"]}
        print(f"[{i}/{len(selecionadas)}] {r['configuracao']} / {r['id']} ({r['tipo']})", file=sys.stderr)
        texto = r.get("texto_resposta")
        if texto is None:
            linhas.append({**base, "metrica": "ragas_fidelidade", "valor": "", "erro": "respostas.json sem texto_resposta (gerado por uma versão antiga do executor)"})
            continue
        if r["tipo"] in TIPOS_COM_TEXTO:
            contextos = [r["contexto"]] if r.get("contexto") else []
            inicio = time.time()
            valor, erro = await nota(fidelidade.ascore(user_input=r["pergunta"], response=texto, retrieved_contexts=contextos))
            linhas.append({**base, "metrica": "ragas_fidelidade", "valor": "" if valor is None else f"{valor:.6f}", "erro": erro})
            valor, erro = await nota(relevancia.ascore(user_input=r["pergunta"], response=texto))
            linhas.append({**base, "metrica": "ragas_relevancia_resposta", "valor": "" if valor is None else f"{valor:.6f}", "erro": erro})
            print(f"    {time.time() - inicio:.1f} s", file=sys.stderr)
        if r["tipo"] in TIPOS_COM_DADOS and r.get("resultado_sql"):
            achados, total = numeros_presentes(texto, [tuple(l) for l in r["resultado_sql"]["linhas"]])
            if total:
                linhas.append({**base, "metrica": "numeros_no_texto", "valor": f"{achados / total:.6f}", "erro": ""})
    return linhas


# Médias por configuração e tipo, com o número de respostas com nota e sem nota.
def resumir(linhas: list[dict]) -> dict:
    grupos = {}
    for l in linhas:
        grupos.setdefault((l["configuracao"], l["tipo"], l["metrica"]), []).append(l)
    resumo = {}
    for chave, itens in grupos.items():
        valores = [float(l["valor"]) for l in itens if l["valor"] != ""]
        resumo[chave] = {"media": mean(valores) if valores else None, "n": len(valores), "falhas": len(itens) - len(valores)}
    return resumo


# Monta o relatório em Markdown.
def montar_relatorio(experimento: str, pasta: str, linhas: list[dict], comando: str, execucao: str) -> str:
    resumo = resumir(linhas)
    configuracoes = list(dict.fromkeys(l["configuracao"] for l in linhas))
    t = [f"# Experimento {experimento}: RAGAS adaptado e conferência dos números", ""]
    t += [
        f"- Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')} por `{comando}`.",
        f"- Execução do experimento: `{execucao}`. Pasta: `{pasta}`.",
        f"- Juiz: `{MODELO_JUIZ}` (Ollama); embeddings: E5 local (`modelos_embedding`, prefixo de pergunta). Bibliotecas: ragas {version('ragas')}, openai {version('openai')}.",
        "- Fidelidade e relevância da resposta seguem as definições do artigo da RAGAS (es2025), calculadas pela biblioteca ragas com um juiz local; a relevância do contexto não é usada. A ragas 0.4.3 só importa com um contorno (módulo falso do Vertex AI), declarado em `contornar_vertexai`.",
        "- A nota mede sustentação pelo contexto, e não a correção do dado, e o juiz é o mesmo modelo que escreve as respostas. Convém medir a concordância dele com a autora na amostra de 20 respostas.",
        "- A conferência dos números é feita por código: fração dos valores numéricos do resultado do SQL (até 30 linhas) que aparecem no texto da resposta, comparando só os dígitos. Não é métrica de artigo.",
        "",
    ]
    n_secao = 1
    for tipo in ("conceitual", "hibrida", "quantitativa"):
        metricas = [(m, r) for m, r in METRICAS if any(k[1] == tipo and k[2] == m for k in resumo)]
        if not metricas:
            continue
        t += [f"## {n_secao}. Perguntas {PLURAL[tipo]}", ""]
        corpo = []
        for nome in configuracoes:
            linha = [nome]
            for m, _ in metricas:
                r = resumo.get((nome, tipo, m))
                linha.append("n/d" if not r else f"{fmt(r['media'])} (n = {r['n']}" + (f", {r['falhas']} sem nota)" if r["falhas"] else ")"))
            corpo.append(linha)
        t += tabela_md(["Configuração"] + [rotulo for _, rotulo in metricas], corpo)
        n_secao += 1
    erros = {}
    for l in linhas:
        if l["erro"]:
            erros[l["erro"]] = erros.get(l["erro"], 0) + 1
    if erros:
        t += [f"## {n_secao}. Respostas sem nota", ""] + [f"- {n} vez(es): {erro}" for erro, n in sorted(erros.items(), key=lambda x: -x[1])] + [""]
    return "\n".join(t)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Pontua com a RAGAS as respostas de um experimento do nível pipeline.")
    parser.add_argument("experimento")
    parser.add_argument("--rascunho", action="store_true", help="lê de metricas/rascunhos/")
    parser.add_argument("--ids", default=None, help="só estas perguntas, separadas por vírgula")
    args = parser.parse_args()

    pasta = os.path.join(PASTA_RASCUNHOS if args.rascunho else PASTA_SAIDA, args.experimento)
    caminho = os.path.join(pasta, "respostas.json")
    if not os.path.exists(caminho):
        sys.exit(f"{caminho} não existe: rode antes o experimento no nível pipeline.")
    with open(caminho, encoding="utf-8") as f:
        respostas = json.load(f)
    execucao = ""
    if os.path.exists(os.path.join(pasta, "execucao.json")):
        with open(os.path.join(pasta, "execucao.json"), encoding="utf-8") as f:
            execucao = json.load(f).get("execucao", "")
    ids = set(args.ids.split(",")) if args.ids else None

    linhas = asyncio.run(pontuar(respostas, ids, execucao, args.experimento))

    campos = ["experimento", "execucao", "rodada", "configuracao", "id_pergunta", "tipo", "metrica", "valor", "erro"]
    with open(os.path.join(pasta, "ragas.csv"), "w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(linhas)
    comando = "python pipeline/avaliacao/ragas_avaliacao.py " + " ".join(sys.argv[1:])
    relatorio = montar_relatorio(args.experimento, pasta, linhas, comando, execucao)
    with open(os.path.join(pasta, "ragas.md"), "w", encoding="utf-8") as f:
        f.write(relatorio)
    print(relatorio)
    print(f"\nGravados {pasta}/ragas.csv e ragas.md", file=sys.stderr)


if __name__ == "__main__":
    main()
