# Experimento pipeline_exploratorio: RAGAS adaptado e conferência dos números

- Gerado em 11/11/2026 11:00 por `python pipeline/avaliacao/ragas_avaliacao.py pipeline_exploratorio`.
- Execução do experimento: `20261111-093954`. Pasta: `metricas\pipeline_exploratorio`.
- Juiz: `qwen2.5-coder:7b` (Ollama); embeddings: E5 local (`modelos_embedding`, prefixo de pergunta). Bibliotecas: ragas 0.4.3, openai 1.109.1.
- Fidelidade e relevância da resposta seguem as definições do artigo da RAGAS (es2025), calculadas pela biblioteca ragas com um juiz local; a relevância do contexto não é usada. A ragas 0.4.3 só importa com um contorno (módulo falso do Vertex AI), declarado em `contornar_vertexai`.
- A nota mede sustentação pelo contexto, e não a correção do dado, e o juiz é o mesmo modelo que escreve as respostas. Convém medir a concordância dele com a autora na amostra de 20 respostas.
- A conferência dos números é feita por código: fração dos valores numéricos do resultado do SQL (até 30 linhas) que aparecem no texto da resposta, comparando só os dígitos. Não é métrica de artigo.

## 1. Perguntas conceituais

| Configuração | Fidelidade (RAGAS) | Relevância da resposta (RAGAS) |
|---|---|---|
| Cenário 1: embeddings (esquema e glossário) | 0,80 (n = 20) | 0,78 (n = 20) |
| Cenário 2: embeddings (esquema) + RRF (glossário) | 0,73 (n = 20) | 0,82 (n = 20) |

## 2. Perguntas híbridas

| Configuração | Fidelidade (RAGAS) | Relevância da resposta (RAGAS) | Números do resultado no texto (por código) |
|---|---|---|---|
| Cenário 1: embeddings (esquema e glossário) | 0,25 (n = 20) | 0,47 (n = 20) | 0,93 (n = 19) |
| Cenário 2: embeddings (esquema) + RRF (glossário) | 0,31 (n = 20) | 0,52 (n = 20) | 0,93 (n = 19) |

## 3. Perguntas quantitativas

| Configuração | Números do resultado no texto (por código) |
|---|---|
| Cenário 1: embeddings (esquema e glossário) | 0,96 (n = 17) |
| Cenário 2: embeddings (esquema) + RRF (glossário) | 0,96 (n = 17) |
