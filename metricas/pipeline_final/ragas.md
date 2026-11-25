# Experimento pipeline_final: RAGAS adaptado e conferência dos números

- Gerado em 25/11/2026 15:25 por `python pipeline/avaliacao/ragas_avaliacao.py pipeline_final`.
- Execução do experimento: `20261125-115934`. Pasta: `metricas\pipeline_final`.
- Juiz: `qwen2.5-coder:7b` (Ollama); embeddings: E5 local (`modelos_embedding`, prefixo de pergunta). Bibliotecas: ragas 0.4.3, openai 1.109.1.
- Fidelidade e relevância da resposta seguem as definições do artigo da RAGAS (es2025), calculadas pela biblioteca ragas com um juiz local; a relevância do contexto não é calculada aqui (vem do validador do sistema e está no `relatorio.md` do executor). A ragas 0.4.3 só importa com um contorno (módulo falso do Vertex AI), declarado em `contornar_vertexai`.
- A nota mede sustentação pelo contexto, e não a correção do dado, e o juiz é o mesmo modelo que escreve as respostas. Convém medir a concordância dele com a autora na amostra de 20 respostas.
- A conferência dos números é feita por código: fração dos valores numéricos do resultado do SQL (até 30 linhas) que aparecem no texto da resposta, comparando só os dígitos. Não é métrica de artigo.

## 1. Perguntas conceituais

| Configuração | Fidelidade (RAGAS) | Relevância da resposta (RAGAS) |
|---|---|---|
| Embeddings (esquema) + RRF (glossário) | 0,66 (n = 96) | 0,72 (n = 96) |

## 2. Perguntas híbridas

| Configuração | Fidelidade (RAGAS) | Relevância da resposta (RAGAS) | Números do resultado no texto (por código) |
|---|---|---|---|
| Embeddings (esquema) + RRF (glossário) | 0,16 (n = 108) | 0,38 (n = 108) | 0,92 (n = 100) |

## 3. Perguntas quantitativas

| Configuração | Números do resultado no texto (por código) |
|---|---|
| Embeddings (esquema) + RRF (glossário) | 0,94 (n = 93) |
