# Arquitetura Híbrida RAG-to-SQL para Consulta em Linguagem Natural a Bancos de Dados Financeiros Estruturados

Pipeline que permite consultar, em linguagem natural, as Demonstrações Financeiras Padronizadas
(DFP) publicadas pela CVM. Combina recuperação semântica sobre a documentação do esquema e sobre
normativos contábeis com geração de SQL, para responder tanto perguntas quantitativas (que exigem
consulta ao banco) quanto conceituais (que exigem apenas os normativos).

---

## Estrutura

```
rag-nl2sql-financas-br/
├── dados/
│   ├── dfp_cia_aberta_2025/         # CSVs oficiais da CVM (658 companhias, exercício 2025)
│   ├── meta_dfp_cia_aberta_txt/     # dicionário de dados oficial da CVM
│   ├── normativos/
│   │   ├── cvm/resol080consolid.pdf # Resolução CVM 80/2022 consolidada
│   │   └── cpc/                     # CPC 03 (R2), CPC 09 (R1), CPC 26 (R1)
│   ├── dfp.db                       # SQLite gerado por carregar_dfp.py (fora do git, 121 MB)
│   ├── perguntas_dfp.json           # conjunto de perguntas de teste
│   ├── perguntas_validacao.json     # perguntas independentes, só para medir
│   └── sinonimos_contas.json        # termos de analista de cada conta (ex.: receita líquida)
├── embeddings/
│   ├── esquema.json                 # documentação do esquema, uma entrada por tabela
│   ├── index_esquema.json           # índice vetorial de esquema
│   └── index_glossario.json         # índice vetorial conceitual
├── pipeline/
│   ├── offline/
│   │   ├── carregar_dfp.py          # CSVs da CVM -> dados/dfp.db
│   │   ├── gerar_indice_esquema.py  # dados/dfp.db -> esquema.json -> index_esquema.json
│   │   └── gerar_indice_glossario.py  # normativos/*.pdf -> index_glossario.json
│   └── online/
│       ├── busca.py                 # busca por similaridade nos dois índices
│       ├── avaliar_recuperacao.py   # Recall@k e MRR da busca
│       ├── classificador.py         # classifica a pergunta em conceitual, quantitativa ou híbrida
│       ├── avaliar_classificador.py # acurácia e matriz de confusão do classificador
│       ├── gerador_sql.py           # gera e executa o SQL das perguntas quantitativas
│       ├── avaliar_sql.py           # EX do gerador de SQL
│       ├── sintetizador.py          # escreve a resposta em linguagem natural
│       ├── avaliar_sintese.py       # inspeciona a síntese
│       ├── validador.py             # relevância do contexto e fidelidade da resposta
│       ├── avaliar_validador.py     # inspeciona a validação
│       ├── formatador.py            # resposta final com proveniência
│       ├── avaliar_formatador.py    # inspeciona o formatador
│       ├── main.py                  # orquestrador: pergunta -> resposta final
│       └── avaliar_pipeline.py      # mede o fluxo completo (classe, EX, fidelidade, tempo)
└── requirements.txt
```

## Ambiente

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Requer um servidor Ollama rodando localmente, com um modelo de geração de texto compatível com
formato JSON estruturado (ex.: `qwen2.5-coder:7b`) baixado.

Os embeddings são gerados pelo modelo `intfloat/multilingual-e5-base` (cerca de 1,1 GB). Os scripts
o carregam do cache local, então ele precisa ser baixado uma vez:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-base')"
```

## Passo a passo

```bash
python pipeline/offline/carregar_dfp.py            # 1: CSVs -> dados/dfp.db
python pipeline/offline/gerar_indice_esquema.py    # 2: dados/dfp.db -> esquema.json + index_esquema.json
python pipeline/offline/gerar_indice_glossario.py  # 3: normativos -> index_glossario.json
python pipeline/online/avaliar_recuperacao.py      # 4: mede a busca (Recall@k, MRR)
python pipeline/online/avaliar_classificador.py    # 5: mede o classificador de intenção (acurácia)
python pipeline/online/avaliar_sql.py              # 6: mede o gerador de SQL (EX)
python pipeline/online/avaliar_sintese.py          # 7: inspeciona a síntese da resposta
python pipeline/online/avaliar_validador.py        # 8: inspeciona a validação
python pipeline/online/avaliar_formatador.py       # 9: inspeciona a resposta formatada
python pipeline/online/avaliar_pipeline.py         # 10: mede o fluxo completo
python pipeline/online/main.py "Qual foi a receita líquida consolidada da Ambev em 2025?"   # uma pergunta
```

## Recorte de dados

- Exercício social: só `DT_REFER = 2025-12-31` (658 das 668 companhias do lote de 2025).
- Reapresentações: mantida só a `VERSAO` mais alta por companhia.
- Demonstrações: BPA, BPP, DRE, DFC (só método indireto), DVA, DMPL — consolidado e individual
  (12 tabelas). DRA fora do escopo (residual nos dados e sem CPC dedicado).

## Carga (`dados/dfp.db`)

658 companhias, 1.131.701 linhas nas 12 tabelas de demonstração, todas com chave estrangeira para
`companhias`. O CSV de origem da CVM tem linhas com conteúdo idêntico repetido em algumas
companhias; `carregar_dfp.py` descarta essas repetições exatas na carga.

## Esquema (`esquema.json` / `index_esquema.json`)

A estrutura (colunas, tipos, chaves e relações) é lida direto do SQLite. Como o significado das
DFPs está nos valores de `CD_CONTA`/`DS_CONTA` e não nos nomes de tabela ou coluna, o script também
extrai, direto do banco e sem LLM:

- o plano de contas de cada tabela (códigos e descrições das contas fixas, `ST_CONTA_FIXA='S'`);
- os valores possíveis das colunas com poucos valores distintos (`ESCALA_MOEDA`, `ORDEM_EXERC`,
  `ST_CONTA_FIXA`).

O LLM escreve só os textos em português (resumo, propósito, entidades e descrição das colunas),
recebendo no prompt o nome por extenso de cada demonstração (ex.: "BPA" → "Balanço Patrimonial
Ativo") e se é consolidada ou individual.

Cada tabela gera um registro por aspecto (resumo e propósito, entidades, relações, colunas e
contas), com até 120 tokens, e o E5-base gera o vetor de cada um. Para trocar o modelo de
embeddings sem refazer os textos do LLM, `python pipeline/offline/gerar_indice_esquema.py
--so-indice` regera só o índice a partir do `esquema.json`.

Resultado: 13 tabelas (`companhias` + 12 de demonstração), 171 registros no índice.

## Normativos (`index_glossario.json`)

Extrai o texto de cada PDF em `dados/normativos/` (varredura recursiva das subpastas `cvm/` e
`cpc/`), remove cabeçalhos e rodapés repetidos (detectados por frequência entre páginas) e divide
o texto em trechos de até 120 tokens (medidos com o tokenizador do E5-base), com sobreposição de
~15%, respeitando limites de parágrafo quando possível. O E5-base aceita até 512 tokens, mas
trechos de 250 e 400 tokens recuperaram pior que os de 120 nos testes.

Resultado: 1.634 trechos (165 do CPC 03, 202 do CPC 09, 425 do CPC 26, 842 da Resolução 80).

## Conjunto de perguntas de teste (`dados/perguntas_dfp.json`)

28 perguntas (7 conceituais, 14 quantitativas, 7 híbridas), com `id`, `pergunta`, `tipo`,
`tabelas_esperadas`, `sql_referencia` e `evidencia`. Serve de base para métricas de recuperação
(Recall@k, MRR), de resposta conceitual (faithfulness, answer relevance) e de geração de SQL (EX,
EM).

- Conceituais: evidência conferida contra o texto real de `index_glossario.json`.
- Quantitativas e híbridas: todos os SQLs de referência executados e conferidos contra
  `dados/dfp.db`.

`dados/perguntas_validacao.json` tem 13 perguntas no mesmo formato, escritas no estilo de um
analista e com outras empresas. Serve só para medir a busca, sem ajustá-la a elas. Os valores dos
SQLs de referência não foram conferidos contra os documentos originais.

## Busca (`pipeline/online/busca.py`)

O `multilingual-e5-base` gera o vetor da pergunta, com o prefixo `query: ` (os textos dos índices
usam `passage: `, como o modelo exige). O produto interno com os vetores normalizados do índice
(igual ao cosseno) ordena os registros.

- **Glossário:** devolve os k trechos mais parecidos.
- **Esquema:** cada tabela é pontuada pelo melhor registro de resumo ou de entidades. Registros de
  colunas, contas e relações não entram na pontuação. As k tabelas escolhidas vêm acompanhadas das
  tabelas que elas referenciam por chave estrangeira (hoje, `companhias`), lidas de `esquema.json`.

`avaliar_recuperacao.py` calcula Recall@k e MRR nos dois índices e aceita o arquivo de perguntas
como argumento:

```bash
python pipeline/online/avaliar_recuperacao.py dados/perguntas_validacao.json
```

## Classificador de intenção (`pipeline/online/classificador.py`)

Um LLM local classifica a pergunta em `conceitual`, `quantitativa` ou `hibrida`. O tipo decide o
caminho: conceitual usa só o índice do glossário, quantitativa usa o índice de esquema e o SQL, e
híbrida usa os dois. O prompt traz a definição de cada classe e 9 exemplos (few-shot), com as
classes intercaladas, que não fazem parte de nenhum dos dois arquivos de perguntas. A saída é um
JSON `{"tipo": ...}`, com temperatura 0 na primeira tentativa. Se o LLM não devolver uma classe
válida em 2 tentativas, a pergunta é tratada como híbrida, que aciona os dois caminhos.

`avaliar_classificador.py` calcula a acurácia geral e por classe, a matriz de confusão e o tempo por
pergunta, e aceita o arquivo de perguntas como argumento:

```bash
python pipeline/online/avaliar_classificador.py dados/perguntas_validacao.json
```

## Gerador de SQL (`pipeline/online/gerador_sql.py`)

Para as perguntas quantitativas e híbridas. A busca escolhe as `k` tabelas mais parecidas com a
pergunta (padrão `k = 3`) e acrescenta as que elas referenciam por chave estrangeira. O contexto do
LLM traz só essas tabelas, lidas de `esquema.json`: colunas, tipos, valores possíveis das colunas
categóricas, junções e o plano de contas. Cada conta aparece com o filtro pronto e, quando há, com os
termos de analista de `dados/sinonimos_contas.json` (65 termos de 41 contas, todos presentes nos
normativos indexados):

```
- Receita de Venda de Bens e/ou Serviços (também chamada de: receita líquida) -> CD_CONTA = '3.01'
```

As regras do prompt tratam de consolidado (`_con`) como padrão, `ORDEM_EXERC`, filtro de companhia
por `LIKE` em `DENOM_CIA` e valores sempre em reais (`CASE WHEN ESCALA_MOEDA = 'MIL' THEN VL_CONTA *
1000 ELSE VL_CONTA END`). Um nome pode combinar com mais de uma companhia (`%JBS%` traz JBS S.A. e
JBS N.V.): o SQL não soma nem limita, e devolve uma linha por companhia, com `DENOM_CIA` na primeira
coluna.

O LLM responde com um JSON `{"sql": ...}`. O código só aceita um único `SELECT`, executa o banco em
modo somente leitura, com limite de tempo, e devolve o erro ao LLM (ou avisa que o resultado veio
vazio) por até 3 tentativas. `gerar_sql(pergunta, k, tabelas)` também aceita a lista de tabelas, sem
usar a busca.

`avaliar_sql.py` calcula a acurácia de execução (EX) nas perguntas com SQL de referência. Dois
resultados são equivalentes se só diferem pela ordem das colunas e das linhas, com números
arredondados em 2 casas. Como o gerador lista todas as companhias que combinam com o nome, o script
também conta o EX que aceita, além das linhas da referência, as de outras companhias que combinam
com o mesmo nome (marcado `ok+`; `DENOM_CIA` é a única coluna extra aceita). Aceita o arquivo de
perguntas, o número de tabelas e a opção de usar as tabelas esperadas das perguntas:

```bash
python pipeline/online/avaliar_sql.py dados/perguntas_validacao.json --k 5
python pipeline/online/avaliar_sql.py --tabelas-ideais
```

## Síntese da resposta (`pipeline/online/sintetizador.py`)

Escreve a resposta em português a partir do que o LLM recebe, e só dele. No tipo conceitual, o
contexto são os 3 trechos do glossário mais parecidos com a pergunta; no quantitativo, o resultado
do SQL; no híbrido, os dois. O resultado do SQL entra no prompt com cada valor rotulado pelo nome
da coluna e os números no padrão brasileiro (valores monetários já em reais). Se os trechos não
trouxerem a informação pedida, o LLM responde que não encontrou nos documentos disponíveis; numa
pergunta híbrida, a parte de dados é respondida mesmo assim. A resposta tem limite de tokens. A
função devolve também o contexto mostrado ao LLM, que a validação usa.

`avaliar_sintese.py` usa o SQL de referência (para isolar a síntese dos erros do gerador), imprime
as respostas, conta quantos números do resultado aparecem no texto e confere se perguntas fora do
escopo dos normativos são recusadas.

## Validação (`pipeline/online/validador.py`)

Duas verificações com o mesmo LLM, em temperatura 0:

- **Relevância do contexto:** cada trecho do glossário é julgado como relevante ou não para a
  pergunta (fração de trechos relevantes).
- **Fidelidade:** a resposta é decomposta em afirmações e cada uma é verificada contra o contexto
  (fração de afirmações sustentadas). Uma resposta sem afirmações, como uma recusa, tem fidelidade 1.

A resposta é aprovada se a fidelidade for pelo menos 0,8. `avaliar_validador.py` aplica a validação
a respostas geradas e a casos conhecidos (número trocado, fato inventado e resposta correta).

## Resposta final (`pipeline/online/formatador.py`)

Monta o texto entregue: a resposta, a tabela de dados do resultado do SQL (até 20 linhas) e a
proveniência. Os quatro campos da proveniência vêm do SQL gerado e do banco, sem LLM: **empresa e
código CVM** (as companhias do resultado ou as que o SQL filtra), **demonstração financeira** (as
tabelas citadas no SQL, com consolidada ou individual) e **exercício social** (período das linhas da
demonstração para o exercício usado). Também entram a consulta SQL e os documentos do glossário
usados; numa recusa nenhum documento é listado. Se a fidelidade ficar abaixo do limiar, a resposta
recebe um aviso. `avaliar_formatador.py` mostra casos de cada tipo, um com nome ambíguo e um com
fidelidade baixa.

## Fluxo completo (`pipeline/online/main.py`)

`executar(pergunta)` classifica a pergunta e segue o caminho do tipo: gera o SQL (quantitativa e
híbrida), sintetiza, valida e formata. Se o SQL falha em todas as tentativas, a pergunta
quantitativa recebe uma mensagem de falha e a híbrida é respondida só com os normativos, com um
aviso. Pela linha de comando, recebe a pergunta como argumento:

```bash
python pipeline/online/main.py "Qual foi o lucro líquido consolidado da Ambev em 2025?"
```

`avaliar_pipeline.py` roda o fluxo em todas as perguntas de um arquivo e resume: acerto da classe,
EX do SQL, fidelidade e relevância médias, respostas aprovadas e tempo. A opção `--saida` grava as
respostas completas em JSON.
