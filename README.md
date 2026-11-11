# Arquitetura Híbrida RAG-to-SQL para Consulta em Linguagem Natural a Bancos de Dados Financeiros Estruturados

Pipeline que permite consultar, em linguagem natural, as Demonstrações Financeiras Padronizadas
(DFP) publicadas pela CVM. Combina recuperação semântica sobre a documentação do esquema e sobre
normativos contábeis com geração de SQL, para responder tanto perguntas quantitativas (que exigem
consulta ao banco) quanto conceituais (que exigem apenas os normativos) e híbridas (que exigem os
dois).

---

## Estrutura

```
rag-nl2sql-financas-br/
├── configuracao.json                   # escolhas do pipeline: modelo, métodos de busca, k, divisor, stemmer
├── experimentos.json                   # experimentos de avaliação: configurações nomeadas
├── requirements.txt                    # dependências Python
├── dados/
│   ├── dfp_cia_aberta_2025/            # CSVs oficiais da CVM (lote anual de 2025)
│   ├── meta_dfp_cia_aberta_txt/        # dicionário de dados oficial da CVM
│   ├── normativos/
│   │   ├── cvm/resol080consolid.pdf    # Resolução CVM 80/2022 consolidada
│   │   └── cpc/                        # CPC 03 (R2), CPC 09 (R1), CPC 26 (R1)
│   ├── dfp.db                          # SQLite gerado por carregar_dfp.py (fora do git, 121 MB)
│   ├── perguntas_validacao.json        # as 60 perguntas oficiais de avaliação
│   └── perguntas_validacao_ficha.md    # ficha de conferência das 60 perguntas
├── embeddings/
│   ├── esquema.json                    # documentação do esquema, uma entrada por tabela
│   ├── corpus_esquema.json             # registros de busca do esquema (id, fonte, conteudo), sem vetores
│   ├── corpus_glossario.json           # trechos dos normativos (id, fonte, conteudo), sem vetores
│   ├── index_esquema.json              # índice vetorial do esquema (modelo padrão)
│   ├── index_glossario.json            # índice vetorial dos normativos (modelo padrão)
│   ├── index_bm25_esquema.json         # índice de palavras (BM25) do esquema
│   ├── index_bm25_glossario.json       # índice de palavras (BM25) dos normativos
│   └── modelos/<chave>/                # índices vetoriais dos outros modelos (gerados, fora do git)
├── metricas/                           # resultados dos experimentos
│   ├── <experimento>/                  # uma pasta por experimento: relatorio.md, resultados.csv, resumo.csv, execucao.json
│   ├── rascunhos/                      # execuções de teste e relatórios de apoio (fora do git)
│   └── divisoes_hibridas.json          # divisões das perguntas híbridas, reaproveitadas por todas as configurações
└── pipeline/
    ├── modelos_embedding.py            # registro dos modelos de embeddings (usado pelos dois estágios)
    ├── offline/
    │   ├── carregar_dfp.py             # CSVs da CVM -> dados/dfp.db
    │   ├── gerar_corpus.py             # esquema.json e normativos/*.pdf -> corpus_esquema.json e corpus_glossario.json
    │   ├── gerar_indice_esquema.py     # dados/dfp.db -> esquema.json; corpus_esquema.json -> index_esquema.json
    │   ├── gerar_indice_glossario.py   # corpus_glossario.json -> index_glossario.json
    │   ├── gerar_indice_bm25.py        # corpus -> índices de palavras (BM25)
    │   └── rodar_offline.py            # prepara o corpus e gera os índices acima, para um ou mais modelos
    ├── online/
    │   ├── configuracao.py             # lê, confere e sobrescreve o configuracao.json
    │   ├── busca.py                    # busca por embeddings nos dois índices
    │   ├── busca_bm25.py               # busca por palavras (BM25)
    │   ├── fusao_rrf.py                # fusão por posição (RRF) de embeddings e BM25
    │   ├── busca_configuravel.py       # ponto único de busca: escolhe o método pela configuração
    │   ├── classificador.py            # classifica a pergunta em conceitual, quantitativa ou híbrida
    │   ├── divisor.py                  # divide a pergunta híbrida em parte de dados e parte conceitual
    │   ├── gerador_sql.py              # gera e executa o SQL das perguntas quantitativas e híbridas
    │   ├── sintetizador.py             # escreve a resposta em linguagem natural
    │   ├── validador.py                # relevância do contexto e fidelidade da resposta
    │   ├── formatador.py               # resposta final com proveniência
    │   └── main.py                     # orquestrador: pergunta -> resposta final
    └── avaliacao/
        ├── executar_experimentos.py    # executor dos experimentos (recuperação e pipeline completo)
        ├── metricas.py                 # fórmulas de métrica e EX
        ├── ragas_avaliacao.py          # RAGAS (fidelidade e relevância da resposta) e conferência dos números
        ├── verificacoes.py             # testes de sanidade da síntese, do validador e do formatador
        ├── validar_perguntas.py        # confere o arquivo de perguntas e a base de cada uma
        └── medir_tempo_busca.py        # tempo de partida e por pergunta de cada busca
```

---

## Ambiente

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m nltk.downloader stopwords      # lista de palavras vazias usada pelo BM25
```

Requer um servidor Ollama rodando localmente, com um modelo de geração de texto compatível com
formato JSON estruturado (ex.: `qwen2.5-coder:7b`) baixado.

**Modelos de embeddings.** O modelo padrão é o `intfloat/multilingual-e5-base` (cerca de 1,1 GB). Os
scripts só carregam modelos do cache local e nunca baixam nada sozinhos. O download é feito pelos
scripts do estágio offline, com a opção `--baixar-modelos`, que baixa só os modelos que faltarem
(sem ela, o script para e diz o que falta, com nome, fonte e tamanho):

```bash
python pipeline/offline/rodar_offline.py --baixar-modelos
python pipeline/modelos_embedding.py        # lista os modelos e diz quais já estão no cache
```

---

## Passo a passo

Os comandos rodam de dentro de `rag-nl2sql-financas-br/`, com o ambiente ativo. O `esquema.json`, os
índices e os arquivos de configuração já estão no repositório; o único arquivo que precisa ser
gerado para consultar é o `dados/dfp.db` (passo 1).

**1. Estágio offline (prepara os dados e os índices)**

```bash
python pipeline/offline/carregar_dfp.py              # CSVs -> dados/dfp.db
python pipeline/offline/gerar_indice_esquema.py      # opcional: refaz esquema.json, o corpus e o índice do esquema (usa o LLM)
python pipeline/offline/rodar_offline.py             # corpus, índice do esquema, do glossário e BM25 (modelo padrão)
python pipeline/offline/rodar_offline.py --modelos e5-base bge-m3   # também para outros modelos
```

**2. Estágio online (responde perguntas)**

```bash
python pipeline/online/main.py "Qual foi a receita líquida consolidada da Ambev em 2025?"
python pipeline/online/main.py --interativo                  # várias perguntas, sem recarregar o sistema
python pipeline/online/main.py --arquivo perguntas.txt       # uma pergunta por linha (ou JSON)
python pipeline/online/main.py "pergunta" --set divisor.ativo=false   # troca um valor da configuração
```

**3. Avaliação**

```bash
python pipeline/avaliacao/validar_perguntas.py dados/perguntas_validacao.json --base   # confere o conjunto
python pipeline/avaliacao/executar_experimentos.py --listar                            # experimentos disponíveis
python pipeline/avaliacao/executar_experimentos.py metodos                             # embeddings, BM25 e RRF
python pipeline/avaliacao/executar_experimentos.py pipeline_exploratorio               # pipeline completo, dois cenários de busca
python pipeline/avaliacao/ragas_avaliacao.py pipeline_exploratorio                     # RAGAS sobre o respostas.json do experimento
python pipeline/avaliacao/verificacoes.py todas                                        # testes de sanidade (usam o LLM)
```

Os demais scripts de avaliação estão descritos na seção "Avaliação e experimentos".

---

## Configuração

`configuracao.json` reúne as escolhas do pipeline. Os scripts do estágio online leem o arquivo, e
qualquer valor pode ser trocado na linha de comando com `--set chave=valor` (chaves aninhadas
separadas por ponto), sem editar o arquivo. `python pipeline/online/configuracao.py` mostra a
configuração efetiva.

| Chave | Padrão | O que controla |
|---|---|---|
| `modelo_embedding` | `e5-base` | Modelo dos embeddings (as chaves estão em `pipeline/modelos_embedding.py`) |
| `busca.esquema.metodo` | `embeddings` | Método que busca as tabelas: `embeddings`, `bm25` ou `rrf` |
| `busca.glossario.metodo` | `embeddings` | Método que busca os trechos dos normativos |
| `busca.<índice>.por_tipo` | vazio | Troca o método para um tipo de pergunta (`quantitativa`, `conceitual`, `hibrida`) |
| `k_tabelas` | `3` | Tabelas escolhidas pela busca (as referenciadas por chave estrangeira entram além delas) |
| `k_trechos` | `3` | Trechos dos normativos que vão à síntese |
| `divisor.ativo` | `true` | Divide as perguntas híbridas em parte de dados e parte conceitual |
| `divisor.bloquear_pronomes` | `true` | Recusa a parte conceitual que tenha pronome ("nesse valor") |
| `bm25.radicalizar` | `true` | O BM25 usa o radicalizador (stemmer) |

`experimentos.json` descreve os experimentos de avaliação (ver "Avaliação e experimentos").

---

## Dados

**Recorte.** Exercício social `DT_REFER = 2025-12-31` (658 das 668 companhias do lote de 2025);
reapresentações: só a `VERSAO` mais alta por companhia; demonstrações BPA, BPP, DRE, DFC (só método
indireto), DVA e DMPL, consolidadas e individuais (12 tabelas). A DRA fica fora do escopo.

**Carga (`dados/dfp.db`).** 658 companhias e 1.131.701 linhas nas 12 tabelas de demonstração, todas
com chave estrangeira para `companhias`. O CSV da CVM tem linhas idênticas repetidas em algumas
companhias; `carregar_dfp.py` descarta essas repetições exatas. Os valores ficam como publicados
(`VL_CONTA`), com `ESCALA_MOEDA` ao lado (`MIL` ou `UNIDADE`); o gerador de SQL converte para reais.

**Normativos.** Resolução CVM 80/2022 consolidada, CPC 03 (R2, fluxos de caixa), CPC 09 (R1, valor
adicionado) e CPC 26 (R1, apresentação das demonstrações), em `dados/normativos/`.

**Conjunto de perguntas (`dados/perguntas_validacao.json`).** 60 perguntas, 20 de cada tipo
(quantitativa, conceitual e híbrida), usadas em todas as avaliações. Cada uma tem `id` (`q01` a
`q20`, `c01` a `c20`, `h01` a `h20`), `pergunta`, `tipo`, `tabelas_esperadas`, `sql_referencia` (nas
quantitativas e híbridas), `evidencia` (frase literal do normativo, nas conceituais e híbridas) e os
campos de base: `base_dados` (companhia, código CVM, tabela e conta) e `base_normativo` (arquivo,
página, trechos do índice e dispositivo). `perguntas_validacao_ficha.md` registra a conferência de
cada pergunta contra os dados e os normativos, e `pipeline/avaliacao/validar_perguntas.py --base` repete as conferências
automáticas.

---

## Estágio offline

Prepara tudo o que a consulta usa: o banco, a documentação do esquema e os índices de busca.

| Arquivo | Função |
|---|---|
| `carregar_dfp.py` | Lê os CSVs da CVM e cria `dados/dfp.db` (SQLite) com `companhias` e as 12 tabelas de demonstração |
| `gerar_corpus.py` | Prepara o corpus de busca: extrai o texto dos PDFs, remove cabeçalhos, rodapés e texto riscado e divide em trechos (glossário), e escreve os textos de cada aspecto das tabelas a partir do `esquema.json` (esquema). Grava `corpus_esquema.json` e `corpus_glossario.json`; `--partes` escolhe quais |
| `gerar_indice_esquema.py` | Lê a estrutura do banco, pede ao LLM os textos em português de cada tabela, grava `esquema.json`, o corpus do esquema e o índice vetorial. Com `--so-indice`, regera só o índice a partir do corpus; `--modelo` escolhe o modelo |
| `gerar_indice_glossario.py` | Lê o corpus do glossário e grava o índice vetorial dos normativos; `--modelo` escolhe o modelo |
| `gerar_indice_bm25.py` | Grava os índices de palavras (BM25) do esquema e do glossário, a partir do corpus |
| `rodar_offline.py` | Roda o corpus, o índice de esquema, o do glossário e o BM25 na ordem certa, para um ou mais modelos (`--modelos`, `--todos-os-modelos`, `--etapas`) |
| `../modelos_embedding.py` | Registro dos modelos (identificador, prefixos, dimensões, limite de tokens) e verificação do cache; é usado também pelo estágio online |

**Esquema.** A estrutura (colunas, tipos, chaves e relações) é lida direto do SQLite. Como o
significado das DFPs está nos valores de `CD_CONTA` e `DS_CONTA`, o script também extrai do banco, sem
LLM, o plano de contas de cada tabela (contas fixas, `ST_CONTA_FIXA='S'`) e os valores possíveis das
colunas com poucos valores distintos (`ESCALA_MOEDA`, `ORDEM_EXERC`, `ST_CONTA_FIXA`). O LLM escreve só
os textos (resumo, propósito, entidades e descrição das colunas), recebendo o nome por extenso de cada
demonstração e se ela é consolidada ou individual. Cada tabela gera um registro por aspecto (resumo e
propósito, entidades, relações, colunas e contas), com até 120 tokens: 13 tabelas e 171 registros.

**Corpus.** O texto que cada método de busca indexa vem de um passo só, `gerar_corpus.py`: uma lista de
registros com `id`, `fonte` e `conteudo`, sem vetores. Os vetores (um índice por modelo) e o índice de
palavras do BM25 partem desse mesmo corpus, então todos os métodos e modelos comparam as mesmas
unidades de busca. O tamanho dos registros é medido sempre com o tokenizador do E5-base, para o corpus
ser o mesmo qualquer que seja o modelo.

**Glossário.** Extrai o texto de cada PDF (varredura recursiva de `cvm/` e `cpc/`), remove cabeçalhos
e rodapés repetidos (detectados por frequência entre páginas) e as linhas riscadas (redação substituída
nos PDFs consolidados) e divide o texto em trechos de até 120 tokens, medidos com o tokenizador do
E5-base, com sobreposição de cerca de 15%. Resultado: 1.283 trechos (157 do CPC 03, 202 do CPC 09, 330
do CPC 26 e 594 da Resolução 80). Os trechos são os mesmos qualquer que seja o modelo de embeddings.

**BM25.** O índice lê o corpus (não depende dos vetores nem do modelo) e guarda, por registro, o texto e as palavras já limpas (minúsculas, sem acentos,
sem palavras vazias do português), com e sem radicalização, para a busca não refazer isso a cada
partida.

**Modelos.** `minilm`, `e5-small`, `e5-base` (padrão), `bge-m3` e `e5-base-v2` (só inglês, usado como
controle). O corpus fica em `embeddings/`. O índice vetorial do modelo padrão fica em `embeddings/`; o dos outros, em
`embeddings/modelos/<chave>/`, fora do git.

---

## Estágio online

Responde uma pergunta: classifica, busca, gera o SQL, escreve a resposta, valida e formata.

| Arquivo | Função |
|---|---|
| `configuracao.py` | Lê o `configuracao.json`, confere os valores e aceita sobrescritas `--set chave=valor` |
| `busca.py` | Busca por embeddings: vetoriza a pergunta e ordena os registros pelo produto interno |
| `busca_bm25.py` | Busca por palavras (BM25, biblioteca `rank_bm25`), com radicalização opcional |
| `fusao_rrf.py` | Funde os rankings de embeddings e BM25 por posição (Reciprocal Rank Fusion, k = 60) |
| `busca_configuravel.py` | Ponto único de busca: escolhe o método e o modelo pela configuração e devolve tabelas e trechos |
| `classificador.py` | Classifica a pergunta em `conceitual`, `quantitativa` ou `hibrida` |
| `divisor.py` | Divide a pergunta híbrida em parte de dados e parte conceitual |
| `gerador_sql.py` | Gera o SQL com o LLM, executa em modo somente leitura e corrige em caso de erro |
| `sintetizador.py` | Escreve a resposta em português a partir do resultado do SQL e dos trechos |
| `validador.py` | Mede a relevância do contexto e a fidelidade da resposta |
| `formatador.py` | Monta a resposta final com a tabela de dados e a proveniência |
| `main.py` | Orquestrador: liga os passos em um fluxo único por pergunta |

**Busca.** Os embeddings usam o prefixo `query: ` na pergunta e `passage: ` nos textos indexados, como
o E5 exige. No índice de esquema, cada tabela é pontuada pelo melhor registro de resumo ou de
entidades (colunas, contas e relações não entram na pontuação), e as tabelas escolhidas vêm
acompanhadas das que elas referenciam por chave estrangeira (hoje, `companhias`). No glossário, a
busca devolve os trechos mais parecidos. O BM25 usa k1 = 2 e b = 0,5.

**Classificador.** Um LLM local recebe a definição de cada classe e 23 exemplos e responde um JSON
`{"tipo": ...}`. O tipo decide o caminho: conceitual usa o glossário, quantitativa usa o esquema e o
SQL, e híbrida usa os dois. Os exemplos têm híbridas de estruturas variadas e também quantitativas e
conceituais de forma parecida, mas sem as duas partes, para o modelo não tratar o "e" ou o nome de
uma companhia como sinal de pergunta híbrida. Se o LLM não devolver uma classe válida em 2 tentativas, a pergunta é
tratada como híbrida.

**Divisor.** Só nas perguntas híbridas e com `divisor.ativo`. O LLM separa a pergunta em parte de dados
e parte conceitual (5 exemplos no prompt); a parte conceitual não pode ter pronome ("nesse valor"), e,
se tiver, o LLM é chamado de novo. Se a divisão falhar, as duas partes viram a pergunta inteira. As
tabelas continuam sendo buscadas com a pergunta inteira, os trechos são buscados com a parte conceitual e o
gerador de SQL recebe só a parte de dados.

**Gerador de SQL.** Para as perguntas quantitativas e híbridas. O contexto do LLM traz só as tabelas
escolhidas, lidas do `esquema.json`: colunas, tipos, valores possíveis, junções e o plano de contas.
Cada conta aparece com a descrição oficial e o filtro pronto:

```
- Resultado Bruto -> CD_CONTA = '3.03'
```

As regras do prompt tratam de consolidado (`_con`) como padrão, `ORDEM_EXERC`, filtro de companhia por
`LIKE` em `DENOM_CIA` e valores sempre em reais (`CASE WHEN ESCALA_MOEDA = 'MIL' THEN VL_CONTA * 1000
ELSE VL_CONTA END`). Um nome pode combinar com mais de uma companhia (`%JBS%` traz JBS S.A. e JBS N.V.):
o SQL devolve uma linha por companhia, com `DENOM_CIA` na primeira coluna. O código só aceita um único
`SELECT`, executado em modo somente leitura e com limite de tempo, e confere o SQL antes de executar:

- recusa `CD_CVM` escrito direto (o LLM inventa esses códigos) e confere os nomes do `LIKE` em
  `DENOM_CIA`: se o `LIKE` literal não casa com nenhuma companhia, mas o termo, sem acento, pontuação e
  espaço, casa com uma ou mais, a condição é trocada por `DENOM_CIA IN (nomes exatos)` e o SQL original
  fica em `sql_original`; se nada casa, o LLM recebe um aviso com nomes parecidos;
- confere se cada `CD_CONTA` existe no plano da tabela em que é usado (o aviso só sai se o código existe
  no plano de outra tabela do contexto, pois o plano guarda só as contas fixas);
- recusa `VL_CONTA` sem `ESCALA_MOEDA`, pedindo a conversão para reais.

Erro de execução, resultado vazio e resultado só com valores nulos (como o `SUM` sobre nenhuma linha)
voltam ao LLM com o problema, por até 3 tentativas. `gerar_sql` aceita também a lista de tabelas, sem usar
a busca, o texto que o LLM recebe (`pergunta_sql`; as tabelas são sempre buscadas com a pergunta) e um
dicionário de cache que reaproveita o SQL de um mesmo par (texto, tabelas do contexto).

**Síntese.** O LLM responde só com o que recebe. Conceitual: os trechos mais parecidos; quantitativa: o
resultado do SQL, com cada valor rotulado pelo nome da coluna e os números no padrão brasileiro;
híbrida: os dois. Se os trechos não trouxerem a informação pedida, a resposta diz que não encontrou nos
documentos disponíveis (na híbrida, a parte de dados é respondida mesmo assim). A resposta tem limite
de tokens, e a função devolve também o contexto mostrado ao LLM, que a validação usa.

**Validação.** Duas verificações com o mesmo LLM, em temperatura 0: a *relevância do contexto* (fração
dos trechos relevantes para a pergunta) e a *fidelidade* (a resposta é decomposta em afirmações e cada
uma é verificada contra o contexto). A resposta é aprovada se a fidelidade for pelo menos 0,8. Ela mede
a sustentação pelo contexto, não a correção do dado.

**Resposta final.** O texto entregue traz a resposta, a tabela de dados do resultado do SQL (até 20
linhas) e a proveniência. Os quatro campos da proveniência vêm do SQL gerado e do banco, sem LLM:
empresa e código CVM, demonstração financeira e exercício social. Também entram a consulta SQL e os
documentos usados (numa recusa, nenhum documento é listado). Fidelidade abaixo do limiar gera um aviso.

**Fluxo (`main.py`).** `executar(pergunta)` classifica e segue o caminho do tipo: divide a pergunta
híbrida, gera o SQL (quantitativa e híbrida; na híbrida dividida, só com a parte de dados), sintetiza,
valida e formata. Se o SQL falha em todas as
tentativas, a quantitativa recebe uma mensagem de falha e a híbrida é respondida só com os normativos,
com um aviso. Pela linha de comando: `--interativo` e `--arquivo` carregam o sistema uma vez e
respondem várias perguntas, `--configuracao` troca o arquivo de configuração e `--set` troca um valor.

---

## Avaliação e experimentos

Ficam em `pipeline/avaliacao/`, separados do fluxo de uma pergunta (`pipeline/online/`), que não depende
deles.

| Arquivo | Função |
|---|---|
| `executar_experimentos.py` | Executor dos experimentos descritos em `experimentos.json`, em dois níveis (abaixo); é o ponto de entrada de todas as medições |
| `metricas.py` | Fórmulas usadas pelo executor: a chamada à biblioteca `ranx` (Precision@k, Recall@k, nDCG@k e MRR), o EX estrito e o EX adaptado, e a formatação dos relatórios |
| `ragas_avaliacao.py` | Lê o `respostas.json` de um experimento do nível `pipeline` e pontua, com a biblioteca `ragas` e o Qwen local como juiz, a fidelidade e a relevância da resposta das conceituais e híbridas (relevância do contexto não é usada); confere por código se os números do resultado do SQL aparecem no texto. Grava `ragas.csv` e `ragas.md` na pasta do experimento. A `ragas` 0.4.3 só importa com um contorno, declarado no arquivo |
| `verificacoes.py` | Testes de sanidade com o LLM: `sintese` (respostas com o SQL de referência e recusas fora do escopo), `validador` (número trocado e fato inventado devem ser reprovados, resposta correta aprovada) e `formatador` (proveniência, aviso de fidelidade baixa e recusa) |
| `validar_perguntas.py` | Confere o arquivo de perguntas (campos, tabelas, evidência nos trechos); com `--base`, confere também `base_dados`, `sql_referencia` e `base_normativo` |
| `medir_tempo_busca.py` | Tempo de partida e por pergunta de cada busca |

Todos usam o conjunto oficial (`dados/perguntas_validacao.json`) e rodam de dentro de
`rag-nl2sql-financas-br/`.

**Executor de experimentos.** `experimentos.json` descreve cada experimento: nível, configurações
nomeadas. Cada configuração é uma lista de trocas `chave=valor` sobre o
`configuracao.json`, como o `--set`, então qualquer escolha pode variar (modelo, método por índice e
por tipo, `k`, divisor, pronomes, stemmer).

- **Nível `recuperacao`** (sem LLM, exceto o divisor): para cada configuração e pergunta, mede se a busca
  traz as tabelas e os trechos certos, com Precision@k, Recall@k, nDCG@k e MRR (biblioteca `ranx`,
  relevância binária). Nas tabelas, a `companhias` fica fora da medição, porque o sistema a acrescenta
  a todas as listas; nos trechos, é relevante o que contém a evidência da pergunta. O relatório mostra
  médias por tipo, e as híbridas têm tabelas e trechos em separado. As divisões das híbridas ficam em
  `metricas/divisoes_hibridas.json`, para todas as configurações usarem as mesmas.
- **Nível `pipeline`** (com LLM): roda `main.py` em cada pergunta, por rodada e configuração, e mede
  classe, EX adaptado, EX estrito e tempo por passo (o EX adaptado também
  aceita a coluna `DENOM_CIA` e linhas de companhias cujo nome tem a palavra que identifica a companhia
  da pergunta); o relatório traz também a matriz de confusão do classificador (tipo
  real contra tipo previsto, uma por configuração). Com `"tabelas_ideais": true`, uma passada extra
  gera o SQL só com as tabelas esperadas, para separar o erro da busca do erro da geração. A fidelidade
  e a relevância da resposta (RAGAS) são pontuadas depois, por `ragas_avaliacao.py`, sobre o
  `respostas.json` da pasta do experimento.

**Resultados e versão.** Cada experimento tem a sua pasta, `metricas/<nome>/`, com nomes fixos que são
sobrescritos a cada execução:

| Arquivo | Conteúdo |
|---|---|
| `relatorio.md` | Tabelas por tipo, métricas e escalas, ressalvas e comandos |
| `resultados.csv` | Uma linha por rodada, configuração, pergunta e métrica |
| `resumo.csv` | Uma linha por configuração, tipo e métrica (soma, média, mínimo e máximo entre rodadas) |
| `respostas.json` | Só no nível `pipeline`: respostas, SQL e divisões de cada pergunta |
| `ragas.csv` e `ragas.md` | Só no nível `pipeline`, depois de `ragas_avaliacao.py`: fidelidade e relevância da resposta e conferência dos números |
| `execucao.json` | A ficha que identifica o teste: comando, commit do código (com aviso de alterações não commitadas), impressão digital (SHA-256) da configuração, dos experimentos, das perguntas e dos índices, modelos, versões das bibliotecas, configurações efetivas, rodadas e duração |

Cada execução recebe um identificador (data e hora, `AAAAMMDD-HHMMSS`) que aparece no cabeçalho do
`relatorio.md` e na coluna `execucao` dos CSV. A versão de uma execução oficial é o commit que guarda
a pasta, com o identificador na mensagem, e uma tag `<nome>-v1`, `<nome>-v2`...; o executor imprime os
comandos sugeridos no fim, e o resultado antigo continua acessível pelo git:

```bash
git add metricas/metodos
git commit -m "data: resultados do experimento metodos (execução 20261112-103000)"
git tag metodos-v1
git show metodos-v1:metricas/metodos/relatorio.md
```

Execuções de teste (`--ids`, ou `"rascunho": true` no experimento) vão para `metricas/rascunhos/<nome>/`,
pasta ignorada pelo git, e o relatório as marca como rascunho. `medir_tempo_busca.py` grava o relatório
dele nessa mesma pasta.

**Experimentos executados.** Cada execução oficial está no commit que guarda a sua pasta e tem uma tag.

| Experimento | Nível | O que compara | Tags |
|---|---|---|---|
| `metodos` | `recuperacao` | embeddings, BM25 e RRF, o BM25 e o RRF com e sem radicalizador | `metodos-v1` |
| `modelos` | `recuperacao` | cinco modelos de embeddings (`minilm`, `e5-small`, `e5-base`, `bge-m3` e `e5-base-v2`) | `modelos-v1` |
| `pipeline_exploratorio` | `pipeline` | dois cenários de busca (embeddings nos dois índices; embeddings no esquema e RRF sem radicalizador no glossário), com a passada de tabelas ideais | `pipeline_exploratorio-v1` a `-v3` |

O experimento `teste_equivalencia` serve para conferir que uma mudança no código não alterou as
métricas de recuperação: rode-o antes e depois, e compare os `resultados.csv` (a coluna `execucao` é a
única que muda).

```bash
python pipeline/avaliacao/executar_experimentos.py metodos
python pipeline/avaliacao/executar_experimentos.py teste_pipeline --ids q01,c01,h01 --rodadas 1
```
