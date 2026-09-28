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
│   └── perguntas_dfp.json           # conjunto de perguntas de teste
├── embeddings/
│   ├── esquema.json                 # documentação do esquema, uma entrada por tabela
│   ├── index_esquema.json           # índice vetorial de esquema
│   └── index_glossario.json         # índice vetorial conceitual
├── pipeline/
│   ├── carregar_dfp.py              # CSVs da CVM -> dados/dfp.db
│   ├── gerar_indice_esquema.py      # dados/dfp.db -> esquema.json -> index_esquema.json
│   └── gerar_indice_glossario.py    # normativos/*.pdf -> index_glossario.json
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

## Passo a passo

```bash
python pipeline/carregar_dfp.py            # 1: CSVs -> dados/dfp.db
python pipeline/gerar_indice_esquema.py    # 2: dados/dfp.db -> esquema.json + index_esquema.json
python pipeline/gerar_indice_glossario.py  # 3: normativos -> index_glossario.json
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

Resultado: 13 tabelas (`companhias` + 12 de demonstração), 171 registros no índice.

## Normativos (`index_glossario.json`)

Extrai o texto de cada PDF em `dados/normativos/` (varredura recursiva das subpastas `cvm/` e
`cpc/`), remove cabeçalhos e rodapés repetidos (detectados por frequência entre páginas) e divide
o texto em trechos de até 120 tokens, com sobreposição de ~15%, respeitando limites de parágrafo
quando possível.

Resultado: 1.634 trechos (163 do CPC 03, 202 do CPC 09, 425 do CPC 26, 842 da Resolução 80).

## Conjunto de perguntas de teste (`dados/perguntas_dfp.json`)

28 perguntas (7 conceituais, 14 quantitativas, 7 híbridas), com `id`, `pergunta`, `tipo`,
`tabelas_esperadas`, `sql_referencia` e `evidencia`. Serve de base para métricas de recuperação
(Recall@k, MRR), de resposta conceitual (faithfulness, answer relevance) e de geração de SQL (EX,
EM).

- Conceituais: evidência conferida contra o texto real de `index_glossario.json`.
- Quantitativas e híbridas: todos os SQLs de referência executados e conferidos contra
  `dados/dfp.db`.
