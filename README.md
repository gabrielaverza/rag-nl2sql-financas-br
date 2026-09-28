# rag-nl2sql-financas-br

Pipeline com dados reais da CVM para o TCC "Arquitetura Híbrida RAG-to-SQL
para Consulta em Linguagem Natural a Bancos de Dados Financeiros
Estruturados". A partir da Etapa 2 do cronograma, este repositório é a
pasta de trabalho para os dados e o código sobre DFP/CVM; o
`tcc-prototype` fica congelado como a validação da Etapa 1 sobre o banco
Northwind.

---

## Estrutura

```
rag-nl2sql-financas-br/
├── dados/
│   ├── dfp_cia_aberta_2025/         # CSVs oficiais da CVM (658 companhias, exercicio 2025)
│   ├── meta_dfp_cia_aberta_txt/     # dicionario de dados oficial da CVM
│   └── normativos/
│       ├── cvm/resol080consolid.pdf # Resolucao CVM 80/2022 consolidada
│       └── cpc/                     # CPC 03 (R2), CPC 09 (R1), CPC 26 (R1)
├── data/
│   └── dfp.db                       # SQLite gerado por carregar_dfp.py (fora do git, 121 MB)
├── embeddings/
│   ├── esquema.json                 # documentacao do esquema, uma entrada por tabela
│   └── index_esquema.json           # indice vetorial de esquema
├── pipeline/
│   ├── carregar_dfp.py              # CSVs da CVM -> data/dfp.db
│   ├── gerar_indice_esquema.py      # data/dfp.db -> esquema.json -> index_esquema.json
│   └── gerar_indice_glossario.py    # normativos/*.pdf -> index_glossario.json
├── requirements.txt
└── .venv/                           # ambiente virtual (nao versionado)
```

## Ambiente

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Precisa do Ollama rodando (`ollama serve`) com `qwen2.5-coder:7b` baixado, igual ao `tcc-prototype`
(ver `CLAUDE.md` na raiz do TCC para detalhes do ambiente, incluindo a correção do driver NVIDIA).

## Passo a passo

```bash
python pipeline/carregar_dfp.py              # 1: CSVs -> data/dfp.db
python pipeline/gerar_indice_esquema.py      # 2: data/dfp.db -> esquema.json + index_esquema.json
python pipeline/gerar_indice_glossario.py    # 3: normativos -> index_glossario.json (ainda nao adaptado às subpastas)
```

## Recorte de dados (fechado)

- Exercício social: só `DT_REFER = 2025-12-31` (658 das 668 companhias do lote de 2025).
- Reapresentações: mantida só a `VERSAO` mais alta por companhia.
- Demonstrações: BPA, BPP, DRE, DFC (só método indireto), DVA, DMPL — consolidado e individual (12 tabelas).
  DRA fora do escopo (residual nos dados e sem CPC dedicado).

## Carga em `data/dfp.db`

658 companhias, 1.131.701 linhas nas 12 tabelas de demonstração, 0 linhas órfãs de chave
estrangeira. Valores conferidos contra o documento original de duas companhias conhecidas
(Ambev: receita líquida R$ 88.242.467 mil na DRE consolidada 2025; Alpargatas: ativo total
R$ 6.096.758 mil no BPA consolidado 2025). O CSV de origem da CVM tinha linhas com conteúdo
idêntico repetido em 2 das 658 companhias (FGR Incorporações, VLI Multimodal); `carregar_dfp.py`
descarta essas repetições exatas na carga.

## Esquema (`esquema.json` / `index_esquema.json`)

Adaptado de `tcc-prototype/pipeline/gerar_indice_esquema.py`, com duas extensões sobre os dados
reais da CVM:

1. **Plano de contas.** Diferente do Northwind, o significado das DFPs está nos valores de
   `CD_CONTA`/`DS_CONTA`, não nos nomes de tabela/coluna. Para cada tabela de demonstração, os
   códigos e descrições das contas fixas (`ST_CONTA_FIXA='S'`) são extraídos direto do banco (sem
   LLM) e viram um aspecto novo do esquema ("Contas: ..."). A mesma conta pode ter descrições
   diferentes por setor (ex.: `3.01` é "Receita de Venda de Bens e/ou Serviços" para a maioria,
   mas "Receitas da Intermediação Financeira" para bancos); fica só a descrição de maior
   frequência, então setores minoritários (bancos, seguradoras) não são cobertos por este aspecto.
2. **Valores possíveis de colunas de poucos valores distintos** (`ESCALA_MOEDA`, `ORDEM_EXERC`,
   `ST_CONTA_FIXA`), extraídos direto do banco e anexados à descrição da coluna. Colunas de data
   (`DT_*`) são excluídas dessa checagem, porque teriam poucos valores só pelo recorte de exercício
   atual, não por serem um código estável.
3. O prompt do LLM recebe o nome por extenso de cada demonstração (ex.: "BPA" -> "Balanço
   Patrimonial Ativo") e se é consolidada ou individual, para não depender só da sigla da tabela.

Resultado (13 tabelas: `companhias` + 12 de demonstração, GPU): 171 registros no índice, gerados em
2m04s (contra ~10 min em CPU para as 13 tabelas do Northwind). Validação (`validar_esquema`) sem
problemas. Resumos conferidos manualmente: todos mencionam o nome completo da demonstração e a CVM.

## Normativos (`index_glossario.json`)

Ainda não gerado sobre os PDFs reais. `pipeline/gerar_indice_glossario.py` hoje varre um único
diretório plano; com `normativos/cvm/` e `normativos/cpc/` como subpastas, precisa rodar uma vez por
subpasta ou ganhar varredura recursiva antes do próximo passo.
