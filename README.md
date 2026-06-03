# HotpotQA RAG Evaluation Pipeline

This repository is being reset into a phased Haystack 2.x RAG evaluation project based on HotpotQA.

The implementation plan lives in `plan.md`. The project is organized around two main pipelines:

- ingestion: parsing, chunking, metadata handling, dense/sparse indexing, and index updates.
- retrieval: query processing, metadata filtering, hybrid search, rerank, context compression, and chunk expansion.
- generation/evaluation: prompt assembly, answer generation, citations, groundedness/no-answer checks, retrieval metrics, answer metrics, and system metrics.

## Status

Pipeline 1 has a small HotpotQA ingestion implementation:

- parse HotpotQA records into Haystack `Document` objects
- split documents with Haystack `DocumentSplitter`
- generate Anthropic-style contextual retrieval text for each chunk
- persist chunks to `data/hotpotqa_chunks.jsonl` for sparse/BM25 work
- persist a Haystack BM25 document store to `data/hotpotqa_bm25_store.json`
- index DashScope/Qwen dense embeddings into a local Chroma store
- write an ingestion manifest to `results/ingestion_manifest.json`

Pipeline 2 has a retrieval implementation:

- implemented as a Haystack `Pipeline` graph, not a hand-written stage loop
- query processing hooks for rewrite, expansion, HyDE, and routing
- metadata filters for source, title, level, type, and permissions
- BM25 retrieval from `InMemoryDocumentStore`
- optional dense retrieval from Chroma using DashScope embeddings
- hybrid fusion with RRF or weighted scores
- optional rerank
- context compression and deduplication
- small-to-big parent document expansion

Pipeline 3 has generation and evaluation:

- implemented as a Haystack `Pipeline` graph that extends Pipeline 2
- Haystack `PromptBuilder` assembles query plus retrieved sources into a cited prompt
- Haystack `OpenAIGenerator` calls the configured OpenAI-compatible chat endpoint
- answers are attributed to numbered source chunks or parent documents
- no-answer fallback runs when there is no retrieved context or scores are below threshold
- low groundedness can also trigger the no-answer fallback
- groundedness and answer relevance use lightweight lexical checks
- retrieval metrics include recall@k, precision@k, MRR, nDCG, and hit rate
- system metrics include latency, estimated token cost, retrieved document count, and context size

Business outputs keep only end-to-end latency. Component-level runtime inspection should use Haystack tracing, OpenTelemetry, or MLflow autologging instead of per-component `perf_counter` code.

## Setup

```bash
uv sync --extra dev
```

Alternatively, use a regular virtual environment and run `pip install -e ".[dev]"`.

Dense indexing uses Haystack `OpenAIDocumentEmbedder` with DashScope's OpenAI-compatible endpoint by default:

```bash
cat > .env <<'EOF'
DASHSCOPE_API_KEY=your_api_key
INGEST_CONTEXTUAL_RETRIEVAL=true
INGEST_CONTEXTUAL_MODEL=qwen-flash
INGEST_CONTEXTUAL_MAX_TOKENS=120
INGEST_CONTEXTUAL_TEMPERATURE=0
BM25_STORE_PATH=data/hotpotqa_bm25_store.json
BM25_ALGORITHM=BM25L
INGEST_EMBEDDING_MODEL=text-embedding-v4
INGEST_EMBEDDING_DIMENSION=1024
INGEST_EMBEDDING_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
INGEST_EMBEDDING_BATCH_SIZE=10
GENERATION_MODEL=qwen-flash
GENERATION_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
GENERATION_MAX_TOKENS=512
GENERATION_TEMPERATURE=0
GENERATION_MIN_GROUNDEDNESS=0.2
EOF
```

Contextual retrieval follows Anthropic's pattern: for each chunk, the pipeline prompts a generator with the whole parent document and the chunk, then prepends the generated concise context before writing JSONL or embedding into Chroma.

## Run Ingestion

Run a small Hugging Face slice without Chroma and without generator calls:

```bash
uv run python -m src.stage2_ingestion \
  --limit 2 \
  --skip-chroma \
  --skip-contextual-retrieval
```

This still writes the BM25 store unless you pass `--skip-bm25`.

Run dense indexing into local Chroma:

```bash
uv run python -m src.stage2_ingestion \
  --limit 2 \
  --rebuild
```

If you change embedding model or dimension, rebuild Chroma so the collection uses one consistent vector shape.
If you change chunking or contextual retrieval settings, rebuild both Chroma and BM25 so dense and sparse retrieval use the same chunk text.

The loader uses Hugging Face:

```python
from datasets import load_dataset

ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki")["validation"]
```

The ingestion CLI uses the same dataset/config and loads `validation[:limit]` by default.

## Run Retrieval

Run a local BM25-only retrieval smoke test:

```bash
uv run python -m src.stage3_retrieval \
  "Scott Derrickson nationality" \
  --search-mode bm25
```

Apply metadata filtering:

```bash
uv run python -m src.stage3_retrieval \
  "Scott Derrickson nationality" \
  --search-mode bm25 \
  --filter-title "Adam Collis" \
  --no-parent-expansion
```

Hybrid retrieval uses RRF by default:

```bash
uv run python -m src.stage3_retrieval \
  "Scott Derrickson nationality" \
  --search-mode hybrid \
  --fusion rrf
```

Dense or hybrid retrieval requires a built Chroma index and `DASHSCOPE_API_KEY`.

## Run Generation And Evaluation

Run a BM25-only end-to-end RAG query:

```bash
uv run python -m src.stage4_rag \
  "Scott Derrickson nationality" \
  --search-mode bm25
```

Pass HotpotQA ground-truth parent document ids to compute retrieval metrics:

```bash
uv run python -m src.stage4_rag \
  "Scott Derrickson nationality" \
  --search-mode bm25 \
  --relevant-parent-doc-id p1
```

Set cost estimates through `.env` when you want system-level cost reporting:

```bash
EVALUATION_INPUT_TOKEN_COST_PER_1K=0.001
EVALUATION_OUTPUT_TOKEN_COST_PER_1K=0.002
```
