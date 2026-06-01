# StratRAG Retrieval Evaluation Pipeline

This project is a phased Haystack 2.x RAG evaluation pipeline for multi-hop retrieval experiments on StratRAG-style data.

The immediate goal is to build the project in independently runnable stages. Retrieval evaluation must run without any LLM or API key, and later generation evaluation will use local Ollama by default.

## Current Status

- Stage 0: project skeleton and configuration.
- Stage 1: StratRAG data loading and dataset statistics.
- Stage 2+: intentionally not implemented yet. Wait for data-format validation before retrieval code is added.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Configuration

Runtime parameters live in `src/config.py`. The default data path is:

```text
data/stratrag.jsonl
```

You can override selected settings with environment variables:

```bash
STRATRAG_DATA_PATH=data/your_file.jsonl \
STRATRAG_BENCHMARK_NAME=stratrag \
STRATRAG_TOP_K=5 \
STRATRAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
python -m src.stage1_data
```

## Stage 0 Validation

```bash
python -c "import src; from src.config import get_config; print(get_config())"
```

## Stage 1 Validation

Stage 1 accepts a local StratRAG JSON/JSONL file and normalizes each question to:

- `question`
- `candidate_docs`
- `gold_indices`
- `answer`
- `question_type`

Internally, the loader now has a benchmark adapter layer:

- `load_benchmark_records(...)`: generic entry point for registered benchmarks.
- `load_stratrag_records(...)`: StratRAG-compatible wrapper kept for Stage 1.
- `BenchmarkRecord.to_haystack_documents()`: converts each candidate pool into Haystack `Document` objects with gold labels in `meta`.
- `BenchmarkLoader`: Haystack custom component that outputs normalized records, Haystack documents, and stats.

Run:

```bash
python -m src.stage1_data
```

Or validate the included smoke fixture:

```bash
python -m src.stage1_data --benchmark stratrag --data-path tests/fixtures/sample_stratrag.jsonl
```

If the real StratRAG field names differ, update the field aliases in `src/config.py`.

## Data Format Assumption

The loader is intentionally conservative because the exact local StratRAG export schema has not been verified in this repository yet.

Expected canonical shape:

```json
{
  "question": "...",
  "candidate_docs": ["doc 0", "doc 1", "..."],
  "gold_indices": [0, 4],
  "answer": "...",
  "question_type": "bridge"
}
```

Also supported through configurable aliases:

- candidate documents: `candidate_docs`, `context`, `contexts`, `documents`, `docs`, `paragraphs`
- gold indices: `gold_indices`, `gold_doc_indices`, `supporting_doc_indices`, `supporting_indices`
- HotpotQA-style support facts: `supporting_facts` as `[[title, sentence_id], ...]`, mapped to candidate titles
- answer: `answer`, `final_answer`
- question type: `question_type`, `type`, `q_type`

TODO: once the actual StratRAG file is placed under `data/`, confirm the concrete schema here.

## Adding More Benchmarks

Add a new adapter in `src/data_loader.py` that implements `BenchmarkAdapter.normalize(...)`, then register it in `BENCHMARK_ADAPTERS`.

The rest of the retrieval/evaluation code should depend on `BenchmarkRecord` and Haystack `Document`, not raw dataset fields.

## Haystack API Notes

Haystack MCP documentation was checked for the planned Stage 2 API surface. `InMemoryDocumentStore` and `InMemoryEmbeddingRetriever` are the intended in-memory retrieval components for the next stage:

- https://docs.haystack.deepset.ai/docs/inmemorydocumentstore
- https://docs.haystack.deepset.ai/reference/retrievers-api
