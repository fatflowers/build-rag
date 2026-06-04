"""Smoke tests for Pipeline 1 ingestion."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Mapping

import src.ingestion as ingestion
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

from src.config import AppConfig, BM25Config, ContextualRetrievalConfig, get_config
from src.hotpotqa_loader import HotpotQAStats, normalize_hotpotqa_record
from src.ingestion import (
    ContextualRetrievalAnnotator,
    EmbeddingIntegrityValidator,
    run_ingestion,
    split_documents,
)


def _hotpotqa_row() -> dict:
    return {
        "id": "sample-1",
        "question": "Which city contains both the Eiffel Tower and the Louvre?",
        "answer": "Paris",
        "type": "bridge",
        "level": "easy",
        "context": {
            "title": ["Eiffel Tower", "Louvre", "Berlin"],
            "sentences": [
                ["The Eiffel Tower is a wrought-iron lattice tower in Paris, France."],
                ["The Louvre is a national art museum in Paris, France."],
                ["Berlin is the capital of Germany."],
            ],
        },
        "supporting_facts": {
            "title": ["Eiffel Tower", "Louvre"],
            "sent_id": [0, 0],
        },
    }


def test_normalize_hotpotqa_record_to_documents() -> None:
    """A HotpotQA row becomes document-level Haystack Documents."""

    documents = normalize_hotpotqa_record(_hotpotqa_row(), position=0)

    assert len(documents) == 3
    assert documents[0].meta["title"] == "Eiffel Tower"
    assert documents[0].meta["is_supporting_doc"] is True


def test_split_documents_adds_chunk_relationships() -> None:
    """Chunking preserves parent document ids and labels supporting chunks."""

    config = AppConfig(
        chunking=replace(AppConfig().chunking, contextual_retrieval=False),
    )
    documents = normalize_hotpotqa_record(_hotpotqa_row(), position=0)

    chunks = split_documents(documents, config)

    assert chunks
    assert chunks[0].id == chunks[0].meta["chunk_id"]
    assert "parent_doc_id" in chunks[0].meta
    assert any(chunk.meta["is_supporting_chunk"] for chunk in chunks)
    assert all(isinstance(value, (str, int, float, bool)) for chunk in chunks for value in chunk.meta.values())


def test_contextual_retrieval_annotator_prepends_generated_context() -> None:
    """Contextual retrieval prepends generated chunk-specific context."""

    class FakeGenerator:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def run(
            self,
            prompt: str,
            *,
            generation_kwargs: Mapping[str, int | float] | None = None,
        ) -> dict[str, list[str]]:
            self.prompts.append(prompt)
            return {"replies": ["This chunk explains that the Louvre is in Paris."]}

    source_document = Document(
        id="parent-1",
        content="The Eiffel Tower is in Paris. The Louvre is in Paris.",
        meta={"parent_doc_id": "parent-1", "title": "Paris landmarks"},
    )
    chunk = Document(
        id="chunk-1",
        content="The Louvre is in Paris.",
        meta={"parent_doc_id": "parent-1"},
    )
    generator = FakeGenerator()
    annotator = ContextualRetrievalAnnotator(
        config=ContextualRetrievalConfig(),
        generator=generator,
    )

    result = annotator.run(documents=[chunk], source_documents=[source_document])
    contextualized = result["documents"][0]

    assert contextualized.content is not None
    assert contextualized.content.startswith("This chunk explains that the Louvre is in Paris.")
    assert "The Louvre is in Paris." in contextualized.content
    assert contextualized.meta["contextual_retrieval_context"] == (
        "This chunk explains that the Louvre is in Paris."
    )
    assert "<document>" in generator.prompts[0]
    assert "<chunk>" in generator.prompts[0]


def test_embedding_integrity_validator_fails_missing_embeddings() -> None:
    """Dense indexing fails fast when embedder output has no vectors."""

    validator = EmbeddingIntegrityValidator(dimension=3)
    document = Document(id="chunk-1", content="No embedding")

    try:
        validator.run([document])
    except RuntimeError as exc:
        assert "Dense embedding failed before Chroma indexing" in str(exc)
        assert "chunk-1" in str(exc)
    else:
        raise AssertionError("EmbeddingIntegrityValidator did not reject missing embeddings.")


def test_embedding_integrity_validator_accepts_expected_dimension() -> None:
    """Dense indexing continues when every document has a valid vector."""

    validator = EmbeddingIntegrityValidator(dimension=3)
    document = Document(id="chunk-1", content="Embedded", embedding=[0.1, 0.2, 0.3])

    result = validator.run([document])

    assert result["documents"] == [document]


def test_embedding_integrity_validator_allows_unconfigured_dimension() -> None:
    """Dense indexing can skip dimension validation for providers that infer dimensions."""

    validator = EmbeddingIntegrityValidator(dimension=None)
    document = Document(id="chunk-1", content="Embedded", embedding=[0.1, 0.2])

    result = validator.run([document])

    assert result["documents"] == [document]


def test_run_ingestion_without_chroma_writes_chunks_and_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The ingestion pipeline can run without dense indexing for cheap validation."""

    config = AppConfig(
        chunking=replace(AppConfig().chunking, contextual_retrieval=False),
        bm25=BM25Config(store_path=tmp_path / "bm25_store.json"),
        chunks_path=tmp_path / "chunks.jsonl",
        manifest_path=tmp_path / "manifest.json",
    )
    documents = normalize_hotpotqa_record(_hotpotqa_row(), position=0)
    stats = HotpotQAStats(
        records=1,
        source_documents=3,
        supporting_documents=2,
        supporting_facts=2,
        matched_supporting_facts=2,
    )

    def fake_load_hotpotqa_documents(**kwargs):
        return documents, stats

    monkeypatch.setattr(ingestion, "load_hotpotqa_documents", fake_load_hotpotqa_documents)

    manifest = run_ingestion(config, skip_chroma=True)

    assert manifest["counts"]["records"] == 1
    assert manifest["chroma"]["enabled"] is False
    assert manifest["bm25"]["enabled"] is True
    assert manifest["bm25"]["document_count"] == manifest["counts"]["chunks"]
    assert config.chunks_path.exists()
    assert config.bm25.store_path.exists()
    assert config.manifest_path.exists()
    bm25_store = InMemoryDocumentStore.load_from_disk(str(config.bm25.store_path))
    assert bm25_store.count_documents() == manifest["counts"]["chunks"]
    first_chunk = json.loads(config.chunks_path.read_text(encoding="utf-8").splitlines()[0])
    assert first_chunk["id"]


def test_run_ingestion_can_skip_bm25_writer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The ingestion pipeline can skip the BM25 writer component."""

    config = AppConfig(
        chunking=replace(AppConfig().chunking, contextual_retrieval=False),
        bm25=BM25Config(store_path=tmp_path / "bm25_store.json"),
        chunks_path=tmp_path / "chunks.jsonl",
        manifest_path=tmp_path / "manifest.json",
    )
    documents = normalize_hotpotqa_record(_hotpotqa_row(), position=0)
    stats = HotpotQAStats(
        records=1,
        source_documents=3,
        supporting_documents=2,
        supporting_facts=2,
        matched_supporting_facts=2,
    )

    def fake_load_hotpotqa_documents(**kwargs):
        return documents, stats

    monkeypatch.setattr(ingestion, "load_hotpotqa_documents", fake_load_hotpotqa_documents)

    manifest = run_ingestion(config, skip_chroma=True, skip_bm25=True)

    assert manifest["bm25"]["enabled"] is False
    assert manifest["bm25"]["document_count"] is None
    assert config.chunks_path.exists()
    assert not config.bm25.store_path.exists()


def test_get_config_loads_dotenv(tmp_path: Path, monkeypatch) -> None:
    """Config reads .env values from the current working directory."""

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "HOTPOTQA_LIMIT=7",
                "CHROMA_COLLECTION_NAME=dotenv_chunks",
                "BM25_STORE_PATH=data/custom_bm25.json",
                "BM25_ALGORITHM=BM25Okapi",
                "INGEST_CONTEXTUAL_RETRIEVAL=false",
                "INGEST_CONTEXTUAL_MODEL=qwen-flash",
                "INGEST_CONTEXTUAL_MAX_TOKENS=120",
                "INGEST_CONTEXTUAL_TEMPERATURE=0",
                "INGEST_EMBEDDING_MODEL=text-embedding-v4",
                "INGEST_EMBEDDING_DIMENSION=none",
                "INGEST_EMBEDDING_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1",
                "INGEST_EMBEDDING_API_KEY_ENV_VAR=DASHSCOPE_API_KEY",
                "INGEST_EMBEDDING_BATCH_SIZE=10",
                "RETRIEVAL_SEARCH_MODE=bm25",
                "RETRIEVAL_FUSION_ALGORITHM=weighted",
                "RETRIEVAL_FINAL_TOP_K=5",
                "RETRIEVAL_ENABLE_PARENT_DOCUMENT_EXPANSION=false",
                "GENERATION_MODEL=qwen-flash",
                "GENERATION_MAX_TOKENS=300",
                "GENERATION_MIN_GROUNDEDNESS=0.4",
                "GENERATION_NO_ANSWER_TEXT=INSUFFICIENT_CONTEXT",
                "EVALUATION_RAGAS_ENABLED=true",
                "EVALUATION_INPUT_TOKEN_COST_PER_1K=0.001",
                "EVALUATION_OUTPUT_TOKEN_COST_PER_1K=0.002",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = get_config()

    assert config.data.limit == 7
    assert config.chroma.collection_name == "dotenv_chunks"
    assert config.bm25.store_path == Path("data/custom_bm25.json")
    assert config.bm25.algorithm == "BM25Okapi"
    assert config.chunking.contextual_retrieval is False
    assert config.contextual_retrieval.model == "qwen-flash"
    assert config.contextual_retrieval.max_tokens == 120
    assert config.embedding.model == "text-embedding-v4"
    assert config.embedding.dimension is None
    assert config.embedding.api_key_env_var == "DASHSCOPE_API_KEY"
    assert config.embedding.batch_size == 10
    assert config.retrieval.search_mode == "bm25"
    assert config.retrieval.fusion_algorithm == "weighted"
    assert config.retrieval.final_top_k == 5
    assert config.retrieval.enable_parent_document_expansion is False
    assert config.generation.model == "qwen-flash"
    assert config.generation.max_tokens == 300
    assert config.generation.min_groundedness == 0.4
    assert config.generation.no_answer_text == "INSUFFICIENT_CONTEXT"
    assert config.evaluation.ragas_enabled is True
    assert config.evaluation.input_token_cost_per_1k == 0.001
    assert config.evaluation.output_token_cost_per_1k == 0.002
