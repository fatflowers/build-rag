"""Smoke tests for Pipeline 2 retrieval."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

from src.config import AppConfig, BM25Config
from src.retrieval import (
    MetadataFilterCriteria,
    build_metadata_filter,
    fuse_hybrid_results,
    run_retrieval,
)


def _write_bm25_store(path: Path) -> None:
    store = InMemoryDocumentStore(return_embedding=False)
    store.write_documents(
        [
            Document(
                id="p1:c0",
                content="Adam Collis worked with Scott Derrickson.",
                meta={
                    "parent_doc_id": "p1",
                    "split_id": 0,
                    "title": "Adam Collis",
                    "source": "hotpotqa",
                    "permissions": "public",
                },
            ),
            Document(
                id="p1:c1",
                content="Scott Derrickson is an American filmmaker.",
                meta={
                    "parent_doc_id": "p1",
                    "split_id": 1,
                    "title": "Adam Collis",
                    "source": "hotpotqa",
                    "permissions": "public",
                },
            ),
            Document(
                id="p2:c0",
                content="Ed Wood was also an American filmmaker.",
                meta={
                    "parent_doc_id": "p2",
                    "split_id": 0,
                    "title": "Ed Wood",
                    "source": "hotpotqa",
                    "permissions": "public",
                },
            ),
        ]
    )
    store.save_to_disk(str(path))


def test_retrieval_runs_bm25_with_parent_expansion(tmp_path: Path) -> None:
    """BM25 retrieval loads the ingested store and expands chunk hits to parent context."""

    store_path = tmp_path / "bm25.json"
    _write_bm25_store(store_path)
    base = AppConfig()
    config = AppConfig(
        bm25=BM25Config(store_path=store_path),
        retrieval=replace(
            base.retrieval,
            search_mode="bm25",
            final_top_k=1,
            enable_parent_document_expansion=True,
            max_context_chars_per_document=400,
        ),
    )

    result = run_retrieval(config, "Scott Derrickson")

    assert result.query.route == "bm25"
    assert result.documents
    assert result.documents[0].id == "p1:parent"
    assert "Adam Collis worked" in (result.documents[0].content or "")
    assert "American filmmaker" in (result.documents[0].content or "")


def test_metadata_filter_limits_retrieval_scope(tmp_path: Path) -> None:
    """Metadata filters are passed into Haystack retrieval."""

    store_path = tmp_path / "bm25.json"
    _write_bm25_store(store_path)
    base = AppConfig()
    config = AppConfig(
        bm25=BM25Config(store_path=store_path),
        retrieval=replace(
            base.retrieval,
            search_mode="bm25",
            final_top_k=2,
            enable_parent_document_expansion=False,
        ),
    )

    result = run_retrieval(
        config,
        "American filmmaker",
        metadata_filters=MetadataFilterCriteria(title="Ed Wood"),
    )

    assert result.filters == build_metadata_filter(MetadataFilterCriteria(title="Ed Wood"))
    assert result.documents
    assert all(document.meta["title"] == "Ed Wood" for document in result.documents)


def test_rrf_fusion_prefers_consistent_hits() -> None:
    """RRF fusion rewards documents retrieved by both dense and sparse search."""

    base = AppConfig()
    config = replace(base.retrieval, fusion_algorithm="rrf", rrf_k=60)
    dense_documents = [
        Document(id="shared", content="Dense shared", score=0.9),
        Document(id="dense-only", content="Dense only", score=0.8),
    ]
    bm25_documents = [
        Document(id="shared", content="Sparse shared", score=12.0),
        Document(id="bm25-only", content="Sparse only", score=10.0),
    ]

    fused = fuse_hybrid_results(
        dense_documents=dense_documents,
        bm25_documents=bm25_documents,
        config=config,
    )

    assert fused[0].id == "shared"
    assert fused[0].meta["fusion_algorithm"] == "rrf"
