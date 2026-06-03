"""Smoke tests for Pipeline 3 RAG graph."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Mapping

from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

from src.config import AppConfig, BM25Config
from src.rag import run_rag


class FakeGenerator:
    """Deterministic generator for RAG pipeline tests."""

    def run(self, prompt: str) -> Mapping[str, list[str]]:
        return {"replies": ["Scott Derrickson is American [1]."]}


def _write_bm25_store(path: Path) -> None:
    store = InMemoryDocumentStore(return_embedding=False)
    store.write_documents(
        [
            Document(
                id="p1:c0",
                content="Scott Derrickson is an American filmmaker.",
                meta={
                    "parent_doc_id": "p1",
                    "split_id": 0,
                    "title": "Adam Collis",
                    "source": "hotpotqa",
                    "permissions": "public",
                },
            ),
        ]
    )
    store.save_to_disk(str(path))


def test_rag_pipeline_runs_retrieval_generation_and_evaluation(tmp_path: Path) -> None:
    """Pipeline 3 is executed as a Haystack graph."""

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
        ),
    )

    result = run_rag(
        config,
        "What nationality is Scott Derrickson?",
        relevant_parent_doc_ids={"p1"},
        generator=FakeGenerator(),
    )

    assert result.retrieval.documents
    assert result.generation.answer == "Scott Derrickson is American [1]."
    assert result.evaluation.retrieval.hit_rate == 1.0
    assert result.evaluation.generation.citation_coverage == 1.0
