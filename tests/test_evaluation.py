"""Tests for retrieval, generation, and system evaluation metrics."""

from __future__ import annotations

from dataclasses import replace

from haystack.dataclasses import Document

from src.config import AppConfig
from src.evaluation import evaluate_rag_result, evaluate_retrieval
from src.generation import Citation, GeneratedAnswer
from src.retrieval import ProcessedQuery, RetrievalResult


def test_retrieval_metrics_use_parent_document_ids() -> None:
    """Retrieval metrics support HotpotQA parent-document relevance."""

    documents = [
        Document(id="p2:parent", content="Distractor", meta={"expanded_parent_doc_id": "p2"}),
        Document(id="p1:parent", content="Relevant", meta={"expanded_parent_doc_id": "p1"}),
        Document(id="p3:parent", content="Other", meta={"expanded_parent_doc_id": "p3"}),
    ]

    metrics = evaluate_retrieval(
        documents,
        relevant_parent_doc_ids={"p1"},
        k=3,
    )

    assert metrics.recall_at_k == 1.0
    assert metrics.precision_at_k == 1 / 3
    assert metrics.mrr == 0.5
    assert 0.0 < metrics.ndcg < 1.0
    assert metrics.hit_rate == 1.0


def test_rag_evaluation_report_includes_generation_and_system_metrics() -> None:
    """End-to-end report combines retrieval, generation, and cost metrics."""

    base = AppConfig()
    config = AppConfig(
        retrieval=replace(base.retrieval, final_top_k=2),
        evaluation=replace(
            base.evaluation,
            input_token_cost_per_1k=0.001,
            output_token_cost_per_1k=0.002,
        ),
    )
    documents = [
        Document(id="p1:parent", content="Scott Derrickson is American.", meta={"parent_doc_id": "p1"}),
        Document(id="p2:parent", content="Distractor.", meta={"parent_doc_id": "p2"}),
    ]
    retrieval_result = RetrievalResult(
        query=ProcessedQuery(
            original_query="What nationality is Scott Derrickson?",
            rewritten_query="What nationality is Scott Derrickson?",
            expanded_queries=[],
            hyde_document=None,
            route="bm25",
        ),
        documents=documents,
        filters=None,
        fusion_algorithm="rrf",
        timings={"total_seconds": 0.2},
    )
    answer = GeneratedAnswer(
        query="What nationality is Scott Derrickson?",
        answer="Scott Derrickson is American [1].",
        citations=[
            Citation(
                citation_id="1",
                document_id="p1:parent",
                title="Adam Collis",
                source="hotpotqa",
                score=0.8,
            )
        ],
        prompt="Sources: Scott Derrickson is American.",
        groundedness=1.0,
        answer_relevance=0.75,
        no_answer=False,
        timings={"total_seconds": 0.3},
    )

    report = evaluate_rag_result(
        config,
        retrieval_result,
        answer,
        relevant_parent_doc_ids={"p1"},
    )

    assert report.retrieval.hit_rate == 1.0
    assert report.generation.faithfulness == 1.0
    assert report.generation.citation_coverage == 1.0
    assert report.system.latency_seconds == 0.5
    assert report.system.estimated_cost > 0.0
    assert report.system.retrieved_documents == 2
