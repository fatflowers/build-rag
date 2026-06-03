"""Retrieval, generation, and system-level evaluation helpers."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from haystack.dataclasses import Document

from src.config import AppConfig
from src.generation import GeneratedAnswer
from src.retrieval import JsonValue, RetrievalResult

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class RetrievalMetrics:
    """Standard retrieval quality metrics at k."""

    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg: float
    hit_rate: float


@dataclass(frozen=True)
class GenerationMetrics:
    """Answer quality metrics."""

    faithfulness: float
    answer_relevance: float
    citation_coverage: float
    no_answer: bool


@dataclass(frozen=True)
class SystemMetrics:
    """System-level observability metrics."""

    latency_seconds: float
    estimated_cost: float
    estimated_input_tokens: int
    estimated_output_tokens: int
    retrieved_documents: int
    prompt_chars: int
    answer_chars: int
    hit_rate: float


@dataclass(frozen=True)
class RagEvaluationReport:
    """Complete evaluation report for one RAG query."""

    retrieval: RetrievalMetrics
    generation: GenerationMetrics
    system: SystemMetrics
    ragas_enabled: bool
    ragas_status: str


def evaluate_retrieval(
    documents: list[Document],
    *,
    relevant_document_ids: set[str] | None = None,
    relevant_parent_doc_ids: set[str] | None = None,
    k: int,
) -> RetrievalMetrics:
    """Compute recall@k, precision@k, MRR, nDCG, and hit rate."""

    doc_ids = relevant_document_ids or set()
    parent_ids = relevant_parent_doc_ids or set()
    target_count = len(parent_ids) if parent_ids else len(doc_ids)
    if target_count == 0 or k <= 0:
        return RetrievalMetrics(0.0, 0.0, 0.0, 0.0, 0.0)

    ranked = documents[:k]
    relevance = [_is_relevant(document, doc_ids, parent_ids) for document in ranked]
    hit_count = sum(1 for relevant in relevance if relevant)
    recall = min(hit_count / target_count, 1.0)
    precision = hit_count / k
    reciprocal_rank = _reciprocal_rank(relevance)
    ndcg = _ndcg(relevance, min(target_count, k))
    hit_rate = 1.0 if hit_count > 0 else 0.0
    return RetrievalMetrics(recall, precision, reciprocal_rank, ndcg, hit_rate)


def evaluate_generation(answer: GeneratedAnswer) -> GenerationMetrics:
    """Compute lightweight faithfulness, relevance, and citation coverage."""

    return GenerationMetrics(
        faithfulness=answer.groundedness,
        answer_relevance=answer.answer_relevance,
        citation_coverage=_citation_coverage(answer),
        no_answer=answer.no_answer,
    )


def evaluate_system(
    config: AppConfig,
    retrieval_result: RetrievalResult,
    answer: GeneratedAnswer,
    *,
    retrieval_hit_rate: float,
) -> SystemMetrics:
    """Compute latency, cost, context size, and hit-rate observability metrics."""

    input_tokens = _estimate_tokens(answer.prompt)
    output_tokens = _estimate_tokens(answer.answer)
    estimated_cost = (
        (input_tokens / 1000) * config.evaluation.input_token_cost_per_1k
        + (output_tokens / 1000) * config.evaluation.output_token_cost_per_1k
    )
    return SystemMetrics(
        latency_seconds=retrieval_result.timings.get("total_seconds", 0.0)
        + answer.timings.get("total_seconds", 0.0),
        estimated_cost=estimated_cost,
        estimated_input_tokens=input_tokens,
        estimated_output_tokens=output_tokens,
        retrieved_documents=len(retrieval_result.documents),
        prompt_chars=len(answer.prompt),
        answer_chars=len(answer.answer),
        hit_rate=retrieval_hit_rate,
    )


def evaluate_rag_result(
    config: AppConfig,
    retrieval_result: RetrievalResult,
    answer: GeneratedAnswer,
    *,
    relevant_document_ids: set[str] | None = None,
    relevant_parent_doc_ids: set[str] | None = None,
) -> RagEvaluationReport:
    """Evaluate a retrieved-and-generated answer."""

    retrieval = evaluate_retrieval(
        retrieval_result.documents,
        relevant_document_ids=relevant_document_ids,
        relevant_parent_doc_ids=relevant_parent_doc_ids,
        k=config.retrieval.final_top_k,
    )
    generation = evaluate_generation(answer)
    system = evaluate_system(
        config,
        retrieval_result,
        answer,
        retrieval_hit_rate=retrieval.hit_rate,
    )
    return RagEvaluationReport(
        retrieval=retrieval,
        generation=generation,
        system=system,
        ragas_enabled=config.evaluation.ragas_enabled,
        ragas_status=_ragas_status(config),
    )


def evaluation_report_to_json(report: RagEvaluationReport) -> dict[str, JsonValue]:
    """Serialize evaluation output for CLI use."""

    return {
        "retrieval": {
            "recall_at_k": report.retrieval.recall_at_k,
            "precision_at_k": report.retrieval.precision_at_k,
            "mrr": report.retrieval.mrr,
            "ndcg": report.retrieval.ndcg,
            "hit_rate": report.retrieval.hit_rate,
        },
        "generation": {
            "faithfulness": report.generation.faithfulness,
            "answer_relevance": report.generation.answer_relevance,
            "citation_coverage": report.generation.citation_coverage,
            "no_answer": report.generation.no_answer,
        },
        "system": {
            "latency_seconds": report.system.latency_seconds,
            "estimated_cost": report.system.estimated_cost,
            "estimated_input_tokens": report.system.estimated_input_tokens,
            "estimated_output_tokens": report.system.estimated_output_tokens,
            "retrieved_documents": report.system.retrieved_documents,
            "prompt_chars": report.system.prompt_chars,
            "answer_chars": report.system.answer_chars,
            "hit_rate": report.system.hit_rate,
        },
        "ragas": {
            "enabled": report.ragas_enabled,
            "status": report.ragas_status,
        },
    }


def _is_relevant(
    document: Document,
    relevant_document_ids: set[str],
    relevant_parent_doc_ids: set[str],
) -> bool:
    if str(document.id) in relevant_document_ids:
        return True
    parent_id = document.meta.get("parent_doc_id") or document.meta.get("expanded_parent_doc_id")
    return isinstance(parent_id, str) and parent_id in relevant_parent_doc_ids


def _reciprocal_rank(relevance: list[bool]) -> float:
    for rank, relevant in enumerate(relevance, start=1):
        if relevant:
            return 1.0 / rank
    return 0.0


def _ndcg(relevance: list[bool], ideal_relevant_count: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, relevant in enumerate(relevance, start=1)
        if relevant
    )
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_relevant_count + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def _citation_coverage(answer: GeneratedAnswer) -> float:
    if answer.no_answer:
        return 1.0
    referenced_ids = set(_CITATION_PATTERN.findall(answer.answer))
    if not referenced_ids:
        return 0.0
    valid_ids = {citation.citation_id for citation in answer.citations}
    return len(referenced_ids & valid_ids) / len(referenced_ids)


def _estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _ragas_status(config: AppConfig) -> str:
    if config.evaluation.ragas_enabled:
        return "configured_for_external_ragas_run"
    return "disabled"
