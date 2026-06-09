"""Batch retrieval evaluation over HotpotQA records."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Awaitable, Mapping, Protocol, cast

from src.config import AppConfig
from src.data_sources import read_huggingface_records
from src.evaluation import RetrievalMetrics, evaluate_retrieval
from src.hotpotqa import HotpotQAFormatError, normalize_hotpotqa_record
from src.retrieval import (
    JsonValue,
    MetadataFilterCriteria,
    RetrievalResult,
    build_retrieval_pipeline,
)


@dataclass(frozen=True)
class HotpotQARetrievalExample:
    """One HotpotQA query and its gold supporting parent documents."""

    question_id: str
    question: str
    answer: str
    relevant_parent_doc_ids: set[str]


@dataclass(frozen=True)
class BatchRetrievalEvaluationCase:
    """Retrieval metrics for one HotpotQA example."""

    question_id: str
    question: str
    answer: str
    relevant_parent_doc_ids: list[str]
    retrieved_document_ids: list[str]
    retrieved_parent_doc_ids: list[str]
    metrics: RetrievalMetrics
    latency_seconds: float


@dataclass(frozen=True)
class BatchRetrievalSummary:
    """Aggregate retrieval metrics across a dataset slice."""

    evaluated_count: int
    skipped_count: int
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg: float
    hit_rate: float
    latency_seconds: float
    average_latency_seconds: float


@dataclass(frozen=True)
class BatchRetrievalEvaluationReport:
    """Dataset-level retrieval evaluation report."""

    dataset_name: str
    dataset_config: str
    split: str
    limit: int
    k: int
    search_mode: str
    fusion_algorithm: str
    summary: BatchRetrievalSummary
    cases: list[BatchRetrievalEvaluationCase]


class RetrievalEvaluationPipeline(Protocol):
    """Pipeline interface used by batch retrieval evaluation."""

    def run(
        self,
        data: Mapping[str, Mapping[str, object]],
        *,
        include_outputs_from: set[str] | None = None,
    ) -> Mapping[str, Mapping[str, object]]:
        """Run the pipeline synchronously."""
        ...

    def run_async(
        self,
        data: Mapping[str, Mapping[str, object]],
        *,
        include_outputs_from: set[str] | None = None,
        concurrency_limit: int = 4,
    ) -> Awaitable[Mapping[str, Mapping[str, object]]]:
        """Run the pipeline asynchronously."""
        ...


def load_hotpotqa_retrieval_examples(config: AppConfig) -> list[HotpotQARetrievalExample]:
    """Load HotpotQA records and derive gold supporting parent document ids."""

    rows = read_huggingface_records(
        config.data.dataset_name,
        config.data.dataset_config,
        config.data.split,
        config.data.limit,
    )
    examples: list[HotpotQARetrievalExample] = []
    for position, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise HotpotQAFormatError(f"Record {position} is not a JSON object.")
        documents = normalize_hotpotqa_record(raw, position)
        if not documents:
            continue
        question_id_value = documents[0].meta.get("question_id")
        question_value = documents[0].meta.get("question")
        answer_value = documents[0].meta.get("answer")
        if not isinstance(question_id_value, str) or not isinstance(question_value, str):
            raise HotpotQAFormatError(f"Record {position} is missing normalized question metadata.")
        relevant_parent_doc_ids = {
            str(document.meta["parent_doc_id"])
            for document in documents
            if document.meta.get("is_supporting_doc") is True
        }
        examples.append(
            HotpotQARetrievalExample(
                question_id=question_id_value,
                question=question_value,
                answer=answer_value if isinstance(answer_value, str) else "",
                relevant_parent_doc_ids=relevant_parent_doc_ids,
            )
        )
    return examples


def evaluate_hotpotqa_retrieval_batch(
    config: AppConfig,
    *,
    metadata_filters: MetadataFilterCriteria | None = None,
) -> BatchRetrievalEvaluationReport:
    """Run retrieval for a HotpotQA slice and aggregate retrieval metrics."""

    examples = load_hotpotqa_retrieval_examples(config)
    pipeline = build_retrieval_pipeline(config)
    cases: list[BatchRetrievalEvaluationCase] = []
    skipped_count = 0

    for example in examples:
        if not example.relevant_parent_doc_ids:
            skipped_count += 1
            continue
        cases.append(
            _evaluate_retrieval_example(
                pipeline=pipeline,
                config=config,
                example=example,
                metadata_filters=metadata_filters,
            )
        )

    return _build_batch_retrieval_report(config, cases=cases, skipped_count=skipped_count)


async def evaluate_hotpotqa_retrieval_batch_async(
    config: AppConfig,
    *,
    metadata_filters: MetadataFilterCriteria | None = None,
    concurrency_limit: int = 4,
) -> BatchRetrievalEvaluationReport:
    """Run retrieval evaluation with Haystack AsyncPipeline.run_async."""

    examples = load_hotpotqa_retrieval_examples(config)
    pipeline = build_retrieval_pipeline(config)
    cases: list[BatchRetrievalEvaluationCase] = []
    skipped_count = 0

    for example in examples:
        if not example.relevant_parent_doc_ids:
            skipped_count += 1
            continue
        cases.append(
            await _evaluate_retrieval_example_async(
                pipeline=pipeline,
                config=config,
                example=example,
                metadata_filters=metadata_filters,
                concurrency_limit=concurrency_limit,
            )
        )

    return _build_batch_retrieval_report(config, cases=cases, skipped_count=skipped_count)


def _evaluate_retrieval_example(
    *,
    pipeline: RetrievalEvaluationPipeline,
    config: AppConfig,
    example: HotpotQARetrievalExample,
    metadata_filters: MetadataFilterCriteria | None,
) -> BatchRetrievalEvaluationCase:
    start = time.perf_counter()
    output = cast(
        Mapping[str, Mapping[str, object]],
        pipeline.run(
            _pipeline_inputs(example, metadata_filters),
            include_outputs_from={"result_builder"},
        ),
    )
    return _build_batch_retrieval_case(
        config=config,
        example=example,
        output=output,
        elapsed_seconds=time.perf_counter() - start,
    )


async def _evaluate_retrieval_example_async(
    *,
    pipeline: RetrievalEvaluationPipeline,
    config: AppConfig,
    example: HotpotQARetrievalExample,
    metadata_filters: MetadataFilterCriteria | None,
    concurrency_limit: int,
) -> BatchRetrievalEvaluationCase:
    start = time.perf_counter()
    output = cast(
        Mapping[str, Mapping[str, object]],
        await pipeline.run_async(
            _pipeline_inputs(example, metadata_filters),
            include_outputs_from={"result_builder"},
            concurrency_limit=concurrency_limit,
        ),
    )
    return _build_batch_retrieval_case(
        config=config,
        example=example,
        output=output,
        elapsed_seconds=time.perf_counter() - start,
    )


def _pipeline_inputs(
    example: HotpotQARetrievalExample,
    metadata_filters: MetadataFilterCriteria | None,
) -> dict[str, dict[str, object]]:
    return {
        "query_processor": {"query": example.question},
        "metadata_filter": {
            "criteria": metadata_filters or MetadataFilterCriteria(),
        },
    }


def _build_batch_retrieval_case(
    *,
    config: AppConfig,
    example: HotpotQARetrievalExample,
    output: Mapping[str, Mapping[str, object]],
    elapsed_seconds: float,
) -> BatchRetrievalEvaluationCase:
    result = output["result_builder"]["result"]
    if not isinstance(result, RetrievalResult):
        raise TypeError("Retrieval pipeline did not return a RetrievalResult.")

    metrics = evaluate_retrieval(
        result.documents,
        relevant_parent_doc_ids=example.relevant_parent_doc_ids,
        k=config.retrieval.final_top_k,
    )
    retrieved_parent_doc_ids: list[str] = []
    for document in result.documents:
        parent_doc_id = document.meta.get("expanded_parent_doc_id") or document.meta.get(
            "parent_doc_id"
        )
        if isinstance(parent_doc_id, str):
            retrieved_parent_doc_ids.append(parent_doc_id)
    return BatchRetrievalEvaluationCase(
        question_id=example.question_id,
        question=example.question,
        answer=example.answer,
        relevant_parent_doc_ids=sorted(example.relevant_parent_doc_ids),
        retrieved_document_ids=[str(document.id) for document in result.documents],
        retrieved_parent_doc_ids=retrieved_parent_doc_ids,
        metrics=metrics,
        latency_seconds=elapsed_seconds,
    )


def _build_batch_retrieval_report(
    config: AppConfig,
    *,
    cases: list[BatchRetrievalEvaluationCase],
    skipped_count: int,
) -> BatchRetrievalEvaluationReport:
    total_latency_seconds = sum(case.latency_seconds for case in cases)
    evaluated_count = len(cases)
    if evaluated_count == 0:
        summary = BatchRetrievalSummary(
            evaluated_count=0,
            skipped_count=skipped_count,
            recall_at_k=0.0,
            precision_at_k=0.0,
            mrr=0.0,
            ndcg=0.0,
            hit_rate=0.0,
            latency_seconds=0.0,
            average_latency_seconds=0.0,
        )
    else:
        summary = BatchRetrievalSummary(
            evaluated_count=evaluated_count,
            skipped_count=skipped_count,
            recall_at_k=sum(case.metrics.recall_at_k for case in cases) / evaluated_count,
            precision_at_k=sum(case.metrics.precision_at_k for case in cases) / evaluated_count,
            mrr=sum(case.metrics.mrr for case in cases) / evaluated_count,
            ndcg=sum(case.metrics.ndcg for case in cases) / evaluated_count,
            hit_rate=sum(case.metrics.hit_rate for case in cases) / evaluated_count,
            latency_seconds=total_latency_seconds,
            average_latency_seconds=total_latency_seconds / evaluated_count,
        )

    return BatchRetrievalEvaluationReport(
        dataset_name=config.data.dataset_name,
        dataset_config=config.data.dataset_config,
        split=config.data.split,
        limit=config.data.limit,
        k=config.retrieval.final_top_k,
        search_mode=config.retrieval.search_mode,
        fusion_algorithm=config.retrieval.fusion_algorithm,
        summary=summary,
        cases=cases,
    )


def batch_retrieval_evaluation_report_to_json(
    report: BatchRetrievalEvaluationReport,
    *,
    include_cases: bool = True,
) -> dict[str, JsonValue]:
    """Serialize a batch retrieval evaluation report for CLI output."""

    return {
        "dataset": {
            "name": report.dataset_name,
            "config": report.dataset_config,
            "split": report.split,
            "limit": report.limit,
        },
        "retrieval": {
            "k": report.k,
            "search_mode": report.search_mode,
            "fusion_algorithm": report.fusion_algorithm,
        },
        "summary": {
            "evaluated_count": report.summary.evaluated_count,
            "skipped_count": report.summary.skipped_count,
            "recall_at_k": report.summary.recall_at_k,
            "precision_at_k": report.summary.precision_at_k,
            "mrr": report.summary.mrr,
            "ndcg": report.summary.ndcg,
            "hit_rate": report.summary.hit_rate,
            "latency_seconds": report.summary.latency_seconds,
            "average_latency_seconds": report.summary.average_latency_seconds,
        },
        "cases": [
            {
                "question_id": case.question_id,
                "question": case.question,
                "answer": case.answer,
                "relevant_parent_doc_ids": case.relevant_parent_doc_ids,
                "retrieved_document_ids": case.retrieved_document_ids,
                "retrieved_parent_doc_ids": case.retrieved_parent_doc_ids,
                "metrics": {
                    "recall_at_k": case.metrics.recall_at_k,
                    "precision_at_k": case.metrics.precision_at_k,
                    "mrr": case.metrics.mrr,
                    "ndcg": case.metrics.ndcg,
                    "hit_rate": case.metrics.hit_rate,
                },
                "latency_seconds": case.latency_seconds,
            }
            for case in report.cases
        ]
        if include_cases
        else [],
    }
