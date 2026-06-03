"""Pipeline 3: retrieval-augmented generation and evaluation as a Haystack Pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, cast

from haystack import Pipeline, component

from src.config import AppConfig
from src.evaluation import RagEvaluationReport, evaluate_rag_result, evaluation_report_to_json
from src.generation import GeneratedAnswer, TextGenerator, generate_answer, generated_answer_to_json
from src.retrieval import (
    JsonValue,
    MetadataFilterCriteria,
    RetrievalResult,
    add_retrieval_pipeline_components,
    retrieval_result_to_json,
)


@dataclass(frozen=True)
class RagPipelineResult:
    """Complete Pipeline 3 output."""

    retrieval: RetrievalResult
    generation: GeneratedAnswer
    evaluation: RagEvaluationReport


@component
class AnswerGeneratorComponent:
    """Generate a grounded answer from a retrieval result."""

    def __init__(
        self,
        config: AppConfig,
        generator: TextGenerator | None = None,
    ) -> None:
        self.config = config
        self.generator = generator

    @component.output_types(answer=GeneratedAnswer)
    def run(self, retrieval_result: RetrievalResult) -> dict[str, GeneratedAnswer]:
        """Generate the RAG answer for the retrieved documents."""

        answer = generate_answer(
            self.config,
            retrieval_result.query.original_query,
            retrieval_result.documents,
            generator=self.generator,
        )
        return {"answer": answer}


@component
class RagEvaluationComponent:
    """Evaluate a full RAG result."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    @component.output_types(report=RagEvaluationReport)
    def run(
        self,
        retrieval_result: RetrievalResult,
        answer: GeneratedAnswer,
        relevant_document_ids: list[str] | None = None,
        relevant_parent_doc_ids: list[str] | None = None,
    ) -> dict[str, RagEvaluationReport]:
        """Evaluate retrieval, generation, and system metrics."""

        report = evaluate_rag_result(
            self.config,
            retrieval_result,
            answer,
            relevant_document_ids=set(relevant_document_ids or []),
            relevant_parent_doc_ids=set(relevant_parent_doc_ids or []),
        )
        return {"report": report}


def build_rag_pipeline(
    config: AppConfig,
    *,
    generator: TextGenerator | None = None,
) -> Pipeline:
    """Build Pipeline 3 as a Haystack Pipeline graph."""

    pipeline = Pipeline()
    add_retrieval_pipeline_components(pipeline, config)
    pipeline.add_component("answer_generator", AnswerGeneratorComponent(config, generator))
    pipeline.add_component("rag_evaluator", RagEvaluationComponent(config))
    pipeline.connect("result_builder.result", "answer_generator.retrieval_result")
    pipeline.connect("result_builder.result", "rag_evaluator.retrieval_result")
    pipeline.connect("answer_generator.answer", "rag_evaluator.answer")
    return pipeline


def run_rag(
    config: AppConfig,
    query: str,
    *,
    metadata_filters: MetadataFilterCriteria | None = None,
    relevant_document_ids: set[str] | None = None,
    relevant_parent_doc_ids: set[str] | None = None,
    generator: TextGenerator | None = None,
) -> RagPipelineResult:
    """Run Pipeline 3 through a Haystack Pipeline graph."""

    pipeline = build_rag_pipeline(config, generator=generator)
    output = cast(
        Mapping[str, Mapping[str, object]],
        pipeline.run(
            {
                "query_processor": {"query": query},
                "metadata_filter": {
                    "criteria": metadata_filters or MetadataFilterCriteria(),
                },
                "rag_evaluator": {
                    "relevant_document_ids": list(relevant_document_ids or set()),
                    "relevant_parent_doc_ids": list(relevant_parent_doc_ids or set()),
                },
            },
            include_outputs_from={"result_builder", "answer_generator", "rag_evaluator"},
        ),
    )
    retrieval = output["result_builder"]["result"]
    generation = output["answer_generator"]["answer"]
    evaluation = output["rag_evaluator"]["report"]
    if not isinstance(retrieval, RetrievalResult):
        raise TypeError("RAG pipeline did not return a RetrievalResult.")
    if not isinstance(generation, GeneratedAnswer):
        raise TypeError("RAG pipeline did not return a GeneratedAnswer.")
    if not isinstance(evaluation, RagEvaluationReport):
        raise TypeError("RAG pipeline did not return a RagEvaluationReport.")
    return RagPipelineResult(
        retrieval=retrieval,
        generation=generation,
        evaluation=evaluation,
    )


def rag_pipeline_result_to_json(result: RagPipelineResult) -> dict[str, JsonValue]:
    """Serialize Pipeline 3 output for CLI use."""

    return {
        "retrieval": retrieval_result_to_json(result.retrieval),
        "generation": generated_answer_to_json(result.generation),
        "evaluation": evaluation_report_to_json(result.evaluation),
    }
