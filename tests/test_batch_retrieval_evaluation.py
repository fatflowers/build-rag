"""Tests for batch HotpotQA retrieval evaluation."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

import src.batch_retrieval_evaluation as batch_eval
from haystack.dataclasses import Document

from src.config import AppConfig, HotpotQAConfig
from src.retrieval import ProcessedQuery, RetrievalResult


def test_batch_retrieval_evaluation_aggregates_hotpotqa_metrics(monkeypatch) -> None:
    """Batch evaluation derives gold parent ids and averages retrieval metrics."""

    rows: list[Mapping[str, object]] = [
        {
            "id": "q1",
            "question": "Where is the Eiffel Tower?",
            "answer": "Paris",
            "type": "bridge",
            "level": "easy",
            "context": {
                "title": ["Eiffel Tower", "Berlin"],
                "sentences": [
                    ["The Eiffel Tower is in Paris."],
                    ["Berlin is in Germany."],
                ],
            },
            "supporting_facts": {"title": ["Eiffel Tower"], "sent_id": [0]},
        },
        {
            "id": "q2",
            "question": "Who worked with Scott Derrickson?",
            "answer": "Adam Collis",
            "type": "bridge",
            "level": "medium",
            "context": {
                "title": ["Scott Derrickson", "Adam Collis"],
                "sentences": [
                    ["Scott Derrickson is a filmmaker."],
                    ["Adam Collis worked with Scott Derrickson."],
                ],
            },
            "supporting_facts": {"title": ["Adam Collis"], "sent_id": [0]},
        },
    ]

    class FakePipeline:
        def run(
            self,
            data: Mapping[str, Mapping[str, object]],
            *,
            include_outputs_from: set[str] | None = None,
        ) -> Mapping[str, Mapping[str, object]]:
            query = data["query_processor"]["query"]
            documents = [
                Document(
                    id="hotpotqa:q1:doc:0:parent",
                    content="The Eiffel Tower is in Paris.",
                    meta={"expanded_parent_doc_id": "hotpotqa:q1:doc:0"},
                ),
                Document(
                    id="hotpotqa:q1:doc:1:parent",
                    content="Berlin is in Germany.",
                    meta={"expanded_parent_doc_id": "hotpotqa:q1:doc:1"},
                ),
            ]
            if query == "Who worked with Scott Derrickson?":
                documents = [
                    Document(
                        id="hotpotqa:q2:doc:0:parent",
                        content="Scott Derrickson is a filmmaker.",
                        meta={"expanded_parent_doc_id": "hotpotqa:q2:doc:0"},
                    ),
                    Document(
                        id="hotpotqa:q2:doc:1:parent",
                        content="Adam Collis worked with Scott Derrickson.",
                        meta={"expanded_parent_doc_id": "hotpotqa:q2:doc:1"},
                    ),
                ]
            return {
                "result_builder": {
                    "result": RetrievalResult(
                        query=ProcessedQuery(
                            original_query=str(query),
                            rewritten_query=str(query),
                            expanded_queries=[],
                            hyde_document=None,
                            route="bm25",
                        ),
                        documents=documents,
                        filters=None,
                        fusion_algorithm="rrf",
                        timings={},
                    )
                }
            }

    def fake_read_huggingface_records(
        dataset_name: str,
        dataset_config: str,
        split: str,
        limit: int,
    ) -> list[Mapping[str, object]]:
        return rows[:limit]

    def fake_build_retrieval_pipeline(config: AppConfig) -> FakePipeline:
        return FakePipeline()

    monkeypatch.setattr(batch_eval, "read_huggingface_records", fake_read_huggingface_records)
    monkeypatch.setattr(batch_eval, "build_retrieval_pipeline", fake_build_retrieval_pipeline)
    config = AppConfig(
        data=HotpotQAConfig(limit=2),
        retrieval=replace(AppConfig().retrieval, search_mode="bm25", final_top_k=2),
    )

    report = batch_eval.evaluate_hotpotqa_retrieval_batch(config)
    payload = batch_eval.batch_retrieval_evaluation_report_to_json(report)

    assert report.summary.evaluated_count == 2
    assert report.summary.skipped_count == 0
    assert report.summary.recall_at_k == 1.0
    assert report.summary.precision_at_k == 0.5
    assert report.summary.mrr == 0.75
    assert report.summary.hit_rate == 1.0
    assert payload["summary"]["evaluated_count"] == 2
    assert payload["cases"][0]["relevant_parent_doc_ids"] == ["hotpotqa:q1:doc:0"]
