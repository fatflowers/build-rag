"""Smoke tests for the StratRAG data loader."""

from __future__ import annotations

from pathlib import Path

from src.config import DataConfig
from src.data_loader import (
    BenchmarkLoader,
    load_benchmark_records,
    load_stratrag_records,
    normalize_record,
    records_to_haystack_documents,
    summarize_records,
)


def test_load_sample_stratrag_fixture() -> None:
    """The loader normalizes canonical and HotpotQA-style context records."""

    records = load_stratrag_records(
        Path("tests/fixtures/sample_stratrag.jsonl"),
        config=DataConfig(),
    )

    assert len(records) == 2
    assert records[0].question_type == "bridge"
    assert records[0].gold_indices == [0, 2]
    assert len(records[0].candidate_docs) == 15
    assert records[1].candidate_docs[0].title == "Eiffel Tower"
    assert records[1].question_type == "comparison"

    stats = summarize_records(records)
    assert stats["total_questions"] == 2
    assert stats["benchmark_distribution"] == {"stratrag": 2}
    assert stats["question_type_distribution"] == {"bridge": 1, "comparison": 1}


def test_supporting_facts_titles_map_to_gold_indices() -> None:
    """HotpotQA-style supporting facts are mapped through candidate titles."""

    raw_record = {
        "question": "Which city is shared?",
        "context": [
            ["Doc A", ["A sentence."]],
            ["Doc B", ["B sentence."]],
            ["Doc C", ["C sentence."]],
        ],
        "supporting_facts": [["Doc B", 0], ["Doc C", 1], ["Doc B", 2]],
        "answer": "Doc B and Doc C",
        "type": "bridge",
    }

    record = normalize_record(raw_record, record_position=0, config=DataConfig())

    assert record.gold_indices == [1, 2]


def test_records_convert_to_haystack_documents_with_gold_metadata() -> None:
    """Candidate pools become Haystack Documents with deterministic labels."""

    records = load_benchmark_records(
        Path("tests/fixtures/sample_stratrag.jsonl"),
        benchmark_name="stratrag",
        config=DataConfig(),
        limit=1,
    )
    documents = records_to_haystack_documents(records)

    assert len(documents) == 15
    assert documents[0].content
    assert documents[0].meta["benchmark"] == "stratrag"
    assert documents[0].meta["source_index"] == 0
    assert documents[0].meta["is_gold"] is True
    assert documents[1].meta["is_gold"] is False
    assert documents[2].meta["is_gold"] is True


def test_benchmark_loader_haystack_component_outputs_records_documents_and_stats() -> None:
    """The loader can be used as a Haystack component in later pipelines."""

    loader = BenchmarkLoader(benchmark_name="stratrag", config=DataConfig())
    output = loader.run(data_path="tests/fixtures/sample_stratrag.jsonl", limit=1)

    assert len(output["records"]) == 1
    assert len(output["documents"]) == 15
    assert output["stats"]["total_questions"] == 1
