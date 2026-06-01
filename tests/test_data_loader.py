"""Smoke tests for the StratRAG data loader."""

from __future__ import annotations

from pathlib import Path

from src.config import DataConfig
from src.data_loader import load_stratrag_records, normalize_record, summarize_records


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
