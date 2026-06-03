"""Load HotpotQA records and normalize them into Documents."""

from __future__ import annotations

from typing import Mapping

from haystack.dataclasses import Document

from src.data_sources import read_huggingface_records
from src.hotpotqa import (
    HotpotQAFormatError,
    HotpotQAStats,
    RawRecord,
    count_supporting_facts,
    normalize_hotpotqa_record,
)


def load_hotpotqa_documents(
    *,
    dataset_name: str,
    dataset_config: str,
    split: str,
    limit: int,
) -> tuple[list[Document], HotpotQAStats]:
    """Load HotpotQA from Hugging Face datasets."""

    rows = read_huggingface_records(dataset_name, dataset_config, split, limit)
    documents: list[Document] = []
    supporting_doc_count = 0
    supporting_fact_count = 0
    matched_supporting_fact_count = 0

    for position, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise HotpotQAFormatError(f"Record {position} is not a JSON object.")
        row_documents = normalize_hotpotqa_record(raw, position)
        supporting_fact_count += count_supporting_facts(raw.get("supporting_facts"))
        supporting_doc_count += sum(
            1 for document in row_documents if document.meta["is_supporting_doc"]
        )
        matched_supporting_fact_count += sum(
            int(document.meta["supporting_sentence_count"]) for document in row_documents
        )
        documents.extend(row_documents)

    stats = HotpotQAStats(
        records=len(rows),
        source_documents=len(documents),
        supporting_documents=supporting_doc_count,
        supporting_facts=supporting_fact_count,
        matched_supporting_facts=matched_supporting_fact_count,
    )
    return documents, stats


__all__ = [
    "HotpotQAFormatError",
    "HotpotQAStats",
    "RawRecord",
    "load_hotpotqa_documents",
    "normalize_hotpotqa_record",
]
