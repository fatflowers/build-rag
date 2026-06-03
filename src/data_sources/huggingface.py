"""Generic Hugging Face dataset access."""

from __future__ import annotations

from typing import Mapping, TypeAlias, cast

from datasets import load_dataset


DatasetRecord: TypeAlias = Mapping[str, object]


def read_huggingface_records(
    dataset_name: str,
    dataset_config: str,
    split: str,
    limit: int,
) -> list[DatasetRecord]:
    """Read mapping records from a Hugging Face dataset split."""

    split_expr = f"{split}[:{limit}]" if limit > 0 else split
    dataset = load_dataset(dataset_name, dataset_config, split=split_expr)
    records: list[DatasetRecord] = []
    for record in dataset:
        if not isinstance(record, Mapping):
            raise ValueError("Hugging Face dataset returned a non-mapping record.")
        records.append(cast(DatasetRecord, dict(record)))
    return records
