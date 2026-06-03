"""Generic external data source helpers."""

from src.data_sources.huggingface import DatasetRecord, read_huggingface_records

__all__ = [
    "DatasetRecord",
    "read_huggingface_records",
]
