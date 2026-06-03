"""Shared HotpotQA ingestion types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


RawRecord = Mapping[str, object]


class HotpotQAFormatError(ValueError):
    """Raised when a HotpotQA record cannot be normalized."""


@dataclass(frozen=True)
class HotpotQAStats:
    """Basic dataset loading statistics."""

    records: int
    source_documents: int
    supporting_documents: int
    supporting_facts: int
    matched_supporting_facts: int
