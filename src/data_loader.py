"""Load benchmark QA records and convert candidate pools to Haystack documents."""

from __future__ import annotations

import json
import logging
import hashlib
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence

from haystack import Document, component

from src.config import DataConfig

logger = logging.getLogger(__name__)


class DataFormatError(ValueError):
    """Raised when a StratRAG record cannot be normalized."""


@dataclass(frozen=True)
class CandidateDocument:
    """A normalized candidate document from a benchmark candidate pool."""

    source_index: int
    content: str
    title: Optional[str] = None


@dataclass(frozen=True)
class BenchmarkRecord:
    """Canonical QA record shape used by all pipeline stages."""

    question: str
    candidate_docs: list[CandidateDocument]
    gold_indices: list[int]
    answer: str
    question_type: str
    record_id: Optional[str] = None
    benchmark_name: str = "stratrag"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the normalized record for JSON logging or fixtures."""

        return asdict(self)

    def to_haystack_documents(self) -> list[Document]:
        """Convert the record's candidate pool into Haystack `Document` objects.

        The gold labels are preserved in document metadata so deterministic
        evaluators can recover relevance labels after retrieval.
        """

        record_key = self.record_id or _stable_record_key(self.question)
        gold_index_set = set(self.gold_indices)
        documents: list[Document] = []

        for candidate in self.candidate_docs:
            documents.append(
                Document(
                    id=f"{self.benchmark_name}:{record_key}:doc:{candidate.source_index}",
                    content=candidate.content,
                    meta={
                        "benchmark": self.benchmark_name,
                        "record_id": self.record_id,
                        "question": self.question,
                        "answer": self.answer,
                        "question_type": self.question_type,
                        "source_index": candidate.source_index,
                        "title": candidate.title,
                        "is_gold": candidate.source_index in gold_index_set,
                        "gold_indices": self.gold_indices,
                    },
                )
            )

        return documents


StratRAGRecord = BenchmarkRecord


class BenchmarkAdapter(Protocol):
    """Adapter interface for benchmark-specific raw record normalization."""

    name: str

    def normalize(
        self,
        raw: Mapping[str, Any],
        record_position: int,
        config: DataConfig,
    ) -> BenchmarkRecord:
        """Map one raw benchmark record to the project canonical shape."""


@dataclass(frozen=True)
class StratRAGAdapter:
    """Adapter for StratRAG records derived from HotpotQA distractor pools."""

    name: str = "stratrag"

    def normalize(
        self,
        raw: Mapping[str, Any],
        record_position: int,
        config: DataConfig,
    ) -> BenchmarkRecord:
        """Normalize one raw StratRAG-style record."""

        return _normalize_stratrag_record(raw, record_position, config, self.name)


BENCHMARK_ADAPTERS: dict[str, BenchmarkAdapter] = {
    "stratrag": StratRAGAdapter(),
}


@component
class BenchmarkLoader:
    """Haystack component that loads benchmark records and candidate documents."""

    def __init__(
        self,
        benchmark_name: str = "stratrag",
        config: Optional[DataConfig] = None,
    ) -> None:
        """Create a benchmark loader component for Haystack pipelines."""

        self.benchmark_name = benchmark_name
        self.config = config or DataConfig(benchmark_name=benchmark_name)

    @component.output_types(records=list, documents=list, stats=dict)
    def run(self, data_path: Optional[str] = None, limit: Optional[int] = None) -> dict[str, Any]:
        """Load records and expose Haystack documents as component outputs."""

        path = Path(data_path) if data_path is not None else self.config.path
        records = load_benchmark_records(
            path,
            benchmark_name=self.benchmark_name,
            config=self.config,
            limit=limit,
        )
        documents = records_to_haystack_documents(records)
        return {
            "records": records,
            "documents": documents,
            "stats": summarize_records(records),
        }


def load_benchmark_records(
    path: Path,
    benchmark_name: str = "stratrag",
    config: Optional[DataConfig] = None,
    limit: Optional[int] = None,
) -> list[BenchmarkRecord]:
    """Load records for any registered benchmark adapter."""

    data_config = config or DataConfig(benchmark_name=benchmark_name)
    adapter = get_benchmark_adapter(benchmark_name)
    raw_records = _read_json_records(path)
    normalized: list[BenchmarkRecord] = []

    for idx, raw in enumerate(raw_records):
        if limit is not None and len(normalized) >= limit:
            break
        if not isinstance(raw, Mapping):
            raise DataFormatError(f"Record {idx} is not a JSON object: {type(raw)!r}")
        normalized.append(adapter.normalize(raw, idx, data_config))

    logger.info("Loaded %d %s records from %s", len(normalized), benchmark_name, path)
    return normalized


def load_stratrag_records(
    path: Path,
    config: Optional[DataConfig] = None,
    limit: Optional[int] = None,
) -> list[StratRAGRecord]:
    """Load StratRAG-style records from a JSON or JSONL file.

    Args:
        path: Local JSON/JSONL file.
        config: Data field aliases and validation expectations.
        limit: Optional maximum number of records to load.

    Returns:
        Normalized records in the canonical project shape.

    Raises:
        FileNotFoundError: If the file does not exist.
        DataFormatError: If records cannot be mapped to the expected shape.
    """

    return load_benchmark_records(
        path,
        benchmark_name="stratrag",
        config=config or DataConfig(benchmark_name="stratrag"),
        limit=limit,
    )


def get_benchmark_adapter(benchmark_name: str) -> BenchmarkAdapter:
    """Return a registered benchmark adapter by name."""

    try:
        return BENCHMARK_ADAPTERS[benchmark_name]
    except KeyError as exc:
        available = ", ".join(sorted(BENCHMARK_ADAPTERS))
        raise DataFormatError(
            f"Unsupported benchmark '{benchmark_name}'. Available adapters: {available}."
        ) from exc


def records_to_haystack_documents(records: Sequence[BenchmarkRecord]) -> list[Document]:
    """Flatten benchmark candidate pools into Haystack documents."""

    return [document for record in records for document in record.to_haystack_documents()]


def normalize_record(
    raw: Mapping[str, Any],
    record_position: int,
    config: Optional[DataConfig] = None,
) -> StratRAGRecord:
    """Normalize one raw StratRAG-style record."""

    data_config = config or DataConfig()
    return _normalize_stratrag_record(raw, record_position, data_config, "stratrag")


def _normalize_stratrag_record(
    raw: Mapping[str, Any],
    record_position: int,
    config: DataConfig,
    benchmark_name: str,
) -> BenchmarkRecord:
    """Normalize one raw StratRAG-style record."""

    data_config = config
    question = _require_str(raw, data_config.question_field, record_position)
    candidate_value = _require_first_present(
        raw,
        data_config.candidate_doc_fields,
        "candidate documents",
        record_position,
    )
    candidate_docs = _normalize_candidate_docs(candidate_value, record_position)
    gold_value = _require_first_present(
        raw,
        data_config.gold_index_fields,
        "gold indices",
        record_position,
    )
    gold_indices = _normalize_gold_indices(gold_value, candidate_docs, record_position)
    answer = _require_first_present(raw, data_config.answer_fields, "answer", record_position)
    if not isinstance(answer, str):
        answer = str(answer)

    question_type = _first_present(raw, data_config.question_type_fields)
    if question_type is None:
        question_type = "unknown"
        logger.warning(
            "Record %d has no question_type field; using 'unknown'. Available keys: %s",
            record_position,
            sorted(raw.keys()),
        )
    else:
        question_type = str(question_type)

    if len(candidate_docs) != data_config.expected_candidate_count:
        logger.warning(
            "Record %d has %d candidate docs; expected %d.",
            record_position,
            len(candidate_docs),
            data_config.expected_candidate_count,
        )
    if len(gold_indices) != data_config.expected_gold_count:
        logger.warning(
            "Record %d has %d gold indices; expected %d.",
            record_position,
            len(gold_indices),
            data_config.expected_gold_count,
        )

    _validate_gold_indices(gold_indices, len(candidate_docs), record_position)

    record_id_value = _first_present(raw, ("id", "_id", "qid", "question_id"))
    record_id = None if record_id_value is None else str(record_id_value)

    return BenchmarkRecord(
        question=question,
        candidate_docs=candidate_docs,
        gold_indices=gold_indices,
        answer=answer,
        question_type=question_type,
        record_id=record_id,
        benchmark_name=benchmark_name,
    )


def summarize_records(records: Sequence[BenchmarkRecord]) -> dict[str, Any]:
    """Compute deterministic Stage 1 dataset statistics."""

    benchmark_counts = Counter(record.benchmark_name for record in records)
    type_counts = Counter(record.question_type for record in records)
    candidate_counts = Counter(len(record.candidate_docs) for record in records)
    gold_counts = Counter(len(record.gold_indices) for record in records)
    return {
        "total_questions": len(records),
        "benchmark_distribution": dict(sorted(benchmark_counts.items())),
        "question_type_distribution": dict(sorted(type_counts.items())),
        "candidate_doc_count_distribution": {
            str(key): value for key, value in sorted(candidate_counts.items())
        },
        "gold_count_distribution": {str(key): value for key, value in sorted(gold_counts.items())},
    }


def _read_json_records(path: Path) -> list[Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"StratRAG data file not found: {path}. "
            "Set STRATRAG_DATA_PATH or place a JSON/JSONL file at data/stratrag.jsonl."
        )

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    records.append(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    raise DataFormatError(f"Invalid JSONL on line {line_number}: {exc}") from exc
        return records

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        for key in ("data", "records", "examples", "train", "validation", "test"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise DataFormatError(
        f"Expected {path} to contain a JSON array, JSONL records, or a dict with a list split."
    )


def _stable_record_key(question: str) -> str:
    """Create a stable fallback key for records without explicit IDs."""

    return hashlib.sha1(question.encode("utf-8")).hexdigest()[:16]


def _normalize_candidate_docs(value: Any, record_position: int) -> list[CandidateDocument]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise DataFormatError(
            f"Record {record_position} candidate docs must be a list; got {type(value)!r}."
        )

    return [
        _normalize_candidate_doc(raw_doc, source_index, record_position)
        for source_index, raw_doc in enumerate(value)
    ]


def _normalize_candidate_doc(
    raw_doc: Any,
    source_index: int,
    record_position: int,
) -> CandidateDocument:
    if isinstance(raw_doc, str):
        return CandidateDocument(source_index=source_index, content=raw_doc)

    if isinstance(raw_doc, Mapping):
        title = _first_present(raw_doc, ("title", "name", "doc_title", "wikipedia_title"))
        content = _first_present(
            raw_doc,
            ("content", "text", "passage", "paragraph", "body", "document"),
        )
        sentences = _first_present(raw_doc, ("sentences", "sentence_list"))
        if content is None and isinstance(sentences, Sequence) and not isinstance(sentences, str):
            content = " ".join(str(sentence) for sentence in sentences)
        if content is None:
            raise DataFormatError(
                f"Record {record_position} candidate doc {source_index} has no content field. "
                f"Available keys: {sorted(raw_doc.keys())}"
            )
        return CandidateDocument(
            source_index=source_index,
            title=None if title is None else str(title),
            content=str(content),
        )

    if isinstance(raw_doc, Sequence) and not isinstance(raw_doc, (str, bytes)):
        if len(raw_doc) >= 2 and isinstance(raw_doc[1], Sequence) and not isinstance(raw_doc[1], str):
            title = str(raw_doc[0])
            content = " ".join(str(sentence) for sentence in raw_doc[1])
            return CandidateDocument(source_index=source_index, title=title, content=content)
        content = " ".join(str(part) for part in raw_doc)
        return CandidateDocument(source_index=source_index, content=content)

    raise DataFormatError(
        f"Record {record_position} candidate doc {source_index} has unsupported type: "
        f"{type(raw_doc)!r}."
    )


def _normalize_gold_indices(
    value: Any,
    candidate_docs: Sequence[CandidateDocument],
    record_position: int,
) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise DataFormatError(f"Record {record_position} gold indices must be a list.")

    title_to_index = {
        doc.title: doc.source_index for doc in candidate_docs if doc.title is not None
    }
    gold_indices: list[int] = []

    for item in value:
        if isinstance(item, int):
            gold_indices.append(item)
            continue
        if isinstance(item, str) and item in title_to_index:
            gold_indices.append(title_to_index[item])
            continue
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            title_value = item[0] if item else None
            if isinstance(title_value, str) and title_value in title_to_index:
                gold_indices.append(title_to_index[title_value])
                continue
        if isinstance(item, Mapping):
            index_value = _first_present(item, ("index", "doc_index", "gold_index"))
            if isinstance(index_value, int):
                gold_indices.append(index_value)
                continue
            title_value = _first_present(item, ("title", "doc_title", "name"))
            if isinstance(title_value, str) and title_value in title_to_index:
                gold_indices.append(title_to_index[title_value])
                continue
        raise DataFormatError(
            f"Record {record_position} gold item {item!r} cannot be mapped to a candidate index."
        )

    deduplicated_indices: list[int] = []
    seen_indices: set[int] = set()
    for gold_index in gold_indices:
        if gold_index not in seen_indices:
            deduplicated_indices.append(gold_index)
            seen_indices.add(gold_index)
    return deduplicated_indices


def _validate_gold_indices(gold_indices: Iterable[int], doc_count: int, record_position: int) -> None:
    for gold_index in gold_indices:
        if gold_index < 0 or gold_index >= doc_count:
            raise DataFormatError(
                f"Record {record_position} gold index {gold_index} is outside candidate "
                f"range [0, {doc_count})."
            )


def _require_str(raw: Mapping[str, Any], field: str, record_position: int) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise DataFormatError(
            f"Record {record_position} requires non-empty string field '{field}'. "
            f"Available keys: {sorted(raw.keys())}"
        )
    return value


def _require_first_present(
    raw: Mapping[str, Any],
    fields: Sequence[str],
    label: str,
    record_position: int,
) -> Any:
    value = _first_present(raw, fields)
    if value is None:
        raise DataFormatError(
            f"Record {record_position} has no {label} field. Tried aliases {tuple(fields)}. "
            f"Available keys: {sorted(raw.keys())}"
        )
    return value


def _first_present(raw: Mapping[str, Any], fields: Sequence[str]) -> Any:
    for field in fields:
        if field in raw and raw[field] is not None:
            return raw[field]
    return None
