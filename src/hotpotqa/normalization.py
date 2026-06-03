"""Normalize HotpotQA rows into Haystack Documents."""

from __future__ import annotations

import json
from typing import Iterable, Mapping, Sequence, TypeGuard, cast

from haystack.dataclasses import Document

from src.hotpotqa.types import HotpotQAFormatError, RawRecord


def normalize_hotpotqa_record(raw: RawRecord, position: int) -> list[Document]:
    """Map one HotpotQA row to document-level Haystack Documents."""

    question_id = str(_first_present(raw, ("id", "_id", "qid")) or f"record-{position}")
    question = _require_str(raw, "question", position)
    answer = str(_first_present(raw, ("answer", "final_answer")) or "")
    question_type = str(_first_present(raw, ("type", "question_type")) or "unknown")
    level = str(_first_present(raw, ("level", "difficulty")) or "unknown")
    context_items = _normalize_context(raw.get("context"), position)
    support_map = _normalize_supporting_facts(raw.get("supporting_facts"))

    documents: list[Document] = []
    for doc_index, (title, sentences) in enumerate(context_items):
        content = " ".join(sentence.strip() for sentence in sentences if sentence.strip())
        if not content:
            continue

        supporting_sentence_ids = sorted(support_map.get(title, set()))
        supporting_texts = [
            sentences[sentence_id]
            for sentence_id in supporting_sentence_ids
            if 0 <= sentence_id < len(sentences)
        ]
        parent_doc_id = f"hotpotqa:{question_id}:doc:{doc_index}"
        documents.append(
            Document(
                id=parent_doc_id,
                content=content,
                meta={
                    "source": "hotpotqa",
                    "question_id": question_id,
                    "question": question,
                    "answer": answer,
                    "type": question_type,
                    "level": level,
                    "title": title,
                    "section": title,
                    "date": "",
                    "permissions": "public",
                    "parent_doc_id": parent_doc_id,
                    "source_doc_index": doc_index,
                    "is_supporting_doc": bool(supporting_sentence_ids),
                    "supporting_sentence_ids": ",".join(str(i) for i in supporting_sentence_ids),
                    "supporting_sentence_count": len(supporting_sentence_ids),
                    "supporting_sentence_texts_json": json.dumps(
                        supporting_texts,
                        ensure_ascii=False,
                    ),
                },
            )
        )

    return documents


def count_supporting_facts(value: object) -> int:
    """Count supporting fact references in a HotpotQA row."""

    if isinstance(value, Mapping):
        supporting_facts = cast(RawRecord, value)
        titles = supporting_facts.get("title") or supporting_facts.get("titles")
        sentence_ids = (
            supporting_facts.get("sent_id")
            or supporting_facts.get("sent_ids")
            or supporting_facts.get("sentence_id")
        )
        if _is_sequence(titles) and _is_sequence(sentence_ids):
            return min(len(titles), len(sentence_ids))
        return 0
    if _is_sequence(value):
        return sum(
            1
            for item in value
            if _is_sequence(item) and len(item) >= 2
        )
    return 0


def _normalize_context(value: object, position: int) -> list[tuple[str, list[str]]]:
    if isinstance(value, Mapping):
        context = cast(RawRecord, value)
        titles = context.get("title") or context.get("titles")
        sentences = context.get("sentences") or context.get("sentence")
        if _is_sequence(titles) and _is_sequence(sentences):
            return [
                (str(title), _stringify_sentences(sentence_list))
                for title, sentence_list in zip(titles, sentences)
            ]

    if _is_sequence(value):
        context_items: list[tuple[str, list[str]]] = []
        for item in value:
            if _is_sequence(item) and len(item) >= 2:
                title = str(item[0])
                raw_sentences = item[1]
                if _is_sequence(raw_sentences):
                    context_items.append((title, [str(sentence) for sentence in raw_sentences]))
                else:
                    context_items.append((title, [str(raw_sentences)]))
        if context_items:
            return context_items

    raise HotpotQAFormatError(f"Record {position} has unsupported HotpotQA context format.")


def _normalize_supporting_facts(value: object) -> dict[str, set[int]]:
    support_map: dict[str, set[int]] = {}
    if value is None:
        return support_map

    pairs: Iterable[tuple[object, object]]
    if isinstance(value, Mapping):
        supporting_facts = cast(RawRecord, value)
        titles = supporting_facts.get("title") or supporting_facts.get("titles")
        sentence_ids = (
            supporting_facts.get("sent_id")
            or supporting_facts.get("sent_ids")
            or supporting_facts.get("sentence_id")
        )
        if not _is_sequence(titles) or not _is_sequence(sentence_ids):
            return support_map
        pairs = zip(titles, sentence_ids)
    elif _is_sequence(value):
        pairs = (
            (item[0], item[1])
            for item in value
            if _is_sequence(item) and len(item) >= 2
        )
    else:
        return support_map

    for title, sentence_id in pairs:
        try:
            support_map.setdefault(str(title), set()).add(int(sentence_id))
        except (TypeError, ValueError):
            continue
    return support_map


def _require_str(raw: RawRecord, field: str, position: int) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise HotpotQAFormatError(f"Record {position} requires non-empty string field '{field}'.")
    return value


def _first_present(raw: RawRecord, fields: Sequence[str]) -> object | None:
    for field in fields:
        if field in raw and raw[field] is not None:
            return raw[field]
    return None


def _is_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _stringify_sentences(value: object) -> list[str]:
    if _is_sequence(value):
        return [str(sentence) for sentence in value]
    return [str(value)]
