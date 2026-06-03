"""Generation, citation, and groundedness checks for retrieved HotpotQA context."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Mapping, Protocol, cast

from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.dataclasses import Document
from haystack.utils import Secret

from src.config import AppConfig
from src.retrieval import JsonValue

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_WORD_PATTERN = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}

_PROMPT_TEMPLATE = """
You are a precise RAG answer generator.
Use only the provided sources to answer the question.
Every factual claim must be supported by one or more citations like [1].
If the sources do not contain enough evidence, answer exactly: {{ no_answer_text }}

Sources:
{% for source in sources %}
[{{ source.citation_id }}]
Title: {{ source.title }}
Source: {{ source.source }}
Document ID: {{ source.document_id }}
Content:
{{ source.content }}

{% endfor %}
Question: {{ query }}
Answer:
""".strip()


@dataclass(frozen=True)
class Citation:
    """A source used for answer attribution."""

    citation_id: str
    document_id: str
    title: str
    source: str
    score: float | None


@dataclass(frozen=True)
class PromptSource:
    """Prompt-friendly source payload."""

    citation_id: str
    document_id: str
    title: str
    source: str
    content: str


@dataclass(frozen=True)
class GeneratedAnswer:
    """Generated answer plus attribution and quality signals."""

    query: str
    answer: str
    citations: list[Citation]
    prompt: str
    groundedness: float
    answer_relevance: float
    no_answer: bool
    timings: dict[str, float]


class TextGenerator(Protocol):
    """Text generation interface used by answer generation."""

    def run(self, prompt: str) -> Mapping[str, list[str]]:
        """Generate text for a prompt."""
        ...


def generate_answer(
    config: AppConfig,
    query: str,
    documents: list[Document],
    *,
    generator: TextGenerator | None = None,
) -> GeneratedAnswer:
    """Build a grounded prompt, generate an answer, and attach citations."""

    timings: dict[str, float] = {}
    start = time.perf_counter()
    prompt, citations = build_generation_prompt(config, query, documents)
    timings["prompt_build_seconds"] = time.perf_counter() - start

    if _should_fallback(config, documents):
        return _fallback_answer(config, query, prompt, timings)

    generation_start = time.perf_counter()
    active_generator = generator or _build_generator(config)
    answer = _first_reply(active_generator, prompt) or config.generation.no_answer_text
    timings["generation_seconds"] = time.perf_counter() - generation_start

    check_start = time.perf_counter()
    no_answer = _is_no_answer(answer, config.generation.no_answer_text)
    used_citations = _select_citations(answer, citations, no_answer)
    context = "\n".join(document.content or "" for document in documents)
    groundedness = 1.0 if no_answer else lexical_support_score(answer, context)
    if _is_ungrounded(config, groundedness, no_answer):
        answer = config.generation.no_answer_text
        no_answer = True
        used_citations = []
    relevance = 0.0 if no_answer else lexical_support_score(query, answer)
    timings["groundedness_check_seconds"] = time.perf_counter() - check_start
    timings["total_seconds"] = time.perf_counter() - start

    return GeneratedAnswer(
        query=query,
        answer=answer,
        citations=used_citations,
        prompt=prompt,
        groundedness=groundedness,
        answer_relevance=relevance,
        no_answer=no_answer,
        timings=timings,
    )


def build_generation_prompt(
    config: AppConfig,
    query: str,
    documents: list[Document],
) -> tuple[str, list[Citation]]:
    """Render the final RAG prompt with numbered sources."""

    sources: list[PromptSource] = []
    citations: list[Citation] = []
    for index, document in enumerate(documents, start=1):
        citation = _citation_from_document(str(index), document)
        citations.append(citation)
        sources.append(
            PromptSource(
                citation_id=citation.citation_id,
                document_id=citation.document_id,
                title=citation.title,
                source=citation.source,
                content=document.content or "",
            )
        )
    builder = PromptBuilder(
        template=_PROMPT_TEMPLATE,
        required_variables={"query", "sources", "no_answer_text"},
    )
    rendered = cast(
        Mapping[str, str],
        builder.run(
            query=query,
            sources=sources,
            no_answer_text=config.generation.no_answer_text,
        ),
    )
    return rendered["prompt"], citations


def lexical_support_score(text: str, evidence: str) -> float:
    """Estimate how much content vocabulary in text is present in evidence."""

    text_words = _content_words(text)
    if not text_words:
        return 0.0
    evidence_words = _content_words(evidence)
    return len(text_words & evidence_words) / len(text_words)


def generated_answer_to_json(answer: GeneratedAnswer) -> dict[str, JsonValue]:
    """Serialize generation output for CLI use."""

    return {
        "query": answer.query,
        "answer": answer.answer,
        "no_answer": answer.no_answer,
        "citations": [_citation_to_json(citation) for citation in answer.citations],
        "groundedness": answer.groundedness,
        "answer_relevance": answer.answer_relevance,
        "prompt": answer.prompt,
        "timings": {key: value for key, value in answer.timings.items()},
    }


def _build_generator(config: AppConfig) -> OpenAIGenerator:
    return OpenAIGenerator(
        model=config.generation.model,
        api_base_url=config.generation.api_base_url,
        api_key=Secret.from_env_var(config.generation.api_key_env_var),
        generation_kwargs={
            "max_tokens": config.generation.max_tokens,
            "temperature": config.generation.temperature,
        },
    )


def _first_reply(generator: TextGenerator, prompt: str) -> str | None:
    result = generator.run(prompt)
    replies = result.get("replies", [])
    if not replies:
        return None
    reply = replies[0].strip()
    return reply or None


def _should_fallback(config: AppConfig, documents: list[Document]) -> bool:
    if not documents:
        return True
    if config.generation.min_context_score <= 0:
        return False
    return max(_score(document) for document in documents) < config.generation.min_context_score


def _fallback_answer(
    config: AppConfig,
    query: str,
    prompt: str,
    timings: dict[str, float],
) -> GeneratedAnswer:
    timings["generation_seconds"] = 0.0
    timings["groundedness_check_seconds"] = 0.0
    timings["total_seconds"] = sum(timings.values())
    return GeneratedAnswer(
        query=query,
        answer=config.generation.no_answer_text,
        citations=[],
        prompt=prompt,
        groundedness=1.0,
        answer_relevance=0.0,
        no_answer=True,
        timings=timings,
    )


def _citation_from_document(citation_id: str, document: Document) -> Citation:
    title = _string_meta(document.meta, "title", "Untitled")
    source = _string_meta(document.meta, "source", "unknown")
    return Citation(
        citation_id=citation_id,
        document_id=str(document.id),
        title=title,
        source=source,
        score=document.score,
    )


def _select_citations(
    answer: str,
    citations: list[Citation],
    no_answer: bool,
) -> list[Citation]:
    if no_answer:
        return []
    cited_ids = set(_CITATION_PATTERN.findall(answer))
    selected = [citation for citation in citations if citation.citation_id in cited_ids]
    if selected or not citations:
        return selected
    return [citations[0]]


def _is_no_answer(answer: str, no_answer_text: str) -> bool:
    return answer.strip().upper() == no_answer_text.upper()


def _is_ungrounded(config: AppConfig, groundedness: float, no_answer: bool) -> bool:
    if no_answer:
        return False
    return groundedness < config.generation.min_groundedness


def _content_words(text: str) -> set[str]:
    words = {match.group(0).lower() for match in _WORD_PATTERN.finditer(text)}
    return {word for word in words if word not in _STOPWORDS and len(word) > 1}


def _string_meta(meta: Mapping[str, object], key: str, default: str) -> str:
    value = meta.get(key)
    if isinstance(value, str) and value:
        return value
    return default


def _score(document: Document) -> float:
    if document.score is None:
        return 0.0
    return float(document.score)


def _citation_to_json(citation: Citation) -> dict[str, JsonValue]:
    return {
        "citation_id": citation.citation_id,
        "document_id": citation.document_id,
        "title": citation.title,
        "source": citation.source,
        "score": citation.score,
    }
