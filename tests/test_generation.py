"""Smoke tests for grounded answer generation."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from haystack.dataclasses import Document

from src.config import AppConfig
from src.generation import build_generation_prompt, generate_answer


class FakeGenerator:
    """Deterministic generator for generation tests."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def run(self, prompt: str) -> Mapping[str, list[str]]:
        self.prompts.append(prompt)
        return {"replies": [self.reply]}


def test_generation_builds_prompt_and_citations() -> None:
    """Prompt assembly numbers retrieved documents for attribution."""

    document = Document(
        id="p1:parent",
        content="Scott Derrickson is an American filmmaker.",
        score=0.8,
        meta={"title": "Adam Collis", "source": "hotpotqa"},
    )

    prompt, citations = build_generation_prompt(
        AppConfig(),
        "What nationality is Scott Derrickson?",
        [document],
    )

    assert "[1]" in prompt
    assert "Scott Derrickson is an American filmmaker." in prompt
    assert citations[0].document_id == "p1:parent"
    assert citations[0].title == "Adam Collis"


def test_generate_answer_with_grounded_citation() -> None:
    """Generation returns answer text, cited source, and groundedness signals."""

    document = Document(
        id="p1:parent",
        content="Scott Derrickson is an American filmmaker.",
        score=0.8,
        meta={"title": "Adam Collis", "source": "hotpotqa"},
    )
    generator = FakeGenerator("Scott Derrickson is American [1].")

    answer = generate_answer(
        AppConfig(),
        "What nationality is Scott Derrickson?",
        [document],
        generator=generator,
    )

    assert answer.no_answer is False
    assert answer.answer == "Scott Derrickson is American [1]."
    assert answer.citations[0].citation_id == "1"
    assert answer.groundedness > 0.9
    assert answer.answer_relevance > 0.5
    assert generator.prompts


def test_generate_answer_falls_back_without_documents() -> None:
    """No retrieved context produces the no-answer fallback without an API call."""

    config = AppConfig(
        generation=replace(AppConfig().generation, no_answer_text="INSUFFICIENT_CONTEXT"),
    )
    generator = FakeGenerator("This should not be called.")

    answer = generate_answer(config, "Unknown question", [], generator=generator)

    assert answer.no_answer is True
    assert answer.answer == "INSUFFICIENT_CONTEXT"
    assert answer.citations == []
    assert generator.prompts == []


def test_generate_answer_falls_back_when_ungrounded() -> None:
    """Low lexical support triggers the groundedness no-answer fallback."""

    config = AppConfig(
        generation=replace(AppConfig().generation, min_groundedness=0.5),
    )
    document = Document(
        id="p1:parent",
        content="Scott Derrickson is an American filmmaker.",
        score=0.8,
        meta={"title": "Adam Collis", "source": "hotpotqa"},
    )
    generator = FakeGenerator("The answer is Canadian actor Ryan Reynolds [1].")

    answer = generate_answer(
        config,
        "What nationality is Scott Derrickson?",
        [document],
        generator=generator,
    )

    assert answer.no_answer is True
    assert answer.answer == "NO_ANSWER"
    assert answer.citations == []
    assert answer.groundedness < 0.5
