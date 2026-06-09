"""Tests for optional Langfuse tracing integration."""

from __future__ import annotations

from haystack import AsyncPipeline, component

import src.langfuse_tracing as langfuse_tracing
from src.config import AppConfig, LangfuseConfig
from src.langfuse_tracing import LANGFUSE_COMPONENT_NAME, add_langfuse_connector


@component
class FakeLangfuseConnector:
    """Minimal stand-in for the real Langfuse connector."""

    def __init__(self, name: str) -> None:
        self.name = name

    @component.output_types(name=str)
    def run(self) -> dict[str, str]:
        return {"name": self.name}


class FakeTracer:
    """Tracer with a flush method for shutdown tests."""

    def __init__(self) -> None:
        self.flush_count = 0

    def flush(self) -> None:
        self.flush_count += 1


def test_add_langfuse_connector_is_noop_when_disabled() -> None:
    """Langfuse stays out of the pipeline unless explicitly enabled."""

    pipeline = AsyncPipeline()

    add_langfuse_connector(pipeline, AppConfig(), "retrieval")

    assert LANGFUSE_COMPONENT_NAME not in pipeline.graph.nodes


def test_add_langfuse_connector_adds_unconnected_component(monkeypatch) -> None:
    """Enabled Langfuse tracing adds a Haystack connector component."""

    trace_names: list[str] = []

    def fake_builder(config: LangfuseConfig, trace_name: str) -> FakeLangfuseConnector:
        trace_names.append(trace_name)
        return FakeLangfuseConnector(trace_name)

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setattr(langfuse_tracing, "_build_langfuse_connector", fake_builder)
    pipeline = AsyncPipeline()
    config = AppConfig(
        langfuse=LangfuseConfig(enabled=True, trace_name_prefix="tests"),
    )

    add_langfuse_connector(pipeline, config, "ingestion")

    assert LANGFUSE_COMPONENT_NAME in pipeline.graph.nodes
    assert trace_names == ["tests: ingestion"]
    assert pipeline.inputs() == {}


def test_flush_langfuse_traces_calls_active_tracer(monkeypatch) -> None:
    """Shutdown flushing delegates to Haystack's active tracer when supported."""

    from haystack.tracing import tracer

    fake_tracer = FakeTracer()
    monkeypatch.setattr(tracer, "actual_tracer", fake_tracer)

    langfuse_tracing.flush_langfuse_traces()

    assert fake_tracer.flush_count == 1
