"""Langfuse integration helpers for Haystack pipelines."""

from __future__ import annotations

import os

from haystack import AsyncPipeline
from haystack.utils import Secret

from src.config import AppConfig, LangfuseConfig

LANGFUSE_COMPONENT_NAME = "langfuse_tracer"


def add_langfuse_connector(pipeline: AsyncPipeline, config: AppConfig, pipeline_name: str) -> None:
    """Attach a Langfuse tracing component when Langfuse is enabled."""

    if not config.langfuse.enabled:
        return
    _validate_langfuse_environment(config.langfuse)
    os.environ.setdefault("HAYSTACK_CONTENT_TRACING_ENABLED", "true")
    from haystack import tracing

    tracing.tracer.is_content_tracing_enabled = True
    pipeline.add_component(
        LANGFUSE_COMPONENT_NAME,
        _build_langfuse_connector(config.langfuse, _trace_name(config, pipeline_name)),
    )


def flush_langfuse_traces() -> None:
    """Flush pending Langfuse spans if the active Haystack tracer supports flushing."""

    from haystack.tracing import tracer

    flush = getattr(tracer.actual_tracer, "flush", None)
    if callable(flush):
        flush()


def _build_langfuse_connector(config: LangfuseConfig, trace_name: str) -> object:
    from haystack_integrations.components.connectors.langfuse import LangfuseConnector

    return LangfuseConnector(
        trace_name,
        public=config.public,
        public_key=Secret.from_env_var(config.public_key_env_var),
        secret_key=Secret.from_env_var(config.secret_key_env_var),
    )


def _trace_name(config: AppConfig, pipeline_name: str) -> str:
    return f"{config.langfuse.trace_name_prefix}: {pipeline_name}"


def _validate_langfuse_environment(config: LangfuseConfig) -> None:
    missing = [
        env_var
        for env_var in (config.public_key_env_var, config.secret_key_env_var)
        if not os.getenv(env_var)
    ]
    if missing:
        raise RuntimeError(
            "Langfuse tracing is enabled but required environment variables are missing: "
            + ", ".join(missing)
        )
