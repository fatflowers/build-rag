"""Logging and tracing setup shared by CLI entry points."""

from __future__ import annotations

import logging
import os


def configure_observability(
    *,
    log_level: str,
    trace_content: bool,
    openai_debug: bool,
    langfuse_enabled: bool = False,
) -> None:
    """Configure Python logging, Haystack content tracing, and OpenAI SDK debug logs."""

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(levelname)s:%(name)s:%(message)s",
    )
    if openai_debug or os.getenv("OPENAI_LOG") == "debug":
        for logger_name in ("openai", "openai._base_client", "httpx"):
            logging.getLogger(logger_name).setLevel(logging.DEBUG)
    if langfuse_enabled:
        os.environ.setdefault("HAYSTACK_CONTENT_TRACING_ENABLED", "true")
    haystack_trace_content = _truthy_env("HAYSTACK_TRACE_CONTENT")
    if langfuse_enabled or trace_content or haystack_trace_content:
        from haystack import tracing

        logging.getLogger("haystack").setLevel(logging.DEBUG)
        tracing.tracer.is_content_tracing_enabled = True

    if trace_content or haystack_trace_content:
        from haystack import tracing
        from haystack.tracing.logging_tracer import LoggingTracer

        tracing.enable_tracing(LoggingTracer())


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "false").lower() in {"1", "true", "yes"}
