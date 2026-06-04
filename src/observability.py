"""Logging and tracing setup shared by CLI entry points."""

from __future__ import annotations

import logging
import os


def configure_observability(*, log_level: str, trace_content: bool, openai_debug: bool) -> None:
    """Configure Python logging, Haystack content tracing, and OpenAI SDK debug logs."""

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(levelname)s:%(name)s:%(message)s",
    )
    if openai_debug or os.getenv("OPENAI_LOG") == "debug":
        for logger_name in ("openai", "openai._base_client", "httpx"):
            logging.getLogger(logger_name).setLevel(logging.DEBUG)
    if trace_content or os.getenv("HAYSTACK_TRACE_CONTENT", "false").lower() in {
        "1",
        "true",
        "yes",
    }:
        from haystack import tracing
        from haystack.tracing.logging_tracer import LoggingTracer

        logging.getLogger("haystack").setLevel(logging.DEBUG)
        tracing.tracer.is_content_tracing_enabled = True
        tracing.enable_tracing(LoggingTracer())
