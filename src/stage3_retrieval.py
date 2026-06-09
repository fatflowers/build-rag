"""CLI entry point for Pipeline 2: Retrieval."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from src.config import AppConfig, BM25Config, RetrievalConfig, get_config
from src.langfuse_tracing import flush_langfuse_traces
from src.observability import configure_observability
from src.retrieval import MetadataFilterCriteria, retrieval_result_to_json, run_retrieval_async


def main() -> None:
    """Run retrieval over BM25 and optionally Chroma."""

    defaults = get_config()
    parser = argparse.ArgumentParser(description="Run HotpotQA retrieval.")
    parser.add_argument("query")
    parser.add_argument("--bm25-store-path", type=Path, default=defaults.bm25.store_path)
    parser.add_argument(
        "--search-mode",
        choices=["hybrid", "dense", "bm25"],
        default=defaults.retrieval.search_mode,
    )
    parser.add_argument(
        "--fusion",
        choices=["rrf", "weighted"],
        default=defaults.retrieval.fusion_algorithm,
    )
    parser.add_argument("--dense-top-k", type=int, default=defaults.retrieval.dense_top_k)
    parser.add_argument("--bm25-top-k", type=int, default=defaults.retrieval.bm25_top_k)
    parser.add_argument("--final-top-k", type=int, default=defaults.retrieval.final_top_k)
    parser.add_argument("--rerank", action="store_true", default=defaults.retrieval.enable_rerank)
    parser.add_argument("--no-rerank", action="store_false", dest="rerank")
    parser.add_argument(
        "--parent-expansion",
        action="store_true",
        default=defaults.retrieval.enable_parent_document_expansion,
    )
    parser.add_argument("--no-parent-expansion", action="store_false", dest="parent_expansion")
    parser.add_argument(
        "--context-compression",
        action="store_true",
        default=defaults.retrieval.enable_context_compression,
    )
    parser.add_argument("--no-context-compression", action="store_false", dest="context_compression")
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=defaults.retrieval.max_context_chars_per_document,
    )
    parser.add_argument("--filter-source")
    parser.add_argument("--filter-title")
    parser.add_argument("--filter-level")
    parser.add_argument("--filter-type")
    parser.add_argument("--filter-permissions")
    parser.add_argument("--trace-content", action="store_true")
    parser.add_argument("--openai-debug", action="store_true")
    parser.add_argument("--concurrency-limit", type=int, default=4)
    args = parser.parse_args()

    configure_observability(
        log_level=defaults.log_level,
        trace_content=args.trace_content,
        openai_debug=args.openai_debug,
        langfuse_enabled=defaults.langfuse.enabled,
    )

    config = AppConfig(
        data=defaults.data,
        chunking=defaults.chunking,
        chroma=defaults.chroma,
        bm25=BM25Config(
            store_path=args.bm25_store_path,
            algorithm=defaults.bm25.algorithm,
            tokenization_regex=defaults.bm25.tokenization_regex,
        ),
        contextual_retrieval=defaults.contextual_retrieval,
        embedding=defaults.embedding,
        langfuse=defaults.langfuse,
        query_processing=defaults.query_processing,
        retrieval=RetrievalConfig(
            search_mode=args.search_mode,
            fusion_algorithm=args.fusion,
            rrf_k=defaults.retrieval.rrf_k,
            dense_weight=defaults.retrieval.dense_weight,
            bm25_weight=defaults.retrieval.bm25_weight,
            dense_top_k=args.dense_top_k,
            bm25_top_k=args.bm25_top_k,
            final_top_k=args.final_top_k,
            enable_rerank=args.rerank,
            reranker_model=defaults.retrieval.reranker_model,
            rerank_top_k=defaults.retrieval.rerank_top_k,
            enable_context_compression=args.context_compression,
            max_context_chars_per_document=args.max_context_chars,
            enable_parent_document_expansion=args.parent_expansion,
        ),
        chunks_path=defaults.chunks_path,
        manifest_path=defaults.manifest_path,
        log_level=defaults.log_level,
    )
    filters = MetadataFilterCriteria(
        source=args.filter_source,
        title=args.filter_title,
        level=args.filter_level,
        question_type=args.filter_type,
        permissions=args.filter_permissions,
    )
    try:
        result = asyncio.run(
            run_retrieval_async(
                config,
                args.query,
                metadata_filters=filters,
                concurrency_limit=args.concurrency_limit,
            )
        )
        print(json.dumps(retrieval_result_to_json(result), ensure_ascii=False, sort_keys=True))
    finally:
        flush_langfuse_traces()


if __name__ == "__main__":
    main()
