"""CLI entry point for Pipeline 1: Ingestion."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from src.config import AppConfig, BM25Config, ChromaConfig, get_config
from src.ingestion import run_ingestion
from src.observability import configure_observability


def main() -> None:
    """Run the HotpotQA ingestion pipeline."""

    defaults = get_config()
    parser = argparse.ArgumentParser(description="Run HotpotQA ingestion into Chroma.")
    parser.add_argument("--limit", type=int, default=defaults.data.limit)
    parser.add_argument("--split", default=defaults.data.split)
    parser.add_argument("--dataset-config", default=defaults.data.dataset_config)
    parser.add_argument("--chunks-path", type=Path, default=defaults.chunks_path)
    parser.add_argument("--manifest-path", type=Path, default=defaults.manifest_path)
    parser.add_argument("--chroma-path", type=Path, default=defaults.chroma.persist_path)
    parser.add_argument("--collection-name", default=defaults.chroma.collection_name)
    parser.add_argument("--bm25-store-path", type=Path, default=defaults.bm25.store_path)
    parser.add_argument("--skip-chroma", action="store_true")
    parser.add_argument("--skip-bm25", action="store_true")
    parser.add_argument(
        "--contextual-retrieval",
        dest="contextual_retrieval",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--skip-contextual-retrieval",
        dest="contextual_retrieval",
        action="store_false",
    )
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--trace-content", action="store_true")
    parser.add_argument("--openai-debug", action="store_true")
    args = parser.parse_args()

    configure_observability(
        log_level=defaults.log_level,
        trace_content=args.trace_content,
        openai_debug=args.openai_debug,
    )

    config = AppConfig(
        data=replace(
            defaults.data,
            limit=args.limit,
            split=args.split,
            dataset_config=args.dataset_config,
        ),
        chunking=replace(
            defaults.chunking,
            contextual_retrieval=(
                defaults.chunking.contextual_retrieval
                if args.contextual_retrieval is None
                else args.contextual_retrieval
            ),
        ),
        chroma=ChromaConfig(
            persist_path=args.chroma_path,
            collection_name=args.collection_name,
            distance_function=defaults.chroma.distance_function,
        ),
        bm25=BM25Config(
            store_path=args.bm25_store_path,
            algorithm=defaults.bm25.algorithm,
            tokenization_regex=defaults.bm25.tokenization_regex,
        ),
        contextual_retrieval=defaults.contextual_retrieval,
        embedding=defaults.embedding,
        chunks_path=args.chunks_path,
        manifest_path=args.manifest_path,
        log_level=defaults.log_level,
    )
    manifest = run_ingestion(
        config,
        skip_chroma=args.skip_chroma,
        skip_bm25=args.skip_bm25,
        rebuild=args.rebuild,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
