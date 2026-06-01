"""Stage 1 entry point: load StratRAG data and print dataset statistics."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from src.config import get_config
from src.data_loader import load_benchmark_records, summarize_records


def main(argv: Optional[list[str]] = None) -> None:
    """Run Stage 1 data loading validation."""

    config = get_config()
    parser = argparse.ArgumentParser(description="Validate and summarize StratRAG data.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=config.data.path,
        help="Path to StratRAG JSON or JSONL data.",
    )
    parser.add_argument(
        "--benchmark",
        default=config.data.benchmark_name,
        help="Registered benchmark adapter name.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Record index to log as a complete normalized sample.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of records to load for quick inspection.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(levelname)s:%(name)s:%(message)s",
    )

    records = load_benchmark_records(
        args.data_path,
        benchmark_name=args.benchmark,
        config=config.data,
        limit=args.limit,
    )
    stats = summarize_records(records)
    logging.info("dataset_stats=%s", json.dumps(stats, ensure_ascii=False, sort_keys=True))

    if not records:
        logging.warning("No records found in %s", args.data_path)
        return

    if args.sample_index < 0 or args.sample_index >= len(records):
        raise IndexError(
            f"sample-index {args.sample_index} outside loaded record range [0, {len(records)})."
        )

    sample = records[args.sample_index].to_dict()
    logging.info("sample_record=%s", json.dumps(sample, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
