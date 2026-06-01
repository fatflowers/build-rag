"""Central configuration for the phased StratRAG evaluation pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    """Configuration for loading StratRAG-style records."""

    path: Path = Path("data/stratrag.jsonl")
    question_field: str = "question"
    candidate_doc_fields: tuple[str, ...] = (
        "candidate_docs",
        "context",
        "contexts",
        "documents",
        "docs",
        "paragraphs",
    )
    gold_index_fields: tuple[str, ...] = (
        "gold_indices",
        "gold_doc_indices",
        "supporting_doc_indices",
        "supporting_indices",
        "supporting_facts",
    )
    answer_fields: tuple[str, ...] = ("answer", "final_answer")
    question_type_fields: tuple[str, ...] = ("question_type", "type", "q_type")
    expected_candidate_count: int = 15
    expected_gold_count: int = 2


@dataclass(frozen=True)
class RetrievalConfig:
    """Configuration shared by deterministic retrieval experiments."""

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    alternate_embedding_model: str = "BAAI/bge-m3"
    top_k: int = 5
    strategy: str = "dense"
    enable_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for the later local-Ollama generation evaluation stage."""

    ollama_model: str = "qwen2.5"
    ragas_max_retries: int = 2
    ragas_timeout_seconds: float = 120.0


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    results_dir: Path = Path("results")
    log_level: str = "INFO"


def get_config() -> AppConfig:
    """Build configuration from defaults plus environment variable overrides."""

    data_path = Path(os.getenv("STRATRAG_DATA_PATH", str(DataConfig.path)))
    top_k = int(os.getenv("STRATRAG_TOP_K", str(RetrievalConfig.top_k)))

    data = DataConfig(path=data_path)
    retrieval = RetrievalConfig(
        embedding_model=os.getenv(
            "STRATRAG_EMBEDDING_MODEL",
            RetrievalConfig.embedding_model,
        ),
        top_k=top_k,
        strategy=os.getenv("STRATRAG_RETRIEVAL_STRATEGY", RetrievalConfig.strategy),
        enable_reranker=os.getenv("STRATRAG_ENABLE_RERANKER", "false").lower()
        in {"1", "true", "yes"},
        reranker_model=os.getenv("STRATRAG_RERANKER_MODEL", RetrievalConfig.reranker_model),
    )
    generation = GenerationConfig(
        ollama_model=os.getenv("STRATRAG_OLLAMA_MODEL", GenerationConfig.ollama_model),
    )

    return AppConfig(
        data=data,
        retrieval=retrieval,
        generation=generation,
        results_dir=Path(os.getenv("STRATRAG_RESULTS_DIR", "results")),
        log_level=os.getenv("STRATRAG_LOG_LEVEL", "INFO"),
    )
