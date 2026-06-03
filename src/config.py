"""Configuration for the HotpotQA RAG ingestion pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class HotpotQAConfig:
    """HotpotQA loading settings."""

    dataset_name: str = "hotpotqa/hotpot_qa"
    dataset_config: str = "fullwiki"
    split: str = "validation"
    limit: int = 20


@dataclass(frozen=True)
class ChunkingConfig:
    """Haystack DocumentSplitter settings."""

    split_by: str = "sentence"
    split_length: int = 4
    split_overlap: int = 1
    split_threshold: int = 0
    contextual_retrieval: bool = True


@dataclass(frozen=True)
class ChromaConfig:
    """Chroma dense index settings."""

    persist_path: Path = Path("data/chroma_hotpotqa")
    collection_name: str = "hotpotqa_chunks"
    distance_function: str = "cosine"


@dataclass(frozen=True)
class BM25Config:
    """Haystack in-memory BM25 index settings."""

    store_path: Path = Path("data/hotpotqa_bm25_store.json")
    algorithm: Literal["BM25Okapi", "BM25L", "BM25Plus"] = "BM25L"
    tokenization_regex: str = r"(?u)\b\w+\b"


@dataclass(frozen=True)
class ContextualRetrievalConfig:
    """LLM settings for Anthropic-style contextual retrieval."""

    model: str = "qwen-flash"
    api_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key_env_var: str = "DASHSCOPE_API_KEY"
    max_tokens: int = 120
    temperature: float = 0.0


@dataclass(frozen=True)
class EmbeddingConfig:
    """Dense embedding model settings."""

    provider: str = "dashscope"
    model: str = "text-embedding-v4"
    dimension: int = 1024
    api_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key_env_var: str = "DASHSCOPE_API_KEY"
    batch_size: int = 10
    language_support: str = "Multilingual"
    asymmetric: bool = False


@dataclass(frozen=True)
class QueryProcessingConfig:
    """Query processing switches for retrieval."""

    enable_rewrite: bool = False
    enable_expand: bool = False
    enable_hyde: bool = False
    model: str = "qwen-flash"
    api_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key_env_var: str = "DASHSCOPE_API_KEY"
    max_tokens: int = 256
    temperature: float = 0.0


@dataclass(frozen=True)
class RetrievalConfig:
    """Pipeline 2 retrieval settings."""

    search_mode: Literal["hybrid", "dense", "bm25"] = "hybrid"
    fusion_algorithm: Literal["rrf", "weighted"] = "rrf"
    rrf_k: int = 60
    dense_weight: float = 0.5
    bm25_weight: float = 0.5
    dense_top_k: int = 20
    bm25_top_k: int = 20
    final_top_k: int = 8
    enable_rerank: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 8
    enable_context_compression: bool = True
    max_context_chars_per_document: int = 1200
    enable_parent_document_expansion: bool = True


@dataclass(frozen=True)
class GenerationConfig:
    """Downstream answer generation settings."""

    model: str = "qwen-flash"
    api_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key_env_var: str = "DASHSCOPE_API_KEY"
    max_tokens: int = 512
    temperature: float = 0.0
    min_context_score: float = 0.0
    min_groundedness: float = 0.2
    no_answer_text: str = "NO_ANSWER"


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation and system metric settings."""

    ragas_enabled: bool = False
    input_token_cost_per_1k: float = 0.0
    output_token_cost_per_1k: float = 0.0


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    data: HotpotQAConfig = HotpotQAConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    chroma: ChromaConfig = ChromaConfig()
    bm25: BM25Config = BM25Config()
    contextual_retrieval: ContextualRetrievalConfig = ContextualRetrievalConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    query_processing: QueryProcessingConfig = QueryProcessingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    generation: GenerationConfig = GenerationConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    chunks_path: Path = Path("data/hotpotqa_chunks.jsonl")
    manifest_path: Path = Path("results/ingestion_manifest.json")
    log_level: str = "INFO"


def get_config() -> AppConfig:
    """Build configuration from defaults and environment variables."""

    load_dotenv(Path.cwd() / ".env")

    limit = int(os.getenv("HOTPOTQA_LIMIT", str(HotpotQAConfig.limit)))

    return AppConfig(
        data=HotpotQAConfig(
            dataset_name=os.getenv("HOTPOTQA_DATASET_NAME", HotpotQAConfig.dataset_name),
            dataset_config=os.getenv("HOTPOTQA_DATASET_CONFIG", HotpotQAConfig.dataset_config),
            split=os.getenv("HOTPOTQA_SPLIT", HotpotQAConfig.split),
            limit=limit,
        ),
        chunking=ChunkingConfig(
            split_by=os.getenv("INGEST_SPLIT_BY", ChunkingConfig.split_by),
            split_length=int(os.getenv("INGEST_SPLIT_LENGTH", str(ChunkingConfig.split_length))),
            split_overlap=int(os.getenv("INGEST_SPLIT_OVERLAP", str(ChunkingConfig.split_overlap))),
            split_threshold=int(
                os.getenv("INGEST_SPLIT_THRESHOLD", str(ChunkingConfig.split_threshold))
            ),
            contextual_retrieval=os.getenv("INGEST_CONTEXTUAL_RETRIEVAL", "true").lower()
            in {"1", "true", "yes"},
        ),
        chroma=ChromaConfig(
            persist_path=Path(os.getenv("CHROMA_PERSIST_PATH", str(ChromaConfig.persist_path))),
            collection_name=os.getenv("CHROMA_COLLECTION_NAME", ChromaConfig.collection_name),
            distance_function=os.getenv("CHROMA_DISTANCE_FUNCTION", ChromaConfig.distance_function),
        ),
        bm25=BM25Config(
            store_path=Path(os.getenv("BM25_STORE_PATH", str(BM25Config.store_path))),
            algorithm=_parse_bm25_algorithm(os.getenv("BM25_ALGORITHM", BM25Config.algorithm)),
            tokenization_regex=os.getenv(
                "BM25_TOKENIZATION_REGEX",
                BM25Config.tokenization_regex,
            ),
        ),
        contextual_retrieval=ContextualRetrievalConfig(
            model=os.getenv("INGEST_CONTEXTUAL_MODEL", ContextualRetrievalConfig.model),
            api_base_url=os.getenv(
                "INGEST_CONTEXTUAL_API_BASE_URL",
                ContextualRetrievalConfig.api_base_url,
            ),
            api_key_env_var=os.getenv(
                "INGEST_CONTEXTUAL_API_KEY_ENV_VAR",
                ContextualRetrievalConfig.api_key_env_var,
            ),
            max_tokens=int(
                os.getenv("INGEST_CONTEXTUAL_MAX_TOKENS", str(ContextualRetrievalConfig.max_tokens))
            ),
            temperature=float(
                os.getenv("INGEST_CONTEXTUAL_TEMPERATURE", str(ContextualRetrievalConfig.temperature))
            ),
        ),
        embedding=EmbeddingConfig(
            provider=os.getenv("INGEST_EMBEDDING_PROVIDER", EmbeddingConfig.provider),
            model=os.getenv("INGEST_EMBEDDING_MODEL", EmbeddingConfig.model),
            dimension=int(os.getenv("INGEST_EMBEDDING_DIMENSION", str(EmbeddingConfig.dimension))),
            api_base_url=os.getenv("INGEST_EMBEDDING_API_BASE_URL", EmbeddingConfig.api_base_url),
            api_key_env_var=os.getenv(
                "INGEST_EMBEDDING_API_KEY_ENV_VAR",
                EmbeddingConfig.api_key_env_var,
            ),
            batch_size=int(
                os.getenv("INGEST_EMBEDDING_BATCH_SIZE", str(EmbeddingConfig.batch_size))
            ),
            language_support=os.getenv(
                "INGEST_EMBEDDING_LANGUAGE_SUPPORT",
                EmbeddingConfig.language_support,
            ),
            asymmetric=os.getenv("INGEST_EMBEDDING_ASYMMETRIC", "false").lower()
            in {"1", "true", "yes"},
        ),
        query_processing=QueryProcessingConfig(
            enable_rewrite=os.getenv("RETRIEVAL_ENABLE_REWRITE", "false").lower()
            in {"1", "true", "yes"},
            enable_expand=os.getenv("RETRIEVAL_ENABLE_EXPAND", "false").lower()
            in {"1", "true", "yes"},
            enable_hyde=os.getenv("RETRIEVAL_ENABLE_HYDE", "false").lower()
            in {"1", "true", "yes"},
            model=os.getenv("RETRIEVAL_QUERY_MODEL", QueryProcessingConfig.model),
            api_base_url=os.getenv(
                "RETRIEVAL_QUERY_API_BASE_URL",
                QueryProcessingConfig.api_base_url,
            ),
            api_key_env_var=os.getenv(
                "RETRIEVAL_QUERY_API_KEY_ENV_VAR",
                QueryProcessingConfig.api_key_env_var,
            ),
            max_tokens=int(
                os.getenv("RETRIEVAL_QUERY_MAX_TOKENS", str(QueryProcessingConfig.max_tokens))
            ),
            temperature=float(
                os.getenv("RETRIEVAL_QUERY_TEMPERATURE", str(QueryProcessingConfig.temperature))
            ),
        ),
        retrieval=RetrievalConfig(
            search_mode=_parse_search_mode(
                os.getenv("RETRIEVAL_SEARCH_MODE", RetrievalConfig.search_mode)
            ),
            fusion_algorithm=_parse_fusion_algorithm(
                os.getenv("RETRIEVAL_FUSION_ALGORITHM", RetrievalConfig.fusion_algorithm)
            ),
            rrf_k=int(os.getenv("RETRIEVAL_RRF_K", str(RetrievalConfig.rrf_k))),
            dense_weight=float(
                os.getenv("RETRIEVAL_DENSE_WEIGHT", str(RetrievalConfig.dense_weight))
            ),
            bm25_weight=float(os.getenv("RETRIEVAL_BM25_WEIGHT", str(RetrievalConfig.bm25_weight))),
            dense_top_k=int(os.getenv("RETRIEVAL_DENSE_TOP_K", str(RetrievalConfig.dense_top_k))),
            bm25_top_k=int(os.getenv("RETRIEVAL_BM25_TOP_K", str(RetrievalConfig.bm25_top_k))),
            final_top_k=int(os.getenv("RETRIEVAL_FINAL_TOP_K", str(RetrievalConfig.final_top_k))),
            enable_rerank=os.getenv("RETRIEVAL_ENABLE_RERANK", "false").lower()
            in {"1", "true", "yes"},
            reranker_model=os.getenv("RETRIEVAL_RERANKER_MODEL", RetrievalConfig.reranker_model),
            rerank_top_k=int(
                os.getenv("RETRIEVAL_RERANK_TOP_K", str(RetrievalConfig.rerank_top_k))
            ),
            enable_context_compression=os.getenv(
                "RETRIEVAL_ENABLE_CONTEXT_COMPRESSION",
                "true",
            ).lower()
            in {"1", "true", "yes"},
            max_context_chars_per_document=int(
                os.getenv(
                    "RETRIEVAL_MAX_CONTEXT_CHARS_PER_DOCUMENT",
                    str(RetrievalConfig.max_context_chars_per_document),
                )
            ),
            enable_parent_document_expansion=os.getenv(
                "RETRIEVAL_ENABLE_PARENT_DOCUMENT_EXPANSION",
                "true",
            ).lower()
            in {"1", "true", "yes"},
        ),
        generation=GenerationConfig(
            model=os.getenv("GENERATION_MODEL", GenerationConfig.model),
            api_base_url=os.getenv("GENERATION_API_BASE_URL", GenerationConfig.api_base_url),
            api_key_env_var=os.getenv(
                "GENERATION_API_KEY_ENV_VAR",
                GenerationConfig.api_key_env_var,
            ),
            max_tokens=int(os.getenv("GENERATION_MAX_TOKENS", str(GenerationConfig.max_tokens))),
            temperature=float(
                os.getenv("GENERATION_TEMPERATURE", str(GenerationConfig.temperature))
            ),
            min_context_score=float(
                os.getenv("GENERATION_MIN_CONTEXT_SCORE", str(GenerationConfig.min_context_score))
            ),
            min_groundedness=float(
                os.getenv("GENERATION_MIN_GROUNDEDNESS", str(GenerationConfig.min_groundedness))
            ),
            no_answer_text=os.getenv("GENERATION_NO_ANSWER_TEXT", GenerationConfig.no_answer_text),
        ),
        evaluation=EvaluationConfig(
            ragas_enabled=os.getenv("EVALUATION_RAGAS_ENABLED", "false").lower()
            in {"1", "true", "yes"},
            input_token_cost_per_1k=float(
                os.getenv(
                    "EVALUATION_INPUT_TOKEN_COST_PER_1K",
                    str(EvaluationConfig.input_token_cost_per_1k),
                )
            ),
            output_token_cost_per_1k=float(
                os.getenv(
                    "EVALUATION_OUTPUT_TOKEN_COST_PER_1K",
                    str(EvaluationConfig.output_token_cost_per_1k),
                )
            ),
        ),
        chunks_path=Path(os.getenv("INGEST_CHUNKS_PATH", "data/hotpotqa_chunks.jsonl")),
        manifest_path=Path(os.getenv("INGEST_MANIFEST_PATH", "results/ingestion_manifest.json")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )


def _parse_bm25_algorithm(value: str) -> Literal["BM25Okapi", "BM25L", "BM25Plus"]:
    if value in {"BM25Okapi", "BM25L", "BM25Plus"}:
        return value
    raise ValueError("BM25_ALGORITHM must be one of BM25Okapi, BM25L, or BM25Plus.")


def _parse_search_mode(value: str) -> Literal["hybrid", "dense", "bm25"]:
    if value in {"hybrid", "dense", "bm25"}:
        return value
    raise ValueError("RETRIEVAL_SEARCH_MODE must be one of hybrid, dense, or bm25.")


def _parse_fusion_algorithm(value: str) -> Literal["rrf", "weighted"]:
    if value in {"rrf", "weighted"}:
        return value
    raise ValueError("RETRIEVAL_FUSION_ALGORITHM must be one of rrf or weighted.")
