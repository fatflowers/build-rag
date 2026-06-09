"""Haystack ingestion pipeline for HotpotQA records."""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Protocol, TypeAlias, cast

from haystack import AsyncPipeline, component
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DocumentStore, DuplicatePolicy
from haystack.utils import Secret
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

from src.config import AppConfig, ContextualRetrievalConfig
from src.hotpotqa_loader import HotpotQAStats, load_hotpotqa_documents
from src.langfuse_tracing import add_langfuse_connector

logger = logging.getLogger(__name__)


MetaScalar: TypeAlias = str | int | float | bool
JsonValue: TypeAlias = MetaScalar | None | list["JsonValue"] | dict[str, "JsonValue"]
GenerationKwargs: TypeAlias = Mapping[str, int | float]


class ContextGenerator(Protocol):
    """Text generator interface used for contextual retrieval."""

    def run(
        self,
        prompt: str,
        *,
        generation_kwargs: GenerationKwargs | None = None,
    ) -> Mapping[str, list[str]]:
        """Generate text for a prompt."""
        ...


@component
class ContextualRetrievalAnnotator:
    """Generate Anthropic-style chunk-specific retrieval context."""

    def __init__(
        self,
        config: ContextualRetrievalConfig,
        generator: ContextGenerator | None = None,
    ) -> None:
        self.config = config
        self.generator = generator or OpenAIGenerator(
            model=config.model,
            api_base_url=config.api_base_url,
            api_key=Secret.from_env_var(config.api_key_env_var),
        )

    @component.output_types(documents=list[Document])
    def run(
        self,
        documents: list[Document],
        source_documents: list[Document],
    ) -> dict[str, list[Document]]:
        """Prepend generated context to each chunk before indexing."""

        sources_by_parent_id: dict[str, Document] = {}
        for source_document in source_documents:
            parent_doc_id = source_document.meta.get("parent_doc_id") or source_document.id
            sources_by_parent_id[str(parent_doc_id)] = source_document

        chunks_per_parent: dict[str, int] = {}
        for document in documents:
            parent_doc_id = str(document.meta["parent_doc_id"])
            chunks_per_parent[parent_doc_id] = chunks_per_parent.get(parent_doc_id, 0) + 1

        contextualized_documents: list[Document] = []
        for document in documents:
            parent_doc_id = str(document.meta["parent_doc_id"])
            chunk_text = document.content or ""
            # A document that produced a single chunk was never split (e.g. the
            # source text was short enough to fit in one chunk). The chunk already
            # contains the full document, so situating it adds nothing — skip the
            # LLM call entirely.
            if chunks_per_parent[parent_doc_id] <= 1:
                meta = dict(document.meta)
                meta["contextual_retrieval_context"] = ""
                contextualized_documents.append(replace(document, meta=meta))
                continue
            source_document = sources_by_parent_id.get(parent_doc_id)
            if source_document is None:
                document_text = ""
            else:
                title = str(source_document.meta.get("title") or "")
                content = source_document.content or ""
                document_text = f"Title: {title}\n{content}" if title else content
            prompt = (
                "<document>\n"
                f"{document_text}\n"
                "</document>\n"
                "Here is the chunk we want to situate within the whole document\n"
                "<chunk>\n"
                f"{chunk_text}\n"
                "</chunk>\n"
                "Please give a short succinct context to situate this chunk within the overall "
                "document for the purposes of improving search retrieval of the chunk. Answer "
                "only with the succinct context and nothing else."
            )
            result = self.generator.run(
                prompt,
                generation_kwargs={
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                },
            )
            replies = result.get("replies", [])
            contextual_text = replies[0].strip() if replies else ""
            content = f"{contextual_text}\n\n{chunk_text}" if contextual_text else chunk_text
            meta = dict(document.meta)
            meta["contextual_retrieval_context"] = contextual_text
            contextualized_documents.append(replace(document, content=content, meta=meta))

        return {"documents": contextualized_documents}


@component
class ChunkMetadataNormalizer:
    """Normalize chunk metadata after Haystack splitting."""

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """Add stable chunk ids and labels to Haystack-split documents."""

        normalized_chunks: list[Document] = []
        for chunk_index, chunk in enumerate(documents):
            parent_doc_id = str(chunk.meta["parent_doc_id"])
            supporting_texts = cast(
                list[str],
                json.loads(str(chunk.meta.get("supporting_sentence_texts_json") or "[]")),
            )
            is_supporting_chunk = any(
                text and text in (chunk.content or "") for text in supporting_texts
            )
            meta = dict(chunk.meta)
            meta.pop("supporting_sentence_texts_json", None)
            meta.update(
                {
                    "chunk_id": f"{parent_doc_id}:chunk:{meta.get('split_id', chunk_index)}",
                    "is_supporting_chunk": is_supporting_chunk,
                }
            )
            content = chunk.content or ""
            normalized_chunks.append(
                replace(
                    chunk,
                    id=str(meta["chunk_id"]),
                    content=content,
                    meta=_sanitize_meta(meta),
                )
            )

        return {"documents": normalized_chunks}


@component
class BM25DocumentStoreWriter:
    """Write normalized chunks into a persisted Haystack BM25 store."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    @component.output_types(bm25_count=int)
    def run(self, documents: list[Document]) -> dict[str, int]:
        """Persist chunks to the BM25 document store."""

        logger.info("Writing BM25 document store to %s", self.config.bm25.store_path)
        self.config.bm25.store_path.parent.mkdir(parents=True, exist_ok=True)
        document_store = InMemoryDocumentStore(
            bm25_algorithm=self.config.bm25.algorithm,
            bm25_tokenization_regex=self.config.bm25.tokenization_regex,
            return_embedding=False,
        )
        document_store.write_documents(documents, policy=DuplicatePolicy.OVERWRITE)
        document_store.save_to_disk(str(self.config.bm25.store_path))
        return {"bm25_count": document_store.count_documents()}


@component
class EmbeddingIntegrityValidator:
    """Fail ingestion when dense embedding did not produce usable vectors."""

    def __init__(self, dimension: int | None) -> None:
        self.dimension = dimension

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """Validate that every document has an embedding with the expected dimension."""

        missing_ids: list[str] = []
        wrong_dimension_ids: list[str] = []
        for document in documents:
            embedding = document.embedding
            if embedding is None:
                missing_ids.append(str(document.id))
            elif self.dimension is not None and len(embedding) != self.dimension:
                wrong_dimension_ids.append(str(document.id))

        if missing_ids or wrong_dimension_ids:
            details: list[str] = []
            if missing_ids:
                details.append(f"missing embeddings for {len(missing_ids)} documents")
            if wrong_dimension_ids:
                details.append(
                    f"wrong embedding dimension for {len(wrong_dimension_ids)} documents"
                )
            sample_ids = [*missing_ids, *wrong_dimension_ids][:5]
            raise RuntimeError(
                "Dense embedding failed before Chroma indexing: "
                + ", ".join(details)
                + f". Sample document ids: {sample_ids}"
            )

        return {"documents": documents}


@dataclass(frozen=True)
class IngestionPipelineRun:
    """Outputs produced by the Haystack ingestion pipeline."""

    chunks: list[Document]
    bm25_count: int | None


def build_chroma_ingestion_pipeline(
    config: AppConfig,
    document_store: DocumentStore | None = None,
    *,
    skip_chroma: bool = False,
    skip_bm25: bool = False,
) -> AsyncPipeline:
    """Build the Haystack ingestion pipeline, optionally skipping Chroma or BM25 indexing."""

    pipeline = AsyncPipeline()
    add_langfuse_connector(pipeline, config, "ingestion")
    pipeline.add_component(
        "splitter",
        DocumentSplitter(
            split_by=config.chunking.split_by,
            split_length=config.chunking.split_length,
            split_overlap=config.chunking.split_overlap,
            split_threshold=config.chunking.split_threshold,
        ),
    )
    pipeline.add_component(
        "chunk_normalizer",
        ChunkMetadataNormalizer(),
    )
    if config.chunking.contextual_retrieval:
        pipeline.add_component(
            "contextualizer",
            ContextualRetrievalAnnotator(config.contextual_retrieval),
        )
        pipeline.connect("splitter.documents", "contextualizer.documents")
        pipeline.connect("contextualizer.documents", "chunk_normalizer.documents")
    else:
        pipeline.connect("splitter.documents", "chunk_normalizer.documents")

    if not skip_bm25:
        pipeline.add_component(
            "bm25_writer",
            BM25DocumentStoreWriter(config),
        )
        pipeline.connect("chunk_normalizer.documents", "bm25_writer.documents")

    if not skip_chroma:
        if document_store is None:
            raise ValueError("document_store is required when skip_chroma is false.")

        embedder = OpenAIDocumentEmbedder(
            model=config.embedding.model,
            dimensions=config.embedding.dimension,
            api_base_url=config.embedding.api_base_url,
            api_key=Secret.from_env_var(config.embedding.api_key_env_var),
            batch_size=config.embedding.batch_size,
        )
        pipeline.add_component("embedder", embedder)
        pipeline.add_component(
            "writer",
            DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE),
        )
        pipeline.add_component(
            "embedding_validator",
            EmbeddingIntegrityValidator(config.embedding.dimension),
        )
        pipeline.connect("chunk_normalizer.documents", "embedder.documents")
        pipeline.connect("embedder.documents", "embedding_validator.documents")
        pipeline.connect("embedding_validator.documents", "writer.documents")
    return pipeline


def split_documents(documents: list[Document], config: AppConfig) -> list[Document]:
    """Split documents by running the Haystack ingestion pipeline without Chroma."""

    return _run_haystack_ingestion_pipeline(
        documents,
        config,
        skip_chroma=True,
        skip_bm25=True,
    ).chunks


def _run_haystack_ingestion_pipeline(
    documents: list[Document],
    config: AppConfig,
    *,
    skip_chroma: bool,
    skip_bm25: bool,
    document_store: DocumentStore | None = None,
) -> IngestionPipelineRun:
    pipeline = build_chroma_ingestion_pipeline(
        config,
        document_store=document_store,
        skip_chroma=skip_chroma,
        skip_bm25=skip_bm25,
    )
    result = pipeline.run(
        _ingestion_pipeline_inputs(documents, config),
        include_outputs_from=_ingestion_pipeline_outputs(skip_bm25=skip_bm25),
    )
    return _parse_ingestion_pipeline_result(result, skip_bm25=skip_bm25)


async def _run_haystack_ingestion_pipeline_async(
    documents: list[Document],
    config: AppConfig,
    *,
    skip_chroma: bool,
    skip_bm25: bool,
    document_store: DocumentStore | None = None,
    concurrency_limit: int = 4,
) -> IngestionPipelineRun:
    pipeline = build_chroma_ingestion_pipeline(
        config,
        document_store=document_store,
        skip_chroma=skip_chroma,
        skip_bm25=skip_bm25,
    )
    result = await pipeline.run_async(
        _ingestion_pipeline_inputs(documents, config),
        include_outputs_from=_ingestion_pipeline_outputs(skip_bm25=skip_bm25),
        concurrency_limit=concurrency_limit,
    )
    return _parse_ingestion_pipeline_result(result, skip_bm25=skip_bm25)


def _ingestion_pipeline_inputs(
    documents: list[Document],
    config: AppConfig,
) -> dict[str, dict[str, list[Document]]]:
    pipeline_inputs: dict[str, dict[str, list[Document]]] = {"splitter": {"documents": documents}}
    if config.chunking.contextual_retrieval:
        pipeline_inputs["contextualizer"] = {"source_documents": documents}
    return pipeline_inputs


def _ingestion_pipeline_outputs(*, skip_bm25: bool) -> set[str]:
    include_outputs = {"chunk_normalizer"}
    if not skip_bm25:
        include_outputs.add("bm25_writer")
    return include_outputs


def _parse_ingestion_pipeline_result(
    result: Mapping[str, Mapping[str, object]],
    *,
    skip_bm25: bool,
) -> IngestionPipelineRun:
    outputs = cast(Mapping[str, Mapping[str, object]], result)
    chunks = cast(list[Document], outputs["chunk_normalizer"]["documents"])
    bm25_count: int | None = None
    if not skip_bm25:
        value = outputs["bm25_writer"]["bm25_count"]
        if not isinstance(value, int):
            raise TypeError("Haystack ingestion pipeline returned a non-integer BM25 count.")
        bm25_count = value
    return IngestionPipelineRun(chunks=chunks, bm25_count=bm25_count)


def run_ingestion(
    config: AppConfig,
    *,
    skip_chroma: bool = False,
    skip_bm25: bool = False,
    rebuild: bool = False,
) -> dict[str, JsonValue]:
    """Run HotpotQA parsing, chunking, BM25 indexing, Chroma indexing, and manifest writing."""

    return _run_ingestion_with_pipeline_run(
        config,
        skip_chroma=skip_chroma,
        skip_bm25=skip_bm25,
        rebuild=rebuild,
    )


async def run_ingestion_async(
    config: AppConfig,
    *,
    skip_chroma: bool = False,
    skip_bm25: bool = False,
    rebuild: bool = False,
    concurrency_limit: int = 4,
) -> dict[str, JsonValue]:
    """Run ingestion with Haystack AsyncPipeline.run_async."""

    return await _run_ingestion_with_pipeline_run_async(
        config,
        skip_chroma=skip_chroma,
        skip_bm25=skip_bm25,
        rebuild=rebuild,
        concurrency_limit=concurrency_limit,
    )


def _prepare_ingestion_inputs(
    config: AppConfig,
    *,
    skip_chroma: bool,
    rebuild: bool,
) -> tuple[list[Document], HotpotQAStats, ChromaDocumentStore | None, float]:
    start = time.perf_counter()
    logger.info("Loading HotpotQA documents")
    source_documents, stats = load_hotpotqa_documents(
        dataset_name=config.data.dataset_name,
        dataset_config=config.data.dataset_config,
        split=config.data.split,
        limit=config.data.limit,
    )

    document_store = None
    if not skip_chroma:
        if rebuild and config.chroma.persist_path.exists():
            shutil.rmtree(config.chroma.persist_path)
        config.chroma.persist_path.mkdir(parents=True, exist_ok=True)
        document_store = ChromaDocumentStore(
            collection_name=config.chroma.collection_name,
            persist_path=str(config.chroma.persist_path),
            distance_function=config.chroma.distance_function,
        )

    return source_documents, stats, document_store, start


def _run_ingestion_with_pipeline_run(
    config: AppConfig,
    *,
    skip_chroma: bool,
    skip_bm25: bool,
    rebuild: bool,
) -> dict[str, JsonValue]:
    source_documents, stats, document_store, start = _prepare_ingestion_inputs(
        config,
        skip_chroma=skip_chroma,
        rebuild=rebuild,
    )
    logger.info("Running Haystack ingestion pipeline for %d documents", len(source_documents))
    pipeline_run = _run_haystack_ingestion_pipeline(
        source_documents,
        config,
        skip_chroma=skip_chroma,
        skip_bm25=skip_bm25,
        document_store=document_store,
    )
    return _finish_ingestion_run(
        config=config,
        stats=stats,
        chunks=pipeline_run.chunks,
        bm25_count=pipeline_run.bm25_count,
        document_store=document_store,
        start=start,
        skip_chroma=skip_chroma,
        skip_bm25=skip_bm25,
    )


async def _run_ingestion_with_pipeline_run_async(
    config: AppConfig,
    *,
    skip_chroma: bool,
    skip_bm25: bool,
    rebuild: bool,
    concurrency_limit: int,
) -> dict[str, JsonValue]:
    source_documents, stats, document_store, start = _prepare_ingestion_inputs(
        config,
        skip_chroma=skip_chroma,
        rebuild=rebuild,
    )
    logger.info("Running Haystack async ingestion pipeline for %d documents", len(source_documents))
    pipeline_run = await _run_haystack_ingestion_pipeline_async(
        source_documents,
        config,
        skip_chroma=skip_chroma,
        skip_bm25=skip_bm25,
        document_store=document_store,
        concurrency_limit=concurrency_limit,
    )
    return _finish_ingestion_run(
        config=config,
        stats=stats,
        chunks=pipeline_run.chunks,
        bm25_count=pipeline_run.bm25_count,
        document_store=document_store,
        start=start,
        skip_chroma=skip_chroma,
        skip_bm25=skip_bm25,
    )


def _finish_ingestion_run(
    *,
    config: AppConfig,
    stats: HotpotQAStats,
    chunks: list[Document],
    bm25_count: int | None,
    document_store: ChromaDocumentStore | None,
    start: float,
    skip_chroma: bool,
    skip_bm25: bool,
) -> dict[str, JsonValue]:
    chroma_count: int | None = None
    if document_store is not None:
        chroma_count = document_store.count_documents()

    logger.info("Writing chunk JSONL to %s", config.chunks_path)
    _write_chunk_jsonl(chunks, config.chunks_path)

    manifest = _build_manifest(
        config=config,
        stats=stats,
        chunks=chunks,
        chroma_count=chroma_count,
        elapsed_seconds=time.perf_counter() - start,
        skip_chroma=skip_chroma,
        skip_bm25=skip_bm25,
        bm25_count=bm25_count,
    )
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info("Wrote ingestion manifest to %s", config.manifest_path)
    return manifest


def _write_chunk_jsonl(chunks: list[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(
                json.dumps(
                    {"id": chunk.id, "content": chunk.content, "meta": chunk.meta},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def _build_manifest(
    *,
    config: AppConfig,
    stats: HotpotQAStats,
    chunks: list[Document],
    chroma_count: int | None,
    bm25_count: int | None,
    elapsed_seconds: float,
    skip_chroma: bool,
    skip_bm25: bool,
) -> dict[str, JsonValue]:
    supporting_chunks = sum(1 for chunk in chunks if chunk.meta.get("is_supporting_chunk"))
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "dataset_name": config.data.dataset_name,
            "dataset_config": config.data.dataset_config,
            "split": config.data.split,
            "limit": config.data.limit,
        },
        "chunking": {
            "split_by": config.chunking.split_by,
            "split_length": config.chunking.split_length,
            "split_overlap": config.chunking.split_overlap,
            "split_threshold": config.chunking.split_threshold,
            "contextual_retrieval": config.chunking.contextual_retrieval,
        },
        "contextual_retrieval": {
            "enabled": config.chunking.contextual_retrieval,
            "model": config.contextual_retrieval.model,
            "api_base_url": config.contextual_retrieval.api_base_url,
            "api_key_env_var": config.contextual_retrieval.api_key_env_var,
            "max_tokens": config.contextual_retrieval.max_tokens,
            "temperature": config.contextual_retrieval.temperature,
        },
        "embedding": {
            "provider": config.embedding.provider,
            "model": config.embedding.model,
            "dimension": config.embedding.dimension,
            "api_base_url": config.embedding.api_base_url,
            "api_key_env_var": config.embedding.api_key_env_var,
            "batch_size": config.embedding.batch_size,
            "language_support": config.embedding.language_support,
            "asymmetric": config.embedding.asymmetric,
        },
        "chroma": {
            "persist_path": str(config.chroma.persist_path),
            "collection_name": config.chroma.collection_name,
            "distance_function": config.chroma.distance_function,
            "enabled": not skip_chroma,
            "document_count": chroma_count,
        },
        "bm25": {
            "store_path": str(config.bm25.store_path),
            "algorithm": config.bm25.algorithm,
            "tokenization_regex": config.bm25.tokenization_regex,
            "enabled": not skip_bm25,
            "document_count": bm25_count,
        },
        "outputs": {
            "chunks_path": str(config.chunks_path),
            "bm25_store_path": str(config.bm25.store_path),
            "manifest_path": str(config.manifest_path),
        },
        "counts": {
            "records": stats.records,
            "source_documents": stats.source_documents,
            "supporting_documents": stats.supporting_documents,
            "supporting_facts": stats.supporting_facts,
            "matched_supporting_facts": stats.matched_supporting_facts,
            "chunks": len(chunks),
            "supporting_chunks": supporting_chunks,
        },
        "timings": {
            "total_seconds": elapsed_seconds,
        },
    }


def _sanitize_meta(meta: Mapping[str, object]) -> dict[str, MetaScalar]:
    sanitized_meta: dict[str, MetaScalar] = {}
    for key, value in meta.items():
        if isinstance(value, (str, int, float, bool)):
            sanitized_meta[key] = value
        elif value is None:
            sanitized_meta[key] = ""
        else:
            sanitized_meta[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return sanitized_meta
