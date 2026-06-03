"""Pipeline 2: retrieval over HotpotQA BM25 and Chroma indexes."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Mapping, Protocol, TypedDict, TypeGuard, cast

from haystack import Pipeline, component
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.rankers import SentenceTransformersSimilarityRanker
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import Secret
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

from src.config import AppConfig, QueryProcessingConfig, RetrievalConfig

MetaScalar = str | int | float | bool
JsonValue = MetaScalar | None | list["JsonValue"] | dict[str, "JsonValue"]
Route = Literal["hybrid", "dense", "bm25"]
FusionAlgorithm = Literal["rrf", "weighted"]


class ComparisonFilter(TypedDict):
    """Haystack comparison filter."""

    field: str
    operator: Literal["=="]
    value: MetaScalar


class LogicalFilter(TypedDict):
    """Haystack logical filter."""

    operator: Literal["AND"]
    conditions: list[ComparisonFilter]


MetadataFilter = ComparisonFilter | LogicalFilter


@dataclass(frozen=True)
class MetadataFilterCriteria:
    """Supported exact-match metadata filters."""

    source: str | None = None
    title: str | None = None
    level: str | None = None
    question_type: str | None = None
    permissions: str | None = None


@dataclass(frozen=True)
class ProcessedQuery:
    """Query processing output used by retrievers."""

    original_query: str
    rewritten_query: str
    expanded_queries: list[str]
    hyde_document: str | None
    route: Route

    @property
    def search_queries(self) -> list[str]:
        """Queries to use for sparse retrieval."""

        queries = [self.rewritten_query, *self.expanded_queries]
        deduped: list[str] = []
        seen: set[str] = set()
        for query in queries:
            normalized = query.strip()
            if normalized and normalized not in seen:
                deduped.append(normalized)
                seen.add(normalized)
        return deduped


@dataclass(frozen=True)
class RetrievalResult:
    """Final retrieval result and trace metadata."""

    query: ProcessedQuery
    documents: list[Document]
    filters: MetadataFilter | None
    fusion_algorithm: FusionAlgorithm
    timings: dict[str, float]


class TextGenerator(Protocol):
    """Text generation interface used by query processing."""

    def run(self, prompt: str) -> Mapping[str, list[str]]:
        """Generate text for a prompt."""
        ...


class QueryProcessor:
    """Rewrite, expand, HyDE, and route incoming queries."""

    def __init__(
        self,
        config: QueryProcessingConfig,
        retrieval_config: RetrievalConfig,
        generator: TextGenerator | None = None,
    ) -> None:
        self.config = config
        self.retrieval_config = retrieval_config
        self.generator = generator
        if self.generator is None and (
            config.enable_rewrite or config.enable_expand or config.enable_hyde
        ):
            self.generator = OpenAIGenerator(
                model=config.model,
                api_base_url=config.api_base_url,
                api_key=Secret.from_env_var(config.api_key_env_var),
                generation_kwargs={
                    "max_tokens": config.max_tokens,
                    "temperature": config.temperature,
                },
            )

    def run(self, query: str) -> ProcessedQuery:
        """Process a query into retrieval inputs."""

        rewritten_query = self._rewrite(query) if self.config.enable_rewrite else query
        expanded_queries = self._expand(rewritten_query) if self.config.enable_expand else []
        hyde_document = self._hyde(rewritten_query) if self.config.enable_hyde else None
        return ProcessedQuery(
            original_query=query,
            rewritten_query=rewritten_query,
            expanded_queries=expanded_queries,
            hyde_document=hyde_document,
            route=self.retrieval_config.search_mode,
        )

    def _rewrite(self, query: str) -> str:
        prompt = (
            "Rewrite this retrieval query to be precise and self-contained. "
            "Return only the rewritten query.\n\n"
            f"Query: {query}"
        )
        return self._first_reply(prompt) or query

    def _expand(self, query: str) -> list[str]:
        prompt = (
            "Generate up to three alternate retrieval queries with synonyms or entity aliases. "
            "Return one query per line and no commentary.\n\n"
            f"Query: {query}"
        )
        reply = self._first_reply(prompt)
        if not reply:
            return []
        return [line.strip("- ").strip() for line in reply.splitlines() if line.strip()]

    def _hyde(self, query: str) -> str | None:
        prompt = (
            "Write a short hypothetical document that would answer this query. "
            "Return only the hypothetical document.\n\n"
            f"Query: {query}"
        )
        return self._first_reply(prompt)

    def _first_reply(self, prompt: str) -> str | None:
        if self.generator is None:
            return None
        result = self.generator.run(prompt)
        replies = result.get("replies", [])
        if not replies:
            return None
        reply = replies[0].strip()
        return reply or None


@component
class MetadataFilterComponent:
    """Build metadata filters inside a Haystack Pipeline."""

    @component.output_types(filters=MetadataFilter | None)
    def run(self, criteria: MetadataFilterCriteria | None = None) -> dict[str, MetadataFilter | None]:
        """Return Haystack filters for exact-match metadata criteria."""

        return {"filters": build_metadata_filter(criteria or MetadataFilterCriteria())}


@component
class QueryProcessorComponent:
    """Haystack component wrapper for query processing."""

    def __init__(
        self,
        config: QueryProcessingConfig,
        retrieval_config: RetrievalConfig,
        generator: TextGenerator | None = None,
    ) -> None:
        self.processor = QueryProcessor(config, retrieval_config, generator)

    @component.output_types(
        processed_query=ProcessedQuery,
        query=str,
        query_timings=dict[str, float],
    )
    def run(self, query: str) -> dict[str, ProcessedQuery | str | dict[str, float]]:
        """Rewrite, expand, apply HyDE, and route a query."""

        start = time.perf_counter()
        processed_query = self.processor.run(query)
        return {
            "processed_query": processed_query,
            "query": query,
            "query_timings": {"query_processing_seconds": time.perf_counter() - start},
        }


@component
class BM25StoreLoaderComponent:
    """Load the persisted Haystack BM25 document store."""

    def __init__(self, store_path: Path) -> None:
        self.store_path = store_path

    @component.output_types(bm25_store=InMemoryDocumentStore)
    def run(self) -> dict[str, InMemoryDocumentStore]:
        """Load the BM25 store from disk."""

        return {"bm25_store": _load_bm25_store(self.store_path)}


@component
class BM25RetrievalComponent:
    """Run BM25 retrieval for processed queries."""

    def __init__(self, config: RetrievalConfig) -> None:
        self.config = config

    @component.output_types(bm25_documents=list[Document], bm25_timings=dict[str, float])
    def run(
        self,
        processed_query: ProcessedQuery,
        filters: MetadataFilter | None,
        bm25_store: InMemoryDocumentStore,
    ) -> dict[str, list[Document] | dict[str, float]]:
        """Retrieve sparse candidates when the route needs BM25."""

        start = time.perf_counter()
        documents: list[Document] = []
        if processed_query.route in {"hybrid", "bm25"}:
            documents = _run_bm25_retrieval(
                bm25_store,
                processed_query.search_queries,
                filters,
                self.config.bm25_top_k,
            )
        return {
            "bm25_documents": documents,
            "bm25_timings": {"bm25_retrieval_seconds": time.perf_counter() - start},
        }


@component
class DenseRetrievalComponent:
    """Run dense retrieval against Chroma."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    @component.output_types(dense_documents=list[Document], dense_timings=dict[str, float])
    def run(
        self,
        processed_query: ProcessedQuery,
        filters: MetadataFilter | None,
    ) -> dict[str, list[Document] | dict[str, float]]:
        """Retrieve dense candidates when the route needs vector search."""

        start = time.perf_counter()
        documents: list[Document] = []
        if processed_query.route in {"hybrid", "dense"}:
            documents = _run_dense_retrieval(self.config, processed_query, filters)
        return {
            "dense_documents": documents,
            "dense_timings": {"dense_retrieval_seconds": time.perf_counter() - start},
        }


@component
class HybridFusionComponent:
    """Fuse dense and BM25 rankings."""

    def __init__(self, config: RetrievalConfig) -> None:
        self.config = config

    @component.output_types(fused_documents=list[Document], fusion_timings=dict[str, float])
    def run(
        self,
        dense_documents: list[Document],
        bm25_documents: list[Document],
    ) -> dict[str, list[Document] | dict[str, float]]:
        """Fuse candidate rankings with RRF or weighted fusion."""

        start = time.perf_counter()
        documents = fuse_hybrid_results(
            dense_documents=dense_documents,
            bm25_documents=bm25_documents,
            config=self.config,
        )
        return {
            "fused_documents": documents,
            "fusion_timings": {"fusion_seconds": time.perf_counter() - start},
        }


@component
class RerankComponent:
    """Optionally rerank fused candidates."""

    def __init__(self, config: RetrievalConfig) -> None:
        self.config = config

    @component.output_types(ranked_documents=list[Document], rerank_timings=dict[str, float])
    def run(
        self,
        query: str,
        fused_documents: list[Document],
    ) -> dict[str, list[Document] | dict[str, float]]:
        """Apply the configured reranker."""

        start = time.perf_counter()
        documents = _rerank_documents(query, fused_documents, self.config)
        return {
            "ranked_documents": documents,
            "rerank_timings": {"rerank_seconds": time.perf_counter() - start},
        }


@component
class ContextCompressionComponent:
    """Deduplicate and trim retrieved context."""

    def __init__(self, config: RetrievalConfig) -> None:
        self.config = config

    @component.output_types(compressed_documents=list[Document], compression_timings=dict[str, float])
    def run(self, ranked_documents: list[Document]) -> dict[str, list[Document] | dict[str, float]]:
        """Compress and deduplicate candidate documents."""

        start = time.perf_counter()
        documents = compress_and_deduplicate_documents(ranked_documents, self.config)
        return {
            "compressed_documents": documents,
            "compression_timings": {"compression_seconds": time.perf_counter() - start},
        }


@component
class ParentExpansionComponent:
    """Expand chunk hits to parent-document context."""

    def __init__(self, config: RetrievalConfig) -> None:
        self.config = config

    @component.output_types(expanded_documents=list[Document], expansion_timings=dict[str, float])
    def run(
        self,
        compressed_documents: list[Document],
        bm25_store: InMemoryDocumentStore,
    ) -> dict[str, list[Document] | dict[str, float]]:
        """Apply small-to-big parent document expansion."""

        start = time.perf_counter()
        documents = expand_parent_documents(compressed_documents, bm25_store, self.config)
        return {
            "expanded_documents": documents,
            "expansion_timings": {"expansion_seconds": time.perf_counter() - start},
        }


@component
class RetrievalResultBuilderComponent:
    """Build the final retrieval result object."""

    def __init__(self, config: RetrievalConfig) -> None:
        self.config = config

    @component.output_types(result=RetrievalResult)
    def run(
        self,
        processed_query: ProcessedQuery,
        filters: MetadataFilter | None,
        expanded_documents: list[Document],
        query_timings: dict[str, float],
        bm25_timings: dict[str, float],
        dense_timings: dict[str, float],
        fusion_timings: dict[str, float],
        rerank_timings: dict[str, float],
        compression_timings: dict[str, float],
        expansion_timings: dict[str, float],
    ) -> dict[str, RetrievalResult]:
        """Collect final documents and timings."""

        timings = _merge_timings(
            query_timings,
            bm25_timings,
            dense_timings,
            fusion_timings,
            rerank_timings,
            compression_timings,
            expansion_timings,
        )
        timings["retrieval_seconds"] = (
            timings.get("bm25_retrieval_seconds", 0.0)
            + timings.get("dense_retrieval_seconds", 0.0)
        )
        timings["total_seconds"] = sum(
            value for key, value in timings.items() if key != "total_seconds"
        )
        return {
            "result": RetrievalResult(
                query=processed_query,
                documents=expanded_documents,
                filters=filters,
                fusion_algorithm=self.config.fusion_algorithm,
                timings=timings,
            )
        }


def build_retrieval_pipeline(config: AppConfig) -> Pipeline:
    """Build Pipeline 2 as a Haystack Pipeline graph."""

    pipeline = Pipeline()
    add_retrieval_pipeline_components(pipeline, config)
    return pipeline


def add_retrieval_pipeline_components(pipeline: Pipeline, config: AppConfig) -> None:
    """Add Pipeline 2 retrieval components to an existing Haystack Pipeline."""

    pipeline.add_component("query_processor", QueryProcessorComponent(config.query_processing, config.retrieval))
    pipeline.add_component("metadata_filter", MetadataFilterComponent())
    pipeline.add_component("bm25_store_loader", BM25StoreLoaderComponent(config.bm25.store_path))
    pipeline.add_component("bm25_retriever", BM25RetrievalComponent(config.retrieval))
    pipeline.add_component("dense_retriever", DenseRetrievalComponent(config))
    pipeline.add_component("hybrid_fusion", HybridFusionComponent(config.retrieval))
    pipeline.add_component("reranker", RerankComponent(config.retrieval))
    pipeline.add_component("context_compressor", ContextCompressionComponent(config.retrieval))
    pipeline.add_component("parent_expander", ParentExpansionComponent(config.retrieval))
    pipeline.add_component("result_builder", RetrievalResultBuilderComponent(config.retrieval))

    pipeline.connect("query_processor.processed_query", "bm25_retriever.processed_query")
    pipeline.connect("query_processor.processed_query", "dense_retriever.processed_query")
    pipeline.connect("query_processor.processed_query", "result_builder.processed_query")
    pipeline.connect("query_processor.query", "reranker.query")
    pipeline.connect("query_processor.query_timings", "result_builder.query_timings")

    pipeline.connect("metadata_filter.filters", "bm25_retriever.filters")
    pipeline.connect("metadata_filter.filters", "dense_retriever.filters")
    pipeline.connect("metadata_filter.filters", "result_builder.filters")

    pipeline.connect("bm25_store_loader.bm25_store", "bm25_retriever.bm25_store")
    pipeline.connect("bm25_store_loader.bm25_store", "parent_expander.bm25_store")

    pipeline.connect("bm25_retriever.bm25_documents", "hybrid_fusion.bm25_documents")
    pipeline.connect("bm25_retriever.bm25_timings", "result_builder.bm25_timings")
    pipeline.connect("dense_retriever.dense_documents", "hybrid_fusion.dense_documents")
    pipeline.connect("dense_retriever.dense_timings", "result_builder.dense_timings")

    pipeline.connect("hybrid_fusion.fused_documents", "reranker.fused_documents")
    pipeline.connect("hybrid_fusion.fusion_timings", "result_builder.fusion_timings")
    pipeline.connect("reranker.ranked_documents", "context_compressor.ranked_documents")
    pipeline.connect("reranker.rerank_timings", "result_builder.rerank_timings")
    pipeline.connect("context_compressor.compressed_documents", "parent_expander.compressed_documents")
    pipeline.connect("context_compressor.compression_timings", "result_builder.compression_timings")
    pipeline.connect("parent_expander.expanded_documents", "result_builder.expanded_documents")
    pipeline.connect("parent_expander.expansion_timings", "result_builder.expansion_timings")


def run_retrieval(
    config: AppConfig,
    query: str,
    *,
    metadata_filters: MetadataFilterCriteria | None = None,
) -> RetrievalResult:
    """Run Pipeline 2 through a Haystack Pipeline graph."""

    pipeline = build_retrieval_pipeline(config)
    output = cast(
        Mapping[str, Mapping[str, object]],
        pipeline.run(
            {
                "query_processor": {"query": query},
                "metadata_filter": {
                    "criteria": metadata_filters or MetadataFilterCriteria(),
                },
            }
        ),
    )
    result = output["result_builder"]["result"]
    if isinstance(result, RetrievalResult):
        return result
    raise TypeError("Retrieval pipeline did not return a RetrievalResult.")


def build_metadata_filter(criteria: MetadataFilterCriteria) -> MetadataFilter | None:
    """Build Haystack exact-match filters from supported metadata fields."""

    conditions: list[ComparisonFilter] = []
    _add_filter_condition(conditions, "meta.source", criteria.source)
    _add_filter_condition(conditions, "meta.title", criteria.title)
    _add_filter_condition(conditions, "meta.level", criteria.level)
    _add_filter_condition(conditions, "meta.type", criteria.question_type)
    _add_filter_condition(conditions, "meta.permissions", criteria.permissions)
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"operator": "AND", "conditions": conditions}


def fuse_hybrid_results(
    *,
    dense_documents: list[Document],
    bm25_documents: list[Document],
    config: RetrievalConfig,
) -> list[Document]:
    """Fuse dense and BM25 rankings with RRF or weighted score fusion."""

    if config.fusion_algorithm == "weighted":
        return _weighted_fusion(dense_documents, bm25_documents, config)
    return _rrf_fusion(dense_documents, bm25_documents, config)


def compress_and_deduplicate_documents(
    documents: list[Document],
    config: RetrievalConfig,
) -> list[Document]:
    """Deduplicate documents and trim content to save context window."""

    deduped: list[Document] = []
    seen_ids: set[str] = set()
    for document in documents:
        document_id = str(document.id)
        if document_id in seen_ids:
            continue
        seen_ids.add(document_id)
        content = document.content or ""
        if config.enable_context_compression:
            content = content[: config.max_context_chars_per_document]
        deduped.append(replace(document, content=content))
        if len(deduped) >= config.final_top_k:
            break
    return deduped


def expand_parent_documents(
    documents: list[Document],
    bm25_store: InMemoryDocumentStore,
    config: RetrievalConfig,
) -> list[Document]:
    """Expand small chunk hits to parent-document context."""

    if not config.enable_parent_document_expansion:
        return documents

    expanded: list[Document] = []
    expanded_parent_ids: set[str] = set()
    for document in documents:
        parent_doc_id = str(document.meta.get("parent_doc_id") or "")
        if not parent_doc_id or parent_doc_id in expanded_parent_ids:
            expanded.append(document)
            continue
        parent_chunks = _parent_chunks(bm25_store, parent_doc_id)
        if not parent_chunks:
            expanded.append(document)
            continue
        expanded_parent_ids.add(parent_doc_id)
        content = "\n".join(chunk.content or "" for chunk in parent_chunks)
        if config.enable_context_compression:
            content = content[: config.max_context_chars_per_document]
        meta = dict(document.meta)
        meta["expanded_from_chunk_id"] = str(document.id)
        meta["expanded_parent_doc_id"] = parent_doc_id
        meta["expanded_chunk_count"] = len(parent_chunks)
        expanded.append(
            replace(
                document,
                id=f"{parent_doc_id}:parent",
                content=content,
                meta=meta,
            )
        )
    return expanded


def retrieval_result_to_json(result: RetrievalResult) -> dict[str, JsonValue]:
    """Serialize retrieval output for CLI use."""

    return {
        "query": {
            "original": result.query.original_query,
            "rewritten": result.query.rewritten_query,
            "expanded": result.query.expanded_queries,
            "hyde_document": result.query.hyde_document,
            "route": result.query.route,
        },
        "fusion_algorithm": result.fusion_algorithm,
        "filters": _filter_to_json(result.filters),
        "documents": [_document_to_json(document) for document in result.documents],
        "timings": {key: value for key, value in result.timings.items()},
    }


def _run_bm25_retrieval(
    document_store: InMemoryDocumentStore,
    queries: list[str],
    filters: MetadataFilter | None,
    top_k: int,
) -> list[Document]:
    retriever = InMemoryBM25Retriever(document_store=document_store, top_k=top_k)
    documents: list[Document] = []
    for query in queries:
        result = retriever.run(query=query, filters=filters, top_k=top_k)
        outputs = cast(Mapping[str, list[Document]], result)
        documents.extend(_tag_documents(outputs["documents"], "bm25"))
    return documents


def _run_dense_retrieval(
    config: AppConfig,
    processed_query: ProcessedQuery,
    filters: MetadataFilter | None,
) -> list[Document]:
    query_text = processed_query.hyde_document or processed_query.rewritten_query
    embedder = OpenAITextEmbedder(
        model=config.embedding.model,
        dimensions=config.embedding.dimension,
        api_base_url=config.embedding.api_base_url,
        api_key=Secret.from_env_var(config.embedding.api_key_env_var),
    )
    embedding_result = cast(Mapping[str, object], embedder.run(text=query_text))
    query_embedding = _extract_embedding(embedding_result)
    document_store = ChromaDocumentStore(
        collection_name=config.chroma.collection_name,
        persist_path=str(config.chroma.persist_path),
        distance_function=config.chroma.distance_function,
    )
    retriever = ChromaEmbeddingRetriever(
        document_store=document_store,
        top_k=config.retrieval.dense_top_k,
    )
    result = retriever.run(
        query_embedding=query_embedding,
        filters=filters,
        top_k=config.retrieval.dense_top_k,
    )
    outputs = cast(Mapping[str, list[Document]], result)
    return _tag_documents(outputs["documents"], "dense")


def _rerank_documents(
    query: str,
    documents: list[Document],
    config: RetrievalConfig,
) -> list[Document]:
    if not config.enable_rerank or not documents:
        return documents[: config.final_top_k]
    ranker = SentenceTransformersSimilarityRanker(
        model=config.reranker_model,
        top_k=config.rerank_top_k,
    )
    result = ranker.run(query=query, documents=documents, top_k=config.rerank_top_k)
    outputs = cast(Mapping[str, list[Document]], result)
    return outputs["documents"][: config.final_top_k]


def _rrf_fusion(
    dense_documents: list[Document],
    bm25_documents: list[Document],
    config: RetrievalConfig,
) -> list[Document]:
    scores: dict[str, float] = {}
    documents_by_id: dict[str, Document] = {}
    for rank, document in enumerate(dense_documents, start=1):
        _add_rrf_score(scores, documents_by_id, document, rank, config.rrf_k)
    for rank, document in enumerate(bm25_documents, start=1):
        _add_rrf_score(scores, documents_by_id, document, rank, config.rrf_k)
    return _rank_fused_documents(scores, documents_by_id, "rrf")


def _weighted_fusion(
    dense_documents: list[Document],
    bm25_documents: list[Document],
    config: RetrievalConfig,
) -> list[Document]:
    scores: dict[str, float] = {}
    documents_by_id: dict[str, Document] = {}
    dense_max = _max_score(dense_documents)
    bm25_max = _max_score(bm25_documents)
    for document in dense_documents:
        document_id = str(document.id)
        documents_by_id.setdefault(document_id, document)
        scores[document_id] = scores.get(document_id, 0.0) + (
            config.dense_weight * (_score(document) / dense_max)
        )
    for document in bm25_documents:
        document_id = str(document.id)
        documents_by_id.setdefault(document_id, document)
        scores[document_id] = scores.get(document_id, 0.0) + (
            config.bm25_weight * (_score(document) / bm25_max)
        )
    return _rank_fused_documents(scores, documents_by_id, "weighted")


def _add_rrf_score(
    scores: dict[str, float],
    documents_by_id: dict[str, Document],
    document: Document,
    rank: int,
    rrf_k: int,
) -> None:
    document_id = str(document.id)
    documents_by_id.setdefault(document_id, document)
    scores[document_id] = scores.get(document_id, 0.0) + (1.0 / (rrf_k + rank))


def _rank_fused_documents(
    scores: dict[str, float],
    documents_by_id: dict[str, Document],
    algorithm: str,
) -> list[Document]:
    ranked_ids = sorted(scores, key=lambda document_id: scores[document_id], reverse=True)
    ranked_documents: list[Document] = []
    for document_id in ranked_ids:
        document = documents_by_id[document_id]
        meta = dict(document.meta)
        meta["fusion_algorithm"] = algorithm
        meta["fusion_score"] = scores[document_id]
        ranked_documents.append(replace(document, score=scores[document_id], meta=meta))
    return ranked_documents


def _tag_documents(documents: list[Document], source: Literal["dense", "bm25"]) -> list[Document]:
    tagged_documents: list[Document] = []
    for document in documents:
        meta = dict(document.meta)
        meta[f"{source}_score"] = _score(document)
        tagged_documents.append(replace(document, meta=meta))
    return tagged_documents


def _parent_chunks(document_store: InMemoryDocumentStore, parent_doc_id: str) -> list[Document]:
    filters: ComparisonFilter = {
        "field": "meta.parent_doc_id",
        "operator": "==",
        "value": parent_doc_id,
    }
    chunks = document_store.filter_documents(filters=filters)
    return sorted(chunks, key=_split_id)


def _load_bm25_store(path: Path) -> InMemoryDocumentStore:
    if not path.exists():
        raise FileNotFoundError(f"BM25 store not found at {path}. Run ingestion first.")
    return InMemoryDocumentStore.load_from_disk(str(path))


def _add_filter_condition(
    conditions: list[ComparisonFilter],
    field: str,
    value: str | None,
) -> None:
    if value:
        conditions.append({"field": field, "operator": "==", "value": value})


def _extract_embedding(result: Mapping[str, object]) -> list[float]:
    embedding = result.get("embedding")
    if _is_number_list(embedding):
        return [float(value) for value in embedding]
    raise ValueError("Text embedder did not return a numeric embedding.")


def _is_number_list(value: object) -> TypeGuard[list[int | float]]:
    return isinstance(value, list) and all(isinstance(item, (int, float)) for item in value)


def _score(document: Document) -> float:
    if document.score is None:
        return 0.0
    return float(document.score)


def _max_score(documents: list[Document]) -> float:
    scores = [_score(document) for document in documents]
    return max(scores) if scores and max(scores) > 0 else 1.0


def _merge_timings(*timing_groups: dict[str, float]) -> dict[str, float]:
    timings: dict[str, float] = {}
    for group in timing_groups:
        timings.update(group)
    return timings


def _split_id(document: Document) -> int:
    value = document.meta.get("split_id")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0


def _document_to_json(document: Document) -> dict[str, JsonValue]:
    return {
        "id": str(document.id),
        "score": document.score,
        "content": document.content,
        "meta": _meta_to_json(document.meta),
    }


def _meta_to_json(meta: Mapping[str, object]) -> dict[str, JsonValue]:
    return {str(key): _json_value(value) for key, value in meta.items()}


def _filter_to_json(filters: MetadataFilter | None) -> JsonValue:
    if filters is None:
        return None
    return _json_value(filters)


def _json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return str(value)
