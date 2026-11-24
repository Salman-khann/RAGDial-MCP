"""Compact, structured tool outputs.

Every MCP tool returns one of these models instead of raw chunk text, so an
agent pulls metric tables and a handful of failure IDs into its context, never
whole documents. Full detail stays behind the read-only ``ragdial://``
resources, fetched only when needed.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DocSummary(BaseModel):
    doc_id: str
    tokens: int
    sections: int
    tables: int
    code_blocks: int


class IngestResult(BaseModel):
    corpus_id: str
    folder: str
    file_count: int
    total_tokens: int
    tokenizer: str
    detected: dict[str, int] = Field(description="Structure counts: headers, tables, nested_tables, code_blocks, lists, paragraphs")
    documents: list[DocSummary] = Field(description="Per-document summary (first 50)")
    warnings: list[str] = Field(default_factory=list)
    next_step: str


class ChunkConfigResult(BaseModel):
    config_id: str
    strategy: str
    size: int
    overlap: int
    chunks: int
    tokens_p50: float
    tokens_p95: float
    tokens_max: int
    damage_score: float = Field(description="0 = no structure severed, 1 = every structure severed (weighted)")
    severed: dict[str, str] = Field(description="severed/total per structure kind")
    mitigated_tables: int = Field(description="Severed tables whose header row was replicated into every piece")


class ChunkingAblationResult(BaseModel):
    run_id: str
    corpus_id: str
    embedder: str | None = Field(description="Embedder used by the semantic chunker, if any")
    configs: list[ChunkConfigResult] = Field(description="Sorted by damage_score ascending")
    least_damage: str
    skipped: list[str] = Field(default_factory=list)
    next_step: str


class BoundarySplit(BaseModel):
    doc_id: str
    section_id: str
    kind: str
    mitigated: bool
    oversized: bool
    left_chunk: str | None
    right_chunk: str | None
    before_cut: str
    after_cut: str


class BoundaryInspection(BaseModel):
    run_id: str
    config_id: str
    damage_score: float
    severed_total: int
    severed_by_kind: dict[str, int]
    splits: list[BoundarySplit]
    hint: str


class VariantMetrics(BaseModel):
    variant_id: str
    chunk_config: str
    retriever: str
    reranker: str
    chunks: int
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    hit_rate: float
    p50_ms: float
    p95_ms: float


class QueryFailure(BaseModel):
    query_id: str
    question: str
    first_rank: int | None
    recall_at_k: float


class BenchmarkResult(BaseModel):
    run_id: str
    chunking_run_id: str
    dataset: str
    queries: int
    top_k: int
    embedder: str
    vector_store: str
    reranker: str
    results: list[VariantMetrics] = Field(description="Sorted by nDCG@k descending")
    best_variant: str
    worst_queries: list[QueryFailure] = Field(description="Bottom-3 queries for the best variant")
    warnings: list[str] = Field(default_factory=list)
    next_step: str


class MetricDelta(BaseModel):
    a: float
    b: float
    delta: float


class QueryFlip(BaseModel):
    query_id: str
    rank_a: int | None
    rank_b: int | None


class RunDiff(BaseModel):
    run_a: str
    variant_a: str
    run_b: str
    variant_b: str
    top_k: int
    metrics: dict[str, MetricDelta]
    fixed: list[QueryFlip] = Field(description="Queries that went from miss to hit within top_k")
    regressed: list[QueryFlip] = Field(description="Queries that went from hit to miss within top_k")
    rank_improved: int
    rank_worsened: int
    unchanged: int
    verdict: str
