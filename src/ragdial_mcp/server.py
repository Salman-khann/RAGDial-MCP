"""MCP server: tools, resources and prompts over the :class:`~ragdial_mcp.lab.Lab` engine.

Run over stdio (IDE clients)::

    ragdial-mcp

or Streamable HTTP (containers / remote)::

    ragdial-mcp --transport streamable-http --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import json
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceNotFoundError, ToolError
from mcp_types import ToolAnnotations
from pydantic import Field

from . import __version__
from .lab import Lab, LabError
from .models import (
    BenchmarkResult,
    BoundaryInspection,
    ChunkingAblationResult,
    IngestResult,
    RunDiff,
)

INSTRUCTIONS = """\
RAGDial MCP, a RAG retrieval ablation & chunking lab. Tune retrieval with controlled experiments, not guesses.

Typical loop:
1. ingest_corpus(folder_path) to parse docs and count tables, code fences and headers ("@sample" = bundled benchmark pack)
2. run_chunking_ablation(corpus_id, strategies, chunk_sizes, overlaps) for chunk stats and boundary-damage scores
3. inspect_boundary_splits(run_id) to see exactly which tables/code/lists were severed
4. benchmark_retrieval(run_id, dataset_path, retrievers, reranker, top_k) for Recall@k, MRR, nDCG@k, p95 latency
5. diff_experiment_runs(run_a, run_b) for metric deltas and the query IDs that flipped

Tool outputs are compact summaries by design. Full per-query tables live in
ragdial://experiments/{run_id}/report; read them only when you need them.
"""


def build_server(lab: Lab | None = None) -> MCPServer:
    state: dict[str, Lab] = {}

    def get_lab() -> Lab:
        if "lab" not in state:
            state["lab"] = lab or Lab()
        return state["lab"]

    def tool_call(method: str, *args):
        """Run a Lab method; failures reach the model as readable tool errors, never a bare crash."""
        try:
            return getattr(get_lab(), method)(*args)
        except LabError as exc:
            raise ToolError(str(exc)) from exc
        except (ValueError, TypeError, KeyError, OSError, ArithmeticError) as exc:
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc

    def resource_call(method: str, *args) -> str:
        try:
            # compact JSON: resources exist to save context, so don't spend it on indentation
            return json.dumps(getattr(get_lab(), method)(*args), separators=(",", ":"))
        except LabError as exc:
            raise ResourceNotFoundError(str(exc)) from exc

    mcp = MCPServer(name="ragdial", title="RAGDial MCP", version=__version__, instructions=INSTRUCTIONS)
    readonly = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
    writes = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)
    replaces = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False)

    # ------------------------------------------------------------------ tools
    @mcp.tool(annotations=replaces)
    def ingest_corpus(
        folder_path: Annotated[str, Field(description="Folder of .md/.txt/.pdf files (searched recursively), or '@sample' for the bundled 13-doc benchmark pack")],
        corpus_id: Annotated[str | None, Field(description="ID to store the corpus under; defaults to the folder name. Re-ingesting an ID replaces it and its runs.")] = None,
    ) -> IngestResult:
        """Parse a document folder: file count, total tokens, detected tables/headers/code, document IDs."""
        return tool_call("ingest_corpus", folder_path, corpus_id)

    @mcp.tool(annotations=writes)
    def run_chunking_ablation(
        corpus_id: str,
        strategies: Annotated[list[str], Field(description="Any of: fixed, recursive, markdown (header-aware), semantic")],
        chunk_sizes: Annotated[list[int], Field(description="Target chunk sizes in tokens, e.g. [128, 256, 512]")],
        overlaps: Annotated[list[int], Field(description="Overlaps in tokens, e.g. [0, 32]")] = [0],  # noqa: B006
        embedder: Annotated[str, Field(description="Embedder for the semantic chunker: auto, hash, bge-small, or a sentence-transformers id")] = "auto",
    ) -> ChunkingAblationResult:
        """Chunk the corpus under every strategy x size x overlap combination. Returns chunk counts,
        token-length p50/p95 and a boundary-damage score per config (no chunk text)."""
        return tool_call("run_chunking_ablation", corpus_id, strategies, chunk_sizes, overlaps, embedder)

    @mcp.tool(annotations=readonly)
    def inspect_boundary_splits(
        run_id: Annotated[str, Field(description="A chunking run, or a benchmark run (inspects its best config)")],
        limit: Annotated[int, Field(ge=1, le=25)] = 5,
        config_id: Annotated[str | None, Field(description="Config to inspect, e.g. 'fixed-256-o0'. Defaults to the most damaged config.")] = None,
        kinds: Annotated[list[str] | None, Field(description="Filter: table, code, list, paragraph")] = None,
    ) -> BoundaryInspection:
        """Short excerpts on both sides of the cut wherever a table, code block, list or paragraph
        was severed mid-structure."""
        return tool_call("inspect_boundary_splits", run_id, limit, config_id, kinds)

    @mcp.tool(annotations=writes)
    def benchmark_retrieval(
        run_id: Annotated[str, Field(description="Chunking run to evaluate")],
        dataset_path: Annotated[str, Field(description="Golden Q&A JSONL (absolute, relative to cwd, or next to the corpus), or '@sample' for the bundled 30 questions")],
        retrievers: Annotated[list[str], Field(description="Any of: bm25, dense, hybrid (BM25 + dense fused with RRF)")] = ["bm25", "dense", "hybrid"],  # noqa: B006
        reranker: Annotated[str, Field(description="'none' or a cross-encoder such as 'bge-reranker-base'")] = "none",
        top_k: Annotated[int, Field(ge=1, le=100)] = 5,
        configs: Annotated[list[str] | None, Field(description="Subset of chunk config IDs to evaluate; default all")] = None,
        embedder: Annotated[str, Field(description="auto, hash, bge-small, or a sentence-transformers id")] = "auto",
        vector_store: Annotated[str, Field(description="auto, chroma or numpy")] = "auto",
    ) -> BenchmarkResult:
        """Run the golden queries against every chunk config x retriever. Returns Recall@k, MRR,
        nDCG@k and p50/p95 latency per variant plus the 3 worst queries of the best variant."""
        return tool_call("benchmark_retrieval", run_id, dataset_path, retrievers, reranker, top_k, configs,
                                             embedder, vector_store)

    @mcp.tool(annotations=readonly)
    def diff_experiment_runs(
        run_a: Annotated[str, Field(description="Baseline benchmark run")],
        run_b: Annotated[str, Field(description="Candidate benchmark run (may equal run_a to compare its worst vs best variant)")],
        variant_a: Annotated[str | None, Field(description="Variant ID in run_a; defaults to its best")] = None,
        variant_b: Annotated[str | None, Field(description="Variant ID in run_b; defaults to its best")] = None,
    ) -> RunDiff:
        """Metric deltas (B - A) plus the query IDs that flipped from miss to hit, or regressed."""
        return tool_call("diff_experiment_runs", run_a, run_b, variant_a, variant_b)

    # -------------------------------------------------------------- resources
    @mcp.resource("ragdial://experiments", name="experiments", mime_type="application/json",
                  description="Index of corpora and saved ablation/benchmark runs with headline metrics.")
    def experiments() -> str:
        return resource_call("experiments_index")

    @mcp.resource("ragdial://experiments/{run_id}/report", name="experiment_report", mime_type="application/json",
                  description="Full JSON metrics for a run, including the per-query hit/miss table.")
    def experiment_report(run_id: str) -> str:
        return resource_call("run_report", run_id)

    @mcp.resource("ragdial://corpus/{corpus_id}/stats", name="corpus_stats", mime_type="application/json",
                  description="Document metadata, token counts and structural breakdown for a corpus.")
    def corpus_stats(corpus_id: str) -> str:
        return resource_call("corpus_stats", corpus_id)

    # ---------------------------------------------------------------- prompts
    @mcp.prompt(title="Diagnose retrieval failures")
    def diagnose_retrieval_failures(run_id: str) -> str:
        """Find the worst queries in a benchmark run, trace them to chunk boundary damage, propose a fix."""
        return f"""Diagnose why retrieval fails in benchmark run `{run_id}`. Keep your context small: use the
compact tool outputs and read the full report only for the queries you investigate.

1. Read `ragdial://experiments/{run_id}/report`. For the best variant, list the 5 queries with the lowest
   nDCG (misses first). Note each query's first_rank and top_chunks.
2. Call `inspect_boundary_splits(run_id="{run_id}", limit=8)`. Match severed structures (especially
   tables and code) to the failing queries by doc_id / section_id.
3. Classify each failure as one of:
   - boundary damage (evidence split across chunks)
   - ranking problem (evidence chunk exists but ranks below top_k, so try hybrid or a reranker)
   - vocabulary mismatch (BM25 misses but dense hits, or the reverse)
   - chunk too large (evidence diluted in a big chunk)
4. Propose ONE targeted change (chunking strategy/size/overlap, retriever, or reranker), run it with
   `run_chunking_ablation` and/or `benchmark_retrieval`, then confirm with
   `diff_experiment_runs(run_a="{run_id}", run_b=<new run>)`.
5. Report: the root cause per failing query, the change, the metric deltas, and the query IDs that
   flipped or regressed. Do not paste document text beyond short excerpts."""

    @mcp.prompt(title="Generate production RAG spec")
    def generate_rag_spec(corpus_id: str, latency_budget_ms: str = "200") -> str:
        """Recommend a production retrieval configuration for a corpus within a latency ceiling."""
        return f"""Produce a production retrieval spec for corpus `{corpus_id}` with a p95 retrieval latency
budget of {latency_budget_ms} ms.

1. Read `ragdial://corpus/{corpus_id}/stats` (table/code density shapes the chunking choice) and
   `ragdial://experiments`. Use benchmark runs for this corpus. If none exist, run
   `run_chunking_ablation` (fixed, recursive, markdown, semantic at 256/512 tokens) and
   `benchmark_retrieval` with bm25, dense and hybrid first.
2. Drop every variant whose p95_ms exceeds {latency_budget_ms}. Among the rest, start from the best by
   nDCG@k, but treat every variant in that run's `tied_with_best` (and any whose ndcg_ci95 overlaps the
   best's) as equally good: choose among the tied ones by lower latency, then simpler setup (BM25 before
   hybrid, no reranker before reranker). State the confidence intervals; never call a winner on a
   difference inside them.
3. If any run used the 'hash' embedder (see its warnings), do NOT recommend dense or hybrid retrieval
   from it; recommend rerunning with embedder='bge-small' first.
4. If a reranker run exists, state whether its quality gain (with CI) justifies its latency cost.
5. Output a spec as a YAML block with: chunking (strategy, size, overlap), retriever (type, fusion,
   top_k, candidate depth), embedder, reranker, vector store, and expected metrics (Recall@k, MRR,
   nDCG@k with 95% CI, p95 ms) with the run_id that supports each number.
6. Note that the numbers only hold for questions like the golden set's, list the open failure modes
   (query IDs), and give one experiment to run next."""

    return mcp


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ragdial-mcp", description="RAGDial MCP: RAG ablation & chunking lab server")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--home", default=None, help="Data directory (default: $RAGDIAL_HOME or ~/.ragdial)")
    args = parser.parse_args(argv)
    server = build_server(Lab(args.home) if args.home else None)
    if args.transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
