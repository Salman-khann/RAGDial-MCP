"""``ragdial-demo``: the full before/after loop on the bundled benchmark pack, in seconds.

    uv run ragdial-demo                      # offline, deterministic (hash embedder, BM25/RRF)
    uv run --extra ml ragdial-demo --embedder bge-small --reranker bge-reranker-base
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

from .lab import Lab

ROOT = Path(__file__).resolve().parents[2]


def _table(rows: list[list[str]], header: list[str]) -> str:
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    line = lambda r: "  ".join(str(x).ljust(w) for x, w in zip(r, widths))  # noqa: E731
    return "\n".join([line(header), line(["-" * w for w in widths]), *(line(r) for r in rows)])


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ragdial-demo")
    p.add_argument("--corpus", default=str(ROOT / "sample_data" / "corpus"))
    p.add_argument("--dataset", default=str(ROOT / "sample_data" / "golden_qa.jsonl"))
    p.add_argument("--embedder", default="hash")
    p.add_argument("--reranker", default="none")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--home", default=None, help="Persist runs here (default: a temp dir)")
    args = p.parse_args(argv)

    t0 = time.perf_counter()
    lab = Lab(args.home or tempfile.mkdtemp(prefix="ragdial-demo-"))
    ing = lab.ingest_corpus(args.corpus, "demo")
    print(f"[1] ingest: {ing.file_count} files, {ing.total_tokens:,} tokens, {ing.detected}\n")

    before = lab.run_chunking_ablation("demo", ["fixed"], [256], [0], args.embedder)
    after = lab.run_chunking_ablation("demo", ["markdown"], [256], [0], args.embedder)
    rows = [[c.config_id, c.chunks, c.tokens_p50, c.tokens_p95, c.damage_score, c.severed.get("table", "-")]
            for c in before.configs + after.configs]
    print("[2] chunking ablation")
    print(_table(rows, ["config", "chunks", "p50 tok", "p95 tok", "damage", "tables cut"]), "\n")

    insp = lab.inspect_boundary_splits(before.run_id, limit=2, kinds=["table"])
    print("[3] boundary splits in", insp.config_id)
    for s in insp.splits:
        print(f"  - {s.section_id} ({s.kind}) cut between {s.left_chunk} | {s.right_chunk}")
        print(f"      ...{s.before_cut[-70:]!r}  <CUT>  {s.after_cut[:70]!r}")
    print()

    kw = dict(retrievers=["bm25", "dense", "hybrid"], reranker=args.reranker, top_k=args.top_k,
              embedder=args.embedder)
    b1 = lab.benchmark_retrieval(before.run_id, args.dataset, **kw)
    b2 = lab.benchmark_retrieval(after.run_id, args.dataset, **kw)
    rows = [[v.variant_id, f"{v.recall_at_k:.3f}", f"{v.mrr:.3f}", f"{v.ndcg_at_k:.3f}", v.p95_ms]
            for v in sorted(b1.results + b2.results, key=lambda v: -v.ndcg_at_k)]
    print(f"[4] retrieval benchmark ({b1.queries} queries, embedder={b1.embedder}, store={b1.vector_store})")
    print(_table(rows, ["variant", f"R@{args.top_k}", "MRR", f"nDCG@{args.top_k}", "p95 ms"]), "\n")

    d = lab.diff_experiment_runs(b1.run_id, b2.run_id)
    print(f"[5] diff {d.variant_a}  ->  {d.variant_b}")
    for name, m in d.metrics.items():
        print(f"  {name:12s} {m.a:>8} -> {m.b:<8} ({m.delta:+})")
    print(f"  fixed: {[f.query_id for f in d.fixed]}")
    print(f"  regressed: {[f.query_id for f in d.regressed]}")
    print(f"  {d.verdict}")
    print(f"\nDone in {time.perf_counter() - t0:.1f}s. Runs stored in {lab.home}")


if __name__ == "__main__":
    main()
