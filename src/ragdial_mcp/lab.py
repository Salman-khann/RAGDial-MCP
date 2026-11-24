"""Evaluation engine. Pure Python, no MCP imports.

``server.py`` is a thin adapter over this class; a CLI, notebook or web
dashboard can drive exactly the same experiments.
"""

from __future__ import annotations

import itertools
import json
import os
import sqlite3
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .chunking.boundary_audit import KINDS, AuditResult, audit_boundaries, excerpt, trim_span
from .chunking.strategies import Chunk, ChunkConfig, chunk_corpus, measured_overlap, normalize_strategy
from .ingestion.parser import Document, load_folder, slugify
from .models import (
    BenchmarkResult,
    BoundaryInspection,
    BoundarySplit,
    ChunkConfigResult,
    ChunkingAblationResult,
    DocSummary,
    IngestResult,
    MetricDelta,
    QueryFailure,
    QueryFlip,
    RunDiff,
    VariantMetrics,
)
from .retrieval.embedders import get_embedder
from .retrieval.hybrid import RETRIEVERS, ChunkSearcher, get_reranker
from .samples import SAMPLE_TOKEN, sample_corpus, sample_golden
from .retrieval.metrics import groups_covered, normalize, percentile, score_ranking
from .retrieval.stats import mean_ci, paired_diff_ci, sign_test_p
from .storage.db import ExperimentDB
from .tokens import TokenIndex, get_tokenizer

MAX_CONFIGS = 48
REPORT_SEVERED_CAP = 25


class LabError(ValueError):
    """User-facing error with an actionable message."""


def default_home() -> Path:
    return Path(os.environ.get("RAGDIAL_HOME", Path.home() / ".ragdial")).expanduser()


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _pct(values: list[int] | list[float], p: float) -> float:
    return round(percentile(values, p), 1)


class GoldenQuery:
    __slots__ = ("query_id", "question", "groups", "tags")

    def __init__(self, raw: Any, line: int) -> None:
        if not isinstance(raw, dict):
            raise LabError(f"golden line {line}: expected a JSON object, got {type(raw).__name__}")
        self.query_id = str(raw.get("id") or raw.get("query_id") or f"q{line:03d}")
        where = f"golden line {line} ({self.query_id})"
        self.question = raw.get("question") or raw.get("query")
        if not isinstance(self.question, str) or not self.question.strip():
            raise LabError(f"{where}: missing 'question'")
        groups = raw.get("evidence_groups")
        if groups is None:
            ev = raw.get("evidence")
            if ev is None:
                raise LabError(f"{where}: needs 'evidence' or 'evidence_groups'")
            groups = [[ev]] if isinstance(ev, str) else [ev]
        # strict shape: a list of groups, each a non-empty list of non-empty strings. A list of
        # bare strings would otherwise be iterated character by character and match anything.
        if not isinstance(groups, list) or not groups or not all(
                isinstance(g, list) and g and all(isinstance(s, str) and s.strip() for s in g) for g in groups):
            raise LabError(f"{where}: 'evidence' must be a string or list of strings, and 'evidence_groups' "
                           "a list of lists of strings, e.g. [[\"header row\", \"data row\"], [\"other fact\"]]")
        self.groups = [[normalize(s) for s in g] for g in groups]
        self.tags = raw.get("tags", [])


class Lab:
    def __init__(self, home: str | Path | None = None) -> None:
        self.home = Path(home) if home else default_home()
        self.db = ExperimentDB(self.home / "ragdial.db")
        self._docs: dict[str, list[Document]] = {}
        self._token_indexes: dict[str, dict[str, TokenIndex]] = {}

    # ================================================================ ingestion
    def ingest_corpus(self, folder_path: str, corpus_id: str | None = None) -> IngestResult:
        if folder_path.strip() == SAMPLE_TOKEN:
            folder = sample_corpus()
            corpus_id = corpus_id or "sample"
        else:
            folder = Path(folder_path).expanduser().resolve()
        try:
            docs, warnings = load_folder(folder)
        except (FileNotFoundError, ValueError) as exc:
            raise LabError(str(exc)) from exc
        corpus_id = slugify(corpus_id or folder.name)
        detected = Counter()
        for d in docs:
            for b in d.blocks:
                detected[{"header": "headers", "table": "tables", "code": "code_blocks",
                          "list": "lists", "paragraph": "paragraphs"}[b.kind]] += 1
            detected["nested_tables"] += len(d.nested_tables)
        detected_d = {k: detected.get(k, 0) for k in
                      ("headers", "tables", "nested_tables", "code_blocks", "lists", "paragraphs")}
        summaries = [DocSummary(doc_id=d.doc_id, tokens=d.tokens, sections=len(d.sections),
                                tables=sum(b.kind == "table" for b in d.blocks) + len(d.nested_tables),
                                code_blocks=sum(b.kind == "code" for b in d.blocks)) for d in docs]
        stats = {"file_count": len(docs), "total_tokens": sum(d.tokens for d in docs),
                 "tokenizer": get_tokenizer().name, "detected": detected_d,
                 "documents": [s.model_dump() | {"title": d.title, "path": d.path, "format": d.source_format}
                               for s, d in zip(summaries, docs)]}
        self.db.save_corpus(corpus_id, str(folder), stats, [d.model_dump() for d in docs])
        self._docs[corpus_id] = docs
        self._token_indexes.pop(corpus_id, None)
        return IngestResult(
            corpus_id=corpus_id, folder=str(folder), file_count=len(docs), total_tokens=stats["total_tokens"],
            tokenizer=stats["tokenizer"], detected=detected_d, documents=summaries[:50], warnings=warnings,
            next_step=f"run_chunking_ablation(corpus_id='{corpus_id}', strategies=[...], chunk_sizes=[...], overlaps=[...])",
        )

    def _corpus(self, corpus_id: str) -> tuple[list[Document], dict[str, Any]]:
        row = self.db.get_corpus(corpus_id)
        if row is None:
            known = ", ".join(c["corpus_id"] for c in self.db.list_corpora()) or "none"
            raise LabError(f"Unknown corpus_id '{corpus_id}'. Known corpora: {known}. Run ingest_corpus first.")
        if corpus_id not in self._docs:
            self._docs[corpus_id] = [Document.model_validate(d) for d in row["docs"]]
        return self._docs[corpus_id], row

    def _tis(self, corpus_id: str) -> dict[str, TokenIndex]:
        return self._token_indexes.setdefault(corpus_id, {})

    def _run(self, run_id: str, kind: str | None = None) -> dict[str, Any]:
        run = self.db.get_run(run_id)
        if run is None:
            raise LabError(f"Unknown run_id '{run_id}'. Read ragdial://experiments for saved runs.")
        if kind and run["kind"] != kind:
            raise LabError(f"Run '{run_id}' is a {run['kind']} run; this needs a {kind} run.")
        return run

    # ================================================================ chunking
    def run_chunking_ablation(self, corpus_id: str, strategies: list[str], chunk_sizes: list[int],
                              overlaps: list[int] | None = None, embedder: str = "auto") -> ChunkingAblationResult:
        docs, _ = self._corpus(corpus_id)
        overlaps = overlaps or [0]
        try:
            strategies = list(dict.fromkeys(normalize_strategy(s) for s in strategies))
        except ValueError as exc:
            raise LabError(str(exc)) from exc
        if not strategies or not chunk_sizes:
            raise LabError("Provide at least one strategy and one chunk size.")
        configs: list[ChunkConfig] = []
        skipped: list[str] = []
        for strat, size, ov in itertools.product(strategies, sorted(set(chunk_sizes)), sorted(set(overlaps))):
            if size < 16 or size > 8192:
                skipped.append(f"{strat}-{size}-o{ov}: size must be 16..8192")
            elif ov < 0 or ov >= size:
                skipped.append(f"{strat}-{size}-o{ov}: overlap must be >= 0 and < size")
            else:
                configs.append(ChunkConfig(strategy=strat, size=size, overlap=ov))
        if not configs:
            raise LabError("No valid configurations: " + "; ".join(skipped))
        if len(configs) > MAX_CONFIGS:
            raise LabError(f"{len(configs)} configurations requested; the limit is {MAX_CONFIGS}. Narrow the grid.")

        try:
            emb = get_embedder(embedder) if "semantic" in strategies else None
        except ValueError as exc:
            raise LabError(str(exc)) from exc
        tis = self._tis(corpus_id)
        block_tokens: dict[tuple[str, int, int], int] = {}
        for d in docs:
            ti = tis.get(d.doc_id) or tis.setdefault(d.doc_id, TokenIndex(d.text))
            for b in d.structures:
                s, e = trim_span(d.text, b.start, b.end)
                block_tokens[(d.doc_id, s, e)] = ti.count(s, e)

        run_id = _new_id("chk")
        results: list[ChunkConfigResult] = []
        chunk_sets: list[tuple[str, list[dict[str, Any]], dict[str, Any]]] = []
        for cfg in configs:  # compute everything first; nothing is stored unless the whole run succeeds
            chunks = chunk_corpus(docs, cfg, emb, tis)
            audit = audit_boundaries(docs, chunks, cfg.size, block_tokens)
            toks = [c.tokens for c in chunks]
            results.append(ChunkConfigResult(
                config_id=cfg.config_id, strategy=cfg.strategy, size=cfg.size, overlap=cfg.overlap,
                chunks=len(chunks), tokens_p50=_pct(toks, 50), tokens_p95=_pct(toks, 95),
                tokens_max=max(toks) if toks else 0, measured_overlap=measured_overlap(chunks, tis),
                damage_score=audit.damage_score,
                severed={k: f"{v.severed}/{v.total}" for k, v in audit.by_kind.items()},
                mitigated_tables=audit.mitigated, lost_structures=audit.lost,
            ))
            chunk_sets.append((cfg.config_id, [c.model_dump() for c in chunks], audit.model_dump()))
        results.sort(key=lambda r: (r.damage_score, r.config_id))
        no_effect = sorted({r.config_id for r in results if r.overlap > 0 and r.measured_overlap == 0})
        notes = ([f"Overlap had no effect for {', '.join(no_effect)}: these chunkers only carry whole "
                  "sentences/blocks within a section, and none fit."] if no_effect else [])
        summary = {"configs": [r.model_dump() for r in results], "least_damage": results[0].config_id,
                   "skipped": skipped, "notes": notes}
        config = {"strategies": strategies, "chunk_sizes": sorted(set(chunk_sizes)),
                  "overlaps": sorted(set(overlaps)), "embedder": emb.name if emb else None,
                  "tokenizer": get_tokenizer().name}
        try:
            self.db.save_chunking_run(run_id, corpus_id, config, summary, chunk_sets)
        except sqlite3.IntegrityError as exc:
            raise LabError(f"Corpus '{corpus_id}' changed while chunking (re-ingested?). Run it again.") from exc
        return ChunkingAblationResult(
            run_id=run_id, corpus_id=corpus_id, embedder=emb.name if emb else None, configs=results,
            least_damage=results[0].config_id, skipped=skipped, notes=notes,
            next_step=(f"inspect_boundary_splits(run_id='{run_id}') to see what got cut, or "
                       f"benchmark_retrieval(run_id='{run_id}', dataset_path=...) to measure retrieval."),
        )

    def inspect_boundary_splits(self, run_id: str, limit: int = 5, config_id: str | None = None,
                                kinds: list[str] | None = None) -> BoundaryInspection:
        run = self._run(run_id)
        chunk_run_id = run_id if run["kind"] == "chunking" else run["parent_run_id"]
        if chunk_run_id is None:
            raise LabError(f"Run '{run_id}' has no chunking run attached.")
        chunk_run = self._run(chunk_run_id, "chunking")
        configs = chunk_run["summary"]["configs"]
        if config_id is None:
            if run["kind"] == "benchmark":
                best = run["summary"]["best_variant"]
                config_id = best.split("|")[0]
            else:  # the most damaged config is the most instructive
                config_id = max(configs, key=lambda c: (c["damage_score"], c["config_id"]))["config_id"]
        audit_raw = self.db.get_audit(chunk_run_id, config_id)
        if audit_raw is None:
            ids = ", ".join(c["config_id"] for c in configs)
            raise LabError(f"Config '{config_id}' not in run {chunk_run_id}. Available: {ids}")
        audit = AuditResult.model_validate(audit_raw)
        docs, _ = self._corpus(chunk_run["corpus_id"])
        by_id = {d.doc_id: d for d in docs}
        limit = max(1, min(limit, 25))
        bad_kinds = [k for k in kinds or [] if k not in KINDS]
        if bad_kinds:
            raise LabError(f"Unknown kind(s) {bad_kinds}. Choose from {list(KINDS)}")
        items = [s for s in audit.severed_items if not kinds or s.kind in kinds]
        # lost content first, then unmitigated damage, one example per document first, for variety
        items.sort(key=lambda s: (not s.lost, s.mitigated, {"table": 0, "code": 1, "list": 2}.get(s.kind, 3)))
        picked, seen_docs = [], set()
        for s in items:
            if s.doc_id not in seen_docs:
                picked.append(s)
                seen_docs.add(s.doc_id)
            if len(picked) >= limit:
                break
        for s in items:
            if len(picked) >= limit:
                break
            if s not in picked:
                picked.append(s)
        splits = []
        for s in picked:
            before, after = excerpt(by_id[s.doc_id], s)
            splits.append(BoundarySplit(doc_id=s.doc_id, section_id=s.section_id, kind=s.kind,
                                        mitigated=s.mitigated, oversized=s.oversized, lost=s.lost,
                                        left_chunk=s.left_chunk,
                                        right_chunk=s.right_chunk, before_cut=before, after_cut=after))
        by_kind = Counter(s.kind for s in audit.severed_items)
        if not audit.severed_items:
            hint = "No structures were severed."
        else:
            ranked = sorted(configs, key=lambda c: (c["damage_score"], c["config_id"]))
            others = [c["config_id"] for c in ranked if c["config_id"] != config_id][:2]
            hint = (("Some content is in no chunk at all (lost=true): that is a chunker bug, please report it. "
                     if audit.lost else "")
                    + "Severed tables/code lose their header or context. Lower damage does not guarantee "
                    "better retrieval: confirm with benchmark_retrieval"
                    + (f", e.g. against the least-damaged configs in this run: {', '.join(others)}." if others
                       else "."))
        return BoundaryInspection(run_id=chunk_run_id, config_id=config_id, damage_score=audit.damage_score,
                                  severed_total=len(audit.severed_items), severed_by_kind=dict(by_kind),
                                  splits=splits, hint=hint)

    # ================================================================ retrieval
    def _resolve_dataset(self, dataset_path: str, corpus_folder: str) -> Path:
        if dataset_path.strip() == SAMPLE_TOKEN:
            return sample_golden()
        p = Path(dataset_path).expanduser()
        # exact path only: never substitute a different file that merely shares the name
        candidates = [p] if p.is_absolute() else [Path.cwd() / p, *(
            base / p for base in [Path(corpus_folder), *Path(corpus_folder).parents][:3])]
        for c in candidates:
            if c.is_file():
                return c.resolve()
        raise LabError(f"Dataset not found: {dataset_path} (looked in cwd and next to the corpus folder)")

    def _load_golden(self, path: Path) -> list[GoldenQuery]:
        queries: list[GoldenQuery] = []
        seen: dict[str, int] = {}
        for i, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if line.strip() and not line.lstrip().startswith("//"):
                try:
                    q = GoldenQuery(json.loads(line), i)
                except json.JSONDecodeError as exc:
                    raise LabError(f"{path.name} line {i}: invalid JSON ({exc.msg})") from exc
                if q.query_id in seen:
                    raise LabError(f"{path.name} line {i}: duplicate query id '{q.query_id}' "
                                   f"(first used on line {seen[q.query_id]})")
                seen[q.query_id] = i
                queries.append(q)
        if not queries:
            raise LabError(f"{path} has no queries")
        return queries

    def benchmark_retrieval(self, run_id: str, dataset_path: str, retrievers: list[str] | None = None,
                            reranker: str = "none", top_k: int = 5, configs: list[str] | None = None,
                            embedder: str = "auto", vector_store: str = "auto",
                            rerank_depth: int = 20) -> BenchmarkResult:
        chunk_run = self._run(run_id, "chunking")
        docs, corpus = self._corpus(chunk_run["corpus_id"])
        retrievers = list(dict.fromkeys(r.lower() for r in (retrievers or ["bm25", "dense", "hybrid"])))
        bad = [r for r in retrievers if r not in RETRIEVERS]
        if bad:
            raise LabError(f"Unknown retriever(s) {bad}. Choose from {list(RETRIEVERS)}")
        if not 1 <= top_k <= 100:
            raise LabError("top_k must be between 1 and 100")
        all_cfgs = [c["config_id"] for c in chunk_run["summary"]["configs"]]
        cfg_ids = list(dict.fromkeys(configs)) if configs else all_cfgs
        missing = [c for c in cfg_ids if c not in all_cfgs]
        if missing:
            raise LabError(f"Config(s) {missing} not in run {run_id}. Available: {all_cfgs}")
        try:
            rr = get_reranker(reranker)
            emb = get_embedder(embedder) if any(r != "bm25" for r in retrievers) else None
        except (ValueError, OSError) as exc:
            raise LabError(f"Could not load model: {exc}") from exc
        reranker_name = rr.name if rr else "none"
        dataset = self._resolve_dataset(dataset_path, corpus["folder"])
        queries = self._load_golden(dataset)

        warnings: list[str] = []
        corpus_norm = "\n".join(normalize(d.text) for d in docs)
        orphan = [q.query_id for q in queries if not any(all(s in corpus_norm for s in g) for g in q.groups)]
        if orphan:
            warnings.append(f"{len(orphan)} queries have evidence not found verbatim in the corpus "
                            f"(always a miss): {orphan[:10]}")
        if emb is not None and emb.name.startswith("hash"):
            warnings.append("Dense/hybrid used the 'hash' embedder, a lexical stand-in for a real model. Do not "
                            "choose dense or hybrid retrieval from these numbers; rerun with embedder='bge-small' "
                            "(needs the ml extra).")

        bench_id = _new_id("bench")
        rows: list[tuple] = []
        variants: list[VariantMetrics] = []
        per_variant_ndcg: dict[str, list[float]] = {}
        store_used = "n/a"
        for cfg_id in cfg_ids:
            stored = self.db.get_chunk_set(run_id, cfg_id)
            if stored is None:
                raise LabError(f"Chunk set {cfg_id} of run {run_id} no longer exists (corpus re-ingested?). "
                               "Run the chunking ablation again.")
            chunks = [Chunk.model_validate(c) for c in stored[0]]
            if not chunks:
                raise LabError(f"Config {cfg_id} produced no chunks; nothing to search.")
            texts = [c.text for c in chunks]
            norm = [normalize(t) for t in texts]
            searcher = ChunkSearcher(texts, emb or get_embedder("hash"), vector_store)
            try:
                try:
                    searcher.warm(retrievers)
                except ValueError as exc:
                    raise LabError(str(exc)) from exc
                if emb is not None:
                    store_used = searcher.vector_store
                for retriever in retrievers:
                    variant = f"{cfg_id}|{retriever}|{reranker_name}"
                    searcher.search(queries[0].question, retriever, top_k, rr, rerank_depth)  # warm-up
                    lat, rec, mrr, ndcg = [], [], [], []
                    for q in queries:
                        ranked, ms = searcher.search(q.question, retriever, top_k, rr, rerank_depth)
                        covered = [groups_covered(norm[i], q.groups) for i in ranked]
                        m = score_ranking(covered, len(q.groups), top_k)
                        lat.append(ms)
                        rec.append(m["recall"])
                        mrr.append(m["rr"])
                        ndcg.append(m["ndcg"])
                        rows.append((variant, q.query_id, m["recall"], m["rr"], m["ndcg"], m["first_rank"],
                                     round(ms, 3), [chunks[i].chunk_id for i in ranked[:top_k]]))
                    per_variant_ndcg[variant] = ndcg
                    lo, hi = mean_ci(ndcg)
                    variants.append(VariantMetrics(
                        variant_id=variant, chunk_config=cfg_id, retriever=retriever, reranker=reranker_name,
                        chunks=len(chunks), recall_at_k=round(float(np.mean(rec)), 4),
                        mrr=round(float(np.mean(mrr)), 4), ndcg_at_k=round(float(np.mean(ndcg)), 4),
                        ndcg_ci95=(round(lo, 3), round(hi, 3)),
                        hit_rate=round(float(np.mean([r > 0 for r in rec])), 4),
                        p50_ms=round(percentile(lat, 50), 2), p95_ms=round(percentile(lat, 95), 2),
                    ))
            finally:
                searcher.close()

        variants.sort(key=lambda v: (-v.ndcg_at_k, -v.recall_at_k, v.p95_ms, v.variant_id))
        best = variants[0]
        tied = [v.variant_id for v in variants[1:]
                if paired_diff_ci(per_variant_ndcg[v.variant_id], per_variant_ndcg[best.variant_id])[0] <= 0]
        qtext = {q.query_id: q.question for q in queries}
        best_rows = sorted((r for r in rows if r[0] == best.variant_id), key=lambda r: (r[4], r[2], r[1]))
        worst = [QueryFailure(query_id=r[1], question=qtext[r[1]][:100], first_rank=r[5], recall_at_k=r[2])
                 for r in best_rows[:3] if r[4] < 1.0]
        config = {"dataset": str(dataset), "retrievers": retrievers, "reranker": reranker_name, "top_k": top_k,
                  "configs": cfg_ids, "embedder": emb.name if emb else None, "vector_store": store_used,
                  "rerank_depth": rerank_depth, "queries": len(queries)}
        summary = {"variants": [v.model_dump() for v in variants], "best_variant": best.variant_id,
                   "tied_with_best": tied, "questions": qtext, "warnings": warnings}
        try:
            self.db.save_benchmark_run(bench_id, chunk_run["corpus_id"], run_id, config, summary, rows)
        except sqlite3.IntegrityError as exc:
            raise LabError(f"Chunking run {run_id} was deleted while benchmarking (corpus re-ingested?). "
                           "Run it again.") from exc
        return BenchmarkResult(
            run_id=bench_id, chunking_run_id=run_id, dataset=dataset.name, queries=len(queries), top_k=top_k,
            embedder=emb.name if emb else "n/a (bm25 only)", vector_store=store_used, reranker=reranker_name,
            results=variants, best_variant=best.variant_id, tied_with_best=tied, worst_queries=worst,
            warnings=warnings,
            next_step=(f"diff_experiment_runs(run_a='{bench_id}', run_b=...) after a change, or read "
                       f"ragdial://experiments/{bench_id}/report for the per-query table."),
        )

    # ================================================================ diffing
    def diff_experiment_runs(self, run_a: str, run_b: str, variant_a: str | None = None,
                             variant_b: str | None = None) -> RunDiff:
        ra, rb = self._run(run_a, "benchmark"), self._run(run_b, "benchmark")
        va = variant_a or ra["summary"]["best_variant"]
        vb = variant_b or rb["summary"]["best_variant"]
        if run_a == run_b and variant_a is None and variant_b is None:
            ordered = [v["variant_id"] for v in ra["summary"]["variants"]]
            if len(ordered) < 2:
                raise LabError("Same run with a single variant: nothing to diff.")
            va, vb = ordered[-1], ordered[0]  # worst vs best within one run
        rows_a = {r["query_id"]: r for r in self.db.get_query_results(run_a, va)}
        rows_b = {r["query_id"]: r for r in self.db.get_query_results(run_b, vb)}
        for run, v, rows in ((ra, va, rows_a), (rb, vb, rows_b)):
            if not rows:
                avail = [x["variant_id"] for x in run["summary"]["variants"]]
                raise LabError(f"Variant '{v}' not in run {run['run_id']}. Available: {avail}")
        shared = sorted(set(rows_a) & set(rows_b))
        if not shared:
            raise LabError("The two runs share no query IDs (different datasets?).")
        k_a, k_b = ra["config"]["top_k"], rb["config"]["top_k"]
        if k_a != k_b:
            raise LabError(f"Runs use different top_k ({k_a} vs {k_b}); their Recall@k and nDCG@k are not "
                           "comparable. Rerun one of them with the same top_k.")
        k = k_a

        def hit(r: dict[str, Any]) -> bool:
            return r["first_rank"] is not None and r["first_rank"] <= k

        def mean(rows: dict[str, dict[str, Any]], key: str) -> float:
            return float(np.mean([rows[q][key] for q in shared]))

        metrics: dict[str, MetricDelta] = {}
        for name, key in (("recall_at_k", "recall"), ("mrr", "rr"), ("ndcg_at_k", "ndcg")):
            a, b = mean(rows_a, key), mean(rows_b, key)
            metrics[name] = MetricDelta(a=round(a, 4), b=round(b, 4), delta=round(b - a, 4))
        lat_a = [rows_a[q]["latency_ms"] for q in shared]
        lat_b = [rows_b[q]["latency_ms"] for q in shared]
        pa, pb = percentile(lat_a, 95), percentile(lat_b, 95)
        metrics["p95_ms"] = MetricDelta(a=round(pa, 3), b=round(pb, 3), delta=round(pb - pa, 3))

        fixed, regressed = [], []
        improved = worsened = unchanged = 0
        for q in shared:
            a, b = rows_a[q], rows_b[q]
            if not hit(a) and hit(b):
                fixed.append(QueryFlip(query_id=q, rank_a=a["first_rank"], rank_b=b["first_rank"]))
            elif hit(a) and not hit(b):
                regressed.append(QueryFlip(query_id=q, rank_a=a["first_rank"], rank_b=b["first_rank"]))
            ra_, rb_ = a["first_rank"] or 10_000, b["first_rank"] or 10_000
            if rb_ < ra_:
                improved += 1
            elif rb_ > ra_:
                worsened += 1
            else:
                unchanged += 1
        d = metrics["ndcg_at_k"].delta
        lo, hi = paired_diff_ci([rows_a[q]["ndcg"] for q in shared], [rows_b[q]["ndcg"] for q in shared])
        p_sign = sign_test_p(len(fixed), len(regressed))
        significant = lo > 0 or hi < 0
        verdict = (f"B {'improves' if d > 0 else 'regresses' if d < 0 else 'matches'} nDCG@{k} by {d:+.3f} "
                   f"(95% CI {lo:+.3f} to {hi:+.3f}; "
                   f"{'a real difference' if significant else 'not distinguishable from noise'} on "
                   f"{len(shared)} questions). {len(fixed)} fixed, {len(regressed)} regressed "
                   f"(sign test p={p_sign:.3f}); p95 latency {metrics['p95_ms'].delta:+.2f} ms.")
        return RunDiff(run_a=run_a, variant_a=va, run_b=run_b, variant_b=vb, top_k=k, metrics=metrics,
                       ndcg_delta_ci95=(round(lo, 4), round(hi, 4)), sign_test_p=round(p_sign, 4),
                       significant=significant, fixed=fixed, regressed=regressed, rank_improved=improved,
                       rank_worsened=worsened, unchanged=unchanged, verdict=verdict)

    # ================================================================ resources
    def experiments_index(self) -> dict[str, Any]:
        runs = []
        for r in self.db.list_runs():
            entry = {"run_id": r["run_id"], "kind": r["kind"], "corpus_id": r["corpus_id"],
                     "parent_run_id": r["parent_run_id"], "created_at": r["created_at"]}
            if r["kind"] == "chunking":
                entry["configs"] = [c["config_id"] for c in r["summary"].get("configs", [])]
                entry["least_damage"] = r["summary"].get("least_damage")
            else:
                best = r["summary"]["variants"][0]
                entry.update(retrievers=r["config"]["retrievers"], reranker=r["config"]["reranker"],
                             top_k=r["config"]["top_k"], best_variant=best["variant_id"],
                             best_ndcg=best["ndcg_at_k"], best_recall=best["recall_at_k"], best_p95_ms=best["p95_ms"])
            runs.append(entry)
        return {"corpora": self.db.list_corpora(), "runs": runs}

    def run_report(self, run_id: str) -> dict[str, Any]:
        run = self._run(run_id)
        report: dict[str, Any] = {k: run[k] for k in ("run_id", "kind", "corpus_id", "parent_run_id", "created_at")}
        report["config"] = run["config"]
        if run["kind"] == "chunking":
            report["configs"] = []
            for c in run["summary"].get("configs", []):
                audit = self.db.get_audit(run_id, c["config_id"]) or {}
                severed = audit.get("severed_items", [])
                report["configs"].append(c | {"severed_items": severed[:REPORT_SEVERED_CAP],
                                              "severed_items_truncated": max(0, len(severed) - REPORT_SEVERED_CAP)})
            return report
        report["variants"] = run["summary"]["variants"]
        report["best_variant"] = run["summary"]["best_variant"]
        report["warnings"] = run["summary"].get("warnings", [])
        questions = run["summary"].get("questions", {})
        per_query: dict[str, list[dict[str, Any]]] = {}
        for r in self.db.get_query_results(run_id):
            per_query.setdefault(r["variant_id"], []).append({
                "query_id": r["query_id"], "question": questions.get(r["query_id"], ""),
                "hit": r["recall"] > 0, "recall": r["recall"], "rr": round(r["rr"], 4),
                "ndcg": round(r["ndcg"], 4), "first_rank": r["first_rank"], "latency_ms": r["latency_ms"],
                "top_chunks": r["top_chunks"]})
        report["per_query"] = per_query
        return report

    def corpus_stats(self, corpus_id: str) -> dict[str, Any]:
        _, row = self._corpus(corpus_id)
        return {"corpus_id": corpus_id, "folder": row["folder"], "created_at": row["created_at"], **row["stats"]}
