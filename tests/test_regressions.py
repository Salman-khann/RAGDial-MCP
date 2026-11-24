"""Regression tests for the bugs found in the 0.1.2 audit. Each test names the defect it pins."""

from __future__ import annotations

import json
import sqlite3

import pytest

from ragdial_mcp.chunking.boundary_audit import audit_boundaries
from ragdial_mcp.chunking.strategies import ChunkConfig, chunk_document
from ragdial_mcp.ingestion.parser import parse_markdown
from ragdial_mcp.lab import LabError
from ragdial_mcp.retrieval.embedders import get_embedder
from ragdial_mcp.retrieval.metrics import normalize, score_ranking
from ragdial_mcp.retrieval.stats import mean_ci, paired_diff_ci, sign_test_p
from ragdial_mcp.tokens import TokenIndex, get_tokenizer

from .conftest import GOLDEN

STRATEGIES = ["fixed", "recursive", "markdown", "semantic"]
WIDE_TABLE = "# T\n\nIntro words.\n\n| " + " | ".join(f"col{i} value{i}" for i in range(60)) + " |\n|---|\n"


def _chunks(text, strategy, size, overlap=0):
    doc = parse_markdown(text, "d")
    return doc, chunk_document(doc, ChunkConfig(strategy=strategy, size=size, overlap=overlap), get_embedder("hash"))


def _uncovered(doc, chunks) -> str:
    """Non-whitespace text that is in no chunk (markdown carries headings as breadcrumbs, so skip them)."""
    covered = [False] * len(doc.text)
    for c in chunks:
        covered[c.start:c.end] = [True] * (c.end - c.start)
    headers = {i for b in doc.blocks if b.kind == "header" for i in range(b.start, b.end)}
    return "".join(ch for i, ch in enumerate(doc.text) if not covered[i] and not ch.isspace() and i not in headers)


# ------------------------------------------------------------ content loss
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_oversized_one_line_table_is_not_dropped(strategy):
    doc, chunks = _chunks(WIDE_TABLE, strategy, 64)
    assert _uncovered(doc, chunks) == ""


@pytest.mark.parametrize("strategy", ["recursive", "markdown", "semantic"])
def test_token_window_fallback_keeps_straddling_tokens(strategy):
    doc, chunks = _chunks("intro words here x" + "aB3{i}Zq" * 200 + " tail words end", strategy, 32)
    assert _uncovered(doc, chunks) == ""


def test_audit_reports_lost_content():
    doc = parse_markdown("# T\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\nTail paragraph here.\n", "d")
    table = next(b for b in doc.blocks if b.kind == "table")
    # a chunking that skips the table entirely
    from ragdial_mcp.chunking.strategies import Chunk
    tail = doc.text.index("Tail")
    chunks = [Chunk(chunk_id="d::0", doc_id="d", start=tail, end=len(doc.text.rstrip()), tokens=3,
                    text=doc.text[tail:].strip(), section_ids=[])]
    audit = audit_boundaries([doc], chunks, 64)
    lost = [s for s in audit.severed_items if s.kind == "table"]
    assert lost and lost[0].lost and not lost[0].mitigated and audit.lost >= 1
    assert table.start == lost[0].start


def test_empty_and_whitespace_documents_give_no_chunks():
    for text in ("", "   \n\n\t\n"):
        for s in STRATEGIES:
            assert _chunks(text, s, 32)[1] == []


# ------------------------------------------------------------ size cap and tokens
def test_tokens_counted_for_cjk_and_emoji():
    text = "数据库连接池配置说明 " * 30 + "🚀🔥💡🎉 " * 30
    assert len(TokenIndex(text)) == get_tokenizer().count(text)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_size_cap_holds_within_two_tokens(strategy):
    tok = get_tokenizer()
    texts = [WIDE_TABLE,
             "# A long document title for breadcrumbs\n\n## A long section heading here\n\n"
             + "\n\n".join(f"Short paragraph number {i} here." for i in range(40)),
             "数据库连接池配置说明 " * 40 + "\n\n" + "🚀🔥 " * 40]
    for text in texts:
        for size in (32, 64, 128):
            _, chunks = _chunks(text, strategy, size, 8)
            assert max(tok.count(c.text) for c in chunks) <= size + 2


def test_reported_tokens_match_chunk_text_for_markdown():
    _, chunks = _chunks(WIDE_TABLE, "markdown", 64)
    tok = get_tokenizer()
    assert all(c.tokens == tok.count(c.text) for c in chunks)


# ------------------------------------------------------------ parser
def test_inline_fence_does_not_swallow_document():
    doc = parse_markdown("```x``` is inline code.\n\n# Later\n\npara\n", "d")
    assert [b.kind for b in doc.blocks] == ["paragraph", "header", "paragraph"]


def test_nested_fences_close_at_matching_length():
    doc = parse_markdown("````md\n```py\nx\n```\n````\n\nafter\n", "d")
    assert [b.kind for b in doc.blocks] == ["code", "paragraph"]


def test_header_keeps_trailing_hash_without_space():
    assert parse_markdown("# Intro to C#\n\ntext\n", "d").sections[0].title == "Intro to C#"
    assert parse_markdown("# Closed ##\n\ntext\n", "d").sections[0].title == "Closed"


def test_pipe_rows_without_separator_are_not_a_table():
    doc = parse_markdown("| r1 | a |\n| r2 | b |\n", "d")
    assert [b.kind for b in doc.blocks] == ["paragraph"]


def test_thematic_break_is_not_a_list():
    doc = parse_markdown("text\n\n* * *\n\nmore\n", "d")
    assert [b.kind for b in doc.blocks] == ["paragraph", "paragraph"]


# ------------------------------------------------------------ audit: mitigation and oversized
def test_shared_header_from_another_table_is_not_mitigation(sample_docs):
    from ragdial_mcp.chunking.strategies import chunk_corpus
    chunks = chunk_corpus(sample_docs, ChunkConfig(strategy="fixed", size=512, overlap=0))
    audit = audit_boundaries(sample_docs, chunks, 512)
    assert audit.mitigated == 0  # fixed chunking never repeats headers


def test_nested_table_in_list_gets_header_replication(sample_docs):
    runbook = next(d for d in sample_docs if d.doc_id == "deployment_runbook")
    chunks = chunk_document(runbook, ChunkConfig(strategy="markdown", size=128, overlap=0))
    large = "| large | 9 | 16 | 64 GiB | 4 TiB | 600 MB/s |"
    header = "| Profile | Brokers | vCPU per broker | Memory per broker | Disk per broker | Sustained ingress |"
    assert any(large in c.text and header in c.text for c in chunks)


def test_oversized_flag_works_for_indented_blocks():
    from ragdial_mcp.chunking.boundary_audit import trim_span
    text = "\n".join(f"  - item {i} with some words" for i in range(80)) + "\n"
    doc = parse_markdown(text, "d")
    _, chunks = _chunks(text, "fixed", 32)
    ti = TokenIndex(doc.text)
    counts = {}
    for b in doc.structures:
        s, e = trim_span(doc.text, b.start, b.end)
        counts[("d", s, e)] = ti.count(s, e)
    audit = audit_boundaries([doc], chunks, 32, counts)
    assert any(s.kind == "list" and s.oversized for s in audit.severed_items)


# ------------------------------------------------------------ metrics and stats
def test_ndcg_one_chunk_covering_all_groups_is_perfect():
    assert score_ranking([{0, 1}], 2, 5)["ndcg"] == 1.0
    assert score_ranking([{0}, {1}], 2, 5)["ndcg"] == pytest.approx(1.0)


def test_stats_helpers():
    lo, hi = mean_ci([1.0] * 10)
    assert lo == hi == 1.0
    lo, hi = paired_diff_ci([0.0] * 30, [1.0] * 30)
    assert lo == hi == 1.0
    assert sign_test_p(5, 0) == pytest.approx(0.0625)
    assert sign_test_p(0, 0) == 1.0


# ------------------------------------------------------------ golden set integrity
def test_required_table_headers_are_unique(sample_docs):
    corpus = "\n".join(normalize(d.text) for d in sample_docs)
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        for g in r.get("evidence_groups") or [r["evidence"]]:
            if len(g) > 1:
                for snippet in g:
                    assert corpus.count(normalize(snippet)) == 1, (r["id"], snippet)


# ------------------------------------------------------------ lab: validation and integrity
@pytest.fixture
def ran(lab):
    lab.ingest_corpus("@sample")
    return lab, lab.run_chunking_ablation("sample", ["fixed", "markdown"], [128], [0], "hash")


def _golden(tmp_path, *rows):
    p = tmp_path / "g.jsonl"
    p.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(p)


@pytest.mark.parametrize("row, message", [
    ({"id": "x", "question": "q", "evidence_groups": ["a b"]}, "list of lists"),
    ('["a"]', "expected a JSON object"),
    ({"id": "x", "question": "q", "evidence": ""}, "list of strings"),
    ({"id": "x", "evidence": "512MiB"}, "missing 'question'"),
])
def test_malformed_golden_rows_are_rejected(ran, tmp_path, row, message):
    lab, abl = ran
    with pytest.raises(LabError, match=message):
        lab.benchmark_retrieval(abl.run_id, _golden(tmp_path, row), ["bm25"])


def test_duplicate_query_ids_rejected_without_orphan_runs(ran, tmp_path):
    lab, abl = ran
    row = {"id": "x", "question": "segment size", "evidence": "512MiB"}
    with pytest.raises(LabError, match="duplicate query id"):
        lab.benchmark_retrieval(abl.run_id, _golden(tmp_path, row, row), ["bm25"])
    assert not [r for r in lab.db.list_runs() if r["kind"] == "benchmark"]


def test_bom_in_golden_file_is_tolerated(ran, tmp_path):
    lab, abl = ran
    p = tmp_path / "bom.jsonl"
    p.write_text(json.dumps({"id": "x", "question": "segment size", "evidence": "512MiB"}) + "\n",
                 encoding="utf-8-sig")
    assert lab.benchmark_retrieval(abl.run_id, str(p), ["bm25"]).queries == 1


def test_wrong_dataset_path_is_not_silently_replaced(ran):
    lab, abl = ran
    with pytest.raises(LabError, match="Dataset not found"):
        lab.benchmark_retrieval(abl.run_id, "C:/nonexistent/dir/golden_qa.jsonl", ["bm25"])


def test_duplicate_configs_are_deduplicated(ran):
    lab, abl = ran
    b = lab.benchmark_retrieval(abl.run_id, "@sample", ["bm25"], configs=["fixed-128-o0", "fixed-128-o0"])
    assert len(b.results) == 1


def test_bad_inputs_give_lab_errors(ran):
    lab, abl = ran
    with pytest.raises(LabError, match="hash-<dim>"):
        lab.run_chunking_ablation("sample", ["semantic"], [256], [0], "hash-0")
    with pytest.raises(LabError, match="Unknown kind"):
        lab.inspect_boundary_splits(abl.run_id, kinds=["tables"])
    with pytest.raises(LabError, match="vector_store"):
        lab.benchmark_retrieval(abl.run_id, "@sample", ["dense"], embedder="hash", vector_store="bogus")
    assert all(r["summary"] for r in lab.db.list_runs())  # no half-written runs


def test_failed_save_leaves_no_partial_run(ran, monkeypatch):
    lab, abl = ran
    before = {r["run_id"] for r in lab.db.list_runs()}

    def failing(run_id, *a, **k):  # the run row is written, then the per-query insert fails
        with lab.db._lock, lab.db._conn:
            lab.db._insert_run(run_id, "benchmark", "sample", abl.run_id, {}, {"x": 1})
            raise sqlite3.IntegrityError("simulated")
    monkeypatch.setattr(lab.db, "save_benchmark_run", failing)
    with pytest.raises(LabError):
        lab.benchmark_retrieval(abl.run_id, "@sample", ["bm25"])
    assert {r["run_id"] for r in lab.db.list_runs()} == before  # the transaction rolled back


def test_diff_refuses_different_top_k_and_reports_uncertainty(ran):
    lab, abl = ran
    b5 = lab.benchmark_retrieval(abl.run_id, "@sample", ["bm25"], top_k=5)
    b1 = lab.benchmark_retrieval(abl.run_id, "@sample", ["bm25"], top_k=1)
    with pytest.raises(LabError, match="different top_k"):
        lab.diff_experiment_runs(b5.run_id, b1.run_id)
    d = lab.diff_experiment_runs(b5.run_id, b5.run_id, "fixed-128-o0|bm25|none", "markdown-128-o0|bm25|none")
    lo, hi = d.ndcg_delta_ci95
    assert lo <= d.metrics["ndcg_at_k"].delta <= hi
    assert d.significant == (lo > 0 or hi < 0)
    assert "95% CI" in d.verdict


def test_hash_embedder_warning_and_measured_overlap(ran):
    lab, _ = ran
    abl = lab.run_chunking_ablation("sample", ["fixed", "markdown"], [512], [64], "hash")
    by_id = {c.config_id: c for c in abl.configs}
    assert by_id["fixed-512-o64"].measured_overlap == 64.0
    assert by_id["markdown-512-o64"].measured_overlap == 0.0
    assert any("no effect" in n for n in abl.notes)
    b = lab.benchmark_retrieval(abl.run_id, "@sample", ["dense"], embedder="hash")
    assert any("hash" in w for w in b.warnings)
