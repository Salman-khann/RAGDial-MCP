from __future__ import annotations

import json

import pytest

from ragdial_mcp.chunking.boundary_audit import audit_boundaries
from ragdial_mcp.chunking.strategies import ChunkConfig, chunk_corpus, chunk_document, normalize_strategy
from ragdial_mcp.ingestion.parser import parse_markdown
from ragdial_mcp.retrieval.embedders import get_embedder
from ragdial_mcp.retrieval.metrics import normalize
from ragdial_mcp.tokens import TokenIndex

from .conftest import GOLDEN

TABLE_DOC = """# Limits

Intro paragraph about limits.

## Quotas

| Plan | RPS | Burst |
|---|---|---|
""" + "\n".join(f"| plan-{i:02d} | {i * 10} | {i * 20} |" for i in range(40)) + """

## Example

```python
def handler(event):
    return event["id"]
```

- first item
- second item
"""

ALL = ["fixed", "recursive", "markdown", "semantic"]


def _chunks(doc, strategy, size, overlap=0):
    return chunk_document(doc, ChunkConfig(strategy=strategy, size=size, overlap=overlap), get_embedder("hash"))


def test_parser_detects_structures():
    doc = parse_markdown(TABLE_DOC, "limits")
    kinds = [b.kind for b in doc.blocks]
    assert kinds.count("table") == 1 and kinds.count("code") == 1 and kinds.count("list") == 1
    assert [s.title for s in doc.sections] == ["Limits", "Quotas", "Example"]
    assert doc.section("limits#quotas").path == ["Limits", "Quotas"]
    code = next(b for b in doc.blocks if b.kind == "code")
    assert doc.text[code.start:code.end].startswith("```python") and doc.text[code.start:code.end].endswith("```")


def test_parser_ignores_hash_lines_inside_code_fences():
    doc = parse_markdown("# T\n\n```bash\n# a comment\nls\n```\n", "d")
    assert [s.title for s in doc.sections] == ["T"]


def test_nested_tables_inside_lists(sample_docs):
    runbook = next(d for d in sample_docs if d.doc_id == "deployment_runbook")
    assert len(runbook.nested_tables) == 2
    t = runbook.nested_tables[0]
    assert runbook.text[t.start:t.end].startswith("| Profile |")


@pytest.mark.parametrize("strategy", ALL)
def test_offsets_are_valid_and_ordered(strategy, sample_docs):
    for doc in sample_docs[:4]:
        chunks = _chunks(doc, strategy, 128, 16)
        assert chunks
        for c in chunks:
            assert 0 <= c.start < c.end <= len(doc.text)
            assert doc.text[c.start:c.end].strip()
        assert [c.start for c in chunks] == sorted(c.start for c in chunks)


@pytest.mark.parametrize("strategy", ["fixed", "recursive"])
def test_size_cap_respected(strategy, sample_docs):
    for doc in sample_docs:
        assert max(c.tokens for c in _chunks(doc, strategy, 100)) <= 100


def test_fixed_overlap_shares_exact_tokens():
    doc = parse_markdown(" ".join(f"word{i}" for i in range(400)), "d")
    chunks = _chunks(doc, "fixed", 50, 10)
    ti = TokenIndex(doc.text)
    for a, b in zip(chunks, chunks[1:]):
        assert b.start < a.end  # windows overlap
        assert ti.count(b.start, a.end) == 10


def test_markdown_never_crosses_sections_and_adds_breadcrumb():
    doc = parse_markdown(TABLE_DOC, "limits")
    for c in _chunks(doc, "markdown", 512):
        assert len(c.section_ids) == 1
        assert c.text.startswith("Limits")


def test_markdown_replicates_table_header_when_splitting():
    doc = parse_markdown(TABLE_DOC, "limits")
    table_chunks = [c for c in _chunks(doc, "markdown", 120) if "plan-" in c.text]
    assert len(table_chunks) > 1
    assert all("| Plan | RPS | Burst |" in c.text for c in table_chunks)


def test_boundary_audit_flags_fixed_but_not_markdown():
    doc = parse_markdown(TABLE_DOC, "limits")
    fixed = audit_boundaries([doc], _chunks(doc, "fixed", 64), 64)
    md = audit_boundaries([doc], _chunks(doc, "markdown", 1024), 1024)
    assert fixed.by_kind["table"].severed == 1 and not fixed.severed_items[0].mitigated
    assert md.severed == 0 and md.damage_score == 0.0
    md_small = audit_boundaries([doc], _chunks(doc, "markdown", 120), 120)
    table = next(s for s in md_small.severed_items if s.kind == "table")
    assert table.mitigated  # header replicated into every piece
    assert md_small.damage_score < fixed.damage_score


def test_overlap_rescues_structures():
    text = "# T\n\n" + "\n\n".join(f"Paragraph {i} " + "filler " * 20 for i in range(6))
    doc = parse_markdown(text, "d")
    no_ov = audit_boundaries([doc], _chunks(doc, "fixed", 40, 0), 40)
    big_ov = audit_boundaries([doc], _chunks(doc, "fixed", 40, 39), 40)
    assert big_ov.severed <= no_ov.severed


def test_chunking_is_deterministic(sample_docs):
    for strategy in ALL:
        cfg = ChunkConfig(strategy=strategy, size=256, overlap=32)
        a = chunk_corpus(sample_docs, cfg, get_embedder("hash"))
        b = chunk_corpus(sample_docs, cfg, get_embedder("hash"))
        assert [c.model_dump() for c in a] == [c.model_dump() for c in b]


def test_strategy_aliases():
    assert normalize_strategy("header_aware") == "markdown"
    assert normalize_strategy("Fixed-Token") == "fixed"
    with pytest.raises(ValueError):
        normalize_strategy("magic")


def test_golden_evidence_exists_verbatim(sample_docs):
    corpus = "\n".join(normalize(d.text) for d in sample_docs)
    doc_ids = {d.doc_id for d in sample_docs}
    rows = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 30
    assert len({r["id"] for r in rows}) == 30
    for r in rows:
        groups = r.get("evidence_groups") or [r["evidence"]]
        for g in groups:
            for snippet in g:
                assert normalize(snippet) in corpus, (r["id"], snippet)
        if r.get("doc_id"):
            assert r["doc_id"] in doc_ids
