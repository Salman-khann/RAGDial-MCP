"""Boundary-damage audit.

A structure (table, code fence, list, paragraph) is **severed** when no single
chunk's source span contains it whole. Overlap can rescue a structure: if a
cut lands inside a table but another chunk still holds the full table, it is
intact somewhere and not counted.

A structure is **lost** when some of its text is in no chunk at all. Lost
structures always count as severed at full weight, so a chunker can never
score well by dropping content.

A severed table is **mitigated** when every chunk touching it carries *that
table's own* header row: either the header lies inside the chunk's source
span, or the chunker prepended it (the ``markdown`` chunker replicates
headers). A matching header that belongs to a different table does not count.

``damage_score`` is the weighted share of severed structures, 0.0 (perfect)
to 1.0 (everything cut). Tables and code are weighted higher than prose
because a split row or function loses its meaning; mitigated tables count
at half weight.
"""

from __future__ import annotations

import bisect
from collections import defaultdict

from pydantic import BaseModel

from ..ingestion.parser import Block, Document
from .strategies import Chunk

WEIGHTS = {"table": 3.0, "code": 3.0, "list": 1.5, "paragraph": 1.0}
KIND_PRIORITY = {"table": 0, "code": 1, "list": 2, "paragraph": 3}
KINDS = tuple(WEIGHTS)


class SeveredStructure(BaseModel):
    doc_id: str
    section_id: str
    kind: str
    start: int
    end: int
    cut_at: int
    left_chunk: str | None
    right_chunk: str | None
    mitigated: bool
    oversized: bool  # larger than the chunk size, so it could never fit in one chunk
    lost: bool = False  # some of its text is in no chunk at all


class KindDamage(BaseModel):
    total: int
    severed: int


class AuditResult(BaseModel):
    damage_score: float
    structures: int
    severed: int
    mitigated: int
    lost: int = 0
    by_kind: dict[str, KindDamage]
    severed_items: list[SeveredStructure]


def trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Strip surrounding whitespace from a span; the audit and its token counts both use this."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _uncovered(text: str, start: int, end: int, chunks: list[Chunk]) -> bool:
    """True if any non-whitespace character in [start, end) lies in no chunk."""
    pos = start
    for c in sorted(chunks, key=lambda c: c.start):
        if c.start > pos and text[pos:min(c.start, end)].strip():
            return True
        pos = max(pos, c.end)
        if pos >= end:
            return False
    return bool(text[pos:end].strip())


def _carries_own_header(doc: Document, block: Block, c: Chunk) -> bool:
    header = block.header_line or ""
    if not header:
        return False
    header_end = block.start + len(header)
    if c.start <= block.start and c.end >= header_end:
        return True  # the real header row is inside this chunk's source span
    body = doc.text[c.start:c.end]
    prefix = c.text[:-len(body)] if body and c.text.endswith(body) else ""
    return header in prefix  # header the chunker prepended for this piece


def _severed(doc: Document, block: Block, chunks: list[Chunk], starts: list[int], max_len: int,
             size: int, token_count: int) -> SeveredStructure | None:
    lo = bisect.bisect_left(starts, block.start - max_len)
    hi = bisect.bisect_left(starts, block.end)
    near = chunks[lo:hi]
    if any(c.start <= block.start and c.end >= block.end for c in near):
        return None
    touching = [c for c in near if c.start < block.end and c.end > block.start]
    lost = _uncovered(doc.text, block.start, block.end, touching)
    cut, left, right = block.start, None, None
    candidates: list[int] = []
    for c in touching:
        if block.start < c.end < block.end:
            candidates.append(c.end)
        if block.start < c.start < block.end:
            candidates.append(c.start)
    if candidates:
        cut = min(candidates)
        left = next((c.chunk_id for c in touching if c.start < cut <= c.end), None)
        right = next((c.chunk_id for c in touching if c.start <= cut < c.end and c.chunk_id != left), None)
    mitigated = (not lost and block.kind == "table" and bool(touching)
                 and all(_carries_own_header(doc, block, c) for c in touching))
    return SeveredStructure(doc_id=doc.doc_id, section_id=block.section_id, kind=block.kind,
                            start=block.start, end=block.end, cut_at=cut, left_chunk=left,
                            right_chunk=right, mitigated=mitigated, oversized=token_count > size, lost=lost)


def audit_boundaries(docs: list[Document], chunks: list[Chunk], size: int,
                     token_counts: dict[tuple[str, int, int], int] | None = None) -> AuditResult:
    """``token_counts`` is keyed by ``(doc_id, *trim_span(...))`` and feeds the ``oversized`` flag."""
    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        by_doc[c.doc_id].append(c)
    totals: dict[str, int] = defaultdict(int)
    severed_counts: dict[str, int] = defaultdict(int)
    items: list[SeveredStructure] = []
    weight_total = weight_damage = 0.0
    for doc in docs:
        doc_chunks = sorted(by_doc.get(doc.doc_id, []), key=lambda c: c.start)
        starts = [c.start for c in doc_chunks]
        max_len = max((c.end - c.start for c in doc_chunks), default=0)
        for block in doc.structures:
            s, e = trim_span(doc.text, block.start, block.end)
            if e <= s:
                continue
            block = block.model_copy(update={"start": s, "end": e})
            w = WEIGHTS.get(block.kind, 1.0)
            totals[block.kind] += 1
            weight_total += w
            n = (token_counts or {}).get((doc.doc_id, s, e), 0)
            hit = _severed(doc, block, doc_chunks, starts, max_len, size, n)
            if hit:
                severed_counts[block.kind] += 1
                weight_damage += w * (0.5 if hit.mitigated else 1.0)
                items.append(hit)
    items.sort(key=lambda s: (not s.lost, KIND_PRIORITY.get(s.kind, 9), s.mitigated, s.doc_id, s.start))
    return AuditResult(
        damage_score=round(weight_damage / weight_total, 4) if weight_total else 0.0,
        structures=sum(totals.values()),
        severed=len(items),
        mitigated=sum(1 for s in items if s.mitigated),
        lost=sum(1 for s in items if s.lost),
        by_kind={k: KindDamage(total=totals[k], severed=severed_counts[k]) for k in sorted(totals)},
        severed_items=items,
    )


def excerpt(doc: Document, item: SeveredStructure, width: int = 110) -> tuple[str, str]:
    """Return (text before the cut, text after the cut), clipped to the structure."""
    lo = max(item.start, item.cut_at - width)
    hi = min(item.end, item.cut_at + width)
    before = doc.text[lo:item.cut_at]
    after = doc.text[item.cut_at:hi]
    if lo > item.start:
        before = "..." + before
    if hi < item.end:
        after = after + "..."
    return before, after
