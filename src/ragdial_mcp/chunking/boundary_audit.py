"""Boundary-damage audit.

A structure (table, code fence, list, paragraph) is **severed** when no single
chunk's source span contains it whole. Overlap can rescue a structure: if a
cut lands inside a table but another chunk still holds the full table, it is
intact somewhere and not counted.

A severed table is **mitigated** when every chunk touching its rows also
carries the table's header row in its text (the ``markdown`` chunker
replicates headers), so each piece is still self-describing.

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


class KindDamage(BaseModel):
    total: int
    severed: int


class AuditResult(BaseModel):
    damage_score: float
    structures: int
    severed: int
    mitigated: int
    by_kind: dict[str, KindDamage]
    severed_items: list[SeveredStructure]


def _severed(doc: Document, block: Block, chunks: list[Chunk], starts: list[int],
             size: int, token_count: int) -> SeveredStructure | None:
    # candidate chunks are those starting at or before the structure start
    hi = bisect.bisect_right(starts, block.start)
    for c in chunks[:hi]:
        if c.start <= block.start and c.end >= block.end:
            return None
    touching = [c for c in chunks if c.start < block.end and c.end > block.start]
    if not touching:  # structure fell entirely outside chunks (e.g. whitespace-only)
        return None
    # first boundary strictly inside the structure
    cut, left, right = block.start, None, None
    candidates: list[tuple[int, str | None, str | None]] = []
    for c in touching:
        if block.start < c.end < block.end:
            candidates.append((c.end, c.chunk_id, None))
        if block.start < c.start < block.end:
            candidates.append((c.start, None, c.chunk_id))
    if candidates:
        cut = min(x[0] for x in candidates)
        left = next((c.chunk_id for c in touching if c.start < cut <= c.end), None)
        right = next((c.chunk_id for c in touching if c.start <= cut < c.end and c.chunk_id != left), None)
    mitigated = False
    if block.kind == "table" and block.header_line:
        mitigated = all(block.header_line in c.text for c in touching)
    return SeveredStructure(doc_id=doc.doc_id, section_id=block.section_id, kind=block.kind,
                            start=block.start, end=block.end, cut_at=cut, left_chunk=left,
                            right_chunk=right, mitigated=mitigated, oversized=token_count > size)


def audit_boundaries(docs: list[Document], chunks: list[Chunk], size: int,
                     token_counts: dict[tuple[str, int, int], int] | None = None) -> AuditResult:
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
        for block in doc.structures:
            s, e = block.start, block.end
            while s < e and doc.text[s].isspace():
                s += 1
            while e > s and doc.text[e - 1].isspace():
                e -= 1
            if e <= s:
                continue
            block = block.model_copy(update={"start": s, "end": e})
            w = WEIGHTS.get(block.kind, 1.0)
            totals[block.kind] += 1
            weight_total += w
            n = (token_counts or {}).get((doc.doc_id, block.start, block.end), 0)
            hit = _severed(doc, block, doc_chunks, starts, size, n)
            if hit:
                severed_counts[block.kind] += 1
                weight_damage += w * (0.5 if hit.mitigated else 1.0)
                items.append(hit)
    items.sort(key=lambda s: (KIND_PRIORITY.get(s.kind, 9), s.mitigated, s.doc_id, s.start))
    return AuditResult(
        damage_score=round(weight_damage / weight_total, 4) if weight_total else 0.0,
        structures=sum(totals.values()),
        severed=len(items),
        mitigated=sum(1 for s in items if s.mitigated),
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
