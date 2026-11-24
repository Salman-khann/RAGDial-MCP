"""Four chunkers that all emit chunks with exact source offsets.

=================  ==========================================================
``fixed``          sliding token windows (``size`` tokens, ``overlap`` shared)
``recursive``      recursive character splitting on paragraph -> line ->
                   sentence -> word separators, merged up to ``size`` tokens
``markdown``       header-aware: never crosses a section boundary, keeps
                   tables / code fences / lists atomic when they fit, splits
                   oversized tables (including tables nested in lists) by rows
                   *with the header row replicated*, and prefixes every chunk
                   with its section breadcrumb. ``size`` includes the
                   breadcrumb and any replicated header.
``semantic``       embedding-based: breaks where adjacent sentences/blocks
                   are least similar, capped at ``size`` tokens
=================  ==========================================================

Overlap: ``fixed`` shares exactly ``overlap`` tokens. The other strategies
carry whole trailing units (sentences, lines, blocks) up to ``overlap``
tokens into the next chunk. ``markdown`` never carries across a section
boundary or into a table/code piece, so on corpora with short sections the
effective overlap can be near zero; the ablation reports the measured
overlap for every config.

Each chunk records ``start``/``end`` into the document text (the span of
source text it covers), while ``text`` is what gets embedded and indexed
(it may carry a breadcrumb or replicated table header).
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pydantic import BaseModel

from ..ingestion.parser import Block, Document, is_table_separator
from ..retrieval.embedders import Embedder, get_embedder
from ..tokens import TokenIndex, get_tokenizer

Strategy = Literal["fixed", "recursive", "markdown", "semantic"]
STRATEGIES: tuple[str, ...] = ("fixed", "recursive", "markdown", "semantic")
STRATEGY_ALIASES = {
    "fixed_token": "fixed", "fixed-token": "fixed", "token": "fixed",
    "recursive_character": "recursive", "recursive-character": "recursive",
    "header": "markdown", "header_aware": "markdown", "markdown_header": "markdown",
    "header-aware": "markdown", "md": "markdown",
    "embedding": "semantic",
}

_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`\"'(\[])")


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    start: int
    end: int
    tokens: int
    text: str
    section_ids: list[str]


class ChunkConfig(BaseModel):
    strategy: Strategy
    size: int
    overlap: int

    @property
    def config_id(self) -> str:
        return f"{self.strategy}-{self.size}-o{self.overlap}"


def normalize_strategy(name: str) -> str:
    key = name.strip().lower()
    key = STRATEGY_ALIASES.get(key, key)
    if key not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{name}'. Choose from {list(STRATEGIES)}")
    return key


@dataclass
class _Unit:
    start: int
    end: int
    tokens: int
    prefix: str = ""       # text prepended when this unit is emitted (e.g. replicated table header)
    solo: bool = False     # must be emitted as its own chunk
    breadcrumb: str = ""   # section breadcrumb, for header-aware chunks
    overhead: int = 0      # tokens the breadcrumb adds to any chunk this unit lands in
    section_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- helpers

def _split_on(text: str, start: int, end: int, sep: str) -> list[tuple[int, int]]:
    pieces: list[tuple[int, int]] = []
    pos = start
    while pos < end:
        idx = text.find(sep, pos, end)
        if idx == -1:
            pieces.append((pos, end))
            break
        pieces.append((pos, idx + len(sep)))
        pos = idx + len(sep)
    return [(s, e) for s, e in pieces if text[s:e].strip()]


def _token_windows(ti: TokenIndex, start: int, end: int, size: int, step: int) -> list[tuple[int, int]]:
    """Windows of ``size`` tokens over ``[start, end)``, every ``step`` tokens.

    Windows are clamped to the region, so a token that straddles the region edge
    (e.g. ``" xa..."`` after a split on spaces) never drops characters.
    """
    lo = bisect.bisect_left(ti.starts, start)
    hi = bisect.bisect_left(ti.starts, end)
    if lo >= hi:
        return [(start, end)] if ti.text[start:end].strip() else []
    out: list[tuple[int, int]] = []
    i = lo
    while i < hi:
        j = min(i + size, hi)
        s = start if i == lo else ti.spans[i][0]
        e = end if j == hi else ti.spans[j][0]
        if e > s:
            out.append((s, e))
        if j >= hi:
            break
        i += step
    return out


def _split_region(text: str, ti: TokenIndex, start: int, end: int, size: int,
                  seps: list[str] = _SEPARATORS) -> list[tuple[int, int]]:
    """Recursively split ``[start, end)`` until every piece is <= ``size`` tokens."""
    if ti.count(start, end) <= size:
        return [(start, end)] if text[start:end].strip() else []
    for k, sep in enumerate(seps):
        if text.find(sep, start, end) != -1:
            out: list[tuple[int, int]] = []
            for s, e in _split_on(text, start, end, sep):
                out.extend(_split_region(text, ti, s, e, size, seps[k + 1:]))
            return out
    return _token_windows(ti, start, end, size, size)


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _units_from_spans(doc: Document, ti: TokenIndex, spans: list[tuple[int, int]]) -> list[_Unit]:
    units = []
    for s, e in spans:
        s, e = _trim(doc.text, s, e)
        if e > s:
            units.append(_Unit(s, e, ti.count(s, e)))
    return units


def _sections_for(doc: Document, start: int, end: int) -> list[str]:
    ids = [s.section_id for s in doc.sections if s.start < end and s.end > start]
    return ids or ([doc.sections[0].section_id] if doc.sections else [])


def _pack(doc: Document, ti: TokenIndex, units: list[_Unit], size: int, overlap: int,
          hard_breaks: set[int] | None = None, soft_breaks: set[int] | None = None) -> list[Chunk]:
    """Greedily merge units into chunks of at most ``size`` tokens.

    A hard break (section boundary, solo table/code piece) always starts a new
    chunk with no carried overlap. A soft break (semantic topic shift) starts a
    new chunk but still carries up to ``overlap`` tokens of trailing units.
    """
    hard_breaks = hard_breaks or set()
    soft_breaks = soft_breaks or set()
    tok = get_tokenizer()
    groups: list[list[_Unit]] = []
    cur: list[_Unit] = []

    def cur_tokens(us: list[_Unit]) -> int:
        return ti.count(us[0].start, us[-1].end) + us[0].overhead if us else 0

    for i, u in enumerate(units):
        hard = i in hard_breaks or u.solo or bool(cur and cur[-1].solo)
        if cur and (hard or i in soft_breaks or cur_tokens(cur + [u]) > size):
            groups.append(cur)
            carry: list[_Unit] = []
            if overlap > 0 and not hard:
                for prev in reversed(cur):
                    if prev.solo or ti.count(prev.start, (carry or [prev])[-1].end) > overlap:
                        break
                    carry.insert(0, prev)
                if carry and cur_tokens(carry + [u]) > size:
                    carry = []
            cur = carry
        cur.append(u)
    if cur:
        groups.append(cur)

    chunks: list[Chunk] = []
    for g in groups:
        start, end = g[0].start, g[-1].end
        body = doc.text[start:end]
        if g[0].solo and g[0].prefix:
            body = g[0].prefix + body
        crumb = g[0].breadcrumb
        text = f"{crumb}\n\n{body}" if crumb else body
        n = tok.count(text)
        sections = list(dict.fromkeys(s for u in g for s in u.section_ids)) or _sections_for(doc, start, end)
        chunks.append(Chunk(chunk_id=f"{doc.doc_id}::{len(chunks):04d}", doc_id=doc.doc_id, start=start,
                            end=end, tokens=n, text=text, section_ids=sections))
    return chunks


# ---------------------------------------------------------------- strategies

def chunk_fixed(doc: Document, ti: TokenIndex, size: int, overlap: int) -> list[Chunk]:
    step = max(1, size - overlap)
    chunks = []
    for s, e in _token_windows(ti, 0, len(doc.text), size, step):
        if not doc.text[s:e].strip():
            continue  # whitespace-only window (tiktoken tokenizes whitespace): nothing to retrieve
        chunks.append(Chunk(chunk_id=f"{doc.doc_id}::{len(chunks):04d}", doc_id=doc.doc_id, start=s,
                            end=e, tokens=ti.count(s, e), text=doc.text[s:e],
                            section_ids=_sections_for(doc, s, e)))
    return chunks


def chunk_recursive(doc: Document, ti: TokenIndex, size: int, overlap: int) -> list[Chunk]:
    units = _units_from_spans(doc, ti, _split_region(doc.text, ti, 0, len(doc.text), size))
    return _pack(doc, ti, units, size, overlap)


def _table_units(doc: Document, ti: TokenIndex, start: int, end: int, budget: int) -> list[_Unit]:
    """Split an oversized table by rows, replicating its header row(s) into every later piece."""
    text = doc.text
    lines = _split_on(text, start, end, "\n")
    if not lines:
        return []
    n_header = 2 if len(lines) > 1 and is_table_separator(text[lines[1][0]:lines[1][1]]) else 1
    tok = get_tokenizer()
    header = "\n".join(text[s:e].strip() for s, e in lines[:n_header]) + "\n"
    header_tokens = tok.count(header)
    rows = lines[n_header:]
    if not rows or header_tokens >= budget // 2:
        # one giant line, or a header too big to repeat: split it like text rather than drop it
        return _units_from_spans(doc, ti, _split_region(text, ti, start, end, budget))
    row_budget = budget - header_tokens
    pieces: list[tuple[int, int]] = []        # each piece is a span of whole rows (or part of one huge row)
    group: list[tuple[int, int]] = []
    for r in rows:
        if ti.count(*r) > row_budget:          # a single row bigger than the budget: split the row itself
            if group:
                pieces.append((group[0][0], group[-1][1]))
                group = []
            pieces.extend(_split_region(text, ti, r[0], r[1], row_budget))
            continue
        if group and ti.count(group[0][0], r[1]) > row_budget:
            pieces.append((group[0][0], group[-1][1]))
            group = []
        group.append(r)
    if group:
        pieces.append((group[0][0], group[-1][1]))
    out: list[_Unit] = []
    for pi, (s, e) in enumerate(pieces):
        s, e = _trim(text, s, e)
        if e <= s:
            continue
        if pi == 0 and ti.count(start, e) <= budget:  # first piece keeps the real header in its own span
            out.append(_Unit(start, e, ti.count(start, e), solo=True))
        else:
            out.append(_Unit(s, e, ti.count(s, e) + header_tokens, prefix=header, solo=True))
    return out


def _block_units(doc: Document, ti: TokenIndex, block: Block, budget: int) -> list[_Unit]:
    """Units for one block: atomic if it fits, otherwise structure-aware sub-splits."""
    text = doc.text
    n = ti.count(block.start, block.end)
    if n <= budget:
        return [_Unit(block.start, block.end, n)]
    if block.kind == "table":
        return _table_units(doc, ti, block.start, block.end, budget)
    if block.kind == "code":
        spans = _split_region(text, ti, block.start, block.end, budget, ["\n\n", "\n", " "])
        return [_Unit(u.start, u.end, u.tokens, solo=True) for u in _units_from_spans(doc, ti, spans)]
    if block.kind == "list":
        # tables nested inside list items get the same table-aware treatment as top-level tables
        nested = sorted((t for t in doc.nested_tables if block.start <= t.start and t.end <= block.end),
                        key=lambda t: t.start)
        units: list[_Unit] = []
        pos = block.start
        for t in nested:
            units += _units_from_spans(doc, ti, _split_region(text, ti, pos, t.start, budget))
            tn = ti.count(t.start, t.end)
            units += [_Unit(t.start, t.end, tn, solo=True)] if tn <= budget else _table_units(
                doc, ti, t.start, t.end, budget)
            pos = t.end
        units += _units_from_spans(doc, ti, _split_region(text, ti, pos, block.end, budget))
        return units
    return _units_from_spans(doc, ti, _split_region(text, ti, block.start, block.end, budget))


def _breadcrumb(section_path: list[str], size: int) -> tuple[str, int]:
    """Section breadcrumb, shortened (or dropped) so it never eats more than a quarter of a chunk."""
    tok = get_tokenizer()
    for crumb in (" > ".join(section_path), section_path[-1] if section_path else "", ""):
        # +2: re-encoding "crumb + body" can merge tokens differently at the seam
        n = tok.count(f"{crumb}\n\n") + 2 if crumb else 0
        if n <= size // 4:
            return crumb, n
    return "", 0


def chunk_markdown(doc: Document, ti: TokenIndex, size: int, overlap: int) -> list[Chunk]:
    units: list[_Unit] = []
    breaks: set[int] = set()
    by_section: dict[str, list[Block]] = {}
    for b in doc.blocks:
        if b.kind != "header":
            by_section.setdefault(b.section_id, []).append(b)
    for section in doc.sections:
        blocks = by_section.get(section.section_id)
        if not blocks:
            continue
        crumb, overhead = _breadcrumb(section.path, size)
        budget = max(8, size - overhead)
        breaks.add(len(units))
        for b in blocks:
            for u in _block_units(doc, ti, b, budget):
                u.breadcrumb, u.overhead = crumb, overhead
                u.section_ids = [section.section_id]
                units.append(u)
    return _pack(doc, ti, units, size, overlap, hard_breaks=breaks)


def chunk_semantic(doc: Document, ti: TokenIndex, size: int, overlap: int, embedder: Embedder,
                   percentile: float = 75.0) -> list[Chunk]:
    units: list[_Unit] = []
    is_header: list[bool] = []
    for b in doc.blocks:
        if b.kind in ("paragraph", "list") and ti.count(b.start, b.end) > 0:
            spans: list[tuple[int, int]] = []
            for s, e in _split_on(doc.text, b.start, b.end, "\n") if b.kind == "list" else [(b.start, b.end)]:
                pos = s
                for m in _SENTENCE_END.finditer(doc.text, s, e):
                    spans.append((pos, m.start()))
                    pos = m.end()
                spans.append((pos, e))
            pieces: list[tuple[int, int]] = []
            for s, e in spans:
                pieces.extend(_split_region(doc.text, ti, s, e, size))
            new = _units_from_spans(doc, ti, pieces)
        else:
            new = _units_from_spans(doc, ti, _split_region(doc.text, ti, b.start, b.end, size))
        units.extend(new)
        is_header.extend([b.kind == "header"] * len(new))
    if not units:
        return []
    vecs = embedder.embed_documents([doc.text[u.start:u.end] for u in units])
    sims = np.einsum("ij,ij->i", vecs[:-1], vecs[1:]) if len(units) > 1 else np.array([])
    topic_breaks: set[int] = set()
    if sims.size:
        distances = 1.0 - sims
        threshold = float(np.percentile(distances, percentile))
        for i, d in enumerate(distances):
            if d > threshold and not is_header[i]:
                topic_breaks.add(i + 1)
    header_breaks = {i for i, h in enumerate(is_header) if h and i > 0}  # a header always opens a new chunk
    return _pack(doc, ti, units, size, overlap, hard_breaks=header_breaks,
                 soft_breaks=topic_breaks - header_breaks)


def chunk_document(doc: Document, config: ChunkConfig, embedder: Embedder | None = None,
                   ti: TokenIndex | None = None) -> list[Chunk]:
    ti = ti or TokenIndex(doc.text)
    if config.strategy == "fixed":
        return chunk_fixed(doc, ti, config.size, config.overlap)
    if config.strategy == "recursive":
        return chunk_recursive(doc, ti, config.size, config.overlap)
    if config.strategy == "markdown":
        return chunk_markdown(doc, ti, config.size, config.overlap)
    if config.strategy == "semantic":
        return chunk_semantic(doc, ti, config.size, config.overlap, embedder or get_embedder("hash"))
    raise ValueError(config.strategy)


def chunk_corpus(docs: list[Document], config: ChunkConfig, embedder: Embedder | None = None,
                 token_indexes: dict[str, TokenIndex] | None = None) -> list[Chunk]:
    token_indexes = token_indexes if token_indexes is not None else {}
    out: list[Chunk] = []
    for doc in docs:
        ti = token_indexes.get(doc.doc_id)
        if ti is None:
            ti = token_indexes[doc.doc_id] = TokenIndex(doc.text)
        out.extend(chunk_document(doc, config, embedder, ti))
    return out


def measured_overlap(chunks: list[Chunk], token_indexes: dict[str, TokenIndex]) -> float:
    """Mean tokens of source text shared by consecutive chunks of the same document."""
    shared: list[int] = []
    for a, b in zip(chunks, chunks[1:]):
        if a.doc_id == b.doc_id:
            shared.append(token_indexes[a.doc_id].count(b.start, a.end) if b.start < a.end else 0)
    return round(sum(shared) / len(shared), 1) if shared else 0.0
