"""Four chunkers that all emit chunks with exact source offsets.

=================  ==========================================================
``fixed``          sliding token windows (``size`` tokens, ``overlap`` shared)
``recursive``      recursive character splitting on paragraph -> line ->
                   sentence -> word separators, merged up to ``size`` tokens
``markdown``       header-aware: never crosses a section boundary, keeps
                   tables / code fences / lists atomic when they fit, splits
                   oversized tables by rows *with the header row replicated*,
                   and prefixes every chunk with its section breadcrumb
``semantic``       embedding-based: breaks where adjacent sentences/blocks
                   are least similar, capped at ``size`` tokens
=================  ==========================================================

Each chunk records ``start``/``end`` into the document text (the span of
source text it covers), while ``text`` is what gets embedded and indexed
(it may carry a breadcrumb or replicated table header).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pydantic import BaseModel

from ..ingestion.parser import Block, Document
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
    spans = [sp for sp in ti.spans if sp[0] >= start and sp[1] <= end]
    out: list[tuple[int, int]] = []
    for i in range(0, len(spans), step):
        window = spans[i:i + size]
        out.append((window[0][0], window[-1][1]))
        if i + size >= len(spans):
            break
    return out or [(start, end)]


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
          breaks: set[int] | None = None) -> list[Chunk]:
    """Greedily merge units into chunks <= size tokens, carrying ``overlap`` tokens of tail units."""
    breaks = breaks or set()
    tok = get_tokenizer()
    crumb_tokens: dict[str, int] = {}
    groups: list[list[_Unit]] = []
    cur: list[_Unit] = []

    def cur_tokens(us: list[_Unit]) -> int:
        return ti.count(us[0].start, us[-1].end) if us else 0

    for i, u in enumerate(units):
        hard = i in breaks or u.solo or (cur and cur[-1].solo)
        if cur and (hard or cur_tokens(cur + [u]) > size):
            groups.append(cur)
            carry: list[_Unit] = []
            if overlap > 0 and not hard:
                for prev in reversed(cur):
                    if prev.solo or cur_tokens([prev] + carry) > overlap:
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
        n = ti.count(start, end)
        if crumb:
            n += crumb_tokens.setdefault(crumb, tok.count(crumb))
        if g[0].solo and g[0].prefix:
            n += tok.count(g[0].prefix)
        sections = list(dict.fromkeys(s for u in g for s in u.section_ids)) or _sections_for(doc, start, end)
        chunks.append(Chunk(chunk_id=f"{doc.doc_id}::{len(chunks):04d}", doc_id=doc.doc_id, start=start,
                            end=end, tokens=n, text=text, section_ids=sections))
    return chunks


# ---------------------------------------------------------------- strategies

def chunk_fixed(doc: Document, ti: TokenIndex, size: int, overlap: int) -> list[Chunk]:
    step = max(1, size - overlap)
    units = [_Unit(s, e, ti.count(s, e)) for s, e in _token_windows(ti, 0, len(doc.text), size, step)]
    chunks = []
    for u in units:
        chunks.append(Chunk(chunk_id=f"{doc.doc_id}::{len(chunks):04d}", doc_id=doc.doc_id, start=u.start,
                            end=u.end, tokens=u.tokens, text=doc.text[u.start:u.end],
                            section_ids=_sections_for(doc, u.start, u.end)))
    return chunks


def chunk_recursive(doc: Document, ti: TokenIndex, size: int, overlap: int) -> list[Chunk]:
    units = _units_from_spans(doc, ti, _split_region(doc.text, ti, 0, len(doc.text), size))
    return _pack(doc, ti, units, size, overlap)


def _block_units(doc: Document, ti: TokenIndex, block: Block, size: int, budget: int) -> list[_Unit]:
    """Units for one block: atomic if it fits, otherwise structure-aware sub-splits."""
    text = doc.text
    n = ti.count(block.start, block.end)
    if n <= budget:
        return [_Unit(block.start, block.end, n)]
    if block.kind == "table":
        lines = text[block.start:block.end].split("\n")
        header = "\n".join(lines[:2]) + "\n"
        header_tokens = get_tokenizer().count(header)
        row_start = block.start + len(header)
        row_budget = max(8, budget - header_tokens)
        units = []
        # first piece keeps the real header in its span; later pieces get it replicated
        rows = _split_on(text, row_start, block.end, "\n")
        group: list[tuple[int, int]] = []
        for r in rows:
            if group and ti.count(group[0][0], r[1]) > row_budget:
                units.append(group)
                group = []
            group.append(r)
        if group:
            units.append(group)
        out: list[_Unit] = []
        for gi, grp in enumerate(units):
            s, e = _trim(text, grp[0][0], grp[-1][1])
            if gi == 0:
                out.append(_Unit(block.start, e, ti.count(block.start, e), solo=True))
            else:
                out.append(_Unit(s, e, ti.count(s, e) + header_tokens, prefix=header, solo=True))
        return out
    if block.kind == "code":
        spans = _split_region(text, ti, block.start, block.end, budget, ["\n\n", "\n", " "])
        return [_Unit(u.start, u.end, u.tokens, solo=True) for u in _units_from_spans(doc, ti, spans)]
    return _units_from_spans(doc, ti, _split_region(text, ti, block.start, block.end, budget))


def chunk_markdown(doc: Document, ti: TokenIndex, size: int, overlap: int) -> list[Chunk]:
    tok = get_tokenizer()
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
        crumb = " > ".join(section.path)
        budget = max(16, size - tok.count(crumb) - 2)
        breaks.add(len(units))
        for b in blocks:
            for u in _block_units(doc, ti, b, size, budget):
                u.breadcrumb = crumb
                u.section_ids = [section.section_id]
                units.append(u)
    return _pack(doc, ti, units, size, overlap, breaks)


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
    breaks: set[int] = set()
    if sims.size:
        distances = 1.0 - sims
        threshold = float(np.percentile(distances, percentile))
        for i, d in enumerate(distances):
            if d > threshold and not is_header[i]:
                breaks.add(i + 1)
    for i, h in enumerate(is_header):  # a header always opens a new chunk
        if h and i > 0:
            breaks.add(i)
    return _pack(doc, ti, units, size, overlap, breaks)


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
