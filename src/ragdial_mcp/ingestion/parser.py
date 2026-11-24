"""Corpus loader: Markdown/text/PDF -> structured documents.

Every document is reduced to Markdown text plus character-offset annotations:

* **sections**   - spans between Markdown headers, with a breadcrumb path
* **blocks**     - top-level header / paragraph / table / code / list blocks
* **structures** - the things a chunker can damage (tables, code fences,
  lists, paragraphs, including tables nested inside list items)

Offsets always index into ``Document.text`` so the boundary auditor can tell
exactly which structure a chunk boundary cut through.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ..tokens import get_tokenizer

BlockKind = Literal["header", "paragraph", "table", "code", "list"]

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}

_HEADER = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_TABLE = re.compile(r"^\s*\|.*\|\s*$")
_LIST = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+")


class Block(BaseModel):
    kind: BlockKind
    start: int
    end: int
    section_id: str
    header_line: str | None = None  # first row of a table (used for header replication)


class Section(BaseModel):
    section_id: str
    title: str
    level: int
    start: int
    end: int
    path: list[str] = Field(default_factory=list)


class Document(BaseModel):
    doc_id: str
    path: str
    title: str
    source_format: str
    text: str
    tokens: int
    sections: list[Section]
    blocks: list[Block]
    nested_tables: list[Block] = Field(default_factory=list)

    @property
    def structures(self) -> list[Block]:
        """Blocks a chunker can sever (everything except header lines)."""
        return [b for b in self.blocks if b.kind != "header"] + list(self.nested_tables)

    def section(self, section_id: str) -> Section | None:
        return next((s for s in self.sections if s.section_id == section_id), None)


def slugify(value: str) -> str:
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[\s-]+", "-", value).strip("-_") or "section"


def _doc_id_for(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    return "/".join(slugify(p) for p in rel.parts)


def _read_text(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            import pymupdf4llm  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                f"{path.name}: PDF support needs the 'pdf' extra (uv sync --extra pdf)"
            ) from exc
        return pymupdf4llm.to_markdown(str(path)), "pdf"
    return path.read_text(encoding="utf-8", errors="replace"), suffix.lstrip(".") or "txt"


def _is_table(line: str) -> bool:
    return bool(_TABLE.match(line))


def parse_markdown(text: str, doc_id: str, path: str = "", source_format: str = "md") -> Document:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.splitlines(keepends=True)
    offsets: list[int] = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line)

    def end_of(idx: int) -> int:
        """End offset of line ``idx`` without its trailing newline."""
        return offsets[idx] + len(lines[idx].rstrip("\n"))

    title: str | None = None
    used_ids: set[str] = set()
    # (level, title) stack used to build breadcrumbs
    path_stack: list[tuple[int, str]] = []

    root = Section(section_id=f"{doc_id}#_root", title="", level=0, start=0, end=len(text))
    used_ids.add(root.section_id)
    sections: list[Section] = [root]
    current = root

    def new_section(htitle: str, level: int, start: int) -> Section:
        base = f"{doc_id}#{slugify(htitle)}"
        sid, n = base, 2
        while sid in used_ids:
            sid, n = f"{base}-{n}", n + 1
        used_ids.add(sid)
        while path_stack and path_stack[-1][0] >= level:
            path_stack.pop()
        path_stack.append((level, htitle))
        return Section(section_id=sid, title=htitle, level=level, start=start, end=len(text),
                       path=[t for _, t in path_stack])

    blocks: list[Block] = []
    nested: list[Block] = []
    i, n_lines = 0, len(lines)

    while i < n_lines:
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue

        m = _HEADER.match(line.rstrip("\n"))
        if m:
            level, htitle = len(m.group(1)), m.group(2).strip()
            if title is None and level == 1:
                title = htitle
            current.end = offsets[i]
            current = new_section(htitle, level, offsets[i])
            sections.append(current)
            blocks.append(Block(kind="header", start=offsets[i], end=end_of(i), section_id=current.section_id))
            i += 1
            continue

        fm = _FENCE.match(line)
        if fm:
            fence = fm.group(1)
            j = i + 1
            while j < n_lines and not lines[j].strip().startswith(fence):
                j += 1
            j = min(j, n_lines - 1)
            blocks.append(Block(kind="code", start=offsets[i], end=end_of(j), section_id=current.section_id))
            i = j + 1
            continue

        if _is_table(line):
            j = i
            while j + 1 < n_lines and _is_table(lines[j + 1]):
                j += 1
            blocks.append(Block(kind="table", start=offsets[i], end=end_of(j),
                                section_id=current.section_id, header_line=stripped))
            i = j + 1
            continue

        if _LIST.match(line):
            j = i
            while j + 1 < n_lines:
                nxt = lines[j + 1]
                indented = bool(nxt.strip()) and nxt[:1] in " \t"
                if _LIST.match(nxt) or indented:
                    j += 1
                    continue
                if not nxt.strip() and j + 2 < n_lines:
                    after = lines[j + 2]
                    if _LIST.match(after) or (after.strip() and after[:1] in " \t"):
                        j += 1  # blank line inside a loose list
                        continue
                break
            blocks.append(Block(kind="list", start=offsets[i], end=end_of(j), section_id=current.section_id))
            k = i
            while k <= j:  # tables nested inside list items
                if _is_table(lines[k]):
                    t = k
                    while t + 1 <= j and _is_table(lines[t + 1]):
                        t += 1
                    indent = len(lines[k]) - len(lines[k].lstrip())
                    nested.append(Block(kind="table", start=offsets[k] + indent, end=end_of(t),
                                        section_id=current.section_id, header_line=lines[k].strip()))
                    k = t + 1
                else:
                    k += 1
            i = j + 1
            continue

        j = i  # paragraph: runs until a blank line or another block starts
        while j + 1 < n_lines:
            nxt = lines[j + 1]
            if (not nxt.strip() or _HEADER.match(nxt.rstrip("\n")) or _FENCE.match(nxt)
                    or _is_table(nxt) or _LIST.match(nxt)):
                break
            j += 1
        blocks.append(Block(kind="paragraph", start=offsets[i], end=end_of(j), section_id=current.section_id))
        i = j + 1

    current.end = len(text)
    doc_title = title or (Path(path).stem if path else doc_id)
    root.title = doc_title
    if not any(b.section_id == root.section_id for b in blocks):
        sections.remove(root)
    for s in sections:
        # breadcrumb always starts with the document title, without repeating it
        crumbs = [p for p in s.path]
        if not crumbs or crumbs[0] != doc_title:
            crumbs = [doc_title, *crumbs]
        s.path = crumbs

    return Document(doc_id=doc_id, path=path, title=doc_title, source_format=source_format, text=text,
                    tokens=get_tokenizer().count(text), sections=sections, blocks=blocks,
                    nested_tables=nested)


def load_folder(folder: str | Path) -> tuple[list[Document], list[str]]:
    """Parse every supported file under ``folder``. Returns (documents, warnings)."""
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")
    docs: list[Document] = []
    warnings: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            text, fmt = _read_text(path)
        except Exception as exc:  # keep ingesting the rest of the corpus
            warnings.append(str(exc))
            continue
        docs.append(parse_markdown(text, _doc_id_for(path, root), path.relative_to(root).as_posix(), fmt))
    if not docs:
        raise ValueError(f"No supported documents ({', '.join(sorted(SUPPORTED_SUFFIXES))}) in {root}")
    return docs, warnings
