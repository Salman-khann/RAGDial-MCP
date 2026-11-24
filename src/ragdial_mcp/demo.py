"""``ragdial-demo``: the full before/after loop on the bundled benchmark pack, in seconds.

    uvx --from ragdial-mcp ragdial-demo        # offline, deterministic (hash embedder, BM25/RRF)
    uvx --from "ragdial-mcp[ml]" ragdial-demo --embedder bge-small --reranker bge-reranker-base

Output design: two entity colours (orange = fixed-token baseline, blue =
header-aware candidate; validated for colour-vision deficiency on the dark
surface), status colours only for outcomes and always paired with an icon,
numbers in neutral ink.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

from rich import box
from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .lab import Lab
from .samples import sample_corpus, sample_golden

BASE = "#d95926"      # series 1: fixed-token chunking (baseline)
CAND = "#3987e5"      # series 2: header-aware markdown chunking (candidate)
GOOD = "#0ca30c"      # status: good
BAD = "#d03b3b"       # status: critical
INK = "#f0efec"       # primary text
SUB = "#c3c2b7"       # secondary text
MUTED = "#8a8984"     # captions, rules

RETRIEVER = {"bm25": "BM25", "dense": "Dense", "hybrid": "Hybrid (RRF)"}
EIGHTHS = " ▏▎▍▌▋▊▉█"


def bar(value: float, width: int, color: str) -> Text:
    """Horizontal bar of ``value`` in [0, 1], drawn with eighth-block precision."""
    cells = max(0.0, min(1.0, value)) * width
    full, frac = int(cells), int(round((cells - int(cells)) * 8))
    if frac == 8:
        full, frac = full + 1, 0
    s = "█" * full + (EIGHTHS[frac] if frac and full < width else "")
    return Text(s.ljust(width), style=color)


def chunker(config_id: str) -> tuple[str, str]:
    """('fixed-256', colour) for a config id like 'fixed-256-o0'."""
    name = config_id.rsplit("-o", 1)[0]
    return name, BASE if name.startswith("fixed") else CAND


def swatch(config_id: str) -> Text:
    name, color = chunker(config_id)
    return Text.assemble(("● ", color), (name, INK))


def step(console: Console, n: int, title: str, caption: str) -> None:
    console.print()
    console.print(Text.assemble((f" {n} ", f"bold {INK} on #383835"), " ", (title, f"bold {INK}")))
    for line in caption.split("\n"):
        console.print(Text(f"    {line}", style=MUTED))


def section_name(section_id: str) -> str:
    doc, _, slug = section_id.partition("#")
    return f"{doc} › {slug.replace('-', ' ')}"


def clip_lines(text: str, keep: int, from_end: bool) -> list[str]:
    lines = [ln for ln in text.split("\n")]
    lines = lines[-keep:] if from_end else lines[:keep]
    return [ln.strip().replace("...", "…") for ln in lines]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ragdial-demo")
    p.add_argument("--corpus", default=None, help="Corpus folder (default: bundled sample pack)")
    p.add_argument("--dataset", default=None, help="Golden JSONL (default: bundled sample pack)")
    p.add_argument("--embedder", default="hash")
    p.add_argument("--reranker", default="none")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--home", default=None, help="Persist runs here (default: a temp dir)")
    args = p.parse_args(argv)
    args.corpus = args.corpus or str(sample_corpus())
    args.dataset = args.dataset or str(sample_golden())
    k = args.top_k

    # FORCE_COLOR means "emit ANSI even when piped"; rich's legacy Windows mode would bypass the pipe
    legacy = False if os.environ.get("FORCE_COLOR") else None
    truecolor = os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit")
    # the output uses symbols (✓ ✂ ●); never crash on a pipe or file with a legacy code page
    if (getattr(sys.stdout, "encoding", "") or "").lower().replace("-", "") != "utf8":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
    probe = Console(legacy_windows=legacy)
    width = min(probe.width, 100) if probe.is_terminal else 100  # pipes/files: don't fall back to 80 cols
    console = Console(highlight=False, legacy_windows=legacy, color_system="truecolor" if truecolor else "auto",
                      width=width)
    t0 = time.perf_counter()
    lab = Lab(args.home or tempfile.mkdtemp(prefix="ragdial-demo-"))

    console.print(Panel(
        Text.assemble(("Does header-aware chunking beat fixed-size windows?\n", f"bold {INK}"),
                      ("Same documents, same questions, two ways of cutting them into chunks.", SUB)),
        title=Text("RAGDial demo", style=f"bold {INK}"), title_align="left",
        border_style=MUTED, box=box.ROUNDED, padding=(0, 2)))

    # 1 ── ingest
    ing = lab.ingest_corpus(args.corpus, "demo")
    det = ing.detected
    step(console, 1, "Read the documents", "Parse Markdown/text into sections, tables, code and lists.")
    console.print(Padding(Text.assemble(
        (f"{ing.file_count}", f"bold {INK}"), (" docs   ", SUB),
        (f"{ing.total_tokens:,}", f"bold {INK}"), (" tokens   ", SUB),
        (f"{det['tables'] + det['nested_tables']}", f"bold {INK}"), (" tables   ", SUB),
        (f"{det['code_blocks']}", f"bold {INK}"), (" code blocks   ", SUB),
        (f"{det['headers']}", f"bold {INK}"), (" headings", SUB)), (0, 0, 0, 4)))

    # 2 ── chunk
    before = lab.run_chunking_ablation("demo", ["fixed"], [256], [0], args.embedder)
    after = lab.run_chunking_ablation("demo", ["markdown"], [256], [0], args.embedder)
    step(console, 2, "Cut them into chunks two ways",
         "fixed = every 256 tokens, wherever that lands · markdown = along headings, tables kept whole")
    t = Table(box=box.SIMPLE_HEAD, header_style=SUB, padding=(0, 1), pad_edge=False, show_edge=False)
    t.add_column("chunker", min_width=14)
    t.add_column("chunks", justify="right")
    t.add_column("median size", justify="right")
    t.add_column("tables cut in half", min_width=30)
    t.add_column("damage", justify="right")
    for c in before.configs + after.configs:
        cut, total = (int(x) for x in c.severed.get("table", "0/0").split("/"))
        _, color = chunker(c.config_id)
        t.add_row(swatch(c.config_id), f"{c.chunks}", f"{c.tokens_p50:.0f} tok",
                  Text.assemble(bar(cut / total if total else 0, 16, color), f" {cut:>2} of {total}"),
                  f"{c.damage_score:.2f}")
    console.print(Padding(t, (0, 0, 0, 4)))
    console.print(Text("    damage: weighted share of tables, code and lists cut apart (0 = none, 1 = all)",
                       style=MUTED))

    # 3 ── show one cut
    insp = lab.inspect_boundary_splits(before.run_id, limit=1, kinds=["table"])
    step(console, 3, "See what fixed-size chunking breaks",
         "A real cut from the run above. Part of this table lands in a chunk without its header row,\n"
         "so those values no longer say which column they belong to.")
    for s in insp.splits:
        above = clip_lines(s.before_cut, 2, from_end=True)
        below = clip_lines(s.after_cut, 2, from_end=False)
        body = [Text(section_name(s.section_id), style=f"bold {INK}")]
        body += [Text.assemble(("│ ", MUTED), (ln, SUB)) for ln in above]
        body.append(Text.assemble(("✂ ", BAD), ("─" * 18, BAD),
                                  (f"  {s.left_chunk} ends · {s.right_chunk} starts", MUTED)))
        body += [Text.assemble(("│ ", MUTED), (ln, SUB)) for ln in below]
        console.print(Padding(Group(*body), (0, 0, 0, 4)))

    # 4 ── benchmark
    kw = dict(retrievers=["bm25", "dense", "hybrid"], reranker=args.reranker, top_k=k, embedder=args.embedder)
    b1 = lab.benchmark_retrieval(before.run_id, args.dataset, **kw)
    b2 = lab.benchmark_retrieval(after.run_id, args.dataset, **kw)
    step(console, 4, f"Ask {b1.queries} questions with known answers",
         f"Recall@{k}: answer found in the top {k}   MRR: how high the first hit ranks\n"
         f"nDCG@{k}: overall ranking quality, 1 = perfect   p95: search time per question")
    t = Table(box=box.SIMPLE_HEAD, header_style=SUB, padding=(0, 1), pad_edge=False, show_edge=False)
    t.add_column("chunker", min_width=14)
    t.add_column("search")
    t.add_column(f"Recall@{k}", justify="right")
    t.add_column("MRR", justify="right")
    t.add_column(f"nDCG@{k}", min_width=28)
    t.add_column("p95", justify="right")
    variants = sorted(b1.results + b2.results, key=lambda v: -v.ndcg_at_k)
    for i, v in enumerate(variants):
        _, color = chunker(v.chunk_config)
        t.add_row(swatch(v.chunk_config), RETRIEVER.get(v.retriever, v.retriever),
                  f"{v.recall_at_k:.2f}", f"{v.mrr:.2f}",
                  Text.assemble(bar(v.ndcg_at_k, 20, color), f" {v.ndcg_at_k:.3f}",
                                (" ★" if i == 0 else "", INK)),
                  f"{v.p95_ms:.1f} ms" if v.p95_ms >= 1 else f"{v.p95_ms:.2f} ms")
    console.print(Padding(t, (0, 0, 0, 4)))

    # 5 ── verdict
    d = lab.diff_experiment_runs(b1.run_id, b2.run_id)
    questions = lab.db.get_run(b2.run_id)["summary"]["questions"]
    step(console, 5, "Verdict", "Best fixed setup vs best markdown setup, compared question by question.")

    def delta_row(label: str, m) -> Text:
        return Text.assemble((f"{label:<10}", SUB), (f"{m.a:.3f}", f"bold {BASE}"), ("  ──▶  ", MUTED),
                             (f"{m.b:.3f}", f"bold {CAND}"),
                             (f"   {m.delta:+.3f}", f"bold {GOOD if m.delta > 0 else BAD if m.delta < 0 else INK}"))

    lines: list[Text] = [delta_row(f"nDCG@{k}", d.metrics["ndcg_at_k"]),
                         delta_row(f"Recall@{k}", d.metrics["recall_at_k"]), Text("")]
    lines.append(Text.assemble((f"✓ {len(d.fixed)} questions now answered", f"bold {GOOD}"), ("   ", ""),
                               (f"✗ {len(d.regressed)} broken" if d.regressed else "✓ none broken",
                                f"bold {BAD if d.regressed else GOOD}")))
    for f in d.fixed:
        q = questions.get(f.query_id, "")
        q = q if len(q) <= 72 else q[:71] + "…"
        lines.append(Text.assemble(("  ✓ ", GOOD), (f"{f.query_id}  ", MUTED), (q, SUB)))
    for r in d.regressed:
        lines.append(Text.assemble(("  ✗ ", BAD), (f"{r.query_id}  ", MUTED), (questions.get(r.query_id, ""), SUB)))
    delta = d.metrics["ndcg_at_k"].delta
    if delta > 0:
        verdict, color = f"{chunker(d.variant_b.split('|')[0])[0]} wins", CAND
    elif delta < 0:
        verdict, color = f"{chunker(d.variant_a.split('|')[0])[0]} wins", BASE
    else:
        verdict, color = "no difference", MUTED
    console.print(Padding(Panel(Group(*lines), border_style=color, box=box.ROUNDED, padding=(0, 2),
                                title=Text.assemble(("● ", color), (verdict, f"bold {INK}")),
                                title_align="left"), (0, 0, 0, 4)))

    console.print()
    console.print(Text.assemble((f"  Done in {time.perf_counter() - t0:.1f}s · try it in your agent:  ", MUTED),
                                ("claude mcp add ragdial -- uvx ragdial-mcp", SUB)))


if __name__ == "__main__":
    main()
