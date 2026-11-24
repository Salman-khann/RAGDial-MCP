"""``ragdial-demo``: the full before/after loop on the bundled benchmark pack, in seconds.

    uvx --from ragdial-mcp ragdial-demo        # offline, deterministic (hash embedder, BM25/RRF)
    uvx --from "ragdial-mcp[ml]" ragdial-demo --embedder bge-small --reranker bge-reranker-base

The demo compares header-aware chunking against two fixed-size baselines (a
naive one and a tuned one) and reports confidence intervals, so it shows
where the difference is real and where it is noise.

Output design: three entity colours (orange = naive fixed, aqua = tuned
fixed, blue = header-aware; validated all-pairs for colour-vision deficiency
on the dark surface), status colours only for outcomes and always paired with
an icon, numbers in neutral ink.
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

BASE = "#d95926"      # naive fixed-size chunking
TUNED = "#199e70"     # tuned fixed-size chunking (bigger chunks + overlap)
CAND = "#3987e5"      # header-aware markdown chunking
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


NAIVE_CFG, TUNED_CFG, MD_CFG = "fixed-256-o0", "fixed-512-o64", "markdown-256-o0"
SERIES = {NAIVE_CFG: ("fixed-256", BASE), TUNED_CFG: ("fixed-512 +64", TUNED), MD_CFG: ("markdown-256", CAND)}


def chunker(config_id: str) -> tuple[str, str]:
    """(label, colour) for one of the demo's three chunk configs."""
    return SERIES.get(config_id, (config_id.rsplit("-o", 1)[0], MUTED))


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
                      ("Same documents and questions; a naive and a tuned fixed-size baseline.", SUB)),
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
    naive = lab.run_chunking_ablation("demo", ["fixed"], [256], [0], args.embedder)
    tuned = lab.run_chunking_ablation("demo", ["fixed"], [512], [64], args.embedder)
    md = lab.run_chunking_ablation("demo", ["markdown"], [256], [0], args.embedder)
    step(console, 2, "Cut them into chunks three ways",
         "fixed-256 = every 256 tokens, wherever that lands · fixed-512 +64 = bigger, overlapping\n"
         "markdown-256 = along headings, tables kept whole or split with their header row repeated")
    t = Table(box=box.SIMPLE_HEAD, header_style=SUB, padding=(0, 1), pad_edge=False, show_edge=False)
    t.add_column("chunker", min_width=15)
    t.add_column("chunks", justify="right")
    t.add_column("median size", justify="right")
    t.add_column("tables cut in half", min_width=30)
    t.add_column("damage", justify="right")
    for c in naive.configs + tuned.configs + md.configs:
        cut, total = (int(x) for x in c.severed.get("table", "0/0").split("/"))
        _, color = chunker(c.config_id)
        t.add_row(swatch(c.config_id), f"{c.chunks}", f"{c.tokens_p50:.0f} tok",
                  Text.assemble(bar(cut / total if total else 0, 16, color), f" {cut:>2} of {total}"),
                  f"{c.damage_score:.2f}")
    console.print(Padding(t, (0, 0, 0, 4)))
    console.print(Text("    damage: weighted share of tables, code and lists cut apart (0 = none, 1 = all)",
                       style=MUTED))

    # 3 ── show one cut
    insp = lab.inspect_boundary_splits(naive.run_id, limit=1, kinds=["table"])
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
    # BM25 only with the offline hash embedder: hash "dense" is a lexical stand-in and would mislead
    retrievers = ["bm25"] if args.embedder.startswith("hash") else ["bm25", "dense", "hybrid"]
    kw = dict(retrievers=retrievers, reranker=args.reranker, top_k=k, embedder=args.embedder)
    bn = lab.benchmark_retrieval(naive.run_id, args.dataset, **kw)
    bt = lab.benchmark_retrieval(tuned.run_id, args.dataset, **kw)
    bm = lab.benchmark_retrieval(md.run_id, args.dataset, **kw)
    step(console, 4, f"Ask {bn.queries} questions with known answers",
         f"Recall@{k}: answer found in the top {k}   MRR: how high the first hit ranks\n"
         f"nDCG@{k}: overall ranking quality, 1 = perfect; the range is its 95% confidence interval")
    t = Table(box=box.SIMPLE_HEAD, header_style=SUB, padding=(0, 1), pad_edge=False, show_edge=False)
    t.add_column("chunker", min_width=15)
    t.add_column("search")
    t.add_column(f"Recall@{k}", justify="right")
    t.add_column("MRR", justify="right")
    t.add_column(f"nDCG@{k}  (95% CI)", min_width=36)
    variants = sorted(bn.results + bt.results + bm.results, key=lambda v: -v.ndcg_at_k)
    for v in variants:
        _, color = chunker(v.chunk_config)
        lo, hi = v.ndcg_ci95
        t.add_row(swatch(v.chunk_config), RETRIEVER.get(v.retriever, v.retriever),
                  f"{v.recall_at_k:.2f}", f"{v.mrr:.2f}",
                  Text.assemble(bar(v.ndcg_at_k, 16, color), f" {v.ndcg_at_k:.3f}", (f"  ({lo:.2f}–{hi:.2f})", MUTED)))
    console.print(Padding(t, (0, 0, 0, 4)))
    if args.embedder.startswith("hash"):
        console.print(Text("    BM25 only: dense/hybrid need a real embedder (--embedder bge-small, ml extra)",
                           style=MUTED))

    # 5 ── verdict: markdown vs each baseline, with uncertainty
    questions = lab.db.get_run(bm.run_id)["summary"]["questions"]
    step(console, 5, "Verdict", "Best markdown setup vs each baseline, compared question by question.")
    lines: list[Text] = []
    real = {}
    for label, base_run, color in (("vs naive", bn, BASE), ("vs tuned", bt, TUNED)):
        d = lab.diff_experiment_runs(base_run.run_id, bm.run_id)
        m, (lo, hi) = d.metrics["ndcg_at_k"], d.ndcg_delta_ci95
        real[label] = d
        status = (("✓ real difference", GOOD) if d.significant and m.delta > 0 else
                  ("✗ markdown is worse", BAD) if d.significant else ("≈ within noise", SUB))
        lines.append(Text.assemble((f"{label:<9}", SUB), (f"{m.a:.3f}", f"bold {color}"), (" ──▶ ", MUTED),
                                   (f"{m.b:.3f}", f"bold {CAND}"), (f"  {m.delta:+.3f}", f"bold {INK}"),
                                   (f"  CI {lo:+.2f} to {hi:+.2f}  ", MUTED), (status[0], f"bold {status[1]}")))
    d = real["vs naive"]
    lines.append(Text(""))
    lines.append(Text.assemble((f"vs naive: {len(d.fixed)} questions now answered, "
                                f"{len(d.regressed)} broken", f"bold {INK}"),
                               (f"  (sign test p={d.sign_test_p:.2f})", MUTED)))
    for f in d.fixed:
        q = questions.get(f.query_id, "")
        q = q if len(q) <= 70 else q[:69] + "…"
        lines.append(Text.assemble(("  ✓ ", GOOD), (f"{f.query_id}  ", MUTED), (q, SUB)))
    for r in d.regressed:
        lines.append(Text.assemble(("  ✗ ", BAD), (f"{r.query_id}  ", MUTED), (questions.get(r.query_id, ""), SUB)))
    naive_win = real["vs naive"].significant and real["vs naive"].metrics["ndcg_at_k"].delta > 0
    tuned_win = real["vs tuned"].significant and real["vs tuned"].metrics["ndcg_at_k"].delta > 0
    if naive_win and tuned_win:
        title, color = "markdown-256 beats both baselines", CAND
    elif naive_win:
        title, color = "beats naive chunking; tied with a tuned baseline", CAND
    else:
        title, color = "no clear winner on this data", MUTED
    console.print(Padding(Panel(Group(*lines), border_style=color, box=box.ROUNDED, padding=(0, 2),
                                title=Text.assemble(("● ", color), (title, f"bold {INK}")),
                                title_align="left"), (0, 0, 0, 4)))

    console.print()
    console.print(Text("  Synthetic 13-doc demo pack: run RAGDial on your own documents before deciding anything.",
                       style=MUTED))
    console.print(Text.assemble((f"  Done in {time.perf_counter() - t0:.1f}s · try it in your agent:  ", MUTED),
                                ("claude mcp add ragdial -- uvx ragdial-mcp", SUB)))


if __name__ == "__main__":
    main()
