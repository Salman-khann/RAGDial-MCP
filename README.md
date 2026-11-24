# RAGDial MCP

<!-- mcp-name: io.github.Salman-khann/ragdial-mcp -->

[![CI](https://github.com/Salman-khann/RAGDial-MCP/actions/workflows/ci.yml/badge.svg)](https://github.com/Salman-khann/RAGDial-MCP/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ragdial-mcp)](https://pypi.org/project/ragdial-mcp/)
[![MCP Registry](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fregistry.modelcontextprotocol.io%2Fv0.1%2Fservers%2Fio.github.Salman-khann%252Fragdial-mcp%2Fversions%2Flatest&query=%24.server.version&label=MCP%20Registry&prefix=v&color=6E56CF)](https://registry.modelcontextprotocol.io/v0.1/servers?search=io.github.Salman-khann/ragdial-mcp)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![MCP SDK](https://img.shields.io/badge/MCP%20SDK-v2-6E56CF)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**RAGDial MCP is a RAG retrieval ablation & chunking lab, packaged as an MCP server.** It turns an AI coding assistant (Claude Code, Cursor, Claude Desktop) into a systematic retrieval tuner. Instead of guessing chunk sizes and overlaps, the agent runs controlled experiments, looks at exactly which tables and code blocks got cut in half, and compares pipelines on Recall@k, MRR, nDCG@k and p95 latency, with confidence intervals so small differences aren't mistaken for wins.

```
ingest_corpus → run_chunking_ablation → inspect_boundary_splits → benchmark_retrieval → diff_experiment_runs
```

Everything runs locally: no API keys and no paid services. The first run downloads tiktoken's tokenizer file (about 1.7 MB); offline, a regex tokenizer takes over and every result says which one was used. On the bundled benchmark pack, the full loop takes a few seconds on a laptop.

![ragdial-demo: chunking ablation, severed tables, retrieval benchmark and before/after diff](docs/demo.gif)

<sub>A real run of `ragdial-demo`, captured with its colours and rendered by [`scripts/make_demo_gif.py`](scripts/make_demo_gif.py). Orange = naive fixed-size chunking, green = tuned fixed-size chunking, blue = header-aware chunking.</sub>

---

## Quick start (under 2 minutes)

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) and nothing else; uv fetches Python 3.11+ if it's missing.

**Add it to Claude Code:**

```bash
claude mcp add ragdial -- uvx ragdial-mcp
```

Then ask: *"Ingest @sample, ablate fixed vs markdown chunking at 256/512 tokens, show me which tables got split, and benchmark against @sample."* The `@sample` shortcut points the tools at the bundled 13-document benchmark pack and its 30 golden questions. Point `ingest_corpus` at your own docs folder when you're ready.

**See the whole loop in your terminal, no agent needed:**

```bash
uvx --from ragdial-mcp ragdial-demo
```

**Work on the code:**

```bash
git clone https://github.com/Salman-khann/RAGDial-MCP.git
cd RAGDial-MCP
uv sync                  # core deps: MCP SDK, tiktoken, rank-bm25, numpy
uv run ragdial-demo      # full before/after loop on sample_data/ (about 4 s)
uv run pytest            # 32 deterministic tests, no downloads
```

To run the latest unreleased `main` instead of the PyPI release, use `uvx --from git+https://github.com/Salman-khann/RAGDial-MCP ragdial-mcp`.

### What the demo shows, and what it doesn't

The demo compares three chunkers on the bundled pack (30 questions, BM25, top-5):

| Chunker | Median chunk | Tables cut | nDCG@5 | 95% CI |
|---|---|---|---|---|
| fixed-256, no overlap (naive baseline) | 256 tok | 12 of 14 | 0.763 | 0.63 to 0.88 |
| fixed-512, 64-token overlap (tuned baseline) | 512 tok | 3 of 14 | 0.865 | 0.79 to 0.94 |
| markdown-256 (header-aware) | 110 tok | 5 of 14, header repeated | **0.926** | 0.86 to 0.98 |

Read it this way:

- **Against naive fixed-size chunking, header-aware chunking clearly wins:** +0.163 nDCG@5 (95% CI +0.06 to +0.28). 4 questions go from miss to hit and none go the other way, though 4-0 alone is not significant (sign test p = 0.125).
- **Against a tuned baseline, the difference is within noise:** +0.061 against fixed-512 with overlap (95% CI −0.01 to +0.14). On this pack the tool reports six variants as statistically tied with the best.
- **Where markdown's gain comes from:** its chunks are smaller (median 110 vs 256 tokens) and start with a `Doc > Section` breadcrumb. Without the breadcrumb, markdown-256 scores 0.897, so about 0.03 of the 0.163 comes from that prefix, not from where the text is split.
- **This is a demo, not evidence.** The corpus and questions are synthetic, table-heavy, and written by the same author as the chunkers. Run RAGDial on your own documents and questions before deciding anything.

To use real models (`BAAI/bge-small-en-v1.5`, the `BAAI/bge-reranker-base` cross-encoder, and ChromaDB):

```bash
uvx --from "ragdial-mcp[ml]" ragdial-demo --embedder bge-small --reranker bge-reranker-base
```

An earlier run with real models (v0.1.2, before the 0.2 metric fixes, laptop CPU, no GPU) showed the same direction: nDCG@5 0.779 for fixed-256 vs 0.950 for markdown-256 with dense retrieval and the bge reranker. The cross-encoder took 13 to 17 seconds per question on CPU, and that time grows with chunk length. That's the trade-off `generate_rag_spec` weighs against a latency budget.

## Connect it to an agent

**Claude Code:**

```bash
claude mcp add ragdial -- uvx ragdial-mcp
# with real models:
claude mcp add ragdial -- uvx --from "ragdial-mcp[ml]" ragdial-mcp
```

Inside a clone, the project-scoped [`.mcp.json`](.mcp.json) also offers the `ragdial` server automatically, running your local checkout.

**Cursor / Claude Desktop** (`mcpServers` entry):

```json
{ "ragdial": { "command": "uvx", "args": ["ragdial-mcp"] } }
```

**Remote / container** (Streamable HTTP on `http://localhost:8000/mcp`):

```bash
uv run ragdial-mcp --transport streamable-http --port 8000
# or
docker build -t ragdial-mcp . && docker run --rm -p 8000:8000 -v ragdial:/data ragdial-mcp
```

**Inspect the schemas** with the MCP Inspector:

```bash
npx @modelcontextprotocol/inspector uv run ragdial-mcp
```

Then ask the agent something like: *"Ingest sample_data/corpus, ablate fixed/recursive/markdown/semantic at 128/256/512 with 0 and 64 overlap, show me the worst table splits, benchmark against sample_data/golden_qa.jsonl and tell me which config to ship."*

---

## Local or hosted?

RAGDial does its work (parsing, chunking, embedding, ranking) in its own process, so whoever runs that process supplies the CPU, RAM and disk. The AI model only receives the compact JSON results.

**Run it locally (the default, recommended for most people).** Your MCP client starts `ragdial-mcp` on your machine over stdio, as set up in [`.mcp.json`](.mcp.json) and the quick start.

- Your documents never leave your machine. Only metric summaries and short excerpts reach the model.
- `ingest_corpus` reads any folder on your disk directly, with no upload step.
- There's no server to pay for and no account or API key to manage.
- The core mode is light: the full sample loop runs in seconds. Expect real CPU and RAM use only with the `ml` extra. Its models total about 1.3 GB on disk, and the cross-encoder reranker is slow on CPU but uses an NVIDIA GPU automatically if PyTorch finds one.

**Host it as a shared server** when one of these applies:

- **Heavy models on weak laptops:** a GPU server runs `bge-reranker-base` many times faster than a laptop CPU.
- **Teams:** everyone shares one experiment history and can diff each other's runs.
- **Large corpora:** tens of thousands of documents need more memory and time than a laptop comfortably gives.
- **Zero install for users:** people connect with a URL instead of installing uv and Python.

```bash
docker build -t ragdial-mcp:ml --build-arg EXTRAS=ml .
docker run -d -p 8000:8000 --gpus all \
  -v ragdial:/data \
  -v /srv/corpora:/corpora:ro \
  ragdial-mcp:ml
# clients connect to http://<host>:8000/mcp and ingest e.g. folder_path="/corpora/my-docs"
```

Drop `--gpus all` on a machine without an NVIDIA GPU.

Keep in mind when hosting:

- **Paths are server paths.** `ingest_corpus` reads folders on the *server*, so mount the document folders into the container (read-only, as above).
- **There is no built-in authentication.** Anyone who can reach the port can run experiments and read any folder the container can see. Keep it on a private network or behind a reverse proxy that handles auth and TLS. Never expose it directly to the internet.
- **Data in `/data` is shared** by all users of that server: corpora, runs and reports.

| | Local (stdio) | Hosted (Streamable HTTP) |
|---|---|---|
| Who pays for compute | Each user's machine | The server |
| Where documents live | User's disk | Mounted on the server |
| Setup per user | Install uv, clone or `uvx` | Paste a URL |
| Experiment history | Per user (`~/.ragdial`) | Shared (`/data`) |
| GPU for reranking | Only if the user has one | Provision once on the server |
| Access control | Your OS account | Your network or reverse proxy (none built in) |

---

## MCP surface

### Tools (model-driven actions)

| Tool | Key parameters | Returns |
|---|---|---|
| `ingest_corpus` | `folder_path`, `corpus_id` | File count, total tokens, detected headers/tables/nested tables/code/lists, document IDs |
| `run_chunking_ablation` | `corpus_id`, `strategies[]`, `chunk_sizes[]`, `overlaps[]`, `embedder` | Per config: chunk count, token p50/p95/max, **boundary-damage score**, severed/total per structure kind |
| `inspect_boundary_splits` | `run_id`, `limit=5`, `config_id`, `kinds[]` | Excerpts of about 110 characters on each side of every cut through a table, code fence, list or paragraph |
| `benchmark_retrieval` | `run_id`, `dataset_path`, `retrievers[]`, `reranker`, `top_k`, `configs[]`, `embedder`, `vector_store` | Recall@k, MRR, nDCG@k, hit rate, p50/p95 latency per *chunk config × retriever*, plus the 3 worst queries |
| `diff_experiment_runs` | `run_a`, `run_b`, `variant_a`, `variant_b` | Metric deltas (B − A), query IDs that flipped miss→hit or regressed, and rank-movement counts |

### Resources (read-only context)

| URI | Content |
|---|---|
| `ragdial://experiments` | Index of corpora and runs with configs, timestamps and headline metrics |
| `ragdial://experiments/{run_id}/report` | Full JSON: every variant and the per-query hit/miss table with top chunk IDs |
| `ragdial://corpus/{corpus_id}/stats` | Document metadata, token counts and structural breakdown |

### Prompts (user-triggered workflows)

| Prompt | What it makes the agent do |
|---|---|
| `diagnose_retrieval_failures(run_id)` | Pull the worst queries from the report, run `inspect_boundary_splits`, classify each miss (boundary damage / ranking / vocabulary / dilution), test **one** targeted fix and confirm it with `diff_experiment_runs` |
| `generate_rag_spec(corpus_id, latency_budget_ms)` | Drop variants over the p95 budget, treat variants statistically tied on nDCG as equal and choose among them by latency and simplicity, never recommend dense/hybrid from `hash`-embedder runs, and emit a YAML production spec that cites the supporting run IDs and confidence intervals |

### Context-window discipline, by design

Each tool returns a **typed Pydantic model** (exposed as an MCP `outputSchema` with `structuredContent`) holding metric tables, IDs and a `next_step` hint. **No tool returns chunk text.** The only text that reaches the model is the clipped excerpts from `inspect_boundary_splits`, at most 25 of them. Full per-query detail sits behind `ragdial://experiments/{run_id}/report`, and the agent reads it only for the queries it is investigating. A 48-config ablation costs a few KB of context, not a few MB.

---

## How it measures things

### Chunkers ([`chunking/strategies.py`](src/ragdial_mcp/chunking/strategies.py))

Each chunker records the exact source span (`start`/`end` character offsets) of every chunk, which is what makes the boundary audit exact.

| Strategy | Behaviour |
|---|---|
| `fixed` | Sliding windows of exactly `size` tiktoken tokens, `overlap` tokens shared |
| `recursive` | Recursive splitting on paragraph → line → sentence → clause → word, greedily merged up to `size` |
| `markdown` | Header-aware: never crosses a section, keeps tables/code/lists atomic when they fit, splits oversized tables (including tables nested in lists) by rows **with the header row replicated**, prefixes every chunk with its breadcrumb (`Doc > Section > Subsection`). `size` includes the breadcrumb. Sections are often smaller than `size`, so markdown chunks are usually much smaller than the cap |
| `semantic` | Embeds sentences/blocks and breaks where adjacent similarity falls in the bottom quartile, capped at `size` |

`size` is a cap. Chunk text re-tokenized on its own can land a token or two above it, because the boundaries of a token window rarely align exactly with how the substring tokenizes.

**Overlap:** `fixed` shares exactly `overlap` tokens. The other strategies carry whole trailing sentences or blocks, and `markdown` never carries across a section boundary, so a requested overlap can have little or no effect. Every config reports `measured_overlap` (tokens actually shared by neighbouring chunks), and the ablation adds a note when overlap did nothing.

### Boundary damage ([`chunking/boundary_audit.py`](src/ragdial_mcp/chunking/boundary_audit.py))

A structure is **severed** when no single chunk contains it whole, so overlap that keeps a full copy somewhere rescues it. A structure with text in no chunk at all is **lost**; that always counts as full damage, so a chunker can't score well by dropping content. The damage score is the weighted share of severed structures (tables and code ×3, lists ×1.5, paragraphs ×1). A severed table counts as **mitigated** (half weight) only when every piece carries *that table's own* header row; a matching header from a different table doesn't count. Every severed item also records whether it was *oversized*, meaning bigger than the chunk size and so impossible to keep intact.

Damage measures exactly what header-aware chunking is designed to avoid, so the lowest-damage config is usually a `markdown` one. Low damage doesn't guarantee better retrieval; confirm with `benchmark_retrieval`.

### Retrieval ([`retrieval/hybrid.py`](src/ragdial_mcp/retrieval/hybrid.py))

* `bm25`: Okapi BM25 via `rank-bm25`
* `dense`: cosine similarity in an **ephemeral ChromaDB collection per experiment** (or an in-memory NumPy index without the `ml` extra)
* `hybrid`: **Reciprocal Rank Fusion** (k = 60) of the top 50 from BM25 and the top 50 from dense
* optional cross-encoder reranking (`bge-reranker-base`) of the top 20 fused candidates

Latency is measured per query (query embedding + search + rerank) after a warm-up query, and reported as p50/p95. Each question is timed once, so with 30 questions p95 is roughly the second-slowest query. Latency also scales with chunk count (BM25 scores every chunk) and chunk length (the reranker reads whole chunks), so it reflects chunking as much as the retriever.

### Ground truth that can't be fooled by chunking ([`retrieval/metrics.py`](src/ragdial_mcp/retrieval/metrics.py))

Golden queries reference **evidence snippets quoted verbatim from the corpus**, not chunk IDs. Chunk IDs change with every config, so they can't serve as ground truth. A chunk is relevant only if it contains *every* snippet of an evidence group:

```json
{"id": "q09", "question": "What does extra storage cost per GB-month on the Business plan?",
 "evidence": ["| Feature | Developer | Starter | Growth | Business | Enterprise |",
              "| Storage overage per GB-month | n/a | $0.10 | $0.08 | $0.05 | custom |"]}
```

A chunk holding the price row without the column header can't say which plan `$0.05` belongs to, and the metric scores it as a miss. Splitting a table therefore shows up as a measurable retrieval failure. Two caveats:

* **The header must be unique.** If several tables share the same header row, requiring it proves nothing, because any of those tables' headers would match. The bundled set only requires headers that appear once in the corpus, and a test enforces that.
* **The requirement favours header replication.** Only a chunker that repeats headers (or keeps whole tables) can meet it, so use it where the header carries meaning (wide pricing or plan tables), not for self-describing key/value rows.

Queries can also carry several `evidence_groups` for answers that span documents.

* **Recall@k**: share of evidence groups found in the top k. With one group per question (28 of the 30 bundled ones), this is the hit rate.
* **MRR**: 1 / rank of the first relevant chunk, searched at least 10 deep (MRR@10)
* **nDCG@k**: gain counts each evidence group once, at the rank where it's first covered. "Ranked #1" beats "ranked #5", and one chunk that answers everything scores 1.0.

### Statistics

Golden sets are small, so every benchmark reports a bootstrap 95% confidence interval on nDCG@k per variant, plus `tied_with_best`: the variants not distinguishable from the top one. `diff_experiment_runs` adds a paired CI on the nDCG@k change, a sign test on fixed vs regressed questions, and a `significant` flag. It refuses to compare runs with different `top_k`.

Results computed with the `hash` embedder carry a warning not to choose dense or hybrid retrieval from them.

---

## Benchmark pack

[`sample_data/corpus/`](sample_data/corpus) holds 13 documents for a fictional event-streaming platform: about 10k tokens, 14 tables (two nested inside numbered list items), 16 code fences in Python/Bash/TOML/JSON/diff, and Markdown plus plain text. They include a config reference with 4 wide tables, a rate-limit and quota guide, an error catalogue, an SDK guide, a Kubernetes runbook, pricing, security, a migration guide, a changelog, a postmortem and a plain-text FAQ.

[`sample_data/golden_qa.jsonl`](sample_data/golden_qa.jsonl) holds 30 questions:

- **14 about tables:** 7 require a unique header row and the data row in the same chunk (tagged `needs-header`), and 7 need only the row.
- **5 about code:** 4 need two lines of the same block, 1 needs a single line.
- **9 about prose or lists.**
- **2 cross-document questions.**

Tests check that every evidence snippet exists verbatim in the corpus, and that every required header row is unique.

To use your own data, point `ingest_corpus` at any folder of `.md` / `.txt` / `.pdf` files (PDFs need `uv sync --extra pdf`, which converts them with `pymupdf4llm` and keeps tables) and write a JSONL file in the same format.

## Embedders

| `embedder=` | What it is | When |
|---|---|---|
| `hash` | Deterministic feature hashing of unigrams + bigrams (2048-d, sublinear TF) | Tests, CI, offline demo: identical numbers on every machine with the same tokenizer backend |
| `bge-small` | `BAAI/bge-small-en-v1.5` via sentence-transformers, with the BGE query instruction | Real dense retrieval (`ml` extra) |
| any HF id | Any sentence-transformers model | Your own comparisons |
| `auto` (default) | `bge-small` if the `ml` extra is installed, otherwise `hash` | |

The `hash` embedder is a lexical proxy, so on it "dense" is close to BM25 and usually weaker. Draw conclusions about dense vs hybrid from `bge-small` runs. Every run records the embedder and vector store it used.

---

## Project layout

```text
RAGDial-MCP/
├── pyproject.toml              # uv project, extras: ml, pdf; entry points
├── Dockerfile                  # Streamable HTTP server on :8000/mcp
├── .mcp.json                   # Claude Code project config
├── server.json                 # MCP Registry listing (io.github.Salman-khann/ragdial-mcp)
├── docs/demo.gif               # README demo, rendered from a real run
├── scripts/make_demo_gif.py    # re-records the demo GIF (uv run --with pillow ...)
├── .github/workflows/
│   ├── ci.yml                  # tests on Linux/Windows/macOS, wheel check, Docker HTTP smoke test
│   └── release.yml             # GitHub release -> build, verify, publish to PyPI
├── sample_data/
│   ├── corpus/                 # 13 technical docs (tables, nested tables, code)
│   └── golden_qa.jsonl         # 30 Q&A pairs with verbatim evidence
├── src/ragdial_mcp/
│   ├── server.py               # MCP tools, resources, prompts (thin adapter)
│   ├── lab.py                  # evaluation engine: no MCP imports
│   ├── models.py               # compact Pydantic tool outputs
│   ├── demo.py                 # `ragdial-demo` CLI
│   ├── samples.py              # locates the bundled pack ("@sample")
│   ├── tokens.py               # tiktoken spans (regex fallback offline)
│   ├── ingestion/parser.py     # MD/TXT/PDF → sections, blocks, structures w/ offsets
│   ├── chunking/strategies.py  # fixed, recursive, markdown, semantic
│   ├── chunking/boundary_audit.py
│   ├── retrieval/embedders.py  # hash / sentence-transformers
│   ├── retrieval/hybrid.py     # BM25, Chroma/NumPy dense, RRF, cross-encoder
│   ├── retrieval/metrics.py    # Recall@k, MRR, nDCG@k, percentiles
│   └── storage/db.py           # SQLite run registry
└── tests/
    ├── test_chunkers.py        # offsets, size caps, header replication, audit, golden validity
    ├── test_metrics.py         # metric edge cases, RRF, searcher
    └── test_mcp_tools.py       # end-to-end through a real MCP client session
```

**The engine is decoupled from the server.** `lab.Lab` is plain Python that returns Pydantic models, and `server.py` is a thin layer of decorators over it. The same engine already drives the `ragdial-demo` CLI and could back a side-by-side web dashboard without duplicating logic.

Runs persist in SQLite at `$RAGDIAL_HOME/ragdial.db` (default `~/.ragdial`), so experiments survive restarts and `diff_experiment_runs` works across sessions.

## Notes and limits

* Built on the official Python MCP SDK **v2** (`mcp.server.mcpserver.MCPServer`, the successor to v1's `FastMCP`).
* Token counts use tiktoken `cl100k_base`. If its BPE file can't be downloaded on first run, a regex tokenizer takes over and the backend name is reported in every result.
* The semantic chunker's quality depends on the embedder. With `hash`, it mostly finds lexical topic shifts.
* The sample pack is Markdown and plain text only. PDF ingestion works through the `pdf` extra, but multi-column PDF layouts depend on `pymupdf4llm`'s reconstruction.

## Releasing

1. Bump `__version__` in [`src/ragdial_mcp/__init__.py`](src/ragdial_mcp/__init__.py) and both `version` fields in [`server.json`](server.json), then push. A test fails if they disagree.
2. Publish a GitHub release tagged `v<version>` (for example `v0.1.1`).
3. [`release.yml`](.github/workflows/release.yml) checks that the tag matches the version, runs the tests, builds the sdist and wheel, runs the wheel in isolation, and uploads it to PyPI with trusted publishing (no API token stored).
4. It then lists the new version in the [official MCP Registry](https://registry.modelcontextprotocol.io/v0.1/servers?search=io.github.Salman-khann/ragdial-mcp) as `io.github.Salman-khann/ragdial-mcp`, authenticating with GitHub OIDC. The registry verifies PyPI ownership through the `mcp-name` comment at the top of this README.

## License

MIT
