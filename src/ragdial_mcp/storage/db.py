"""SQLite experiment registry.

Tables
------
corpora        one row per ingested corpus (parsed documents stored as JSON)
runs           chunking ablations and retrieval benchmarks (config + summary)
chunk_sets     the chunks + boundary audit for each config of a chunking run
query_results  per-query metrics for each variant of a benchmark run
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS corpora (
    corpus_id   TEXT PRIMARY KEY,
    folder      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    stats_json  TEXT NOT NULL,
    docs_json   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id         TEXT PRIMARY KEY,
    kind           TEXT NOT NULL CHECK (kind IN ('chunking', 'benchmark')),
    corpus_id      TEXT NOT NULL REFERENCES corpora(corpus_id),
    parent_run_id  TEXT REFERENCES runs(run_id),
    created_at     TEXT NOT NULL,
    config_json    TEXT NOT NULL,
    summary_json   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunk_sets (
    run_id       TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    config_id    TEXT NOT NULL,
    chunks_json  TEXT NOT NULL,
    audit_json   TEXT NOT NULL,
    PRIMARY KEY (run_id, config_id)
);
CREATE TABLE IF NOT EXISTS query_results (
    run_id       TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    variant_id   TEXT NOT NULL,
    query_id     TEXT NOT NULL,
    recall       REAL NOT NULL,
    rr           REAL NOT NULL,
    ndcg         REAL NOT NULL,
    first_rank   INTEGER,
    latency_ms   REAL NOT NULL,
    top_chunks   TEXT NOT NULL,
    PRIMARY KEY (run_id, variant_id, query_id)
);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ExperimentDB:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------ corpora
    def save_corpus(self, corpus_id: str, folder: str, stats: dict[str, Any], docs: list[dict[str, Any]]) -> None:
        with self._lock, self._conn:
            # re-ingesting a corpus id replaces it and its experiments
            old = [r["run_id"] for r in self._conn.execute(
                "SELECT run_id FROM runs WHERE corpus_id = ? ORDER BY created_at DESC", (corpus_id,))]
            for run_id in old:
                self._conn.execute("UPDATE runs SET parent_run_id = NULL WHERE parent_run_id = ?", (run_id,))
            self._conn.execute("DELETE FROM runs WHERE corpus_id = ?", (corpus_id,))
            self._conn.execute(
                "INSERT OR REPLACE INTO corpora VALUES (?, ?, ?, ?, ?)",
                (corpus_id, folder, utcnow(), json.dumps(stats), json.dumps(docs)),
            )

    def get_corpus(self, corpus_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM corpora WHERE corpus_id = ?", (corpus_id,)).fetchone()
        if row is None:
            return None
        return {"corpus_id": row["corpus_id"], "folder": row["folder"], "created_at": row["created_at"],
                "stats": json.loads(row["stats_json"]), "docs": json.loads(row["docs_json"])}

    def list_corpora(self) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT corpus_id, folder, created_at FROM corpora ORDER BY created_at").fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------ runs
    def save_run(self, run_id: str, kind: str, corpus_id: str, parent_run_id: str | None,
                 config: dict[str, Any], summary: dict[str, Any]) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, kind, corpus_id, parent_run_id, utcnow(), json.dumps(config), json.dumps(summary)),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return self._run_dict(row) if row else None

    def list_runs(self, limit: int = 200) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM runs ORDER BY created_at DESC, rowid DESC LIMIT ?",
                                  (limit,)).fetchall()
        return [self._run_dict(r) for r in rows]

    @staticmethod
    def _run_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {"run_id": row["run_id"], "kind": row["kind"], "corpus_id": row["corpus_id"],
                "parent_run_id": row["parent_run_id"], "created_at": row["created_at"],
                "config": json.loads(row["config_json"]), "summary": json.loads(row["summary_json"])}

    # ------------------------------------------------------------ chunk sets
    def save_chunk_set(self, run_id: str, config_id: str, chunks: list[dict[str, Any]], audit: dict[str, Any]) -> None:
        with self._lock, self._conn:
            self._conn.execute("INSERT OR REPLACE INTO chunk_sets VALUES (?, ?, ?, ?)",
                               (run_id, config_id, json.dumps(chunks), json.dumps(audit)))

    def get_chunk_set(self, run_id: str, config_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
        row = self._conn.execute("SELECT chunks_json, audit_json FROM chunk_sets WHERE run_id = ? AND config_id = ?",
                                 (run_id, config_id)).fetchone()
        return (json.loads(row["chunks_json"]), json.loads(row["audit_json"])) if row else None

    def get_audit(self, run_id: str, config_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT audit_json FROM chunk_sets WHERE run_id = ? AND config_id = ?",
                                 (run_id, config_id)).fetchone()
        return json.loads(row["audit_json"]) if row else None

    # ------------------------------------------------------------ query results
    def save_query_results(self, run_id: str, rows: list[tuple]) -> None:
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT INTO query_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(run_id, *r[:7], json.dumps(r[7])) for r in rows],
            )

    def get_query_results(self, run_id: str, variant_id: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM query_results WHERE run_id = ?"
        args: tuple = (run_id,)
        if variant_id is not None:
            sql += " AND variant_id = ?"
            args += (variant_id,)
        rows = self._conn.execute(sql + " ORDER BY variant_id, query_id", args).fetchall()
        return [{"variant_id": r["variant_id"], "query_id": r["query_id"], "recall": r["recall"], "rr": r["rr"],
                 "ndcg": r["ndcg"], "first_rank": r["first_rank"], "latency_ms": r["latency_ms"],
                 "top_chunks": json.loads(r["top_chunks"])} for r in rows]
