"""Sparse, dense and hybrid retrieval with optional cross-encoder reranking.

``bm25``    - Okapi BM25 over lowercase word tokens (rank-bm25)
``dense``   - cosine similarity over embeddings, in ChromaDB (ephemeral
              collection per experiment) or an in-memory NumPy index
``hybrid``  - Reciprocal Rank Fusion of the BM25 and dense rankings
              (score = sum 1 / (rrf_k + rank), rrf_k = 60)

Any of them can feed a cross-encoder reranker (``bge-reranker-base``), which
rescores the top ``rerank_depth`` fused candidates.
"""

from __future__ import annotations

import time
import uuid
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from .embedders import Embedder, words

RETRIEVERS = ("bm25", "dense", "hybrid")
RERANKER_ALIASES = {
    "bge-reranker-base": "BAAI/bge-reranker-base",
    "bge-reranker": "BAAI/bge-reranker-base",
    "cross-encoder": "BAAI/bge-reranker-base",
}
RRF_K = 60


class VectorIndex(Protocol):
    name: str

    def search(self, query_vec: np.ndarray, n: int) -> list[int]: ...

    def close(self) -> None: ...


class NumpyIndex:
    name = "numpy"

    def __init__(self, vectors: np.ndarray) -> None:
        self._m = vectors.astype(np.float32)

    def search(self, query_vec: np.ndarray, n: int) -> list[int]:
        if not len(self._m):
            return []
        scores = self._m @ query_vec
        n = min(n, len(scores))
        top = np.argpartition(-scores, n - 1)[:n]
        return [int(i) for i in top[np.argsort(-scores[top], kind="stable")]]

    def close(self) -> None:
        pass


class ChromaIndex:  # pragma: no cover - needs the ml extra
    name = "chroma"

    def __init__(self, vectors: np.ndarray) -> None:
        import chromadb
        from chromadb.config import Settings

        self._client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
        self._name = f"ragdial-{uuid.uuid4().hex[:12]}"
        self._col = self._client.create_collection(self._name, metadata={"hnsw:space": "cosine"})
        ids = [str(i) for i in range(len(vectors))]
        for s in range(0, len(ids), 4000):
            self._col.add(ids=ids[s:s + 4000], embeddings=vectors[s:s + 4000].tolist())

    def search(self, query_vec: np.ndarray, n: int) -> list[int]:
        n = min(n, self._col.count())
        if n <= 0:
            return []
        res = self._col.query(query_embeddings=[query_vec.tolist()], n_results=n)
        return [int(i) for i in res["ids"][0]]

    def close(self) -> None:
        try:
            self._client.delete_collection(self._name)
        except Exception:
            pass


def chroma_available() -> bool:
    try:
        import chromadb  # noqa: F401
    except ImportError:
        return False
    return True


def make_vector_index(vectors: np.ndarray, store: str = "auto") -> VectorIndex:
    if store == "chroma" or (store == "auto" and chroma_available()):
        if not chroma_available():
            raise ValueError("vector_store='chroma' needs chromadb (uv sync --extra ml)")
        return ChromaIndex(vectors)
    if store not in ("auto", "numpy"):
        raise ValueError(f"Unknown vector_store '{store}'. Use auto, numpy or chroma.")
    return NumpyIndex(vectors)


class CrossEncoderReranker:  # pragma: no cover - needs the ml extra + model download
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self.name = RERANKER_ALIASES.get(model_name, model_name)
        self._model = CrossEncoder(self.name)

    def rerank(self, query: str, texts: list[str]) -> list[int]:
        if not texts:
            return []
        scores = self._model.predict([(query, t) for t in texts], show_progress_bar=False)
        return [int(i) for i in np.argsort(-np.asarray(scores), kind="stable")]


_RERANKERS: dict[str, CrossEncoderReranker] = {}


def get_reranker(name: str | None) -> CrossEncoderReranker | None:
    if not name or name.lower() == "none":
        return None
    key = RERANKER_ALIASES.get(name, name)
    if key not in _RERANKERS:
        try:
            import sentence_transformers  # noqa: F401
        except ImportError as exc:
            raise ValueError(
                f"Reranker '{name}' needs sentence-transformers (uv sync --extra ml), or pass reranker='none'."
            ) from exc
        _RERANKERS[key] = CrossEncoderReranker(key)
    return _RERANKERS[key]


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return [i for i, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


class ChunkSearcher:
    """Indexes one chunk set once and serves bm25 / dense / hybrid queries over it."""

    def __init__(self, texts: list[str], embedder: Embedder, vector_store: str = "auto",
                 candidate_depth: int = 50) -> None:
        self.texts = texts
        self.embedder = embedder
        self.depth = candidate_depth
        self._bm25 = BM25Okapi([words(t) or ["_"] for t in texts]) if texts else None
        self._dense: VectorIndex | None = None
        self._vector_store = vector_store

    @property
    def vector_store(self) -> str:
        return self._dense.name if self._dense else "unbuilt"

    def _dense_index(self) -> VectorIndex:
        if self._dense is None:
            self._dense = make_vector_index(self.embedder.embed_documents(self.texts), self._vector_store)
        return self._dense

    def warm(self, retrievers: list[str]) -> None:
        if any(r in ("dense", "hybrid") for r in retrievers):
            self._dense_index()

    def _bm25_rank(self, query: str, n: int) -> list[int]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(words(query))
        n = min(n, len(scores))
        top = np.argpartition(-scores, n - 1)[:n]
        return [int(i) for i in top[np.argsort(-scores[top], kind="stable")]]

    def _dense_rank(self, query: str, n: int) -> list[int]:
        return self._dense_index().search(self.embedder.embed_query(query), n)

    def search(self, query: str, retriever: str, top_k: int,
               reranker: CrossEncoderReranker | None = None, rerank_depth: int = 20) -> tuple[list[int], float]:
        """Return (ranked chunk indexes, latency in ms)."""
        t0 = time.perf_counter()
        # always rank at least 10 deep so MRR is comparable across top_k settings
        n = max(top_k, 10, rerank_depth if reranker else 0)
        if retriever == "bm25":
            ranked = self._bm25_rank(query, n)
        elif retriever == "dense":
            ranked = self._dense_rank(query, n)
        elif retriever == "hybrid":
            ranked = rrf([self._bm25_rank(query, self.depth), self._dense_rank(query, self.depth)])[:n]
        else:
            raise ValueError(f"Unknown retriever '{retriever}'. Choose from {list(RETRIEVERS)}")
        if reranker is not None:
            head = ranked[:rerank_depth]
            order = reranker.rerank(query, [self.texts[i] for i in head])
            ranked = [head[i] for i in order] + ranked[rerank_depth:]
        return ranked[:max(top_k, 10)], (time.perf_counter() - t0) * 1000.0

    def close(self) -> None:
        if self._dense is not None:
            self._dense.close()
