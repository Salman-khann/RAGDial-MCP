"""Dense embedders.

* ``hash``      - deterministic feature-hashing embedder (unigrams + bigrams,
  sublinear TF, L2-normalised). No downloads, no GPU, identical results on
  every machine, which makes it the default for tests and the offline demo.
* ``bge-small`` - ``BAAI/bge-small-en-v1.5`` via sentence-transformers
  (``ml`` extra). Any other sentence-transformers model id also works.
* ``auto``      - ``bge-small`` when sentence-transformers is installed,
  otherwise ``hash``.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from functools import lru_cache
from typing import Protocol

import numpy as np

_WORD = re.compile(r"[a-z0-9_]+")

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
MODEL_ALIASES = {
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "bge-base": "BAAI/bge-base-en-v1.5",
}


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@lru_cache(maxsize=200_000)
def _bucket(feature: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    return value % dim, (1.0 if (value >> 63) & 1 else -1.0)


class HashEmbedder:
    def __init__(self, dim: int = 2048) -> None:
        self.dim = dim
        self.name = f"hash-{dim}"
        self._cache: dict[str, np.ndarray] = {}

    def _embed(self, text: str) -> np.ndarray:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        toks = words(text)
        feats = Counter(toks)
        feats.update(f"{a} {b}" for a, b in zip(toks, toks[1:]))
        vec = np.zeros(self.dim, dtype=np.float32)
        for feat, tf in feats.items():
            idx, sign = _bucket(feat, self.dim)
            vec[idx] += sign * (1.0 + math.log(tf))
        norm = float(np.linalg.norm(vec))
        if norm:
            vec /= norm
        self._cache[text] = vec
        return vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._embed(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text)


class SentenceTransformerEmbedder:  # pragma: no cover - needs the ml extra + model download
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = MODEL_ALIASES.get(model_name, model_name)
        self.name = self.model_name
        self._model = SentenceTransformer(self.model_name)
        self._query_prefix = BGE_QUERY_INSTRUCTION if "bge" in self.model_name.lower() else ""
        self._cache: dict[str, np.ndarray] = {}

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            vecs = self._model.encode(missing, batch_size=32, normalize_embeddings=True,
                                      show_progress_bar=False, convert_to_numpy=True)
            self._cache.update(zip(missing, vecs.astype(np.float32)))
        if not texts:
            return np.zeros((0, self._model.get_sentence_embedding_dimension()), dtype=np.float32)
        return np.vstack([self._cache[t] for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._model.encode(self._query_prefix + text, normalize_embeddings=True,
                                  show_progress_bar=False, convert_to_numpy=True).astype(np.float32)


def sentence_transformers_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


_EMBEDDERS: dict[str, Embedder] = {}


def get_embedder(name: str = "auto") -> Embedder:
    """Return a cached embedder instance. See module docstring for names."""
    key = name
    if name == "auto":
        key = "bge-small" if sentence_transformers_available() else "hash"
    if key not in _EMBEDDERS:
        if key == "hash" or key.startswith("hash-"):
            suffix = key.split("-", 1)[1] if "-" in key else "2048"
            if not suffix.isdigit() or not 16 <= int(suffix) <= 65536:
                raise ValueError(f"Embedder '{name}': use 'hash' or 'hash-<dim>' with dim between 16 and 65536.")
            _EMBEDDERS[key] = HashEmbedder(int(suffix))
        else:
            if not sentence_transformers_available():
                raise ValueError(
                    f"Embedder '{name}' needs sentence-transformers. Install the ml extra "
                    "(uv sync --extra ml) or use embedder='hash'."
                )
            _EMBEDDERS[key] = SentenceTransformerEmbedder(key)
    return _EMBEDDERS[key]
