"""Retrieval metrics against evidence-based ground truth.

Each golden query lists one or more **evidence groups**. A group is a list of
exact snippets from the corpus that must all appear in the *same* chunk for
that chunk to count as relevant. A table question can therefore require the
header row and the data row together, which is how chunk boundary damage
shows up as a retrieval miss rather than a silent pass.

* ``recall@k`` - share of evidence groups covered within the top ``k``
* ``mrr``      - 1 / rank of the first relevant chunk (0 if none in the ranking)
* ``ndcg@k``   - binary-gain nDCG where each evidence group can contribute once,
  so "found somewhere in the top 10" scores lower than "ranked #1"
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

import numpy as np

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def groups_covered(chunk_text_norm: str, groups: Sequence[Sequence[str]]) -> set[int]:
    """Indexes of evidence groups whose snippets all occur in this (normalized) chunk text."""
    return {gi for gi, g in enumerate(groups) if all(s in chunk_text_norm for s in g)}


def score_ranking(ranked: Sequence[set[int]], n_groups: int, k: int) -> dict[str, float | int | None]:
    """Compute recall@k, reciprocal rank, nDCG@k and first relevant rank.

    ``ranked[i]`` is the set of evidence groups covered by the chunk at rank i+1.
    """
    seen: set[int] = set()
    dcg = 0.0
    first_rank: int | None = None
    for i, groups in enumerate(ranked):
        if groups and first_rank is None:
            first_rank = i + 1
        if i < k:
            new = groups - seen
            if new:
                # gain = number of groups this chunk newly covers, so one chunk that answers
                # everything at rank 1 scores as well as one group per rank (it used to score 0.61)
                dcg += len(new) / math.log2(i + 2)
                seen |= new
    ideal_hits = min(k, n_groups)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return {
        "recall": len(seen) / n_groups if n_groups else 0.0,
        "rr": 1.0 / first_rank if first_rank else 0.0,
        "ndcg": min(1.0, dcg / idcg) if idcg else 0.0,
        "first_rank": first_rank,
    }


def percentile(values: Sequence[float], p: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), p)) if len(values) else 0.0
