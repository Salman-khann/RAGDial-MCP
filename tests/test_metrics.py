from __future__ import annotations

import math

import pytest

from ragdial_mcp.retrieval.hybrid import ChunkSearcher, rrf
from ragdial_mcp.retrieval.embedders import HashEmbedder
from ragdial_mcp.retrieval.metrics import groups_covered, normalize, percentile, score_ranking


def test_perfect_ranking():
    m = score_ranking([{0}, set(), set()], n_groups=1, k=5)
    assert m == {"recall": 1.0, "rr": 1.0, "ndcg": 1.0, "first_rank": 1}


def test_rank_three_vs_rank_one():
    m = score_ranking([set(), set(), {0}], n_groups=1, k=5)
    assert m["recall"] == 1.0
    assert m["rr"] == pytest.approx(1 / 3)
    assert m["ndcg"] == pytest.approx(1 / math.log2(4))
    assert m["first_rank"] == 3


def test_hit_beyond_k_counts_for_mrr_not_recall():
    m = score_ranking([set()] * 5 + [{0}], n_groups=1, k=5)
    assert m["recall"] == 0.0 and m["ndcg"] == 0.0
    assert m["rr"] == pytest.approx(1 / 6)


def test_duplicate_groups_do_not_inflate_ndcg():
    m = score_ranking([{0}, {0}, {0}], n_groups=1, k=3)
    assert m["ndcg"] == pytest.approx(1.0)


def test_multi_group_recall_and_ndcg():
    m = score_ranking([{1}, set(), {0}], n_groups=2, k=3)
    assert m["recall"] == 1.0
    ideal = 1 + 1 / math.log2(3)
    assert m["ndcg"] == pytest.approx((1 + 1 / math.log2(4)) / ideal)
    half = score_ranking([{1}, set(), set()], n_groups=2, k=3)
    assert half["recall"] == 0.5


def test_no_hits():
    m = score_ranking([set(), set()], n_groups=1, k=2)
    assert m == {"recall": 0.0, "rr": 0.0, "ndcg": 0.0, "first_rank": None}


def test_groups_require_all_snippets_in_same_chunk():
    groups = [[normalize("| Key | Default |"), normalize("| a | 1 |")]]
    assert groups_covered(normalize("| Key | Default |\n|--|--|\n| a | 1 |"), groups) == {0}
    assert groups_covered(normalize("| a | 1 |"), groups) == set()


def test_normalize_collapses_whitespace_and_case():
    assert normalize("  Hello\n\n  WORLD \t x") == "hello world x"


def test_rrf_fuses_rankings():
    fused = rrf([[1, 2, 3], [3, 1, 4]])
    assert fused[0] == 1  # ranked 1st and 2nd beats 3rd and 1st
    assert set(fused) == {1, 2, 3, 4}


def test_percentile():
    assert percentile([1, 2, 3, 4, 100], 50) == 3
    assert percentile([], 95) == 0.0


def test_searcher_modes_rank_the_obvious_chunk_first():
    texts = ["the broker rolls segments at 512MiB", "consumer groups rebalance cooperatively",
             "pricing for the growth plan is 399 dollars"]
    s = ChunkSearcher(texts, HashEmbedder(512), vector_store="numpy")
    for mode in ("bm25", "dense", "hybrid"):
        ranked, ms = s.search("when does the broker roll segments", mode, top_k=2)
        assert ranked[0] == 0, mode
        assert ms >= 0
    with pytest.raises(ValueError):
        s.search("x", "sparse-magic", 1)
