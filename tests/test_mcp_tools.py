"""End-to-end tests through a real MCP client session (in-process transport)."""

from __future__ import annotations

import json

import pytest
from mcp import Client

from ragdial_mcp.server import build_server

from .conftest import CORPUS, GOLDEN

pytestmark = pytest.mark.anyio


async def _call(client: Client, name: str, args: dict) -> dict:
    res = await client.call_tool(name, args)
    assert not res.is_error, res.content
    assert res.structured_content is not None
    return res.structured_content


async def test_tool_and_prompt_surface(lab):
    async with Client(build_server(lab)) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        assert set(tools) == {"ingest_corpus", "run_chunking_ablation", "inspect_boundary_splits",
                              "benchmark_retrieval", "diff_experiment_runs"}
        assert all(t.output_schema for t in tools.values())
        prompts = {p.name for p in (await client.list_prompts()).prompts}
        assert prompts == {"diagnose_retrieval_failures", "generate_rag_spec"}
        templates = {t.uri_template for t in (await client.list_resource_templates()).resource_templates}
        assert templates == {"ragdial://experiments/{run_id}/report", "ragdial://corpus/{corpus_id}/stats"}


async def test_full_experiment_loop(lab):
    async with Client(build_server(lab)) as client:
        ing = await _call(client, "ingest_corpus", {"folder_path": str(CORPUS), "corpus_id": "halyard"})
        assert ing["file_count"] == 13 and ing["detected"]["tables"] >= 10

        abl = await _call(client, "run_chunking_ablation", {
            "corpus_id": "halyard", "strategies": ["fixed", "markdown"], "chunk_sizes": [256],
            "overlaps": [0], "embedder": "hash"})
        damage = {c["config_id"]: c["damage_score"] for c in abl["configs"]}
        assert damage["markdown-256-o0"] < damage["fixed-256-o0"]
        assert abl["least_damage"] == "markdown-256-o0"
        assert "text" not in json.dumps(abl["configs"][0])  # no chunk text leaks into context

        insp = await _call(client, "inspect_boundary_splits",
                           {"run_id": abl["run_id"], "limit": 3, "config_id": "fixed-256-o0", "kinds": ["table"]})
        assert 1 <= len(insp["splits"]) <= 3
        assert all(s["kind"] == "table" for s in insp["splits"])
        assert all(len(s["before_cut"]) <= 120 and len(s["after_cut"]) <= 120 for s in insp["splits"])

        bench = await _call(client, "benchmark_retrieval", {
            "run_id": abl["run_id"], "dataset_path": str(GOLDEN), "retrievers": ["bm25", "hybrid"],
            "top_k": 5, "embedder": "hash", "vector_store": "numpy"})
        assert bench["queries"] == 30 and len(bench["results"]) == 4
        assert bench["best_variant"].startswith("markdown-256-o0")
        assert len(bench["worst_queries"]) <= 3
        for v in bench["results"]:
            assert 0 <= v["recall_at_k"] <= 1 and 0 <= v["ndcg_at_k"] <= 1 and v["p95_ms"] >= v["p50_ms"] >= 0

        diff = await _call(client, "diff_experiment_runs", {
            "run_a": bench["run_id"], "run_b": bench["run_id"],
            "variant_a": "fixed-256-o0|bm25|none", "variant_b": "markdown-256-o0|bm25|none"})
        assert diff["metrics"]["ndcg_at_k"]["delta"] > 0
        assert diff["fixed"] and all(f["rank_b"] is not None for f in diff["fixed"])
        assert diff["rank_improved"] + diff["rank_worsened"] + diff["unchanged"] == 30

        index = json.loads((await client.read_resource("ragdial://experiments")).contents[0].text)
        assert {r["run_id"] for r in index["runs"]} == {abl["run_id"], bench["run_id"]}

        report = json.loads((await client.read_resource(f"ragdial://experiments/{bench['run_id']}/report")).contents[0].text)
        assert len(report["per_query"]["markdown-256-o0|bm25|none"]) == 30
        stats = json.loads((await client.read_resource("ragdial://corpus/halyard/stats")).contents[0].text)
        assert stats["file_count"] == 13

        prompt = await client.get_prompt("diagnose_retrieval_failures", {"run_id": bench["run_id"]})
        assert bench["run_id"] in prompt.messages[0].content.text
        spec = await client.get_prompt("generate_rag_spec", {"corpus_id": "halyard", "latency_budget_ms": "150"})
        assert "150 ms" in spec.messages[0].content.text


async def test_errors_are_actionable(lab):
    async with Client(build_server(lab)) as client:
        res = await client.call_tool("run_chunking_ablation",
                                     {"corpus_id": "nope", "strategies": ["fixed"], "chunk_sizes": [128]})
        assert res.is_error
        assert "ingest_corpus" in res.content[0].text

        await _call(client, "ingest_corpus", {"folder_path": str(CORPUS), "corpus_id": "c"})
        res = await client.call_tool("run_chunking_ablation",
                                     {"corpus_id": "c", "strategies": ["fixed"], "chunk_sizes": [128], "overlaps": [128]})
        assert res.is_error and "overlap" in res.content[0].text

        res = await client.call_tool("diff_experiment_runs", {"run_a": "bench-x", "run_b": "bench-y"})
        assert res.is_error and "ragdial://experiments" in res.content[0].text

        with pytest.raises(Exception, match="Unknown run_id"):
            await client.read_resource("ragdial://experiments/bench-missing/report")
