"""Keep server.json, the package version and the README ownership marker in sync.

The MCP Registry rejects a publish when any of these disagree, so catch it in CI
instead of in the release workflow.
"""

from __future__ import annotations

import json
import re

from ragdial_mcp import __version__

from .conftest import ROOT

SERVER = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))


def test_server_json_versions_match_package():
    assert SERVER["version"] == __version__
    for pkg in SERVER["packages"]:
        assert pkg["version"] == __version__


def test_server_json_points_at_this_pypi_package():
    (pkg,) = SERVER["packages"]
    assert pkg["registryType"] == "pypi"
    assert pkg["identifier"] == "ragdial-mcp"
    assert pkg["transport"]["type"] == "stdio"


def test_registry_description_limit():
    assert len(SERVER["description"]) <= 100


def test_readme_carries_ownership_marker():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    m = re.search(r"mcp-name:\s*(\S+?)\s*-->", readme)
    assert m, "README must contain <!-- mcp-name: ... --> for MCP Registry PyPI verification"
    assert m.group(1) == SERVER["name"]
    assert SERVER["name"].startswith("io.github.Salman-khann/")
