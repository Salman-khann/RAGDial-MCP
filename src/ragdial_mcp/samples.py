"""Locate the bundled benchmark pack.

In a wheel (``pip``/``uvx`` install) the pack lives at ``ragdial_mcp/sample_data``;
in a source checkout it lives at the repository root. Tools accept the
shortcut ``"@sample"`` for both the corpus folder and the golden dataset.
"""

from __future__ import annotations

from pathlib import Path

SAMPLE_TOKEN = "@sample"

_PACKAGED = Path(__file__).resolve().parent / "sample_data"
_REPO = Path(__file__).resolve().parents[2] / "sample_data"


def sample_dir() -> Path:
    for candidate in (_PACKAGED, _REPO):
        if (candidate / "corpus").is_dir():
            return candidate
    raise FileNotFoundError("Bundled sample_data not found; reinstall ragdial-mcp or clone the repository.")


def sample_corpus() -> Path:
    return sample_dir() / "corpus"


def sample_golden() -> Path:
    return sample_dir() / "golden_qa.jsonl"
