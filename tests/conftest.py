from __future__ import annotations

from pathlib import Path

import pytest

from ragdial_mcp.lab import Lab

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "sample_data" / "corpus"
GOLDEN = ROOT / "sample_data" / "golden_qa.jsonl"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def lab(tmp_path: Path) -> Lab:
    lab = Lab(tmp_path / "home")
    yield lab
    lab.db.close()


@pytest.fixture(scope="session")
def sample_docs():
    from ragdial_mcp.ingestion.parser import load_folder

    docs, warnings = load_folder(CORPUS)
    assert not warnings
    return docs
