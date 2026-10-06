from __future__ import annotations

from pathlib import Path

import pytest

import g4utils

tomllib = pytest.importorskip("tomllib")  # Python 3.11+

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_version_matches_pyproject():
    with PYPROJECT.open("rb") as f:
        expected = tomllib.load(f)["project"]["version"]
    assert g4utils.__version__ == expected
