"""Shared pytest fixtures: paths into tests/data (see generate_fixtures.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def fits_files() -> list[Path]:
    return sorted(DATA_DIR.glob("*_proc.fits"))


@pytest.fixture
def csv_files() -> list[Path]:
    return sorted(DATA_DIR.glob("*_proc_catalog.csv"))
