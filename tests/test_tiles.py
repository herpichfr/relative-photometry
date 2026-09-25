"""Tests for relphot.tiles: adaptive rectangular tiling and merging."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_synthetic_night

from relphot.config import Settings
from relphot.exceptions import TilingError
from relphot.tiles import build_tilemap


def test_uniform_field_tiles_and_meets_minimum() -> None:
    night, _airmass, _flux0 = make_synthetic_night(n_stars=2000, n_frames=5, size_px=3000.0)
    settings = Settings()
    candidates = np.ones(night.n_stars, dtype=bool)
    tilemap = build_tilemap(night, candidates, settings)
    assert tilemap.n_tiles > 1
    assert np.all(tilemap.n_candidates >= settings.tile.min_ref_candidates)


def test_core_indices_partition_every_star_exactly_once() -> None:
    night, _airmass, _flux0 = make_synthetic_night(n_stars=1500, n_frames=5, size_px=2500.0)
    settings = Settings()
    candidates = np.ones(night.n_stars, dtype=bool)
    tilemap = build_tilemap(night, candidates, settings)
    covered = (
        np.concatenate(tilemap.core_indices) if tilemap.n_tiles else np.array([], dtype=np.int64)
    )
    assert sorted(covered.tolist()) == list(range(night.n_stars))
    assert np.all(tilemap.core_tile >= 0)


def test_sparse_region_triggers_merging_and_meets_minimum() -> None:
    night, _airmass, _flux0 = make_synthetic_night(
        n_stars=2000, n_frames=5, size_px=3000.0, seed=11
    )
    # Thin out one corner so its own ~1000x1000 grid cell alone would be far
    # short of min_ref_candidates, forcing the adaptive loop to merge it.
    rng = np.random.default_rng(99)
    sparse_zone = (night.x < 1000.0) & (night.y < 1000.0)
    keep = ~sparse_zone | (rng.random(night.n_stars) < 0.02)
    kept_idx = np.nonzero(keep)[0]

    night.ra = night.ra[kept_idx]
    night.dec = night.dec[kept_idx]
    night.x = night.x[kept_idx]
    night.y = night.y[kept_idx]
    night.frame_x = night.frame_x[kept_idx]
    night.frame_y = night.frame_y[kept_idx]
    night.flux = night.flux[kept_idx]
    night.fluxerr = night.fluxerr[kept_idx]
    night.fwhm = night.fwhm[kept_idx]
    night.snr = night.snr[kept_idx]
    night.background = night.background[kept_idx]
    night.flags = night.flags[kept_idx]
    night.presence = night.presence[kept_idx]

    settings = Settings()
    candidates = np.ones(night.n_stars, dtype=bool)
    tilemap = build_tilemap(night, candidates, settings)
    assert tilemap.n_tiles < 9  # fewer than the unmerged 3x3 grid
    assert np.all(tilemap.n_candidates >= settings.tile.min_ref_candidates)


def test_hard_minimum_violation_raises() -> None:
    night, _airmass, _flux0 = make_synthetic_night(n_stars=15, n_frames=5, size_px=3000.0, seed=3)
    candidates = np.ones(night.n_stars, dtype=bool)
    settings = Settings()
    with pytest.raises(TilingError):
        build_tilemap(night, candidates, settings)


def test_to_csv_writes_one_row_per_tile(tmp_path) -> None:
    night, _airmass, _flux0 = make_synthetic_night(n_stars=2000, n_frames=5, size_px=3000.0)
    settings = Settings()
    candidates = np.ones(night.n_stars, dtype=bool)
    tilemap = build_tilemap(night, candidates, settings)
    out = tmp_path / "tiles.csv"
    tilemap.to_csv(out)
    lines = out.read_text().splitlines()
    assert len(lines) == tilemap.n_tiles + 1
    assert lines[0].split(",") == [
        "tile", "xmin", "xmax", "ymin", "ymax", "n_core", "n_extended", "n_candidates",
    ]
