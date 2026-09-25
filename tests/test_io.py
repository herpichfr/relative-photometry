"""Tests for relphot.io: exact .npz round trip of a matched night."""

from __future__ import annotations

import numpy as np

from relphot.config import Settings
from relphot.ingest import read_catalogs
from relphot.io import load_night, load_reference, save_night, save_reference
from relphot.match import match_night
from relphot.reference import build_references, select_candidates, select_reference_frames_and_stars
from relphot.tiles import build_tilemap


def test_round_trip_is_exact(fits_files, tmp_path) -> None:
    settings = Settings()
    cats = read_catalogs(list(fits_files), settings, fmt="fits")
    night = match_night(cats, settings)

    out = tmp_path / "night.npz"
    save_night(night, settings, out)
    night2, settings2 = load_night(out)

    assert settings2 == settings
    assert night2.master_frame_index == night.master_frame_index
    assert night2.n_stars_before_cut == night.n_stars_before_cut
    assert night2.n_stars_after_cut == night.n_stars_after_cut
    assert night2.reports == night.reports
    assert [m.to_dict() for m in night2.frame_meta] == [m.to_dict() for m in night.frame_meta]

    attrs = (
        "ra",
        "dec",
        "x",
        "y",
        "frame_x",
        "frame_y",
        "flux",
        "fluxerr",
        "fwhm",
        "snr",
        "background",
        "flags",
        "presence",
    )
    for attr in attrs:
        a = getattr(night, attr)
        b = getattr(night2, attr)
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(a, b, strict=True)


def test_reference_frame_kept_round_trip(fits_files, tmp_path) -> None:
    """Test that frame_kept and dropped_frames round-trip through save/load."""
    settings = Settings()
    cats = read_catalogs(list(fits_files), settings, fmt="fits")
    night = match_night(cats, settings)

    variable_mask = np.zeros(night.n_stars, dtype=bool)
    candidates = select_candidates(night, variable_mask, settings, aper=0)
    tilemap = build_tilemap(night, candidates, settings)
    frame_selection = select_reference_frames_and_stars(
        night, tilemap, candidates, settings, aper=0
    )
    result = build_references(night, tilemap, frame_selection, settings)

    ref_out = tmp_path / "ref.npz"
    save_reference(tilemap, result, settings, ref_out)
    _tilemap2, result2, _settings2 = load_reference(ref_out)

    np.testing.assert_array_equal(result2.frame_kept, result.frame_kept)
    assert result2.frame_kept.dtype == result.frame_kept.dtype
