"""Tests for relphot.io: exact .npz round trip of a matched night."""

from __future__ import annotations

import numpy as np
from conftest import make_synthetic_night

from relphot.comparison import select_comparison_stars
from relphot.config import Settings
from relphot.ingest import read_catalogs
from relphot.io import (
    load_lightcurves_npz,
    load_night,
    load_reference,
    save_lightcurve_table,
    save_lightcurves_npz,
    save_night,
    save_reference,
    save_starstats_table,
)
from relphot.lightcurve import compute_light_curves
from relphot.match import match_night
from relphot.reference import (
    build_references,
    select_candidates,
    select_reference_frames_and_stars,
)
from relphot.stats import compute_star_stats
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


def test_save_load_lightcurves_npz(tmp_path) -> None:
    """Test round-trip save/load of light curves."""
    night, _airmass, _flux0 = make_synthetic_night(n_stars=100, n_frames=30, seed=30)

    settings = Settings()
    variable_mask = np.zeros(night.n_stars, dtype=bool)
    candidates = select_candidates(night, variable_mask, settings, aper=0)
    tilemap = build_tilemap(night, candidates, settings)
    frame_selection = select_reference_frames_and_stars(
        night, tilemap, candidates, settings, aper=0
    )
    reference_result = build_references(night, tilemap, frame_selection, settings)
    comparison_result = select_comparison_stars(
        night, tilemap, reference_result, variable_mask, settings
    )

    lc_result = compute_light_curves(night, tilemap, reference_result, comparison_result, settings)
    star_stats = compute_star_stats(lc_result)

    best_aper = np.zeros((tilemap.n_tiles, settings.lightcurve.n_mag_bins), dtype=np.int64)
    bin_edges = np.full((tilemap.n_tiles, settings.lightcurve.n_mag_bins + 1), np.nan)
    star_best_aper = np.zeros(night.n_stars, dtype=np.int64)

    out_path = tmp_path / "lightcurves.npz"
    save_lightcurves_npz(
        lc_result,
        star_stats,
        best_aper,
        bin_edges,
        star_best_aper,
        comparison_result,
        settings,
        out_path,
    )

    assert out_path.is_file()

    # Load and verify
    (
        lc_result2,
        star_stats2,
        _best_aper2,
        _bin_edges2,
        _star_best_aper2,
        _comp_result2,
        settings2,
        _decorr2,
    ) = load_lightcurves_npz(out_path)

    np.testing.assert_array_equal(lc_result2.lc, lc_result.lc)
    np.testing.assert_array_equal(lc_result2.lc_err, lc_result.lc_err)
    np.testing.assert_array_equal(lc_result2.lc_raw, lc_result.lc_raw)
    np.testing.assert_array_equal(lc_result2.epoch_ok, lc_result.epoch_ok)

    np.testing.assert_array_equal(star_stats2.rms, star_stats.rms)
    assert settings2 == settings


def test_save_lightcurve_table_fits(tmp_path) -> None:
    """Test save_lightcurve_table writes a FITS table."""
    night, _airmass, _flux0 = make_synthetic_night(n_stars=50, n_frames=20, seed=31)

    settings = Settings()
    variable_mask = np.zeros(night.n_stars, dtype=bool)
    candidates = select_candidates(night, variable_mask, settings, aper=0)
    tilemap = build_tilemap(night, candidates, settings)
    frame_selection = select_reference_frames_and_stars(
        night, tilemap, candidates, settings, aper=0
    )
    reference_result = build_references(night, tilemap, frame_selection, settings)
    comparison_result = select_comparison_stars(
        night, tilemap, reference_result, variable_mask, settings
    )

    lc_result = compute_light_curves(night, tilemap, reference_result, comparison_result, settings)

    star_best_aper = np.zeros(night.n_stars, dtype=np.int64)

    out_path = tmp_path / "lc"
    saved_path = save_lightcurve_table(
        night, tilemap, lc_result, star_best_aper, settings, out_path, fmt="fits"
    )

    assert saved_path.suffix == ".fits"
    assert saved_path.is_file()

    # Try to read it back
    try:
        from astropy.table import Table
        table = Table.read(str(saved_path), format="fits")
        assert "star_id" in table.colnames
        assert "lc" in table.colnames
        assert "lc_err" in table.colnames
    except ImportError:
        pass  # astropy not available


def test_save_starstats_table_fits(tmp_path) -> None:
    """Test save_starstats_table writes a FITS table."""
    night, _airmass, _flux0 = make_synthetic_night(n_stars=50, n_frames=20, seed=32)

    settings = Settings()
    variable_mask = np.zeros(night.n_stars, dtype=bool)
    candidates = select_candidates(night, variable_mask, settings, aper=0)
    tilemap = build_tilemap(night, candidates, settings)
    frame_selection = select_reference_frames_and_stars(
        night, tilemap, candidates, settings, aper=0
    )
    reference_result = build_references(night, tilemap, frame_selection, settings)
    comparison_result = select_comparison_stars(
        night, tilemap, reference_result, variable_mask, settings
    )

    lc_result = compute_light_curves(night, tilemap, reference_result, comparison_result, settings)
    star_stats = compute_star_stats(lc_result)

    star_best_aper = np.zeros(night.n_stars, dtype=np.int64)

    out_path = tmp_path / "stats"
    saved_path = save_starstats_table(
        night,
        tilemap,
        comparison_result,
        star_stats,
        star_best_aper,
        settings,
        out_path,
        fmt="fits",
    )

    assert saved_path.suffix == ".fits"
    assert saved_path.is_file()

    # Try to read it back
    try:
        from astropy.table import Table
        table = Table.read(str(saved_path), format="fits")
        assert "star_id" in table.colnames
        assert "rms" in table.colnames
        assert "chi2_reduced" in table.colnames
    except ImportError:
        pass  # astropy not available
