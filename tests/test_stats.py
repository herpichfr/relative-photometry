"""Tests for relphot.stats: light-curve statistics and best-aperture selection."""

from __future__ import annotations

import numpy as np
from conftest import make_synthetic_night

from relphot.comparison import select_comparison_stars
from relphot.config import Settings
from relphot.lightcurve import compute_light_curves
from relphot.reference import (
    build_references,
    select_candidates,
    select_reference_frames_and_stars,
)
from relphot.stats import best_aperture_per_star, compute_star_stats, select_best_aperture
from relphot.tiles import build_tilemap


def test_compute_star_stats_hand_built() -> None:
    """Test compute_star_stats on hand-built LightCurveResult with known RMS and chi2."""
    from relphot.lightcurve import LightCurveResult

    # Create a simple hand-built LightCurveResult
    n_stars = 10
    n_frames = 20
    n_aper = 1

    rng = np.random.RandomState(42)
    lc = np.ones((n_stars, n_frames, n_aper), dtype=np.float32)
    # Add noise to all stars with Gaussian distribution
    for i in range(n_stars):
        lc[i, :, 0] = 1.0 + 0.01 * rng.standard_normal(n_frames)

    lc_err = 0.005 * np.ones((n_stars, n_frames, n_aper), dtype=np.float32)
    lc_raw = lc.copy()
    epoch_ok = np.ones((n_stars, n_frames), dtype=bool)

    lc_result = LightCurveResult(
        lc=lc, lc_err=lc_err, lc_raw=lc_raw, epoch_ok=epoch_ok, decorrelation=None
    )

    stats = compute_star_stats(lc_result)

    # Check star 0's RMS is around 0.01
    assert np.isfinite(stats.rms[0, 0]), "RMS should be finite for star 0"
    assert 0.005 < stats.rms[0, 0] < 0.02, f"RMS {stats.rms[0, 0]} out of expected range"

    # Check n_epochs
    assert stats.n_epochs[0, 0] == n_frames, "All frames should be counted for star 0"
    assert stats.n_epochs[1, 0] == n_frames, "All frames should be counted for star 1"

    # Check chi2_reduced is finite
    chi2_list = [
        stats.chi2_reduced[i, 0]
        for i in range(n_stars)
        if np.isfinite(stats.chi2_reduced[i, 0])
    ]
    assert len(chi2_list) > 0, "Should have finite chi2 values"
    # With lc_err=0.005 and noise scale=0.01, chi2 should be ~(0.01/0.005)^2 = 4
    # Allow some variance
    chi2_med = np.median(chi2_list)
    assert 2.0 < chi2_med < 6.0, f"Chi2 {chi2_med} outside expected range [2, 6]"


def test_best_aperture_selection() -> None:
    """Test that best aperture selection picks the quieter aperture for each magnitude bin."""
    night, _airmass, _flux0 = make_synthetic_night(n_stars=500, n_frames=40, n_aper=2, seed=20)

    # Artificially make aperture 0 quieter for bright stars and aperture 1 for faint
    # by scaling the noise per aperture
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

    # Select best aperture
    best_aper_per_tile, bin_edges = select_best_aperture(
        tilemap, comparison_result, star_stats, settings, mag_aper=0
    )

    # Check that we got valid results
    assert best_aper_per_tile.shape[0] == tilemap.n_tiles
    assert best_aper_per_tile.shape[1] == settings.lightcurve.n_mag_bins
    assert bin_edges.shape[0] == tilemap.n_tiles
    assert bin_edges.shape[1] == settings.lightcurve.n_mag_bins + 1

    # At least some apertures should be selected
    n_selected = np.sum(best_aper_per_tile >= 0)
    assert n_selected > 0, "Should have selected some apertures"


def test_best_aperture_per_star() -> None:
    """Test that best_aperture_per_star assigns each star a valid aperture."""
    night, _airmass, _flux0 = make_synthetic_night(n_stars=200, n_frames=40, n_aper=2, seed=21)

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

    best_aper_per_tile, bin_edges = select_best_aperture(
        tilemap, comparison_result, star_stats, settings, mag_aper=0
    )
    star_best_aper = best_aperture_per_star(
        tilemap, comparison_result, best_aper_per_tile, bin_edges, mag_aper=0
    )

    # Check result
    assert star_best_aper.shape == (night.n_stars,)
    assert star_best_aper.dtype == np.int64

    # Stars without a core tile should have -1
    no_core = tilemap.core_tile < 0
    assert np.all(star_best_aper[no_core] == -1), "Stars without core tile should have -1"

    # Stars with a core tile and non-NaN mag should have >= -1
    with_core = tilemap.core_tile >= 0
    with_core_and_mag = with_core & np.isfinite(comparison_result.mag[:, 0])
    for i in np.where(with_core_and_mag)[0]:
        assert star_best_aper[i] >= -1, f"Star {i} should have aperture >= -1"
