"""Tests for relphot.stats: light-curve statistics and best-aperture selection."""

from __future__ import annotations

import numpy as np
from conftest import make_synthetic_night

from relphot.comparison import select_comparison_stars
from relphot.config import Settings
from relphot.lightcurve import LightCurveResult, compute_light_curves
from relphot.reference import (
    build_references,
    select_candidates,
    select_reference_frames_and_stars,
)
from relphot.stats import (
    best_aperture_per_star,
    bright_star_floor,
    compute_star_stats,
    select_best_aperture,
)
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


def test_expected_noise_stays_formal_and_chi2_uses_the_inflated_error() -> None:
    """With error inflation ``expected_noise`` is the formal prediction, ``chi2`` the used error."""
    from relphot.lightcurve import LightCurveResult

    n_stars, n_frames = 5, 200
    rng = np.random.default_rng(7)
    lc = (1.0 + 0.03 * rng.standard_normal((n_stars, n_frames, 1))).astype(np.float32)
    raw = np.full((n_stars, n_frames, 1), 0.01, dtype=np.float32)
    scale = np.full((n_stars, 1), 3.0)
    lc_result = LightCurveResult(
        lc=lc, lc_err=raw * 3.0, lc_raw=lc.copy(), epoch_ok=np.ones((n_stars, n_frames), bool),
        decorrelation=None, lc_err_raw=raw, err_scale=scale,
        blended=np.array([True, False, False, False, False]),
    )
    stats = compute_star_stats(lc_result)

    np.testing.assert_allclose(stats.expected_noise[:, 0], 0.01, rtol=1e-2)
    # scatter 0.03 against the inflated error 0.03: chi2_red ~ 1 (it would be ~9 against 0.01)
    assert np.all((stats.chi2_reduced[:, 0] > 0.7) & (stats.chi2_reduced[:, 0] < 1.4))
    assert stats.err_scale is scale
    assert stats.blended.tolist() == [True, False, False, False, False]

    # a result without inflation information behaves exactly as before
    plain = LightCurveResult(
        lc=lc, lc_err=raw, lc_raw=lc.copy(), epoch_ok=np.ones((n_stars, n_frames), bool),
        decorrelation=None,
    )
    stats0 = compute_star_stats(plain)
    np.testing.assert_allclose(stats0.expected_noise[:, 0], 0.01, rtol=1e-2)
    assert stats0.err_scale is None and stats0.blended is None


def test_compute_star_stats_near_edge_passthrough() -> None:
    """compute_star_stats passes through near_edge array."""
    n_stars = 5
    n_frames = 10
    n_aper = 1
    lc = np.ones((n_stars, n_frames, n_aper))
    raw = np.ones((n_stars, n_frames, n_aper)) * 0.01
    lc_result = LightCurveResult(
        lc=lc, lc_err=raw, lc_raw=lc.copy(), epoch_ok=np.ones((n_stars, n_frames), bool),
        decorrelation=None,
    )

    # Create near_edge mask
    near_edge = np.array([True, False, True, False, True])

    stats = compute_star_stats(lc_result, near_edge=near_edge)

    assert stats.near_edge is not None
    np.testing.assert_array_equal(stats.near_edge, near_edge)


def test_compute_star_stats_near_edge_defaults_none() -> None:
    """compute_star_stats defaults to near_edge=None."""
    n_stars = 5
    n_frames = 10
    n_aper = 1
    lc = np.ones((n_stars, n_frames, n_aper))
    raw = np.ones((n_stars, n_frames, n_aper)) * 0.01
    lc_result = LightCurveResult(
        lc=lc, lc_err=raw, lc_raw=lc.copy(), epoch_ok=np.ones((n_stars, n_frames), bool),
        decorrelation=None,
    )

    stats = compute_star_stats(lc_result)

    assert stats.near_edge is None


def test_compute_star_stats_tailed_passthrough() -> None:
    """compute_star_stats passes through the tailed array and defaults it to None."""
    n_stars = 5
    n_frames = 10
    lc = np.ones((n_stars, n_frames, 1))
    raw = np.ones((n_stars, n_frames, 1)) * 0.01
    lc_result = LightCurveResult(
        lc=lc, lc_err=raw, lc_raw=lc.copy(), epoch_ok=np.ones((n_stars, n_frames), bool),
        decorrelation=None,
    )
    tailed = np.array([False, True, False, False, True])

    stats = compute_star_stats(lc_result, tailed=tailed)

    assert stats.tailed is not None
    np.testing.assert_array_equal(stats.tailed, tailed)
    assert compute_star_stats(lc_result).tailed is None


def _floor_catalogue() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1000 stars, 0.01 mag apart, flat 2 mmag-ish RMS below mag 2, rising above."""
    mag = 0.005 + 0.01 * np.arange(1000)
    rms = np.where(mag < 2.0, 0.002, 0.01)
    return mag, rms, np.ones(mag.size, dtype=bool)


def test_bright_star_floor_hand_built() -> None:
    """The bin starts at the given percentile of all stars; the floor is the bin's median RMS."""
    mag, rms, is_comp = _floor_catalogue()
    res = bright_star_floor(mag, rms, is_comp, percentile=1.0)
    assert res is not None
    assert np.isclose(res.mag_lo, 0.005 + 0.01 * 9.99)
    assert np.isclose(res.mag_hi, res.mag_lo + 1.0)
    assert res.n == 100
    assert np.isclose(res.floor_mmag, 0.002 * 1085.7)


def test_bright_star_floor_default_percentile() -> None:
    """The default anchor is the 0.25th percentile."""
    mag, rms, is_comp = _floor_catalogue()
    res = bright_star_floor(mag, rms, is_comp)
    assert res is not None
    assert np.isclose(res.mag_lo, 0.005 + 0.01 * 2.4975)
    assert res.n == 100


def test_bright_star_floor_ignores_a_few_very_bright_stars() -> None:
    """A handful of far brighter stars must not drag the bin to them (the old min anchor did)."""
    mag, rms, is_comp = _floor_catalogue()
    mag = np.concatenate([np.full(5, -5.0), mag])
    rms = np.concatenate([np.full(5, 0.0005), rms])
    is_comp = np.concatenate([np.ones(5, dtype=bool), is_comp])
    res = bright_star_floor(mag, rms, is_comp, percentile=1.0)
    assert res is not None
    assert res.mag_lo > 0.0
    assert res.n >= 95
    assert np.isclose(res.floor_mmag, 0.002 * 1085.7)


def test_bright_star_floor_uses_comparison_stars_only_for_the_median() -> None:
    """Non-comparison stars set the anchor but not the median."""
    mag, rms, is_comp = _floor_catalogue()
    is_comp[::2] = False
    rms[::2] = 0.05
    res = bright_star_floor(mag, rms, is_comp, percentile=1.0)
    assert res is not None
    assert res.n == 50
    assert np.isclose(res.floor_mmag, 0.002 * 1085.7)


def test_bright_star_floor_skips_nan_and_returns_none_when_empty() -> None:
    """NaN magnitudes or RMS never count; no comparison star in the bin gives None."""
    mag, rms, is_comp = _floor_catalogue()
    mag_nan = mag.copy()
    mag_nan[:20] = np.nan
    res = bright_star_floor(mag_nan, rms, is_comp)
    assert res is not None
    assert np.isfinite(res.floor_mmag)
    all_nan = np.full(10, np.nan)
    assert bright_star_floor(all_nan, np.full(10, 0.01), np.ones(10, dtype=bool)) is None
    assert bright_star_floor(mag, rms, np.zeros(mag.size, dtype=bool)) is None
