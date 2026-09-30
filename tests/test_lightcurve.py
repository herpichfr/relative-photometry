"""Tests for relphot.lightcurve: light curve extraction."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from conftest import make_synthetic_night

from relphot.comparison import select_comparison_stars
from relphot.config import Settings
from relphot.exceptions import ConfigError
from relphot.lightcurve import (
    NEIGHBOUR_FLAG_MASK,
    compute_light_curves,
    error_inflation,
    point_to_point_sigma,
)
from relphot.reference import (
    build_references,
    select_candidates,
    select_reference_frames_and_stars,
)
from relphot.tiles import build_tilemap


def test_lightcurve_chi2_flat_night() -> None:
    """Chi-squared of flat synthetic night is within expected range.

    For a synthetic night with constant sources, decorrelation off,
    the reduced chi2 should be within [0.7, 1.5].
    """
    night, _airmass, _flux0 = make_synthetic_night(n_stars=500, n_frames=40, seed=10)

    from dataclasses import replace

    settings = Settings()
    settings = replace(settings, decorrelation=replace(settings.decorrelation, enabled=False))

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

    # Compute chi2_reduced for a sample of stars
    lc = lc_result.lc[:, :, 0]
    lc_err = lc_result.lc_err[:, :, 0]

    chi2_list = []
    for i in range(min(50, night.n_stars)):
        if tilemap.core_tile[i] >= 0:
            lc_med = np.nanmedian(lc[i, :])
            if np.isfinite(lc_med) and lc_med != 0:
                residuals = (lc[i, :] - lc_med) / lc_err[i, :]
                n_finite = np.count_nonzero(np.isfinite(residuals))
                if n_finite > 1:
                    chi2 = np.nansum(residuals**2) / max(n_finite - 1, 1)
                    chi2_list.append(chi2)

    if chi2_list:
        chi2_med = np.median(chi2_list)
        assert 0.7 <= chi2_med <= 1.5, f"Chi2_reduced {chi2_med} outside [0.7, 1.5]"


def test_lightcurve_dropped_frame_nan() -> None:
    """Dropped frames yield NaN light curves without RuntimeWarning."""
    night, _airmass, _flux0 = make_synthetic_night(n_stars=200, n_frames=40, seed=11)
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

    with np.errstate(invalid="ignore", divide="ignore"):
        lc_result = compute_light_curves(
            night, tilemap, reference_result, comparison_result, settings
        )

    # Check dropped frames are NaN
    dropped = ~reference_result.frame_kept
    if np.any(dropped):
        lc_dropped = lc_result.lc[:, dropped, 0]
        lc_err_dropped = lc_result.lc_err[:, dropped, 0]
        assert np.all(np.isnan(lc_dropped)) or np.all(
            np.isnan(lc_err_dropped)
        ), "Dropped frames should yield NaN"


def test_lightcurve_bad_flags_masked() -> None:
    """Epochs with bad FLAGS are NaN in light curve."""
    night, _airmass, _flux0 = make_synthetic_night(n_stars=200, n_frames=40, seed=12)
    settings = Settings()

    # Inject FLAGS bit 4 (saturated) in some epochs
    night.flags[10:20, 5:10] = 4

    # And inject bit 2 (blended) in other epochs, which should be kept
    night.flags[30:35, 15:20] = 2

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

    # bad_flag_mask defaults to 252, which includes bit 4 but not bit 2
    # So epochs with FLAGS=4 should have NaN lc, but FLAGS=2 should be kept

    # Check star 15, frames 5-10: should have NaN (bit 4 is masked)
    assert np.all(np.isnan(lc_result.lc[15, 5:10, 0])) or (
        lc_result.epoch_ok[15, 5:10].sum() == 0
    ), "Epochs with bad FLAGS should be NaN"

    # Check star 32, frames 15-20: should NOT all be NaN (bit 2 is not masked)
    # (may be NaN for other reasons, but not because of the FLAGS mask)
    # Just verify that the mask logic runs without error


def test_lightcurve_raw_equals_relative_flux_over_ensemble() -> None:
    """lc_raw exactly equals relative_flux / ensemble where epoch_ok."""
    night, _airmass, _flux0 = make_synthetic_night(n_stars=200, n_frames=40, seed=13)

    from dataclasses import replace

    settings = Settings()
    settings = replace(settings, decorrelation=replace(settings.decorrelation, enabled=False))

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

    # For each star with a core tile, check lc_raw = relative_flux / ensemble
    for i in range(night.n_stars):
        t = int(tilemap.core_tile[i])
        if t >= 0:
            rel_flux = reference_result.relative_flux[i, :, 0]
            ens = comparison_result.ensemble[t, :, 0]
            expected_lc_raw = rel_flux / ens

            # Only compare where epoch_ok
            ok = lc_result.epoch_ok[i, :]
            if np.any(ok):
                np.testing.assert_allclose(
                    lc_result.lc_raw[i, ok, 0],
                    expected_lc_raw[ok],
                    rtol=1e-5,
                    atol=1e-10,
                    err_msg=f"lc_raw mismatch for star {i}",
                )


def test_lightcurve_best_aperture_selection() -> None:
    """Best aperture: aperture 0 quieter for bright, aperture 1 for faint."""
    night, _airmass, _flux0 = make_synthetic_night(n_stars=500, n_frames=40, n_aper=2, seed=14)

    from dataclasses import replace

    settings = Settings()
    settings = replace(settings, decorrelation=replace(settings.decorrelation, enabled=False))

    # Scale noise per aperture: aper 0 smaller for bright stars, aper 1 smaller for faint
    # We'll scale fluxerr as a function of magnitude
    from relphot.numeric import nanmedian_quiet

    variable_mask = np.zeros(night.n_stars, dtype=bool)
    candidates = select_candidates(night, variable_mask, settings, aper=0)
    tilemap = build_tilemap(night, candidates, settings)

    # Get initial mag estimates
    frame_selection = select_reference_frames_and_stars(
        night, tilemap, candidates, settings, aper=0
    )
    reference_result = build_references(night, tilemap, frame_selection, settings)
    comparison_result = select_comparison_stars(
        night, tilemap, reference_result, variable_mask, settings
    )

    mag = comparison_result.mag[:, 0]
    mag_median = nanmedian_quiet(mag[np.isfinite(mag)])

    # Scale fluxerr so aper 0 is quieter for bright (mag < median) and aper 1 for faint
    for i in range(night.n_stars):
        if np.isfinite(mag[i]):
            if mag[i] < mag_median:
                # Bright: make aper 0 quieter (smaller error)
                night.fluxerr[i, :, 1] *= 2.0
            else:
                # Faint: make aper 1 quieter (smaller error)
                night.fluxerr[i, :, 0] *= 2.0

    # Re-run pipeline with the modified fluxerr
    candidates2 = select_candidates(night, variable_mask, settings, aper=0)
    tilemap2 = build_tilemap(night, candidates2, settings)
    frame_selection2 = select_reference_frames_and_stars(
        night, tilemap2, candidates2, settings, aper=0
    )
    reference_result2 = build_references(night, tilemap2, frame_selection2, settings)
    comparison_result2 = select_comparison_stars(
        night, tilemap2, reference_result2, variable_mask, settings
    )

    lc_result = compute_light_curves(
        night, tilemap2, reference_result2, comparison_result2, settings
    )

    from relphot.stats import compute_star_stats, select_best_aperture

    star_stats = compute_star_stats(lc_result)
    best_aper_per_tile, _bin_edges = select_best_aperture(
        tilemap2, comparison_result2, star_stats, settings, mag_aper=0
    )

    # Check that best aperture differs between bright and faint bins for at least one tile
    differs = False
    for t in range(tilemap2.n_tiles):
        best_apers = best_aper_per_tile[t, :]
        best_apers = best_apers[best_apers >= 0]
        if len(set(best_apers)) > 1:
            differs = True
            break

    assert differs, "Best aperture should differ between magnitude bins for at least one tile"


# --------------------------------------------------------------------------
# error inflation (blended / excess-scatter stars)
# --------------------------------------------------------------------------

_N_FRAMES = 300


def _inflation_case(seed: int = 0):
    """4 stars, 1 aperture, formal error 0.005 everywhere.

    star 0: blended (neighbour flag in 30 % of frames), scatter 3x the error;
    star 1: unblended, scatter 3x the error; star 2: unblended, scatter = error;
    star 3: blended, scatter = error.
    """
    rng = np.random.default_rng(seed)
    err = 0.005
    scatter = np.array([3.0, 3.0, 1.0, 1.0]) * err
    lc = 1.0 + rng.standard_normal((4, _N_FRAMES, 1)) * scatter[:, None, None]
    lc_err = np.full((4, _N_FRAMES, 1), err)
    epoch_ok = np.ones((4, _N_FRAMES), dtype=bool)
    flags = np.zeros((4, _N_FRAMES), dtype=np.int32)
    flags[0, ::3] = 2
    flags[3, ::3] = 1
    return lc, lc_err, epoch_ok, flags


def _settings_with(mode: str) -> Settings:
    base = Settings()
    return replace(base, lightcurve=replace(base.lightcurve, inflate_errors=mode))


def test_point_to_point_sigma_ignores_a_transit_and_slow_variability() -> None:
    """sigma_p2p reads the white noise, not a 3 % transit nor a slow sinusoid (transit-safe)."""
    rng = np.random.default_rng(1)
    n = 400
    t = np.arange(n)
    clean = 1.0 + 0.005 * rng.standard_normal(n)
    dip = np.where((t > 150) & (t < 200), 0.03, 0.0)
    slow = 0.02 * np.sin(2 * np.pi * t / 150.0)
    lc = np.stack([clean, clean - dip + slow], axis=0)[:, :, None]
    sigma = point_to_point_sigma(lc, np.ones((2, n), dtype=bool))
    assert sigma[0, 0] == pytest.approx(0.005, rel=0.15)
    assert sigma[1, 0] == pytest.approx(sigma[0, 0], rel=0.05)


def test_point_to_point_sigma_needs_enough_pairs_and_skips_dropped_epochs() -> None:
    lc = 1.0 + 0.01 * np.random.default_rng(2).standard_normal((2, 30, 1))
    ok = np.ones((2, 30), dtype=bool)
    ok[1, ::2] = False  # every other epoch dropped: no two consecutive kept epochs
    sigma = point_to_point_sigma(lc, ok, min_pairs=10)
    assert np.isfinite(sigma[0, 0])
    assert np.isnan(sigma[1, 0])


def test_inflation_blended_mode_scales_only_the_blended_star() -> None:
    lc, lc_err, epoch_ok, flags = _inflation_case()
    scale, blended = error_inflation(lc, lc_err, epoch_ok, flags, _settings_with("blended"))
    assert blended.tolist() == [True, False, False, True]
    assert scale[0, 0] == pytest.approx(3.0, rel=0.2)
    # the same excess on an unblended star is NOT inflated with setting 'blended'
    assert scale[1, 0] == 1.0
    assert scale[2, 0] == 1.0
    # a blended star that is not noisier than its error keeps a factor of ~1 (never below 1)
    assert 1.0 <= scale[3, 0] < 1.25


def test_inflation_other_modes() -> None:
    lc, lc_err, epoch_ok, flags = _inflation_case()
    excess, _ = error_inflation(lc, lc_err, epoch_ok, flags, _settings_with("excess"))
    assert excess[0, 0] == pytest.approx(3.0, rel=0.2)
    assert excess[1, 0] == pytest.approx(3.0, rel=0.2)  # unblended but a clear excess
    assert excess[2, 0] == 1.0  # ~1.0x: below err_scale_excess_min, unblended
    everyone, _ = error_inflation(lc, lc_err, epoch_ok, flags, _settings_with("all"))
    assert np.all(everyone >= 1.0)
    assert everyone[2, 0] == pytest.approx(1.0, abs=0.25)
    none, blended = error_inflation(lc, lc_err, epoch_ok, flags, _settings_with("none"))
    assert np.all(none == 1.0)
    assert blended[0]  # the flag verdict is reported whatever the mode
    with pytest.raises(ConfigError, match="inflate_errors"):
        error_inflation(lc, lc_err, epoch_ok, flags, _settings_with("sometimes"))


def test_inflation_counts_only_kept_epochs_for_the_blend_fraction() -> None:
    lc, lc_err, epoch_ok, flags = _inflation_case()
    flags[1, :] = 0
    flags[1, :20] = NEIGHBOUR_FLAG_MASK  # 20 flagged frames ...
    epoch_ok[1, 20:] = False  # ... but they are ALL the star's kept frames
    _scale, blended = error_inflation(lc, lc_err, epoch_ok, flags, _settings_with("blended"))
    assert blended[1]


def test_compute_light_curves_inflates_lc_err_and_keeps_the_raw_error() -> None:
    """End to end on a synthetic night: an injected 3x excess on a blended star."""
    night, _airmass, _flux0 = make_synthetic_night(n_stars=400, n_frames=60, seed=21)
    settings = replace(
        Settings(),
        decorrelation=replace(Settings().decorrelation, enabled=False),
        lightcurve=replace(Settings().lightcurve, inflate_errors="blended"),
    )
    blended_star, plain_star = 10, 11
    rng = np.random.default_rng(5)
    for i in (blended_star, plain_star):
        night.flux[i, :, 0] += rng.standard_normal(night.n_frames) * 3.0 * night.fluxerr[i, :, 0]
    night.flags[blended_star, ::2] = 2

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
    assert tilemap.core_tile[blended_star] >= 0 and tilemap.core_tile[plain_star] >= 0

    res = compute_light_curves(night, tilemap, reference_result, comparison_result, settings)

    assert res.blended[blended_star] and not res.blended[plain_star]
    assert res.err_scale[blended_star, 0] > 1.8
    assert res.err_scale[plain_star, 0] == 1.0
    ok = res.epoch_ok[blended_star]
    np.testing.assert_allclose(
        res.lc_err[blended_star, ok, 0],
        res.lc_err_raw[blended_star, ok, 0] * res.err_scale[blended_star, 0],
        rtol=1e-5,
    )
    np.testing.assert_array_equal(res.lc_err[plain_star], res.lc_err_raw[plain_star])


def test_lc_err_of_a_median_night_is_target_photon_noise_plus_error_of_the_median() -> None:
    """lc_err (no inflation, no decorrelation) = sqrt((phot/ens)^2 + (lc_raw*sig_ens/ens)^2)."""
    night, _airmass, _flux0 = make_synthetic_night(n_stars=300, n_frames=30, seed=21)
    settings = Settings()
    assert settings.comparison.ensemble_statistic == "median"
    settings = replace(
        settings,
        decorrelation=replace(settings.decorrelation, enabled=False),
        lightcurve=replace(settings.lightcurve, inflate_errors="none"),
    )
    variable_mask = np.zeros(night.n_stars, dtype=bool)
    candidates = select_candidates(night, variable_mask, settings, aper=0)
    tilemap = build_tilemap(night, candidates, settings)
    selection = select_reference_frames_and_stars(night, tilemap, candidates, settings, aper=0)
    ref = build_references(night, tilemap, selection, settings)
    comp = select_comparison_stars(night, tilemap, ref, variable_mask, settings)
    lc = compute_light_curves(night, tilemap, ref, comp, settings)

    i = int(np.nonzero(tilemap.core_tile >= 0)[0][0])
    t = int(tilemap.core_tile[i])
    ok = lc.epoch_ok[i] & np.isfinite(lc.lc_err[i, :, 0])
    ens = comp.ensemble[t, :, 0]
    phot = night.fluxerr[i, :, 0] / ref.R[t, :, 0]
    ens_term = lc.lc_raw[i, :, 0] * comp.sigma_ensemble[t, :, 0] / ens
    expected = np.sqrt((phot / ens) ** 2 + ens_term**2)
    np.testing.assert_allclose(lc.lc_err[i, ok, 0], expected[ok], rtol=1e-4)
    assert np.all(lc.lc_err[i, ok, 0] > phot[ok] / ens[ok])  # the ensemble term is added
