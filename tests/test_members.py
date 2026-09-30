"""Tests for relphot.members: reference and comparison member traceability."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_synthetic_night

from relphot.comparison import select_comparison_stars
from relphot.config import Settings
from relphot.exceptions import MembersError
from relphot.lightcurve import compute_light_curves
from relphot.members import (
    build_members,
    load_members_npz,
    save_members_npz,
)
from relphot.reference import (
    build_references,
    select_candidates,
    select_reference_frames_and_stars,
)
from relphot.stats import best_aperture_per_star, compute_star_stats, select_best_aperture
from relphot.tiles import build_tilemap


def _compute_best_aperture(night, tilemap, ref_result, comp_result, settings):
    """Helper to compute best_aper_per_tile and bin_edges for a synthetic night."""
    lc_result = compute_light_curves(night, tilemap, ref_result, comp_result, settings)
    star_stats = compute_star_stats(lc_result)
    best_aper_per_tile, bin_edges = select_best_aperture(
        tilemap, comp_result, star_stats, settings, mag_aper=0
    )
    return best_aper_per_tile, bin_edges


def test_build_members_reproduces_ensemble() -> None:
    """build_members ensemble asserts internally and reproduces comparison_result.ensemble."""
    night, _airmass, _flux0 = make_synthetic_night(n_stars=500, n_frames=30, seed=42)
    settings = Settings()

    variable_mask = np.zeros(night.n_stars, dtype=bool)
    candidates = select_candidates(night, variable_mask, settings, aper=0)
    tilemap = build_tilemap(night, candidates, settings)
    frame_sel = select_reference_frames_and_stars(night, tilemap, candidates, settings, aper=0)
    ref_result = build_references(night, tilemap, frame_sel, settings)
    comp_result = select_comparison_stars(night, tilemap, ref_result, variable_mask, settings)

    best_aper_per_tile, _bin_edges = _compute_best_aperture(
        night, tilemap, ref_result, comp_result, settings
    )

    # This should not raise and should have internal assertions pass
    product = build_members(
        night, tilemap, ref_result, frame_sel.tile_stars, 0,
        comp_result, best_aper_per_tile, settings
    )

    # Verify ensembles match
    assert product.tile_ens.shape == comp_result.ensemble.shape
    np.testing.assert_allclose(
        product.tile_ens, comp_result.ensemble,
        rtol=1e-9, atol=1e-12, equal_nan=True
    )


def test_build_members_raises_on_tampered_ensemble() -> None:
    """build_members raises MembersError when comparison_result.ensemble is tampered."""
    night, _airmass, _flux0 = make_synthetic_night(n_stars=500, n_frames=30, seed=43)
    settings = Settings()

    variable_mask = np.zeros(night.n_stars, dtype=bool)
    candidates = select_candidates(night, variable_mask, settings, aper=0)
    tilemap = build_tilemap(night, candidates, settings)
    frame_sel = select_reference_frames_and_stars(night, tilemap, candidates, settings, aper=0)
    ref_result = build_references(night, tilemap, frame_sel, settings)
    comp_result = select_comparison_stars(night, tilemap, ref_result, variable_mask, settings)

    # Tamper with the ensemble
    comp_result.ensemble[0, 0, 0] *= 1.01

    best_aper_per_tile, _bin_edges = _compute_best_aperture(
        night, tilemap, ref_result, comp_result, settings
    )

    with pytest.raises(MembersError, match="ensemble mismatch"):
        build_members(
            night, tilemap, ref_result, frame_sel.tile_stars, 0,
            comp_result, best_aper_per_tile, settings
        )


def test_reference_weights_sum_to_one() -> None:
    """Reference weights sum to 1 per non-empty tile."""
    night, _airmass, _flux0 = make_synthetic_night(n_stars=500, n_frames=30, seed=44)
    settings = Settings()

    variable_mask = np.zeros(night.n_stars, dtype=bool)
    candidates = select_candidates(night, variable_mask, settings, aper=0)
    tilemap = build_tilemap(night, candidates, settings)
    frame_sel = select_reference_frames_and_stars(night, tilemap, candidates, settings, aper=0)
    ref_result = build_references(night, tilemap, frame_sel, settings)
    comp_result = select_comparison_stars(night, tilemap, ref_result, variable_mask, settings)

    best_aper_per_tile, _bin_edges = _compute_best_aperture(
        night, tilemap, ref_result, comp_result, settings
    )

    product = build_members(
        night, tilemap, ref_result, frame_sel.tile_stars, 0,
        comp_result, best_aper_per_tile, settings
    )

    # Check ref weights sum to 1 per tile
    for t in range(tilemap.n_tiles):
        start = product.ref_offsets[t]
        end = product.ref_offsets[t + 1]
        if end > start:
            w_sum = np.sum(product.ref_weight[start:end])
            np.testing.assert_allclose(w_sum, 1.0, rtol=1e-5, atol=1e-10)


def test_reference_weighted_combination() -> None:
    """Reference weights multiplied by normalised flux equals reference result."""
    night, _airmass, _flux0 = make_synthetic_night(n_stars=500, n_frames=30, seed=45)
    settings = Settings()

    variable_mask = np.zeros(night.n_stars, dtype=bool)
    candidates = select_candidates(night, variable_mask, settings, aper=0)
    tilemap = build_tilemap(night, candidates, settings)
    frame_sel = select_reference_frames_and_stars(night, tilemap, candidates, settings, aper=0)
    ref_result = build_references(night, tilemap, frame_sel, settings)
    comp_result = select_comparison_stars(night, tilemap, ref_result, variable_mask, settings)

    best_aper_per_tile, _bin_edges = _compute_best_aperture(
        night, tilemap, ref_result, comp_result, settings
    )

    product = build_members(
        night, tilemap, ref_result, frame_sel.tile_stars, 0,
        comp_result, best_aper_per_tile, settings
    )

    # Verify reference formula: R[t,j,a] = sum_i(w_i * f_i[j,a] / baseline_i)
    frame_kept = ref_result.frame_kept
    ref_aper = product.ref_aper

    for t in range(tilemap.n_tiles):
        s_t = frame_sel.tile_stars[t]
        start = product.ref_offsets[t]
        end = product.ref_offsets[t + 1]

        if end <= start or len(s_t) < 2:
            continue

        # Get flux and baseline
        f_s = night.flux[s_t, :, ref_aper][:, frame_kept].astype(np.float64)
        # Extract baselines from magnitudes
        ref_mag = product.ref_mag[start:end]
        baseline = 10.0 ** (-0.4 * ref_mag)
        weights = product.ref_weight[start:end]

        # Compute weighted combination: sum_i(w_i * f_i / baseline_i)
        with np.errstate(divide="ignore", invalid="ignore"):
            norm_f = f_s.astype(np.float32) / baseline[:, None]
            weighted_sum = np.sum(
                weights[:, None] * norm_f,
                axis=0
            )

        expected_R = ref_result.R[t, frame_kept, ref_aper]
        np.testing.assert_allclose(
            weighted_sum, expected_R,
            rtol=1e-6, atol=1e-10
        )


def test_comparison_members_per_pair() -> None:
    """Comparison members per pair match core tile selection."""
    night, _airmass, _flux0 = make_synthetic_night(n_stars=500, n_frames=30, seed=46)
    settings = Settings()

    variable_mask = np.zeros(night.n_stars, dtype=bool)
    candidates = select_candidates(night, variable_mask, settings, aper=0)
    tilemap = build_tilemap(night, candidates, settings)
    frame_sel = select_reference_frames_and_stars(night, tilemap, candidates, settings, aper=0)
    ref_result = build_references(night, tilemap, frame_sel, settings)
    comp_result = select_comparison_stars(night, tilemap, ref_result, variable_mask, settings)

    best_aper_per_tile, _bin_edges = _compute_best_aperture(
        night, tilemap, ref_result, comp_result, settings
    )

    product = build_members(
        night, tilemap, ref_result, frame_sel.tile_stars, 0,
        comp_result, best_aper_per_tile, settings
    )

    # For each used pair, check members match expected set
    for pair_idx, (t, a) in enumerate(product.used_pairs):
        comp_start = product.comp_offsets[pair_idx]
        comp_end = product.comp_offsets[pair_idx + 1]
        comp_stars = product.comp_star[comp_start:comp_end]
        comp_weights = product.comp_weight[comp_start:comp_end]

        # Expected: stars in core tile with comparison_result.mask[:, a]
        core_mask = tilemap.core_tile == t
        expected_sel = np.nonzero(comp_result.mask[:, a] & core_mask)[0]

        assert len(comp_stars) == len(expected_sel), \
            f"Pair ({t}, {a}): got {len(comp_stars)} members, expected {len(expected_sel)}"

        np.testing.assert_array_equal(np.sort(comp_stars), np.sort(expected_sel))

        # Weights should sum to 1
        w_sum = np.sum(comp_weights)
        np.testing.assert_allclose(w_sum, 1.0, rtol=1e-5, atol=1e-5)


def test_comparison_median_weights() -> None:
    """For median ensemble_statistic, comparison weights are 1/n_members."""
    from dataclasses import replace

    night, _airmass, _flux0 = make_synthetic_night(n_stars=500, n_frames=30, seed=47)
    settings = Settings()
    settings = replace(settings,
        comparison=replace(settings.comparison, ensemble_statistic="median")
    )

    variable_mask = np.zeros(night.n_stars, dtype=bool)
    candidates = select_candidates(night, variable_mask, settings, aper=0)
    tilemap = build_tilemap(night, candidates, settings)
    frame_sel = select_reference_frames_and_stars(night, tilemap, candidates, settings, aper=0)
    ref_result = build_references(night, tilemap, frame_sel, settings)
    comp_result = select_comparison_stars(night, tilemap, ref_result, variable_mask, settings)

    best_aper_per_tile, _bin_edges = _compute_best_aperture(
        night, tilemap, ref_result, comp_result, settings
    )

    product = build_members(
        night, tilemap, ref_result, frame_sel.tile_stars, 0,
        comp_result, best_aper_per_tile, settings
    )

    # For each pair, all weights should be 1/n_members
    for pair_idx, (_t, _a) in enumerate(product.used_pairs):
        comp_start = product.comp_offsets[pair_idx]
        comp_end = product.comp_offsets[pair_idx + 1]
        n_members = comp_end - comp_start

        if n_members > 0:
            expected_weight = 1.0 / n_members
            weights = product.comp_weight[comp_start:comp_end]
            np.testing.assert_allclose(
                weights, expected_weight,
                rtol=1e-6, atol=1e-10
            )


def test_comparison_normalised_flux() -> None:
    """Normalised comparison flux: nanmedian of each member should be 1."""
    night, _airmass, _flux0 = make_synthetic_night(n_stars=500, n_frames=30, seed=48)
    settings = Settings()

    variable_mask = np.zeros(night.n_stars, dtype=bool)
    candidates = select_candidates(night, variable_mask, settings, aper=0)
    tilemap = build_tilemap(night, candidates, settings)
    frame_sel = select_reference_frames_and_stars(night, tilemap, candidates, settings, aper=0)
    ref_result = build_references(night, tilemap, frame_sel, settings)
    comp_result = select_comparison_stars(night, tilemap, ref_result, variable_mask, settings)

    best_aper_per_tile, _bin_edges = _compute_best_aperture(
        night, tilemap, ref_result, comp_result, settings
    )

    product = build_members(
        night, tilemap, ref_result, frame_sel.tile_stars, 0,
        comp_result, best_aper_per_tile, settings
    )

    # Each member's nanmedian(norm_flux) should be 1
    for member_idx in range(len(product.comp_star)):
        norm_f = product.comp_norm_flux[member_idx, :]
        median_norm_f = np.nanmedian(norm_f)
        np.testing.assert_allclose(
            median_norm_f, 1.0,
            rtol=1e-6, atol=1e-6
        )


def test_comparison_clipped_frames_are_subset_of_finite() -> None:
    """Clipped frame indices are a subset of finite-flux frames."""
    night, _airmass, _flux0 = make_synthetic_night(n_stars=500, n_frames=30, seed=49)
    settings = Settings()

    variable_mask = np.zeros(night.n_stars, dtype=bool)
    candidates = select_candidates(night, variable_mask, settings, aper=0)
    tilemap = build_tilemap(night, candidates, settings)
    frame_sel = select_reference_frames_and_stars(night, tilemap, candidates, settings, aper=0)
    ref_result = build_references(night, tilemap, frame_sel, settings)
    comp_result = select_comparison_stars(night, tilemap, ref_result, variable_mask, settings)

    best_aper_per_tile, _bin_edges = _compute_best_aperture(
        night, tilemap, ref_result, comp_result, settings
    )

    product = build_members(
        night, tilemap, ref_result, frame_sel.tile_stars, 0,
        comp_result, best_aper_per_tile, settings
    )

    # For each member, clipped frames should be subset of finite frames
    for member_idx in range(len(product.comp_star)):
        clip_start = product.comp_clip_offsets[member_idx]
        clip_end = product.comp_clip_offsets[member_idx + 1]
        clipped_frames = product.comp_clip_frames[clip_start:clip_end]

        norm_f = product.comp_norm_flux[member_idx, :]
        finite_frames = np.nonzero(np.isfinite(norm_f))[0]

        # All clipped frames should be in finite frames
        for frame_idx in clipped_frames:
            assert frame_idx in finite_frames, \
                f"Clipped frame {frame_idx} not in finite frames for member {member_idx}"


def test_every_core_star_with_best_aper_in_used_pairs() -> None:
    """Every star with core_tile >= 0 and a best aperture has (tile, best_aper) in used_pairs."""
    night, _airmass, _flux0 = make_synthetic_night(n_stars=500, n_frames=30, seed=50)
    settings = Settings()

    variable_mask = np.zeros(night.n_stars, dtype=bool)
    candidates = select_candidates(night, variable_mask, settings, aper=0)
    tilemap = build_tilemap(night, candidates, settings)
    frame_sel = select_reference_frames_and_stars(night, tilemap, candidates, settings, aper=0)
    ref_result = build_references(night, tilemap, frame_sel, settings)
    comp_result = select_comparison_stars(night, tilemap, ref_result, variable_mask, settings)

    best_aper_per_tile, bin_edges = _compute_best_aperture(
        night, tilemap, ref_result, comp_result, settings
    )
    star_best_aper = best_aperture_per_star(
        tilemap, comp_result, best_aper_per_tile, bin_edges, mag_aper=0
    )

    product = build_members(
        night, tilemap, ref_result, frame_sel.tile_stars, 0,
        comp_result, best_aper_per_tile, settings
    )

    used_pairs_set = set(map(tuple, product.used_pairs))

    for star_idx in range(night.n_stars):
        core_t = tilemap.core_tile[star_idx]
        best_a = star_best_aper[star_idx]

        if core_t >= 0 and best_a >= 0:
            assert (core_t, best_a) in used_pairs_set, \
                f"Star {star_idx}: (tile={core_t}, best_aper={best_a}) not in used_pairs"


def test_recover_reference_stars_matches_frame_sel() -> None:
    """recover_reference_stars returns tile_stars equal to FrameSelection.tile_stars."""
    from relphot.members import recover_reference_stars

    night, _airmass, _flux0 = make_synthetic_night(n_stars=500, n_frames=30, seed=51)
    settings = Settings()

    variable_mask = np.zeros(night.n_stars, dtype=bool)
    candidates = select_candidates(night, variable_mask, settings, aper=0)
    tilemap = build_tilemap(night, candidates, settings)
    frame_sel = select_reference_frames_and_stars(night, tilemap, candidates, settings, aper=0)
    ref_result = build_references(night, tilemap, frame_sel, settings)

    recovered_tile_stars, recovered_aper = recover_reference_stars(
        night, tilemap, ref_result, settings
    )

    assert recovered_aper == 0
    assert len(recovered_tile_stars) == len(frame_sel.tile_stars)

    for t in range(tilemap.n_tiles):
        np.testing.assert_array_equal(
            np.sort(recovered_tile_stars[t]),
            np.sort(frame_sel.tile_stars[t])
        )


def test_recover_reference_stars_raises_on_perturbed_R() -> None:
    """recover_reference_stars raises MembersError when reference_result.R is perturbed."""
    from relphot.members import recover_reference_stars

    night, _airmass, _flux0 = make_synthetic_night(n_stars=500, n_frames=30, seed=52)
    settings = Settings()

    variable_mask = np.zeros(night.n_stars, dtype=bool)
    candidates = select_candidates(night, variable_mask, settings, aper=0)
    tilemap = build_tilemap(night, candidates, settings)
    frame_sel = select_reference_frames_and_stars(night, tilemap, candidates, settings, aper=0)
    ref_result = build_references(night, tilemap, frame_sel, settings)

    # Perturb R
    ref_result.R[0, 0, 0] *= 1.01

    with pytest.raises(MembersError):
        recover_reference_stars(night, tilemap, ref_result, settings)


def test_save_load_members_npz_round_trip() -> None:
    """save_members_npz and load_members_npz round trip exactly."""
    import tempfile
    from pathlib import Path

    night, _airmass, _flux0 = make_synthetic_night(n_stars=500, n_frames=30, seed=53)
    settings = Settings()

    variable_mask = np.zeros(night.n_stars, dtype=bool)
    candidates = select_candidates(night, variable_mask, settings, aper=0)
    tilemap = build_tilemap(night, candidates, settings)
    frame_sel = select_reference_frames_and_stars(night, tilemap, candidates, settings, aper=0)
    ref_result = build_references(night, tilemap, frame_sel, settings)
    comp_result = select_comparison_stars(night, tilemap, ref_result, variable_mask, settings)

    best_aper_per_tile, _bin_edges = _compute_best_aperture(
        night, tilemap, ref_result, comp_result, settings
    )

    product = build_members(
        night, tilemap, ref_result, frame_sel.tile_stars, 0,
        comp_result, best_aper_per_tile, settings
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "members.npz"
        save_members_npz(product, out_path)
        product2 = load_members_npz(out_path)

        # Compare all fields
        assert product2.n_frames == product.n_frames
        assert product2.n_aper == product.n_aper
        assert product2.ref_aper == product.ref_aper
        assert product2.meta == product.meta

        for attr in (
            "tile_xmin", "tile_xmax", "tile_ymin", "tile_ymax",
            "tile_n_core", "tile_n_extended", "ref_offsets", "ref_star",
            "ref_ra", "ref_dec", "ref_mag", "ref_weight", "ref_in_core",
            "tile_R", "tile_sigma_R", "tile_ens", "tile_sigma_ens",
            "tile_n_ensemble", "tile_n_rounds", "used_pairs", "comp_offsets",
            "comp_star", "comp_ra", "comp_dec", "comp_mag", "comp_weight",
            "comp_n_clipped", "comp_clip_offsets", "comp_clip_frames", "comp_norm_flux"
        ):
            a = getattr(product, attr)
            b = getattr(product2, attr)
            np.testing.assert_array_equal(a, b, strict=True)
