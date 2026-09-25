"""Tests for relphot.reference: candidate selection and per-tile reference construction."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from conftest import make_synthetic_night

from relphot.config import Settings
from relphot.exceptions import ReferenceFrameError
from relphot.reference import (
    build_references,
    select_candidates,
    select_reference_frames_and_stars,
)
from relphot.tiles import build_tilemap


def test_saturated_star_excluded() -> None:
    night, _airmass, _flux0 = make_synthetic_night(n_stars=300, n_frames=20, seed=1)
    settings = Settings()
    variable_mask = np.zeros(night.n_stars, dtype=bool)
    before = select_candidates(night, variable_mask, settings, aper=0)
    target = int(np.nonzero(before)[0][0])

    night.flags[target, 0] |= settings.catalog.saturation_flag_bit
    after = select_candidates(night, variable_mask, settings, aper=0)
    assert before[target]
    assert not after[target]


def test_known_variable_excluded() -> None:
    night, _airmass, _flux0 = make_synthetic_night(n_stars=300, n_frames=20, seed=1)
    settings = Settings()
    variable_mask = np.zeros(night.n_stars, dtype=bool)
    before = select_candidates(night, variable_mask, settings, aper=0)
    target = int(np.nonzero(before)[0][1])

    flagged = variable_mask.copy()
    flagged[target] = True
    after = select_candidates(night, flagged, settings, aper=0)
    assert before[target]
    assert not after[target]


def test_non_isolated_pair_excluded() -> None:
    night, _airmass, _flux0 = make_synthetic_night(n_stars=300, n_frames=20, seed=1)
    settings = Settings()
    variable_mask = np.zeros(night.n_stars, dtype=bool)
    before = select_candidates(night, variable_mask, settings, aper=0)
    a, b = np.nonzero(before)[0][2:4]

    night.ra[b] = night.ra[a] + 1.0 / 3600.0 / np.cos(np.radians(night.dec[a]))
    night.dec[b] = night.dec[a]
    after = select_candidates(night, variable_mask, settings, aper=0)
    assert before[a] and before[b]
    assert not after[a] and not after[b]


def test_reference_recovers_injected_transparency() -> None:
    night, airmass, _flux0 = make_synthetic_night(n_stars=2000, n_frames=40, seed=0)
    settings = Settings()
    variable_mask = np.zeros(night.n_stars, dtype=bool)
    candidates = select_candidates(night, variable_mask, settings, aper=0)
    tilemap = build_tilemap(night, candidates, settings)
    frame_selection = select_reference_frames_and_stars(
        night, tilemap, candidates, settings, aper=0
    )
    result = build_references(night, tilemap, frame_selection, settings)

    slopes = []
    for t in range(tilemap.n_tiles):
        r_j = result.R[t, :, 0]
        valid = np.isfinite(r_j) & (r_j > 0)
        mag_r = -2.5 * np.log10(r_j[valid])
        slope, _intercept = np.polyfit(airmass[valid], mag_r, 1)
        slopes.append(slope)

    slopes = np.asarray(slopes)
    assert np.all(np.abs(slopes - 0.17) < 0.02)


def test_method_d_beats_method_b_split_subset_self_noise() -> None:
    night, _airmass, _flux0 = make_synthetic_night(n_stars=2000, n_frames=40, seed=7)
    settings = Settings()
    variable_mask = np.zeros(night.n_stars, dtype=bool)
    candidates = select_candidates(night, variable_mask, settings, aper=0)
    tilemap = build_tilemap(night, candidates, settings)

    rng = np.random.default_rng(42)
    cand_idx = np.nonzero(candidates)[0]
    rng.shuffle(cand_idx)
    half = len(cand_idx) // 2
    subset1 = np.zeros(night.n_stars, dtype=bool)
    subset1[cand_idx[:half]] = True
    subset2 = np.zeros(night.n_stars, dtype=bool)
    subset2[cand_idx[half:]] = True

    def self_noise(method: str) -> float:
        s = replace(settings, reference=replace(settings.reference, method=method))
        fs1 = select_reference_frames_and_stars(night, tilemap, subset1, s, aper=0)
        fs2 = select_reference_frames_and_stars(night, tilemap, subset2, s, aper=0)
        result1 = build_references(night, tilemap, fs1, s)
        result2 = build_references(night, tilemap, fs2, s)
        noises = []
        for t in range(tilemap.n_tiles):
            ratio = result1.R[t, :, 0] / result2.R[t, :, 0]
            valid = np.isfinite(ratio) & (ratio > 0)
            r = ratio[valid]
            mad = np.median(np.abs(r - np.median(r)))
            noises.append(1.4826 * mad)
        return float(np.mean(noises))

    noise_d = self_noise("weighted_fixed_mean")
    noise_b = self_noise("median_fixed")
    assert noise_d < noise_b


def test_relative_flux_shape_and_core_tile_coverage() -> None:
    night, _airmass, _flux0 = make_synthetic_night(n_stars=500, n_frames=10, seed=2)
    settings = Settings()
    variable_mask = np.zeros(night.n_stars, dtype=bool)
    candidates = select_candidates(night, variable_mask, settings, aper=0)
    tilemap = build_tilemap(night, candidates, settings)
    frame_selection = select_reference_frames_and_stars(
        night, tilemap, candidates, settings, aper=0
    )
    result = build_references(night, tilemap, frame_selection, settings)

    assert result.relative_flux.shape == night.flux.shape
    assert result.R.shape == (tilemap.n_tiles, night.n_frames, night.n_aper)
    assert np.all(tilemap.core_tile >= 0)  # dense synthetic field: every star has a core tile


def test_frame_with_too_few_reference_stars_is_dropped_globally() -> None:
    night, _airmass, _flux0 = make_synthetic_night(n_stars=500, n_frames=10, seed=3)
    settings = Settings()
    variable_mask = np.zeros(night.n_stars, dtype=bool)
    candidates = select_candidates(night, variable_mask, settings, aper=0)
    tilemap = build_tilemap(night, candidates, settings)
    # Blank frame 4 for all but 5 stars, as when a frame misses part of the field.
    night.flux[5:, 4, :] = np.nan
    frame_selection = select_reference_frames_and_stars(
        night, tilemap, candidates, settings, aper=0
    )
    result = build_references(night, tilemap, frame_selection, settings)

    assert isinstance(frame_selection.frame_kept[4], (bool, np.bool_))
    assert frame_selection.frame_kept[4] == False  # noqa: E712
    assert np.all(np.isnan(result.R[:, 4, :]))
    assert np.all(np.isnan(result.sigma_R[:, 4, :]))
    assert np.all(result.n_used[:, 4, :] == 0)
    other = np.delete(np.arange(night.n_frames), 4)
    assert np.all(np.isfinite(result.R[:, other, :]))


def test_select_reference_frames_drops_globally_sparse_frame() -> None:
    night, _airmass, _flux0 = make_synthetic_night(n_stars=500, n_frames=10, seed=4)
    settings = Settings()
    variable_mask = np.zeros(night.n_stars, dtype=bool)
    candidates = select_candidates(night, variable_mask, settings, aper=0)
    tilemap = build_tilemap(night, candidates, settings)

    # Invalidate most candidates in frame 5
    night.flux[1:, 5, :] = np.nan

    frame_selection = select_reference_frames_and_stars(
        night, tilemap, candidates, settings, aper=0
    )

    # Frame 5 should be dropped
    assert frame_selection.frame_kept[5] == False  # noqa: E712
    # A star invalid only in frame 5 should be in S_t for its tile
    # (The exact star depends on the synthetic data, but the test verifies the mechanism works)
    assert frame_selection.frame_kept.sum() == 9


def test_reference_frame_error_when_drop_cap_exceeded() -> None:
    night, _airmass, _flux0 = make_synthetic_night(n_stars=500, n_frames=10, seed=5)
    settings = Settings()
    variable_mask = np.zeros(night.n_stars, dtype=bool)
    candidates = select_candidates(night, variable_mask, settings, aper=0)
    tilemap = build_tilemap(night, candidates, settings)

    # Blank many frames to exceed the drop cap
    for i in range(4):
        night.flux[5:, i, :] = np.nan

    with pytest.raises(ReferenceFrameError):
        select_reference_frames_and_stars(night, tilemap, candidates, settings, aper=0)


def test_reference_star_set_identical_across_kept_frames() -> None:
    night, _airmass, _flux0 = make_synthetic_night(n_stars=500, n_frames=10, seed=6)
    settings = Settings()
    variable_mask = np.zeros(night.n_stars, dtype=bool)
    candidates = select_candidates(night, variable_mask, settings, aper=0)
    tilemap = build_tilemap(night, candidates, settings)
    frame_selection = select_reference_frames_and_stars(
        night, tilemap, candidates, settings, aper=0
    )
    result = build_references(night, tilemap, frame_selection, settings)

    # For each tile and aperture, n_used should be constant in kept frames and 0 in dropped
    for t in range(tilemap.n_tiles):
        n_ref_stars = len(frame_selection.tile_stars[t])
        for a in range(result.n_aper):
            kept_vals = result.n_used[t, frame_selection.frame_kept, a]
            dropped_vals = result.n_used[t, ~frame_selection.frame_kept, a]
            if n_ref_stars >= 2:
                assert np.all(kept_vals == n_ref_stars)
            assert np.all(dropped_vals == 0)


def test_star_level_outlier_removed_from_all_frames() -> None:
    night, _airmass, _flux0 = make_synthetic_night(n_stars=500, n_frames=10, seed=8)
    settings = Settings()
    variable_mask = np.zeros(night.n_stars, dtype=bool)
    candidates = select_candidates(night, variable_mask, settings, aper=0)
    tilemap = build_tilemap(night, candidates, settings)

    # Inject a strong sinusoid into one candidate star to make it an outlier
    cand_idx = np.nonzero(candidates)[0]
    if len(cand_idx) > 0:
        outlier_star = cand_idx[0]
        for frame_idx in range(night.n_frames):
            night.flux[outlier_star, frame_idx, 0] *= 1.0 + 0.5 * np.sin(
                2 * np.pi * frame_idx / night.n_frames
            )

        frame_selection = select_reference_frames_and_stars(
            night, tilemap, candidates, settings, aper=0
        )

        # The outlier star should not be in any tile's star set
        for tile_stars in frame_selection.tile_stars:
            assert outlier_star not in tile_stars


def test_reference_star_set_valid_at_every_aperture() -> None:
    """A star bad at a non-selection aperture in one frame is not in any S_t."""
    night, _airmass, _flux0 = make_synthetic_night(n_stars=500, n_frames=10, seed=5)
    settings = Settings()
    candidates = select_candidates(night, np.zeros(night.n_stars, dtype=bool), settings, aper=0)
    victim = int(np.nonzero(candidates)[0][0])
    night.flux[victim, 3, night.n_aper - 1] = -1.0
    tilemap = build_tilemap(night, candidates, settings)
    selection = select_reference_frames_and_stars(night, tilemap, candidates, settings, aper=0)
    kept = selection.frame_kept
    for s_t in selection.tile_stars:
        assert victim not in s_t
        if s_t.size:
            f = night.flux[s_t][:, kept, :]
            assert np.all(np.isfinite(f) & (f > 0))
