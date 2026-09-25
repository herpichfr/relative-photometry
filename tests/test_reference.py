"""Tests for relphot.reference: candidate selection and per-tile reference construction."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from conftest import make_synthetic_night

from relphot.config import Settings
from relphot.reference import build_references, select_candidates
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
    result = build_references(night, tilemap, candidates, settings)

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
        result1 = build_references(night, tilemap, subset1, s)
        result2 = build_references(night, tilemap, subset2, s)
        noises = []
        for t in range(tilemap.n_tiles):
            ratio = result1.R[t, :, 0] / result2.R[t, :, 0]
            valid = np.isfinite(ratio) & (ratio > 0)
            r = ratio[valid]
            mad = np.median(np.abs(r - np.median(r)))
            noises.append(1.4826 * mad)
        return float(np.mean(noises))

    noise_d = self_noise("weighted_clipped_mean")
    noise_b = self_noise("median_normalised")
    assert noise_d < noise_b


def test_relative_flux_shape_and_core_tile_coverage() -> None:
    night, _airmass, _flux0 = make_synthetic_night(n_stars=500, n_frames=10, seed=2)
    settings = Settings()
    variable_mask = np.zeros(night.n_stars, dtype=bool)
    candidates = select_candidates(night, variable_mask, settings, aper=0)
    tilemap = build_tilemap(night, candidates, settings)
    result = build_references(night, tilemap, candidates, settings)

    assert result.relative_flux.shape == night.flux.shape
    assert result.R.shape == (tilemap.n_tiles, night.n_frames, night.n_aper)
    assert np.all(tilemap.core_tile >= 0)  # dense synthetic field: every star has a core tile


def test_frame_with_too_few_reference_stars_is_nan() -> None:
    night, _airmass, _flux0 = make_synthetic_night(n_stars=500, n_frames=10, seed=3)
    settings = Settings()
    variable_mask = np.zeros(night.n_stars, dtype=bool)
    candidates = select_candidates(night, variable_mask, settings, aper=0)
    tilemap = build_tilemap(night, candidates, settings)
    # Blank frame 4 for all but 5 stars, as when a frame misses part of the field.
    night.flux[5:, 4, :] = np.nan
    result = build_references(night, tilemap, candidates, settings)

    assert np.all(np.isnan(result.R[:, 4, :]))
    assert np.all(np.isnan(result.sigma_R[:, 4, :]))
    assert np.all(result.n_used[:, 4, :] < settings.reference.min_used_per_frame)
    other = np.delete(np.arange(night.n_frames), 4)
    assert np.all(np.isfinite(result.R[:, other, :]))
