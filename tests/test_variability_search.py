"""Tests for relphot.variables' Stage-6c variability search (compute_star_variability)."""

from __future__ import annotations

import numpy as np

from relphot.config import Settings
from relphot.cotrend import compute_cbvs, detect_systematic_frames
from relphot.variables import compute_star_variability


class _FrameMeta:
    def __init__(self, bjd: float) -> None:
        self.bjd_tdb = bjd


class _FakeNight:
    def __init__(self, n_stars: int, n_frames: int, n_aper: int, bjd: np.ndarray) -> None:
        self.n_stars = n_stars
        self.n_frames = n_frames
        self.n_aper = n_aper
        self.frame_meta = [_FrameMeta(float(b)) for b in bjd]


class _FakeTileMap:
    def __init__(self, core_tile: np.ndarray) -> None:
        self.core_tile = core_tile
        self.n_tiles = int(core_tile.max()) + 1 if core_tile.size else 0


class _FakeComparisonResult:
    def __init__(self, mask: np.ndarray, mag: np.ndarray) -> None:
        self.mask = mask
        self.mag = mag


def _scenario(n_stars: int = 100, n_frames: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    cadence_d = 31.7 / 86400.0
    bjd = 2460930.5 + np.arange(n_frames) * cadence_d
    flux = 1.0 + rng.standard_normal((n_stars, n_frames)) * 0.004
    lc = flux[:, :, None]
    lc_err = np.full((n_stars, n_frames, 1), 0.004)
    epoch_ok = np.ones((n_stars, n_frames), dtype=bool)
    frame_kept = np.ones(n_frames, dtype=bool)
    star_best_aper = np.zeros(n_stars, dtype=np.int64)
    core_tile = np.zeros(n_stars, dtype=np.int64)
    mag = rng.uniform(12.0, 16.0, (n_stars, 1))
    night = _FakeNight(n_stars, n_frames, 1, bjd)
    tilemap = _FakeTileMap(core_tile)
    return night, tilemap, lc, lc_err, epoch_ok, frame_kept, star_best_aper, bjd, mag


def _run(night, tilemap, lc, lc_err, epoch_ok, frame_kept, star_best_aper, mask, mag):
    settings = Settings()
    comparison_result = _FakeComparisonResult(mask, mag)
    cotrend_result = compute_cbvs(tilemap, comparison_result, lc, frame_kept, settings.search)
    systematic_frames = detect_systematic_frames(
        tilemap, comparison_result, lc, frame_kept, settings.search
    )
    return compute_star_variability(
        night, tilemap, comparison_result, cotrend_result, lc, lc_err, epoch_ok, frame_kept,
        star_best_aper, systematic_frames, settings,
    )


def test_sinusoidal_variable_detected_with_correct_period() -> None:
    n_stars, n_comp, target = 100, 80, 99
    night, tilemap, lc, lc_err, epoch_ok, frame_kept, star_best_aper, bjd, mag = _scenario(
        n_stars=n_stars, seed=11
    )
    period_d = 1.5 / 24.0
    amplitude = 0.03
    lc[target, :, 0] += amplitude * np.sin(2.0 * np.pi * (bjd - bjd[0]) / period_d)

    mask = np.zeros((n_stars, 1), dtype=bool)
    mask[:n_comp, 0] = True
    result = _run(night, tilemap, lc, lc_err, epoch_ok, frame_kept, star_best_aper, mask, mag)

    assert result.searched[target]
    assert result.variable_candidate[target]
    assert result.variable_class[target] == "periodic"
    assert np.isfinite(result.ls_period_days[target])
    assert abs(result.ls_period_days[target] - period_d) / period_d < 0.1


def test_white_noise_star_not_flagged() -> None:
    n_stars, n_comp, target = 80, 60, 79
    night, tilemap, lc, lc_err, epoch_ok, frame_kept, star_best_aper, _bjd, mag = _scenario(
        n_stars=n_stars, seed=12
    )
    mask = np.zeros((n_stars, 1), dtype=bool)
    mask[:n_comp, 0] = True
    result = _run(night, tilemap, lc, lc_err, epoch_ok, frame_kept, star_best_aper, mask, mag)

    assert result.searched[target]
    assert not result.variable_candidate[target]
    assert result.variable_class[target] == "none"


def test_shared_systematic_epochs_excluded_from_variability() -> None:
    """A star whose only 'variability' is a shared glitch is not credited as a real variable."""
    n_stars, n_comp, target = 100, 80, 99
    night, tilemap, lc, lc_err, epoch_ok, frame_kept, star_best_aper, bjd, mag = _scenario(
        n_stars=n_stars, n_frames=400, seed=13
    )
    glitch = np.abs(bjd - bjd[200]) < (10 * 31.7 / 86400.0)
    # Every comparison star (and the target) share the same glitch.
    lc[:n_comp, glitch, 0] -= 0.02
    lc[target, glitch, 0] -= 0.02

    mask = np.zeros((n_stars, 1), dtype=bool)
    mask[:n_comp, 0] = True
    result = _run(night, tilemap, lc, lc_err, epoch_ok, frame_kept, star_best_aper, mask, mag)

    assert result.searched[target]
    # Whether or not the glitch alone crossed the excess threshold, it must
    # never be credited as a genuine variable once explained by a shared,
    # instrumental epoch.
    if result.excess[target] >= Settings().search.excess_rms_threshold:
        assert result.systematic_excluded[target] or not result.variable_candidate[target]


def test_eclipse_covering_20_percent_of_points_detected() -> None:
    """A dip covering only ~20% of points must not be missed by a MAD-blind rms.

    ``excess`` is deliberately defined as ``rms_std / floor`` (a plain std,
    not a MAD) for exactly this case: a MAD-based scatter is insensitive to
    an eclipse/dip affecting less than about half the points.
    """
    n_stars, n_comp, target = 100, 80, 99
    night, tilemap, lc, lc_err, epoch_ok, frame_kept, star_best_aper, bjd, mag = _scenario(
        n_stars=n_stars, seed=14
    )
    depth = 0.03
    in_eclipse = np.zeros(bjd.size, dtype=bool)
    in_eclipse[: bjd.size // 5] = True  # first 20% of points
    lc[target, in_eclipse, 0] -= depth

    mask = np.zeros((n_stars, 1), dtype=bool)
    mask[:n_comp, 0] = True
    result = _run(night, tilemap, lc, lc_err, epoch_ok, frame_kept, star_best_aper, mask, mag)

    assert result.searched[target]
    assert np.isfinite(result.rms_std[target]) and np.isfinite(result.rms_robust[target])
    assert result.excess[target] >= Settings().search.excess_rms_threshold
    assert result.variable_candidate[target]
