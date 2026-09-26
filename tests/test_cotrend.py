"""Tests for relphot.cotrend: cotrending basis vectors and systematic-frame detection."""

from __future__ import annotations

import numpy as np

from relphot.config import Settings
from relphot.cotrend import compute_cbvs, compute_frame_error_scale, detect_systematic_frames


class _FakeTileMap:
    def __init__(self, core_tile: np.ndarray) -> None:
        self.core_tile = core_tile
        self.n_tiles = int(core_tile.max()) + 1 if core_tile.size else 0


class _FakeComparisonResult:
    def __init__(self, mask: np.ndarray) -> None:
        self.mask = mask


def _synthetic_ensemble(
    n_stars: int = 60, n_frames: int = 200, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """A common-mode ramp shared by every star, plus independent per-star noise.

    Returns ``(lc, common_mode)`` -- ``lc`` is ``(n_stars, n_frames, 1)``,
    ``common_mode`` is ``(n_frames,)`` the injected shared signal (mean-zero).
    """
    rng = np.random.default_rng(seed)
    phase = np.linspace(0.0, 1.0, n_frames)
    common_mode = 0.02 * np.sin(2.0 * np.pi * phase * 1.5)
    common_mode -= common_mode.mean()
    noise = rng.standard_normal((n_stars, n_frames)) * 0.003
    lc = 1.0 + common_mode[None, :] + noise
    return lc[:, :, None], common_mode


def test_compute_cbvs_recovers_common_mode_signal() -> None:
    lc, common_mode = _synthetic_ensemble()
    n_stars = lc.shape[0]
    core_tile = np.zeros(n_stars, dtype=np.int64)
    mask = np.ones((n_stars, 1), dtype=bool)
    tilemap = _FakeTileMap(core_tile)
    comparison_result = _FakeComparisonResult(mask)
    frame_kept = np.ones(lc.shape[1], dtype=bool)
    settings = Settings()

    result = compute_cbvs(tilemap, comparison_result, lc, frame_kept, settings.search)

    assert result.basis.shape[0] == 1  # one tile
    assert result.n_used[0, 0] == n_stars
    first_cbv = result.basis[0, 0, 0, :]
    assert np.all(np.isfinite(first_cbv))

    # The leading CBV should closely trace the injected common-mode signal,
    # up to an overall sign and scale (PCA components are only defined that
    # way).
    common_norm = (common_mode - common_mode.mean()) / np.std(common_mode)
    cbv_norm = (first_cbv - first_cbv.mean()) / np.std(first_cbv)
    corr = float(np.abs(np.corrcoef(common_norm, cbv_norm)[0, 1]))
    assert corr > 0.95

    # The first component should explain the bulk of the variance.
    assert result.explained_variance_ratio[0, 0, 0] > 0.5


def test_compute_cbvs_too_few_comparison_stars_warns_and_skips(caplog) -> None:
    lc, _common = _synthetic_ensemble(n_stars=2)
    core_tile = np.zeros(2, dtype=np.int64)
    mask = np.ones((2, 1), dtype=bool)
    tilemap = _FakeTileMap(core_tile)
    comparison_result = _FakeComparisonResult(mask)
    frame_kept = np.ones(lc.shape[1], dtype=bool)
    settings = Settings()

    result = compute_cbvs(tilemap, comparison_result, lc, frame_kept, settings.search)
    assert result.n_used[0, 0] == 0
    assert np.all(np.isnan(result.basis[0, 0]))
    assert "comparison stars" in caplog.text.lower()


def test_detect_systematic_frames_flags_injected_glitch() -> None:
    n_stars, n_frames = 80, 200
    rng = np.random.default_rng(1)
    lc = 1.0 + rng.standard_normal((n_stars, n_frames, 1)) * 0.003
    glitch_frames = slice(100, 110)
    lc[:, glitch_frames, 0] -= 0.03  # a shared, sharp dip well above the noise

    core_tile = np.zeros(n_stars, dtype=np.int64)
    mask = np.ones((n_stars, 1), dtype=bool)
    tilemap = _FakeTileMap(core_tile)
    comparison_result = _FakeComparisonResult(mask)
    frame_kept = np.ones(n_frames, dtype=bool)
    settings = Settings()

    flagged = detect_systematic_frames(tilemap, comparison_result, lc, frame_kept, settings.search)
    assert flagged.shape == (n_frames,)
    assert flagged[glitch_frames].all()
    # Quiet frames well away from the glitch should not be flagged.
    assert not flagged[:50].any()
    assert not flagged[150:].any()


def test_detect_systematic_frames_quiet_night_flags_nothing() -> None:
    n_stars, n_frames = 60, 150
    rng = np.random.default_rng(2)
    lc = 1.0 + rng.standard_normal((n_stars, n_frames, 1)) * 0.003
    core_tile = np.zeros(n_stars, dtype=np.int64)
    mask = np.ones((n_stars, 1), dtype=bool)
    tilemap = _FakeTileMap(core_tile)
    comparison_result = _FakeComparisonResult(mask)
    frame_kept = np.ones(n_frames, dtype=bool)
    settings = Settings()

    flagged = detect_systematic_frames(tilemap, comparison_result, lc, frame_kept, settings.search)
    assert not flagged.any()


def test_compute_frame_error_scale_flags_noisier_stretch() -> None:
    """A stretch where the comparison ensemble is genuinely noisier gets a > 1 scale.

    Also checks the [min, max] clip and the "no rescaling" fallback for a
    tile/aperture with too few comparison stars.
    """
    n_stars, n_frames = 80, 200
    rng = np.random.default_rng(5)
    lc = 1.0 + rng.standard_normal((n_stars, n_frames, 1)) * 0.003
    lc_err = np.full((n_stars, n_frames, 1), 0.003)
    noisy = slice(150, 200)
    lc[:, noisy, 0] += rng.standard_normal((n_stars, 50)) * 0.02  # extra scatter, same stars

    core_tile = np.zeros(n_stars, dtype=np.int64)
    mask = np.ones((n_stars, 1), dtype=bool)
    tilemap = _FakeTileMap(core_tile)
    comparison_result = _FakeComparisonResult(mask)
    frame_kept = np.ones(n_frames, dtype=bool)
    settings = Settings()

    scale = compute_frame_error_scale(
        tilemap, comparison_result, lc, lc_err, frame_kept, settings.search
    )
    assert scale.shape == (1, 1, n_frames)
    assert np.nanmedian(scale[0, 0, :100]) < np.nanmedian(scale[0, 0, noisy])
    assert scale[0, 0, noisy].mean() > 1.5
    assert scale.min() >= settings.search.frame_error_scale_min
    assert scale.max() <= settings.search.frame_error_scale_max


def test_compute_frame_error_scale_too_few_comparators_no_rescaling(caplog) -> None:
    n_stars, n_frames = 2, 100
    lc = np.ones((n_stars, n_frames, 1))
    lc_err = np.full((n_stars, n_frames, 1), 0.003)
    core_tile = np.zeros(n_stars, dtype=np.int64)
    mask = np.ones((n_stars, 1), dtype=bool)
    tilemap = _FakeTileMap(core_tile)
    comparison_result = _FakeComparisonResult(mask)
    frame_kept = np.ones(n_frames, dtype=bool)
    settings = Settings()

    scale = compute_frame_error_scale(
        tilemap, comparison_result, lc, lc_err, frame_kept, settings.search
    )
    assert np.all(scale == 1.0)
    assert "comparison stars" in caplog.text.lower()
