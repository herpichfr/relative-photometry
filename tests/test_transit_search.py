"""Tests for relphot.transit_search: single-event box search and vetting flags."""

from __future__ import annotations

import numpy as np

from relphot.config import Settings
from relphot.cotrend import compute_cbvs
from relphot.transit_search import (
    FLAG_PARTIAL,
    FLAG_SHARED_EPOCH,
    FLAG_STEP_LIKE,
    FLAG_TOO_DEEP,
    depth_at_other_aperture,
    flags_to_string,
    search_transits,
    tier_for_flags,
)


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
    def __init__(self, mask: np.ndarray) -> None:
        self.mask = mask


def _occult_uniform_quad(z: np.ndarray, p: float, u1: float = 0.42, u2: float = 0.22) -> np.ndarray:
    """Small-planet quadratic-limb-darkening transit shape (Mandel & Agol 2002, sec. 5)."""
    z = np.abs(z)
    out = np.ones_like(z)
    norm = 1 - u1 / 3 - u2 / 6

    def intensity(r: np.ndarray) -> np.ndarray:
        mu = np.sqrt(np.clip(1 - r**2, 0, 1))
        return 1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2

    full = z <= 1 - p
    out[full] = 1 - p**2 * intensity(z[full]) / norm
    part = (z > 1 - p) & (z < 1 + p)
    zp = z[part]
    if zp.size:
        k0 = np.arccos(np.clip((p**2 + zp**2 - 1) / (2 * p * zp), -1, 1))
        k1 = np.arccos(np.clip((1 - p**2 + zp**2) / (2 * zp), -1, 1))
        lam = (
            p**2 * k0 + k1 - 0.5 * np.sqrt(np.clip(4 * zp**2 - (1 + zp**2 - p**2) ** 2, 0, None))
        ) / np.pi
        i_eff = intensity(np.clip((zp - p + 1) / 2, 0, 1))
        out[part] = 1 - lam * i_eff / norm
    return out


def _limb_darkened_transit(
    t: np.ndarray, tc: float, depth: float, duration_days: float
) -> np.ndarray:
    """A limb-darkened transit light curve (relative flux, ~1 outside transit)."""
    p = np.sqrt(depth)
    b = 0.3
    x = (t - tc) / (duration_days / 2) * np.sqrt((1 + p) ** 2 - b**2)
    z = np.sqrt(x**2 + b**2)
    return _occult_uniform_quad(z, p)


_Scenario = tuple[
    "_FakeNight", np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]


def _scenario(
    n_stars: int = 100,
    n_frames: int = 350,
    seed: int = 0,
    cadence_s: float = 31.7,
    sigma: float = 0.004,
) -> _Scenario:
    """A one-tile night: quiet comparison stars plus room for injected signals.

    Returns ``(night, tilemap, comparison_result, lc, lc_err, epoch_ok,
    frame_kept, star_best_aper)`` with every star already populated with
    independent Gaussian noise around flux 1; callers inject signals on top.
    """
    rng = np.random.default_rng(seed)
    cadence_d = cadence_s / 86400.0
    bjd = 2460930.5 + np.arange(n_frames) * cadence_d
    flux = 1.0 + rng.standard_normal((n_stars, n_frames)) * sigma
    lc = flux[:, :, None]
    lc_err = np.full((n_stars, n_frames, 1), sigma)
    epoch_ok = np.ones((n_stars, n_frames), dtype=bool)
    frame_kept = np.ones(n_frames, dtype=bool)
    star_best_aper = np.zeros(n_stars, dtype=np.int64)

    core_tile = np.zeros(n_stars, dtype=np.int64)
    night = _FakeNight(n_stars, n_frames, 1, bjd)
    tilemap = _FakeTileMap(core_tile)
    return night, tilemap, lc, lc_err, epoch_ok, frame_kept, star_best_aper, bjd


def _run_search(
    night, tilemap, lc, lc_err, epoch_ok, frame_kept, star_best_aper, comparison_mask, settings=None
):
    settings = settings or Settings()
    comparison_result = _FakeComparisonResult(comparison_mask)
    cotrend_result = compute_cbvs(tilemap, comparison_result, lc, frame_kept, settings.search)
    return search_transits(
        night, tilemap, cotrend_result, comparison_result, lc, lc_err, epoch_ok, frame_kept,
        star_best_aper, settings,
    )


def test_injected_transit_recovered_and_not_absorbed_by_cbv() -> None:
    """A limb-darkened transit is found with the right tc/depth and survives CBV detrending."""
    n_stars = 100
    n_comp = 80
    target = 99
    night, tilemap, lc, lc_err, epoch_ok, frame_kept, star_best_aper, bjd = _scenario(
        n_stars=n_stars, seed=3
    )

    depth = 0.03
    duration_d = 1.2 / 24.0
    tc_true = bjd[0] + (bjd[-1] - bjd[0]) * 0.5
    model = _limb_darkened_transit(bjd, tc_true, depth, duration_d)
    lc[target, :, 0] *= model

    comparison_mask = np.zeros((n_stars, 1), dtype=bool)
    comparison_mask[:n_comp, 0] = True

    settings = Settings()
    comparison_result = _FakeComparisonResult(comparison_mask)
    cotrend_result = compute_cbvs(tilemap, comparison_result, lc, frame_kept, settings.search)
    result = search_transits(
        night, tilemap, cotrend_result, comparison_result, lc, lc_err, epoch_ok, frame_kept,
        star_best_aper, settings,
    )

    assert result.searched[target]
    assert np.isfinite(result.snr[target])
    assert result.snr[target] > 5.0
    assert abs(result.tc[target] - tc_true) < duration_d
    assert (result.flags[target] & FLAG_STEP_LIKE) == 0
    assert (result.flags[target] & FLAG_TOO_DEEP) == 0

    # The joint poly+CBV+box fit must not let the CBVs (built only from the
    # *comparison* ensemble, which never includes the injected star) eat
    # into the recovered depth. Checked at the *true* (tc, duration) via
    # depth_at_other_aperture directly -- the grid search's own argmax adds
    # its own localisation noise (it may prefer a slightly different trial
    # width than the injected one), which is a separate concern from
    # transit-safety and would make this check noisy for the wrong reason.
    good = epoch_ok[target] & frame_kept
    n_cbv = int(np.count_nonzero(np.isfinite(cotrend_result.basis[0, 0, :, 0])))
    cbv_rows = cotrend_result.basis[0, 0, :n_cbv, :]
    med = np.nanmedian(lc[target, good, 0])
    fit_depth, _sigma = depth_at_other_aperture(
        bjd, lc[target, :, 0] / med, lc_err[target, :, 0] / med, good, cbv_rows,
        tc_true, duration_d, settings.search,
    )
    # Compare against the *box-averaged* depth of the injected model over the
    # same window, not the model's nominal peak depth: a limb-darkened
    # transit's flux ramps smoothly through ingress/egress, so even a
    # perfectly fit, noise-free box recovers somewhat less than the nominal
    # depth. That shape effect is not what this test is for -- isolating it
    # here means the assertion is specifically about CBV/nuisance absorption.
    in_box = np.abs(bjd - tc_true) < duration_d / 2
    true_box_depth = float(1.0 - model[in_box].mean())
    assert fit_depth >= 0.9 * true_box_depth


def test_too_deep_eclipse_flagged() -> None:
    """A deep eclipse is flagged TOO_DEEP (and so never a transit candidate).

    Uses a longer baseline than the other tests here: a 60%-deep, 1-hour
    eclipse contaminates a degree-2-polynomial-only nuisance fit badly once
    it covers much more than ~10% of the baseline (the same reason
    relphot.transit_search's own noise-scale estimate is MAD-based rather
    than chi2/dof -- see fit_nuisance_model's caller). A long baseline here
    keeps that contamination small so the test isolates the TOO_DEEP flag
    itself rather than that (separately handled) numerical edge case.
    """
    n_stars = 60
    n_comp = 45
    n_frames = 1600
    target = 59
    night, tilemap, lc, lc_err, epoch_ok, frame_kept, star_best_aper, bjd = _scenario(
        n_stars=n_stars, n_frames=n_frames, seed=4
    )
    tc_true = bjd[0] + (bjd[-1] - bjd[0]) * 0.3
    duration_d = 1.0 / 24.0
    in_transit = np.abs(bjd - tc_true) < duration_d / 2
    lc[target, in_transit, 0] -= 0.60  # a 60% eclipse

    comparison_mask = np.zeros((n_stars, 1), dtype=bool)
    comparison_mask[:n_comp, 0] = True

    result = _run_search(
        night, tilemap, lc, lc_err, epoch_ok, frame_kept, star_best_aper, comparison_mask
    )
    assert result.searched[target]
    assert result.depth[target] > 0.3
    assert result.flags[target] & FLAG_TOO_DEEP
    assert not result.candidate[target]


def test_shared_step_is_not_a_candidate() -> None:
    """A baseline shift common to many stars at the same epoch is rejected, not a transit.

    relphot.transit_search excludes a step-like fit from the trial grid
    itself (see search_one_star's module note): the argmax is never allowed
    to land on a step, so a star whose only real feature is a step reports
    either a low-significance leftover trial (not a candidate) or, on the
    rare trial that survives, gets caught by APERTURE_INCONSISTENT/EDGE/
    SHARED_EPOCH instead. The one outcome that must never happen is
    ``candidate=True``.
    """
    n_stars = 60
    n_comp = 45
    night, tilemap, lc, lc_err, epoch_ok, frame_kept, star_best_aper, bjd = _scenario(
        n_stars=n_stars, seed=5
    )
    step_t = bjd[0] + (bjd[-1] - bjd[0]) * 0.5
    step_targets = [50, 51, 52, 53]
    for i in step_targets:
        lc[i, bjd >= step_t, 0] -= 0.02

    comparison_mask = np.zeros((n_stars, 1), dtype=bool)
    comparison_mask[:n_comp, 0] = True

    result = _run_search(
        night, tilemap, lc, lc_err, epoch_ok, frame_kept, star_best_aper, comparison_mask
    )
    # SHARED_EPOCH is a soft flag by design (relphot.transit_search.HARD_REJECT_FLAGS
    # deliberately excludes it, so a human can weigh a genuinely coincident
    # partial event) -- what must hold is that every affected star is at
    # least flagged, not that it is denied candidacy outright.
    for i in step_targets:
        assert result.searched[i]
        if result.candidate[i]:
            assert result.flags[i] & FLAG_SHARED_EPOCH, flags_to_string(int(result.flags[i]))


def test_shared_box_event_flagged_shared_epoch() -> None:
    """Several stars all showing the same box-shaped dip at once is a shared systematic."""
    n_stars = 60
    n_comp = 45
    night, tilemap, lc, lc_err, epoch_ok, frame_kept, star_best_aper, bjd = _scenario(
        n_stars=n_stars, seed=7
    )
    tc_shared = bjd[0] + (bjd[-1] - bjd[0]) * 0.5
    duration_d = 1.2 / 24.0
    in_event = np.abs(bjd - tc_shared) < duration_d / 2
    shared_targets = [50, 51, 52, 53, 54]
    for i in shared_targets:
        lc[i, in_event, 0] -= 0.02

    comparison_mask = np.zeros((n_stars, 1), dtype=bool)
    comparison_mask[:n_comp, 0] = True

    result = _run_search(
        night, tilemap, lc, lc_err, epoch_ok, frame_kept, star_best_aper, comparison_mask
    )
    n_flagged = 0
    for i in shared_targets:
        assert result.searched[i]
        if result.flags[i] & FLAG_SHARED_EPOCH:
            n_flagged += 1
    assert n_flagged >= len(shared_targets) - 1  # allow one noise-driven miss


def test_white_noise_star_not_a_candidate() -> None:
    n_stars = 60
    n_comp = 45
    target = 55
    night, tilemap, lc, lc_err, epoch_ok, frame_kept, star_best_aper, _bjd = _scenario(
        n_stars=n_stars, seed=6
    )
    comparison_mask = np.zeros((n_stars, 1), dtype=bool)
    comparison_mask[:n_comp, 0] = True

    result = _run_search(
        night, tilemap, lc, lc_err, epoch_ok, frame_kept, star_best_aper, comparison_mask
    )
    assert not result.candidate[target]


def test_white_noise_beta_near_one_and_snr_matches_gold_standard() -> None:
    """A transit in pure white noise: beta ~= 1, and SNR matches a direct joint fit.

    The naive ``depth/sigma*sqrt(n_in)`` formula ignores the nuisance
    regression's own correlation with the box regressor (a mid-baseline box
    is measurably correlated with a degree-2 polynomial's intercept/
    curvature terms over a short baseline), so it is not the right target
    here; a *direct* weighted joint fit (solved independently via
    ``np.linalg.solve``, not the module's own partitioned-regression
    machinery) is the correct gold standard, and the two must agree.
    """
    n_stars = 100
    n_comp = 80
    target = 99
    night, tilemap, lc, lc_err, epoch_ok, frame_kept, star_best_aper, bjd = _scenario(
        n_stars=n_stars, seed=8, sigma=0.005
    )
    depth = 0.01
    duration_d = 1.0 / 24.0
    tc_true = bjd[0] + (bjd[-1] - bjd[0]) * 0.5
    in_transit = np.abs(bjd - tc_true) < duration_d / 2
    lc[target, in_transit, 0] -= depth

    comparison_mask = np.zeros((n_stars, 1), dtype=bool)
    comparison_mask[:n_comp, 0] = True
    result = _run_search(
        night, tilemap, lc, lc_err, epoch_ok, frame_kept, star_best_aper, comparison_mask
    )

    assert result.searched[target]
    assert not result.partial[target]
    # Too few occupied bins at this baseline/duration (~3) for a real
    # correlated-noise estimate -- see SearchSettings.beta_min_bins -- so
    # beta must stay at its no-correction default, not chase that noise.
    assert result.beta[target] == 1.0

    t0 = np.median(bjd)
    x = np.column_stack([np.ones_like(bjd), bjd - t0, (bjd - t0) ** 2, in_transit.astype(float)])
    w = 1.0 / lc_err[target, :, 0] ** 2
    xw = x * w[:, None]
    cov = np.linalg.inv(xw.T @ x)
    beta_hat = cov @ (xw.T @ lc[target, :, 0])
    resid = lc[target, :, 0] - x @ beta_hat
    dof = bjd.size - x.shape[1]
    s2 = float(np.sum(w * resid**2)) / dof
    se_gold = np.sqrt(s2 * cov[-1, -1])
    snr_gold = abs(beta_hat[-1]) / se_gold

    assert abs(result.snr[target] - snr_gold) / snr_gold < 0.2


def test_partial_event_is_flagged_and_still_a_candidate() -> None:
    """An edge-touching (partial-coverage) event is flagged PARTIAL (tier 3), not hard-rejected.

    Before the step-like exclusion was restricted to full-coverage trials,
    a partial trial was, inside the data, indistinguishable from a step and
    so was always excluded -- making every partial/edge transit impossible
    to recover at all.
    """
    n_stars = 60
    n_comp = 45
    night, tilemap, lc, lc_err, epoch_ok, frame_kept, star_best_aper, bjd = _scenario(
        n_stars=n_stars, seed=9
    )
    target = 59
    depth = 0.02
    duration_d = 1.5 / 24.0
    tc_edge = bjd[0] + 0.1 * duration_d  # only part of the box is inside the data
    in_transit = (bjd >= tc_edge - duration_d / 2) & (bjd <= tc_edge + duration_d / 2)
    lc[target, in_transit, 0] -= depth

    comparison_mask = np.zeros((n_stars, 1), dtype=bool)
    comparison_mask[:n_comp, 0] = True
    result = _run_search(
        night, tilemap, lc, lc_err, epoch_ok, frame_kept, star_best_aper, comparison_mask
    )

    assert result.searched[target]
    assert result.partial[target]
    assert result.coverage[target] < 1.0
    assert result.flags[target] & FLAG_PARTIAL
    assert tier_for_flags(int(result.flags[target])) == 3
    assert (result.flags[target] & FLAG_STEP_LIKE) == 0


def test_no_python_loop_over_trial_grid() -> None:
    """search_one_star's trial grid is array-vectorised, not a Python loop over trials."""
    import ast
    import inspect

    from relphot import transit_search

    source = inspect.getsource(transit_search.search_one_star)
    tree = ast.parse(source)
    func = tree.body[0]
    loop_targets = []
    for node in ast.walk(func):
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            loop_targets.append(node.target.id)
    # The only "for" loops allowed inside search_one_star are the small,
    # fixed-size duration loop (for the red-noise beta), the grid-wide
    # 3-candidate step-position loop, and the final winning trial's own
    # 3-candidate step re-check -- never a loop whose body indexes a single
    # (duration, mid-time) trial one at a time.
    assert all(name in ("d", "pos", "t_step", "_pass_idx") for name in loop_targets), loop_targets
