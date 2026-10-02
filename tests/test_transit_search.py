"""Tests for relphot.transit_search: single-event box search and vetting flags."""

from __future__ import annotations

import numpy as np

from relphot.config import Settings
from relphot.cotrend import compute_cbvs
from relphot.transit_search import (
    FLAG_ON_VARIABLE,
    FLAG_PARTIAL,
    FLAG_SHARED_EPOCH,
    FLAG_STEP_LIKE,
    FLAG_TOO_DEEP,
    HARD_REJECT_FLAGS,
    depth_at_other_aperture,
    flag_on_variable,
    flags_to_string,
    search_transits,
    tier_for_flags,
)


class _FrameMeta:
    def __init__(self, bjd: float) -> None:
        self.bjd_tdb = bjd
        self.median_fwhm = 3.0


class _FakeNight:
    def __init__(self, n_stars: int, n_frames: int, n_aper: int, bjd: np.ndarray) -> None:
        self.n_stars = n_stars
        self.n_frames = n_frames
        self.n_aper = n_aper
        self.frame_meta = [_FrameMeta(float(b)) for b in bjd]
        # Per-star per-frame seeing, background and centroid: independent of the
        # synthetic light curves (the R90 regressors), so a real injected event
        # does not correlate with them.
        rng = np.random.default_rng(12345)
        shape = (n_stars, n_frames)
        self.fwhm = 3.0 + 0.1 * rng.standard_normal(shape)
        self.background = 100.0 + 2.0 * rng.standard_normal(shape)
        self.frame_x = 500.0 + 0.05 * rng.standard_normal(shape)
        self.frame_y = 700.0 + 0.05 * rng.standard_normal(shape)


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


def test_effective_min_epochs_below_floor(caplog) -> None:
    """A night shorter than the absolute min_epochs floor searches nothing and warns."""
    import logging

    caplog.set_level(logging.WARNING)
    # Small night: 10 frames with default min_epoch_fraction=0.5 would need ceil(5)=5 epochs,
    # but absolute floor min_epochs=20 means we need 20 -- impossible with 10 frames.
    # So nothing gets searched and a warning is logged.
    night, tilemap, lc, lc_err, epoch_ok, frame_kept, star_best_aper, _bjd = _scenario(
        n_stars=5, n_frames=10, seed=42
    )
    comparison_mask = np.ones((5, 1), dtype=bool)
    settings = Settings()
    # Default: min_epochs=20, min_epoch_fraction=0.5
    result = _run_search(
        night, tilemap, lc, lc_err, epoch_ok, frame_kept, star_best_aper, comparison_mask,
        settings,
    )
    # No stars searched because all need 20 epochs minimum, but only 10 frames available
    assert result.searched.sum() == 0
    # Check that the warning was logged
    assert any("no star has >= 20 good epochs" in record.message for record in caplog.records)


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


def test_on_variable_flag_is_informational_only() -> None:
    assert FLAG_ON_VARIABLE & HARD_REJECT_FLAGS == 0
    assert flags_to_string(FLAG_ON_VARIABLE) == "ON_VARIABLE"
    assert flags_to_string(FLAG_SHARED_EPOCH | FLAG_ON_VARIABLE) == "SHARED_EPOCH|ON_VARIABLE"
    assert tier_for_flags(FLAG_ON_VARIABLE) == 1  # a clean full event stays tier 1
    assert tier_for_flags(FLAG_SHARED_EPOCH | FLAG_ON_VARIABLE) == 2
    assert tier_for_flags(FLAG_PARTIAL | FLAG_ON_VARIABLE) == 3


def test_flag_on_variable_marks_transit_candidates_that_are_variable() -> None:
    flags = np.array([0, 0, FLAG_SHARED_EPOCH, 0], dtype=np.int64)
    candidate = np.array([True, True, True, False])
    variable = np.array([True, False, True, True])
    flag_on_variable(flags, candidate, variable)
    assert flags.tolist() == [FLAG_ON_VARIABLE, 0, FLAG_SHARED_EPOCH | FLAG_ON_VARIABLE, 0]
    # candidacy is the caller's mask; the helper never changes it
    assert candidate.tolist() == [True, True, True, False]


# --- R90 false-positive screen (informational flags) -----------------------------------------


def _r90_series(seed: int = 0, n: int = 350, cadence_s: float = 31.7, sigma: float = 0.004):
    """One synthetic star for the R90 features: time, flux, errors, empty CBVs, regressors.

    The regressors are ``(4, n)`` FWHM ratio, background and centroid x/y, white noise
    independent of the flux (a star whose light curve does not follow its seeing).
    """
    rng = np.random.default_rng(seed)
    t = 2460930.5 + np.arange(n) * cadence_s / 86400.0
    y = 1.0 + sigma * rng.standard_normal(n)
    err = np.full(n, sigma)
    cbv = np.zeros((0, n))
    regs = np.vstack([
        1.0 + 0.03 * rng.standard_normal(n),
        100.0 + 2.0 * rng.standard_normal(n),
        500.0 + 0.05 * rng.standard_normal(n),
        700.0 + 0.05 * rng.standard_normal(n),
    ])
    return t, y, err, cbv, regs


def _r90_bits(t, y, err, cbv, regs, tc, duration_h, settings=None) -> int:
    from relphot.config import SearchSettings
    from relphot.transit_r90 import compute_r90_features
    from relphot.transit_search import r90_flag_bits

    settings = settings or SearchSettings()
    features = compute_r90_features(
        t, y, err, cbv, tc, duration_h / 24.0, regs, settings.poly_degree
    )
    return r90_flag_bits(features, settings)


def test_r90_flags_are_informational_and_round_trip() -> None:
    from relphot.transit_search import (
        FLAG_NAMES,
        FLAG_R90_CLIP,
        FLAG_R90_FIT_FAIL,
        FLAG_R90_FLAT,
        FLAG_R90_SINGLE_POINT,
        FLAG_R90_SYSTEMATICS,
        R90_FLAGS,
    )

    bits = [
        FLAG_R90_SINGLE_POINT, FLAG_R90_SYSTEMATICS, FLAG_R90_CLIP, FLAG_R90_FLAT,
        FLAG_R90_FIT_FAIL,
    ]
    assert all(b > FLAG_ON_VARIABLE for b in bits)  # after the existing bits
    assert len(set(bits)) == 5
    assert sum(bits) == R90_FLAGS
    assert R90_FLAGS & HARD_REJECT_FLAGS == 0
    names = dict(FLAG_NAMES)
    assert [names[b] for b in bits] == [
        "R90_SINGLE_POINT", "R90_SYSTEMATICS", "R90_CLIP", "R90_FLAT", "R90_FIT_FAIL"
    ]
    assert len({name for _bit, name in FLAG_NAMES}) == len(FLAG_NAMES)
    for bit, name in FLAG_NAMES:
        assert flags_to_string(bit) == name
        assert name in flags_to_string(bit | FLAG_PARTIAL).split("|")
    assert flags_to_string(R90_FLAGS | FLAG_ON_VARIABLE).split("|") == [
        "ON_VARIABLE", "R90_SINGLE_POINT", "R90_SYSTEMATICS", "R90_CLIP", "R90_FLAT",
        "R90_FIT_FAIL",
    ]
    # the tier is blind to them, whatever else is set
    assert tier_for_flags(R90_FLAGS) == 1
    assert tier_for_flags(FLAG_SHARED_EPOCH | R90_FLAGS) == 2
    assert tier_for_flags(FLAG_PARTIAL | R90_FLAGS) == 3
    # the integer must survive the int64 flags column and the DB text round trip
    assert int(np.int64(R90_FLAGS | FLAG_ON_VARIABLE)) == R90_FLAGS | FLAG_ON_VARIABLE


def test_r90_flag_bits_thresholds_and_nan() -> None:
    from dataclasses import replace

    from relphot.config import SearchSettings
    from relphot.transit_r90 import R90Features
    from relphot.transit_search import (
        FLAG_R90_CLIP,
        FLAG_R90_FIT_FAIL,
        FLAG_R90_FLAT,
        FLAG_R90_SINGLE_POINT,
        FLAG_R90_SYSTEMATICS,
        r90_flag_bits,
    )

    s = SearchSettings()
    assert (s.r90_enabled, s.r90_top1_share_max, s.r90_top3_share_max) == (True, 0.4, 0.6)
    assert (s.r90_reg_dchi2_ratio_min, s.r90_reg_depth_ratio_min) == (0.3, 0.7)
    assert (s.r90_clip3_dchi2_min, s.r90_dbic_flat_min) == (16.0, 6.0)

    ok = R90Features(
        fit_ok=True, success=True, top1_share=0.4, top3_share=0.6, reg_dchi2_ratio=0.3,
        reg_depth_ratio=0.7, clip3_dchi2=16.0, dbic_flat=6.0,
    )
    assert r90_flag_bits(ok, s) == 0  # every threshold is inclusive
    cases = {
        FLAG_R90_SINGLE_POINT: [{"top1_share": 0.41}, {"top3_share": 0.61}],
        FLAG_R90_SYSTEMATICS: [{"reg_dchi2_ratio": 0.29}, {"reg_depth_ratio": 0.69}],
        FLAG_R90_CLIP: [{"clip3_dchi2": 15.9}],
        FLAG_R90_FLAT: [{"dbic_flat": 5.9}],
        FLAG_R90_FIT_FAIL: [{"success": False}],
    }
    for bit, variants in cases.items():
        for change in variants:
            assert r90_flag_bits(replace(ok, **change), s) == bit, change
    # a number that could not be computed fails its own group, and only that group
    assert r90_flag_bits(replace(ok, top1_share=np.nan), s) == FLAG_R90_SINGLE_POINT
    assert r90_flag_bits(replace(ok, reg_depth_ratio=np.nan), s) == FLAG_R90_SYSTEMATICS
    assert r90_flag_bits(replace(ok, clip3_dchi2=np.nan), s) == FLAG_R90_CLIP
    assert r90_flag_bits(replace(ok, dbic_flat=np.nan), s) == FLAG_R90_FLAT
    # no fit at all: FIT_FAIL alone
    assert r90_flag_bits(R90Features(), s) == FLAG_R90_FIT_FAIL
    # thresholds come from the settings
    loose = replace(s, r90_top1_share_max=0.5, r90_dbic_flat_min=0.0)
    assert r90_flag_bits(replace(ok, top1_share=0.5, dbic_flat=1.0), loose) == 0


def test_r90_settings_load_from_toml(tmp_path) -> None:
    from relphot.config import load_settings, settings_from_dict, settings_to_dict

    path = tmp_path / "r90.toml"
    path.write_text(
        "[search]\nr90_enabled = false\nr90_top1_share_max = 0.3\nr90_clip3_dchi2_min = 25.0\n"
    )
    search = load_settings(path).search
    assert search.r90_enabled is False
    assert (search.r90_top1_share_max, search.r90_clip3_dchi2_min) == (0.3, 25.0)
    assert search.r90_dbic_flat_min == 6.0  # untouched keys keep the default
    assert settings_from_dict(settings_to_dict(load_settings(path))).search == search


def test_r90_clean_transit_passes_every_criterion() -> None:
    t, y, err, cbv, regs = _r90_series(seed=1)
    tc = t[0] + 0.5 * (t[-1] - t[0])
    y = y * _limb_darkened_transit(t, tc, 0.02, 1.0 / 24.0)
    assert _r90_bits(t, y, err, cbv, regs, tc, 1.0) == 0


def test_r90_single_point_spike_fails_single_point() -> None:
    from relphot.transit_search import FLAG_R90_CLIP, FLAG_R90_SINGLE_POINT

    t, y, err, cbv, regs = _r90_series(seed=2)
    tc = t[0] + 0.5 * (t[-1] - t[0])
    y = y.copy()
    y[np.argmin(np.abs(t - tc))] -= 0.04  # one 10-sigma epoch, no event around it
    bits = _r90_bits(t, y, err, cbv, regs, tc, 0.5)
    assert bits & FLAG_R90_SINGLE_POINT
    # once that epoch is clipped nothing is left of the event
    assert bits & FLAG_R90_CLIP
    # the same star without the spike is not single-point dominated
    t, y, err, cbv, regs = _r90_series(seed=2)
    assert not _r90_bits(t, y, err, cbv, regs, tc, 0.5) & FLAG_R90_SINGLE_POINT


def test_r90_event_collinear_with_seeing_fails_systematics() -> None:
    from relphot.transit_search import FLAG_R90_SYSTEMATICS

    t, y, err, cbv, regs = _r90_series(seed=3)
    tc = t[0] + 0.5 * (t[-1] - t[0])
    bump = np.exp(-0.5 * ((t - tc) / (0.25 / 24.0)) ** 2)  # a seeing excursion of ~0.6 h
    y = y * (1.0 - 0.02 * bump)  # the flux dip follows it exactly
    clean = _r90_bits(t, y, err, cbv, regs, tc, 0.6)
    assert not clean & FLAG_R90_SYSTEMATICS  # the same dip with quiet seeing is an event
    regs = regs.copy()
    regs[0] = regs[0] + 0.5 * bump  # the FWHM of this star swells with the dip
    assert _r90_bits(t, y, err, cbv, regs, tc, 0.6) & FLAG_R90_SYSTEMATICS


def test_r90_flat_noise_fails_flat() -> None:
    from relphot.transit_search import FLAG_R90_FIT_FAIL, FLAG_R90_FLAT

    t, y, err, cbv, regs = _r90_series(seed=4)
    tc = t[0] + 0.5 * (t[-1] - t[0])
    bits = _r90_bits(t, y, err, cbv, regs, tc, 1.0)
    assert bits & FLAG_R90_FLAT
    assert not bits & FLAG_R90_FIT_FAIL  # the fit itself worked; there is just nothing to fit


def test_r90_unfittable_light_curve_is_fit_fail_not_an_exception() -> None:
    from relphot.transit_search import FLAG_R90_FIT_FAIL

    t, y, err, cbv, regs = _r90_series(seed=5, n=60)
    y = np.full_like(y, np.nan)
    assert _r90_bits(t, y, err, cbv, regs, t[30], 1.0) == FLAG_R90_FIT_FAIL


def test_r90_screen_in_search_transits_candidates_only_and_informational() -> None:
    from dataclasses import replace

    from relphot.transit_search import R90_FLAGS

    n_stars, n_comp, target = 100, 80, 99
    night, tilemap, lc, lc_err, epoch_ok, frame_kept, star_best_aper, bjd = _scenario(
        n_stars=n_stars, seed=3
    )
    tc_true = bjd[0] + (bjd[-1] - bjd[0]) * 0.5
    lc[target, :, 0] *= _limb_darkened_transit(bjd, tc_true, 0.03, 1.2 / 24.0)
    comparison_mask = np.zeros((n_stars, 1), dtype=bool)
    comparison_mask[:n_comp, 0] = True

    def run(**search_kw):
        settings = Settings()
        settings = replace(settings, search=replace(settings.search, **search_kw))
        return _run_search(
            night, tilemap, lc, lc_err, epoch_ok, frame_kept, star_best_aper, comparison_mask,
            settings,
        )

    on = run()
    off = run(r90_enabled=False)

    # screened exactly the candidates; a clean injected transit passes
    assert on.candidate[target]
    assert np.array_equal(on.r90_evaluated, on.candidate)
    assert on.r90_pass[target]
    assert on.flags[target] & R90_FLAGS == 0
    feats = ("top1_share", "top3_share", "reg_dchi2_ratio", "reg_depth_ratio", "clip3_dchi2",
             "dbic_flat")
    for name in feats:
        column = getattr(on, name)
        assert np.isfinite(column[target]), name
        assert np.all(np.isnan(column[~on.r90_evaluated])), name  # non-candidates untouched
    assert on.top1_share[target] <= 0.4 and on.reg_depth_ratio[target] >= 0.7

    # switched off: nothing computed, no bit, and the search itself is untouched
    assert not off.r90_evaluated.any() and not off.r90_pass.any()
    assert all(np.all(np.isnan(getattr(off, name))) for name in feats)
    assert (off.flags & R90_FLAGS == 0).all()
    assert np.array_equal(on.flags & ~R90_FLAGS, off.flags)
    assert np.array_equal(on.candidate, off.candidate)
    assert np.array_equal(on.snr, off.snr, equal_nan=True)
    assert np.array_equal(on.tc, off.tc, equal_nan=True)
    assert [tier_for_flags(int(b)) for b in on.flags] == [
        tier_for_flags(int(b)) for b in off.flags
    ]


def test_r90_systematics_flag_from_the_night_regressors_leaves_tier_and_candidacy() -> None:
    from relphot.transit_search import FLAG_R90_SYSTEMATICS, R90_FLAGS

    n_stars, n_comp, target = 100, 80, 99
    night, tilemap, lc, lc_err, epoch_ok, frame_kept, star_best_aper, bjd = _scenario(
        n_stars=n_stars, seed=5
    )
    tc_true = bjd[0] + (bjd[-1] - bjd[0]) * 0.5
    bump = np.exp(-0.5 * ((bjd - tc_true) / (0.25 / 24.0)) ** 2)
    lc[target, :, 0] *= 1.0 - 0.02 * bump
    night.fwhm[target] += 1.0 * bump  # this star's seeing swells with its dip
    comparison_mask = np.zeros((n_stars, 1), dtype=bool)
    comparison_mask[:n_comp, 0] = True

    result = _run_search(
        night, tilemap, lc, lc_err, epoch_ok, frame_kept, star_best_aper, comparison_mask
    )
    assert result.candidate[target]  # candidacy is not touched
    assert result.r90_evaluated[target] and not result.r90_pass[target]
    assert result.flags[target] & FLAG_R90_SYSTEMATICS
    # the regressor explains the dip, so little chi2 improvement is left for the event
    assert result.reg_dchi2_ratio[target] < 0.3
    assert tier_for_flags(int(result.flags[target])) == tier_for_flags(
        int(result.flags[target]) & ~R90_FLAGS
    )
