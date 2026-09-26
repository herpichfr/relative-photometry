"""Stage 6b: single-event transit search.

The night's baseline is far shorter than any realistic orbital period, so
this searches for a single box-shaped dimming event per star rather than
running a periodic box-least-squares. For every star with enough good
epochs at its own best aperture, a grid of trial durations and mid-times is
evaluated by a joint weighted linear regression

    flux = poly(t, deg) + sum_k a_k * CBV_k(t) + depth * box(t; tc, dur)

solved simultaneously (the nuisance terms and the transit depth come out of
one regression, never a nuisance fit to the star's own light curve followed
by a search on its residuals) via the partitioned-regression (Frisch-Waugh-
Lovell) identity, using cumulative sums over the frame axis so the entire
(duration, mid-time) trial grid is evaluated by array indexing -- no Python
loop ever iterates over a trial. Only the *outer* loop over stars is a
Python loop, as the module docstring's design note allows.

This is the module that keeps the pipeline's transit-safety guarantee (see
:mod:`relphot.cotrend`): the CBVs and the box column are fit jointly, so the
depth estimate is never derived from residuals of a systematics model that
had a chance to absorb part of a real transit.

Noise model, in two independent pieces:

* **Per-trial white variance** comes from the *joint* (nuisance + box) fit's
  own chi2 (``chi2_box_grid = chi2_base - depth**2 * denom``), not from the
  nuisance-only residuals -- a nuisance-only ``resid_base`` still contains
  the event itself, and a 1h event in a 3h night contaminates roughly a
  third of the points, inflating any scale derived from it (a mean-square
  badly, a MAD measurably). Dividing the joint fit's own chi2 by its own
  dof gives an unbiased white-noise estimate that already has the event
  explained away.
* **Red-noise inflation (beta)** is a genuine, separate effect (correlated
  scatter beyond what per-point weights capture) and is estimated the same
  way (Pont et al. 2006 time-averaging) but from the *joint* fit's
  residuals at the current best trial, not from the contaminated
  nuisance-only residuals either. Because beta depends on which trial is
  best and the best trial depends on beta, :func:`search_one_star` iterates
  this to a fixed point (bounded at 3 passes: an initial white-SNR pass
  plus up to two beta-corrected re-picks).

A shared instrumental effect (a genuinely noisier stretch near the end of a
run, a shared frame-level glitch) is handled separately again, upstream in
:func:`search_transits`, by rescaling every star's per-frame error using the
comparison ensemble's own per-frame scatter (:func:`relphot.cotrend.compute_frame_error_scale`)
-- never a per-star fit, so it stays transit-safe.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from relphot.cotrend import compute_frame_error_scale
from relphot.numeric import mad_sigma, nanmedian_quiet, robust_clip_series

if TYPE_CHECKING:
    from relphot.comparison import ComparisonResult
    from relphot.config import SearchSettings, Settings
    from relphot.cotrend import CotrendResult
    from relphot.match import MatchedNight
    from relphot.tiles import TileMap

logger = logging.getLogger(__name__)

__all__ = [
    "FLAG_APERTURE_INCONSISTENT",
    "FLAG_EDGE",
    "FLAG_FEW_POINTS",
    "FLAG_HIGH_BETA",
    "FLAG_NAMES",
    "FLAG_NEIGHBOUR_BLEND",
    "FLAG_PARTIAL",
    "FLAG_SHARED_EPOCH",
    "FLAG_STEP_LIKE",
    "FLAG_TOO_DEEP",
    "HARD_REJECT_FLAGS",
    "StarSearchResult",
    "TransitSearchResult",
    "depth_at_other_aperture",
    "fit_nuisance_model",
    "flags_to_string",
    "plot_transit_candidate",
    "search_one_star",
    "search_transits",
    "tier_for_flags",
]

#: Bit flags for a candidate's vetting outcome (see the module and
#: :class:`SearchSettings` docstrings for what sets each one).
FLAG_SHARED_EPOCH = 1 << 0
FLAG_STEP_LIKE = 1 << 1
FLAG_APERTURE_INCONSISTENT = 1 << 2
FLAG_EDGE = 1 << 3
FLAG_TOO_DEEP = 1 << 4
FLAG_FEW_POINTS = 1 << 5
FLAG_HIGH_BETA = 1 << 6
FLAG_NEIGHBOUR_BLEND = 1 << 7
FLAG_PARTIAL = 1 << 8

FLAG_NAMES: tuple[tuple[int, str], ...] = (
    (FLAG_SHARED_EPOCH, "SHARED_EPOCH"),
    (FLAG_STEP_LIKE, "STEP_LIKE"),
    (FLAG_APERTURE_INCONSISTENT, "APERTURE_INCONSISTENT"),
    (FLAG_EDGE, "EDGE"),
    (FLAG_TOO_DEEP, "TOO_DEEP"),
    (FLAG_FEW_POINTS, "FEW_POINTS"),
    (FLAG_HIGH_BETA, "HIGH_BETA"),
    (FLAG_NEIGHBOUR_BLEND, "NEIGHBOUR_BLEND"),
    (FLAG_PARTIAL, "PARTIAL"),
)

#: Flags that disqualify a star from candidate status outright, regardless of
#: SNR (a judgement call: STEP_LIKE and TOO_DEEP mean the event is not a
#: transit at all; FEW_POINTS means too little data to trust the depth).
#: EDGE and PARTIAL are deliberately *soft*, not hard rejects: a night this
#: short is expected to catch only part of many real transits (partial
#: ingress- or egress-only events are explicitly in scope -- see
#: min_box_coverage), so an edge-touching/partial event is exactly the kind
#: of real candidate this search must not throw away. SHARED_EPOCH,
#: APERTURE_INCONSISTENT, HIGH_BETA, and NEIGHBOUR_BLEND are also kept soft,
#: for a human to weigh.
HARD_REJECT_FLAGS = FLAG_STEP_LIKE | FLAG_TOO_DEEP | FLAG_FEW_POINTS

#: Soft flags other than NEIGHBOUR_BLEND -- used by :func:`tier_for_flags` to
#: tell a clean full event (tier 1) from one with something else worth a
#: second look (tier 2). NEIGHBOUR_BLEND alone does not demote a candidate:
#: it is reported so a human can check the dilution math, not because it is
#: evidence against the event itself.
_SOFT_FLAGS_OTHER_THAN_BLEND = (
    FLAG_SHARED_EPOCH | FLAG_APERTURE_INCONSISTENT | FLAG_HIGH_BETA
)


def flags_to_string(bits: int) -> str:
    """A human-readable ``"|"``-joined name list for a vetting bitmask, or ``"OK"``."""
    names = [name for bit, name in FLAG_NAMES if bits & bit]
    return "|".join(names) if names else "OK"


def tier_for_flags(flags: int) -> int:
    """1 (full event, clean), 2 (full event, soft-flagged), or 3 (partial event).

    Computed from vetting flags alone, regardless of candidate status, so it
    can rank every searched star, not only the ones that cleared
    ``snr_threshold``. A hard-rejected star still gets a tier (for the full
    per-star metrics table); it is simply never a candidate at any tier.
    """
    if flags & FLAG_PARTIAL:
        return 3
    if flags & _SOFT_FLAGS_OTHER_THAN_BLEND:
        return 2
    return 1


@dataclass(slots=True)
class StarSearchResult:
    """The best single-event fit for one star, plus the diagnostics the vetting flags need.

    ``depth_per_aper``/``sigma_depth_per_aper`` are ``(n_aper,)``, NaN where
    that aperture could not be fit. Grid arrays (``snr_grid``, ``tc_grid``,
    ``durations``) and the epoch arrays are only populated when
    ``keep_grid=True`` is passed to :func:`search_one_star` -- used for
    candidate plots, not the bulk per-star table. ``coverage`` is the
    winning trial's fraction of its box actually inside the data span (1.0
    for a fully-contained event); ``partial`` is ``coverage < 1``.
    """

    ok: bool
    aper: int = -1
    n_good: int = 0
    snr: float = np.nan
    depth: float = np.nan
    tc: float = np.nan
    duration: float = np.nan
    n_in: int = 0
    beta: float = np.nan
    coverage: float = np.nan
    partial: bool = False
    dchi2_box_vs_flat: float = np.nan
    dchi2_box_vs_step: float = np.nan
    depth_per_aper: np.ndarray = field(default_factory=lambda: np.array([]))
    sigma_depth_per_aper: np.ndarray = field(default_factory=lambda: np.array([]))
    edge: bool = False
    too_deep: bool = False
    step_like: bool = False
    high_beta: bool = False
    aperture_inconsistent: bool = False
    few_points: bool = False
    t_good: np.ndarray | None = None
    y_good: np.ndarray | None = None
    err_good: np.ndarray | None = None
    model_good: np.ndarray | None = None
    snr_grid: np.ndarray | None = None
    tc_grid: np.ndarray | None = None
    durations: np.ndarray | None = None


@dataclass(slots=True)
class TransitSearchResult:
    """Per-star single-event search outcome, ``(n_stars,)`` unless noted.

    ``flags`` is a bitmask (see the ``FLAG_*`` constants); ``candidate`` is
    ``snr >= settings.search.snr_threshold`` and no bit in
    ``HARD_REJECT_FLAGS`` set. ``event_time_hist`` is ``(n_tiles, n_frames)``
    int, one histogram per tile of stars' best event epoch (only stars with
    ``snr >= settings.search.coincidence_snr_threshold`` contribute).
    ``coverage``/``partial`` mirror :class:`StarSearchResult`.
    ``frame_error_scale`` is ``(n_tiles, n_aper, n_frames)``, the per-frame
    error-inflation factor from :func:`relphot.cotrend.compute_frame_error_scale`
    applied to every star's errors before searching.
    """

    searched: np.ndarray
    aper: np.ndarray
    n_good: np.ndarray
    snr: np.ndarray
    depth: np.ndarray
    tc: np.ndarray
    duration: np.ndarray
    n_in: np.ndarray
    beta: np.ndarray
    coverage: np.ndarray
    partial: np.ndarray
    dchi2_box_vs_flat: np.ndarray
    dchi2_box_vs_step: np.ndarray
    depth_per_aper: np.ndarray
    sigma_depth_per_aper: np.ndarray
    coincidence_count: np.ndarray
    flags: np.ndarray
    candidate: np.ndarray
    event_time_hist: np.ndarray
    frame_error_scale: np.ndarray


def fit_nuisance_model(
    t: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    cbv_rows: np.ndarray,
    poly_degree: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int] | None:
    """Weighted fit of ``y = poly(t - median(t), poly_degree) + CBVs @ a``.

    ``cbv_rows`` is ``(n_cbv, n_points)`` (possibly ``n_cbv == 0``). Returns
    ``(X, beta, resid, chi2, dof)`` or ``None`` if the system is singular or
    underdetermined. ``X`` is the design matrix actually used, so a caller
    can extend it with further columns (e.g. a box regressor) consistently.
    """
    n = t.shape[0]
    t0 = float(np.median(t))
    cols = [np.ones(n)]
    for deg in range(1, max(int(poly_degree), 0) + 1):
        cols.append((t - t0) ** deg)
    for row in cbv_rows:
        cols.append(row)
    x = np.column_stack(cols)
    n_params = x.shape[1]
    if n - n_params < 3:
        return None

    xw = x * w[:, None]
    xtwx = xw.T @ x
    xtwy = xw.T @ y
    try:
        cov = np.linalg.inv(xtwx)
    except np.linalg.LinAlgError:
        return None
    beta = cov @ xtwy
    resid = y - x @ beta
    chi2 = float(np.sum(w * resid**2))
    dof = n - n_params
    return x, beta, resid, chi2, dof


def _red_noise_beta(resid: np.ndarray, t: np.ndarray, bin_days: float, min_bins: int) -> float:
    """Pont et al. (2006) time-averaging red-noise inflation factor at ``bin_days``.

    Bins ``resid`` into non-overlapping windows of width ``bin_days`` over
    ``t`` and compares the scatter of bin means to the white-noise
    expectation ``sigma1 / sqrt(mean bin occupancy)``. Floored at 1.0 (white
    noise cannot make the binned scatter smaller than expected). ``resid``
    should be the *joint* (nuisance + box) fit's residual at the trial
    duration being evaluated, not a nuisance-only residual that still
    contains the event.

    Requires at least ``min_bins`` occupied bins, else returns 1.0 (no
    claim of red-noise inflation): with only a handful of bins -- a night
    only ~3x the trial duration long yields 2-3 -- the MAD of the bin means
    is itself a wildly noisy estimator (a handful of white-noise numbers can
    easily look several times over- or under-dispersed by chance), and
    "correcting" for that noise would chase the estimator's own sampling
    error rather than any real correlated-noise property of the data.
    """
    if bin_days <= 0 or t.size < 4:
        return 1.0
    sigma1 = float(mad_sigma(resid))
    if not np.isfinite(sigma1) or sigma1 <= 0:
        return 1.0

    t0 = t[0]
    bin_idx = np.floor((t - t0) / bin_days).astype(np.int64)
    n_bins = int(bin_idx.max()) + 1
    counts = np.bincount(bin_idx, minlength=n_bins)
    sums = np.bincount(bin_idx, weights=resid, minlength=n_bins)
    occupied = counts > 0
    if np.count_nonzero(occupied) < max(min_bins, 2):
        return 1.0
    means = sums[occupied] / counts[occupied]
    mean_n = float(counts[occupied].mean())
    sigma_n_actual = float(mad_sigma(means))
    if not np.isfinite(sigma_n_actual):
        return 1.0
    expected = sigma1 / np.sqrt(max(mean_n, 1.0))
    if expected <= 0:
        return 1.0
    return max(sigma_n_actual / expected, 1.0)


def search_one_star(
    t: np.ndarray,
    y: np.ndarray,
    err: np.ndarray,
    good: np.ndarray,
    cbv_rows: np.ndarray,
    settings: SearchSettings,
    *,
    aper: int = -1,
    n_aper: int = 1,
    keep_grid: bool = False,
) -> StarSearchResult:
    """Search one star's light curve (one aperture) for the best single dimming event.

    ``t``/``y``/``err`` are ``(n_frames,)`` for the star's chosen aperture
    (``err`` should already include any per-frame rescaling from
    :func:`relphot.cotrend.compute_frame_error_scale`); ``good`` is the
    epoch mask (finite, kept, unflagged) already restricted to that
    aperture. ``cbv_rows`` is ``(n_cbv, n_frames)`` for the star's tile and
    aperture (already NaN-free at kept frames, or shape ``(0, n_frames)``
    when no CBVs are available). Returns ``ok=False`` when there are too
    few good epochs to fit even the nuisance model.
    """
    idx_good = np.nonzero(good)[0]
    n_good = idx_good.size
    if n_good < max(settings.min_epochs, 1):
        return StarSearchResult(ok=False, aper=aper, n_good=n_good)

    order = np.argsort(t[idx_good])
    idx_good = idx_good[order]
    t_good = t[idx_good].astype(np.float64)
    y_good = y[idx_good].astype(np.float64)
    err_good = err[idx_good].astype(np.float64)
    cbv_good = cbv_rows[:, idx_good] if cbv_rows.size else cbv_rows.reshape(0, n_good)

    # Remove isolated single-epoch outliers (cosmic rays, dropped frames)
    # before any fit -- see robust_clip_series's docstring for why this is
    # outlier rejection, not a systematics model, and safe for transit-safety.
    keep = robust_clip_series(y_good, settings.lc_clip_sigma, settings.lc_clip_window)
    if not np.all(keep):
        t_good, y_good, err_good = t_good[keep], y_good[keep], err_good[keep]
        cbv_good = cbv_good[:, keep]
        n_good = t_good.shape[0]
        if n_good < max(settings.min_epochs, 1):
            return StarSearchResult(ok=False, aper=aper, n_good=n_good)

    w_good = 1.0 / err_good**2
    base = fit_nuisance_model(t_good, y_good, w_good, cbv_good, settings.poly_degree)
    if base is None:
        return StarSearchResult(ok=False, aper=aper, n_good=n_good)
    x_base, beta_base, resid_base, chi2_base, _dof_base = base
    n_params = x_base.shape[1]
    # White-noise dof for the *joint* (nuisance + box) fit: one more
    # parameter (the box coefficient) than the nuisance-only fit.
    dof_white = n_good - n_params - 1
    if dof_white < 1:
        return StarSearchResult(ok=False, aper=aper, n_good=n_good)

    xw = x_base * w_good[:, None]
    m_inv = np.linalg.inv(xw.T @ x_base)
    xwy = xw.T @ y_good

    # Prefix sums over the time-sorted good epochs; box sums over [lo, hi)
    # for any trial come out as one indexed subtraction, so the whole
    # (duration, mid-time) grid below never runs a Python loop over trials.
    zero_p = np.zeros((1, n_params))
    s_w = np.concatenate([[0.0], np.cumsum(w_good)])
    s_wy = np.concatenate([[0.0], np.cumsum(w_good * y_good)])
    s_wx = np.concatenate([zero_p, np.cumsum(w_good[:, None] * x_base, axis=0)], axis=0)

    durations_h = np.linspace(
        settings.duration_min_hours, settings.duration_max_hours, max(settings.n_durations, 1)
    )
    durations_d = durations_h / 24.0
    dur_min_d = durations_d.min()
    dur_max_d = durations_d.max()
    step = max(dur_min_d / 4.0, 1e-6)
    tc_start = t_good.min() - dur_max_d / 2.0
    tc_end = t_good.max() + dur_max_d / 2.0
    n_tc = max(int(np.ceil((tc_end - tc_start) / step)) + 1, 1)
    tc_grid = tc_start + step * np.arange(n_tc)

    half = durations_d[:, None] / 2.0  # (D, 1)
    lo_t = tc_grid[None, :] - half
    hi_t = tc_grid[None, :] + half
    lo = np.searchsorted(t_good, lo_t.ravel(), side="left").reshape(lo_t.shape)
    hi = np.searchsorted(t_good, hi_t.ravel(), side="right").reshape(hi_t.shape)
    n_in = hi - lo

    data_lo, data_hi = t_good[0], t_good[-1]
    overlap = np.clip(
        np.minimum(hi_t, data_hi) - np.maximum(lo_t, data_lo), 0.0, None
    )
    coverage = overlap / (2.0 * half)

    valid = (
        (n_in >= settings.min_in_transit_points)
        & (coverage >= settings.min_box_coverage)
        & (lo < hi)
    )

    b_w = s_w[hi] - s_w[lo]
    b_wy = s_wy[hi] - s_wy[lo]
    b_wx = s_wx[hi] - s_wx[lo]  # (D, T, P)

    proj = np.einsum("dtp,pq->dtq", b_wx, m_inv)
    corr_y = np.einsum("dtp,p->dt", proj, xwy)
    corr_box = np.einsum("dtp,dtp->dt", proj, b_wx)
    denom = b_w - corr_box

    with np.errstate(invalid="ignore", divide="ignore"):
        c_hat = (b_wy - corr_y) / denom
        depth = -c_hat

    # Per-trial white variance from the *joint* fit's own chi2 (see the
    # module docstring): chi2_box_grid already has the event explained away,
    # so dividing by its own dof is an unbiased white-noise estimate, unlike
    # anything derived from resid_base (nuisance-only, still contains the
    # event -- a 1h event in a 3h night contaminates roughly a third of the
    # points).
    with np.errstate(invalid="ignore"):
        chi2_box_grid = chi2_base - depth**2 * denom
    with np.errstate(invalid="ignore", divide="ignore"):
        s2_grid = chi2_box_grid / dof_white
        sigma_c = np.sqrt(s2_grid / denom)
        white_snr_grid = depth / sigma_c

    # Grid-wide step-like exclusion: a step (single baseline shift) at either
    # edge of a wide trial or at its centre can rival or beat the box fit;
    # if the argmax were left to land there, the star's reported "best
    # event" would be a false one instead of its real, smaller signal
    # elsewhere in the grid. Restricted to trials whose box lies *fully
    # inside the data* (coverage == 1): a partial trial's box is, inside the
    # data, identical to a step by construction (nothing beyond the data
    # edge to show the flux returning to baseline), so this test would
    # otherwise reject every partial/edge event outright -- exactly the
    # partial transits this search must keep (see min_box_coverage).
    # Evaluated for every full-coverage trial with the same cumulative-sum
    # machinery, only looping over the fixed 3 step-position candidates.
    is_full = coverage >= 0.999
    min_side = settings.step_min_side_points
    chi2_step_best_grid = np.full(chi2_box_grid.shape, np.inf)
    for pos in (lo_t, np.broadcast_to(tc_grid[None, :], lo_t.shape), hi_t):
        step_idx = np.searchsorted(
            t_good, np.clip(pos, data_lo, data_hi).ravel(), side="left"
        ).reshape(pos.shape)
        valid_step = (step_idx >= min_side) & (step_idx <= n_good - min_side)
        b_w_step = s_w[n_good] - s_w[step_idx]
        b_wy_step = s_wy[n_good] - s_wy[step_idx]
        b_wx_step = s_wx[n_good] - s_wx[step_idx]  # (D, T, P)
        proj_step = np.einsum("dtp,pq->dtq", b_wx_step, m_inv)
        denom_step = b_w_step - np.einsum("dtp,dtp->dt", proj_step, b_wx_step)
        with np.errstate(invalid="ignore", divide="ignore"):
            c_step = (b_wy_step - np.einsum("dtp,p->dt", proj_step, xwy)) / denom_step
            chi2_step_this = chi2_base - c_step**2 * denom_step
        chi2_step_this = np.where(
            valid_step & np.isfinite(chi2_step_this) & (denom_step > 0), chi2_step_this, np.inf
        )
        chi2_step_best_grid = np.minimum(chi2_step_best_grid, chi2_step_this)
    with np.errstate(invalid="ignore"):
        is_step_like_grid = is_full & np.isfinite(chi2_step_best_grid) & (
            (chi2_step_best_grid - chi2_box_grid) <= settings.step_delta_chi2_threshold
        )

    ok_trial = (
        valid
        & np.isfinite(denom)
        & (denom > 0)
        & np.isfinite(depth)
        & (depth > 0)
        & np.isfinite(s2_grid)
        & (s2_grid > 0)
        & ~is_step_like_grid
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        white_snr_masked = np.where(ok_trial, white_snr_grid, -np.inf)

    if not np.any(np.isfinite(white_snr_masked) & (white_snr_masked > -np.inf)):
        result = StarSearchResult(ok=False, aper=aper, n_good=n_good)
        if keep_grid:
            result.t_good, result.y_good, result.err_good = t_good, y_good, err_good
            result.snr_grid, result.tc_grid, result.durations = (
                white_snr_masked, tc_grid, durations_d
            )
        return result

    # Two-pass (bounded at 3 total picks) red-noise-corrected argmax: beta
    # depends on which trial is currently best (its residuals are computed
    # at that trial's own duration and depth) and the best trial can depend
    # on beta, so this iterates to a fixed point rather than assuming the
    # white-SNR winner is already the final answer.
    d_idx, t_idx = np.unravel_index(int(np.argmax(white_snr_masked)), white_snr_masked.shape)
    snr_grid = white_snr_masked
    beta_by_dur = np.ones(durations_d.shape[0])
    for _pass_idx in range(3):
        c_hat_here = float(c_hat[d_idx, t_idx])
        in_box_mask = (
            (t_good >= tc_grid[t_idx] - durations_d[d_idx] / 2.0)
            & (t_good <= tc_grid[t_idx] + durations_d[d_idx] / 2.0)
        )
        # Joint residual: adding the box also shifts the nuisance
        # coefficients by -c * M^-1 X^T W box, so the box regressor
        # effectively fit is the box minus its projection onto X.
        box_perp = in_box_mask - x_base @ proj[d_idx, t_idx]
        joint_resid = resid_base - c_hat_here * box_perp
        beta_by_dur = np.array(
            [_red_noise_beta(joint_resid, t_good, d, settings.beta_min_bins) for d in durations_d]
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            snr_grid = np.where(ok_trial, white_snr_grid / beta_by_dur[:, None], -np.inf)
        new_d_idx, new_t_idx = np.unravel_index(int(np.argmax(snr_grid)), snr_grid.shape)
        if (new_d_idx, new_t_idx) == (d_idx, t_idx):
            break
        d_idx, t_idx = new_d_idx, new_t_idx

    best_snr = float(snr_grid[d_idx, t_idx])
    best_depth = float(depth[d_idx, t_idx])
    best_tc = float(tc_grid[t_idx])
    best_dur = float(durations_d[d_idx])
    best_n_in = int(n_in[d_idx, t_idx])
    best_beta = float(beta_by_dur[d_idx])
    best_denom = float(denom[d_idx, t_idx])
    best_coverage = float(coverage[d_idx, t_idx])
    best_s2 = float(s2_grid[d_idx, t_idx])
    chi2_box = chi2_base - best_depth**2 * best_denom
    dchi2_flat = chi2_base - chi2_box
    is_partial = best_coverage < 0.999

    # Step model, same nuisance: 1 for t >= t_step, else 0 -- only evaluated
    # for a fully-inside-data winning trial (see the grid-wide exclusion
    # above for why a partial trial cannot be tested this way). The step's
    # own position is not separately optimised; it is tried at the box's
    # centre and at both edges and the best of the three is kept, since a
    # genuine step can dominate a box fit from either edge as well as from
    # its centre.
    dchi2_step = np.nan
    is_step_like = False
    if not is_partial:
        step_candidates = np.clip(
            [best_tc - best_dur / 2.0, best_tc, best_tc + best_dur / 2.0], data_lo, data_hi
        )
        chi2_step_best = np.inf
        for t_step in step_candidates:
            step_idx = int(np.searchsorted(t_good, t_step, side="left"))
            # A step position within min_side points of either data edge
            # cannot be distinguished from the box itself; skip it rather
            # than let that degeneracy masquerade as evidence of a step.
            if step_idx < min_side or step_idx > n_good - min_side:
                continue
            b_w_step = s_w[n_good] - s_w[step_idx]
            b_wy_step = s_wy[n_good] - s_wy[step_idx]
            b_wx_step = s_wx[n_good] - s_wx[step_idx]
            proj_step = m_inv @ b_wx_step
            denom_step = b_w_step - b_wx_step @ proj_step
            if denom_step <= 0:
                continue
            c_step = (b_wy_step - proj_step @ xwy) / denom_step
            chi2_step_candidate = chi2_base - c_step**2 * denom_step
            chi2_step_best = min(chi2_step_best, chi2_step_candidate)
        dchi2_step = (chi2_step_best - chi2_box) if np.isfinite(chi2_step_best) else np.nan
        is_step_like = np.isfinite(dchi2_step) and dchi2_step <= settings.step_delta_chi2_threshold

    edge_margin = settings.edge_fraction * best_dur
    is_edge = (best_tc - best_dur / 2.0) <= (data_lo + edge_margin) or (
        best_tc + best_dur / 2.0
    ) >= (data_hi - edge_margin)
    is_too_deep = best_depth > settings.too_deep_fraction
    is_few_points = best_n_in < settings.few_points_threshold
    is_high_beta = best_beta > settings.high_beta_threshold

    # Cross-aperture depth check, at the same (tc, duration): recompute the
    # joint fit directly (only n_aper evaluations, not a grid) for whichever
    # apertures the caller supplies via depth_per_aper/sigma_depth_per_aper.
    depth_per_aper = np.full(max(n_aper, 1), np.nan)
    sigma_depth_per_aper = np.full(max(n_aper, 1), np.nan)
    if 0 <= aper < depth_per_aper.size:
        depth_per_aper[aper] = best_depth
        se = np.sqrt(best_s2 / best_denom) if best_denom > 0 and np.isfinite(best_s2) else np.nan
        sigma_depth_per_aper[aper] = se

    result = StarSearchResult(
        ok=True,
        aper=aper,
        n_good=n_good,
        snr=best_snr,
        depth=best_depth,
        tc=best_tc,
        duration=best_dur,
        n_in=best_n_in,
        beta=best_beta,
        coverage=best_coverage,
        partial=bool(is_partial),
        dchi2_box_vs_flat=float(dchi2_flat),
        dchi2_box_vs_step=float(dchi2_step),
        depth_per_aper=depth_per_aper,
        sigma_depth_per_aper=sigma_depth_per_aper,
        edge=bool(is_edge),
        too_deep=bool(is_too_deep),
        step_like=bool(is_step_like),
        high_beta=bool(is_high_beta),
        few_points=bool(is_few_points),
    )
    if keep_grid:
        in_best = (t_good >= best_tc - best_dur / 2.0) & (t_good <= best_tc + best_dur / 2.0)
        model_good = x_base @ beta_base - best_depth * in_best
        result.t_good = t_good
        result.y_good = y_good
        result.err_good = err_good
        result.model_good = model_good
        result.snr_grid = snr_grid
        result.tc_grid = tc_grid
        result.durations = durations_d
    return result


def depth_at_other_aperture(
    t: np.ndarray,
    y: np.ndarray,
    err: np.ndarray,
    good: np.ndarray,
    cbv_rows: np.ndarray,
    tc: float,
    duration: float,
    settings: SearchSettings,
) -> tuple[float, float]:
    """Depth and its formal error at a fixed ``(tc, duration)``, for one aperture.

    Used to fill in the other apertures' entries of ``depth_per_aper`` once
    the search's own best aperture has fixed the event's timing. The noise
    scale is the joint (nuisance + box) fit's own chi2/dof, for the same
    reason :func:`search_one_star` uses it: a nuisance-only residual still
    contains the event.
    """
    idx_good = np.nonzero(good)[0]
    if idx_good.size < 5:
        return np.nan, np.nan
    order = np.argsort(t[idx_good])
    idx_good = idx_good[order]
    t_g = t[idx_good].astype(np.float64)
    y_g = y[idx_good].astype(np.float64)
    err_g = err[idx_good].astype(np.float64)
    cbv_g = cbv_rows[:, idx_good] if cbv_rows.size else cbv_rows.reshape(0, idx_good.size)

    keep = robust_clip_series(y_g, settings.lc_clip_sigma, settings.lc_clip_window)
    if not np.all(keep):
        t_g, y_g, err_g = t_g[keep], y_g[keep], err_g[keep]
        cbv_g = cbv_g[:, keep]
        if t_g.shape[0] < 5:
            return np.nan, np.nan

    w_g = 1.0 / err_g**2
    base = fit_nuisance_model(t_g, y_g, w_g, cbv_g, settings.poly_degree)
    if base is None:
        return np.nan, np.nan
    x_base, _beta, _resid_base, _chi2_base, _dof_base = base

    in_box = (t_g >= tc - duration / 2.0) & (t_g <= tc + duration / 2.0)
    if np.count_nonzero(in_box) < 3:
        return np.nan, np.nan

    x_full = np.column_stack([x_base, in_box.astype(np.float64)])
    dof_white = t_g.shape[0] - x_full.shape[1]
    if dof_white < 1:
        return np.nan, np.nan

    xw = x_full * w_g[:, None]
    try:
        cov = np.linalg.inv(xw.T @ x_full)
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    beta = cov @ (xw.T @ y_g)
    resid_full = y_g - x_full @ beta
    chi2_full = float(np.sum(w_g * resid_full**2))
    s2 = chi2_full / dof_white
    depth = -float(beta[-1])
    se = float(np.sqrt(s2 * cov[-1, -1]))
    return depth, se


def search_transits(
    night: MatchedNight,
    tilemap: TileMap,
    cotrend_result: CotrendResult,
    comparison_result: ComparisonResult,
    lc: np.ndarray,
    lc_err: np.ndarray,
    epoch_ok: np.ndarray,
    frame_kept: np.ndarray,
    star_best_aper: np.ndarray,
    settings: Settings,
) -> TransitSearchResult:
    """Single-event search for every star with enough epochs at its best aperture.

    ``lc``/``lc_err``/``epoch_ok`` are ``(n_stars, n_frames, n_aper)`` /
    ``(n_stars, n_frames, n_aper)`` / ``(n_stars, n_frames)`` -- the
    decorrelated light curves (:class:`~relphot.lightcurve.LightCurveResult`).
    Every star's error is rescaled per frame by
    :func:`relphot.cotrend.compute_frame_error_scale` (built once here, from
    ``comparison_result``) before searching, so a genuinely noisier stretch
    of the night or a shared instrumental glitch is down-weighted for every
    star without any per-star fit.
    """
    search = settings.search
    n_kept = int(np.count_nonzero(frame_kept))
    search = replace(search, min_epochs=search.effective_min_epochs(n_kept))
    logger.info("transit search: min epochs %d (%d kept frames)", search.min_epochs, n_kept)
    n_stars = night.n_stars
    n_frames = night.n_frames
    n_aper = night.n_aper
    bjd = np.array([m.bjd_tdb for m in night.frame_meta], dtype=np.float64)
    core_tile = tilemap.core_tile

    frame_error_scale = compute_frame_error_scale(
        tilemap, comparison_result, lc, lc_err, frame_kept, search
    )

    searched = np.zeros(n_stars, dtype=bool)
    aper_arr = np.full(n_stars, -1, dtype=np.int64)
    n_good_arr = np.zeros(n_stars, dtype=np.int64)
    snr = np.full(n_stars, np.nan)
    depth = np.full(n_stars, np.nan)
    tc = np.full(n_stars, np.nan)
    duration = np.full(n_stars, np.nan)
    n_in = np.zeros(n_stars, dtype=np.int64)
    beta = np.full(n_stars, np.nan)
    coverage = np.full(n_stars, np.nan)
    partial = np.zeros(n_stars, dtype=bool)
    dchi2_flat = np.full(n_stars, np.nan)
    dchi2_step = np.full(n_stars, np.nan)
    depth_per_aper = np.full((n_stars, n_aper), np.nan)
    sigma_depth_per_aper = np.full((n_stars, n_aper), np.nan)
    flags = np.zeros(n_stars, dtype=np.int64)

    warned_no_cbv: set[tuple[int, int]] = set()

    for i in range(n_stars):
        t_tile = core_tile[i]
        a = int(star_best_aper[i])
        if t_tile < 0 or a < 0:
            continue

        good = epoch_ok[i] & frame_kept & np.isfinite(lc[i, :, a]) & np.isfinite(lc_err[i, :, a])
        n_good_arr[i] = int(np.count_nonzero(good))
        if n_good_arr[i] < search.min_epochs:
            continue

        n_cbv_avail = int(np.count_nonzero(np.isfinite(cotrend_result.basis[t_tile, a, :, 0])))
        if n_cbv_avail == 0 and (t_tile, a) not in warned_no_cbv:
            logger.warning("tile %d, aperture %d: no CBVs available; poly-only nuisance", t_tile, a)
            warned_no_cbv.add((t_tile, a))
        cbv_rows = cotrend_result.basis[t_tile, a, :n_cbv_avail, :]

        y = lc[i, :, a]
        err = lc_err[i, :, a] * frame_error_scale[t_tile, a, :]
        med = nanmedian_quiet(np.where(good, y, np.nan))
        if not np.isfinite(med) or med == 0:
            continue
        y_norm = y / med
        err_norm = err / med

        searched[i] = True
        aper_arr[i] = a
        result = search_one_star(
            bjd, y_norm, err_norm, good, cbv_rows, search, aper=a, n_aper=n_aper
        )
        if not result.ok:
            continue

        snr[i] = result.snr
        depth[i] = result.depth
        tc[i] = result.tc
        duration[i] = result.duration
        n_in[i] = result.n_in
        beta[i] = result.beta
        coverage[i] = result.coverage
        partial[i] = result.partial
        dchi2_flat[i] = result.dchi2_box_vs_flat
        dchi2_step[i] = result.dchi2_box_vs_step
        depth_per_aper[i, a] = result.depth
        sigma_depth_per_aper[i, a] = result.sigma_depth_per_aper[a]

        for a2 in range(n_aper):
            if a2 == a:
                continue
            good2 = (
                epoch_ok[i] & frame_kept & np.isfinite(lc[i, :, a2]) & np.isfinite(lc_err[i, :, a2])
            )
            if np.count_nonzero(good2) < search.min_epochs:
                continue
            n_cbv2 = int(np.count_nonzero(np.isfinite(cotrend_result.basis[t_tile, a2, :, 0])))
            cbv_rows2 = cotrend_result.basis[t_tile, a2, :n_cbv2, :]
            err2 = lc_err[i, :, a2] * frame_error_scale[t_tile, a2, :]
            med2 = nanmedian_quiet(np.where(good2, lc[i, :, a2], np.nan))
            if not np.isfinite(med2) or med2 == 0:
                continue
            d2, se2 = depth_at_other_aperture(
                bjd,
                lc[i, :, a2] / med2,
                err2 / med2,
                good2,
                cbv_rows2,
                result.tc,
                result.duration,
                search,
            )
            depth_per_aper[i, a2] = d2
            sigma_depth_per_aper[i, a2] = se2

        flag_bits = 0
        if result.edge:
            flag_bits |= FLAG_EDGE
        if result.too_deep:
            flag_bits |= FLAG_TOO_DEEP
        if result.step_like:
            flag_bits |= FLAG_STEP_LIKE
        if result.high_beta:
            flag_bits |= FLAG_HIGH_BETA
        if result.few_points:
            flag_bits |= FLAG_FEW_POINTS
        if result.partial:
            flag_bits |= FLAG_PARTIAL

        finite_d = np.isfinite(depth_per_aper[i]) & np.isfinite(sigma_depth_per_aper[i])
        finite_d &= sigma_depth_per_aper[i] > 0
        if np.count_nonzero(finite_d) >= 2:
            d_vals = depth_per_aper[i, finite_d]
            se_vals = sigma_depth_per_aper[i, finite_d]
            chi2_aper = float(np.sum((d_vals - result.depth) ** 2 / se_vals**2))
            dof_aper = max(np.count_nonzero(finite_d) - 1, 1)
            if (chi2_aper / dof_aper) > search.aperture_inconsistent_sigma**2:
                flag_bits |= FLAG_APERTURE_INCONSISTENT

        flags[i] = flag_bits

    # Coincidence count: per tile, among stars whose own best event clears
    # coincidence_snr_threshold, how many OTHER such stars in the same tile
    # have their own best tc within half of THIS star's duration.
    coincidence_count = np.zeros(n_stars, dtype=np.int64)
    event_time_hist = np.zeros((tilemap.n_tiles, n_frames), dtype=np.int64)
    high_snr = searched & np.isfinite(snr) & (snr >= search.coincidence_snr_threshold)
    for t_tile in range(tilemap.n_tiles):
        pool = np.nonzero(high_snr & (core_tile == t_tile))[0]
        if pool.size:
            tc_pool = tc[pool]
            dur_pool = duration[pool]
            close = np.abs(tc_pool[:, None] - tc_pool[None, :]) <= (dur_pool[:, None] / 2.0)
            counts = close.sum(axis=1) - 1
            coincidence_count[pool] = counts
            frame_of = np.clip(np.searchsorted(bjd, tc_pool), 0, n_frames - 1)
            np.add.at(event_time_hist[t_tile], frame_of, 1)

        n_pool = pool.size
        if n_pool >= 2:
            frac = coincidence_count[pool].astype(np.float64) / max(n_pool - 1, 1)
            shared = pool[
                (frac >= search.coincidence_fraction_threshold)
                & (coincidence_count[pool] >= search.coincidence_min_count)
            ]
            flags[shared] |= FLAG_SHARED_EPOCH

    candidate = searched & np.isfinite(snr) & (snr >= search.snr_threshold)
    candidate &= (flags & HARD_REJECT_FLAGS) == 0

    if n_stars > 0 and not searched.any():
        logger.warning(
            "transit search: no star has >= %d good epochs (%d kept frames); nothing searched",
            search.min_epochs, n_kept,
        )

    return TransitSearchResult(
        searched=searched,
        aper=aper_arr,
        n_good=n_good_arr,
        snr=snr,
        depth=depth,
        tc=tc,
        duration=duration,
        n_in=n_in,
        beta=beta,
        coverage=coverage,
        partial=partial,
        dchi2_box_vs_flat=dchi2_flat,
        dchi2_box_vs_step=dchi2_step,
        depth_per_aper=depth_per_aper,
        sigma_depth_per_aper=sigma_depth_per_aper,
        coincidence_count=coincidence_count,
        flags=flags,
        candidate=candidate,
        event_time_hist=event_time_hist,
        frame_error_scale=frame_error_scale,
    )


def plot_transit_candidate(
    star_id: int,
    result: StarSearchResult,
    depth_per_aper: np.ndarray,
    sigma_depth_per_aper: np.ndarray,
    shared_epoch_bjd: np.ndarray,
    path: str,
) -> None:
    """Four-panel diagnostic PNG for one transit candidate.

    Panels: raw and detrended light curve with binned points and the best
    joint model; the SNR-vs-mid-time curve for every trial duration; the
    per-aperture depth cross-check; and a text summary of the fit. ``result``
    must come from :func:`search_one_star` with ``keep_grid=True``.
    Shared-epoch mid-times from other stars in the same tile
    (``shared_epoch_bjd``) are marked as vertical lines on the LC panel.

    Raises
    ------
    RelphotError
        If matplotlib is not installed.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        from relphot.exceptions import RelphotError

        msg = "install the 'lightcurve' extra: pip install 'relphot[lightcurve]'"
        raise RelphotError(msg) from None

    t = result.t_good
    y = result.y_good
    model = result.model_good
    tc, dur = result.tc, result.duration

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_lc, ax_snr, ax_aper, ax_grid = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    hours = (t - tc) * 24.0
    ax_lc.plot(hours, y, ".", color="0.6", ms=3, label="light curve")
    order = np.argsort(hours)
    nb = max(len(hours) // 40, 1)
    if len(hours) >= nb:
        hb = [hours[order][i : i + nb].mean() for i in range(0, len(hours) - nb + 1, nb)]
        yb = [y[order][i : i + nb].mean() for i in range(0, len(hours) - nb + 1, nb)]
        ax_lc.plot(hb, yb, "o", color="C0", ms=4, label="binned")
    if model is not None:
        ax_lc.plot(hours[order], model[order], "C3-", lw=1.5, label="joint model")
    for s in (-1, 1):
        ax_lc.axvline(s * dur * 12, color="k", ls=":", lw=1)
    for bjd_other in shared_epoch_bjd:
        ax_lc.axvline((bjd_other - tc) * 24.0, color="orange", ls="--", lw=0.7, alpha=0.6)
    ax_lc.set_xlabel("hours from mid-event")
    ax_lc.set_ylabel("normalised flux")
    ax_lc.set_title(f"star {star_id}: SNR={result.snr:.1f}, depth={result.depth * 100:.2f}%")
    ax_lc.legend(fontsize=8)

    if result.snr_grid is not None:
        for d_idx in range(result.snr_grid.shape[0]):
            row = result.snr_grid[d_idx]
            finite = np.isfinite(row) & (row > -np.inf)
            if np.any(finite):
                # Excluded trials as gaps (NaN), not joined across by lines.
                ax_snr.plot(
                    result.tc_grid, np.where(finite, row, np.nan), lw=1,
                    label=f"{result.durations[d_idx] * 24:.2f}h" if d_idx % 3 == 0 else None,
                )
        ax_snr.axvline(tc, color="k", ls="--", lw=1)
        ax_snr.set_xlabel("trial mid-time (BJD_TDB)")
        ax_snr.set_ylabel("SNR")
        ax_snr.legend(fontsize=6, ncol=2)
        ax_snr.set_title("SNR vs mid-time, per duration")

    aper_idx = np.arange(depth_per_aper.size)
    finite = np.isfinite(depth_per_aper)
    ax_aper.errorbar(
        aper_idx[finite], depth_per_aper[finite] * 100, yerr=sigma_depth_per_aper[finite] * 100,
        fmt="o", color="C0",
    )
    ax_aper.axhline(result.depth * 100, color="C3", ls="--", lw=1)
    ax_aper.set_xlabel("aperture index")
    ax_aper.set_ylabel("depth (%)")
    ax_aper.set_title("depth per aperture")

    ax_grid.axis("off")
    lines = [
        f"tc = {tc:.5f} BJD_TDB",
        f"duration = {dur * 24:.2f} h",
        f"n_in = {result.n_in}",
        f"coverage = {result.coverage:.2f} (partial={result.partial})",
        f"beta = {result.beta:.2f}",
        f"dchi2(box-flat) = {result.dchi2_box_vs_flat:.1f}",
        f"dchi2(box-step) = {result.dchi2_box_vs_step:.1f}",
    ]
    ax_grid.text(0.0, 0.9, "\n".join(lines), va="top", fontsize=10, family="monospace")

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    logger.info("wrote %s", path)
