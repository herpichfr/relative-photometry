"""Night-level tail screen for reference and comparison stars.

A "tail" is a run of isolated single-epoch negative dips (a few percent to ~15 %, far
outside the star's own photon noise) that a star shows even though its neighbours do
not. Such a star must not enter the reference (a dip in one member of the fixed star set
puts an upward spike into every other star's light curve) nor the comparison pool. The
star keeps its light curve and is still searched for transits; only its use as a
reference or comparison star is withdrawn, for the whole night (the reference of a tile
must be the same fixed star set in every frame).

The statistic needs neither the reference nor the comparison ensemble, so it can be
computed before either exists. Per star: the log-flux, minus the star's own median, minus
the median of the same quantity over its ``n_neighbours`` nearest bright non-flagged
stars (a local transparency estimate that follows spatial gradients and clouds), is
detrended with a running median over ``window`` epochs and divided by the star's
point-to-point sigma. A transit lasting more than ``window // 2`` epochs follows the
running median and leaves no residual; an isolated dip does. Frame-wide noise excess
(cloud) is divided out per frame. A star is tailed when it has at least ``min_low``
epochs below ``-k_sigma`` and at least ``asym_ratio`` times as many below as above
``+k_sigma``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial import cKDTree

from relphot.numeric import mad_sigma, nanmedian_quiet

if TYPE_CHECKING:
    from relphot.config import TailSettings
    from relphot.match import MatchedNight

logger = logging.getLogger(__name__)

__all__ = ["TailInfo", "running_median", "tail_eligibility", "tail_residuals"]


@dataclass(frozen=True, slots=True)
class TailInfo:
    """Night-level tail screen result.

    Attributes
    ----------
    tailed : np.ndarray
        (n_stars,) bool, True if the star shows the isolated-dip tail
    eligible : np.ndarray
        (n_stars,) bool, ``~tailed``
    n_low, n_high : np.ndarray
        (n_stars,) int, epochs below ``-k_sigma`` and above ``+k_sigma`` (0 where the star
        was not evaluated)
    n_valid : np.ndarray
        (n_stars,) int, epochs with FLAGS == 0 and finite positive flux at ``aperture``
    frame_noise : np.ndarray
        (n_frames,) per-frame noise inflation factor (>= 1) divided out of the residuals
    aperture : int
        Aperture the screen was computed at
    enabled : bool
        False if the screen was disabled or could not run (too few bright stars); then
        every star is eligible
    """

    tailed: np.ndarray
    eligible: np.ndarray
    n_low: np.ndarray
    n_high: np.ndarray
    n_valid: np.ndarray
    frame_noise: np.ndarray
    aperture: int
    enabled: bool


def running_median(a: np.ndarray, window: int) -> np.ndarray:
    """NaN-aware running median along axis 1 over an odd ``window`` of epochs.

    Same shape as ``a``. NaN where fewer than ``window // 2 + 1`` finite values fall in
    the window. A step wider than ``window // 2`` epochs passes through unchanged; a run
    of ``window // 2`` or fewer outlying epochs is removed.
    """
    half = window // 2
    padded = np.pad(a, ((0, 0), (half, half)), constant_values=np.nan)
    win = sliding_window_view(padded, window, axis=1)
    med = nanmedian_quiet(win, axis=2)
    med[np.isfinite(win).sum(axis=2) < half + 1] = np.nan
    return med


def _tail_aperture(night: MatchedNight, settings: TailSettings) -> int:
    aper = settings.aperture if settings.aperture >= 0 else (1 if night.n_aper >= 2 else 0)
    return min(aper, night.n_aper - 1)


def tail_residuals(
    night: MatchedNight, aper: int, settings: TailSettings, chunk: int = 4000
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Detrended, sigma-normalised, frame-noise-scaled residuals of every evaluated star.

    Returns ``(z, frame_noise, n_valid)``: ``z`` is ``(n_stars, n_frames)`` with NaN for
    unevaluated stars and invalid epochs, ``frame_noise`` ``(n_frames,)``, ``n_valid``
    ``(n_stars,)``. Returns None when fewer than ``n_neighbours + 1`` stars qualify as
    neighbours (``pool_min_snr``, FLAGS == 0 wherever present) or no star can be evaluated.
    """
    n, n_f = night.n_stars, night.n_frames
    window = max(3, int(settings.window))
    window += 1 - window % 2
    k = int(settings.n_neighbours)

    flux = night.flux[:, :, aper].astype(np.float64)
    present = night.flags != -1
    valid = (night.flags == 0) & np.isfinite(flux) & (flux > 0)
    n_valid = valid.sum(axis=1)
    log_flux = np.where(valid, np.log(np.where(valid, flux, 1.0)), np.nan)
    lq = log_flux - nanmedian_quiet(log_flux, axis=1)[:, None]
    snr = nanmedian_quiet(night.snr, axis=1)

    enough = (n_valid >= settings.min_epochs) & np.isfinite(night.x) & np.isfinite(night.y)
    pool = (
        enough
        & np.all((night.flags == 0) | ~present, axis=1)
        & (snr >= settings.pool_min_snr)
    )
    evaluated = enough & (snr >= settings.min_snr)
    pool_idx = np.nonzero(pool)[0]
    eval_idx = np.nonzero(evaluated)[0]
    if pool_idx.size <= k or eval_idx.size == 0:
        return None

    tree = cKDTree(np.column_stack([night.x[pool_idx], night.y[pool_idx]]))
    resid = np.full((n, n_f), np.nan)
    for c0 in range(0, eval_idx.size, chunk):
        e = eval_idx[c0 : c0 + chunk]
        _, nb = tree.query(np.column_stack([night.x[e], night.y[e]]), k=k + 1)
        nb_star = pool_idx[nb]  # (m, k + 1); the star itself, if in the pool, is dropped
        q = lq[nb_star]
        q[nb_star == e[:, None]] = np.nan
        resid[e] = lq[e] - nanmedian_quiet(q, axis=1)

    finite = np.isfinite(resid)
    diff = np.where(finite[:, 1:] & finite[:, :-1], resid[:, 1:] - resid[:, :-1], np.nan)
    sigma = mad_sigma(diff, axis=1) / np.sqrt(2.0)
    usable = (np.isfinite(diff).sum(axis=1) >= 10) & np.isfinite(sigma) & (sigma > 0)
    sigma = np.where(usable, sigma, np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        z = (resid - running_median(resid, window)) / sigma[:, None]

    frame_noise = np.ones(n_f)
    if np.count_nonzero(usable) >= 20:
        per_frame = mad_sigma(z[usable], axis=0)
        frame_noise = np.where(np.isfinite(per_frame), np.maximum(per_frame, 1.0), 1.0)
    return z / frame_noise[None, :], frame_noise, n_valid


def tail_eligibility(night: MatchedNight, settings: TailSettings) -> TailInfo:
    """Compute the night-level tail screen (see the module docstring).

    Disabled nights, and nights with too few bright neighbours, return every star
    eligible with ``enabled=False``.
    """
    n = night.n_stars
    aper = _tail_aperture(night, settings)
    zeros = np.zeros(n, dtype=np.int64)
    off = TailInfo(
        tailed=np.zeros(n, dtype=bool),
        eligible=np.ones(n, dtype=bool),
        n_low=zeros,
        n_high=zeros.copy(),
        n_valid=zeros.copy(),
        frame_noise=np.ones(night.n_frames),
        aperture=aper,
        enabled=False,
    )
    if not settings.enabled:
        logger.info("tail cut disabled")
        return off
    if night.n_aper == 0:
        return off
    out = tail_residuals(night, aper, settings)
    if out is None:
        logger.warning("tail cut skipped: too few bright neighbour stars")
        return off

    z, frame_noise, n_valid = out
    n_low = np.sum(z < -settings.k_sigma, axis=1).astype(np.int64)
    n_high = np.sum(z > settings.k_sigma, axis=1).astype(np.int64)
    tailed = (n_low >= settings.min_low) & (n_low >= settings.asym_ratio * n_high)
    logger.info(
        "tails: aperture %d; tailed %d/%d stars (>= %d epochs below -%.1f sigma, "
        ">= %.1f x the number above); %d frames with noise factor > 1.5",
        aper,
        int(tailed.sum()),
        n,
        settings.min_low,
        settings.k_sigma,
        settings.asym_ratio,
        int(np.count_nonzero(frame_noise > 1.5)),
    )
    return TailInfo(
        tailed=tailed,
        eligible=~tailed,
        n_low=n_low,
        n_high=n_high,
        n_valid=n_valid.astype(np.int64),
        frame_noise=frame_noise,
        aperture=aper,
        enabled=True,
    )
