"""Shared numeric utilities for NaN-aware statistics and weighted combining.

Low-level functions used across :mod:`relphot.reference`, :mod:`relphot.match`,
and future stages.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np

__all__ = [
    "edge_outlier_mask",
    "fit_noise_floor",
    "mad_sigma",
    "nanmedian_quiet",
    "quantile_bin_edges",
    "robust_clip_series",
    "unit_vectors",
    "weighted_clipped_combine",
]


#: ``np.nanmedian`` along an axis this long (or longer) of an array that has NaN falls back to a
#: Python-level ``apply_along_axis`` loop, which is ~50x slower than the vectorised path it uses
#: for shorter axes (a 600-star tile median over 67 frames, for example).
_NUMPY_NANMEDIAN_SLOW_AXIS = 600


def _nanmedian_sorted(arr: np.ndarray, axis: int) -> np.ndarray:
    """``np.nanmedian(arr, axis)`` for a floating array, vectorised by sorting.

    NaN sorts last, so the valid values of every slice come first; the median is the mean of the
    two middle ones (equal when the count is odd). NaN where a slice has no valid value. Gives the
    same values as numpy's, only without its per-slice Python loop.
    """
    arr = np.moveaxis(arr, axis, -1)
    n_valid = np.count_nonzero(~np.isnan(arr), axis=-1)
    srt = np.sort(arr, axis=-1)
    lo = np.take_along_axis(srt, np.maximum((n_valid - 1) // 2, 0)[..., None], axis=-1)[..., 0]
    hi = np.take_along_axis(srt, np.maximum(n_valid // 2, 0)[..., None], axis=-1)[..., 0]
    out = (lo + hi) / 2
    return np.where(n_valid > 0, out, np.nan).astype(arr.dtype, copy=False)


def nanmedian_quiet(arr: np.ndarray, axis: int | None = None) -> np.ndarray:
    """``np.nanmedian``, without the "All-NaN slice" warning an expected NaN result raises.

    Long axes of floating arrays are reduced by :func:`_nanmedian_sorted` (same values, much
    faster than numpy's per-slice fallback).
    """
    arr = np.asarray(arr)
    if (
        axis is not None
        and arr.ndim > 1
        and arr.dtype.kind == "f"
        and arr.shape[axis] >= _NUMPY_NANMEDIAN_SLOW_AXIS
    ):
        return _nanmedian_sorted(arr, axis)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(arr, axis=axis)


def mad_sigma(arr: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Median absolute deviation, scaled to a standard-deviation estimate.

    Returns ``1.4826 * MAD`` where ``MAD = nanmedian(|arr - median(arr)|)``.
    For a Gaussian sample, this equals the standard deviation.
    When ``MAD == 0``, returns ``np.inf``. When all values along the axis
    are NaN, returns ``NaN``.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    axis : int or None, optional
        Axis along which to compute. If None, compute over the flattened array.

    Returns
    -------
    np.ndarray
        MAD * 1.4826, same shape as ``arr`` with ``axis`` reduced away.
    """
    median = nanmedian_quiet(arr, axis=axis)
    if axis is not None:
        median = np.expand_dims(median, axis)
    mad = nanmedian_quiet(np.abs(arr - median), axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isfinite(mad), np.where(mad > 0, 1.4826 * mad, np.inf), np.nan)


def unit_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """(N, 3) unit vectors on the sky sphere for an array of RA/Dec in degrees.

    Parameters
    ----------
    ra_deg : np.ndarray
        Right ascension in degrees, shape (N,).
    dec_deg : np.ndarray
        Declination in degrees, shape (N,).

    Returns
    -------
    np.ndarray
        Shape (N, 3) array of unit vectors.
    """
    ra = np.radians(np.asarray(ra_deg, dtype=np.float64))
    dec = np.radians(np.asarray(dec_deg, dtype=np.float64))
    cosd = np.cos(dec)
    return np.column_stack([cosd * np.cos(ra), cosd * np.sin(ra), np.sin(dec)])


def weighted_clipped_combine(
    values: np.ndarray,
    weights: np.ndarray,
    clip_sigma: float,
    max_iter: int,
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Inverse-variance weighted mean, MAD-clipped, generalized to any axis.

    Iteratively computes a weighted mean while clipping outliers based on
    MAD (median absolute deviation). NaN values in ``values`` and
    non-finite or non-positive weights are never used.

    Parameters
    ----------
    values : np.ndarray
        Data to combine.
    weights : np.ndarray
        Inverse-variance weights, same shape as ``values``.
        Non-positive or non-finite values are treated as zero weight.
    clip_sigma : float
        Clipping threshold in units of the robust standard deviation (MAD).
    max_iter : int
        Maximum number of iterations.
    axis : int, optional
        Axis along which to combine (default: 0).
        The result is reduced along this axis.

    Returns
    -------
    mean : np.ndarray
        Weighted mean, shape of ``values`` with ``axis`` dimension removed.
    sigma : np.ndarray
        Robust standard deviation (1.4826 * MAD / sqrt(max(n_used, 1))),
        same shape as ``mean``.
    n_used : np.ndarray
        Number of unclipped points, dtype int, same shape as ``mean``.
    mask : np.ndarray
        Boolean mask of final-iteration inclusion flags, same shape as ``values``.
    """
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    mask = valid.copy()

    def _weighted_mean(current_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w_masked = np.where(current_mask, weights, 0.0)
        sumw = w_masked.sum(axis=axis)
        with np.errstate(invalid="ignore", divide="ignore"):
            weighted_sum = np.nansum(np.where(current_mask, weights * values, 0.0), axis=axis)
            mean = np.where(sumw > 0, weighted_sum / sumw, np.nan)
        return mean, sumw

    for _ in range(max(int(max_iter), 1)):
        mean_val, _sumw = _weighted_mean(mask)

        # Expand mean to match values' shape for residual computation.
        # If axis is not None, we need to add back the reduced dimension.
        mean_expanded = np.expand_dims(mean_val, axis=axis)
        resid = values - mean_expanded
        masked_resid = np.where(mask, resid, np.nan)
        med = nanmedian_quiet(masked_resid, axis=axis)
        med_expanded = np.expand_dims(med, axis=axis)
        mad = nanmedian_quiet(np.abs(masked_resid - med_expanded), axis=axis)
        sigma_expanded = np.expand_dims(np.where(mad > 0, 1.4826 * mad, np.inf), axis=axis)
        new_mask = valid & (np.abs(resid - med_expanded) <= clip_sigma * sigma_expanded)
        if np.array_equal(new_mask, mask):
            mask = new_mask
            break
        mask = new_mask

    mean_val, sumw = _weighted_mean(mask)
    with np.errstate(invalid="ignore", divide="ignore"):
        n_used = mask.sum(axis=axis, dtype=np.int64)
        sigma = np.where(sumw > 0, np.sqrt(1.0 / sumw), np.nan)

    return mean_val, sigma, n_used, mask


def quantile_bin_edges(mag: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-count bin edges on finite magnitude values.

    Computes ``n_bins+1`` edges such that each bin contains roughly the same
    number of finite values from ``mag``. Duplicate edges are removed via
    ``np.unique``.

    Parameters
    ----------
    mag : np.ndarray
        Magnitude array (1-D).
    n_bins : int
        Number of bins.

    Returns
    -------
    np.ndarray
        Bin edges, shape (n_edges,) where ``n_edges <= n_bins + 1``.
        Sorted and deduplicated.
    """
    finite = np.isfinite(mag)
    mag_finite = mag[finite]
    if mag_finite.size == 0:
        return np.array([], dtype=np.float64)

    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(mag_finite, quantiles)
    return np.unique(edges)


def fit_noise_floor(
    mag: np.ndarray,
    sigma: np.ndarray,
    valid: np.ndarray,
    n_bins: int,
    min_bin_stars: int,
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit a magnitude-dependent noise floor from scatter measurements.

    Bins stars by magnitude using equal-count bins. For each bin, computes
    the median of ``log10(sigma)`` over valid members. Bins with fewer than
    ``min_bin_stars`` members are dropped. The bin count is capped at
    ``max(1, n_valid // min_bin_stars)``, ``n_valid`` being the number of valid
    entries with finite magnitude, so a small sample (e.g. 45 stars) still gives
    bins of at least ``min_bin_stars`` stars; the cap is inactive for
    ``n_valid >= n_bins * min_bin_stars``. Returns a function that
    interpolates the floor at arbitrary magnitudes (constant beyond the ends).

    Parameters
    ----------
    mag : np.ndarray
        Magnitudes (1-D).
    sigma : np.ndarray
        Scatter estimates, same shape as ``mag``.
    valid : np.ndarray
        Boolean mask of valid entries, same shape as ``mag``.
    n_bins : int
        Number of bins to attempt (upper limit, see the cap above).
    min_bin_stars : int
        Minimum number of stars required to keep a bin.

    Returns
    -------
    callable
        Function ``f(m)`` returning ``10 ** np.interp(m, bin_centres, log10_floors)``,
        where ``log10_floors`` are the median log10(sigma) in each bin.
        If no bin survives the cut, returns a function giving NaN.
    """
    n_valid = int(np.count_nonzero(valid & np.isfinite(mag)))
    n_bins_eff = max(1, min(n_bins, n_valid // max(min_bin_stars, 1)))
    edges = quantile_bin_edges(mag[valid], n_bins_eff)
    if edges.size <= 1:
        return lambda m: np.full_like(m, np.nan, dtype=np.float64)

    centres_list = []
    floors_list = []

    for i in range(len(edges) - 1):
        in_bin = valid & (mag >= edges[i]) & (mag < edges[i + 1])
        # For the last bin, include the right edge.
        if i == len(edges) - 2:
            in_bin = valid & (mag >= edges[i]) & (mag <= edges[i + 1])

        n_in_bin = np.count_nonzero(in_bin)
        if n_in_bin >= min_bin_stars:
            sigma_in_bin = sigma[in_bin]
            with np.errstate(invalid="ignore", divide="ignore"):
                log10_sigma = np.log10(sigma_in_bin)
            floor = nanmedian_quiet(log10_sigma)
            if np.isfinite(floor):
                centres_list.append((edges[i] + edges[i + 1]) / 2.0)
                floors_list.append(floor)

    if not centres_list:
        return lambda m: np.full_like(m, np.nan, dtype=np.float64)

    centres = np.asarray(centres_list, dtype=np.float64)
    floors = np.asarray(floors_list, dtype=np.float64)

    def interp_floor(m: np.ndarray) -> np.ndarray:
        """Interpolated floor at magnitudes m."""
        m = np.asarray(m, dtype=np.float64)
        log10_floor = np.interp(m, centres, floors, left=floors[0], right=floors[-1])
        return 10.0**log10_floor

    return interp_floor


def robust_clip_series(y: np.ndarray, clip_sigma: float, window: int) -> np.ndarray:
    """Boolean keep-mask from a rolling-median robust-sigma clip.

    Flags a point bad when it deviates from a rolling median (window
    ``window``, taken over the series in whatever order ``y`` is already
    in -- callers pass a time-sorted series) by more than ``clip_sigma``
    robust standard deviations (:func:`mad_sigma` of the rolling-median
    residual, computed once over the whole series). Meant to remove
    isolated single-epoch outliers (cosmic rays, satellite trails, a
    dropped frame) before a downstream fit, without touching a real,
    sustained multi-epoch feature at ordinary scatter levels -- a fixed,
    global clip threshold this far out cannot mistake a percent-level
    transit dip for a point in need of clipping. Returns all-``True`` when
    ``y`` is too short (< 3 points) or the residual scatter is zero/non-finite.

    Blind spot: ``mode="nearest"`` pads the series with copies of its first/last point, so
    the rolling median of the first and last epoch holds 8 of 15 copies of the point itself
    and its residual is exactly 0. An outlier at either end of the series is therefore
    never clipped here, however large; :func:`edge_outlier_mask` covers that case and callers
    that need it combine the two masks.

    Parameters
    ----------
    y : np.ndarray
        Time-sorted 1-D series.
    clip_sigma : float
        Clipping threshold in robust standard deviations.
    window : int
        Rolling-median window size (points), capped to ``len(y)``.

    Returns
    -------
    np.ndarray
        Boolean keep-mask, same shape as ``y``.
    """
    from scipy.ndimage import median_filter

    if y.size < 3:
        return np.ones_like(y, dtype=bool)
    running_med = median_filter(y, size=max(min(int(window), y.size), 1), mode="nearest")
    resid = y - running_med
    sigma = mad_sigma(resid)
    if not np.isfinite(sigma) or sigma <= 0:
        return np.ones_like(y, dtype=bool)
    return np.abs(resid) <= clip_sigma * sigma


def edge_outlier_mask(
    t: np.ndarray,
    y: np.ndarray,
    err: np.ndarray | None = None,
    z: float = 6.0,
    k_max: int = 2,
    n_ref: int = 6,
    z_ref: float = 3.0,
    min_n: int = 15,
) -> np.ndarray:
    """Boolean keep-mask that drops an isolated outlier run of 1..``k_max`` epochs at either end.

    :func:`robust_clip_series` cannot see the first and last epoch of a series (see its
    docstring), and a high edge epoch makes the rest of a night look like a dip. At each end
    of the time-sorted, finite epochs the ``j`` outermost epochs (``j = k_max .. 1``, the
    largest first) are dropped when all of them deviate from the median of the next
    ``n_ref`` epochs (``y_end[j:j+n_ref]``) by more than ``z * s`` on the same side, and the
    epoch right after them is consistent with that median within ``z_ref * s``, i.e. the run
    ends sharply. A run of ``k_max + 1`` or more deviant epochs, a trend and a ramp therefore
    never qualify (a real partial transit at the start or end of a night is kept).

    ``s = max(1.4826 * MAD(diff(y)) / sqrt(2), median(err), 1e-4)`` is a robust
    point-to-point scatter, floored by the median error (when ``err`` is given) and by
    ``1e-4`` (``y`` is a flux normalised to about 1). Two-sided: a low edge epoch is treated
    like a high one. Non-finite epochs are neither judged nor dropped (kept ``True``), the
    input need not be sorted, and nothing is dropped when fewer than ``min_n`` finite epochs
    exist or when ``k_max < 1``.

    Parameters
    ----------
    t, y : np.ndarray
        Epoch times and the (normalised) flux, same shape.
    err : np.ndarray or None
        Flux errors, same normalisation as ``y``.
    z : float
        Deviation threshold of the dropped epochs, in units of ``s``.
    k_max : int
        Longest run of edge epochs that may be dropped.
    n_ref : int
        Number of epochs, after the run, whose median is the reference level.
    z_ref : float
        The first epoch after the run must lie within ``z_ref * s`` of the reference.
    min_n : int
        Fewest finite epochs for the clip to act at all.

    Returns
    -------
    np.ndarray
        Boolean keep-mask in the input order, same shape as ``y``.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.ones(y.shape, dtype=bool)
    k_max, n_ref = int(k_max), max(int(n_ref), 1)
    finite = np.flatnonzero(np.isfinite(t) & np.isfinite(y))
    if k_max < 1 or finite.size < max(int(min_n), n_ref + k_max + 1):
        return keep
    order = finite[np.argsort(t[finite], kind="stable")]
    ys = y[order]
    diffs = np.diff(ys)
    sigma = 1.4826 * float(np.median(np.abs(diffs - np.median(diffs)))) / np.sqrt(2.0)
    err_floor = 0.0
    if err is not None:
        err_f = np.asarray(err, dtype=np.float64)[order]
        err_f = err_f[np.isfinite(err_f)]
        if err_f.size:
            err_floor = float(np.median(err_f))
    s = max(sigma, err_floor, 1e-4)

    n = ys.size
    for at_start in (True, False):
        y_end = ys if at_start else ys[::-1]
        for j in range(k_max, 0, -1):
            ref = float(np.median(y_end[j : j + n_ref]))
            if abs(y_end[j] - ref) > z_ref * s:
                continue  # the run does not end sharply
            dev = y_end[:j] - ref
            if np.all(np.abs(dev) > z * s) and (np.all(dev > 0) or np.all(dev < 0)):
                dropped = order[:j] if at_start else order[n - j :]
                keep[dropped] = False
                break
    return keep
