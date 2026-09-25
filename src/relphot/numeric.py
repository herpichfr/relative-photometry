"""Shared numeric utilities for NaN-aware statistics and weighted combining.

Low-level functions used across :mod:`relphot.reference`, :mod:`relphot.match`,
and future stages.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np

__all__ = [
    "fit_noise_floor",
    "mad_sigma",
    "nanmedian_quiet",
    "quantile_bin_edges",
    "unit_vectors",
    "weighted_clipped_combine",
]


def nanmedian_quiet(arr: np.ndarray, axis: int | None = None) -> np.ndarray:
    """``np.nanmedian``, without the "All-NaN slice" warning an expected NaN result raises."""
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
    ``min_bin_stars`` members are dropped. Returns a function that
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
        Number of bins to attempt.
    min_bin_stars : int
        Minimum number of stars required to keep a bin.

    Returns
    -------
    callable
        Function ``f(m)`` returning ``10 ** np.interp(m, bin_centres, log10_floors)``,
        where ``log10_floors`` are the median log10(sigma) in each bin.
        If no bin survives the cut, returns a function giving NaN.
    """
    edges = quantile_bin_edges(mag[valid], n_bins)
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
