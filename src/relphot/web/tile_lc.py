"""Tile light-curve helpers for the web API (numpy only; no database access).

Functions to compute member RMS, envelope statistics, and select which members to display
in the comparison light-curve endpoint.
"""

from __future__ import annotations

import numpy as np

__all__ = ["envelope", "member_rms", "select_members"]


def member_rms(norm_flux: np.ndarray, ens: np.ndarray) -> np.ndarray:
    """RMS of each member's normalised flux relative to ensemble, over finite frames.

    Parameters
    ----------
    norm_flux : (M, F) array
        Normalised flux of M members over F frames.
    ens : (F,) array
        Ensemble flux over F frames.

    Returns
    -------
    (M,) array
        RMS per member, or NaN if < 5 finite points.
    """
    norm_flux = np.asarray(norm_flux, dtype=np.float32)
    ens = np.asarray(ens, dtype=np.float32)
    m, f = norm_flux.shape
    assert ens.shape == (f,)
    rms_array = np.full(m, np.nan, dtype=np.float32)
    for i in range(m):
        finite = np.isfinite(norm_flux[i, :]) & np.isfinite(ens)
        if np.sum(finite) >= 5:
            # Relative RMS: 1.4826 * MAD of norm_flux/ens
            ratio = norm_flux[i, finite] / ens[finite]
            median_ratio = np.nanmedian(ratio)
            residuals = np.abs(ratio - median_ratio)
            mad = np.median(residuals)
            rms_array[i] = 1.4826 * mad
    return rms_array


def envelope(
    norm_flux: np.ndarray,
) -> dict[str, list[float | None]] | None:
    """16th, 50th, 84th percentiles of members' normalised flux per frame.

    Parameters
    ----------
    norm_flux : (M, F) array
        Normalised flux of M members over F frames.

    Returns
    -------
    dict or None
        {"median": [...], "lo": [...], "hi": [...]} per-frame percentiles,
        or None where < 3 finite members.
    """
    norm_flux = np.asarray(norm_flux, dtype=np.float32)
    _, f = norm_flux.shape
    with np.errstate(invalid="ignore"):
        median = []
        lo = []
        hi = []
        for j in range(f):
            finite = np.isfinite(norm_flux[:, j])
            if np.sum(finite) < 3:
                median.append(None)
                lo.append(None)
                hi.append(None)
            else:
                vals = norm_flux[finite, j]
                median.append(float(np.nanpercentile(vals, 50)))
                lo.append(float(np.nanpercentile(vals, 16)))
                hi.append(float(np.nanpercentile(vals, 84)))
    return {"median": median, "lo": lo, "hi": hi}


def select_members(
    order: str,
    limit: int,
    mag: np.ndarray | None = None,
    weight: np.ndarray | None = None,
    rms: np.ndarray | None = None,
    must_include: set[int] | None = None,
) -> np.ndarray:
    """Select which members to display, ordered by limit, mag/weight/rms, must_include.

    Parameters
    ----------
    order : str
        Sort criterion: "mag", "weight", "rms".
    limit : int
        Maximum members to display (0 -> all).
    mag : (M,) array, optional
        Apparent magnitude per member (for "mag" order).
    weight : (M,) array, optional
        Weight per member (for "weight" order).
    rms : (M,) array, optional
        RMS per member (for "rms" order).
    must_include : set of int, optional
        Member indices that must be included.

    Returns
    -------
    (L,) array
        Sorted indices of members to display, must_include first.
    """
    must_include = must_include or set()
    if mag is not None:
        n = mag.shape[0]
    elif weight is not None:
        n = weight.shape[0]
    else:
        n = rms.shape[0]
    if limit <= 0 or limit >= n:
        limit = n

    # Sort by the given criterion
    if order == "mag":
        # Magnitude: stratified sampling across the range
        idx_sorted = np.argsort(mag)
        sampled = np.unique(np.linspace(0, n - 1, limit).round().astype(int))
        selected = idx_sorted[sampled]
    elif order == "weight":
        # Weight: top weight
        idx_sorted = np.argsort(-weight)  # descending
        selected = idx_sorted[:limit]
    elif order == "rms":
        # RMS: worst (largest) rms
        # Handle NaN by putting them at the end
        idx_sorted = np.argsort(rms)  # NaN goes to the end
        # Reverse to get worst first, but skip NaN
        finite_indices = idx_sorted[~np.isnan(rms[idx_sorted])]
        selected = np.flip(finite_indices)[:limit]
    else:
        selected = np.arange(n)

    # Add must_include members
    result = np.array(list(must_include))
    for idx in selected:
        if idx not in must_include:
            result = np.append(result, idx)
            if len(result) >= limit:
                break
    return result
