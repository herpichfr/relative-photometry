"""Stage 5b: light-curve statistics and best-aperture selection.

Computes per-star RMS, chi-squared, and expected-noise statistics from light curves.
Selects the best aperture per magnitude bin based on comparison-star RMS, then
determines each star's best aperture.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from relphot.numeric import mad_sigma, nanmedian_quiet, quantile_bin_edges

if TYPE_CHECKING:
    from relphot.comparison import ComparisonResult
    from relphot.config import Settings
    from relphot.decorrelate import DecorrelationResult
    from relphot.lightcurve import LightCurveResult
    from relphot.tiles import TileMap

logger = logging.getLogger(__name__)

__all__ = [
    "StarStats",
    "best_aperture_per_star",
    "build_diagnostics_table",
    "compute_star_stats",
    "plot_rms_vs_magnitude",
    "select_best_aperture",
]


@dataclass(slots=True)
class StarStats:
    """Per-star light-curve statistics at each aperture.

    All arrays are ``(n_stars, n_aper)``. NaN where n_epochs == 0.
    """

    rms: np.ndarray
    chi2_reduced: np.ndarray
    expected_noise: np.ndarray
    n_epochs: np.ndarray


def compute_star_stats(lc_result: LightCurveResult) -> StarStats:
    """Compute RMS, chi2, and expected noise for every star and aperture.

    Parameters
    ----------
    lc_result : LightCurveResult
        Light curve result with lc, lc_err, and epoch_ok arrays.

    Returns
    -------
    StarStats
        Per-star and per-aperture statistics, NaN where n_epochs == 0.
    """
    lc = lc_result.lc.astype(np.float64)
    lc_err = lc_result.lc_err.astype(np.float64)
    use = np.isfinite(lc) & np.isfinite(lc_err) & lc_result.epoch_ok[:, :, None]
    lc = np.where(use, lc, np.nan)
    lc_err = np.where(use, lc_err, np.nan)
    n_epochs = use.sum(axis=1).astype(np.int64)  # (n_stars, n_aper)

    med = nanmedian_quiet(lc, axis=1)  # (n_stars, n_aper)
    med = np.where(np.isfinite(med) & (med != 0), med, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = lc / med[:, None, :]
        norm_err = lc_err / med[:, None, :]
        rms = mad_sigma(norm, axis=1)
        expected_noise = nanmedian_quiet(norm_err, axis=1)
        chi2_sum = np.nansum(((norm - 1.0) / norm_err) ** 2, axis=1)
        chi2_reduced = chi2_sum / np.maximum(n_epochs - 1, 1)
    empty = n_epochs == 0
    rms = np.where(empty, np.nan, rms)
    expected_noise = np.where(empty, np.nan, expected_noise)
    chi2_reduced = np.where(empty, np.nan, chi2_reduced)

    return StarStats(
        rms=rms,
        chi2_reduced=chi2_reduced,
        expected_noise=expected_noise,
        n_epochs=n_epochs,
    )


def select_best_aperture(
    tilemap: TileMap,
    comparison_result: ComparisonResult,
    star_stats: StarStats,
    settings: Settings,
    mag_aper: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the best aperture per (tile, magnitude bin) from comparison stars.

    For each tile and each magnitude bin (using comparison stars as reference),
    choose the aperture minimizing median RMS of comparison stars in that bin.

    Parameters
    ----------
    tilemap : TileMap
        Tile mapping.
    comparison_result : ComparisonResult
        Comparison star selection and ensemble.
    star_stats : StarStats
        Per-star and per-aperture statistics.
    settings : Settings
        Settings with lightcurve.n_mag_bins.
    mag_aper : int
        Aperture index to use for computing star magnitudes (usually the reference
        aperture where reference candidates were selected).

    Returns
    -------
    best : np.ndarray
        Shape (n_tiles, n_bins), int aperture indices (or -1 if no data).
    bin_edges : np.ndarray
        Shape (n_tiles, n_bins+1), magnitude bin edges for each tile (padded with NaN).
    """
    n_tiles = tilemap.n_tiles
    n_mag_bins = settings.lightcurve.n_mag_bins

    best = np.full((n_tiles, n_mag_bins), -1, dtype=np.int64)
    bin_edges = np.full((n_tiles, n_mag_bins + 1), np.nan, dtype=np.float64)

    for t in range(n_tiles):
        # Comparison stars in this tile's core
        comp_in_tile = (comparison_result.mask[:, mag_aper]) & (tilemap.core_tile == t)
        if not np.any(comp_in_tile):
            continue

        # Magnitudes at mag_aper for this tile's comparison stars
        mag_comp = comparison_result.mag[comp_in_tile, mag_aper]
        if not np.any(np.isfinite(mag_comp)):
            continue

        # Bin edges
        edges = quantile_bin_edges(mag_comp, n_mag_bins)
        if edges.size <= 1:
            continue

        # Pad bin_edges to n_mag_bins+1
        if edges.size <= n_mag_bins + 1:
            bin_edges[t, : edges.size] = edges
            # Pad remaining with NaN (already initialized)

        # Per bin, find best aperture
        comp_idx = np.nonzero(comp_in_tile)[0]
        for bin_idx in range(min(len(edges) - 1, n_mag_bins)):
            # Stars in this bin
            lo = edges[bin_idx]
            hi = edges[bin_idx + 1]
            if bin_idx == len(edges) - 2:  # Last bin includes right edge
                in_bin = (mag_comp >= lo) & (mag_comp <= hi)
            else:
                in_bin = (mag_comp >= lo) & (mag_comp < hi)

            if not np.any(in_bin):
                best[t, bin_idx] = -1
                continue

            bin_indices = comp_idx[in_bin]
            rms_in_bin = star_stats.rms[bin_indices, :]  # (n_bin, n_aper)

            # Median RMS per aperture
            median_rms = nanmedian_quiet(rms_in_bin, axis=0)  # (n_aper,)

            # Best aperture: lowest median RMS, ties: lower index
            if np.any(np.isfinite(median_rms)):
                best[t, bin_idx] = int(np.nanargmin(median_rms))
            else:
                best[t, bin_idx] = -1

    return best, bin_edges


def best_aperture_per_star(
    tilemap: TileMap,
    comparison_result: ComparisonResult,
    best: np.ndarray,
    bin_edges: np.ndarray,
    mag_aper: int,
) -> np.ndarray:
    """Determine each star's best aperture from its magnitude and tile.

    For each star with a core tile and finite magnitude, find which bin it
    belongs to and use that bin's best aperture. If the chosen bin has no
    data (-1), use the nearest bin with data.

    Parameters
    ----------
    tilemap : TileMap
        Tile mapping.
    comparison_result : ComparisonResult
        Comparison results.
    best : np.ndarray
        Best apertures per (tile, bin) from select_best_aperture.
    bin_edges : np.ndarray
        Bin edges per tile from select_best_aperture.
    mag_aper : int
        Aperture index for magnitude reference.

    Returns
    -------
    np.ndarray
        Shape (n_stars,), int aperture indices (-1 for stars with no core tile
        or non-finite magnitude).
    """
    n_stars = tilemap.core_tile.shape[0]
    best_aper = np.full(n_stars, -1, dtype=np.int64)
    mag = comparison_result.mag[:, mag_aper]

    for t in range(best.shape[0]):
        edges = bin_edges[t][np.isfinite(bin_edges[t])]
        has_data = np.nonzero(best[t, : max(edges.size - 1, 0)] >= 0)[0]
        if edges.size <= 1 or has_data.size == 0:
            continue
        stars = np.nonzero((tilemap.core_tile == t) & np.isfinite(mag))[0]
        if stars.size == 0:
            continue
        bin_idx = np.clip(np.searchsorted(edges[:-1], mag[stars], side="right") - 1,
                          0, edges.size - 2)
        # empty bins borrow the nearest bin with data (ties: the brighter bin)
        nearest = has_data[np.argmin(np.abs(bin_idx[:, None] - has_data[None, :]), axis=1)]
        best_aper[stars] = best[t, nearest]

    return best_aper


def build_diagnostics_table(
    tilemap: TileMap,
    comparison_result: ComparisonResult,
    _decorrelation: DecorrelationResult | None,
) -> list[dict]:
    """Build per-tile diagnostics table.

    Parameters
    ----------
    tilemap : TileMap
        Tile mapping.
    comparison_result : ComparisonResult
        Comparison results.
    decorrelation : DecorrelationResult or None
        Decorrelation results, or None if disabled.

    Returns
    -------
    list[dict]
        One row per tile: tile, xmin, xmax, ymin, ymax (from TileMap.to_csv),
        n_comparison_aper{a}, n_rounds_aper{a} for each aperture.
    """
    rows = []
    for t in range(tilemap.n_tiles):
        row = {
            "tile": t,
            "xmin": int(tilemap.xmin[t]),
            "xmax": int(tilemap.xmax[t]),
            "ymin": int(tilemap.ymin[t]),
            "ymax": int(tilemap.ymax[t]),
        }
        for a in range(comparison_result.n_comparison.shape[1]):
            row[f"n_comparison_aper{a}"] = int(comparison_result.n_comparison[t, a])
            row[f"n_rounds_aper{a}"] = int(comparison_result.n_rounds_used[t, a])
        rows.append(row)
    return rows


def plot_rms_vs_magnitude(
    mag: np.ndarray,
    rms: np.ndarray,
    expected: np.ndarray,
    is_comparison: np.ndarray,
    path: str,
    title: str = "RMS vs Magnitude",
) -> None:
    """Plot RMS vs magnitude with comparison stars highlighted.

    Parameters
    ----------
    mag : np.ndarray
        Magnitudes, shape (n_stars,).
    rms : np.ndarray
        RMS values, shape (n_stars,), in mmag.
    expected : np.ndarray
        Expected noise, shape (n_stars,), in mmag.
    is_comparison : np.ndarray
        Boolean mask of comparison stars.
    path : str
        Output file path.
    title : str
        Plot title.

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

    fig, ax = plt.subplots(figsize=(10, 6))

    # All stars
    finite = np.isfinite(mag) & np.isfinite(rms)
    ax.scatter(mag[finite & ~is_comparison], rms[finite & ~is_comparison] * 1000,
               alpha=0.5, s=20, label="non-comparison", color="gray")

    # Comparison stars
    comp_finite = finite & is_comparison
    ax.scatter(mag[comp_finite], rms[comp_finite] * 1000,
               alpha=0.7, s=30, label="comparison", color="blue")

    # Running median of expected noise
    if np.any(finite):
        mag_sort_idx = np.argsort(mag[finite])
        mag_sorted = mag[finite][mag_sort_idx]
        exp_sorted = expected[finite][mag_sort_idx]
        window = max(1, len(mag_sorted) // 20)
        med_exp = np.convolve(exp_sorted, np.ones(window) / window, mode="valid")
        mag_med = mag_sorted[window // 2 : len(mag_sorted) - (window - window // 2 - 1)]
        ax.plot(mag_med, med_exp * 1000, "r-", linewidth=2, label="expected median")

    ax.set_xlabel("Magnitude")
    ax.set_ylabel("RMS (mmag)")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)
    logger.info("wrote %s", path)
