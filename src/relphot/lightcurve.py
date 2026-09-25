"""Light curve extraction and decorrelation.

Computes the final light curves from a reference and comparison ensemble,
optionally applying frame-level and stellar-property-dependent decorrelation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from relphot.decorrelate import fit_decorrelation

if TYPE_CHECKING:
    from relphot.comparison import ComparisonResult
    from relphot.config import Settings
    from relphot.decorrelate import DecorrelationResult
    from relphot.match import MatchedNight
    from relphot.reference import ReferenceResult
    from relphot.tiles import TileMap

logger = logging.getLogger(__name__)

__all__ = [
    "LightCurveResult",
    "compute_light_curves",
]


@dataclass(slots=True)
class LightCurveResult:
    """Final light curves, errors, and auxiliary information.

    ``lc`` is the final light curve, ``lc_err`` is the photometric error,
    ``lc_raw`` is before decorrelation. ``epoch_ok`` is a boolean mask of
    valid epochs. ``decorrelation`` is the DecorrelationResult if enabled,
    else None.
    """

    lc: np.ndarray
    lc_err: np.ndarray
    lc_raw: np.ndarray
    epoch_ok: np.ndarray
    decorrelation: DecorrelationResult | None


def compute_light_curves(
    night: MatchedNight,
    tilemap: TileMap,
    reference_result: ReferenceResult,
    comparison_result: ComparisonResult,
    settings: Settings,
) -> LightCurveResult:
    """Compute light curves from reference and comparison ensemble.

    For each star with a core tile, computes the raw light curve as
    relative_flux / ensemble. Errors are propagated from photometric noise
    and ensemble uncertainty. Optionally applies decorrelation to remove
    frame-level and stellar-property-dependent scatter.

    Parameters
    ----------
    night : MatchedNight
        Matched night data.
    tilemap : TileMap
        Tile mapping (used to identify core tiles).
    reference_result : ReferenceResult
        Reference fluxes and relative flux.
    comparison_result : ComparisonResult
        Comparison ensemble and error estimates.
    settings : Settings
        Settings including lightcurve and decorrelation options.

    Returns
    -------
    LightCurveResult
        Computed light curves and metadata.
    """
    n_stars = night.n_stars
    n_frames = night.n_frames
    n_aper = night.n_aper

    # Step 1: Compute raw light curve
    core_tile = tilemap.core_tile
    has_core = core_tile >= 0

    # lc_raw has NaN for stars without a core tile
    lc_raw = np.full((n_stars, n_frames, n_aper), np.nan, dtype=np.float32)

    # For stars with a core tile, divide by that tile's ensemble
    idx = np.nonzero(has_core)[0]
    tile_of = core_tile[idx]
    ens_star = comparison_result.ensemble[tile_of]  # (n_core, n_frames, n_aper)
    with np.errstate(invalid="ignore", divide="ignore"):
        lc_raw[idx] = reference_result.relative_flux[idx] / ens_star

    # Step 2: Determine epoch_ok mask
    # frame_kept & (night.flags != -1) & ((night.flags & bad_flag_mask) == 0)
    frame_kept_exp = reference_result.frame_kept[None, :]  # (1, n_frames)
    flags_present = night.flags != -1  # (n_stars, n_frames)
    bad_flag_mask = settings.lightcurve.bad_flag_mask
    flags_ok = (night.flags & bad_flag_mask) == 0  # (n_stars, n_frames)
    epoch_ok = frame_kept_exp & flags_present & flags_ok  # (n_stars, n_frames)

    # Mask lc_raw where not epoch_ok
    lc_raw = np.where(epoch_ok[:, :, None], lc_raw, np.nan)

    # Step 3: Propagate uncertainties
    # sigma_phot = night.fluxerr / R[core_tile]
    # sigma_ens = comparison.sigma_ensemble[core_tile]
    # lc_err = sqrt((sigma_phot / ens)^2 + (lc_raw * sigma_ens / ens)^2)
    R = reference_result.R  # (n_tiles, n_frames, n_aper)

    lc_err = np.full((n_stars, n_frames, n_aper), np.nan, dtype=np.float32)

    with np.errstate(invalid="ignore", divide="ignore"):
        sigma_phot = night.fluxerr[idx] / R[tile_of]
        sigma_ens = comparison_result.sigma_ensemble[tile_of]
        lc_err[idx] = np.sqrt(
            (sigma_phot / ens_star) ** 2 + (lc_raw[idx] * sigma_ens / ens_star) ** 2
        )
    lc_err = np.where(epoch_ok[:, :, None], lc_err, np.nan)

    # Step 4: Decorrelation (if enabled)
    lc = lc_raw.copy()
    decorrelation = None

    if settings.decorrelation.enabled:
        decorrelation = fit_decorrelation(
            night,
            tilemap,
            comparison_result,
            lc_raw,
            reference_result.frame_kept,
            settings.decorrelation,
        )
        lc = decorrelation.lc_corrected

        # Scale lc_err by the same per-epoch factor: lc_err * (lc / lc_raw)
        with np.errstate(invalid="ignore", divide="ignore"):
            factor = np.where(np.isfinite(lc_raw) & (lc_raw != 0), lc / lc_raw, 1.0)
            lc_err = lc_err * factor

    return LightCurveResult(
        lc=lc,
        lc_err=lc_err,
        lc_raw=lc_raw,
        epoch_ok=epoch_ok,
        decorrelation=decorrelation,
    )
