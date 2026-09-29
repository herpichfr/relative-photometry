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
from relphot.exceptions import ConfigError
from relphot.numeric import mad_sigma, nanmedian_quiet

if TYPE_CHECKING:
    from relphot.comparison import ComparisonResult
    from relphot.config import Settings
    from relphot.decorrelate import DecorrelationResult
    from relphot.match import MatchedNight
    from relphot.reference import ReferenceResult
    from relphot.tiles import TileMap

logger = logging.getLogger(__name__)

__all__ = [
    "INFLATE_MODES",
    "NEIGHBOUR_FLAG_MASK",
    "LightCurveResult",
    "compute_light_curves",
    "error_inflation",
    "point_to_point_sigma",
]

#: SExtractor FLAGS bits 1 (has neighbours) and 2 (blended with a neighbour); the bits
#: ``LightcurveSettings.bad_flag_mask`` (default 252) leaves out so the epoch is kept.
NEIGHBOUR_FLAG_MASK = 3

#: Allowed values of ``LightcurveSettings.inflate_errors``.
INFLATE_MODES = ("none", "blended", "excess", "all")


@dataclass(slots=True)
class LightCurveResult:
    """Final light curves, errors, and auxiliary information.

    ``lc`` is the final light curve, ``lc_err`` is the photometric error
    (inflated by ``err_scale`` where the settings ask for it), ``lc_raw`` is
    before decorrelation. ``epoch_ok`` is a boolean mask of valid epochs.
    ``decorrelation`` is the DecorrelationResult if enabled, else None.

    ``lc_err_raw`` is the error before inflation, ``err_scale``
    ``(n_stars, n_aper)`` the factor applied to it (1 where none was), and
    ``blended`` ``(n_stars,)`` whether the star's neighbour/blend flags are set
    in enough of its kept frames; all three are ``None`` on a result loaded from
    a product written before error inflation existed (its ``lc_err`` is then
    the raw error and the scale is 1).
    """

    lc: np.ndarray
    lc_err: np.ndarray
    lc_raw: np.ndarray
    epoch_ok: np.ndarray
    decorrelation: DecorrelationResult | None
    lc_err_raw: np.ndarray | None = None
    err_scale: np.ndarray | None = None
    blended: np.ndarray | None = None


def point_to_point_sigma(
    lc: np.ndarray, epoch_ok: np.ndarray, min_pairs: int = 10
) -> np.ndarray:
    """Robust point-to-point scatter per star and aperture, ``(n_stars, n_aper)``.

    ``MAD(first differences) / sqrt(2)`` (1.4826-scaled) over pairs of consecutive
    frames that are both kept epochs of the star. A transit's ingress and egress and any
    variability slower than the cadence add a handful of differences at most, so the
    median-based estimate ignores them: it is safe to use on a star's own light curve
    without fitting a trend (the transit-safe rule). NaN where fewer than ``min_pairs``
    differences exist, or where they are all identical.
    """
    n_stars, _n_frames, n_aper = lc.shape
    out = np.full((n_stars, n_aper), np.nan, dtype=np.float64)
    for a in range(n_aper):
        y = lc[:, :, a].astype(np.float64)
        ok = epoch_ok & np.isfinite(y)
        pair_ok = ok[:, 1:] & ok[:, :-1]
        diff = np.where(pair_ok, y[:, 1:] - y[:, :-1], np.nan)
        sigma = mad_sigma(diff, axis=1) / np.sqrt(2.0)
        good = pair_ok.sum(axis=1) >= min_pairs
        out[:, a] = np.where(good & np.isfinite(sigma), sigma, np.nan)
    return out


def error_inflation(
    lc: np.ndarray,
    lc_err: np.ndarray,
    epoch_ok: np.ndarray,
    flags: np.ndarray,
    settings: Settings,
) -> tuple[np.ndarray, np.ndarray]:
    """Per star and aperture error-inflation factor, and the per-star ``blended`` flag.

    The formal ``lc_err`` (photon noise plus ensemble error) misses noise that is not
    white photon noise: a neighbour's light entering the aperture as the seeing changes,
    and, for bright stars, a floor from scintillation, flat-fielding and non-linearity
    (on ROBO43 20250911 the bright comparison stars of WASP-145 A's brightness have the
    same point-to-point excess, a factor ~3.7, as WASP-145 A itself). The measured excess is
    ``sigma_p2p / median(lc_err)`` over kept epochs (see :func:`point_to_point_sigma`) and
    the factor is ``max(1, excess)``.

    ``blended`` is data-defined: the SExtractor neighbour/blend bits (FLAGS 1 and 2) are set
    in at least ``lightcurve.blend_min_frame_fraction`` of the star's kept frames (no
    catalogue is available at this stage). ``lightcurve.inflate_errors`` decides who gets
    the factor: ``"none"`` nobody, ``"blended"`` blended stars, ``"excess"`` blended stars
    and any star whose factor is at least ``lightcurve.err_scale_excess_min``, ``"all"``
    everybody. Stars not selected keep a factor of exactly 1.

    Returns ``(err_scale, blended)`` with shapes ``(n_stars, n_aper)`` and ``(n_stars,)``.
    """
    lcs = settings.lightcurve
    mode = lcs.inflate_errors
    if mode not in INFLATE_MODES:
        msg = f"lightcurve.inflate_errors must be one of {INFLATE_MODES}, got {mode!r}"
        raise ConfigError(msg)
    n_stars, _n_frames, n_aper = lc.shape

    n_ok = epoch_ok.sum(axis=1)
    neighbour = ((flags & NEIGHBOUR_FLAG_MASK) != 0) & epoch_ok
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = neighbour.sum(axis=1) / np.maximum(n_ok, 1)
    blended = (n_ok > 0) & (frac >= lcs.blend_min_frame_fraction)

    if mode == "none":
        return np.ones((n_stars, n_aper), dtype=np.float64), blended

    sigma_p2p = point_to_point_sigma(lc, epoch_ok, lcs.p2p_min_pairs)
    use = epoch_ok[:, :, None] & np.isfinite(lc_err)
    med_err = nanmedian_quiet(np.where(use, lc_err.astype(np.float64), np.nan), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        measured = sigma_p2p / med_err
    measured = np.where(np.isfinite(measured) & (med_err > 0), np.maximum(measured, 1.0), 1.0)

    if mode == "all":
        apply = np.ones((n_stars, n_aper), dtype=bool)
    elif mode == "blended":
        apply = np.broadcast_to(blended[:, None], (n_stars, n_aper))
    else:
        apply = blended[:, None] | (measured >= lcs.err_scale_excess_min)
    return np.where(apply, measured, 1.0), blended


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

    # Step 5: inflate the formal errors by the measured point-to-point excess
    lc_err_raw = lc_err
    err_scale, blended = error_inflation(lc, lc_err_raw, epoch_ok, night.flags, settings)
    lc_err = (lc_err_raw * err_scale[:, None, :]).astype(np.float32)
    n_inflated = int(np.count_nonzero(np.any(err_scale > 1.0, axis=1)))
    logger.info(
        "error inflation (%s): %d stars scaled, %d blended",
        settings.lightcurve.inflate_errors, n_inflated, int(blended.sum()),
    )

    return LightCurveResult(
        lc=lc,
        lc_err=lc_err,
        lc_raw=lc_raw,
        epoch_ok=epoch_ok,
        decorrelation=decorrelation,
        lc_err_raw=lc_err_raw,
        err_scale=err_scale,
        blended=blended,
    )
