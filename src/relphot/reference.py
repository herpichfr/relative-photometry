"""Per-tile reference-star selection and construction.

:func:`select_candidates` picks the clean, non-variable, isolated,
high-SNR stars a tile's reference may be built from (PLAN.md Stage 3).
:func:`build_references` combines each tile's candidates into a per-frame,
per-aperture reference flux ``R`` -- construction D (inverse-variance
weighted, MAD-clipped mean of normalised fluxes) by default, or the simpler
fallback B (plain median of normalised fluxes) -- and divides every star's
own flux by its core tile's reference to get ``relative_flux``.

See PLAN.md's "Reference-star investigation": construction D was the best
performer at every candidate count and on the bright-star noise floor: each
candidate's normalising baseline is not a plain median over frames, because
the night's transparency/airmass trend is 10-25% peak to peak -- computed
iteratively instead, excluding frames whose preliminary reference is an
outlier relative to a quadratic fit in time, not relative to a flat median.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree

from relphot.exceptions import ConfigError
from relphot.numeric import nanmedian_quiet, unit_vectors, weighted_clipped_combine

if TYPE_CHECKING:
    from relphot.config import ReferenceSettings, Settings
    from relphot.match import MatchedNight
    from relphot.tiles import TileMap

logger = logging.getLogger(__name__)

__all__ = ["ReferenceResult", "build_references", "select_candidates"]

_METHODS = ("weighted_clipped_mean", "median_normalised")


@dataclass(slots=True)
class ReferenceResult:
    """Per-tile reference fluxes and every star's flux relative to its core tile.

    ``R``/``sigma_R``/``n_used`` are ``(n_tiles, n_frames, n_aper)``;
    ``relative_flux`` is ``(n_stars, n_frames, n_aper)`` = ``flux / R`` using
    each star's *core* tile (NaN for a star with no core tile). ``method`` is
    the construction actually used (``settings.reference.method``).
    """

    R: np.ndarray
    sigma_R: np.ndarray
    n_used: np.ndarray
    relative_flux: np.ndarray
    method: str

    @property
    def n_tiles(self) -> int:
        return int(self.R.shape[0])

    @property
    def n_frames(self) -> int:
        return int(self.R.shape[1])

    @property
    def n_aper(self) -> int:
        return int(self.R.shape[2])


def select_candidates(
    night: MatchedNight, variable_mask: np.ndarray, settings: Settings, aper: int
) -> np.ndarray:
    """Boolean mask, ``(n_stars,)``, of stars usable as reference candidates.

    A candidate has: presence >= ``settings.catalog.min_presence``; ``FLAGS
    == 0`` in every frame it is present in; a NaN-aware median SNR >=
    ``settings.reference.min_snr``; no other master star within
    ``settings.reference.isolation_radius_arcsec``; is not flagged in
    ``variable_mask``; and finite, positive flux at aperture ``aper``
    wherever present.
    """
    catalog = settings.catalog
    ref = settings.reference
    n = night.n_stars

    presence_ok = night.presence >= catalog.min_presence

    present = night.flags != -1
    flags_ok = np.all((night.flags == 0) | ~present, axis=1)

    finite_snr = np.isfinite(night.snr)
    any_snr = finite_snr.any(axis=1)
    median_snr = np.full(n, np.nan)
    median_snr[any_snr] = nanmedian_quiet(night.snr[any_snr], axis=1)
    snr_ok = median_snr >= ref.min_snr

    finite_pos = np.isfinite(night.ra) & np.isfinite(night.dec)
    isolated = np.zeros(n, dtype=bool)
    if np.any(finite_pos):
        vec = unit_vectors(night.ra[finite_pos], night.dec[finite_pos])
        tree = cKDTree(vec)
        radius_rad = np.radians(ref.isolation_radius_arcsec / 3600.0)
        chord_radius = 2.0 * np.sin(radius_rad / 2.0)
        pairs = tree.query_pairs(chord_radius)
        isolated_local = np.ones(vec.shape[0], dtype=bool)
        if pairs:
            bad = np.fromiter((i for pair in pairs for i in pair), dtype=np.int64)
            isolated_local[bad] = False
        local_idx = np.nonzero(finite_pos)[0]
        isolated[local_idx] = isolated_local

    flux_a = night.flux[:, :, aper]
    flux_ok = np.all(np.where(present, np.isfinite(flux_a) & (flux_a > 0), True), axis=1)

    not_variable = ~np.asarray(variable_mask, dtype=bool)

    candidate_mask = presence_ok & flags_ok & snr_ok & isolated & not_variable & flux_ok
    logger.info(
        "reference candidates: %d/%d stars (aperture %d)", int(candidate_mask.sum()), n, aper
    )
    return candidate_mask


def _fit_frame_outliers(
    r_j: np.ndarray, bjd_tdb: np.ndarray, frame_outlier_sigma: float
) -> np.ndarray:
    """Frames whose ``log10(r_j)`` does not deviate from a quadratic fit in time.

    A plain deviation from the median would mistake the night's own
    airmass/transparency trend for outliers; fitting a quadratic in
    ``bjd_tdb`` first removes that trend. Frames the fit could not evaluate
    (non-finite time or reference) are kept -- there is no evidence against
    them.
    """
    n_frames = r_j.shape[0]
    good = np.isfinite(bjd_tdb) & np.isfinite(r_j) & (r_j > 0)
    if np.count_nonzero(good) < 4:
        return np.ones(n_frames, dtype=bool)

    t0 = np.mean(bjd_tdb[good])
    coeffs = np.polyfit(bjd_tdb[good] - t0, np.log10(r_j[good]), 2)
    fit_all = np.polyval(coeffs, bjd_tdb - t0)

    with np.errstate(invalid="ignore", divide="ignore"):
        log_r = np.where(r_j > 0, np.log10(np.where(r_j > 0, r_j, 1.0)), np.nan)
    resid = log_r - fit_all
    med = nanmedian_quiet(resid[good])
    mad = nanmedian_quiet(np.abs(resid[good] - med))
    robust_sigma = 1.4826 * mad if mad > 0 else np.inf

    ok = np.abs(resid - med) <= frame_outlier_sigma * robust_sigma
    return np.where(np.isfinite(resid), ok, True)


def _weighted_clipped_mean_frames(
    n_ij: np.ndarray, w_ij: np.ndarray, clip_sigma: float, max_iter: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame inverse-variance weighted mean over candidates (axis 0), MAD-clipped.

    ``n_ij``/``w_ij`` are ``(n_candidates, n_frames)``; NaN or non-positive
    weight entries are never used. Returns ``(R, sigma_R, n_used)``, each
    ``(n_frames,)``.
    """
    mean, sigma_r, n_used, _mask = weighted_clipped_combine(
        n_ij, w_ij, clip_sigma, max_iter, axis=0
    )
    return mean, sigma_r, n_used


def _reference_weighted_clipped_mean(
    f: np.ndarray, sigma_f: np.ndarray, bjd_tdb: np.ndarray, ref: ReferenceSettings
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construction D: see the module docstring and PLAN.md."""
    valid0 = np.isfinite(f) & np.isfinite(sigma_f) & (sigma_f > 0) & (f > 0)
    f_masked = np.where(valid0, f, np.nan)

    baseline = nanmedian_quiet(f_masked, axis=1)
    baseline = np.where(np.isfinite(baseline) & (baseline > 0), baseline, np.nan)

    def _normalised(current_baseline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with np.errstate(invalid="ignore", divide="ignore"):
            n_ij = f_masked / current_baseline[:, np.newaxis]
            w_ij = (current_baseline[:, np.newaxis] / sigma_f) ** 2
        return np.where(valid0, n_ij, np.nan), np.where(valid0, w_ij, 0.0)

    for _ in range(ref.baseline_iter):
        n_ij, w_ij = _normalised(baseline)
        prelim_r, _sigma, _n_used = _weighted_clipped_mean_frames(
            n_ij, w_ij, ref.clip_sigma, ref.max_iter
        )
        frame_ok = _fit_frame_outliers(prelim_r, bjd_tdb, ref.frame_outlier_sigma)
        f_for_baseline = np.where(frame_ok[np.newaxis, :], f_masked, np.nan)
        new_baseline = nanmedian_quiet(f_for_baseline, axis=1)
        baseline = np.where(np.isfinite(new_baseline) & (new_baseline > 0), new_baseline, baseline)

    n_ij, w_ij = _normalised(baseline)
    return _weighted_clipped_mean_frames(n_ij, w_ij, ref.clip_sigma, ref.max_iter)


def _reference_median_normalised(
    f: np.ndarray, _sigma_f: np.ndarray, _bjd_tdb: np.ndarray, _ref: ReferenceSettings
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fallback construction B: plain nanmedian of normalised fluxes."""
    valid0 = np.isfinite(f) & (f > 0)
    f_masked = np.where(valid0, f, np.nan)
    baseline = nanmedian_quiet(f_masked, axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        n_ij = f_masked / baseline[:, np.newaxis]

    r_j = nanmedian_quiet(n_ij, axis=0)
    n_used = np.count_nonzero(np.isfinite(n_ij), axis=0)
    mad = nanmedian_quiet(np.abs(n_ij - r_j[np.newaxis, :]), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma_r = np.where(n_used > 0, 1.4826 * mad / np.sqrt(np.maximum(n_used, 1)), np.nan)
    return r_j, sigma_r, n_used


def build_references(
    night: MatchedNight, tilemap: TileMap, candidates: np.ndarray, settings: Settings
) -> ReferenceResult:
    """Build the per-tile, per-aperture reference and every star's relative flux."""
    method = settings.reference.method
    if method not in _METHODS:
        msg = f"unknown reference method {method!r}, expected one of {_METHODS}"
        raise ConfigError(msg)
    combine = _reference_weighted_clipped_mean if method == "weighted_clipped_mean" else (
        _reference_median_normalised
    )

    n_tiles = tilemap.n_tiles
    n_frames = night.n_frames
    n_aper = night.n_aper
    bjd_tdb = np.array([m.bjd_tdb for m in night.frame_meta], dtype=np.float64)

    r_arr = np.full((n_tiles, n_frames, n_aper), np.nan)
    sigma_r_arr = np.full((n_tiles, n_frames, n_aper), np.nan)
    n_used_arr = np.zeros((n_tiles, n_frames, n_aper), dtype=np.int64)

    for t in range(n_tiles):
        ext_idx = tilemap.extended_indices[t]
        cand_idx = ext_idx[candidates[ext_idx]] if ext_idx.size else ext_idx
        if cand_idx.size == 0:
            logger.warning("tile %d: no reference candidates available", t)
            continue
        for a in range(n_aper):
            f = night.flux[cand_idx, :, a].astype(np.float64)
            sigma_f = night.fluxerr[cand_idx, :, a].astype(np.float64)
            r_j, sigma_j, n_j = combine(f, sigma_f, bjd_tdb, settings.reference)
            too_few = n_j < settings.reference.min_used_per_frame
            if np.any(too_few & (n_j > 0)):
                logger.warning(
                    "tile %d aperture %d: reference set to NaN in %d frame(s) with fewer "
                    "than %d stars",
                    t, a, int(np.count_nonzero(too_few & (n_j > 0))),
                    settings.reference.min_used_per_frame,
                )
            r_j = np.where(too_few, np.nan, r_j)
            sigma_j = np.where(too_few, np.nan, sigma_j)
            r_arr[t, :, a] = r_j
            sigma_r_arr[t, :, a] = sigma_j
            n_used_arr[t, :, a] = n_j

    relative_flux = np.full(night.flux.shape, np.nan)
    core_tile = tilemap.core_tile
    has_core = core_tile >= 0
    star_idx = np.nonzero(has_core)[0]
    if star_idx.size:
        with np.errstate(invalid="ignore", divide="ignore"):
            core_of_star = core_tile[star_idx]
            relative_flux[star_idx] = (
                night.flux[star_idx].astype(np.float64) / r_arr[core_of_star]
            )

    return ReferenceResult(
        R=r_arr, sigma_R=sigma_r_arr, n_used=n_used_arr, relative_flux=relative_flux, method=method
    )
