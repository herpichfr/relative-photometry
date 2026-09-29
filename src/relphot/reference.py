"""Per-tile reference-star selection and construction.

Each tile's reference is built from a fixed star set S_t, selected once to be
valid in every kept frame of the whole night. :func:`select_reference_frames_and_stars`
drops frames globally when a tile cannot meet ``min_ref_stars`` and performs
per-star outlier rejection. Dropped frames are represented as NaN in ``R`` and
``sigma_R``, and automatically propagate to ``relative_flux``. The authoritative
frame-kept mask is ``ReferenceResult.frame_kept``; any future stage reading
``night.flux`` directly must intersect with ``frame_kept``.

:func:`select_candidates` picks the clean, non-variable, isolated, high-SNR
stars a tile's reference may be built from (PLAN.md Stage 3). :func:`build_references`
combines each tile's fixed star set into per-frame, per-aperture reference flux
``R`` using ``"weighted_fixed_mean"`` (inverse-variance weighted mean, default)
or ``"median_fixed"`` (plain median of normalised fluxes).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree

from relphot.exceptions import ConfigError, ReferenceFrameError
from relphot.numeric import mad_sigma, nanmedian_quiet, unit_vectors

if TYPE_CHECKING:
    from relphot.config import ReferenceSettings, Settings
    from relphot.match import MatchedNight
    from relphot.tiles import TileMap

logger = logging.getLogger(__name__)

__all__ = [
    "FrameSelection",
    "ReferenceResult",
    "build_references",
    "select_candidates",
    "select_reference_frames_and_stars",
]

_METHODS = ("weighted_fixed_mean", "median_fixed")


@dataclass(slots=True)
class ReferenceResult:
    """Per-tile reference fluxes and every star's flux relative to its core tile.

    ``R``/``sigma_R``/``n_used`` are ``(n_tiles, n_frames, n_aper)``;
    ``relative_flux`` is ``(n_stars, n_frames, n_aper)`` = ``flux / R`` using
    each star's *core* tile (NaN for a star with no core tile). ``method`` is
    the construction actually used (``settings.reference.method``).
    ``frame_kept`` is ``(n_frames,)`` bool, the authoritative whole-night
    frame-drop mask.
    """

    R: np.ndarray
    sigma_R: np.ndarray
    n_used: np.ndarray
    relative_flux: np.ndarray
    method: str
    frame_kept: np.ndarray

    @property
    def n_tiles(self) -> int:
        return int(self.R.shape[0])

    @property
    def n_frames(self) -> int:
        return int(self.R.shape[1])

    @property
    def n_aper(self) -> int:
        return int(self.R.shape[2])


@dataclass(slots=True)
class FrameSelection:
    """Result of selecting which frames to keep and which stars are valid references per tile.

    ``frame_kept`` is ``(n_frames,)`` bool, one whole-night decision shared by every tile
    and aperture. ``tile_stars[t]`` is tile t's fixed star set S_t (master-star indices,
    subset of candidates), with every member having FLAGS == 0 and finite positive flux at
    every aperture in every kept frame, after star-level outlier rejection.
    ``dropped_frames`` is an ascending tuple of False indices.
    """

    frame_kept: np.ndarray
    tile_stars: list[np.ndarray]
    dropped_frames: tuple[int, ...]


def select_candidates(
    night: MatchedNight,
    variable_mask: np.ndarray,
    settings: Settings,
    aper: int,
    star_eligible: np.ndarray | None = None,
) -> np.ndarray:
    """Boolean mask, ``(n_stars,)``, of stars usable as reference candidates.

    A candidate has: presence >= ``settings.catalog.min_presence``; ``FLAGS
    == 0`` in every frame it is present in; a NaN-aware median SNR >=
    ``settings.reference.min_snr``; no other master star within
    ``settings.reference.isolation_radius_arcsec``; is not flagged in
    ``variable_mask``; finite, positive flux at aperture ``aper``
    wherever present; and passes ``star_eligible`` filter.

    Parameters
    ----------
    night : MatchedNight
        The night
    variable_mask : np.ndarray
        (n_stars,) bool mask of known variables
    settings : Settings
        Relphot settings
    aper : int
        Aperture index
    star_eligible : np.ndarray | None, optional
        (n_stars,) bool mask of eligible stars (border and tail cuts).
        If None, computes from star_eligibility(night, settings).eligible.
        Raises ValueError if shape mismatch.
    """
    catalog = settings.catalog
    ref = settings.reference
    n = night.n_stars

    if star_eligible is None:
        from relphot.eligibility import star_eligibility
        star_eligible = star_eligibility(night, settings).eligible
    else:
        star_eligible = np.asarray(star_eligible, dtype=bool)
        if star_eligible.shape != (n,):
            raise ValueError(
                f"star_eligible shape {star_eligible.shape} != ({n},)"
            )

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
    border_ok = np.asarray(star_eligible, dtype=bool)

    candidate_mask = (
        presence_ok & flags_ok & snr_ok & isolated & not_variable & flux_ok & border_ok
    )
    n_removed_border = int((~border_ok).sum())
    logger.info(
        "reference candidates: %d/%d stars (aperture %d)%s",
        int(candidate_mask.sum()),
        n,
        aper,
        (f"; {n_removed_border} removed by border/tail cut" if n_removed_border > 0 else ""),
    )
    return candidate_mask


def select_reference_frames_and_stars(
    night: MatchedNight, tilemap: TileMap, candidates: np.ndarray, settings: Settings, aper: int
) -> FrameSelection:
    """Select frames to keep and build each tile's fixed star set S_t.

    Drops frames globally when min_ref_stars cannot be met in every tile within
    max_dropped_frame_fraction of the night. Performs per-star outlier rejection
    on each tile's reference star set. Returns a FrameSelection with the kept-frame
    mask and per-tile star indices.
    """
    ref = settings.reference
    n_frames = night.n_frames
    n_tiles = tilemap.n_tiles

    # valid[i,j]: FLAGS == 0 and finite, positive flux at EVERY aperture, since
    # S_t chosen here is reused unchanged for every aperture's reference.
    flux_ok = np.all(np.isfinite(night.flux) & (night.flux > 0), axis=2)
    valid_i_j = (night.flags == 0) & flux_ok

    # Frame-dropping loop
    max_drop = int(np.floor(ref.max_dropped_frame_fraction * n_frames))
    dropped_set = set()  # type: set[int]
    warned_empty_tile = False

    for drop_iter in range(max_drop + 1):
        kept = np.array([i not in dropped_set for i in range(n_frames)], dtype=bool)
        kept_idx = np.nonzero(kept)[0]

        # For each tile, compute S_t = candidates in tile t valid in all kept frames
        tile_stars_list = []
        short_tiles = {}  # tile index -> (deficit = min_ref_stars - |S_t|, |S_t|)

        for t in range(n_tiles):
            ext = tilemap.extended_indices[t]
            cand_in_tile = ext[candidates[ext]] if ext.size else np.array([], dtype=np.int64)

            if cand_in_tile.size == 0:
                if not warned_empty_tile:
                    logger.warning("tile %d: no reference candidates available", t)
                    warned_empty_tile = True
                tile_stars_list.append(np.array([], dtype=np.int64))
                continue

            # valid_in_all_kept[i] = True iff candidates[i] is valid in all kept frames
            valid_in_all_kept = np.all(valid_i_j[cand_in_tile][:, kept], axis=1)
            s_t = cand_in_tile[valid_in_all_kept]
            tile_stars_list.append(s_t)

            deficit = max(0, ref.min_ref_stars - len(s_t))
            if deficit > 0:
                short_tiles[t] = (deficit, len(s_t))

        if not short_tiles:
            # All tiles meet min_ref_stars; go to star rejection
            break

        if len(dropped_set) >= max_drop:
            # Hit drop cap; raise error
            msg_parts = []
            for t in sorted(short_tiles.keys()):
                deficit, n_in_t = short_tiles[t]
                min_stars = ref.min_ref_stars
                msg_parts.append(f"tile {t}: {n_in_t}/{min_stars} stars")
            msg = f"cannot meet min_ref_stars after dropping {max_drop} frames: " + ", ".join(
                msg_parts
            )
            raise ReferenceFrameError(msg)

        # Find the worst tile (ties broken by lowest index)
        worst_t = min(short_tiles.keys(), key=lambda t: (-short_tiles[t][0], t))

        # Find which frames are problematic for this tile
        ext = tilemap.extended_indices[worst_t]
        cand_in_tile = ext[candidates[ext]] if ext.size else np.array([], dtype=np.int64)

        # Pool: candidates in C_t invalid in exactly one kept frame
        pool_mask = np.zeros(cand_in_tile.size, dtype=bool)
        for cand_idx, cand_star in enumerate(cand_in_tile):
            invalid_frames = ~valid_i_j[cand_star][kept]
            if np.sum(invalid_frames) == 1:
                pool_mask[cand_idx] = True

        if not np.any(pool_mask):
            # Fallback: use all of C_t
            pool_cand = np.arange(cand_in_tile.size)
        else:
            pool_cand = np.nonzero(pool_mask)[0]

        # Count invalid entries per kept frame over pool
        frame_invalid_counts = np.zeros(len(kept_idx), dtype=np.int64)
        for pool_idx in pool_cand:
            cand_star = cand_in_tile[pool_idx]
            invalid_in_kept = ~valid_i_j[cand_star][kept]
            frame_invalid_counts += invalid_in_kept.astype(np.int64)

        # Drop the frame with highest count (ties: lowest index in kept_idx)
        worst_frame_pos = np.argmax(frame_invalid_counts)
        frame_to_drop = kept_idx[worst_frame_pos]
        dropped_set.add(frame_to_drop)

        logger.info(
            "frame drop iteration %d: tile %d, drop frame %d, count %d",
            drop_iter,
            worst_t,
            frame_to_drop,
            int(frame_invalid_counts[worst_frame_pos]),
        )

    # Star-level outlier rejection per tile
    for t in range(n_tiles):
        s_t = tile_stars_list[t]
        if s_t.size < 2:
            continue

        kept_idx = np.nonzero(kept)[0]
        for _reject_iter in range(ref.star_reject_iter):
            # Compute R_j using the fixed-weight combine at aper
            f_s = night.flux[s_t, :, aper][:, kept].astype(np.float64)
            bjd_kept = np.array([night.frame_meta[i].bjd_tdb for i in kept_idx], dtype=np.float64)

            # Use _reference_weighted_fixed to compute reference
            r_j, _sigma_r_j = _reference_weighted_fixed(
                f_s, night.fluxerr[s_t, :, aper][:, kept].astype(np.float64), bjd_kept, ref
            )

            # Per-star scatter: mad_sigma(log10(f_ij / baseline_i / R_j))
            baseline_i = nanmedian_quiet(f_s, axis=1)
            baseline_i = np.where(np.isfinite(baseline_i) & (baseline_i > 0), baseline_i, np.nan)

            with np.errstate(invalid="ignore", divide="ignore"):
                log_ratio = np.log10(f_s / baseline_i[:, np.newaxis] / r_j[np.newaxis, :])

            s_i = mad_sigma(log_ratio, axis=1)
            pop_med = nanmedian_quiet(s_i)
            pop_sig = mad_sigma(s_i)

            if not np.isfinite(pop_sig) or pop_sig == 0:
                pop_sig = np.inf

            flagged = s_i > pop_med + ref.star_outlier_sigma * pop_sig
            if not np.any(flagged):
                break

            # Check if removing flagged stars leaves >= min_ref_stars
            if np.sum(~flagged) < ref.min_ref_stars:
                break

            s_t = s_t[~flagged]

        tile_stars_list[t] = s_t

    frame_kept = kept
    dropped_frames = tuple(sorted(dropped_set))

    return FrameSelection(
        frame_kept=frame_kept, tile_stars=tile_stars_list, dropped_frames=dropped_frames
    )


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


def _reference_weighted_fixed(
    f_s: np.ndarray, sigma_s: np.ndarray, bjd_tdb_kept: np.ndarray, ref: ReferenceSettings
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-set inverse-variance weighted mean of normalised fluxes.

    Inputs are (|S|, n_kept). Computes baseline iteratively, excluding frame
    outliers relative to a quadratic fit in time, then returns (R_j, sigma_R_j),
    each (n_kept,).
    """
    # Every entry of a fixed set is valid by construction; the mask is a guard,
    # not per-entry clipping. Errors enter only through the fixed per-star weight.
    valid0 = np.isfinite(f_s) & (f_s > 0)
    f_masked = np.where(valid0, f_s, np.nan)

    baseline_i = nanmedian_quiet(f_masked, axis=1)
    baseline_i = np.where(np.isfinite(baseline_i) & (baseline_i > 0), baseline_i, np.nan)

    for _ in range(ref.baseline_iter):
        with np.errstate(invalid="ignore", divide="ignore"):
            norm_f = f_masked / baseline_i[:, np.newaxis]
        # Unweighted preliminary reference, only to find trend-outlier frames
        # for the baseline; S itself never changes here.
        r_j_for_fit = np.nanmean(norm_f, axis=0)

        frame_ok = _fit_frame_outliers(r_j_for_fit, bjd_tdb_kept, ref.frame_outlier_sigma)
        f_for_baseline = np.where(frame_ok[np.newaxis, :], f_masked, np.nan)
        new_baseline = nanmedian_quiet(f_for_baseline, axis=1)
        baseline_i = np.where(
            np.isfinite(new_baseline) & (new_baseline > 0), new_baseline, baseline_i
        )

    # Final weighted combine
    with np.errstate(invalid="ignore", divide="ignore"):
        w_i = (baseline_i**2) / nanmedian_quiet(sigma_s**2, axis=1)
    w_i = np.where(np.isfinite(w_i) & (w_i > 0), w_i, 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        norm_f = f_masked / baseline_i[:, np.newaxis]
    norm_f = np.where(valid0, norm_f, np.nan)

    sum_w = np.nansum(np.where(valid0, w_i[:, np.newaxis], 0.0), axis=0)
    r_j = np.nansum(np.where(valid0, w_i[:, np.newaxis] * norm_f, 0.0), axis=0) / np.where(
        sum_w > 0, sum_w, np.nan
    )
    sigma_r_j = np.where(sum_w > 0, np.sqrt(1.0 / sum_w), np.nan)

    return r_j, sigma_r_j


def _reference_median_fixed(
    f_s: np.ndarray, _sigma_s: np.ndarray, _bjd_tdb_kept: np.ndarray, _ref: ReferenceSettings
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-set median of normalised fluxes.

    Inputs are (|S|, n_kept). Returns (R_j, sigma_R_j), each (n_kept,).
    """
    valid0 = np.isfinite(f_s) & (f_s > 0)
    f_masked = np.where(valid0, f_s, np.nan)

    baseline_i = nanmedian_quiet(f_masked, axis=1)
    baseline_i = np.where(np.isfinite(baseline_i) & (baseline_i > 0), baseline_i, np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        norm_f = f_masked / baseline_i[:, np.newaxis]

    r_j = nanmedian_quiet(norm_f, axis=0)
    mad = nanmedian_quiet(np.abs(norm_f - r_j[np.newaxis, :]), axis=0)
    n_s = f_s.shape[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma_r_j = np.where(
            np.isfinite(mad),
            1.4826 * mad / np.sqrt(np.maximum(n_s, 1)),
            np.nan,
        )
    return r_j, sigma_r_j


def build_references(
    night: MatchedNight, tilemap: TileMap, frame_selection: FrameSelection, settings: Settings
) -> ReferenceResult:
    """Build the per-tile, per-aperture reference and every star's relative flux.

    Uses the fixed star set and kept-frame mask from frame_selection.
    """
    method = settings.reference.method
    if method not in _METHODS:
        msg = f"unknown reference method {method!r}, expected one of {_METHODS}"
        raise ConfigError(msg)
    combine = (
        _reference_weighted_fixed if method == "weighted_fixed_mean" else _reference_median_fixed
    )

    # Reject old method names
    if method in ("weighted_clipped_mean", "median_normalised"):
        msg = f"method {method!r} is no longer supported; use weighted_fixed_mean or median_fixed"
        raise ConfigError(msg)

    n_tiles = tilemap.n_tiles
    n_frames = night.n_frames
    n_aper = night.n_aper
    frame_kept = frame_selection.frame_kept
    kept_idx = np.nonzero(frame_kept)[0]
    bjd_tdb_kept = np.array(
        [night.frame_meta[i].bjd_tdb for i in kept_idx], dtype=np.float64
    )

    r_arr = np.full((n_tiles, n_frames, n_aper), np.nan)
    sigma_r_arr = np.full((n_tiles, n_frames, n_aper), np.nan)
    n_used_arr = np.zeros((n_tiles, n_frames, n_aper), dtype=np.int64)

    for t in range(n_tiles):
        s_t = frame_selection.tile_stars[t]
        if s_t.size < 2:
            logger.warning("tile %d: fewer than 2 reference stars, skipping", t)
            continue

        for a in range(n_aper):
            f_s = night.flux[s_t, :, a][
                :, frame_kept
            ].astype(np.float64)
            sigma_s = night.fluxerr[s_t, :, a][
                :, frame_kept
            ].astype(np.float64)

            r_j, sigma_j = combine(f_s, sigma_s, bjd_tdb_kept, settings.reference)

            # Place results in kept frame positions; dropped frames stay NaN
            r_arr[t, kept_idx, a] = r_j
            sigma_r_arr[t, kept_idx, a] = sigma_j
            n_used_arr[t, kept_idx, a] = len(s_t)

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
        R=r_arr,
        sigma_R=sigma_r_arr,
        n_used=n_used_arr,
        relative_flux=relative_flux,
        method=method,
        frame_kept=frame_kept,
    )
