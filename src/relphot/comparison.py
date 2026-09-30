"""Stage 4: comparison-star selection and ensemble construction.

Builds on relative flux r_ij = f_ij / R_j where R_j is the reference built in
Stage 3. A pool star survives if its robust time-scatter sits within k_floor of
the tile's scatter-vs-magnitude floor. Selection and ensemble are rebuilt each
round in a relaxation loop; leave-one-out (LOO) scoring is used for selected
stars and full scoring for the rest.

Pool diverges from reference.select_candidates: SNR floor 5 not 15 (faint stable
stars are useful and the floor cut already rejects noisy ones); isolation off by
default (each comparison star is cut on its own achieved scatter, so blends are
caught there); FLAGS==0 and presence reused unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree

from relphot.exceptions import ComparisonError, ConfigError
from relphot.numeric import (
    fit_noise_floor,
    mad_sigma,
    nanmedian_quiet,
    unit_vectors,
    weighted_clipped_combine,
)

if TYPE_CHECKING:
    from relphot.config import Settings
    from relphot.match import MatchedNight
    from relphot.reference import ReferenceResult
    from relphot.tiles import TileMap

logger = logging.getLogger(__name__)

#: Standard error of a Gaussian sample median in units of the standard error of the mean,
#: ``sqrt(pi / 2)`` = 1.2533.
MEDIAN_SE_FACTOR = float(np.sqrt(np.pi / 2.0))

__all__ = [
    "MEDIAN_SE_FACTOR",
    "ComparisonResult",
    "select_comparison_pool",
    "select_comparison_stars",
]


@dataclass(slots=True)
class ComparisonResult:
    """Per-tile comparison ensembles and per-star scatter measurements.

    ``mask`` is ``(n_stars, n_aper)`` bool, True for selected comparison stars in
    their core tile. ``ensemble``/``sigma_ensemble`` are ``(n_tiles, n_frames, n_aper)``
    in relative-flux units, normalized to ~1. ``sigma_star`` is ``(n_stars, n_aper)``
    final-round robust scatter; leave-one-out for members, full for the rest; NaN
    outside the pool. ``mag`` is -2.5 log10 of nanmedian relative flux for every star
    with a core tile and finite positive median, independent of pool membership;
    used for decorrelation surface fits (same definition for pool and non-pool stars).
    ``n_comparison`` and ``n_rounds_used`` are ``(n_tiles, n_aper)`` int. ``method``
    is the ensemble statistic used.
    """

    mask: np.ndarray
    ensemble: np.ndarray
    sigma_ensemble: np.ndarray
    sigma_star: np.ndarray
    mag: np.ndarray
    n_comparison: np.ndarray
    n_rounds_used: np.ndarray
    method: str


def _median_ensemble(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame median of ``c`` ``(n_members, n_frames)`` and the standard error of that median.

    ``sigma = MEDIAN_SE_FACTOR * mad_sigma / sqrt(n)`` with ``n`` the finite members of the
    frame and ``mad_sigma`` the 1.4826-scaled MAD across them; NaN where none is finite.
    """
    ens = nanmedian_quiet(c, axis=0)
    n_used = np.count_nonzero(np.isfinite(c), axis=0)
    mad = mad_sigma(c, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sig = np.where(n_used > 0, MEDIAN_SE_FACTOR * mad / np.sqrt(n_used), np.nan)
    return ens, sig


def select_comparison_pool(
    night: MatchedNight,
    variable_mask: np.ndarray,
    settings: Settings,
    aper: int,
    star_eligible: np.ndarray | None = None,
) -> np.ndarray:
    """Boolean mask, ``(n_stars,)``, of stars usable as comparison candidates.

    A pool star has: presence >= ``settings.catalog.min_presence``; ``FLAGS == 0``
    in every frame it is present in; a NaN-aware median SNR >=
    ``settings.comparison.min_snr``; (optionally) no other master star within
    ``settings.comparison.isolation_radius_arcsec``; is not flagged in
    ``variable_mask``; finite, positive flux at aperture ``aper`` wherever present;
    and passes ``star_eligible`` filter.

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
    comp = settings.comparison
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
    snr_ok = median_snr >= comp.min_snr

    isolated = np.ones(n, dtype=bool)
    if comp.require_isolation:
        finite_pos = np.isfinite(night.ra) & np.isfinite(night.dec)
        if np.any(finite_pos):
            vec = unit_vectors(night.ra[finite_pos], night.dec[finite_pos])
            tree = cKDTree(vec)
            radius_rad = np.radians(comp.isolation_radius_arcsec / 3600.0)
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

    pool_mask = (
        presence_ok & flags_ok & snr_ok & isolated & not_variable & flux_ok & border_ok
    )
    n_removed_border = int((~border_ok).sum())
    logger.info(
        "comparison pool: %d/%d stars (aperture %d)%s",
        int(pool_mask.sum()),
        n,
        aper,
        (f"; {n_removed_border} removed by border/tail cut" if n_removed_border > 0 else ""),
    )
    return pool_mask


def select_comparison_stars(
    night: MatchedNight,
    tilemap: TileMap,
    reference_result: ReferenceResult,
    variable_mask: np.ndarray,
    settings: Settings,
    star_eligible: np.ndarray | None = None,
) -> ComparisonResult:
    """Per-tile comparison-star selection and ensemble construction.

    For each (tile, aperture) pair, builds an ensemble from comparison stars
    selected via iterative relaxation. Returns a ComparisonResult with per-tile
    ensembles, per-star scatter, and selection masks.

    Parameters
    ----------
    night : MatchedNight
        The night
    tilemap : TileMap
        Tile map
    reference_result : ReferenceResult
        Reference result
    variable_mask : np.ndarray
        (n_stars,) bool mask of known variables
    settings : Settings
        Relphot settings
    star_eligible : np.ndarray | None, optional
        (n_stars,) bool mask of eligible stars; if None, computed once
        from star_eligibility (border and tail cuts); passed to all pool selections.
    """
    comp = settings.comparison
    method = comp.ensemble_statistic

    if method not in ("weighted_clipped_mean", "median"):
        msg = f"unknown ensemble_statistic {method!r}, expected 'weighted_clipped_mean' or 'median'"
        raise ConfigError(msg)

    n_tiles = tilemap.n_tiles
    n_frames = night.n_frames
    n_stars = night.n_stars
    n_aper = night.n_aper

    # Compute star_eligible once if None (border and tail cuts)
    if star_eligible is None:
        from relphot.eligibility import star_eligibility
        star_eligible = star_eligibility(night, settings).eligible

    # Outputs
    mask = np.zeros((n_stars, n_aper), dtype=bool)
    ensemble = np.full((n_tiles, n_frames, n_aper), np.nan)
    sigma_ensemble = np.full((n_tiles, n_frames, n_aper), np.nan)
    sigma_star = np.full((n_stars, n_aper), np.nan)
    mag = np.full((n_stars, n_aper), np.nan)
    n_comparison = np.zeros((n_tiles, n_aper), dtype=np.int64)
    n_rounds_used = np.zeros((n_tiles, n_aper), dtype=np.int64)

    # Process per aperture, compute pool once
    for a in range(n_aper):
        pool_global = select_comparison_pool(night, variable_mask, settings, a, star_eligible)

        # Process per tile
        for t in range(n_tiles):
            # Step 1: Filter pool to core tile members
            core_members = tilemap.core_tile == t
            pool = pool_global & core_members

            if pool.sum() == 0:
                logger.warning("tile %d, aperture %d: empty pool", t, a)
                continue

            if pool.sum() < comp.min_comparison_stars:
                msg = (
                    f"tile {t}, aperture {a}: pool has {pool.sum()} stars, "
                    f"need {comp.min_comparison_stars}"
                )
                raise ComparisonError(msg)

            # Step 2: Compute baseline magnitudes and initialize selection
            r = reference_result.relative_flux[:, :, a]
            baseline_i = np.full(n_stars, np.nan)
            baseline_i[pool] = nanmedian_quiet(r[pool, :], axis=1)

            # Drop pool stars with non-finite or <= 0 baseline
            valid_baseline = np.isfinite(baseline_i) & (baseline_i > 0) & pool
            bad_baseline_count = pool.sum() - valid_baseline.sum()
            if bad_baseline_count > 0:
                logger.warning(
                    "tile %d, aperture %d: dropped %d pool stars with non-finite baseline",
                    t,
                    a,
                    bad_baseline_count,
                )

            pool = valid_baseline
            if pool.sum() < comp.min_comparison_stars:
                msg = (
                    f"tile {t}, aperture {a}: pool has {pool.sum()} stars "
                    f"after baseline filtering, need {comp.min_comparison_stars}"
                )
                raise ComparisonError(msg)

            mag[pool, a] = -2.5 * np.log10(baseline_i[pool])

            selected = pool.copy()
            k_eff = comp.k_floor
            sigma_pool = np.full(n_stars, np.nan)

            # Step 3: Relaxation loop
            for round_idx in range(comp.n_rounds):
                # Step 3a: Construct c and w matrices
                pool_idx = np.nonzero(selected)[0]
                r_selected = r[pool_idx, :]  # (n_sel, n_frames)
                baseline_selected = baseline_i[pool_idx]  # (n_sel,)

                c = r_selected / baseline_selected[:, None]  # (n_sel, n_frames)

                # w_ij = baseline^2 / fluxerr_rel^2
                fluxerr_rel = night.fluxerr[pool_idx, :, a] / reference_result.R[
                    t, :, a
                ]  # (n_sel, n_frames)
                with np.errstate(invalid="ignore", divide="ignore"):
                    w = baseline_selected[:, None] ** 2 / (fluxerr_rel**2)
                w = np.where(np.isfinite(c) & np.isfinite(w) & (w > 0), w, 0.0)
                c = np.where(np.isfinite(c), c, np.nan)

                # Step 3b: Compute ensemble
                if method == "weighted_clipped_mean":
                    ens, sig, _, m = weighted_clipped_combine(
                        c, w, comp.clip_sigma, comp.max_iter, axis=0
                    )
                else:  # median
                    ens, sig = _median_ensemble(c)
                    m = np.isfinite(c)

                # Step 3c: Full score for every pool star
                pool_all_idx = np.nonzero(pool)[0]
                r_pool = r[pool_all_idx, :]  # (n_pool, n_frames)
                q_pool = r_pool / ens[None, :]  # (n_pool, n_frames)
                with np.errstate(invalid="ignore", divide="ignore"):
                    q_normalized = q_pool / nanmedian_quiet(q_pool, axis=1)[:, None]
                score_full = mad_sigma(q_normalized, axis=1)  # (n_pool,)
                score = np.full(n_stars, np.nan)
                score[pool_all_idx] = score_full

                # Step 3d: LOO score for selected stars
                if method == "weighted_clipped_mean":
                    # O(1) removal of each star from the converged weighted mean
                    # (documented approximation: the clip mask is not re-derived).
                    wm = np.where(m, w, 0.0)
                    sw = wm.sum(axis=0)  # (n_frames,)
                    total = ens * sw
                    with np.errstate(invalid="ignore", divide="ignore"):
                        denom = sw[None, :] - wm
                        ens_loo = np.where(
                            denom > 0,
                            (total[None, :] - np.where(m, wm * c, 0.0)) / denom,
                            np.nan,
                        )
                else:  # median: exact exclusion, one star at a time
                    ens_loo = np.empty_like(c)
                    for k in range(c.shape[0]):
                        ens_loo[k] = nanmedian_quiet(np.delete(c, k, axis=0), axis=0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    q_loo = r[pool_idx, :] / ens_loo
                    q_loo = q_loo / nanmedian_quiet(q_loo, axis=1)[:, None]
                score[pool_idx] = mad_sigma(q_loo, axis=1)

                # Store all scores
                sigma_pool = score.copy()

                # Step 3e: Fit noise floor
                selected_idx = np.nonzero(selected)[0]
                mag_selected = mag[selected_idx, a]
                score_selected = score[selected_idx]
                valid_for_floor = np.isfinite(score_selected)

                floor_func = fit_noise_floor(
                    mag_selected,
                    score_selected,
                    valid_for_floor,
                    comp.n_mag_bins,
                    comp.min_bin_stars,
                )

                # Step 3f: Apply cut
                with np.errstate(invalid="ignore"):
                    k_floor_vals_all = floor_func(mag[:, a])
                    is_finite_score = np.isfinite(score)
                    passes_cut = score <= k_eff * k_floor_vals_all
                new = pool & is_finite_score & passes_cut

                # Step 3g: Relaxation if needed
                while new.sum() < comp.min_comparison_stars and k_eff < comp.max_k_floor:
                    k_eff *= comp.k_floor_growth
                    with np.errstate(invalid="ignore"):
                        k_floor_vals_all = floor_func(mag[:, a])
                        passes_cut = score <= k_eff * k_floor_vals_all
                    new = pool & is_finite_score & passes_cut

                if new.sum() < comp.min_comparison_stars:
                    # Keep lowest-score stars (ties: lower index)
                    pool_idx_arr = np.nonzero(pool)[0]
                    scores_in_pool = score[pool]
                    sorted_idx = np.argsort(scores_in_pool)
                    kept_local = sorted_idx[: comp.min_comparison_stars]
                    new = np.zeros(n_stars, dtype=bool)
                    new[pool_idx_arr[kept_local]] = True
                    logger.warning(
                        "tile %d, aperture %d: relaxed to k_eff=%.2f, keeping lowest %d scores",
                        t,
                        a,
                        k_eff,
                        comp.min_comparison_stars,
                    )

                # Step 3h: Check convergence
                if np.array_equal(new, selected):
                    n_rounds_used[t, a] = round_idx + 1
                    break
                selected = new
                k_eff = comp.k_floor
            else:
                # Loop completed without early exit
                n_rounds_used[t, a] = comp.n_rounds

            # Step 4: Recompute ensemble with final selection if changed
            selected_idx = np.nonzero(selected)[0]
            r_selected = r[selected_idx, :]
            baseline_selected = baseline_i[selected_idx]

            c = r_selected / baseline_selected[:, None]
            fluxerr_rel = night.fluxerr[selected_idx, :, a] / reference_result.R[t, :, a]
            with np.errstate(invalid="ignore", divide="ignore"):
                w = baseline_selected[:, None] ** 2 / (fluxerr_rel**2)
            w = np.where(np.isfinite(c) & np.isfinite(w) & (w > 0), w, 0.0)
            c = np.where(np.isfinite(c), c, np.nan)

            if method == "weighted_clipped_mean":
                ens, sig, _, _m = weighted_clipped_combine(
                    c, w, comp.clip_sigma, comp.max_iter, axis=0
                )
            else:  # median
                ens, sig = _median_ensemble(c)

            ensemble[t, :, a] = ens
            sigma_ensemble[t, :, a] = sig
            mask[selected, a] = True
            n_comparison[t, a] = selected.sum()

            # Store final sigma_star
            # LOO scores for members, full-ensemble scores for the rest of the pool
            sigma_star[pool, a] = sigma_pool[pool]

        # Fill mag for all stars with a core tile and finite positive median
        # (independent of pool membership, for use in decorrelation surface fits)
        todo = (tilemap.core_tile >= 0) & np.isnan(mag[:, a])
        med = nanmedian_quiet(reference_result.relative_flux[todo, :, a], axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mag[todo, a] = np.where(np.isfinite(med) & (med > 0), -2.5 * np.log10(med), np.nan)

    return ComparisonResult(
        mask=mask,
        ensemble=ensemble,
        sigma_ensemble=sigma_ensemble,
        sigma_star=sigma_star,
        mag=mag,
        n_comparison=n_comparison,
        n_rounds_used=n_rounds_used,
        method=method,
    )
